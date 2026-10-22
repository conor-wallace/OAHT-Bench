"""AgentPolicy wrappers for the CNN+GRU networks (``cnn_rnn_actor_critic.py``).

Mirrors ``rnn_actor_critic_agent.py`` one-to-one -- same calling convention
(obs/done shaped ``(seq_len, batch, ...)``, hstate ``(1, batch, -1)``) -- with one
addition: ``obs_shape``, the unflattened ``(width, height, channels)`` the CNN encoder
reshapes the flat observation back into. ``obs_dim`` stays the flattened width so the
rest of the training/eval code (which only ever sees flat obs) is unchanged.
"""

from functools import partial

import jax
import jax.numpy as jnp

from oaht_bench.models.agent_interface import AgentPolicy
from oaht_bench.models.cnn_rnn_actor_critic import (
    CNNRNNActorCritic,
    CNNRNNActorWithConditionalCritic,
)
from oaht_bench.models.rnn_actor_critic import ScannedRNN


class CNNRNNActorCriticPolicy(AgentPolicy):
    """Policy wrapper for CNNRNNActorCritic (shared CNN+GRU trunk)."""

    def __init__(
        self,
        action_dim,
        obs_dim,
        obs_shape,
        activation="relu",
        fc_hidden_dim=128,
        gru_hidden_dim=128,
    ):
        super().__init__(action_dim, obs_dim)
        self.obs_shape = tuple(obs_shape)
        self.network = CNNRNNActorCritic(
            action_dim,
            obs_shape=self.obs_shape,
            fc_hidden_dim=fc_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            activation=activation,
        )
        self.gru_hidden_dim = gru_hidden_dim

    @partial(jax.jit, static_argnums=(0,))
    def get_action(
        self,
        params,
        obs,
        done,
        avail_actions,
        hstate,
        rng,
        aux_obs=None,
        env_state=None,
        test_mode=False,
        reward=None,
    ):
        batch_size = obs.shape[1]
        new_hstate, pi, _ = self.network.apply(
            params, hstate.squeeze(0), (obs, done, avail_actions)
        )
        action = jax.lax.cond(test_mode, lambda: pi.mode(), lambda: pi.sample(seed=rng))
        return action, new_hstate.reshape(1, batch_size, -1)

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(
        self, params, obs, done, avail_actions, hstate, rng, aux_obs=None, env_state=None
    ):
        batch_size = obs.shape[1]
        new_hstate, pi, val = self.network.apply(
            params, hstate.squeeze(0), (obs, done, avail_actions)
        )
        action = pi.sample(seed=rng)
        return action, val, pi, new_hstate.reshape(1, batch_size, -1)

    def init_hstate(self, batch_size, aux_info=None):
        hstate = ScannedRNN.initialize_carry(batch_size, self.gru_hidden_dim)
        return hstate.reshape(1, batch_size, self.gru_hidden_dim)

    def init_params(self, rng):
        batch_size = 1
        init_hstate = self.init_hstate(batch_size)
        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_avail = jnp.ones((1, batch_size, self.action_dim))
        dummy_x = (dummy_obs, dummy_done, dummy_avail)
        return self.network.init(rng, init_hstate.reshape(batch_size, -1), dummy_x)


class CNNRNNActorWithConditionalCriticPolicy(AgentPolicy):
    """Policy wrapper for CNNRNNActorWithConditionalCritic (BRDiv/L-BRDiv, CoMeDi).

    Same convention as ``RNNActorWithConditionalCriticPolicy``: ``aux_obs`` carries
    the teammate one-hot id at training time (load-bearing only in the critic, via
    ``get_action_value_policy``); ``get_action`` substitutes a zero id, since at
    rollout/eval time ``aux_obs`` may hold something else entirely.
    """

    def __init__(
        self,
        action_dim,
        obs_dim,
        obs_shape,
        pop_size,
        activation="relu",
        fc_hidden_dim=128,
        gru_hidden_dim=128,
    ):
        super().__init__(action_dim, obs_dim)
        self.obs_shape = tuple(obs_shape)
        self.pop_size = pop_size
        self.network = CNNRNNActorWithConditionalCritic(
            action_dim,
            obs_shape=self.obs_shape,
            fc_hidden_dim=fc_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            activation=activation,
        )
        self.gru_hidden_dim = gru_hidden_dim

    @partial(jax.jit, static_argnums=(0,))
    def get_action(
        self,
        params,
        obs,
        done,
        avail_actions,
        hstate,
        rng,
        aux_obs=None,
        env_state=None,
        test_mode=False,
        reward=None,
    ):
        batch_size = obs.shape[1]
        dummy_agent_id = jnp.zeros(obs.shape[:-1] + (self.pop_size,))
        new_hstate, pi, _ = self.network.apply(
            params, hstate.squeeze(0), (obs, dummy_agent_id, done, avail_actions)
        )
        action = jax.lax.cond(test_mode, lambda: pi.mode(), lambda: pi.sample(seed=rng))
        return action, new_hstate.reshape(1, batch_size, -1)

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(
        self, params, obs, done, avail_actions, hstate, rng, aux_obs=None, env_state=None
    ):
        batch_size = obs.shape[1]
        new_hstate, pi, val = self.network.apply(
            params, hstate.squeeze(0), (obs, aux_obs, done, avail_actions)
        )
        action = pi.sample(seed=rng)
        return action, val, pi, new_hstate.reshape(1, batch_size, -1)

    def init_hstate(self, batch_size, aux_info=None):
        hstate = ScannedRNN.initialize_carry(batch_size, self.gru_hidden_dim)
        return hstate.reshape(1, batch_size, self.gru_hidden_dim)

    def init_params(self, rng):
        batch_size = 1
        init_hstate = self.init_hstate(batch_size)
        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_id = jnp.zeros((1, batch_size, self.pop_size))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_avail = jnp.ones((1, batch_size, self.action_dim))
        dummy_x = (dummy_obs, dummy_id, dummy_done, dummy_avail)
        return self.network.init(rng, init_hstate.reshape(batch_size, -1), dummy_x)


class PseudoCNNRNNActorWithConditionalCriticPolicy(CNNRNNActorWithConditionalCriticPolicy):
    """Lets the conditional-critic CNN policy stand in as a plain actor-critic by
    feeding a dummy teammate id -- the CNN analogue of
    ``PseudoRNNActorWithConditionalCriticPolicy``. CoMeDi's self-play warmup trains
    its first member through a population-agnostic IPPO trainer, so the params must
    share the main phase's network shape to enter the same BufferedPopulation.
    """

    def get_action_value_policy(
        self, params, obs, done, avail_actions, hstate, rng, aux_obs=None, env_state=None
    ):
        dummy_agent_id = jnp.zeros(obs.shape[:-1] + (self.pop_size,))
        return super().get_action_value_policy(
            params, obs, done, avail_actions, hstate, rng, dummy_agent_id, env_state
        )
