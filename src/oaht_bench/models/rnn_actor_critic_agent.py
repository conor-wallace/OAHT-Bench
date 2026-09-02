from functools import partial

import jax
import jax.numpy as jnp

from oaht_bench.models.agent_interface import AgentPolicy
from oaht_bench.models.rnn_actor_critic import (
    RNNActorCritic,
    RNNActorWithConditionalCritic,
    ScannedRNN,
)


class RNNActorCriticPolicy(AgentPolicy):
    """Policy wrapper for RNN Actor-Critic"""

    def __init__(self, action_dim, obs_dim,
                 activation="tanh", fc_hidden_dim=64, gru_hidden_dim=64):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            activation: str, activation function to use
            fc_hidden_dim: int, dimension of the feed-forward hidden layers
            gru_hidden_dim: int, dimension of the GRU hidden state
        """
        super().__init__(action_dim, obs_dim)
        self.network = RNNActorCritic(
            action_dim,
            fc_hidden_dim=fc_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            activation=activation
        )
        self.gru_hidden_dim = gru_hidden_dim

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the RNN policy.
        Shape of obs, done, avail_actions should correspond to (seq_len, batch_size, ...)
        Shape of hstate should correspond to (1, batch_size, -1). We maintain the extra first dimension for
        compatibility with the learning codes.
        """
        batch_size = obs.shape[1]
        new_hstate, pi, _ = self.network.apply(params, hstate.squeeze(0), (obs, done, avail_actions))
        action = jax.lax.cond(test_mode,
                              lambda: pi.mode(),
                              lambda: pi.sample(seed=rng))
        return action, new_hstate.reshape(1, batch_size, -1)

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Get actions, values, and policy for the RNN policy.
        Shape of obs, done, avail_actions should correspond to (seq_len, batch_size, ...)
        Shape of hstate should correspond to (1, batch_size, -1). We maintain the extra first dimension for
        compatibility with the learning codes.
        """
        batch_size = obs.shape[1]
        new_hstate, pi, val = self.network.apply(params, hstate.squeeze(0), (obs, done, avail_actions))
        action = pi.sample(seed=rng)
        return action, val, pi, new_hstate.reshape(1, batch_size, -1)

    def init_hstate(self, batch_size, aux_info=None):
        """Initialize hidden state for the RNN policy."""
        hstate =  ScannedRNN.initialize_carry(batch_size, self.gru_hidden_dim)
        hstate = hstate.reshape(1, batch_size, self.gru_hidden_dim)
        return hstate

    def init_params(self, rng):
        """Initialize parameters for the RNN policy."""
        batch_size = 1
        # Initialize hidden state
        init_hstate = self.init_hstate(batch_size)

        # Create dummy inputs - add time dimension
        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_avail = jnp.ones((1, batch_size, self.action_dim))
        dummy_x = (dummy_obs, dummy_done, dummy_avail)

        # Initialize model
        return self.network.init(rng, init_hstate.reshape(batch_size, -1), dummy_x)


class RNNActorWithConditionalCriticPolicy(AgentPolicy):
    """Policy wrapper for RNNActorWithConditionalCritic.

    Same calling convention as RNNActorCriticPolicy (obs/done shaped
    (seq_len, batch, ...), hstate shaped (1, batch, -1)) plus aux_obs for the
    teammate one-hot id, matching ActorWithConditionalCriticPolicy's
    convention for that argument. BRDiv.py and LBRDiv.py already call
    get_action_value_policy this exact way (obs/done/aux_obs each
    newaxis'd, avail_actions not) -- built for this before a matching
    policy class existed to receive it; see docs/tuning_record.md.
    """

    def __init__(self, action_dim, obs_dim, pop_size,
                 activation="tanh", fc_hidden_dim=64, gru_hidden_dim=64):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            pop_size: int, number of agents in the population that the critic was trained with
            activation: str, activation function to use
            fc_hidden_dim: int, dimension of the feed-forward hidden layers
            gru_hidden_dim: int, dimension of the GRU hidden state
        """
        super().__init__(action_dim, obs_dim)
        self.pop_size = pop_size
        self.network = RNNActorWithConditionalCritic(
            action_dim,
            fc_hidden_dim=fc_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            activation=activation
        )
        self.gru_hidden_dim = gru_hidden_dim

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the RNN conditional-critic policy.
        Shape of obs, done, avail_actions should correspond to (seq_len, batch_size, ...)
        Shape of hstate should correspond to (1, batch_size, -1).

        aux_obs is ignored here, matching ActorWithConditionalCriticPolicy's own
        get_action: at rollout/evaluation time the caller may not have (or mean)
        a real teammate id in aux_obs -- run_episodes.py passes a completely
        different tuple through this slot for other policy types (e.g. OMIS's
        opponent-history aux_obs). The critic's agent-id input is only load-
        bearing during training, via get_action_value_policy below, so a dummy
        zero id is used here instead of whatever aux_obs happens to hold.
        """
        batch_size = obs.shape[1]
        dummy_agent_id = jnp.zeros(obs.shape[:-1] + (self.pop_size,))
        new_hstate, pi, _ = self.network.apply(
            params, hstate.squeeze(0), (obs, dummy_agent_id, done, avail_actions)
        )
        action = jax.lax.cond(test_mode,
                              lambda: pi.mode(),
                              lambda: pi.sample(seed=rng))
        return action, new_hstate.reshape(1, batch_size, -1)

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Get actions, values, and policy for the RNN conditional-critic policy.
        Shapes as in get_action.
        """
        batch_size = obs.shape[1]
        new_hstate, pi, val = self.network.apply(
            params, hstate.squeeze(0), (obs, aux_obs, done, avail_actions)
        )
        action = pi.sample(seed=rng)
        return action, val, pi, new_hstate.reshape(1, batch_size, -1)

    def init_hstate(self, batch_size, aux_info=None):
        """Initialize hidden state for the RNN policy."""
        hstate = ScannedRNN.initialize_carry(batch_size, self.gru_hidden_dim)
        hstate = hstate.reshape(1, batch_size, self.gru_hidden_dim)
        return hstate

    def init_params(self, rng):
        """Initialize parameters for the RNN conditional-critic policy."""
        batch_size = 1
        init_hstate = self.init_hstate(batch_size)

        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_id = jnp.zeros((1, batch_size, self.pop_size))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_avail = jnp.ones((1, batch_size, self.action_dim))
        dummy_x = (dummy_obs, dummy_id, dummy_done, dummy_avail)

        return self.network.init(rng, init_hstate.reshape(batch_size, -1), dummy_x)


class PseudoRNNActorWithConditionalCriticPolicy(RNNActorWithConditionalCriticPolicy):
    """Enables RNNActorWithConditionalCriticPolicy to act as an RNNActorCriticPolicy,
    by passing in a dummy agent id -- the RNN analogue of
    PseudoActorWithConditionalCriticPolicy (mlp_actor_critic_agent.py).

    CoMeDi trains its first population member via a plain self-play IPPO
    trainer (make_ppo_train/initialize_agent's "pseudo_*" branches) that
    knows nothing about population ids, before any real population exists to
    condition on. That trainer calls get_action_value_policy the same way
    for every actor type, so the dummy id has to be substituted here rather
    than by the caller. Needed so the warmup member's params share the same
    network shape as the conditional-critic members added afterward -- an
    RNN main phase with an MLP-shaped warmup member would fail to add to the
    same BufferedPopulation. See docs/tuning_record.md.
    """

    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        dummy_agent_id = jnp.zeros(obs.shape[:-1] + (self.pop_size,))
        return super().get_action_value_policy(
            params, obs, done, avail_actions, hstate, rng, dummy_agent_id, env_state
        )
