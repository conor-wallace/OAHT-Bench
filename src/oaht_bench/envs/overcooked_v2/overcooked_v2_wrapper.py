from functools import partial

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments import spaces

from oaht_bench.envs.overcooked_v2.overcooked import OvercookedV2

from ..base_env import BaseEnv, WrappedEnvState


class OvercookedV2Wrapper(BaseEnv):
    '''Wrapper for the Overcooked-v2 environment to ensure that it follows a common
    interface with other environments provided in this library.

    Mirrors ``overcooked.overcooked_wrapper.OvercookedWrapper`` exactly, since
    OvercookedV2's action_space/observation_space signatures are unchanged from
    v1's (agent-argument-optional, uniform across agents) despite everything
    else about the environment being new.

    Main features:
    - Flattened observations
    - Base return tracking
    - get_avail_actions is a v1-matching all-actions-available stub: OvercookedV2
      doesn't override the abstract base's get_avail_actions (raises
      NotImplementedError) because its action set has no state-dependent
      restrictions, same as v1 -- see PROVENANCE.md.
    '''
    def __init__(self, *args, **kwargs):
        self.env = OvercookedV2(*args, **kwargs)
        self.agents = self.env.agents
        self.num_agents = len(self.agents)

        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.agents}

    def observation_space(self, agent: str):
        """Returns the flattened observation space."""
        flat_obs_shape = (self.env.obs_shape[0] * self.env.obs_shape[1] * self.env.obs_shape[2],)
        return spaces.Box(0, 255, flat_obs_shape)

    def action_space(self, agent: str):
        return self.env.action_space()

    def reset(self, key: chex.PRNGKey) -> tuple[dict[str, chex.Array], WrappedEnvState]:
        obs, env_state = self.env.reset(key)
        flat_obs = {agent: obs[agent].flatten() for agent in self.agents}
        return flat_obs, WrappedEnvState(env_state, jnp.zeros(self.num_agents), jnp.zeros(self.num_agents), jnp.empty((), dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> dict[str, jnp.ndarray]:
        """All actions available: OvercookedV2's action set has no
        state-dependent restrictions, same as v1 (see PROVENANCE.md)."""
        num_actions = len(self.env.action_set)
        return {agent: jnp.ones(num_actions) for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        """Returns the step count for the environment."""
        return state.env_state.time

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: dict[str, chex.Array],
        reset_state: WrappedEnvState | None = None,
    ) -> tuple[dict[str, chex.Array], WrappedEnvState, dict[str, float], dict[str, bool], dict]:
        '''Wrapped step function. The base return is tracked in the info
        dictionary, so that the return can be obtained from the final info.
        '''
        obs, env_state, rewards, dones, infos = self.env.step(key, state.env_state, actions, reset_state)
        flat_obs = {agent: obs[agent].flatten() for agent in self.agents}
        # v2's step_env doesn't populate infos['base_reward'] the way v1's
        # does (see PROVENANCE.md) -- v1 builds that key from its own
        # `rewards` dict (overcooked_v1.py:139), which is the same base
        # signal v2's `rewards` already is (info['shaped_reward'] is
        # separate, not folded in here either way). Built at the wrapper
        # level instead of relied on from infos.
        base_reward = jnp.array([rewards[agent] for agent in self.agents])
        base_return_so_far = base_reward + state.base_return_so_far

        # v2's raw info['shaped_reward'] is a per-agent dict ({'agent_0':
        # scalar, ...}); the generic IPPO training loop (marl/ippo.py)
        # expects every info value to be a flat (num_agents,) array it can
        # reshape against num_actors, which is what v1's own step_env
        # override does (overcooked_v1.py:137) before this ever reaches
        # training. Flattened here for the same reason, at the same layer.
        # Not folded into `rewards` -- unlike v1's do_reward_shaping, that's
        # a real, separate design decision (whether v2 populations train
        # against shaped or sparse reward) deliberately left open rather
        # than defaulted silently under this fix.
        shaped_reward = jnp.array([infos['shaped_reward'][agent] for agent in self.agents])
        new_info = {'shaped_reward': shaped_reward, 'base_reward': base_reward, 'base_return': base_return_so_far}
        base_return_so_far = jax.lax.select(dones['__all__'], jnp.zeros(self.num_agents), base_return_so_far)
        new_state = WrappedEnvState(env_state=env_state, base_return_so_far=base_return_so_far, avail_actions=jnp.zeros(self.num_agents), step=jnp.empty((), dtype=jnp.int32))
        return flat_obs, new_state, rewards, dones, new_info
