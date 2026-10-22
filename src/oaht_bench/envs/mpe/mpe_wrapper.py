"""Adapt a JaxMARL MPE cooperative task to the jax-aht training interface.

MPE is already a JaxMARL ``MultiAgentEnv`` (auto-resetting ``step``, ``reset``, spaces,
per-agent obs), so unlike the absorbed Jumanji wrappers this is thin. Two adaptations:

* ``get_avail_actions`` -- MPE has no illegal actions, so the base method raises; return
  all-ones masks (every action always available). The trainers ``jax.vmap`` this over the
  env batch, which broadcasts the constant to ``(num_envs, num_actions)``.
* shared rewards -- ``simple_reference`` gives each agent its own local reward; averaging
  across agents makes both seats optimise the *team* objective and makes the ego's
  reported return the team return, matching LBF's ``share_rewards=True`` and the
  fully-cooperative framing teammate generation assumes. ``simple_spread`` is already
  shared, so the mean is a no-op there.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxmarl.wrappers.baselines import JaxMARLWrapper


class MPECooperativeWrapper(JaxMARLWrapper):
    """All-ones availability + team-shared rewards over a JaxMARL MPE env."""

    def __init__(self, env, share_rewards: bool = True):
        super().__init__(env)
        self.share_rewards = share_rewards

    def get_avail_actions(self, state):
        return {
            agent: jnp.ones((self._env.action_space(agent).n,), dtype=jnp.float32)
            for agent in self._env.agents
        }

    def step(self, key, state, actions):
        obs, state, reward, done, info = self._env.step(key, state, actions)
        if self.share_rewards:
            team = sum(reward[a] for a in self._env.agents) / self._env.num_agents
            reward = {a: team for a in self._env.agents}
        return obs, state, reward, done, info
