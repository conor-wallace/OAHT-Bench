"""Annealed dense-reward shaping for the Overcooked-v2 generators.

The Overcooked-v2 wrapper computes a per-step ``info['shaped_reward']`` -- dense
sub-task rewards for placing an ingredient in a pot, picking up a plate while a
dish cooks, picking up a dish -- but returns only the sparse base (delivery)
reward for training (``overcooked_v2_wrapper.py``). Sparse Overcooked is a
long-horizon exploration problem, so following the source paper (Gessler et al.,
ICLR 2025, App. C/D) we add the shaped reward to the base reward with a **linear
1 -> 0 anneal over ``reward_shaping_horizon`` environment steps**: early training
is dense enough to discover the delivery loop, late training optimises the true
sparse task.

Environments that never surface ``shaped_reward`` (LBF, Hanabi, Overcooked-v1) and
any config with ``reward_shaping_horizon == 0`` are untouched -- the guards below
are Python-level (on the static horizon and the static info key), so nothing extra
is traced for them.
"""

from __future__ import annotations

import jax.numpy as jnp


def shaping_coef(horizon: float, global_env_step):
    """Linear anneal from 1 at step 0 to 0 at ``horizon`` (flat 0 after)."""
    return jnp.clip(1.0 - global_env_step / horizon, 0.0, 1.0)


def add_shaped_reward(reward: dict, info: dict, agents, *, horizon: float, global_env_step) -> dict:
    """Fold ``info['shaped_reward']`` into a per-agent base-reward dict.

    ``reward`` is ``{agent: (num_envs,)}`` and ``info['shaped_reward']`` is
    ``(num_envs, num_agents)`` -- both as they come off ``jax.vmap(env.step)``,
    before any per-agent reshaping. Returns ``reward`` unchanged when
    ``horizon <= 0`` or when the env does not surface a shaped reward.
    """
    if horizon <= 0 or "shaped_reward" not in info:
        return reward
    coef = shaping_coef(horizon, global_env_step)
    shaped = info["shaped_reward"]
    return {agent: reward[agent] + coef * shaped[:, i] for i, agent in enumerate(agents)}
