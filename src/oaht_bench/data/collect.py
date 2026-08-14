"""Roll a seated team through an environment and record every transition.

``common/run_episodes.py`` returns only the final ``info`` — enough to score a
population, not enough to build a dataset. This records the full
``(obs, action, reward, done)`` sequence instead.

Seats are filled by iterating ``env.agents`` rather than naming ``agent_0`` and
``agent_1``, so the loop is already N-agent even though every current
environment has exactly two seats. That costs nothing here and keeps the
2-player assumption out of the artifact (see :mod:`oaht_bench.data.schema`).
"""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np


def collect_episode(
    rng,
    env,
    seat_params: Sequence[Any],
    policy,
    *,
    max_episode_steps: int,
    greedy: bool = False,
) -> dict[str, np.ndarray]:
    """Run one episode with ``seat_params[i]`` controlling ``env.agents[i]``.

    Returns arrays with a leading agent axis for per-agent quantities. Steps
    after termination are recorded but marked invalid, since the environment is
    scanned for a fixed length to stay jit-friendly.
    """
    agents = list(env.agents)
    n = len(agents)
    if len(seat_params) != n:
        raise ValueError(
            f"{len(seat_params)} parameter sets for {n} seats ({agents}). Every "
            f"seat needs an occupant."
        )

    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng)
    hstates = [policy.init_hstate(1, aux_info={"agent_id": i}) for i in range(n)]
    done_flags = {k: jnp.zeros((1,), dtype=bool) for k in agents + ["__all__"]}

    rec: dict[str, list] = {k: [] for k in
                            ("obs", "actions", "rewards", "avail", "dones", "valid")}
    finished = False

    for _ in range(max_episode_steps):
        avail = jax.lax.stop_gradient(env.get_avail_actions(state))
        step_obs, step_act, step_avail = [], [], []

        for i, name in enumerate(agents):
            rng, act_rng = jax.random.split(rng)
            a_i = avail[name].astype(jnp.float32)
            o_i = obs[name]
            act, hstates[i] = policy.get_action(
                params=seat_params[i],
                obs=o_i.reshape(1, 1, -1),
                done=done_flags[name].reshape(1, 1),
                avail_actions=a_i,
                hstate=hstates[i],
                rng=act_rng,
                # Conditional-critic policies accept aux_obs; at inference the
                # critic is unused, and crossplay already relies on None here.
                aux_obs=None,
                env_state=state,
                test_mode=greedy,
            )
            step_obs.append(np.asarray(o_i).reshape(-1))
            step_act.append(int(np.asarray(act).reshape(-1)[0]))
            step_avail.append(np.asarray(a_i).reshape(-1))

        rng, step_rng = jax.random.split(rng)
        env_act = {name: jnp.asarray(step_act[i]) for i, name in enumerate(agents)}
        obs, state, reward, done_flags, _ = env.step(step_rng, state, env_act)

        rec["obs"].append(np.stack(step_obs))
        rec["actions"].append(np.asarray(step_act))
        rec["avail"].append(np.stack(step_avail))
        rec["rewards"].append(
            np.asarray([float(np.asarray(reward[name]).reshape(-1)[0]) for name in agents])
        )
        ep_done = bool(np.asarray(done_flags["__all__"]).reshape(-1)[0])
        rec["dones"].append(ep_done)
        rec["valid"].append(not finished)
        if ep_done:
            finished = True
            break

    return {
        # (agent, T, ...) — transpose out of the per-step stacking order.
        "obs": np.stack(rec["obs"]).transpose(1, 0, 2),
        "actions": np.stack(rec["actions"]).T,
        "rewards": np.stack(rec["rewards"]).T,
        "avail_actions": np.stack(rec["avail"]).transpose(1, 0, 2),
        "dones": np.asarray(rec["dones"], dtype=bool),
        "valid": np.asarray(rec["valid"], dtype=bool),
    }


def pad_and_stack(episodes: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Pad ragged episodes to a common length and stack them.

    ``valid`` is what distinguishes a real terminal step from padding; consumers
    must mask with it rather than trusting ``dones``, which is all-False for a
    truncated episode.
    """
    T = max(e["dones"].shape[0] for e in episodes)

    def pad(arr, width, axis):
        if arr.shape[axis] == width:
            return arr
        pad_spec = [(0, 0)] * arr.ndim
        pad_spec[axis] = (0, width - arr.shape[axis])
        return np.pad(arr, pad_spec, mode="constant")

    return {
        "obs": np.stack([pad(e["obs"], T, 1) for e in episodes]),
        "actions": np.stack([pad(e["actions"], T, 1) for e in episodes]),
        "rewards": np.stack([pad(e["rewards"], T, 1) for e in episodes]),
        "avail_actions": np.stack([pad(e["avail_actions"], T, 1) for e in episodes]),
        "dones": np.stack([pad(e["dones"], T, 0) for e in episodes]),
        "valid": np.stack([pad(e["valid"], T, 0) for e in episodes]),
    }
