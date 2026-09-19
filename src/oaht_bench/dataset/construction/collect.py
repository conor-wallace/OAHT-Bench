"""Roll a seated team through an environment and record every transition.

``common/run_episodes.py`` returns only the final ``info`` — enough to score a
population, not enough to build a dataset. This records the full
``(obs, action, reward, done)`` sequence instead.

Seats are filled by iterating ``env.agents`` rather than naming ``agent_0`` and
``agent_1``, so the loop is already N-agent even though every current
environment has exactly two seats. That costs nothing here and keeps the
2-player assumption out of the artifact (see :mod:`oaht_bench.dataset.schema`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from oaht_bench.dataset.schema import Episode


def collect_episode(
    rng,
    env,
    seats: Sequence[tuple[Any, Any]],
    *,
    max_episode_steps: int,
    greedy: bool | Sequence[bool] = False,
    epsilon: float = 0.0,
    noisy_seats: Sequence[int] | None = None,
) -> Episode:
    """Run one episode with ``seats[i]`` controlling ``env.agents[i]``.

    Each seat is its own ``(params, policy)`` pair rather than sharing one
    policy, because BRDiv and L-BRDiv give the two seats different roles: a
    confederate and its best response. It also leaves room for a heuristic
    teammate opposite a learned one without changing this signature again.

    ``epsilon`` injects ε-greedy noise: with probability ``epsilon`` a seat in
    ``noisy_seats`` (default: all seats) takes a uniform *legal* action instead of
    its policy's. The taken action is what gets recorded, so pointing noise at the
    *teammate* seat broadens the state distribution (recovery coverage for a BC
    ego) while leaving the ego seat's recorded actions expert.

    Returns an :class:`~oaht_bench.dataset.schema.Episode` with a leading agent
    axis for per-agent quantities. The loop breaks on termination, so every
    recorded step is real -- there is no padding to mark.
    """
    agents = list(env.agents)
    n = len(agents)
    if len(seats) != n:
        raise ValueError(f"{len(seats)} occupants for {n} seats ({agents}). Every seat needs one.")
    seat_params = [p for p, _ in seats]
    seat_policies = [pol for _, pol in seats]
    # ``greedy`` may be one bool for all seats or one per seat, so an argmax ego
    # can be paired with a sampled teammate (the benchmark's eval regime) without
    # a separate code path.
    greedy_seats = list(greedy) if isinstance(greedy, (list, tuple)) else [greedy] * n

    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng)
    hstates = [seat_policies[i].init_hstate(1, aux_info={"agent_id": i}) for i in range(n)]
    done_flags = {k: jnp.zeros((1,), dtype=bool) for k in agents + ["__all__"]}

    rec: dict[str, list] = {k: [] for k in ("obs", "actions", "rewards", "avail", "dones")}

    for _ in range(max_episode_steps):
        avail = jax.lax.stop_gradient(env.get_avail_actions(state))
        step_obs, step_act, step_avail = [], [], []

        for i, name in enumerate(agents):
            rng, act_rng = jax.random.split(rng)
            a_i = avail[name].astype(jnp.float32)
            o_i = obs[name]
            act, hstates[i] = seat_policies[i].get_action(
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
                test_mode=greedy_seats[i],
            )
            act_i = int(np.asarray(act).reshape(-1)[0])
            if epsilon > 0.0 and (noisy_seats is None or i in noisy_seats):
                rng, coin_rng = jax.random.split(rng)
                if float(jax.random.uniform(coin_rng)) < epsilon:
                    legal = np.flatnonzero(np.asarray(a_i).reshape(-1) > 0)
                    if len(legal) > 0:
                        rng, pick_rng = jax.random.split(rng)
                        act_i = int(legal[int(jax.random.randint(pick_rng, (), 0, len(legal)))])
            step_obs.append(np.asarray(o_i).reshape(-1))
            step_act.append(act_i)
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
        if ep_done:
            break

    return Episode(
        # (agent, T, ...) — transpose out of the per-step stacking order.
        obs=np.stack(rec["obs"]).transpose(1, 0, 2),
        actions=np.stack(rec["actions"]).T,
        rewards=np.stack(rec["rewards"]).T,
        avail_actions=np.stack(rec["avail"]).transpose(1, 0, 2),
        dones=np.asarray(rec["dones"], dtype=bool),
    )


def _tree_freeze(frozen, keep, new):
    """``keep`` where ``frozen`` (a scalar bool), else ``new`` -- leafwise."""
    return jax.tree_util.tree_map(lambda k, v: jnp.where(frozen, k, v), keep, new)


#: Compiled batched rollouts, keyed by everything static about them; params are
#: passed as arguments, so repeated calls that differ only in parameters -- every
#: sub-chunk of a pairing, and every pairing that shares a generator's policy -- reuse
#: one compiled executable. Same rationale (and the same recompile-per-call bug it
#: avoids) as ``run_episodes._ROLLOUT_CACHE``. Key objects are pinned so ids can't be
#: reused while cached.
_BATCH_ROLLOUT_CACHE: dict = {}


def collect_episodes_batched(
    rng,
    env,
    seats: Sequence[tuple[Any, Any]],
    *,
    max_episode_steps: int,
    num_episodes: int,
    batch_size: int = 256,
    greedy: bool | Sequence[bool] = False,
) -> list[Episode]:
    """Collect ``num_episodes`` of one seating in a single vmapped, scanned device call.

    Equivalent to calling :func:`collect_episode` on each of
    ``jax.random.split(rng, num_episodes)`` with ``epsilon=0``, but the whole batch
    runs on device (``lax.scan`` over steps, ``vmap`` over episodes) instead of a
    Python per-step loop -- one to two orders of magnitude faster, and it actually
    uses the accelerator. Every lane shares the *same* seating (policies + params),
    so this batches one ``(ego, teammate)`` pairing; a caller with a mixed plan groups
    it by pairing first.

    The scan runs the full ``max_episode_steps`` and freezes each lane after its
    episode ends (like :func:`~oaht_bench.common.run_episodes.run_single_episode`);
    each episode is then sliced back to its real length, so the returned ragged
    :class:`~oaht_bench.dataset.schema.Episode`\\ s are transition-for-transition
    identical to the eager path. ``epsilon`` noise is not supported here (the
    production collection path uses none); use :func:`collect_episode` for that.
    """
    agents = list(env.agents)
    n = len(agents)
    if len(seats) != n:
        raise ValueError(f"{len(seats)} occupants for {n} seats ({agents}).")
    seat_params = [p for p, _ in seats]
    seat_policies = [pol for _, pol in seats]
    greedy_seats = list(greedy) if isinstance(greedy, (list, tuple)) else [greedy] * n

    def one(ep_rng, seat_params):
        rng, reset_rng = jax.random.split(ep_rng)
        obs0, state0 = env.reset(reset_rng)
        hstates0 = [seat_policies[i].init_hstate(1, aux_info={"agent_id": i}) for i in range(n)]
        done0 = {k: jnp.zeros((1,), dtype=bool) for k in agents + ["__all__"]}

        def step_fn(carry, _):
            state, obs, done_flags, hstates, rng, frozen = carry
            avail = jax.lax.stop_gradient(env.get_avail_actions(state))
            rec_obs, rec_act, rec_avail, new_h = [], [], [], []
            for i, name in enumerate(agents):
                rng, act_rng = jax.random.split(rng)
                av = avail[name].astype(jnp.float32)
                act, h = seat_policies[i].get_action(
                    params=seat_params[i],
                    obs=obs[name].reshape(1, 1, -1),
                    done=done_flags[name].reshape(1, 1),
                    avail_actions=av,
                    hstate=hstates[i],
                    rng=act_rng,
                    aux_obs=None,
                    env_state=state,
                    test_mode=greedy_seats[i],
                )
                rec_obs.append(obs[name].reshape(-1))
                rec_act.append(act.reshape(-1)[0].astype(jnp.int32))
                rec_avail.append(av.reshape(-1))
                new_h.append(h)
            rng, step_rng = jax.random.split(rng)
            env_act = {name: rec_act[i] for i, name in enumerate(agents)}
            nobs, nstate, reward, ndone, _ = env.step(step_rng, state, env_act)
            step_done = ndone["__all__"].reshape(-1)[0]
            carry_out = (
                _tree_freeze(frozen, state, nstate),
                _tree_freeze(frozen, obs, nobs),
                _tree_freeze(frozen, done_flags, ndone),
                _tree_freeze(frozen, hstates, new_h),
                rng,
                frozen | step_done,
            )
            ys = (
                jnp.stack(rec_obs),  # (n, obs_dim) -- pre-step observation acted on
                jnp.stack(rec_act),  # (n,)
                jnp.stack([reward[name].reshape(-1)[0] for name in agents]),  # (n,)
                jnp.stack(rec_avail),  # (n, num_actions)
                frozen | step_done,  # episode-done flag recorded this step
            )
            return carry_out, ys

        init = (state0, obs0, done0, hstates0, rng, jnp.asarray(False))
        _, ys = jax.lax.scan(step_fn, init, None, length=max_episode_steps)
        return ys

    key = (
        id(env),
        tuple(id(p) for p in seat_policies),
        tuple(greedy_seats),
        int(max_episode_steps),
    )
    cached = _BATCH_ROLLOUT_CACHE.get(key)
    if cached is None:

        @jax.jit
        def rollout(ep_rngs, seat_params):
            return jax.vmap(lambda r: one(r, seat_params))(ep_rngs)

        _BATCH_ROLLOUT_CACHE[key] = (rollout, env, seat_policies)
        cached = _BATCH_ROLLOUT_CACHE[key]
    rollout = cached[0]
    ep_rngs = jax.random.split(rng, num_episodes)

    episodes: list[Episode] = []
    # Chunk the vmap so a large group does not materialize (num_episodes x T x
    # obs_dim) on device at once. All full chunks share the compiled rollout; a
    # smaller final chunk recompiles once.
    for start in range(0, num_episodes, batch_size):
        chunk = ep_rngs[start : start + batch_size]
        obs, acts, rews, avail, dones = rollout(chunk, seat_params)
        obs, acts, rews = np.asarray(obs), np.asarray(acts), np.asarray(rews)
        avail, dones = np.asarray(avail), np.asarray(dones)
        any_done = dones.any(axis=1)
        lengths = np.where(any_done, dones.argmax(axis=1) + 1, max_episode_steps)
        for b in range(len(chunk)):
            length = int(lengths[b])
            episodes.append(
                Episode(
                    obs=obs[b, :length].transpose(1, 0, 2).astype(np.float32),
                    actions=acts[b, :length].transpose(1, 0).astype(np.int64),
                    rewards=rews[b, :length].transpose(1, 0).astype(np.float32),
                    avail_actions=avail[b, :length].transpose(1, 0, 2).astype(np.float32),
                    dones=dones[b, :length].astype(bool),
                )
            )
    return episodes
