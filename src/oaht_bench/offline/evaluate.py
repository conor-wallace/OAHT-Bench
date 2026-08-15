"""Roll a trained policy into the environment against real teammates (§8).

Everything before this produces losses. A falling behaviour-cloning loss says
the policy predicts dataset actions better; it does not say the policy
coordinates, and on offline data those come apart — which is the whole reason
the benchmark exists. This is the first number worth reporting.

Return-conditioning follows the reference's deployment loop
(``offline_stage_2/utils.py:eval_episode_rtg``): the policy is given a target
return at reset, and after every step the target is **decremented by the reward
actually received**, so the conditioning tracks what remains rather than staying
fixed. Context is a rolling window of the last ``K`` steps, left-padded, matching
how the model was trained.

Results are reported **per teammate** as well as averaged. An average hides the
failure mode this benchmark is about: a policy that plays well with the
teammates resembling its training data and badly with the rest scores the same
as one that is uniformly mediocre.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvalScores:
    """Returns from evaluating one policy against a set of teammates."""

    #: teammate member index -> mean episode return for the ego agent
    per_teammate: dict[int, float]
    #: teammate member index -> standard error over episodes
    per_teammate_stderr: dict[int, float]
    episodes_per_teammate: int
    target_return: float

    @property
    def mean_return(self) -> float:
        """Averaged over teammates, not over episodes.

        Each teammate gets equal weight regardless of how many episodes it
        appeared in, so the number is not tilted by collection coverage.
        """
        return float(np.mean(list(self.per_teammate.values())))

    @property
    def worst_teammate_return(self) -> float:
        """The teammate the policy plays worst with.

        Ad-hoc teamwork is about the partners you did not train for, so the
        floor is at least as informative as the mean.
        """
        return float(min(self.per_teammate.values()))

    def describe(self) -> str:
        rows = "\n".join(
            f"    teammate {t:3d}   {v:7.4f} ± {self.per_teammate_stderr[t]:.4f}"
            for t, v in sorted(self.per_teammate.items())
        )
        return (
            f"target return {self.target_return:.4f}, "
            f"{self.episodes_per_teammate} episodes each\n{rows}\n"
            f"    mean {self.mean_return:.4f}   worst {self.worst_teammate_return:.4f}"
        )


def dataset_target_return(batch, *, quantile: float = 1.0) -> float:
    """Target return to condition on, taken from the data.

    The reference sets this per opponent from a config table. We have no such
    table, so it comes from the dataset the policy was trained on: the given
    quantile of per-episode ego return. Conditioning on the maximum asks the
    policy for the best behaviour the data contains, which is the usual
    Decision Transformer convention.
    """
    return float(np.quantile(batch.episode_returns()[:, batch.ego_index], quantile))


def _rollout(
    env,
    predict,
    teammate_params,
    teammate_policy,
    *,
    rng,
    context_length: int,
    max_episode_steps: int,
    target_return: float,
    ego_index: int,
    obs_dim: int,
):
    """One episode: ``predict`` drives the ego seat, the teammate drives the other.

    ``predict(rtg, obs, actions, timesteps, mask) -> logits`` is whatever the
    baseline supplies; keeping it a callable is what lets LIAM and TAO share this
    loop despite conditioning differently.
    """
    import jax
    import jax.numpy as jnp

    agents = list(env.agents)
    mate_seat = 1 - ego_index if len(agents) == 2 else None
    if mate_seat is None:
        raise ValueError(f"expected 2 seats for evaluation, got {agents}")

    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng)
    mate_hstate = teammate_policy.init_hstate(1, aux_info={"agent_id": mate_seat})
    done_flags = {k: jnp.zeros((1,), dtype=bool) for k in agents + ["__all__"]}

    K = context_length
    ctx_obs = np.zeros((K, obs_dim), dtype=np.float32)
    ctx_act = np.full(K, -10, dtype=np.int32)
    ctx_rtg = np.zeros(K, dtype=np.float32)
    ctx_t = np.zeros(K, dtype=np.int32)
    ctx_mask = np.zeros(K, dtype=bool)

    rtg = float(target_return)
    total = 0.0
    for t in range(max_episode_steps):
        # Shift the window left and write the current step at the end, which is
        # the position the left-padded training windows put "now".
        ctx_obs = np.roll(ctx_obs, -1, axis=0)
        ctx_act = np.roll(ctx_act, -1)
        ctx_rtg = np.roll(ctx_rtg, -1)
        ctx_t = np.roll(ctx_t, -1)
        ctx_mask = np.roll(ctx_mask, -1)
        ctx_obs[-1] = np.asarray(obs[agents[ego_index]]).reshape(-1)
        ctx_act[-1] = -10  # the ego has not acted yet at t; predicted from o_t
        ctx_rtg[-1] = rtg
        ctx_t[-1] = min(t + 1, K * 64)
        ctx_mask[-1] = True

        logits = predict(
            jnp.asarray(ctx_rtg)[None],
            jnp.asarray(ctx_obs)[None],
            jnp.asarray(ctx_act)[None],
            jnp.asarray(ctx_t)[None],
            jnp.asarray(ctx_mask)[None],
        )
        rng, act_rng = jax.random.split(rng)
        ego_action = int(jax.random.categorical(act_rng, logits[0, -1]))
        ctx_act[-1] = ego_action

        rng, mate_rng = jax.random.split(rng)
        mate_action, mate_hstate = teammate_policy.get_action(
            params=teammate_params,
            obs=obs[agents[mate_seat]].reshape(1, 1, -1),
            done=done_flags[agents[mate_seat]].reshape(1, 1),
            avail_actions=env.get_avail_actions(state)[agents[mate_seat]].astype(jnp.float32),
            hstate=mate_hstate,
            rng=mate_rng,
            aux_obs=None,
            env_state=state,
            test_mode=False,
        )

        acts = [None, None]
        acts[ego_index] = jnp.asarray(ego_action)
        acts[mate_seat] = jnp.asarray(int(np.asarray(mate_action).reshape(-1)[0]))
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done_flags, _ = env.step(
            step_rng, state, {a: acts[i] for i, a in enumerate(agents)}
        )

        r = float(np.asarray(reward[agents[ego_index]]).reshape(-1)[0])
        total += r
        # Reference: the target tracks what is left, not what was asked for.
        rtg -= r
        if bool(np.asarray(done_flags["__all__"]).reshape(-1)[0]):
            break
    return total


def evaluate(
    predict,
    env,
    loaded,
    members,
    *,
    rng,
    context_length: int,
    max_episode_steps: int,
    target_return: float,
    num_episodes: int = 20,
    ego_index: int = 0,
    obs_dim: int,
) -> EvalScores:
    """Play ``num_episodes`` against each member and record the returns.

    Args:
        predict: ``(rtg, obs, actions, timesteps, mask) -> logits`` for the ego.
            For TAO this closes over the teammate embedding; for LIAM over the
            frozen encoder. The loop does not need to know which.
        loaded: The population being evaluated against, from
            :func:`~oaht_bench.population.loading.population_from_run`.
        members: Which members to play against.
    """
    import jax

    from oaht_bench.population.members import get_member_params

    per_teammate, stderr = {}, {}
    for m in members:
        # Paired generators seat a best response opposite a confederate; here the
        # learned policy takes the ego seat, so the teammate is the confederate.
        mate_params = get_member_params(loaded.params, int(m))
        returns = []
        for _ in range(num_episodes):
            rng, ep_rng = jax.random.split(rng)
            returns.append(
                _rollout(
                    env, predict, mate_params, loaded.policy_cls,
                    rng=ep_rng, context_length=context_length,
                    max_episode_steps=max_episode_steps,
                    target_return=target_return, ego_index=ego_index, obs_dim=obs_dim,
                )
            )
        arr = np.asarray(returns)
        per_teammate[int(m)] = float(arr.mean())
        stderr[int(m)] = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0

    return EvalScores(
        per_teammate=per_teammate,
        per_teammate_stderr=stderr,
        episodes_per_teammate=num_episodes,
        target_return=float(target_return),
    )
