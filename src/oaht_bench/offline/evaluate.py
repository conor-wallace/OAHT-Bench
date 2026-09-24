"""Roll a trained policy into the environment against real teammates (§8).

Everything before this produces losses. A falling behaviour-cloning loss says
the policy predicts dataset actions better; it does not say the policy
coordinates, and on offline data those come apart — which is the whole reason
the benchmark exists. This is the first number worth reporting.

Each offline baseline is a
:class:`~oaht_bench.models.return_conditioned_agent.ReturnConditionedAgent`, so
:func:`evaluate_agent` drives it through the shared vmapped ``run_episodes`` loop:
the return-conditioning deployment (a target return decremented by the reward
received, over a rolling left-padded ``K``-window matching training) lives in the
agent's ``get_action``, not here.

Results are reported **per teammate** as well as averaged. An average hides the
failure mode this benchmark is about: a policy that plays well with the
teammates resembling its training data and badly with the rest scores the same
as one that is uniformly mediocre.

**The conditioning target is per teammate, not one dataset-wide number.**
TAO's own reference computes ``OPPO_TARGET[i] = max over egos of that ego's
mean return against teammate i`` -- the best response's expected return
against *that specific* teammate (``offline_stage_2/utils.py``), not a single
value shared by every rollout. That is exactly a column-max over a
crossplay-style matrix, and :mod:`oaht_bench.population.pooled_crossplay`
already computes one for dataset collection with the ego axis being the
trained ``ppo_br`` population -- so a teammate's column max there already
*is* ``OPPO_TARGET`` for that teammate, with no separate computation needed.
:func:`resolve_target_returns` reads it from there when a dataset was
collected in pooled mode (``pooled_matrix_path`` in its meta), applied
identically to every baseline -- not a TAO-specific branch. Datasets with no
matrix (legacy / single-population collections) fall back to
:func:`dataset_target_return`'s single dataset-wide value, broadcast to every
teammate, which is exactly today's behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvalScores:
    """Returns from evaluating one policy against a set of teammates.

    Teammate keys are member indices in the legacy single-population path and
    ``generator:member:role`` labels in the held-out path (test teammates span
    generators, so an int index no longer identifies one) -- so the key type is
    left open.
    """

    #: teammate label -> mean episode return for the ego agent
    per_teammate: dict
    #: teammate label -> standard error over episodes
    per_teammate_stderr: dict
    episodes_per_teammate: int
    #: The mean of per_teammate_target_return's values -- kept as a single
    #: number because every existing reader (describe(), runner.py's
    #: result["target_return"]) expects one; the detail lives alongside it.
    target_return: float
    #: teammate label -> the RTG target that teammate was actually conditioned
    #: on. None only for callers that still pass a bare float (not expected
    #: from this module's own functions, which always resolve per teammate).
    per_teammate_target_return: dict | None = None
    #: teammate label -> online mate_action_acc, or None if the agent has no
    #: mate_action_logits (BC) -- see docs/tuning_record.md. Decoded from the
    #: SAME window get_action conditioned the policy on, at every step of the
    #: SAME rollout the return above comes from, not a separate dataset pass.
    per_teammate_mate_acc: dict | None = None
    #: The modal-action floor, POOLED across every teammate's true actions (not
    #: a per-teammate floor averaged after the fact -- that would silently use
    #: teammate identity, which a floor is supposed to represent NOT having).
    #: Report mate_action_acc against this, never alone -- an accuracy number
    #: by itself is not evidence of anything (docs/tuning_record.md).
    mate_action_floor: float | None = None

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

    @property
    def mate_action_acc(self) -> float | None:
        """Averaged over teammates -- None if this baseline has no mate model."""
        if not self.per_teammate_mate_acc:
            return None
        return float(np.mean(list(self.per_teammate_mate_acc.values())))

    def describe(self) -> str:
        rows = "\n".join(
            f"    teammate {str(t):>14}   {v:7.4f} ± {self.per_teammate_stderr[t]:.4f}"
            for t, v in sorted(self.per_teammate.items(), key=lambda kv: str(kv[0]))
        )
        mate = ""
        if self.per_teammate_mate_acc:
            mate = (
                f"\n    mate_action_acc {self.mate_action_acc:.4f}"
                f"  (modal floor {self.mate_action_floor:.4f})"
            )
        return (
            f"target return {self.target_return:.4f}, "
            f"{self.episodes_per_teammate} episodes each\n{rows}\n"
            f"    mean {self.mean_return:.4f}   worst {self.worst_teammate_return:.4f}{mate}"
        )


def dataset_target_return(batch, *, quantile: float = 1.0) -> float:
    """One dataset-wide target return, taken from the data.

    The fallback used when there is no per-teammate table to read (no
    ``pooled_matrix_path`` in the dataset's meta -- see
    :func:`resolve_target_returns`, which prefers a per-teammate value when
    one is available): the given quantile of per-episode ego return across
    the *whole* dataset. Conditioning on the maximum asks the policy for the
    best behaviour the data contains, which is the usual Decision Transformer
    convention.
    """
    return float(np.quantile(batch.episode_returns()[:, batch.ego_index], quantile))


def crossplay_target_returns(pooled_matrix_path: str, teammates) -> dict:
    """Per-teammate best-response return, read off the pooled crossplay matrix.

    TAO's reference computes ``OPPO_TARGET[i] = max over egos of that ego's
    mean return against teammate i`` from the training data
    (``offline_stage_2/utils.py``). That is exactly a column-max over a
    crossplay-style matrix, and :mod:`oaht_bench.population.pooled_crossplay`
    already produces one for dataset collection -- since its ego axis is the
    trained ``ppo_br`` population (one dedicated best response per teammate),
    a teammate's column max there already *is* ``OPPO_TARGET`` for that
    teammate. Works the same against an older, pre-``ppo_br`` matrix too
    (whatever egos happen to be in the matrix), just with a lower ceiling --
    the lookup itself doesn't care which matrix vintage it reads.

    ``teammates`` is ``[(label, params, policy_cls)]`` (only ``label`` is
    used); labels are the ``"generator:member:role"`` strings
    :func:`~oaht_bench.offline.runner._teammate_policies` already emits.
    Raises if a teammate's identity isn't a column in the matrix -- a stale
    or mismatched matrix should fail loudly, not silently skip a teammate.
    """
    from oaht_bench.dataset.construction.epsilon_sampler import load_pooled

    pooled = load_pooled(pooled_matrix_path)
    index = {
        (str(pooled.generator[i]), int(pooled.member[i]), str(pooled.role[i])): i
        for i in range(pooled.size)
    }
    out = {}
    for label, _, _ in teammates:
        generator, member, role = label.split(":")
        key = (generator, int(member), role)
        if key not in index:
            raise ValueError(
                f"teammate {label!r} is not a column in the crossplay matrix at "
                f"{pooled_matrix_path!r} -- stale or mismatched matrix for this "
                f"dataset's teammates."
            )
        out[label] = float(pooled.matrix[:, index[key]].max())
    return out


def resolve_target_returns(meta: dict, teammates, *, norm=None, fallback_batch=None) -> dict:
    """The RTG target every baseline conditions on, per teammate.

    Prefers :func:`crossplay_target_returns` when the dataset was collected in
    pooled mode (``meta["pooled_matrix_path"]`` set); otherwise falls back to
    :func:`dataset_target_return`'s single dataset-wide value, broadcast to
    every teammate label (``fallback_batch`` is then required -- raises
    otherwise, rather than silently returning a wrong number). ``norm``
    applies the same return-to-go rescaling training used
    (``norm.apply_rtg``), when given.
    """
    matrix_path = meta.get("pooled_matrix_path")
    if matrix_path:
        raw = crossplay_target_returns(matrix_path, teammates)
    else:
        if fallback_batch is None:
            raise ValueError(
                "meta has no pooled_matrix_path, so a dataset-wide fallback is "
                "needed, but no fallback_batch was given to compute it from."
            )
        value = dataset_target_return(fallback_batch)
        raw = {label: value for label, _, _ in teammates}
    if norm is None:
        return raw
    return {label: float(norm.apply_rtg(value)) for label, value in raw.items()}


def evaluate_agent(
    agent,
    params,
    env,
    loaded,
    members,
    *,
    rng,
    target_returns: dict,
    max_episode_steps: int,
    num_episodes: int = 20,
    ego_index: int = 0,
) -> EvalScores:
    """Play each teammate ``num_episodes`` times and record the ego's returns.

    The ego is a :class:`~oaht_bench.models.agent_interface.AgentPolicy` (every
    offline baseline is a
    :class:`~oaht_bench.models.return_conditioned_agent.ReturnConditionedAgent`),
    driven through the shared, vmapped
    :func:`~oaht_bench.common.run_episodes.run_episodes`. The rolling window and
    return-to-go bookkeeping live in the agent, so nothing here needs the context
    length, observation dimension or normalisation -- they are baked into the
    agent. The ego takes seat ``agent_0``; the return is the ego seat of
    LogWrapper's per-agent episode return.
    """
    from oaht_bench.population.members import get_member_params

    teammates = [
        (int(m), get_member_params(loaded.params, int(m)), loaded.policy_cls) for m in members
    ]
    return evaluate_agent_against(
        agent,
        params,
        env,
        teammates,
        rng=rng,
        target_returns=target_returns,
        max_episode_steps=max_episode_steps,
        num_episodes=num_episodes,
        ego_index=ego_index,
    )


def _agent_probes_mate_action(agent, params) -> bool:
    """Whether ``agent.mate_action_logits`` is implemented for this baseline.

    Checked once against a dummy (zero) window, outside jit, so
    :func:`evaluate_agent_against` can pick its rollout path at trace time
    rather than branching every step. ``ReturnConditionedAgent``'s default
    returns ``None``; every override returns an array.
    """
    return agent.mate_action_logits(params, agent.init_hstate(1)) is not None


def _rollout_with_mate_probe(
    rng,
    env,
    agent,
    agent_params,
    mate_params,
    mate_policy,
    *,
    max_episode_steps,
    num_eps,
    ego_index=0,
    greedy=False,
):
    """Like ``run_episodes``, plus the online mate-action probe.

    The ego's action comes from ``agent.get_action`` -- unmodified, the exact
    function every other rollout in this codebase uses -- so this never drifts
    from production the way a hand-copied window replica can (it did, twice,
    while building the diagnostic this replaces; see docs/tuning_record.md).
    The probe calls ``agent.mate_action_logits`` on the ``ContextWindow``
    ``get_action`` just returned; see that method's docstring for why decoding
    from the post-action window is equivalent to decoding from the window
    ``get_action`` actually conditioned on.

    Mirrors ``run_episodes``'s ``_compiled_rollout`` freeze-on-done guard
    (``jax.lax.cond(done, freeze, take_step)``) -- omitting it inflates the
    return by continuing to step (and accrue reward) past episode end, and
    masks the teammate's illegal actions before scoring the probe (both bugs
    hit and fixed while building the diagnostic this replaces).
    """
    import jax
    import jax.numpy as jnp

    from oaht_bench.models.masking import mask_logits

    def one(key):
        k, rk = jax.random.split(key)
        obs0, state0 = env.reset(rk)
        h0 = agent.init_hstate(1, aux_info={"agent_id": 0})
        h1 = mate_policy.init_hstate(1, aux_info={"agent_id": 1})
        ao0 = jnp.zeros((1, 1, agent.action_dim))
        ao1 = jnp.zeros((1, 1, agent.action_dim))
        rw0 = jnp.zeros((1, 1, 1))
        ego_ret0 = jnp.zeros(())
        done0 = jnp.asarray(False)

        def take_step(carry):
            state, obs, h0, h1, ao0, ao1, rw0, k, ego_ret, _done = carry
            k, k0, k1, ks = jax.random.split(k, 4)
            av = jax.lax.stop_gradient(env.get_avail_actions(state))
            joint = jnp.concatenate((ao0, ao1), axis=-1)

            a0, h0n = agent.get_action(
                params=agent_params,
                obs=obs["agent_0"].reshape(1, 1, -1),
                done=jnp.zeros((1, 1), bool),
                avail_actions=av["agent_0"].astype(jnp.float32),
                hstate=h0,
                rng=k0,
                aux_obs=(ao0, joint, rw0),
                env_state=state,
                test_mode=greedy,
                reward=rw0,
            )
            avail1 = jnp.reshape(av["agent_1"], (-1)).astype(jnp.float32)
            pred_mate = jnp.argmax(
                mask_logits(agent.mate_action_logits(agent_params, h0n), avail1)
            ).astype(jnp.int32)

            a1, h1n = mate_policy.get_action(
                params=mate_params,
                obs=obs["agent_1"].reshape(1, 1, -1),
                done=jnp.zeros((1, 1), bool),
                avail_actions=av["agent_1"].astype(jnp.float32),
                hstate=h1,
                rng=k1,
                aux_obs=None,
                env_state=state,
                test_mode=False,
            )
            a0s, a1s = a0.squeeze(), a1.squeeze()
            act = {"agent_0": a0s, "agent_1": a1s}
            obs2, state2, r2, done2, info2 = env.step(ks, state, act)
            n0 = jax.nn.one_hot(a0s, agent.action_dim).reshape(1, 1, -1)
            n1 = jax.nn.one_hot(a1s, agent.action_dim).reshape(1, 1, -1)
            new_ego_ret = jax.lax.select(
                done2["__all__"],
                jnp.asarray(info2["returned_episode_returns"]).reshape(-1)[ego_index],
                ego_ret,
            )
            new_carry = (
                state2,
                obs2,
                h0n,
                h1n,
                n0,
                n1,
                r2["agent_0"].reshape(1, 1, 1),
                k,
                new_ego_ret,
                done2["__all__"],
            )
            return new_carry, (pred_mate, a1s)

        def step(carry, _):
            *_, done = carry
            new_carry, (pred, true) = jax.lax.cond(
                done, lambda c: (c, (jnp.int32(0), jnp.int32(0))), take_step, carry
            )
            return new_carry, (pred, true, ~done)

        init_carry = (state0, obs0, h0, h1, ao0, ao1, rw0, k, ego_ret0, done0)
        final_carry, (pred, true, valid) = jax.lax.scan(
            step, init_carry, None, length=max_episode_steps
        )
        return final_carry[-2], pred, true, valid

    return jax.vmap(one)(jax.random.split(rng, num_eps))


def evaluate_agent_against(
    agent,
    params,
    env,
    teammates,
    *,
    rng,
    target_returns: dict,
    max_episode_steps: int,
    num_episodes: int = 20,
    ego_index: int = 0,
    greedy: bool = False,
) -> EvalScores:
    """Roll the ego against an explicit list of teammate policies.

    ``teammates`` is ``[(label, mate_params, policy_cls)]`` -- a label, the
    teammate's parameters, and its policy class. Unlike :func:`evaluate_agent`,
    which draws members from a single loaded population, this accepts teammates
    from *different* populations (each with its own ``policy_cls``), which is what
    a held-out set spanning generators requires (§8). The ego takes seat
    ``agent_0``; each teammate plays ``num_episodes`` episodes in the other seat.

    ``target_returns`` is ``{label: target}`` (see :func:`resolve_target_returns`)
    -- every teammate can get a different conditioning target, not one value
    shared by the whole rollout. Set via ``agent.set_target_return`` right
    before that teammate's episodes, so it's already in effect by the time
    ``init_hstate`` is called for them.

    ``greedy`` makes only the *ego* act by argmax (the teammate keeps sampling),
    which does not risk the symmetric-argmax deadlock invariant #2 guards against
    and is a diagnostic for whether a high-accuracy policy is being sunk by
    sampling noise rather than a train/deploy mismatch. Defaults False (sampled),
    so the benchmark metric is unchanged.

    When ``agent.mate_action_logits`` is implemented (LIAM/MeLIBA/OMIS; not BC,
    not TAO -- see that method's docstring), the same rollout also scores
    ``mate_action_acc`` at no extra environment cost. Otherwise this is the
    unmodified, shared-cache ``run_episodes`` path.
    """
    import jax

    from oaht_bench.common.run_episodes import run_episodes

    probe = _agent_probes_mate_action(agent, params)
    per_teammate, stderr = {}, {}
    mate_acc = {} if probe else None
    pooled_true = []
    for label, mate_params, policy_cls in teammates:
        agent.set_target_return(target_returns[label])
        rng, ep_rng = jax.random.split(rng)
        if probe:
            returns, pred, true, valid = _rollout_with_mate_probe(
                ep_rng,
                env,
                agent,
                params,
                mate_params,
                policy_cls,
                max_episode_steps=max_episode_steps,
                num_eps=num_episodes,
                ego_index=ego_index,
                greedy=greedy,
            )
            returns = np.asarray(returns)
            v = np.asarray(valid).astype(bool)
            pred, true = np.asarray(pred), np.asarray(true)
            mate_acc[label] = float((pred == true)[v].mean())
            pooled_true.append(true[v])
        else:
            out = run_episodes(
                ep_rng,
                env,
                agent_0_param=params,
                agent_0_policy=agent,
                agent_1_param=mate_params,
                agent_1_policy=policy_cls,
                max_episode_steps=max_episode_steps,
                num_eps=num_episodes,
                agent_0_test_mode=greedy,
            )
            returns = np.asarray(out["returned_episode_returns"])[:, ego_index]
        per_teammate[label] = float(returns.mean())
        stderr[label] = (
            float(returns.std(ddof=1) / np.sqrt(len(returns))) if len(returns) > 1 else 0.0
        )

    mate_floor = None
    if probe and pooled_true:
        pooled = np.concatenate(pooled_true)
        mate_floor = float(np.bincount(pooled, minlength=agent.action_dim).max() / pooled.size)

    return EvalScores(
        per_teammate=per_teammate,
        per_teammate_stderr=stderr,
        episodes_per_teammate=num_episodes,
        target_return=float(np.mean([target_returns[label] for label, *_ in teammates])),
        per_teammate_target_return=dict(target_returns),
        per_teammate_mate_acc=mate_acc,
        mate_action_floor=mate_floor,
    )
