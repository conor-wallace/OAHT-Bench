#!/usr/bin/env python3
"""Replicates TIPR Appendix D (Fig. 9, Eq. 33): pair-wise expected KL
divergence between the Seen-Set opponent (teammate) policies used to
construct an offline dataset -- a direct, independent measure of how
behaviorally diverse the training population actually is, upstream of
anything TAO's training does with it. Where
``diagnose_tao_embedding_separability.py`` asks "did the encoder learn to
tell teammates apart", this asks the prior question: "is there enough real
behavioral difference between teammates for *any* encoder to learn from in
the first place" -- a ceiling the training procedure cannot exceed no matter
how it's tuned.

**What the paper does (papers/tipr.pdf pp. 26-27).** For policies
pi^{-1,i}, pi^{-1,j} in the Seen Set:

    D_KL(pi_i || pi_j) = E_{o ~ P(o)} [ sum_a pi_i(a|o) * log(pi_i(a|o)/pi_j(a|o)) ]   (Eq. 33)

the standard forward KL between their action distributions, in expectation
over an opponent observation distribution P(o). To approximate P(o) for
Seen-Set policy k: sample 1200 episodes with the opponent FIXED at k while
the self-agent "traverses through all the opponent policies" -- cycles
through all 12 dedicated best responses, 100 episodes each -- and collect k's
own observation stream across them. D_KL(pi_i || pi_j) for any pair is then
estimated using the sample collected while opponent i was actually active (so
the expectation is over states i naturally visits), evaluating both pi_i and
pi_j's action probabilities at those same observations.

12 Seen-Set policies -> a 12x12 matrix (diagonal 0 by construction). The
paper reports a "dissimilarity rate": the fraction of pairs whose KL exceeds
1.0, as a one-number diversity summary (LBF: 80.56%).

**OMIS's own Appendix G (papers/omis.pdf pp. 30-32) describes near-identical
methodology for the same MEP mechanism but reports a different number: 66.25%
for LBF, not TIPR's 80.56%.** Two concrete differences from TIPR's recipe,
both directly checkable here:
  - OMIS measures the FULL released population (10 Seen + 10 Unseen = 20 for
    its own MEP population size), not a Seen-only subset -- 1000 episodes per
    fixed opponent, 20 egos x 50 episodes each (TIPR: 1200 episodes, 12 egos x
    100 each). Pass --include-unseen to match this; costs nothing extra to
    try since ppo_br has no held-out concept at all (it trains a best
    response against every member of its source population), so Unseen
    members already have dedicated egos, same as Seen ones.
  - OMIS's own text ("the state space of LBF is much smaller than PP and OC,
    making it difficult for well-trained opponent policies to exhibit
    significant behavioral diversity") suggests LBF specifically may have a
    genuinely lower achievable ceiling than other environments under this
    exact mechanism -- worth treating 66.25% as the more directly comparable
    target for a from-scratch LBF replication, not 80.56%, given the
    near-identical methodology otherwise.

**This script.** Same procedure, against our own released MEP population and
its dedicated ppo_br egos -- reusing the exact rollout machinery dataset
collection uses (collect_episodes_batched), not a reimplementation. Defaults
to fewer episodes per ego than the paper's 1200/policy (cost scales as
n_teammates^2 * episodes_per_ego) -- raise --episodes-per-ego for a
publication-quality number once a quick pass looks interesting.

    uv run python scripts/diagnose_opponent_policy_diversity.py \\
        --population populations/lbf_9x9/mep \\
        --br-population results/teammate_generation/ppo_br_lbf_9x9-<hash> \\
        --dataset results/dataset_collection/pooled_lbf_9x9_tipr-<hash>

    # OMIS's full-population methodology instead of TIPR's Seen-only one:
    uv run python scripts/diagnose_opponent_policy_diversity.py \\
        --population populations/lbf_9x9/mep --include-unseen \\
        --br-population results/teammate_generation/ppo_br_lbf_9x9-<hash> \\
        --dataset results/dataset_collection/pooled_lbf_9x9_tipr-<hash>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def _seen_roster(population_dirs, dataset_path, env, max_teammates, *, include_unseen=False):
    """The teammate roster to measure -- :func:`load_teammate_policies`, by
    default filtered to members NOT recorded as held-out at collection time
    (TIPR's Seen Set), matching
    :func:`~oaht_bench.offline.evaluation._unseen_roster`'s approach.

    ``include_unseen=True`` instead returns the *whole* released population
    (Seen + Unseen together), matching OMIS's own Appendix G methodology
    (papers/omis.pdf pp. 30-32) rather than TIPR's: OMIS computes its
    dissimilarity rate over the full population (10 Seen + 10 Unseen = 20
    for its own MEP-population size), not a Seen-only subset, and reports
    66.25% for LBF under that methodology -- a materially different number
    from TIPR's 80.56% despite near-identical appendix text, suggesting the
    Seen-only restriction itself may explain part of the gap we've been
    chasing against TIPR's figure specifically. ``ppo_br`` has no
    held-out concept at all (ppo_br.py trains a best response against every
    member of its source population) -- the Unseen members already have
    dedicated egos the same way Seen ones do, so this needs no retraining,
    just reading the already-released roster without the held_out filter.
    """
    import types

    from oaht_bench.dataset.vault import VaultReader
    from oaht_bench.population.loading import load_teammate_policies

    roster = load_teammate_policies([Path(p) for p in population_dirs], env)
    meta = VaultReader(f"{dataset_path}/dataset.vlt").meta
    if include_unseen:
        selected = roster
    else:
        held_out = {int(m) for ms in meta.get("held_out", {}).values() for m in ms}
        selected = [m for m in roster if int(m.member) not in held_out]
    if len(selected) < 2:
        raise ValueError(
            f"only {len(selected)} teammates found (roster size {len(roster)}) -- "
            f"need at least 2 to compare."
        )
    if max_teammates is None:
        # Seen-only (TIPR) mode defaults to 12, its own Seen Set size; full-roster
        # (OMIS) mode defaults to uncapped, since 20/20 is the point of that mode.
        max_teammates = 12 if not include_unseen else len(selected)
    return types.SimpleNamespace(meta=meta), selected[:max_teammates]


def _collect_observation_sample(env, teammate, egos, *, episodes_per_ego, max_episode_steps, rng):
    """Teammate's own ``(obs, avail_actions)`` stream across ``episodes_per_ego``
    episodes against EVERY ego (not just its own dedicated one) -- "the
    self-agent traverses through all the opponent policies", Appendix D.
    """
    from oaht_bench.dataset.construction.collect import collect_episodes_batched

    obs_chunks, avail_chunks = [], []
    for ego in egos:
        rng, pair_rng = jax.random.split(rng)
        episodes = collect_episodes_batched(
            pair_rng,
            env,
            [(ego.params, ego.policy_cls), (teammate.params, teammate.policy_cls)],
            max_episode_steps=max_episode_steps,
            num_episodes=episodes_per_ego,
        )
        for ep in episodes:
            obs_chunks.append(ep.obs[1])
            avail_chunks.append(ep.avail_actions[1])
    return np.concatenate(obs_chunks, axis=0), np.concatenate(avail_chunks, axis=0), rng


def _policy_probs(teammate, obs, avail) -> np.ndarray:
    """``pi(a|o)`` for every row of ``obs``/``avail`` -- a pure forward pass, no
    rollout. ``avail`` is a property of the observed env state, shared across
    whichever policy is being evaluated at it."""
    n = obs.shape[0]
    _action, _val, pi, _hstate = teammate.policy_cls.get_action_value_policy(
        params=teammate.params,
        obs=jnp.asarray(obs),
        done=jnp.zeros(n, dtype=bool),
        avail_actions=jnp.asarray(avail),
        hstate=None,
        rng=jax.random.PRNGKey(0),
    )
    return np.asarray(pi.probs)


def _forward_kl(p_i: np.ndarray, p_j: np.ndarray, eps: float = 1e-8) -> float:
    """Eq. 33's Monte-Carlo estimate: mean over the sampled observations of
    sum_a p_i(a|o) * log(p_i(a|o) / p_j(a|o))."""
    ratio = np.log((p_i + eps) / (p_j + eps))
    return float(np.mean(np.sum(p_i * ratio, axis=-1)))


def diagnose(
    population_dirs: list[str],
    br_population: str,
    dataset_path: str,
    *,
    episodes_per_ego: int,
    max_teammates: int,
    threshold: float,
    mat_out: Path | None,
    include_unseen: bool = False,
) -> None:
    # env/job.json come from the dataset collection run, so this matches exactly
    # what that dataset was built against.
    from oaht_bench.configs import load_job
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.population.loading import load_ego_policies
    from oaht_bench.population.members import index_by_identity

    job = load_job(Path(dataset_path) / "job.json")
    env = LogWrapper(make_env(job.env.env_name, job.env.env_kwargs()))

    # NOT job.env.rollout_length -- that only sizes the PPO rollout-collection
    # scan and is explicitly documented (configs/env.py's own EnvConfigBase
    # docstring) as decoupled from the environment's real episode-termination
    # horizon. For LBF, LbfConfig.time_limit is the field that actually reaches
    # Jumanji's constructor; falling back to rollout_length only for env
    # families with no such field (this script has only been run against LBF
    # so far, so that fallback is unverified, not a confirmed-correct value).
    max_episode_steps = getattr(job.env, "time_limit", None) or job.env.rollout_length

    _meta, teammates = _seen_roster(
        population_dirs, dataset_path, env, max_teammates, include_unseen=include_unseen
    )
    egos = load_ego_policies(br_population, env)
    egos_by_id = index_by_identity(egos)
    identities = [(t.generator, int(t.member), t.role) for t in teammates]
    missing = [i for i in identities if i not in egos_by_id]
    if missing:
        raise ValueError(
            f"ppo_br run {br_population!r} has no dedicated best response for "
            f"{missing} -- must cover the whole roster 1:1 "
            f"(ppo_br has no held-out concept, so this means the population itself "
            f"changed since ppo_br was last trained, not an Unseen-member gap)."
        )
    ordered_egos = [egos_by_id[i] for i in identities]

    k = len(teammates)
    roster_label = (
        "Full roster (Seen+Unseen, OMIS Appendix G methodology)" if include_unseen else "Seen Set"
    )
    print(
        f"{roster_label}: {k} teammates, {episodes_per_ego} episodes/ego x {k} egos = "
        f"{episodes_per_ego * k} episodes sampled per teammate ({episodes_per_ego * k * k} total)\n"
    )

    rng = jax.random.PRNGKey(0)
    # probs[i][j] = pi_j evaluated on the observations collected while policy i
    # was the active opponent -- row i's own Monte-Carlo sample for Eq. 33.
    probs: list[list[np.ndarray]] = []
    for i, t in enumerate(teammates):
        obs, avail, rng = _collect_observation_sample(
            env,
            t,
            ordered_egos,
            episodes_per_ego=episodes_per_ego,
            max_episode_steps=max_episode_steps,
            rng=rng,
        )
        probs.append([_policy_probs(other, obs, avail) for other in teammates])
        print(f"  collected {obs.shape[0]:5d} observations under teammate {i} ({identities[i]})")

    matrix = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            matrix[i, j] = 0.0 if i == j else _forward_kl(probs[i][i], probs[i][j])

    offdiag = matrix[~np.eye(k, dtype=bool)]
    dissimilar = float(np.mean(offdiag > threshold))
    print(f"\n{k}x{k} pair-wise expected KL divergence matrix (Eq. 33):")
    print(f"  mean={offdiag.mean():.4f}  min={offdiag.min():.4f}  max={offdiag.max():.4f}")
    reference = (
        "OMIS's own LBF figure (Appendix G, full population): 66.25%"
        if include_unseen
        else "TIPR's own LBF figure (Appendix D, Seen Set only): 80.56%"
    )
    print(f"  dissimilarity rate (KL > {threshold}): {dissimilar * 100:.2f}%  [{reference}]")

    if mat_out:
        np.savez(
            mat_out,
            matrix=matrix,
            roster_generator=[g for g, _m, _r in identities],
            roster_member=[m for _g, m, _r in identities],
            roster_role=[r for _g, _m, r in identities],
        )
        print(f"\nwrote matrix -> {mat_out}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--population",
        action="append",
        required=True,
        dest="population_dirs",
        help="Released population dir(s); repeat for pooled mode.",
    )
    ap.add_argument("--br-population", required=True, help="ppo_br run dir (dedicated egos).")
    ap.add_argument(
        "--dataset",
        required=True,
        dest="dataset_path",
        help="Dataset collection run dir (for env config + held_out roster).",
    )
    ap.add_argument(
        "--episodes-per-ego",
        type=int,
        default=10,
        help="TIPR uses 100; cost is n_teammates^2 * this.",
    )
    ap.add_argument(
        "--max-teammates",
        type=int,
        default=None,
        help="Default: 12 (TIPR's Seen Set size) normally, uncapped with --include-unseen.",
    )
    ap.add_argument(
        "--include-unseen",
        action="store_true",
        help="Measure the full released population (Seen+Unseen), matching OMIS's "
        "Appendix G methodology instead of TIPR's Seen-only Appendix D (66.25%% vs "
        "80.56%% for LBF in the papers themselves). Needs no retraining -- ppo_br has "
        "no held-out concept, so Unseen members already have dedicated egos.",
    )
    ap.add_argument(
        "--threshold", type=float, default=1.0, help="Dissimilarity threshold (TIPR: 1.0)."
    )
    ap.add_argument(
        "--mat-out", type=Path, default=None, help="Write the full matrix to this .npz."
    )
    args = ap.parse_args()
    diagnose(
        args.population_dirs,
        args.br_population,
        args.dataset_path,
        episodes_per_ego=args.episodes_per_ego,
        max_teammates=args.max_teammates,
        threshold=args.threshold,
        mat_out=args.mat_out,
        include_unseen=args.include_unseen,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
