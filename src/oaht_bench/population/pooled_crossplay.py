"""Pooled cross-population coordination-return matrix (§4, dataset_design.md).

The within-population :mod:`~oaht_bench.population.crossplay` matrix scores a
single generator's members against each other. The dataset's ego-response quality
axis needs more, and needs it *competent*: for a teammate ``j``, the ego that
coordinates with it best is a policy dedicated to responding to it, not a reused
population policy -- the teammate-id oracle showed reused-policy egos cap the
dataset at ~45% of achievable competence (``docs/tuning_record.md``). So the ego
axis of this matrix is always the trained ``ppo_br`` population: one dedicated
best-response per designed teammate, never the teammate's own generator's
policies.

**Teammates and egos share one identity space.**
:func:`~oaht_bench.population.loading.load_teammate_policies` builds
the *designed teammates* -- one policy per released member of every generator,
``self`` for homogeneous generators (FCP, CoMeDi), ``conf`` for paired ones
(BRDiv, L-BRDiv); a paired generator's own ``br`` role (its own designed best
response) is never a teammate and, since ``ppo_br`` replaces it, is never an ego
either -- it plays no further part in this matrix. ``ppo_br`` trains exactly one
dedicated best response per teammate identity (:mod:`oaht_bench.teammate_gen.ppo_br`),
so the matrix is a genuine square ``K x K``: ``matrix[i, j]`` is the mean episode
return with teammate ``i``'s dedicated best response in the ego seat (seat 0) and
teammate ``j`` in the teammate seat (seat 1), which for these cooperative
environments is the shared coordination return -- every cell measured, not just
the diagonal, so a teammate's episodes can genuinely be spread across several
different (mostly-competent) egos rather than pinned to one. The dataset sampler
reads this to place each episode at a target point on the best-worst response
spectrum, and stores the (per-teammate normalised) value as
``ego_response_quality``. The roster manifest travels with the matrix so a
column/row can always be traced back to ``(generator, member, role)``.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
from tqdm import tqdm

from oaht_bench.common.run_episodes import run_episodes
from oaht_bench.population.members import Member, index_by_identity


def evaluate_pooled(
    env,
    teammates: list[Member],
    egos: list[Member],
    *,
    rng: jax.Array,
    max_episode_steps: int,
    num_episodes: int = 20,
    greedy: bool = False,
) -> np.ndarray:
    """Score every ordered ``(dedicated best response, teammate)`` pair.

    ``teammates`` is :func:`~oaht_bench.population.loading.load_teammate_policies`'s
    ``K``-sized list; ``egos``
    is :func:`~oaht_bench.population.loading.load_ego_policies`'s list, one
    entry per teammate, each carrying that teammate's own ``(generator,
    member, role)`` identity with its own dedicated best-response ``params``/
    ``policy_cls``. Returns ``matrix`` of shape ``(K, K)`` with ``matrix[i, j]``
    the mean return of teammate ``i``'s dedicated best response (seat 0) with
    teammate ``j`` (seat 1) -- so column ``j``'s diagonal cell is teammate
    ``j`` against its *own* dedicated best response, and every other cell in
    that column is a different teammate's best response cross-played against
    ``j``, which is what gives the dataset sampler real cross-play egos to
    weight over rather than a single fixed point per teammate.

    Raises if ``egos`` doesn't cover exactly the teammate identities in
    ``teammates`` -- ``ppo_br`` is meant to cover the whole released roster
    1:1, so a mismatch means a stale or partial ``ppo_br`` run, not a case to
    silently work around. ``egos`` is not assumed to be in the same order as
    ``teammates`` -- :func:`~oaht_bench.population.members.index_by_identity`
    does the identity-keyed lookup rather than relying on position.

    Cost is ``K**2 * num_episodes`` episodes; ``greedy`` stays off for the
    same reason :mod:`~oaht_bench.population.crossplay` keeps it off (argmax
    deadlocks symmetric coordination). Each pair is a separate
    ``run_episodes`` call -- correct but not fast at large ``K``
    (heterogeneous policies recompile); an all-pairs ``vmap`` is the
    optimisation if it becomes a bottleneck.
    """
    identities = [(t.generator, int(t.member), t.role) for t in teammates]
    egos_by_id = index_by_identity(egos)
    missing = set(identities) - set(egos_by_id)
    extra = set(egos_by_id) - set(identities)
    if missing or extra:
        raise ValueError(
            "egos must cover exactly the teammate roster 1:1 -- "
            f"missing dedicated best responses for {sorted(missing)}, "
            f"and egos has entries with no matching teammate: {sorted(extra)}. "
            "Retrain ppo_br against the current teammate roster."
        )

    k = len(teammates)
    matrix = np.zeros((k, k), dtype=float)
    with tqdm(total=k * k, desc="crossplay pairs", unit="pair") as bar:
        for i, ego_identity in enumerate(identities):
            ego = egos_by_id[ego_identity]
            for j, mate in enumerate(teammates):
                rng, pair_rng = jax.random.split(rng)
                out = run_episodes(
                    pair_rng,
                    env,
                    agent_0_param=ego.params,
                    agent_0_policy=ego.policy_cls,
                    agent_1_param=mate.params,
                    agent_1_policy=mate.policy_cls,
                    max_episode_steps=max_episode_steps,
                    num_eps=num_episodes,
                    agent_0_test_mode=greedy,
                    agent_1_test_mode=greedy,
                )
                matrix[i, j] = float(np.asarray(out["returned_episode_returns"]).mean())
                bar.update(1)
    return matrix


def normalise_per_teammate(matrix: np.ndarray) -> np.ndarray:
    """Column-normalise to ``[0, 1]`` -- the ego-response quality spectrum.

    For each teammate ``j`` (column), map the worst-coordinating ego to 0 and the
    best to 1, so ``quality[i, j]`` is how good ego ``i``'s response is *relative
    to this teammate*. A column with no spread (all egos equal) maps to 0.5 rather
    than dividing by zero.
    """
    lo = matrix.min(axis=0, keepdims=True)
    hi = matrix.max(axis=0, keepdims=True)
    span = hi - lo
    out = np.where(span > 0, (matrix - lo) / np.where(span > 0, span, 1.0), 0.5)
    return out


def save_pooled(matrix: np.ndarray, roster: list[Member], path: Path, *, meta: dict) -> Path:
    """Write the matrix, the roster manifest, and provenance as one ``.npz``.

    The roster arrays are what make a column/row addressable as
    ``(generator, member, role)`` without re-deriving it, so a dataset built off
    this matrix can record exact provenance per episode.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        matrix=matrix,
        roster_generator=np.asarray([e.generator for e in roster]),
        roster_member=np.asarray([e.member for e in roster], dtype=np.int32),
        roster_role=np.asarray([e.role for e in roster]),
        meta=np.asarray(json.dumps(meta, sort_keys=True, default=str)),
    )
    return path


def run(job) -> Path:
    """Execute a :class:`~oaht_bench.configs.job.PooledCrossplayJob` (§4, step 2).

    Builds the designed-teammate roster from ``job.population_path`` (in list
    order, so it matches what a pooled ``dataset_collection`` reconstructs),
    loads the dedicated best response for every teammate from
    ``job.br_population_path``, scores every ordered ``(best response, teammate)``
    pair, and writes ``pooled_crossplay.npz`` plus a readable ``.csv`` and
    ``roster.json`` to ``job.output_path`` (or ``<run_dir>/pooled_crossplay.npz``
    when unset). The resolved config is always recorded in the run directory.
    Returns the directory the matrix was written to.
    """
    import json

    from oaht_bench.configs import save_job
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.population.loading import load_ego_policies, load_teammate_policies

    run_dir = Path(job.run_dir())
    out = Path(job.output_path) if job.output_path else run_dir / "pooled_crossplay.npz"
    if out.exists():
        raise FileExistsError(
            f"{out} already exists and would be overwritten. Delete it to recompute, "
            f"or set a different output_path/label. (Downstream configs reference this "
            f"path, so overwriting silently would change what they read.)"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_job(job, run_dir / "job.json", minimal=False)

    env = LogWrapper(make_env(job.env.env_name, job.env.env_kwargs()))
    teammates = load_teammate_policies([Path(p) for p in job.population_path], env)
    egos = load_ego_policies(job.br_population_path, env)

    matrix = evaluate_pooled(
        env,
        teammates,
        egos,
        rng=jax.random.PRNGKey(job.seed),
        max_episode_steps=job.env.rollout_length,
        num_episodes=job.num_episodes,
    )
    save_pooled(
        matrix,
        teammates,
        out,
        meta={
            "env": job.env.name,
            "populations": [str(p) for p in job.population_path],
            "br_population": str(job.br_population_path),
            "num_episodes": job.num_episodes,
            "seed": job.seed,
        },
    )
    # A readable copy of the matrix and roster alongside the npz.
    np.savetxt(out.with_suffix(".csv"), matrix, delimiter=",")
    out.with_name("roster.json").write_text(
        json.dumps(
            [
                {"index": i, "generator": e.generator, "member": e.member, "role": e.role}
                for i, e in enumerate(teammates)
            ],
            indent=2,
        )
        + "\n"
    )
    return out.parent
