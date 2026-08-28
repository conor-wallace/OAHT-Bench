"""Pooled cross-population coordination-return matrix (§4, dataset_design.md).

The within-population :mod:`~oaht_bench.population.crossplay` matrix scores a
single generator's members against each other. The dataset's ego-response quality
axis needs more: for a teammate ``j``, the *best response* is the ego that
coordinates with it best -- and that ego may come from a *different* generator.
So this module pools the released members of every generator for one environment
into a single roster of policies, and scores every ordered ``(ego, teammate)``
pair across the whole pool.

**The roster is individual policies, not members.** A homogeneous generator (FCP,
CoMeDi) contributes one self-play policy per released member. A paired generator
(BRDiv, L-BRDiv) contributes *two*: the confederate (the designed teammate) and
its best response (the designed ego). Flattening to policies is what lets the
best response to a CoMeDi confederate be, say, an FCP checkpoint -- the
cross-population mixing the dataset design (decision b) is built on.

``matrix[i, j]`` is the mean episode return with roster policy ``i`` in the ego
seat (seat 0) and policy ``j`` in the teammate seat (seat 1), which for these
cooperative environments is the shared coordination return. The dataset sampler
reads this to place each episode at a target point on the best-worst response
spectrum, and stores the (per-teammate normalised) value as
``ego_response_quality``. The roster manifest travels with the matrix so a
column/row can always be traced back to ``(generator, member, role)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

from oaht_bench.common.run_episodes import run_episodes
from oaht_bench.population.loading import artifact_dir
from oaht_bench.population.members import get_member_params, released_members


@dataclass(frozen=True)
class RosterEntry:
    """One policy in the pooled roster, tagged with its provenance and seat role.

    ``role`` is ``self`` for a homogeneous generator's self-play policy, or
    ``conf``/``br`` for a paired generator's confederate / best response. It is
    kept so the dataset sampler can, for example, restrict *teammates* to
    ``{self, conf}`` while allowing any policy as the ego.
    """

    generator: str
    member: int
    role: str
    params: Any
    policy_cls: Any


def _load_population(pop_dir: Path, env):
    """Rebuild a released population the same way scoring and collection do."""
    from oaht_bench.common.save_load_utils import load_train_run
    from oaht_bench.configs import load_job
    from oaht_bench.population.loading import population_from_run

    pop_dir = Path(pop_dir)
    run_dir = pop_dir.parent.parent if pop_dir.name == "saved_train_run" else pop_dir
    job = load_job(run_dir / "job.json")
    out = load_train_run(str(artifact_dir(run_dir)))
    return population_from_run(job, out, env), job


def build_roster(population_dirs: list[Path], env, *, seed_index: int = 0) -> list[RosterEntry]:
    """Flatten released populations into one roster of individual policies.

    ``population_dirs`` are released run directories (``populations/<env>/<gen>/``),
    one per generator. Each contributes its *released* members -- the converged
    checkpoints for FCP, one per convention for the others (see
    :func:`released_members`) -- as self / conf / br policies.
    """
    roster: list[RosterEntry] = []
    for pop_dir in population_dirs:
        loaded, job = _load_population(pop_dir, env)
        generator = job.generator.generator
        for m in released_members(job, loaded.pop_size):
            m = int(m)
            if loaded.paired:
                roster.append(
                    RosterEntry(
                        generator, m, "conf",
                        get_member_params(loaded.params, m, seed_index=seed_index),
                        loaded.policy_cls,
                    )
                )
                roster.append(
                    RosterEntry(
                        generator, m, "br",
                        get_member_params(loaded.partner_params, m, seed_index=seed_index),
                        loaded.policy_cls,
                    )
                )
            else:
                roster.append(
                    RosterEntry(
                        generator, m, "self",
                        get_member_params(loaded.params, m, seed_index=seed_index),
                        loaded.policy_cls,
                    )
                )
    return roster


def evaluate_pooled(
    env,
    roster: list[RosterEntry],
    *,
    rng: jax.Array,
    max_episode_steps: int,
    num_episodes: int = 20,
    greedy: bool = False,
) -> np.ndarray:
    """Score every ordered ``(ego, teammate)`` pair in the roster.

    Returns ``matrix`` of shape ``(K, K)`` with ``matrix[i, j]`` the mean return
    of ego ``roster[i]`` (seat 0) with teammate ``roster[j]`` (seat 1). Cost is
    ``K**2 * num_episodes`` episodes; ``greedy`` stays off for the same reason
    :mod:`~oaht_bench.population.crossplay` keeps it off (argmax deadlocks
    symmetric coordination). Each pair is a separate ``run_episodes`` call --
    correct but not fast at large ``K`` (heterogeneous policies recompile); an
    all-pairs ``vmap`` is the optimisation if it becomes a bottleneck.
    """
    k = len(roster)
    matrix = np.zeros((k, k), dtype=float)
    for i, ego in enumerate(roster):
        for j, mate in enumerate(roster):
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


def save_pooled(matrix: np.ndarray, roster: list[RosterEntry], path: Path, *, meta: dict) -> Path:
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
