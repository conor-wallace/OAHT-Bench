"""ε-targeted seat sampler: turn a target ego-response-quality distribution into
concrete ``(ego, teammate)`` seatings, read off the pooled cross-play matrix.

This is the dataset-side consumer of
:mod:`~oaht_bench.population.pooled_crossplay` (``docs/dataset_design.md`` §3,
piece 3). The pooled matrix gives, for every teammate ``j``, an ε spectrum over
egos -- ``ε(i|j) ∈ [0,1]``, best-response = 1, worst = 0. A dataset *variant* is
a target distribution over that spectrum (``expert`` = top only; ``br_vs_worst``
= bimodal top/bottom; ``mixed`` = top and middle), and this module realises it as
an exact list of seatings.

It generalises :func:`oaht_bench.data.runner._seat_plan` along one axis. That
function splits episodes into matched/mismatched **by count** (not a per-episode
coin flip, so the fraction is an exact stated property) and cycles teammates so
each is covered equally. Here the same two ideas carry over: episodes are split
into ε bands by exact count, and teammates are cycled; what changes is that the
ego is no longer "same member / different member" but "the member whose ε to this
teammate is closest to the band's target level". Matched play falls out as the
``ε=1`` band for a homogeneous teammate; the mismatch axis (pairing correctness)
stays orthogonal and is layered by the caller, not baked in here.

The sampler works on the matrix + roster *manifest* (generator/member/role), not
live policies -- it emits roster indices and the ε value per episode. The caller
rebuilds the live roster (:func:`~oaht_bench.population.pooled_crossplay.build_roster`,
deterministic in the same population order) and seats index ``ego`` against index
``teammate``; :meth:`PooledMatrix.check_roster` guards the two against drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oaht_bench.population.pooled_crossplay import normalise_per_teammate

#: Target ε distributions per dataset variant (``docs/dataset_design.md`` §1).
#: Each entry is ``(target_epsilon, fraction)``; fractions sum to 1. ``expert``
#: is the top of the spectrum; ``br_vs_worst`` the two extremes; ``mixed`` the
#: D4RL medium-expert analogue (best-response + a mid-quality response).
EPSILON_TARGETS: dict[str, list[tuple[float, float]]] = {
    "expert": [(1.0, 1.0)],
    "br_vs_worst": [(1.0, 0.5), (0.0, 0.5)],
    "mixed": [(1.0, 0.5), (0.5, 0.5)],
}

#: Which roster roles may sit in the *teammate* seat by default. Confederates and
#: self-play policies are the designed teammates; a best-response (``br``) is a
#: designed *ego*, not a partner. The ego seat is unrestricted (any policy may be
#: a response), so this only constrains the columns the sampler draws teammates
#: from.
DEFAULT_TEAMMATE_ROLES: tuple[str, ...] = ("self", "conf")


@dataclass(frozen=True)
class Seating:
    """One episode's assignment: roster index ``ego`` (seat 0) vs ``teammate``.

    ``epsilon`` is the pooled-matrix ε(ego|teammate) actually realised by the
    choice -- the stable ``ego_response_quality`` descriptor the dataset records,
    which is the matrix value, not the episode's sampled return. ``target`` is the
    band level the seating was drawn to hit; ``epsilon`` may differ when no ego
    lands exactly on the target (the nearest is taken).
    """

    ego: int
    teammate: int
    epsilon: float
    target: float


@dataclass(frozen=True)
class PooledMatrix:
    """A loaded pooled cross-play matrix with its roster manifest.

    ``matrix[i, j]`` is ego ``i``'s mean coordination return with teammate ``j``;
    ``quality`` is the per-teammate [0,1] normalisation the sampler bands on.
    ``generator``/``member``/``role`` address any row or column back to its
    provenance and are index-aligned with a live
    :func:`~oaht_bench.population.pooled_crossplay.build_roster` roster.
    """

    matrix: np.ndarray
    generator: np.ndarray
    member: np.ndarray
    role: np.ndarray
    quality: np.ndarray
    meta: dict

    @property
    def size(self) -> int:
        return int(self.matrix.shape[0])

    def teammate_pool(self, roles: tuple[str, ...] = DEFAULT_TEAMMATE_ROLES) -> list[int]:
        """Roster indices eligible to sit in the teammate seat, by role."""
        return [j for j in range(self.size) if str(self.role[j]) in roles]

    def check_roster(self, roster) -> None:
        """Fail if a live roster has drifted from this matrix's manifest.

        The sampler emits indices into the manifest; the caller seats the live
        roster at those indices. If a population was re-released or reordered
        since the matrix was computed, the indices silently point at the wrong
        policy -- so verify the ``(generator, member, role)`` sequence matches
        element for element before trusting the plan.
        """
        if len(roster) != self.size:
            raise ValueError(
                f"roster has {len(roster)} policies but the pooled matrix has "
                f"{self.size}; recompute the matrix for this population set."
            )
        for i, e in enumerate(roster):
            got = (e.generator, int(e.member), e.role)
            want = (str(self.generator[i]), int(self.member[i]), str(self.role[i]))
            if got != want:
                raise ValueError(
                    f"roster entry {i} is {got} but the pooled matrix records "
                    f"{want}; the matrix is stale for this roster."
                )


def load_pooled(path: str | Path) -> PooledMatrix:
    """Read a ``pooled_crossplay.npz`` written by
    :func:`~oaht_bench.population.pooled_crossplay.save_pooled`."""
    z = np.load(Path(path), allow_pickle=False)
    matrix = z["matrix"]
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"pooled matrix at {path} is {matrix.shape}, not square; the roster "
            f"indexes both axes, so it must be K×K."
        )
    return PooledMatrix(
        matrix=matrix,
        generator=z["roster_generator"],
        member=z["roster_member"],
        role=z["roster_role"],
        quality=normalise_per_teammate(matrix),
        meta=json.loads(str(z["meta"])),
    )


def _band_counts(targets: list[tuple[float, float]], num_episodes: int) -> list[tuple[float, int]]:
    """Split ``num_episodes`` across the target bands by exact count.

    Like :func:`~oaht_bench.data.runner._seat_plan`'s matched/mismatched split, a
    variant's mixture is a stated property, not a draw: ``br_vs_worst`` over 10
    episodes is exactly 5 best-response and 5 worst-response. Rounding drift is
    absorbed by the first band so the counts always sum to ``num_episodes``.
    """
    counts = [(level, int(round(num_episodes * frac))) for level, frac in targets]
    drift = num_episodes - sum(c for _, c in counts)
    if counts:
        level0, c0 = counts[0]
        counts[0] = (level0, c0 + drift)
    return counts


def _cycle(items: list[int], count: int, rng: np.random.Generator) -> list[int]:
    """Take ``count`` items, cycling through ``items`` so each is used equally."""
    out: list[int] = []
    while len(out) < count:
        out.extend(items[i] for i in rng.permutation(len(items)))
    return out[:count]


def plan_seatings(
    pooled: PooledMatrix,
    targets: list[tuple[float, float]],
    num_episodes: int,
    *,
    rng: np.random.Generator,
    teammate_roles: tuple[str, ...] = DEFAULT_TEAMMATE_ROLES,
    allow_self_pairing: bool = True,
) -> list[Seating]:
    """Realise a target ε distribution as ``num_episodes`` concrete seatings.

    For each band ``(target_level, count)``, cycle teammates for equal coverage
    and, per teammate ``j``, choose the ego whose ``ε(i|j)`` is closest to
    ``target_level``. ``target_level=1`` therefore selects the best response
    (argmax ε), ``0`` the worst; intermediate levels the nearest-quality ego.

    ``allow_self_pairing`` controls whether an ego may be the *same roster entry*
    as the teammate (only possible for homogeneous ``self`` policies). For FCP and
    CoMeDi the best response to member ``j`` is usually ``j`` itself, so keeping it
    ``True`` makes ``expert`` the genuine top of the spectrum; setting it ``False``
    forces the top band to a cross-population responder instead. This is the
    self-pairing decision left open in ``docs/dataset_design.md`` §4, exposed as a
    knob rather than hard-coded.

    Returns the seatings interleaved, so slicing the dataset by index does not
    hand back a single band.
    """
    teammates = pooled.teammate_pool(teammate_roles)
    if not teammates:
        raise ValueError(f"no roster policy has a teammate role in {teammate_roles}.")

    plan: list[Seating] = []
    for level, count in _band_counts(targets, num_episodes):
        for j in _cycle(teammates, count, rng):
            col = pooled.quality[:, j]
            # Egos eligible to respond to this teammate. Only the exact same
            # roster entry is ever excluded as "self"; a paired generator's
            # conf_j and br_j are distinct entries, so br_j stays available.
            egos = np.arange(pooled.size)
            if not allow_self_pairing:
                egos = egos[egos != j]
            ego = int(egos[np.argmin(np.abs(col[egos] - level))])
            plan.append(Seating(ego=ego, teammate=j, epsilon=float(col[ego]), target=float(level)))

    return [plan[i] for i in rng.permutation(len(plan))]


def plan_for_variant(
    pooled: PooledMatrix,
    variant: str,
    num_episodes: int,
    *,
    rng: np.random.Generator,
    **kwargs,
) -> list[Seating]:
    """:func:`plan_seatings` keyed by a named variant in :data:`EPSILON_TARGETS`."""
    if variant not in EPSILON_TARGETS:
        raise NotImplementedError(
            f"variant={variant!r} has no ε target; known ε variants are "
            f"{sorted(EPSILON_TARGETS)}. (τ-ladder variants are deferred, §4.)"
        )
    return plan_seatings(pooled, EPSILON_TARGETS[variant], num_episodes, rng=rng, **kwargs)
