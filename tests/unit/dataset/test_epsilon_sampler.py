"""The ε-targeted seat sampler -- the deterministic parts of dataset variant
construction (``docs/dataset_design.md`` §3). Rollouts are integration-tested
elsewhere; this pins how a target ε distribution becomes concrete seatings.
"""

import numpy as np
import pytest

from oaht_bench.dataset.construction.epsilon_sampler import (
    EPSILON_TARGETS,
    PooledMatrix,
    Seating,
    _band_counts,
    load_pooled,
    plan_for_variant,
    plan_seatings,
)
from oaht_bench.population.pooled_crossplay import normalise_per_teammate


def _pooled(matrix, generator, member, role) -> PooledMatrix:
    matrix = np.asarray(matrix, dtype=float)
    return PooledMatrix(
        matrix=matrix,
        generator=np.asarray(generator),
        member=np.asarray(member, dtype=np.int32),
        role=np.asarray(role),
        quality=normalise_per_teammate(matrix),
        meta={},
    )


def _uniform_roles(k, role="self"):
    return ["gen"] * k, list(range(k)), [role] * k


def test_band_counts_are_exact_not_sampled():
    # br_vs_worst over 10 episodes is exactly 5/5 -- a stated property.
    assert _band_counts(EPSILON_TARGETS["br_vs_worst"], 10) == [(1.0, 5), (0.0, 5)]
    # Rounding drift is absorbed by the first band; counts always sum to N.
    counts = _band_counts(EPSILON_TARGETS["mixed"], 7)
    assert sum(c for _, c in counts) == 7


def test_expert_band_selects_the_best_response_per_teammate():
    # Column j's argmax ego is the best response; expert must pick it every time.
    matrix = np.array(
        [
            [0.5, 0.1, 0.2],
            [0.2, 0.9, 0.1],
            [0.1, 0.3, 0.8],
        ]
    )
    g, m, r = _uniform_roles(3)
    pooled = _pooled(matrix, g, m, r)
    plan = plan_seatings(pooled, EPSILON_TARGETS["expert"], 9, rng=np.random.default_rng(0))

    assert len(plan) == 9
    best_ego = {j: int(np.argmax(matrix[:, j])) for j in range(3)}
    for s in plan:
        assert s.ego == best_ego[s.teammate]
        assert s.epsilon == pytest.approx(1.0)  # top of the normalised spectrum


def test_worst_band_selects_the_argmin_ego():
    matrix = np.array([[0.5, 0.1, 0.2], [0.2, 0.9, 0.7], [0.1, 0.3, 0.9]])
    g, m, r = _uniform_roles(3)
    pooled = _pooled(matrix, g, m, r)
    plan = plan_seatings(pooled, [(0.0, 1.0)], 6, rng=np.random.default_rng(1))
    worst_ego = {j: int(np.argmin(matrix[:, j])) for j in range(3)}
    for s in plan:
        assert s.ego == worst_ego[s.teammate]
        assert s.epsilon == pytest.approx(0.0)


def test_br_vs_worst_is_bimodal_and_covers_both_extremes():
    matrix = np.array([[0.9, 0.1, 0.5], [0.5, 0.5, 0.1], [0.1, 0.9, 0.9]])
    g, m, r = _uniform_roles(3)
    pooled = _pooled(matrix, g, m, r)
    plan = plan_seatings(pooled, EPSILON_TARGETS["br_vs_worst"], 8, rng=np.random.default_rng(2))
    eps = sorted(s.epsilon for s in plan)
    # Exactly half at the top of the spectrum, half at the bottom, nothing mid.
    assert eps.count(pytest.approx(1.0)) == 4
    assert eps.count(pytest.approx(0.0)) == 4


def test_teammates_are_covered_equally():
    # Four teammates, eight episodes at one band -> each teammate seated twice.
    matrix = np.eye(4) + 0.1
    g, m, r = _uniform_roles(4)
    pooled = _pooled(matrix, g, m, r)
    plan = plan_seatings(pooled, [(1.0, 1.0)], 8, rng=np.random.default_rng(3))
    seated = np.bincount([s.teammate for s in plan], minlength=4)
    assert list(seated) == [2, 2, 2, 2]


def test_teammate_role_filter_excludes_br_columns():
    # Only self/conf may be teammates; a br column must never be seated as partner.
    matrix = np.arange(9.0).reshape(3, 3)
    pooled = _pooled(matrix, ["g", "g", "g"], [0, 0, 1], ["conf", "br", "self"])
    plan = plan_seatings(pooled, [(1.0, 1.0)], 6, rng=np.random.default_rng(4))
    seated_roles = {str(pooled.role[s.teammate]) for s in plan}
    assert seated_roles <= {"self", "conf"}
    assert 1 not in {s.teammate for s in plan}  # index 1 is the br column


def test_allow_self_pairing_toggles_expert_ego_for_homogeneous():
    # A teammate whose own self-play is its best response. With self-pairing off,
    # expert must fall back to the best *other* ego.
    matrix = np.array(
        [
            [1.0, 0.2],  # ego 0 best with teammate 0 (itself)
            [0.7, 0.9],
        ]
    )
    g, m, r = _uniform_roles(2)
    pooled = _pooled(matrix, g, m, r)

    with_self = plan_seatings(
        pooled, [(1.0, 1.0)], 2, rng=np.random.default_rng(5), allow_self_pairing=True
    )
    ego_for_0 = next(s.ego for s in with_self if s.teammate == 0)
    assert ego_for_0 == 0  # self is the best response

    without_self = plan_seatings(
        pooled, [(1.0, 1.0)], 2, rng=np.random.default_rng(5), allow_self_pairing=False
    )
    ego_for_0 = next(s.ego for s in without_self if s.teammate == 0)
    assert ego_for_0 == 1  # forced onto the best cross-pairing


def test_plan_for_variant_rejects_unknown_variant():
    pooled = _pooled(np.eye(2), *(_uniform_roles(2)))
    with pytest.raises(NotImplementedError, match="no ε target"):
        plan_for_variant(pooled, "medium", 4, rng=np.random.default_rng(0))


def test_check_roster_catches_drift():
    pooled = _pooled(np.eye(2), ["g", "g"], [0, 1], ["self", "self"])

    class E:
        def __init__(self, generator, member, role):
            self.generator, self.member, self.role = generator, member, role

    pooled.check_roster([E("g", 0, "self"), E("g", 1, "self")])  # matches -> ok
    with pytest.raises(ValueError, match="stale"):
        pooled.check_roster([E("g", 0, "self"), E("g", 9, "self")])
    with pytest.raises(ValueError, match="recompute"):
        pooled.check_roster([E("g", 0, "self")])  # wrong length


def test_load_pooled_round_trip(tmp_path):
    from oaht_bench.population.pooled_crossplay import RosterEntry, save_pooled

    matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
    roster = [
        RosterEntry("fcp", 0, "self", params=None, policy_cls=None),
        RosterEntry("brdiv", 3, "conf", params=None, policy_cls=None),
    ]
    path = save_pooled(matrix, roster, tmp_path / "pooled_crossplay.npz", meta={"env": "lbf"})
    pooled = load_pooled(path)

    np.testing.assert_allclose(pooled.matrix, matrix)
    assert list(pooled.generator) == ["fcp", "brdiv"]
    assert list(pooled.member) == [0, 3]
    assert list(pooled.role) == ["self", "conf"]
    assert pooled.meta["env"] == "lbf"
    # And the manifest addresses the same policies the live roster would.
    pooled.check_roster(roster)


def test_seatings_record_matrix_quality_not_return():
    # ε on a Seating is the normalised matrix value, the stable descriptor.
    matrix = np.array([[0.4, 0.1], [0.2, 0.6]])
    g, m, r = _uniform_roles(2)
    pooled = _pooled(matrix, g, m, r)
    plan = plan_seatings(pooled, [(0.5, 1.0)], 2, rng=np.random.default_rng(7))
    for s in plan:
        assert isinstance(s, Seating)
        assert s.epsilon == pytest.approx(pooled.quality[s.ego, s.teammate])
