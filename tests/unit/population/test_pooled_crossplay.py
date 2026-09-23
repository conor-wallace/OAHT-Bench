"""The pooled-matrix normalization -- the ego-response quality spectrum.

The matrix itself comes from real rollouts (integration, not unit-tested here);
this pins the deterministic per-teammate normalization the dataset sampler reads,
plus the two invariants the BR-only ego redefinition depends on: the teammate
roster excludes `br`-role rows, and the matrix computation refuses to run against
a mismatched (not-exactly-1:1) `br_egos`.
"""

import numpy as np
import pytest

from oaht_bench.population.pooled_crossplay import (
    RosterEntry,
    evaluate_pooled,
    normalise_per_teammate,
    teammate_roster,
)


def test_normalise_is_per_teammate_columnwise():
    # Each column is a teammate; egos are rows. Worst ego -> 0, best -> 1, per column.
    matrix = np.array(
        [
            [0.0, 5.0],
            [2.0, 5.0],
            [4.0, 15.0],
        ]
    )
    quality = normalise_per_teammate(matrix)
    # col 0: 0,2,4 -> min 0, max 4
    np.testing.assert_allclose(quality[:, 0], [0.0, 0.5, 1.0])
    # col 1: 5,5,15 -> min 5, max 15
    np.testing.assert_allclose(quality[:, 1], [0.0, 0.0, 1.0])


def test_best_and_worst_are_the_column_argmax_argmin():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(6, 6))
    quality = normalise_per_teammate(matrix)
    for j in range(matrix.shape[1]):
        assert np.argmax(quality[:, j]) == np.argmax(matrix[:, j])
        assert np.argmin(quality[:, j]) == np.argmin(matrix[:, j])
        assert quality[np.argmax(matrix[:, j]), j] == 1.0
        assert quality[np.argmin(matrix[:, j]), j] == 0.0


def test_flat_column_maps_to_one_half_not_nan():
    # A teammate every ego coordinates with equally has no spread; must not divide
    # by zero (which would read as all-best or NaN).
    matrix = np.array([[3.0], [3.0], [3.0]])
    quality = normalise_per_teammate(matrix)
    np.testing.assert_allclose(quality[:, 0], [0.5, 0.5, 0.5])
    assert np.isfinite(quality).all()


def _fake_roster():
    return [
        RosterEntry("brdiv", 0, "conf", "p_b0c", "clsB"),
        RosterEntry("brdiv", 0, "br", "p_b0b", "clsB"),  # not a teammate, not an ego
        RosterEntry("comedi", 1, "self", "p_c1", "clsC"),
        RosterEntry("fcp", 4, "self", "p_f4", "clsF"),
    ]


def test_teammate_roster_excludes_br_role_rows(monkeypatch):
    import oaht_bench.population.pooled_crossplay as pc

    roster = _fake_roster()
    monkeypatch.setattr(pc, "build_roster", lambda dirs, env, **kw: roster)

    teammates = teammate_roster(["pops/brdiv", "pops/comedi", "pops/fcp"], env=None)

    assert [(t.generator, t.member, t.role) for t in teammates] == [
        ("brdiv", 0, "conf"),
        ("comedi", 1, "self"),
        ("fcp", 4, "self"),
    ]


def test_evaluate_pooled_requires_br_egos_to_cover_the_roster_exactly_1to1():
    teammates = [
        RosterEntry("brdiv", 0, "conf", "p", "cls"),
        RosterEntry("comedi", 1, "self", "p", "cls"),
    ]
    # Missing comedi:1, plus an extra entry for a teammate not in the roster.
    br_egos = {
        ("brdiv", 0, "conf"): ("br_params", "cls"),
        ("fcp", 9, "self"): ("br_params", "cls"),
    }
    with pytest.raises(ValueError, match="1:1"):
        evaluate_pooled(
            env=None,
            teammates=teammates,
            br_egos=br_egos,
            rng=None,
            max_episode_steps=1,
        )
