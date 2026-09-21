"""The pooled-matrix normalization -- the ego-response quality spectrum.

The matrix itself comes from real rollouts (integration, not unit-tested here);
this pins the deterministic per-teammate normalization the dataset sampler reads.
"""

import numpy as np

from oaht_bench.population.pooled_crossplay import normalise_per_teammate


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
