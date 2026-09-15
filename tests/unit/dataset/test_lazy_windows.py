"""LazyWindows must be a drop-in for the eagerly-materialized Windows.

The lazy path exists to fit long-context datasets that O(dataset x context) up-front
materialization cannot, so it is only worth having if it produces the *same*
windows. These pin field-for-field equivalence (raw and normalized) and the
normalization equality that holds when every transition belongs to one window.
"""

from __future__ import annotations

import numpy as np
import pytest

from oaht_bench.dataset.dataset import LazyWindows, _build_windows
from oaht_bench.dataset.schema import Episode, EpisodeBatch

_PER_POS_FLOAT = ("ego_obs", "ego_rtg", "mate_obs", "mate_next_obs")
_PER_POS_INT = (
    "ego_actions",
    "ego_avail",
    "mate_actions",
    "mate_avail",
    "mate_rewards",
    "timesteps",
    "mask",
)
_META = ("episode_id", "teammate_id", "episode_return")


def _batch(lengths, obs_dim=5, n_actions=6, seed=0):
    rng = np.random.default_rng(seed)
    episodes, member_ids = [], []
    for i, ln in enumerate(lengths):
        episodes.append(
            Episode(
                obs=rng.normal(size=(2, ln, obs_dim)).astype(np.float32),
                actions=rng.integers(0, n_actions, size=(2, ln)),
                rewards=rng.normal(size=(2, ln)).astype(np.float32),
                avail_actions=(rng.random((2, ln, n_actions)) > 0.3).astype(np.float32),
                dones=np.zeros(ln, dtype=bool),
            )
        )
        member_ids.append([0, i % 3])
    return EpisodeBatch(episodes=episodes, member_ids=np.asarray(member_ids), ego_index=0, meta={})


def _assert_fields_match(eager, lazy, idx, *, normalized):
    for k in _META:
        assert np.array_equal(getattr(eager, k), getattr(lazy, k)), k
    for k in _PER_POS_INT:
        assert np.array_equal(getattr(eager, k)[idx], getattr(lazy, k)[idx]), k
    for k in _PER_POS_FLOAT:
        atol = 1e-5 if normalized else 0.0
        assert np.allclose(getattr(eager, k)[idx], getattr(lazy, k)[idx], atol=atol), k


def test_raw_fields_identical_multiwindow():
    """With normalize off, every field matches exactly -- even when episodes are
    longer than the context, so stride cuts several overlapping windows each."""
    batch = _batch([16, 16, 16, 16])
    kw = dict(context_length=8, stride=4, normalize=False)
    eager, lazy = _build_windows(batch, **kw), LazyWindows(batch, **kw)
    assert len(eager) == len(lazy) and len(eager) > len(batch.episodes)  # overlapping windows
    _assert_fields_match(eager, lazy, np.arange(len(eager)), normalized=False)


def test_normalized_fields_and_norm_identical_when_one_window_per_episode():
    """Episodes no longer than the context => one window each => the streaming
    normalization equals the per-window-position statistics exactly."""
    batch = _batch([6, 7, 5, 8, 6, 7])  # all <= context 8
    kw = dict(context_length=8, stride=4, normalize=True)
    eager, lazy = _build_windows(batch, **kw), LazyWindows(batch, **kw)
    assert len(eager) == len(lazy) == len(batch.episodes)  # one window per episode
    assert np.allclose(eager.norm.obs_mean, lazy.norm.obs_mean, atol=1e-5)
    assert np.allclose(eager.norm.obs_std, lazy.norm.obs_std, atol=1e-5)
    assert eager.norm.rtg_scale == pytest.approx(lazy.norm.rtg_scale, rel=1e-5)
    _assert_fields_match(eager, lazy, np.array([0, 3, 5, 1]), normalized=True)


def test_two_dimensional_index_matches():
    """Stage 2 indexes with a (batch, C) context array; the lazy proxy must
    reshape the same way."""
    batch = _batch([6, 6, 6, 6, 6, 6])
    kw = dict(context_length=8, stride=4, normalize=True)
    eager, lazy = _build_windows(batch, **kw), LazyWindows(batch, **kw)
    cidx = np.array([[0, 1, 2], [3, 4, 5]])
    for k in ("mate_next_obs", "mate_actions", "mask"):
        a, b = getattr(eager, k)[cidx], getattr(lazy, k)[cidx]
        assert a.shape == b.shape and np.allclose(a, b, atol=1e-5), k
