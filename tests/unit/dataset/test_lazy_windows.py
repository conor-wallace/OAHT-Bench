"""The streaming :class:`Windows` view must be a drop-in for the eager
:class:`_ReferenceWindows` oracle.

The view exists to fit long-context datasets that O(dataset x context) up-front
materialization cannot, so it is only worth having if it produces the *same*
windows. These pin field-for-field equivalence (raw and normalized) and the
normalization equality that holds when every transition belongs to one window.
"""

from __future__ import annotations

import numpy as np
import pytest

from oaht_bench.dataset.dataset import Windows, _build_windows
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
    eager, lazy = _build_windows(batch, **kw), Windows(batch, **kw)
    assert len(eager) == len(lazy) and len(eager) > len(batch.episodes)  # overlapping windows
    _assert_fields_match(eager, lazy, np.arange(len(eager)), normalized=False)


def test_normalized_fields_and_norm_identical_when_one_window_per_episode():
    """Episodes no longer than the context => one window each => the streaming
    normalization equals the per-window-position statistics exactly."""
    batch = _batch([6, 7, 5, 8, 6, 7])  # all <= context 8
    kw = dict(context_length=8, stride=4, normalize=True)
    eager, lazy = _build_windows(batch, **kw), Windows(batch, **kw)
    assert len(eager) == len(lazy) == len(batch.episodes)  # one window per episode
    assert np.allclose(eager.norm.obs_mean, lazy.norm.obs_mean, atol=1e-5)
    assert np.allclose(eager.norm.obs_std, lazy.norm.obs_std, atol=1e-5)
    assert eager.norm.rtg_scale == pytest.approx(lazy.norm.rtg_scale, rel=1e-5)
    _assert_fields_match(eager, lazy, np.array([0, 3, 5, 1]), normalized=True)


def test_dataset_streams_from_vault_matches_eager_reference(tmp_path):
    """The production path -- Dataset, which always streams episodes from the vault
    via VaultReader and windows them with the view -- must produce the same windows
    as the eager _build_windows reference over the same episodes in RAM."""
    from oaht_bench.dataset.dataset import Dataset
    from oaht_bench.dataset.vault import read_vault, write_vault

    batch = _batch([6, 7, 5, 8, 6, 7, 6, 5])  # <= context 8: one window per episode
    vault = tmp_path / "v.vlt"
    write_vault(batch.episodes, batch.member_ids, vault, ego_index=0, meta={"variant": "single"})
    kw = dict(context_length=8, stride=4, normalize=True)
    streamed = Dataset(str(vault), **kw).windows
    # Reference off the same vault (via read_vault -> _build_windows), so the only
    # thing under test is the windowing, not the vault's float32 store round-trip.
    reference = _build_windows(read_vault(str(vault)), **kw)

    assert len(streamed) == len(reference)
    assert np.allclose(streamed.norm.obs_mean, reference.norm.obs_mean, atol=1e-5)
    assert streamed.norm.rtg_scale == pytest.approx(reference.norm.rtg_scale, rel=1e-5)
    _assert_fields_match(reference, streamed, np.array([0, 3, 5, 1, 7]), normalized=True)


def test_vault_writer_chunked_matches_one_shot(tmp_path):
    """VaultWriter appends episodes in chunks (bounded RAM at collection time); the
    resulting vault must read back identically to a single write_vault call."""
    from oaht_bench.dataset.vault import VaultWriter, read_vault, write_vault

    batch = _batch([6, 7, 5, 8, 6, 7, 4, 9])
    eps, mids = batch.episodes, batch.member_ids

    one = tmp_path / "one.vlt"
    write_vault(eps, mids, one, ego_index=0, meta={"variant": "single"})

    chunked = tmp_path / "chunked.vlt"
    w = VaultWriter(chunked, ego_index=0, meta={"variant": "single"})
    w.write(eps[:3], mids[:3])
    w.write(eps[3:7], mids[3:7])
    w.write(eps[7:], mids[7:])

    w.close()

    a, b = read_vault(one), read_vault(chunked)
    assert a.num_episodes == b.num_episodes == len(eps)
    assert np.array_equal(a.episode_lengths(), b.episode_lengths())
    assert np.array_equal(a.member_ids, b.member_ids)
    assert np.allclose(a.episode_returns(), b.episode_returns())
    for i in range(len(eps)):
        assert np.allclose(a.episodes[i].obs, b.episodes[i].obs)
        assert np.array_equal(a.episodes[i].actions, b.episodes[i].actions)


def test_norm_sidecar_matches_recompute_and_guards_on_count(tmp_path):
    """VaultWriter.close writes a norm_stats.json accumulated over the whole
    collection; norm_stats() must return it exactly, and must ignore it once its
    transition count no longer matches the vault (the append-safety guard)."""
    import json

    from oaht_bench.dataset.vault import VaultReader, VaultWriter

    batch = _batch([6, 7, 5, 8, 6, 7, 4, 9])
    eps, mids = batch.episodes, batch.member_ids
    vd = tmp_path / "v.vlt"
    w = VaultWriter(vd, ego_index=0, meta={"variant": "single"})
    w.write(eps[:5], mids[:5])  # two chunks -> cross-chunk accumulation
    w.write(eps[5:], mids[5:])
    w.close()

    sidecar = vd / "single" / "norm_stats.json"
    assert sidecar.exists()

    src = VaultReader(vd)
    stored = src.norm_stats()  # served from the sidecar

    # Exact recompute (chunked pass) with the sidecar hidden.
    sidecar.rename(sidecar.with_suffix(".bak"))
    exact = VaultReader(vd).norm_stats()
    assert np.allclose(stored[0], exact[0], atol=1e-6)  # obs_mean
    assert np.allclose(stored[1], exact[1], atol=1e-6)  # obs_std
    assert abs(stored[2] - exact[2]) < 1e-4  # rtg_scale

    # A stale count (as if the vault grew after the norm was written) is ignored:
    # the reader must fall through to recompute rather than trust it.
    d = json.loads(sidecar.with_suffix(".bak").read_text())
    d["n_transitions"] = int(d["n_transitions"]) + 1
    sidecar.write_text(json.dumps(d))
    guarded = VaultReader(vd)
    assert guarded._read_norm_sidecar() is None


def test_two_dimensional_index_matches():
    """Stage 2 indexes with a (batch, C) context array; the lazy proxy must
    reshape the same way."""
    batch = _batch([6, 6, 6, 6, 6, 6])
    kw = dict(context_length=8, stride=4, normalize=True)
    eager, lazy = _build_windows(batch, **kw), Windows(batch, **kw)
    cidx = np.array([[0, 1, 2], [3, 4, 5]])
    for k in ("mate_next_obs", "mate_actions", "mask"):
        a, b = getattr(eager, k)[cidx], getattr(lazy, k)[cidx]
        assert a.shape == b.shape and np.allclose(a, b, atol=1e-5), k
