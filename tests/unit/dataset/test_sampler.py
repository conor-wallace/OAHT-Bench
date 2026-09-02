"""Batch structure the two stages require, which uniform sampling cannot give.

Both stages need relationships *across* the batch — a cross trajectory of the
same teammate, several positives per anchor, a context drawn from one teammate —
so these test the sampler's guarantees rather than its shapes alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from oaht_bench.dataset.dataset import TeammateIndex, _build_windows
from oaht_bench.dataset.sampler import sample_stage1, sample_stage2
from oaht_bench.dataset.schema import EpisodeBatch


def _windows(n_ep=8, T=16, obs_dim=5, teammates=(0, 1, 2, 3), seed=0):
    from oaht_bench.dataset.schema import Episode

    rng = np.random.default_rng(seed)
    # two episodes per teammate, so "a different episode" is always available
    member_ids = np.array([[0, teammates[i % len(teammates)]] for i in range(n_ep)])
    episodes = [
        Episode(
            obs=rng.normal(size=(2, T, obs_dim)).astype(np.float32),
            actions=rng.integers(0, 6, size=(2, T)),
            rewards=rng.normal(size=(2, T)).astype(np.float32),
            avail_actions=np.ones((2, T, 6), dtype=np.float32),
            dones=np.zeros(T, dtype=bool),
        )
        for _ in range(n_ep)
    ]
    return _build_windows(
        EpisodeBatch(episodes=episodes, member_ids=member_ids, ego_index=0, meta={}),
        context_length=8,
        stride=4,
    )


def test_windows_record_the_episode_they_came_from():
    """Two overlapping windows of one episode are not two trajectories."""
    w = _windows()
    assert w.episode_id.shape == (len(w),)
    # a single episode yields several windows sharing an episode_id
    _, counts = np.unique(w.episode_id, return_counts=True)
    assert counts.max() > 1


def test_stage1_guarantees_positives_for_every_anchor():
    """SupCon drops anchors with no positive; uniform sampling produces many.

    The reference gets positives for free from many trajectories per opponent.
    Our per-teammate coverage is ragged, so the sampler groups by teammate.
    """
    w = _windows()
    idx = TeammateIndex.build(w)
    b = sample_stage1(
        w, idx, np.random.default_rng(0), teammates_per_batch=3, windows_per_teammate=4
    )
    _, counts = np.unique(b["teammate_id"], return_counts=True)
    assert counts.min() >= 4  # so every anchor has >= 3 positives
    assert len(b["teammate_id"]) == 12


def test_stage1_cross_trajectory_is_the_same_teammate_from_another_episode():
    """The generative term's whole point: condition on one episode, score another."""
    w = _windows()
    idx = TeammateIndex.build(w)
    rng = np.random.default_rng(0)
    for _ in range(20):
        t = int(rng.choice(idx.teammates))
        pool = idx.by_teammate[t]
        ep = int(w.episode_id[pool[0]])
        j = idx.cross_trajectory(t, ep, rng)
        assert int(w.teammate_id[j]) == t
        assert int(w.episode_id[j]) != ep  # this fixture always has an alternative


def test_stage1_rejects_a_batch_that_cannot_have_positives():
    w = _windows()
    idx = TeammateIndex.build(w)
    with pytest.raises(ValueError, match="at least 2"):
        sample_stage1(w, idx, np.random.default_rng(0), windows_per_teammate=1)


def test_stage2_context_is_c_fragments_of_one_teammate():
    """GetOffD: C trajectories, a fragment of each, concatenated along time."""
    w = _windows()
    idx = TeammateIndex.build(w)
    C = 5
    b = sample_stage2(w, idx, np.random.default_rng(0), batch_size=6, context_trajectories=C)
    T = w.context_length
    assert b["ego_obs"].shape == (6, T, w.obs_dim)
    assert b["context_mate_next_obs"].shape == (6, C * T, w.obs_dim)
    assert b["context_mask"].shape == (6, C * T)
    # the context is longer than the decoder window, which only works because it
    # enters as cross-attention keys rather than being concatenated
    assert b["context_mask"].shape[1] > b["mask"].shape[1]


def test_stage2_context_matches_the_decoder_window_teammate():
    """Context must describe the teammate actually being played with."""
    w = _windows()
    idx = TeammateIndex.build(w)
    rng = np.random.default_rng(1)
    b = sample_stage2(w, idx, rng, batch_size=8, context_trajectories=3)
    # reconstruct which windows fed each context row by matching observations
    for row in range(len(b["teammate_id"])):
        want = int(b["teammate_id"][row])
        frag = b["context_mate_next_obs"][row].reshape(3, w.context_length, w.obs_dim)
        for f in frag:
            hits = np.flatnonzero((np.abs(w.mate_next_obs - f).sum(axis=(1, 2))) < 1e-6)
            assert hits.size, "context fragment should come from the window pool"
            assert int(w.teammate_id[hits[0]]) == want
