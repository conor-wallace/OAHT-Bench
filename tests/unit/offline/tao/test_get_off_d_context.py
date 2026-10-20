"""GetOffD (Sec 3.2, p.5): the offline opponent in-context data construction.

"GetOffD begins by sampling C trajectories from T^{-1,k}, following which it
samples H consecutive fragments ... from each trajectory and ultimately
stitches them together." Two properties follow directly from that sentence:
the context is C fragments of the *same* teammate (tested, passes), and the C
fragments come from C *distinct* episodes -- "it can showcase different
playing behaviors across various episodes" is the paper's own stated reason
for C>1 (tested below; currently fails -- see the xfail).
"""

from __future__ import annotations

import numpy as np
import pytest

from oaht_bench.dataset.dataset import TeammateIndex, _build_windows
from oaht_bench.dataset.sampler import sample_stage2


def _windows(n_ep=8, T=16, obs_dim=5, teammates=(0, 1, 2, 3), seed=0):
    from oaht_bench.dataset.schema import Episode, EpisodeBatch

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


def _single_teammate_windows(n_ep=10, T=8, obs_dim=4, seed=0):
    """Every episode is the same teammate (id 0), with plenty of them -- enough
    distinct "other" episodes that a without-replacement sampler would never
    need to repeat, isolating whether the sampler itself enforces distinctness
    rather than just running out of candidates."""
    from oaht_bench.dataset.schema import Episode, EpisodeBatch

    rng = np.random.default_rng(seed)
    member_ids = np.array([[0, 0] for _ in range(n_ep)])
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
        context_length=T,
        stride=T,
    )


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


@pytest.mark.xfail(
    strict=True,
    reason="TeammateIndex.cross_trajectory (dataset/dataset.py) samples independently "
    "(with replacement) each of the C times sample_stage2 calls it, so GetOffD's C "
    "context trajectories are not guaranteed to be C distinct episodes. The paper's "
    "own rationale for C>1 (Sec 3.2, papers/tao.pdf p.5): 'it can showcase different "
    "playing behaviors across various episodes -- hence we sample from multiple "
    "trajectories.' This fixture gives 9 other episodes per row (ample) so a correct "
    "without-replacement sampler would never need to repeat; ours still does.",
)
def test_get_off_d_context_trajectories_are_distinct_episodes():
    w = _single_teammate_windows(n_ep=10)
    idx = TeammateIndex.build(w)
    C = 5
    b = sample_stage2(w, idx, np.random.default_rng(0), batch_size=50, context_trajectories=C)
    T = w.context_length
    for row in range(b["context_mate_next_obs"].shape[0]):
        frags = b["context_mate_next_obs"][row].reshape(C, T, w.obs_dim)
        episodes_seen = []
        for f in frags:
            hits = np.flatnonzero((np.abs(w.mate_next_obs - f).sum(axis=(1, 2))) < 1e-6)
            episodes_seen.append(int(w.episode_id[hits[0]]))
        assert len(set(episodes_seen)) == C, (
            f"row {row}: expected {C} distinct episodes, got {episodes_seen}"
        )
