"""GetOffD (Sec 3.2, p.5): the offline opponent in-context data construction.

"GetOffD begins by sampling C trajectories from T^{-1,k}, following which it
samples H consecutive fragments ... from each trajectory and ultimately
stitches them together." Two properties follow directly from that sentence:
the C trajectories are the *same* teammate, and they come from C *distinct*
episodes -- "it can showcase different playing behaviors across various
episodes" is the paper's own stated reason for C>1. The second property used
to fail (``TeammateIndex.cross_trajectory`` sampled each of the ``C``
fragments independently and so could repeat one); ``_get_off_d_context``'s
``rng.choice(..., replace=pool.size < c)`` draws distinct episode ids by
construction, so it now passes.

Both properties are checked on the *episode ids actually drawn* via a
recording ``rng`` wrapper, rather than by reconstructing which episode a
fragment came from from its values -- the latter breaks the moment a
fragment's random start position is less than ``h`` (left-padding), since a
padded fragment no longer matches any real slice of the episode it was read
from.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from oaht_bench.offline.tao import _get_off_d_context


class _RecordingRng:
    """Wraps a real ``np.random.Generator``, remembering the episode ids the
    last ``choice`` call drew."""

    def __init__(self, rng):
        self._rng = rng
        self.last_choice = None

    def choice(self, *args, **kwargs):
        out = self._rng.choice(*args, **kwargs)
        self.last_choice = out
        return out

    def integers(self, *args, **kwargs):
        return self._rng.integers(*args, **kwargs)


def _batch(n_ep=8, T=16, obs_dim=5, teammates=(0, 1, 2, 3), seed=0):
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
    return EpisodeBatch(episodes=episodes, member_ids=member_ids, ego_index=0, meta={})


def _single_teammate_batch(n_ep=10, T=8, obs_dim=4, seed=0):
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
    return EpisodeBatch(episodes=episodes, member_ids=member_ids, ego_index=0, meta={})


def test_get_off_d_context_is_c_fragments_of_one_teammate():
    """GetOffD: C trajectories, a fragment of each, concatenated along time."""
    batch = _batch()
    ds = SimpleNamespace(batch=batch, mate=1)
    rng = np.random.default_rng(0)
    C, h, obs_dim = 5, 8, 5

    ctx = _get_off_d_context(ds, teammate_ids=[1], exclude_episode_ids=[0], h=h, c=C, rng=rng)
    assert ctx["mate_next_obs"].shape == (1, C * h, obs_dim)
    assert ctx["mask"].shape == (1, C * h)


def test_get_off_d_context_is_batched_across_many_decoder_rows_at_once():
    """The same call with several rows' teammate_ids/exclude_episode_ids at
    once must give each row its own independent C fragments, not the first
    row's broadcast to every row."""
    batch = _batch()
    ds = SimpleNamespace(batch=batch, mate=1)
    rng = np.random.default_rng(0)
    C, h, obs_dim = 3, 8, 5

    ctx = _get_off_d_context(
        ds, teammate_ids=[0, 1, 2], exclude_episode_ids=[0, 2, 4], h=h, c=C, rng=rng
    )
    assert ctx["mate_next_obs"].shape == (3, C * h, obs_dim)
    assert ctx["mask"].shape == (3, C * h)


def test_get_off_d_context_draws_episodes_of_the_requested_teammate():
    """Context must describe the teammate actually being played with."""
    batch = _batch()
    ds = SimpleNamespace(batch=batch, mate=1)
    rng = _RecordingRng(np.random.default_rng(1))
    C, h = 3, 16

    for teammate_id in (0, 1, 2, 3):
        episode_ids = np.flatnonzero(batch.member_ids[:, 1] == teammate_id)
        for exclude in episode_ids:
            _get_off_d_context(ds, [teammate_id], [int(exclude)], h=h, c=C, rng=rng)
            for episode_id in rng.last_choice:
                assert int(batch.member_ids[episode_id, 1]) == teammate_id


def test_get_off_d_context_trajectories_are_distinct_episodes():
    """GetOffD's C context trajectories must be C *distinct* episodes (Sec
    3.2, p.5): "it can showcase different playing behaviors across various
    episodes -- hence we sample from multiple trajectories."
    """
    batch = _single_teammate_batch(n_ep=10)
    ds = SimpleNamespace(batch=batch, mate=1)
    rng = _RecordingRng(np.random.default_rng(0))
    C, h = 5, 8

    for row in range(50):
        _get_off_d_context(ds, [0], [row % 10], h=h, c=C, rng=rng)
        episodes_seen = rng.last_choice.tolist()
        assert len(set(episodes_seen)) == C, (
            f"row {row}: expected {C} distinct episodes, got {episodes_seen}"
        )
