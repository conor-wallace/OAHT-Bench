"""``window_slice_batch``/``teammate_episode_ids`` against a hand-built
``EpisodeBatch``, checked on decoded values rather than shapes alone.

Every scalar is ``seat * 100 + t`` (observations repeat that value across
``obs_dim``), so a value read back out of a window decodes to exactly which
seat/timestep it came from -- enough to check padding, the ``next_obs``
shift, and the return-to-go reverse-cumsum without a separate test oracle.
"""

from __future__ import annotations

import numpy as np
import pytest

from oaht_bench.dataset.schema import Episode, EpisodeBatch
from oaht_bench.dataset.windowing import enumerate_windows, teammate_episode_ids, window_slice_batch

OBS_DIM = 3


def _value(seat: int, t: int) -> int:
    return seat * 100 + t


def _episode(length: int, num_agents: int = 2) -> Episode:
    obs = np.array(
        [[[_value(seat, t)] * OBS_DIM for t in range(length)] for seat in range(num_agents)],
        dtype=np.float32,
    )
    actions = np.array(
        [[_value(seat, t) for t in range(length)] for seat in range(num_agents)], dtype=np.int64
    )
    rewards = np.array(
        [[float(t + 1) for t in range(length)] for _ in range(num_agents)], dtype=np.float32
    )
    avail = np.ones((num_agents, length, 4), dtype=np.float32)
    return Episode(
        obs=obs,
        actions=actions,
        rewards=rewards,
        avail_actions=avail,
        dones=np.zeros(length, dtype=bool),
    )


def _batch(lengths, member_ids, num_agents=2, ego_index=0) -> EpisodeBatch:
    episodes = [_episode(n, num_agents=num_agents) for n in lengths]
    return EpisodeBatch(
        episodes=episodes, member_ids=np.array(member_ids), ego_index=ego_index, meta={}
    )


def test_window_slice_batch_reads_the_unpadded_slice_when_t_covers_the_whole_window():
    batch = _batch(lengths=[6], member_ids=[[0, 1]])
    out = window_slice_batch(batch, [0], [6], h=4, fields=("ego_obs", "mate_actions"))
    assert out["ego_obs"].shape == (1, 4, OBS_DIM)
    assert list(out["ego_obs"][0, :, 0]) == [_value(0, t) for t in range(2, 6)]
    assert list(out["mate_actions"][0]) == [_value(1, t) for t in range(2, 6)]


def test_window_slice_batch_left_pads_when_t_is_less_than_h():
    batch = _batch(lengths=[6], member_ids=[[0, 1]])
    out = window_slice_batch(
        batch, [0], [2], h=5, fields=("ego_obs", "ego_actions", "mask", "timesteps")
    )
    # real data only at the last 2 positions; first 3 are left-padding
    assert list(out["ego_actions"][0]) == [-10, -10, -10, _value(0, 0), _value(0, 1)]
    assert list(out["ego_obs"][0, :3, 0]) == [0.0, 0.0, 0.0]
    assert list(out["ego_obs"][0, 3:, 0]) == [_value(0, 0), _value(0, 1)]
    assert list(out["mask"][0]) == [False, False, False, True, True]
    assert list(out["timesteps"][0]) == [0, 0, 0, 1, 2]


def test_mate_next_obs_is_shifted_by_one_and_clamps_at_episode_end():
    batch = _batch(lengths=[4], member_ids=[[0, 1]])
    out = window_slice_batch(batch, [0], [4], h=4, fields=("mate_next_obs",))
    # position t sees mate_obs[t+1], and the last real position repeats the
    # final observation rather than reading past the episode end.
    assert list(out["mate_next_obs"][0, :, 0]) == [
        _value(1, 1),
        _value(1, 2),
        _value(1, 3),
        _value(1, 3),
    ]


def test_ego_rtg_is_the_reverse_cumulative_reward_from_t_to_episode_end():
    batch = _batch(lengths=[4], member_ids=[[0, 1]])
    out = window_slice_batch(batch, [0], [4], h=4, fields=("ego_rtg",))
    # rewards are t+1 for t in 0..3 -> [1, 2, 3, 4]; reverse-cumsum from each position
    assert list(out["ego_rtg"][0]) == [10.0, 9.0, 7.0, 4.0]


def test_window_slice_batch_requires_mate_to_disambiguate_more_than_two_agents():
    batch = _batch(lengths=[4], member_ids=[[0, 1, 2]], num_agents=3)
    with pytest.raises(ValueError, match="ambiguous"):
        window_slice_batch(batch, [0], [4], h=4, fields=("mate_obs",))
    # explicit mate resolves it
    out = window_slice_batch(batch, [0], [4], h=4, fields=("mate_obs",), mate=2)
    assert list(out["mate_obs"][0, :, 0]) == [_value(2, t) for t in range(4)]


def test_window_slice_batch_rows_are_independent_and_match_their_own_episode_and_t():
    """The whole point of batching: many rows in one call, each still
    reading its own (episode_id, t) correctly -- not the first row's data
    broadcast, and not rows bleeding into each other.
    """
    batch = _batch(lengths=[6, 8, 5], member_ids=[[0, 1], [0, 1], [0, 1]])
    out = window_slice_batch(
        batch,
        episode_ids=[0, 1, 2, 1],
        ts=[6, 8, 5, 3],
        h=4,
        fields=("ego_obs", "mate_actions", "mask"),
    )
    assert out["ego_obs"].shape == (4, 4, OBS_DIM)

    # rows 0-2: t == length of a full window, so fully real, no padding
    expected_ego = [
        [_value(0, t) for t in range(2, 6)],  # episode 0, t=6
        [_value(0, t) for t in range(4, 8)],  # episode 1, t=8
        [_value(0, t) for t in range(1, 5)],  # episode 2, t=5
    ]
    expected_mate = [
        [_value(1, t) for t in range(2, 6)],
        [_value(1, t) for t in range(4, 8)],
        [_value(1, t) for t in range(1, 5)],
    ]
    for row in range(3):
        assert list(out["ego_obs"][row, :, 0]) == expected_ego[row]
        assert list(out["mate_actions"][row]) == expected_mate[row]
        assert list(out["mask"][row]) == [True, True, True, True]

    # row 3: episode 1, t=3, h=4 -> start=0, n=3 -- left-padded by one
    assert list(out["mask"][3]) == [False, True, True, True]
    assert list(out["ego_obs"][3, 1:, 0]) == [_value(0, 0), _value(0, 1), _value(0, 2)]
    assert out["ego_obs"][3, 0, 0] == 0.0
    assert list(out["mate_actions"][3]) == [-10, _value(1, 0), _value(1, 1), _value(1, 2)]


def test_enumerate_windows_matches_a_hand_counted_stride_enumeration():
    # length=6, h=4, stride=2 -> starts 0, 2 (range(0, max(1, 6-4+1)=3, 2));
    # ends = start + min(h, length-start) = [4, 6]
    batch = _batch(lengths=[6], member_ids=[[0, 1]])
    episode_ids, ts = enumerate_windows(batch, h=4, stride=2)
    assert list(episode_ids) == [0, 0]
    assert list(ts) == [4, 6]


def test_enumerate_windows_gives_one_start_for_an_episode_shorter_than_h():
    batch = _batch(lengths=[3], member_ids=[[0, 1]])
    episode_ids, ts = enumerate_windows(batch, h=4, stride=2)
    assert list(episode_ids) == [0]
    assert list(ts) == [3]


def test_enumerate_windows_spans_every_episode():
    batch = _batch(lengths=[6, 3], member_ids=[[0, 1], [0, 1]])
    episode_ids, ts = enumerate_windows(batch, h=4, stride=2)
    assert list(episode_ids) == [0, 0, 1]


def test_teammate_episode_ids_filters_only_on_the_mate_seat():
    # episode 0: mate=5 (ego=0); episode 1: mate=5 (ego=1, a different ego
    # policy -- must still count); episode 2: mate=7.
    batch = _batch(lengths=[4, 4, 4], member_ids=[[0, 5], [1, 5], [0, 7]])
    ids = teammate_episode_ids(batch, mate=1, teammate_id=5)
    assert list(ids) == [0, 1]
