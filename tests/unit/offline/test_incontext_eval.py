"""The Opponent Context Window buffer that TAO/OMIS accumulate across episodes.

The cross-episode eval loop and rollout are exercised end-to-end against a real
env + teammate in the validation run; here we pin the buffer semantics that the
adaptation curve depends on -- an empty window must produce *no* context (episode
1 vs a teammate), the most recent trajectory must land last, and the window must
respect its capacity ``C``.
"""

from __future__ import annotations

import numpy as np

from oaht_bench.offline.incontext_eval import OpponentContextWindow


def test_empty_window_has_no_valid_context():
    """Episode 1 vs a teammate: nothing observed yet, so every mask position is 0
    and the encoder/cross-attention sees no context."""
    ocw = OpponentContextWindow(capacity=3, horizon=5, obs_dim=4)
    no, ac, rw, ts, mk = ocw.arrays()
    assert no.shape == (3, 5, 4) and ac.shape == (3, 5) and mk.shape == (3, 5)
    assert mk.sum() == 0.0
    assert len(ocw) == 0


def test_front_padded_most_recent_last():
    """Fragments fill from the back, so the newest trajectory is the last slot and
    unfilled slots stay masked -- the encoder reads a fixed (C, T, ...) either way."""
    ocw = OpponentContextWindow(capacity=3, horizon=5, obs_dim=4)
    ocw.append(np.ones((3, 4)), np.array([1, 2, 3]), np.zeros(3), np.arange(3))  # 3 steps
    ocw.append(np.ones((5, 4)), np.array([0, 1, 0, 1, 0]), np.zeros(5), np.arange(5))  # 5 steps
    no, ac, rw, ts, mk = ocw.arrays()
    assert mk[0].sum() == 0  # front slot empty
    assert int(mk[1].sum()) == 3 and int(mk[2].sum()) == 5  # valid lengths
    assert ac[2, :5].tolist() == [0, 1, 0, 1, 0]  # most recent is last


def test_capacity_is_a_ring_buffer():
    """Only the last C trajectories are kept; older ones roll off (opponent switch
    handling is a full reset, tested implicitly by capacity)."""
    ocw = OpponentContextWindow(capacity=2, horizon=4, obs_dim=2)
    for a in range(4):
        ocw.append(np.zeros((2, 2)), np.array([a, a]), np.zeros(2), np.arange(2))
    assert len(ocw) == 2
    _, ac, _, _, mk = ocw.arrays()
    # the two most recent appends (actions 2 and 3) survive, newest last
    assert ac[0, 0] == 2 and ac[1, 0] == 3


def test_reset_empties_the_window():
    ocw = OpponentContextWindow(capacity=2, horizon=4, obs_dim=2)
    ocw.append(np.zeros((2, 2)), np.array([1, 1]), np.zeros(2), np.arange(2))
    ocw.reset()
    assert len(ocw) == 0
    assert ocw.arrays()[4].sum() == 0.0
