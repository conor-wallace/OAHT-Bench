"""``window_slice_batch``, checked against decoded ground truth: build an
``EpisodeBatch`` fully-decodable per ``(agent, policy, episode, timestep,
channel)`` (``conftest.py``) and confirm every field it hands back decodes to
the exact position it should -- the windowing/gather logic itself, one level
below the sampler-level checks in ``test_deterministic_sampling.py``.
"""

from __future__ import annotations

import numpy as np
from conftest import CHANNEL_ACTION, CHANNEL_OBS, EGO, MATE, decode

from oaht_bench.dataset.windowing import window_slice_batch


def test_window_slice_batch_ego_and_mate_fields_decode_to_the_right_window(deterministic_dataset):
    """No padding, no overlap (``t == h == episode length``): every position
    of every window should decode to its own episode's ``teammate_id``/
    ``episode_id`` and a timestep running ``0..T-1`` -- for all four
    episodes read in ONE batched call, not decoded one at a time.
    """
    T = 6
    teammate_sequence = [1, 1, 3, 3]
    ds = deterministic_dataset(teammate_sequence, T=T)

    episode_ids = list(range(len(teammate_sequence)))
    windows = window_slice_batch(
        ds.batch,
        episode_ids,
        [T] * len(episode_ids),
        T,
        ("ego_obs", "mate_obs", "ego_actions", "mate_actions"),
    )
    for row, (episode_id, k) in enumerate(zip(episode_ids, teammate_sequence, strict=True)):
        for field, agent, channel in (
            ("ego_obs", EGO, CHANNEL_OBS),
            ("mate_obs", MATE, CHANNEL_OBS),
            ("ego_actions", EGO, CHANNEL_ACTION),
            ("mate_actions", MATE, CHANNEL_ACTION),
        ):
            values = np.asarray(windows[field][row])
            if values.ndim == 2:  # obs carries an extra feature axis
                values = values[:, 0]
            for t, v in enumerate(values):
                got_agent, got_k, got_e, got_t, got_c = decode(v)
                assert (got_agent, got_k, got_e, got_t, got_c) == (
                    agent,
                    k,
                    episode_id,
                    t,
                    channel,
                ), field


def test_mate_next_obs_is_shifted_one_step_and_clamped_at_episode_end(deterministic_dataset):
    """``mate_next_obs`` fuses ``(a_{t-1}, r_{t-1}, o_t)`` by reading the
    teammate's observation one step ahead (``windowing.py``'s own
    docstring); the last real position repeats rather than reading past the
    episode end.
    """
    T = 5
    ds = deterministic_dataset([2], T=T)
    k, episode_id = 2, 0
    windows = window_slice_batch(ds.batch, [episode_id], [T], T, ("mate_next_obs",))
    next_obs = np.asarray(windows["mate_next_obs"][0])[:, 0]

    for t in range(T - 1):
        assert decode(next_obs[t]) == (MATE, k, episode_id, t + 1, CHANNEL_OBS)
    # last position: o_{T-1} repeats (no o_T exists)
    assert decode(next_obs[-1]) == (MATE, k, episode_id, T - 1, CHANNEL_OBS)


def test_window_slice_batch_pads_short_episodes_at_the_front_with_real_data_at_the_end(
    deterministic_dataset,
):
    """Left-padding convention: a window shorter than ``h`` keeps real data
    at the END, with ``-10``-filled actions and ``mask=False`` at the front
    -- the most recent timestep is always last.
    """
    h = 6
    ds = deterministic_dataset([5], T=h, episode_lengths=[3])
    k, episode_id = 5, 0
    n = 3  # real length

    windows = window_slice_batch(ds.batch, [episode_id], [n], h, ("ego_actions", "mask"))
    actions = np.asarray(windows["ego_actions"][0])
    mask = np.asarray(windows["mask"][0])
    assert list(actions[: h - n]) == [-10] * (h - n), "front is padding"
    assert list(mask[: h - n]) == [False] * (h - n)
    assert list(mask[h - n :]) == [True] * n
    for t, v in enumerate(actions[h - n :]):
        assert decode(v) == (EGO, k, episode_id, t, CHANNEL_ACTION)
