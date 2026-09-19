"""Batch construction for the two-stage baselines.

Neither stage can be fed by slicing :class:`~oaht_bench.dataset.dataset.Windows`
uniformly, because both need *structure* across the batch rather than just a
random subset:

* **Stage 1** pairs each anchor with a different trajectory of the same teammate
  (``offline_stage_1/utils.py:105-113``), and its contrastive term needs at least
  one other window of that teammate *in the same batch* or the anchor
  contributes nothing.
* **Stage 2** builds TAO's ``GetOffD``: for each decoder window, sample ``C``
  trajectories of the same teammate, take a fragment of each, and concatenate
  them into one context of length ``C × H``
  (``offline_stage_2/utils.py:362-392``).

One deliberate divergence, forced by our data. The reference draws
``batch_size`` trajectories uniformly from the whole pool, which lands positives
in the batch because it has many trajectories per opponent. Our per-teammate
coverage is ragged — 1 to 4 episodes each on the LBF datasets, since seats are
sampled independently at collection — so uniform sampling frequently produces
batches where most anchors have no positive and drop out of the contrastive
mean. :func:`sample_stage1` therefore samples *teammates* first and several
windows each, which guarantees positives by construction. The alternative is to
fix collection, which is the better fix and is tracked separately.
"""

from __future__ import annotations

import numpy as np

from oaht_bench.dataset.dataset import TeammateIndex, Windows

#: Keys a stage-1 batch carries for the anchor trajectory.
_ANCHOR_KEYS = (
    "mate_next_obs",
    "mate_actions",
    "mate_rewards",
    "mask",
    "timesteps",
)


def _take(windows: Windows, idx: np.ndarray, keys, prefix: str = "") -> dict[str, np.ndarray]:
    return {f"{prefix}{k}": getattr(windows, k)[idx] for k in keys}


def sample_stage1(
    windows: Windows,
    index: TeammateIndex,
    rng: np.random.Generator,
    *,
    teammates_per_batch: int = 4,
    windows_per_teammate: int = 4,
) -> dict[str, np.ndarray]:
    """A stage-1 batch: anchors, plus a cross trajectory for each.

    ``batch_size`` is ``teammates_per_batch * windows_per_teammate``. Sampling by
    teammate rather than uniformly is what guarantees each anchor has
    ``windows_per_teammate - 1`` positives available to the contrastive term.
    """
    if windows_per_teammate < 2:
        raise ValueError(
            "windows_per_teammate must be at least 2, or every anchor is its own "
            "only member of its class and the contrastive term has no positives."
        )
    available = index.teammates
    chosen = rng.choice(available, size=min(teammates_per_batch, len(available)), replace=False)

    anchors: list[int] = []
    for t in chosen:
        pool = index.by_teammate[int(t)]
        replace = pool.size < windows_per_teammate
        anchors.extend(rng.choice(pool, size=windows_per_teammate, replace=replace).tolist())
    anchor_idx = np.asarray(anchors)

    cross_idx = np.asarray(
        [
            index.cross_trajectory(int(windows.teammate_id[i]), int(windows.episode_id[i]), rng)
            for i in anchor_idx
        ]
    )

    batch = _take(windows, anchor_idx, _ANCHOR_KEYS)
    batch["teammate_id"] = windows.teammate_id[anchor_idx]
    # The generative term reads the cross trajectory's *observations* and scores
    # its *actions* -- not next-observations, which are the encoder's input.
    batch.update(
        _take(windows, cross_idx, ("mate_obs", "mate_actions", "mate_avail", "mask"), "cross_")
    )
    return batch


def sample_stage2(
    windows: Windows,
    index: TeammateIndex,
    rng: np.random.Generator,
    *,
    batch_size: int = 16,
    context_trajectories: int = 5,
) -> dict[str, np.ndarray]:
    """A stage-2 batch: decoder windows plus a ``GetOffD`` context for each.

    The context is ``context_trajectories`` fragments of the *same* teammate,
    concatenated along time into one sequence of length ``C × T``. The reference
    draws each fragment from a different trajectory with its own random start,
    on the stated rationale that play style is pronounced over consecutive
    timesteps but varies across episodes — hence fragments, and hence several of
    them.
    """
    decoder_idx = rng.choice(len(windows), size=batch_size, replace=len(windows) < batch_size)

    context_rows = []
    for i in decoder_idx:
        t, ep = int(windows.teammate_id[i]), int(windows.episode_id[i])
        context_rows.append(
            [index.cross_trajectory(t, ep, rng) for _ in range(context_trajectories)]
        )
    context_idx = np.asarray(context_rows)  # (batch, C)

    batch = {
        "ego_obs": windows.ego_obs[decoder_idx],
        "ego_actions": windows.ego_actions[decoder_idx],
        "ego_rtg": windows.ego_rtg[decoder_idx],
        "timesteps": windows.timesteps[decoder_idx],
        "mask": windows.mask[decoder_idx],
        "ego_avail": windows.ego_avail[decoder_idx],
        "teammate_id": windows.teammate_id[decoder_idx],
    }
    # Concatenate the C fragments along time: (batch, C, T, ...) -> (batch, C*T, ...)
    for key, out in (
        ("mate_next_obs", "context_mate_next_obs"),
        ("mate_actions", "context_mate_actions"),
        ("mate_rewards", "context_mate_rewards"),
        ("timesteps", "context_timesteps"),
        ("mask", "context_mask"),
    ):
        stacked = getattr(windows, key)[context_idx]
        batch[out] = stacked.reshape(stacked.shape[0], -1, *stacked.shape[3:])
    return batch
