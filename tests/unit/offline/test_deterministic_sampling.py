"""TAO/OMIS's data sampling, checked against decoded ground truth rather than
shapes alone -- see ``conftest.py`` for the encoding scheme.

These assert the paper's *claims* directly on real decoded data instead of
re-deriving them from reading the code:

* Appendix C: a ``GetOffD``/``D_epi`` context fragment is a *consecutive*
  segment, and comes from a *different episode* of the *same* teammate as the
  anchor it's attached to.
* Eq. 2 (``D_step^k``): the opponent-action history at position ``t`` is
  ``a^{-1,k}_{t-1}``, never ``a^{-1,k}_t``.

Every check below reads only ``mask``-true positions: a window's *left*
padding fill (``0.0`` for observations) is indistinguishable by value from a
real ``decode``d ``(0, 0, 0, 0, 0)``, so an unmasked read would intermittently
misread padding as a real teammate-0/episode-0 position.
"""

from __future__ import annotations

import numpy as np
from conftest import MATE, decode

from oaht_bench.dataset.windowing import window_slice_batch
from oaht_bench.offline.omis import build_d_epi, build_d_step
from oaht_bench.offline.tao import _ANCHOR_FIELDS, _CROSS_FIELDS, _anchor_and_cross_indices


def test_build_d_epi_context_is_the_same_teammate_from_a_different_episode(
    deterministic_dataset,
):
    """Appendix C step 1: ``C`` trajectories "from all historical games
    involving pi^{-1,k}" -- not just any teammate, and (per
    :func:`~oaht_bench.offline.omis.build_d_epi`'s documented fallback) not
    the decoder's own episode when an alternative exists.
    """
    # Every teammate needs >= 2 episodes, or landing on a single-episode
    # teammate correctly (per build_d_epi's documented fallback) reuses that
    # same episode -- not what this test is checking.
    teammate_sequence = [1, 1, 1, 1, 2, 2]
    ds = deterministic_dataset(teammate_sequence)
    rng = np.random.default_rng(0)

    episode_ids = list(range(len(teammate_sequence)))
    d_epi = build_d_epi(ds, teammate_sequence, episode_ids, rng, H=9, C=3)
    for row, (episode_id, teammate_id) in enumerate(
        zip(episode_ids, teammate_sequence, strict=True)
    ):
        for pos in np.flatnonzero(d_epi["mask"][row]):
            agent, k, e, _t, _c = decode(d_epi["mate_obs"][row, pos, 0])
            assert agent == MATE
            assert k == teammate_id, "D_epi context must be the SAME teammate"
            assert e != episode_id, "must come from a DIFFERENT episode than the decoder's own"


def test_build_d_epi_fragments_are_consecutive_timesteps(deterministic_dataset):
    """Appendix C step 2: "sample consecutive segments" -- within one fragment
    (one trajectory's contribution to the ``C``-fragment context), timesteps
    must increase by exactly 1, not be shuffled or resampled per-position.
    """
    ds = deterministic_dataset([1, 1, 1, 1, 1], T=6)
    rng = np.random.default_rng(1)
    C, H = 3, 6
    h = H // C
    d_epi = build_d_epi(ds, [1], [0], rng, H=H, C=C)

    fragments = d_epi["mate_actions"][0].reshape(C, h)
    mask_fragments = d_epi["mask"][0].reshape(C, h)
    for frag, frag_mask in zip(fragments, mask_fragments, strict=True):
        real = frag[frag_mask]
        timesteps = [decode(v)[3] for v in real]
        assert timesteps == list(range(timesteps[0], timesteps[0] + len(timesteps))), (
            f"fragment not consecutive: {timesteps}"
        )


def test_anchor_and_cross_share_teammate_but_differ_in_episode(deterministic_dataset):
    """TAO stage 1's contrastive positive (Eq. 2): the cross trajectory used
    for the generative term must be the *same* teammate, a *different*
    episode -- "a different trajectory of the same teammate".
    """
    ds = deterministic_dataset([1, 1, 1, 1, 2, 2, 2, 2])
    rng = np.random.default_rng(2)

    for teammate_id in (1, 2):
        for _ in range(10):
            ae, at, ce, ct = _anchor_and_cross_indices(ds, teammate_id, rng)
            anchor = window_slice_batch(ds.batch, [ae], [at], 6, _ANCHOR_FIELDS, mate=MATE)
            cross = window_slice_batch(ds.batch, [ce], [ct], 6, _CROSS_FIELDS, mate=MATE)
            anchor_pos = np.flatnonzero(anchor["mask"][0])[0]
            cross_pos = np.flatnonzero(cross["mask"][0])[0]
            anchor_agent, anchor_k, anchor_e, _, _ = decode(
                anchor["mate_next_obs"][0, anchor_pos, 0]
            )
            cross_agent, cross_k, cross_e, _, _ = decode(cross["mate_obs"][0, cross_pos, 0])
            assert anchor_agent == cross_agent == MATE
            assert anchor_k == cross_k == teammate_id
            assert cross_e != anchor_e


def test_build_d_step_reads_the_opponents_action_one_step_before(deterministic_dataset):
    """Eq. 2's ``D_step^k = (s_0, a^{-1,k}_0, ..., s_{t-1}, a^{-1,k}_{t-1})``:
    the conditioning input available at position ``t`` is the opponent's
    action from ``t-1``, never ``t`` -- checked against real decoded values,
    not just the shift's index arithmetic.
    """
    ds = deterministic_dataset([4, 4, 4], T=6)
    batch = window_slice_batch(ds.batch, [0], [6], 6, ("mate_actions", "mate_obs"), mate=MATE)

    d_step = build_d_step(batch)
    shifted = d_step["mate_actions"]
    raw_agent, raw_k, raw_e, _, _ = decode(batch["mate_actions"][0, 1])

    for t in range(1, 6):
        agent, k, e, shifted_t, channel = decode(shifted[0, t])
        assert (agent, k, e, channel) == (MATE, raw_k, raw_e, 0)
        assert shifted_t == t - 1, "position t must carry the opponent's action from t-1"

    assert int(shifted[0, 0]) == -10, "position 0 has no history yet"
