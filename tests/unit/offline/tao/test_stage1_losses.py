"""Stage 1, Policy Embedding Learning -- Eq. 2 (generative), Eq. 3
(discriminative, SupCon), Eq. 4 (combined: L_emb = alpha*L_gen + lam*L_dis).

Eq. 3 was already well covered (SupCon aggregation, ragged-positive handling);
relocated here verbatim. Eq. 2/4 were only shape-tested before -- this adds the
numeric cross-check the other equations already get: run the real
``embedding_loss`` on a real stage-1 batch, then recompute both terms
independently from the same tokens and assert they agree.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from oaht_bench.dataset.windowing import window_slice_batch
from oaht_bench.models.masking import mask_logits
from oaht_bench.models.tao_agent import AncillaryActionDecoder, OpponentPolicyEncoder
from oaht_bench.offline import embedding_loss, supervised_contrastive
from oaht_bench.offline.tao import _ANCHOR_FIELDS, _CROSS_FIELDS, _anchor_and_cross_indices
from oaht_bench.offline.utils import safe_cross_entropy


def _dataset(n_ep=8, T=16, obs_dim=5, teammates=(0, 1, 2, 3), seed=0):
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
    batch = EpisodeBatch(episodes=episodes, member_ids=member_ids, ego_index=0, meta={})
    return SimpleNamespace(batch=batch, mate=1)


def _stage1_batch(ds, *, teammates_per_batch=3, windows_per_teammate=3, h=8, seed=0):
    """The same teammate-first contrastive batch ``TaoTrainer._stage1_batch``
    builds, standalone -- a real stage-1 batch through
    ``_anchor_and_cross_indices`` + ``window_slice_batch``."""
    rng = np.random.default_rng(seed)
    teammate_ids = np.unique(ds.batch.member_ids[:, ds.mate])
    chosen = rng.choice(
        teammate_ids, size=min(teammates_per_batch, len(teammate_ids)), replace=False
    )

    anchor_episode_ids, anchor_ts, cross_episode_ids, cross_ts, teammate_col = [], [], [], [], []
    for teammate_id in chosen:
        for _ in range(windows_per_teammate):
            ae, at, ce, ct = _anchor_and_cross_indices(ds, int(teammate_id), rng)
            anchor_episode_ids.append(ae)
            anchor_ts.append(at)
            cross_episode_ids.append(ce)
            cross_ts.append(ct)
            teammate_col.append(int(teammate_id))

    anchors = window_slice_batch(
        ds.batch, anchor_episode_ids, anchor_ts, h, _ANCHOR_FIELDS, mate=ds.mate
    )
    crosses = window_slice_batch(
        ds.batch, cross_episode_ids, cross_ts, h, _CROSS_FIELDS, mate=ds.mate
    )
    batch = dict(anchors)
    batch.update({f"cross_{k}": v for k, v in crosses.items()})
    batch["teammate_id"] = np.asarray(teammate_col, dtype=np.int32)
    return {k: jnp.asarray(v) for k, v in batch.items()}


def test_supervised_contrastive_matches_the_reference_aggregation():
    """SupCon, not plain InfoNCE -- the reference averages log-prob over positives.

    ``-(T/T_base) * mean_over_positives(log_prob)``, on raw dot products with a
    per-row max subtracted (``offline_stage_1/nn_trainer.py:130-156``). Computed
    directly here so a future edit back to log-sum-exp is caught.
    """
    z = jnp.asarray(np.eye(3), dtype=jnp.float32)
    labels = jnp.array([0, 0, 1])
    got = float(supervised_contrastive(z, labels, temperature=0.1, base_temperature=0.1))

    sim = np.asarray(z @ z.T) / 0.1
    sim = sim - sim.max(axis=1, keepdims=True)
    eye = np.eye(3, dtype=bool)
    exp = np.exp(sim) * ~eye
    log_prob = sim - np.log(exp.sum(axis=1, keepdims=True))
    pos = (np.asarray(labels)[:, None] == np.asarray(labels)[None, :]) & ~eye
    rows = pos.sum(1) > 0
    expected = -((pos * log_prob).sum(1)[rows] / pos.sum(1)[rows]).mean()
    assert got == pytest.approx(expected, rel=1e-5)


def test_supervised_contrastive_drops_rows_with_no_positive():
    """Ragged per-teammate coverage is the common case, not an edge case.

    The reference divides by ``dis_mask.sum(1)`` unguarded because its sampler
    guarantees a positive per anchor; ours cannot, since seats are sampled
    independently.
    """
    assert float(supervised_contrastive(jnp.eye(3), jnp.array([0, 1, 2]))) == 0.0


def _stage1_params_and_batch():
    """A real stage-1 batch (through ``_anchor_and_cross``) and
    freshly-initialised encoder/decoder params, the way
    ``TaoTrainer.train_stage_1`` builds them."""
    ds = _dataset()
    batch = _stage1_batch(ds, teammates_per_batch=3, windows_per_teammate=3)

    rng = jax.random.PRNGKey(0)
    enc = OpponentPolicyEncoder(action_dim=6)
    enc_params = enc.init(
        rng,
        batch["mate_next_obs"],
        batch["mate_actions"],
        batch["mate_rewards"],
        mask=batch["mask"],
        timesteps=batch["timesteps"],
    )
    tokens = enc.apply(
        enc_params,
        batch["mate_next_obs"],
        batch["mate_actions"],
        batch["mate_rewards"],
        mask=batch["mask"],
        timesteps=batch["timesteps"],
    )
    z_bar = OpponentPolicyEncoder.pool(tokens)
    dec = AncillaryActionDecoder(action_dim=6)
    dec_params = dec.init(rng, batch["cross_mate_obs"], z_bar)
    return batch, enc, dec, {"encoder": enc_params, "decoder": dec_params}


def test_embedding_loss_is_alpha_gen_plus_lam_dis():
    """Eq. 4: L_emb = alpha*L_gen + lam*L_dis, with mismatched weights so the
    test would fail if either term's coefficient were swapped or dropped."""
    batch, enc, dec, params = _stage1_params_and_batch()
    alpha, lam = 2.0, 0.5

    total, aux = embedding_loss(params, enc, dec, batch, alpha=alpha, lam=lam, train=False)
    assert float(total) == pytest.approx(
        alpha * float(aux["generative"]) + lam * float(aux["discriminative"]), rel=1e-5
    )


def test_embedding_loss_generative_term_scores_a_different_episode_under_z_bar():
    """Eq. 2: the generative term conditions on the anchor's pooled embedding but
    is *scored* on a different trajectory of the same teammate -- recomputed here
    independently of ``embedding_loss`` from the same tokens, the same way
    Eq. 5's masked-mean is cross-checked in test_stage2_response_loss.py."""
    batch, enc, dec, params = _stage1_params_and_batch()
    _, aux = embedding_loss(params, enc, dec, batch, train=False)

    tokens = enc.apply(
        params["encoder"],
        batch["mate_next_obs"],
        batch["mate_actions"],
        batch["mate_rewards"],
        mask=batch["mask"],
        timesteps=batch["timesteps"],
    )
    z_bar = OpponentPolicyEncoder.pool(tokens)
    logits = mask_logits(
        dec.apply(params["decoder"], batch["cross_mate_obs"], z_bar), batch["cross_mate_avail"]
    )
    cross_mask = batch["cross_mask"].astype(jnp.float32)
    gen = safe_cross_entropy(logits, batch["cross_mate_actions"], cross_mask)
    gen = (gen * cross_mask).sum() / jnp.maximum(cross_mask.sum(), 1.0)

    assert float(aux["generative"]) == pytest.approx(float(gen), rel=1e-5)


def test_ancillary_decoder_conditions_on_the_embedding():
    """The decoder's input is (mate_obs, z_bar) -- Eq. 2's pi_phi_d(a | o, z_bar)."""
    n_rows, T, obs_dim = 8, 16, 5
    mate_obs = jnp.asarray(np.random.default_rng(0).normal(size=(n_rows, T, obs_dim)))
    rng = jax.random.PRNGKey(0)
    dec = AncillaryActionDecoder(action_dim=6)
    z = jnp.zeros((n_rows, 32))
    p = dec.init(rng, mate_obs, z)
    out = dec.apply(p, mate_obs, z)
    assert out.shape == (n_rows, T, 6)
