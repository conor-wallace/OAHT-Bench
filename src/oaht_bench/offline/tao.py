"""TAO — offline opponent modelling via policy embeddings (Wang et al., ICLR 2024).

Three stages, per the paper and its Appendix F:

**Stage 1, Policy Embedding Learning.** An encoder ``M_θe`` over the teammate's
``(a⁻¹_{t-1}, r⁻¹_{t-1}, o⁻¹_t)`` stream, trained with two losses:

* *generative* (Eq. 2) — an ancillary decoder predicts the teammate's actions
  from the teammate's own observations, conditioned on an embedding computed
  from a **different trajectory of the same teammate**. The cross-trajectory
  conditioning is the point: it forces the embedding to carry policy identity
  rather than episode specifics.
* *discriminative* (Eq. 3) — InfoNCE with positives defined by **teammate policy
  label**. This is what makes ``teammate_id`` a required dataset field rather
  than a diagnostic one.

``L_emb = α·L_gen + λ·L_dis`` (Eq. 4).

**Stage 2, In-context Control Decoder.** The shared backbone with cross-attention,
taking the full token sequence ``z⁻¹`` as key/value. Trained to predict
**near-optimal** ego actions ``a^{1,*}`` — not merely the actions in the data.
See the module note in :mod:`oaht_bench.offline` about what that requires of the
dataset.

**Stage 3, deployment.** An Opponent Context Window holds the most recent ``C``
teammate trajectories; ``θ`` is frozen, so adaptation is entirely in-context.

Appendix F specifies the encoder as a GPT-2 *encoder*, 3 blocks of single-head
attention + feed-forward, with ELU modality layers and a fusion layer producing
one token per timestep.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from oaht_bench.dataset.windowing import (
    enumerate_windows,
    identity_matched_episode_ids,
    teammate_episode_ids,
    window_slice_batch,
)
from oaht_bench.models.tao_agent import OpponentPolicyEncoder, TaoAgent
from oaht_bench.offline.registry import BaseAhtTrainer
from oaht_bench.offline.utils import mask_logits, normalize_window_batch, safe_cross_entropy, to_jax

#: Stage 1's anchor trajectory -- what the encoder reads.
_ANCHOR_FIELDS = ("mate_next_obs", "mate_actions", "mate_rewards", "mask", "timesteps")
#: Stage 1's cross trajectory -- read by observation, scored by action
#: (the generative term's whole point: predict from a different episode).
_CROSS_FIELDS = ("mate_obs", "mate_actions", "mate_avail", "mask")
#: Stage 2's decoder window -- TAO's policy reads only the ego stream; its
#: opponent conditioning is entirely the cross-attended ``context_*`` fields.
_DECODER_FIELDS = ("ego_obs", "ego_actions", "ego_rtg", "ego_avail", "timesteps", "mask")
#: GetOffD's per-fragment fields -- what the encoder reads, same as the anchor.
_CONTEXT_FIELDS = _ANCHOR_FIELDS


def _random_t(batch, episode_id: int, rng: np.random.Generator) -> int:
    """A uniform random window end-position within ``episode_id`` -- the
    reference's own "random start offsets" (unlike the ego-history baselines'
    ``stride``-enumerated pool, there is no reproducibility argument for a
    grid here: these windows are keyed by *teammate*, not drawn from "any
    window in the dataset", so a flat stride pool does not apply)."""
    length = int(batch.episode_lengths()[episode_id])
    return int(rng.integers(1, length + 1))


def _anchor_and_cross_indices(dataset, teammate_id: int, rng: np.random.Generator):
    """Stage 1's contrastive pair (Eq. 2), as indices only: an anchor
    ``(episode_id, t)`` of ``teammate_id``'s play, and a ``(episode_id, t)``
    of the *same* teammate from a *different* episode -- the cross
    trajectory the generative term is scored against. Falls back to the
    anchor's own episode when the teammate appears in only one (common with
    1-4 episodes per teammate on these datasets).

    Index selection only, no disk I/O -- a caller collects many of these
    across a whole batch and reads them all in one
    :func:`~oaht_bench.dataset.windowing.window_slice_batch` call (see that
    function's own docstring for why reading one row at a time here would
    silently undo ``VaultReader.read_slices``'s concurrency).
    """
    episode_ids = teammate_episode_ids(dataset.batch, dataset.mate, teammate_id)
    anchor_episode_id = int(rng.choice(episode_ids))
    anchor_t = _random_t(dataset.batch, anchor_episode_id, rng)
    other_episode_ids = episode_ids[episode_ids != anchor_episode_id]
    pool = other_episode_ids if other_episode_ids.size else episode_ids
    cross_episode_id = int(rng.choice(pool))
    cross_t = _random_t(dataset.batch, cross_episode_id, rng)
    return anchor_episode_id, anchor_t, cross_episode_id, cross_t


def _get_off_d_context(dataset, teammate_ids, exclude_episode_ids, h: int, c: int, rng):
    """GetOffD (stage 2's cross-attention context), for a whole batch of
    decoder rows at once: ``c`` fragments of the *same* teammate per row,
    each a full ``h``-length window from a *different* episode than that
    row's decoder window, concatenated along time into one context of
    length ``c * h`` per row. Unlike D_epi (``offline/omis.py``), TAO has no
    independent fragment-length hyperparameter -- Appendix F's context is
    ``C`` whole windows, not sub-windows of some other size.

    ``teammate_ids``/``exclude_episode_ids`` are arrays, one entry per
    decoder row. Picking *which* episodes/positions is cheap index
    bookkeeping and stays a Python loop; the actual reads are ONE
    ``window_slice_batch`` call across every row's every fragment, not one
    call per row or per fragment.
    """
    episode_ids_flat, ts_flat = [], []
    for teammate_id, exclude_episode_id in zip(teammate_ids, exclude_episode_ids, strict=True):
        episode_ids = teammate_episode_ids(dataset.batch, dataset.mate, int(teammate_id))
        other_episode_ids = episode_ids[episode_ids != exclude_episode_id]
        pool = other_episode_ids if other_episode_ids.size else episode_ids
        chosen = rng.choice(pool, size=c, replace=pool.size < c)
        episode_ids_flat.extend(int(e) for e in chosen)
        ts_flat.extend(_random_t(dataset.batch, int(e), rng) for e in chosen)
    flat = window_slice_batch(
        dataset.batch, episode_ids_flat, ts_flat, h, _CONTEXT_FIELDS, mate=dataset.mate
    )
    n = len(teammate_ids)
    return {k: v.reshape(n, c * h, *v.shape[2:]) for k, v in flat.items()}


def _masked_accuracy(logits, labels, mask) -> jnp.ndarray:
    """Top-1 accuracy over valid timesteps.

    Reported alongside every cross-entropy term because a loss is not
    interpretable across baselines or datasets — 0.69 nats means one thing with
    two actions and another with six — while "fraction of actions predicted
    correctly" is comparable and has an obvious floor at chance.
    """
    correct = (jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)
    m = mask.astype(jnp.float32)
    return (correct * m).sum() / jnp.maximum(m.sum(), 1.0)


def supervised_contrastive(
    embeddings, labels, *, temperature: float = 0.1, base_temperature: float = 0.1
):
    """Eq. 3, as the reference implements it (``nn_trainer.py:130-156``).

    This is SupCon (Khosla et al. 2020), not plain InfoNCE, and the difference is
    the aggregation: the loss is the **mean over positives of the log-probability**,
    not the log-sum-exp over them. With several positives per anchor the two have
    different gradients — SupCon pulls every positive in equally, InfoNCE is
    dominated by the nearest.

    Two further details taken from the reference: similarities are on **raw dot
    products, not normalised embeddings**, and the max is subtracted per row for
    numerical stability before the log-sum-exp.

    Rows with no positive are dropped rather than dividing by zero. The reference
    divides by ``dis_mask.sum(1)`` unguarded because its sampler guarantees a
    positive per anchor; ours cannot, since seats are sampled independently and
    per-teammate coverage is ragged.
    """
    # Reference matmuls the pooled hidden states directly -- no L2 normalisation.
    sim = (embeddings @ embeddings.T) / temperature
    n = sim.shape[0]
    eye = jnp.eye(n, dtype=bool)
    positive = (labels[:, None] == labels[None, :]) & ~eye

    sim = sim - jax.lax.stop_gradient(sim.max(axis=1, keepdims=True))
    exp_sim = jnp.exp(sim) * (~eye)
    log_prob = sim - jnp.log(jnp.maximum(exp_sim.sum(axis=1, keepdims=True), 1e-12))

    n_pos = positive.sum(axis=1)
    mean_log_prob_pos = (positive * log_prob).sum(axis=1) / jnp.maximum(n_pos, 1)
    per_row = -(temperature / base_temperature) * mean_log_prob_pos
    has_positive = n_pos > 0
    return jnp.where(has_positive, per_row, 0.0).sum() / jnp.maximum(has_positive.sum(), 1)


def embedding_loss(
    params, encoder, decoder, batch, *, alpha=1.0, lam=1.0, rngs=None, train: bool = True
):
    """Stage 1: ``L_emb = α·L_gen + λ·L_dis`` (Eq. 4).

    One encoder pass, on the *anchor* trajectory. The pooled embedding ``z̄`` is
    then used for both terms: the generative decoder predicts a **different**
    trajectory's actions from that different trajectory's observations while
    conditioned on ``z̄``, and the discriminative term contrasts ``z̄`` against
    other anchors by teammate label (``offline_stage_1/nn_trainer.py:100-133``).

    The direction matters. An earlier version encoded the cross trajectory and
    decoded the anchor, which is a different task and cost a second encoder pass;
    the reference conditions on the anchor and is scored on the *other* episode,
    which is what forces the embedding to describe the policy rather than the
    episode it was computed from.
    """
    enc_params, dec_params = params["encoder"], params["decoder"]
    tokens = encoder.apply(
        enc_params,
        batch["mate_next_obs"],
        batch["mate_actions"],
        batch["mate_rewards"],
        mask=batch["mask"],
        timesteps=batch["timesteps"],
        train=train,
        rngs=rngs,
    )
    # Unmasked mean, as the reference pools (see OpponentPolicyEncoder.pool).
    z_bar = OpponentPolicyEncoder.pool(tokens)

    # Generative: a different trajectory of the same teammate, scored under z_bar.
    logits = mask_logits(
        decoder.apply(dec_params, batch["cross_mate_obs"], z_bar),
        batch["cross_mate_avail"],
    )
    cross_mask = batch["cross_mask"].astype(jnp.float32)
    gen = safe_cross_entropy(logits, batch["cross_mate_actions"], cross_mask)
    gen = (gen * cross_mask).sum() / jnp.maximum(cross_mask.sum(), 1.0)

    # Discriminative: the same z_bar, contrasted by teammate label.
    dis = supervised_contrastive(z_bar, batch["teammate_id"])

    # Whether the embedding carries enough about the teammate to predict what a
    # *different* episode of that teammate did.
    gen_acc = _masked_accuracy(logits, batch["cross_mate_actions"], cross_mask)

    total = alpha * gen + lam * dis
    return total, {
        "loss": total,
        "generative": gen,
        "discriminative": dis,
        "generative_accuracy": gen_acc,
    }


def tao_policy_loss(
    params, policy, encoder, batch, *, freeze_encoder: bool = True, rngs=None, train: bool = True
):
    """Stage 2: predict the ego action, cross-attending to the policy embedding.

    Cross-entropy over valid timesteps only. The reference flattens and indexes
    by the mask before calling ``CrossEntropyLoss``, whose default reduction is a
    mean, so this is a masked mean rather than a masked sum
    (``offline_stage_2/nn_trainer.py:83-87``). Its ``CrossEntropy`` helper takes
    ``argmax`` of the one-hot label and calls ``CrossEntropyLoss``, which is
    softmax cross-entropy with integer labels.

    **The encoder is frozen.** ``freeze_encoder`` defaults to True, which follows
    the paper and *not* the released code. Stage 2 there constructs a fresh
    ``GPTEncoder`` (``offline_stage_2/train.py:98``), never loads stage 1's
    weights, and steps ``encoder_optimizer`` alongside ``decoder_optimizer``
    (``nn_trainer.py:89-96``); deployment then loads the encoder stage 2
    produced. On that path stage 1 trains an encoder nothing ever reads.

    That cannot be what produced the paper, because **TAO w/o PEL is one of its
    own baselines** (§4.1) — PEL is stage 1. If stage 2 discarded stage 1, TAO
    and TAO w/o PEL would be the same model and the ablation would report the
    same number. ``offline_stage_2/config.py:59-61`` defines an
    ``ENCODER_PARAM_PATH`` pointing at stage 1's checkpoint which is read
    nowhere, so the load appears to have been dropped from the release.

    Freezing rather than fine-tuning follows from what stage 1 is for. Its
    generative term conditions on a *different trajectory of the same teammate*
    and its discriminative term separates teammates by identity; together they
    make the embedding carry policy identity rather than episode detail. Stage 3
    then places an *unseen* teammate in that space. Continuing to train the
    encoder on ego-action prediction would pull it toward whatever helps predict
    this batch's actions, eroding the structure the method depends on — and the
    w/o-PEL ablation only means anything if stage 1's product survives.

    ``freeze_encoder=False`` reproduces the released code's behaviour, for
    anyone wanting to check how much it matters.

    Args:
        params: ``{"policy": ..., "encoder": ...}``, the encoder's coming from
            stage 1. It stays in the gradient tree either way; ``freeze_encoder``
            stops the gradient at its output rather than removing the entry, so
            one call site serves both and the choice stays visible.
    """
    # GetOffD: C fragments of the same teammate, concatenated, sampled
    # independently of the decoder window (see _get_off_d_context). The
    # context is therefore C*T long while the decoder window is T -- which is
    # fine, because it enters as cross-attention keys rather than being
    # concatenated to anything.
    tokens = encoder.apply(
        params["encoder"],
        batch["context_mate_next_obs"],
        batch["context_mate_actions"],
        batch["context_mate_rewards"],
        mask=batch["context_mask"],
        timesteps=batch["context_timesteps"],
        train=train and not freeze_encoder,
        rngs=rngs,
    )
    if freeze_encoder:
        tokens = jax.lax.stop_gradient(tokens)

    logits = mask_logits(
        policy.apply(
            params["policy"],
            batch["ego_rtg"],
            batch["ego_obs"],
            batch["ego_actions"],
            timesteps=batch["timesteps"],
            context=tokens,
            mask=batch["mask"],
            context_mask=batch["context_mask"],
            train=train,
            rngs=rngs,
        ),
        batch["ego_avail"],
    )
    mask = batch["mask"].astype(jnp.float32)
    bc = safe_cross_entropy(logits, batch["ego_actions"], mask)
    acc = _masked_accuracy(logits, batch["ego_actions"], mask)
    bc = (bc * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    return bc, {"loss": bc, "bc": bc, "action_accuracy": acc}


class TaoTrainer(BaseAhtTrainer):
    """TAO on the two-stage contract.

    Unlike the ego-history baselines, TAO's encoder reads the *teammate* stream
    and its stages use structured batches: stage 1 draws contrastive batches
    (positives per teammate) via :func:`_anchor_and_cross_indices`, stage 2 draws windows
    with a GetOffD context via :func:`_get_off_d_context`. Stage-2 parameters
    carry the encoder (from stage 1, frozen or fine-tuned per ``freeze_encoder``)
    alongside the policy.

    ``act`` needs the deployment context, which the runner cannot supply at
    evaluation time (the policy is rebuilt without ``prepare``). So stage 2 seeds
    the offline ``C``-trajectory context from the training windows and stores it in
    the returned parameters -- the offline ``C = all`` case -- keeping ``act``'s
    signature the same as every other baseline. ``alpha``/``lam``/``freeze_encoder``
    and the sampler sizes are TAO-specific and read from the top-level config.
    """

    name = "tao"

    def build_model(self) -> None:
        # Inference is the composed TaoAgent's; training reads its
        # encoder/decoder/network.
        self.agent = TaoAgent(self.config)
        self.agent.build_model()

    def prepare(self, dataset, logger, *, rng, np_rng, num_seeds: int = 1) -> None:
        super().prepare(dataset, logger, rng=rng, np_rng=np_rng, num_seeds=num_seeds)
        # Stage 2's decoder window (_stage2_batch, Eq. 5) needs a near-optimal
        # ego action target, same reasoning as OMIS's D_step: restrict to
        # episodes where the ego seat IS the mate's own dedicated best
        # response (roster-index identity equality, not a quality/epsilon
        # threshold). A mismatched pairing's ego actions are not that
        # teammate's best response and would pollute the BC target.
        #
        # _deployment_context_one also reads this pool (for the offline C=all
        # Opponent Context Window) -- that use is about encoding TEAMMATE
        # behaviour, closer in spirit to GetOffD/D_epi, which wants diversity
        # across ANY ego, not matched-only. Restricting it here is a side
        # effect of sharing one pool with stage 2, not a requirement; flagged,
        # not resolved -- see that method's own docstring.
        matched_episodes = identity_matched_episode_ids(dataset.batch, dataset.ego, dataset.mate)
        episode_ids, ts = enumerate_windows(
            dataset.batch, h=self.config.context_length, stride=self.config.stride
        )
        keep = np.isin(episode_ids, matched_episodes)
        self._pool = (episode_ids[keep], ts[keep])

    def _stage1_batch(self, _step):
        # Factor the single ``batch_size`` into TAO's teammate-first contrastive
        # batch: use as many teammates as possible (more contrastive negatives),
        # filling with ``batch_size / n_teammates`` windows each -- floored at the 2
        # positives the contrastive term requires and CAPPED at 8. The cap matters
        # on a small roster: contrastive learning saturates on positives, so without
        # it a large batch_size balloons stage 1 (e.g. 12 teammates x 21 windows =
        # 252) for no benefit. With it, stage 1 is min(batch_size, n_teammates * 8).
        h = self.config.context_length
        teammate_ids = np.unique(self.dataset.batch.member_ids[:, self.dataset.mate])
        tpb = max(1, min(len(teammate_ids), self.config.batch_size // 2))
        wpt = min(max(2, round(self.config.batch_size / tpb)), 8)
        chosen = self.np_rng.choice(teammate_ids, size=tpb, replace=False)

        anchor_episode_ids, anchor_ts, cross_episode_ids, cross_ts = [], [], [], []
        teammate_col = []
        for teammate_id in chosen:
            for _ in range(wpt):
                ae, at, ce, ct = _anchor_and_cross_indices(
                    self.dataset, int(teammate_id), self.np_rng
                )
                anchor_episode_ids.append(ae)
                anchor_ts.append(at)
                cross_episode_ids.append(ce)
                cross_ts.append(ct)
                teammate_col.append(int(teammate_id))

        anchors = window_slice_batch(
            self.dataset.batch,
            anchor_episode_ids,
            anchor_ts,
            h,
            _ANCHOR_FIELDS,
            mate=self.dataset.mate,
        )
        crosses = window_slice_batch(
            self.dataset.batch,
            cross_episode_ids,
            cross_ts,
            h,
            _CROSS_FIELDS,
            mate=self.dataset.mate,
        )
        batch = dict(anchors)
        batch.update({f"cross_{k}": v for k, v in crosses.items()})
        batch["teammate_id"] = np.asarray(teammate_col, dtype=np.int32)
        return to_jax(normalize_window_batch(batch, self.dataset.norm))

    def _stage2_batch(self, _step):
        h = self.config.context_length
        c = self.config.context_trajectories
        episode_ids, ts = self._pool
        batch_size = self.config.batch_size
        rows = self.np_rng.choice(
            len(episode_ids), size=batch_size, replace=len(episode_ids) < batch_size
        )
        row_episode_ids, row_ts = episode_ids[rows], ts[rows]

        decoder = window_slice_batch(
            self.dataset.batch, row_episode_ids, row_ts, h, _DECODER_FIELDS, mate=self.dataset.mate
        )
        teammate_ids = self.dataset.batch.member_ids[row_episode_ids, self.dataset.mate]
        contexts = _get_off_d_context(
            self.dataset, teammate_ids, row_episode_ids, h, c, self.np_rng
        )

        batch = dict(decoder)
        batch["teammate_id"] = np.asarray(teammate_ids, dtype=np.int32)
        for key in _CONTEXT_FIELDS:
            batch[f"context_{key}"] = contexts[key]
        return to_jax(normalize_window_batch(batch, self.dataset.norm))

    def train_stage_1(self):
        init_batch = self._stage1_batch(0)
        self.rng, k = jax.random.split(self.rng)

        def init_one(key):
            k1, k2 = jax.random.split(key)
            encoder_params = self.agent.encoder.init(
                k1,
                init_batch["mate_next_obs"],
                init_batch["mate_actions"],
                init_batch["mate_rewards"],
                mask=init_batch["mask"],
                timesteps=init_batch["timesteps"],
            )
            init_tokens = self.agent.encoder.apply(
                encoder_params,
                init_batch["mate_next_obs"],
                init_batch["mate_actions"],
                init_batch["mate_rewards"],
                mask=init_batch["mask"],
                timesteps=init_batch["timesteps"],
            )
            decoder_params = self.agent.decoder.init(
                k2, init_batch["cross_mate_obs"], OpponentPolicyEncoder.pool(init_tokens)
            )
            return {"encoder": encoder_params, "decoder": decoder_params}

        params = self._init_params(init_one, k)

        def loss(p, b, rngs, frozen):
            return embedding_loss(
                p,
                self.agent.encoder,
                self.agent.decoder,
                b,
                alpha=self.config.alpha,
                lam=self.config.lam,
                rngs=rngs,
            )

        return self._run_stage(
            loss,
            params,
            self._stage1_batch,
            learning_rate=self.config.stage1_learning_rate,
            steps=self.config.stage1_steps,
            prefix="Stage1",
        )

    def train_stage_2(self, stage1_params):
        init_batch = self._stage2_batch(0)
        self.rng, k = jax.random.split(self.rng)
        enc_for_shape = (
            jax.tree.map(lambda x: x[0], stage1_params["encoder"])
            if self.num_seeds > 1
            else stage1_params["encoder"]
        )
        init_context = self.agent.encoder.apply(
            enc_for_shape,
            init_batch["context_mate_next_obs"],
            init_batch["context_mate_actions"],
            init_batch["context_mate_rewards"],
            mask=init_batch["context_mask"],
            timesteps=init_batch["context_timesteps"],
        )

        def init_one(key):
            return self.agent.network.init(
                key,
                init_batch["ego_rtg"],
                init_batch["ego_obs"],
                init_batch["ego_actions"],
                timesteps=init_batch["timesteps"],
                context=init_context,
                mask=init_batch["mask"],
                context_mask=init_batch["context_mask"],
            )

        policy_params = self._init_params(init_one, k)
        # The encoder rides in the parameter tree so freeze_encoder can stop the
        # gradient at its output without removing it (see tao_policy_loss). It keeps
        # the stage-1 seed axis, so under multi-seed each seed carries its own encoder.
        params = {"policy": policy_params, "encoder": stage1_params["encoder"]}

        def loss(p, b, rngs, frozen):
            return tao_policy_loss(
                p,
                self.agent.network,
                self.agent.encoder,
                b,
                freeze_encoder=self.config.freeze_encoder,
                rngs=rngs,
            )

        trained = self._run_stage(
            loss,
            params,
            self._stage2_batch,
            learning_rate=self.config.stage2_learning_rate,
            steps=self.config.stage2_steps,
            prefix="Stage2",
        )
        # Seed the deployment context (offline C=all) from the training windows so
        # act is self-contained at evaluation, where the policy is rebuilt without
        # prepare(). Uses the final encoder, which freeze_encoder may have tuned.
        context, context_mask = self._deployment_context(trained["encoder"])
        return {**trained, "context": context, "context_mask": context_mask}

    def _deployment_context(self, encoder_params):
        """Deployment context, seed-aware: with a stage-1 seed axis on
        ``encoder_params`` this vmaps to a per-seed context (and tiles the mask to
        the same axis), so the whole parameter tree stays uniformly seed-indexed and
        evaluation can slice one seed at a time."""
        if getattr(self, "num_seeds", 1) > 1:
            return jax.vmap(self._deployment_context_one)(encoder_params)
        return self._deployment_context_one(encoder_params)

    def _deployment_context_one(self, encoder_params):
        """The C-trajectory Opponent Context Window, encoded from the dataset.

        Reads and encodes ONLY the first ``c`` windows of the training pool
        (mirroring the old ``tokens[:c]``) -- the encoder embeds each
        trajectory independently, so this gives byte-identical tokens to
        encoding the whole pool and slicing afterward, without materializing
        it.

        Reads from the same ``self._pool`` that ``TaoTrainer.prepare`` now
        restricts to identity-matched episodes (stage 2's own near-optimal-
        target requirement) -- a side effect for this method, not a
        requirement of it: this context is about encoding TEAMMATE behaviour
        (closer in spirit to GetOffD/D_epi), which would prefer the same
        diversity-across-any-ego D_epi gets. Not resolved here; flagged as
        acceptable because this method only ever reads the first ``c``
        windows of the pool, a small, likely-inconsequential slice.
        """
        c = self.config.context_trajectories
        hidden = self.config.network.hidden_dim
        h = self.config.context_length
        episode_ids, ts = self._pool
        n = min(c, len(episode_ids))
        batch = normalize_window_batch(
            window_slice_batch(
                self.dataset.batch,
                episode_ids[:n],
                ts[:n],
                h,
                _CONTEXT_FIELDS,
                mate=self.dataset.mate,
            ),
            self.dataset.norm,
        )
        tokens = self.agent.encoder.apply(
            encoder_params,
            jnp.asarray(batch["mate_next_obs"]),
            jnp.asarray(batch["mate_actions"]),
            jnp.asarray(batch["mate_rewards"]),
            mask=jnp.asarray(batch["mask"]),
            timesteps=jnp.asarray(batch["timesteps"]),
            train=False,
        )
        context = tokens.reshape(1, -1, hidden)
        context_mask = jnp.asarray(batch["mask"]).reshape(1, -1)
        return context, context_mask
