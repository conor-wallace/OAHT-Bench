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
import optax

from oaht_bench.dataset.sampler import sample_stage1, sample_stage2
from oaht_bench.models.tao_agent import OpponentPolicyEncoder, TaoAgent
from oaht_bench.offline.registry import BaseAhtTrainer
from oaht_bench.offline.utils import mask_logits, to_jax


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
    gen = optax.softmax_cross_entropy_with_integer_labels(logits, batch["cross_mate_actions"])
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
    # independently of the decoder window (see oaht_bench.dataset.sampler). The
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
    bc = optax.softmax_cross_entropy_with_integer_labels(logits, batch["ego_actions"])
    acc = _masked_accuracy(logits, batch["ego_actions"], mask)
    bc = (bc * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    return bc, {"loss": bc, "bc": bc, "action_accuracy": acc}


class TaoTrainer(BaseAhtTrainer):
    """TAO on the two-stage contract.

    Unlike the ego-history baselines, TAO's encoder reads the *teammate* stream
    and its stages use structured batches: stage 1 draws contrastive batches
    (positives per teammate) via :func:`sample_stage1`, stage 2 draws windows with
    a GetOffD context via :func:`sample_stage2`. Stage-2 parameters carry the
    encoder (from stage 1, frozen or fine-tuned per ``freeze_encoder``) alongside
    the policy.

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

    def _stage1_batch(self, _step):
        return to_jax(
            sample_stage1(
                self.dataset.windows,
                self.dataset.index,
                self.np_rng,
                teammates_per_batch=self.config.teammates_per_batch,
                windows_per_teammate=self.config.windows_per_teammate,
            )
        )

    def _stage2_batch(self, _step):
        return to_jax(
            sample_stage2(
                self.dataset.windows,
                self.dataset.index,
                self.np_rng,
                batch_size=self.config.stage2_batch_size,
                context_trajectories=self.config.context_trajectories,
            )
        )

    def train_stage_1(self):
        init_batch = self._stage1_batch(0)
        self.rng, k1, k2 = jax.random.split(self.rng, 3)
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
        params = {"encoder": encoder_params, "decoder": decoder_params}

        def loss(p, b, rngs):
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
        init_context = self.agent.encoder.apply(
            stage1_params["encoder"],
            init_batch["context_mate_next_obs"],
            init_batch["context_mate_actions"],
            init_batch["context_mate_rewards"],
            mask=init_batch["context_mask"],
            timesteps=init_batch["context_timesteps"],
        )
        policy_params = self.agent.network.init(
            k,
            init_batch["ego_rtg"],
            init_batch["ego_obs"],
            init_batch["ego_actions"],
            timesteps=init_batch["timesteps"],
            context=init_context,
            mask=init_batch["mask"],
            context_mask=init_batch["context_mask"],
        )
        # The encoder rides in the parameter tree so freeze_encoder can stop the
        # gradient at its output without removing it (see tao_policy_loss).
        params = {"policy": policy_params, "encoder": stage1_params["encoder"]}

        def loss(p, b, rngs):
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
        """The C-trajectory Opponent Context Window, encoded from the dataset."""
        c = self.config.context_trajectories
        hidden = self.config.network.hidden_dim
        tokens = self.agent.encoder.apply(
            encoder_params,
            jnp.asarray(self.dataset.windows.mate_next_obs),
            jnp.asarray(self.dataset.windows.mate_actions),
            jnp.asarray(self.dataset.windows.mate_rewards),
            mask=jnp.asarray(self.dataset.windows.mask),
            timesteps=jnp.asarray(self.dataset.windows.timesteps),
            train=False,
        )
        context = tokens[:c].reshape(1, -1, hidden)
        context_mask = jnp.asarray(self.dataset.windows.mask)[:c].reshape(1, -1)
        return context, context_mask
