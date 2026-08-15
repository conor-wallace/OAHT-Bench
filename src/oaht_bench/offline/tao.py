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

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from oaht_bench.offline.backbone import FEEDFORWARD, HIDDEN, NUM_BLOCKS, ControlDecoder


class OpponentPolicyEncoder(nn.Module):
    """``M_θe``: the teammate's stream to a sequence of policy-embedding tokens.

    Returns the full token sequence. Stage 1 average-pools it into ``z̄⁻¹``;
    stages 2 and 3 feed the sequence itself as key/value into the decoder's
    cross-attention.
    """

    action_dim: int
    dropout: float = 0.1

    max_timesteps: int = 4096

    @nn.compact
    def __call__(self, mate_next_obs, mate_actions, mate_rewards, *, mask, timesteps,
                 train: bool = False):
        """``mate_next_obs`` is deliberate.

        The reference feeds ``traj['next_observations']`` alongside ``actions``
        and ``rewards`` at the *same* index (``offline_stage_1/utils.py:109-111``),
        which is how the paper's ``(a_{t-1}, r_{t-1}, o_t)`` fusion is realised —
        by choosing next-observations, not by shifting the action and reward
        streams. An index shift gets the same pairing but labels each token with
        a different timestep, and the timestep embedding below is not symmetric.
        """
        # Modality-specific linear layers with ELU, 32 nodes (Appendix F).
        a = nn.elu(nn.Dense(HIDDEN)(jax.nn.one_hot(mate_actions, self.action_dim)))
        r = nn.elu(nn.Dense(HIDDEN)(mate_rewards[..., None]))
        o = nn.elu(nn.Dense(HIDDEN)(mate_next_obs))

        # Reference net.py:48-55 -- obs and reward take the timestep embedding at
        # t, the action takes t-1, clamped at 0. The paper mentions no positional
        # encoding in the encoder at all.
        embed_t = nn.Embed(self.max_timesteps, HIDDEN)
        pos = embed_t(timesteps)
        pos_m1 = embed_t(jnp.where(timesteps > 0, timesteps - 1, timesteps))
        a, r, o = a + pos_m1, r + pos, o + pos

        # LayerNorm over the concatenated 3*hidden vector, then fuse to one token
        # per timestep (reference `embed_ln` is LayerNorm(3 * hidden_size)).
        fused = nn.Dense(HIDDEN)(nn.LayerNorm()(jnp.concatenate([a, r, o], axis=-1)))

        # Encoder blocks: no causal mask -- the whole teammate trajectory is
        # available when building a policy embedding.
        pad = nn.make_attention_mask(mask, mask)
        x = fused
        for _ in range(NUM_BLOCKS):
            attn = nn.SelfAttention(
                num_heads=1, qkv_features=HIDDEN, dropout_rate=self.dropout,
                deterministic=not train,
            )(x, mask=pad)
            x = nn.LayerNorm()(x + attn)
            ff = nn.Dense(HIDDEN)(nn.relu(nn.Dense(FEEDFORWARD)(x)))
            x = nn.LayerNorm()(x + ff)
        return x

    @staticmethod
    def pool(tokens, mask=None):
        """``AP``, giving ``z̄⁻¹``.

        The reference pools with a plain ``nn.AvgPool1d(kernel_size=NUM_STEPS)``
        (``nn_trainer.py:32,109``) — an unmasked mean over every position,
        padding included. That is what produced the published numbers, so it is
        the default here; ``mask`` is accepted for the masked variant but is not
        what TAO does.
        """
        if mask is None:
            return tokens.mean(axis=1)
        m = mask.astype(tokens.dtype)[..., None]
        return (tokens * m).sum(axis=1) / jnp.maximum(m.sum(axis=1), 1.0)


class AncillaryActionDecoder(nn.Module):
    """Predicts the teammate's actions from its own observations plus ``z̄⁻¹``.

    Stage 1's generative loss (Eq. 2). Deliberately weak — it exists to shape the
    embedding, not to be a good teammate model.
    """

    action_dim: int

    @nn.compact
    def __call__(self, mate_obs, embedding):
        # Reference MLPDecoder (offline_stage_1/net.py:116-133): the *raw*
        # observation is concatenated with the latent, then one hidden layer with
        # ReLU followed by LayerNorm. Embedding the observation first, or
        # dropping the activation, is a different function.
        z = jnp.broadcast_to(embedding[:, None, :], (*mate_obs.shape[:2], embedding.shape[-1]))
        h = nn.LayerNorm()(nn.relu(nn.Dense(HIDDEN)(jnp.concatenate([mate_obs, z], axis=-1))))
        return nn.Dense(self.action_dim)(h)


class TaoPolicy(nn.Module):
    """Stage 2: the shared backbone, cross-attending to the policy embedding."""

    action_dim: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, rtg, obs, actions, *, timesteps, context, mask=None,
                 context_mask=None, train: bool = False):
        logits, _ = ControlDecoder(
            action_dim=self.action_dim,
            use_cross_attention=True,  # Appendix F: z^-1 enters as key/value.
            dropout=self.dropout,
        )(rtg, obs, actions, timesteps=timesteps, mask=mask, context=context,
          context_mask=context_mask, train=train)
        return logits


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


def embedding_loss(params, encoder, decoder, batch, *, alpha=1.0, lam=1.0, rngs=None,
                   train: bool = True):
    """Stage 1: ``L_emb = α·L_gen + λ·L_dis`` (Eq. 4).

    ``batch`` must supply a *second* trajectory of the same teammate under the
    ``cross_*`` keys — the generative term conditions on an embedding from a
    different episode of the same policy, which is what stops the embedding from
    memorising episode specifics.
    """
    enc_params, dec_params = params["encoder"], params["decoder"]
    tokens = encoder.apply(
        enc_params, batch["cross_mate_next_obs"], batch["cross_mate_actions"],
        batch["cross_mate_rewards"], mask=batch["cross_mask"],
        timesteps=batch["cross_timesteps"], train=train, rngs=rngs,
    )
    # Unmasked mean, as the reference pools (see OpponentPolicyEncoder.pool).
    z_bar = OpponentPolicyEncoder.pool(tokens)

    logits = decoder.apply(dec_params, batch["mate_obs"], z_bar)
    mask = batch["mask"].astype(jnp.float32)
    gen = optax.softmax_cross_entropy_with_integer_labels(logits, batch["mate_actions"])
    gen = (gen * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    own_tokens = encoder.apply(
        enc_params, batch["mate_next_obs"], batch["mate_actions"], batch["mate_rewards"],
        mask=batch["mask"], timesteps=batch["timesteps"], train=train, rngs=rngs,
    )
    dis = supervised_contrastive(
        OpponentPolicyEncoder.pool(own_tokens), batch["teammate_id"]
    )

    total = alpha * gen + lam * dis
    return total, {"loss": total, "generative": gen, "discriminative": dis}
