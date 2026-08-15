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

    @nn.compact
    def __call__(self, mate_obs, mate_actions, mate_rewards, *, mask, train: bool = False):
        # Modality-specific linear layers with ELU, 32 nodes (Appendix F).
        a = nn.elu(nn.Dense(HIDDEN)(jax.nn.one_hot(mate_actions, self.action_dim)))
        r = nn.elu(nn.Dense(HIDDEN)(mate_rewards[..., None]))
        o = nn.elu(nn.Dense(HIDDEN)(mate_obs))

        # (a_{t-1}, r_{t-1}, o_t) -> one fused token per timestep.
        prev = lambda x: jnp.concatenate([jnp.zeros_like(x[:, :1]), x[:, :-1]], axis=1)  # noqa: E731
        fused = nn.Dense(HIDDEN)(jnp.concatenate([prev(a), prev(r), o], axis=-1))

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
    def pool(tokens, mask):
        """``AP``: average over real timesteps, giving ``z̄⁻¹``."""
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
        z = jnp.broadcast_to(embedding[:, None, :], (*mate_obs.shape[:2], embedding.shape[-1]))
        h = nn.Dense(HIDDEN)(jnp.concatenate([nn.Dense(HIDDEN)(mate_obs), z], axis=-1))
        return nn.Dense(self.action_dim)(h)


class TaoPolicy(nn.Module):
    """Stage 2: the shared backbone, cross-attending to the policy embedding."""

    action_dim: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, rtg, obs, actions, *, timesteps, context, train: bool = False):
        logits, _ = ControlDecoder(
            action_dim=self.action_dim,
            use_cross_attention=True,  # Appendix F: z^-1 enters as key/value.
            dropout=self.dropout,
        )(rtg, obs, actions, timesteps=timesteps, context=context, train=train)
        return logits


def info_nce(embeddings, labels, *, temperature: float = 0.1):
    """Eq. 3. Positives are pairs sharing a teammate label.

    Windows whose teammate appears only once in the batch have no positive and
    are dropped from the mean rather than contributing a degenerate term — with
    random seating, per-teammate coverage is ragged (1-4 episodes each on the
    LBF datasets), so this is the common case, not an edge case.
    """
    z = embeddings / jnp.maximum(jnp.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-8)
    sim = z @ z.T / temperature
    n = sim.shape[0]
    eye = jnp.eye(n, dtype=bool)
    positive = (labels[:, None] == labels[None, :]) & ~eye

    sim = jnp.where(eye, -jnp.inf, sim)
    log_denom = jax.nn.logsumexp(sim, axis=-1)
    log_num = jax.nn.logsumexp(jnp.where(positive, sim, -jnp.inf), axis=-1)
    has_positive = positive.any(axis=-1)
    per_row = jnp.where(has_positive, log_denom - log_num, 0.0)
    return per_row.sum() / jnp.maximum(has_positive.sum(), 1)


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
        enc_params, batch["cross_mate_obs"], batch["cross_mate_actions"],
        batch["cross_mate_rewards"], mask=batch["cross_mask"], train=train, rngs=rngs,
    )
    z_bar = OpponentPolicyEncoder.pool(tokens, batch["cross_mask"])

    logits = decoder.apply(dec_params, batch["mate_obs"], z_bar)
    mask = batch["mask"].astype(jnp.float32)
    gen = optax.softmax_cross_entropy_with_integer_labels(logits, batch["mate_actions"])
    gen = (gen * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    own_tokens = encoder.apply(
        enc_params, batch["mate_obs"], batch["mate_actions"], batch["mate_rewards"],
        mask=batch["mask"], train=train, rngs=rngs,
    )
    dis = info_nce(OpponentPolicyEncoder.pool(own_tokens, batch["mask"]), batch["teammate_id"])

    total = alpha * gen + lam * dis
    return total, {"loss": total, "generative": gen, "discriminative": dis}
