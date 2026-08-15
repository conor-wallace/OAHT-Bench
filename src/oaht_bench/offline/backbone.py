"""The shared sequence backbone every trajectory-view baseline is built on (§3.1).

This is TAO's In-context Control Decoder, specified in its Appendix F, which is
the published precedent for §3.1's design: one architecture, with each baseline
stating exactly what it adds. Reproducing that structure rather than authoring
our own means differences between baselines are differences the papers describe,
not differences we introduced.

From Appendix F, verbatim in effect:

* GPT-2 decoder, **3 self-attention blocks**; each block is single-head
  self-attention + single-head cross-attention + feed-forward, residual
  connections and LayerNorm after each layer, dropout on both the residual and
  the attention weights.
* ``G_t, o_t, a_t`` pass through **modality-specific linear layers** plus an
  *episodic timestep* positional encoding.
* Actions are predicted autoregressively under a causal mask, from the hidden
  states **at the ``o_t`` token positions**.

Three details come from the authors' reference implementation rather than the
paper, which does not mention them:

* a **LayerNorm on the stacked token sequence** before the blocks
  (``embed_ln``, ``offline_stage_2/net.py:77``);
* the causal mask is **combined with the padding mask**, so real tokens never
  attend to padding — the paper says only "causal mask";
* sequences are **left-padded**, the Decision Transformer convention the
  reference inherits, so the most recent timestep is always last.
* Feed-forward **128 nodes, ReLU**; every other hidden layer **32 nodes, no
  activation**; modality-specific layers **32 nodes, no activation**.

``use_cross_attention`` is the one switch: TAO keeps it and feeds the opponent
embedding in as key/value; LIAM and MeLIBA drop it and attach an auxiliary head
instead. That is precisely how Appendix F describes them, so it is the only
axis this module exposes.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

#: Appendix F: all hidden layers except the feed-forward are 32 nodes.
HIDDEN = 32
#: Appendix F: the feed-forward layer is 128 nodes with ReLU.
FEEDFORWARD = 128
#: Appendix F: 3 self-attention blocks.
NUM_BLOCKS = 3


class Block(nn.Module):
    """One decoder block: self-attention, optional cross-attention, feed-forward."""

    use_cross_attention: bool
    dropout: float

    @nn.compact
    def __call__(self, x, *, context=None, causal_mask, cross_mask=None, train: bool):
        # Single-head throughout, per Appendix F.
        attn = nn.SelfAttention(
            num_heads=1, qkv_features=HIDDEN, dropout_rate=self.dropout,
            deterministic=not train,
        )(x, mask=causal_mask)
        x = nn.LayerNorm()(x + nn.Dropout(self.dropout, deterministic=not train)(attn))

        if self.use_cross_attention:
            if context is None:
                raise ValueError(
                    "use_cross_attention is set but no context was passed. TAO "
                    "feeds the opponent embedding in as key/value here; LIAM and "
                    "MeLIBA should construct the backbone with it disabled."
                )
            cross = nn.MultiHeadDotProductAttention(
                num_heads=1, qkv_features=HIDDEN, dropout_rate=self.dropout,
                deterministic=not train,
            )(x, context, mask=cross_mask)
            x = nn.LayerNorm()(x + nn.Dropout(self.dropout, deterministic=not train)(cross))

        ff = nn.Dense(FEEDFORWARD)(x)
        ff = nn.relu(ff)
        ff = nn.Dense(HIDDEN)(ff)
        return nn.LayerNorm()(x + nn.Dropout(self.dropout, deterministic=not train)(ff))


class ControlDecoder(nn.Module):
    """Return-conditioned causal decoder over ``(G_t, o_t, a_t)`` triples.

    Returns both the action logits and the hidden states at the ``o_t``
    positions, because LIAM's auxiliary decoder reconstructs the teammate from
    exactly those embeddings — Appendix F is specific that they already contain
    ``o_t`` and ``a_{t-1}``.
    """

    action_dim: int
    use_cross_attention: bool = False
    dropout: float = 0.1
    max_timesteps: int = 4096

    @nn.compact
    def __call__(self, rtg, obs, actions, *, timesteps, mask=None, context=None,
                 context_mask=None, train: bool = False):
        B, T = obs.shape[0], obs.shape[1]

        # Modality-specific linear layers, 32 nodes, no activation.
        g_tok = nn.Dense(HIDDEN)(rtg[..., None])
        o_tok = nn.Dense(HIDDEN)(obs)
        a_tok = nn.Dense(HIDDEN)(jax.nn.one_hot(actions, self.action_dim))

        # Episodic timestep encoding, added to every modality (Chen et al. 2021).
        pos = nn.Embed(self.max_timesteps, HIDDEN)(timesteps)
        g_tok, o_tok, a_tok = g_tok + pos, o_tok + pos, a_tok + pos

        # Interleave to (G_0, o_0, a_0, G_1, o_1, a_1, ...).
        x = jnp.stack([g_tok, o_tok, a_tok], axis=2).reshape(B, T * 3, HIDDEN)
        # Reference `embed_ln`: LayerNorm on the stacked sequence, not per token
        # stream. Absent from the paper.
        x = nn.LayerNorm()(x)

        attn_mask = nn.make_causal_mask(jnp.ones((B, T * 3)))
        if mask is not None:
            # Each timestep contributes three tokens, so the padding mask has to
            # be tripled to line up. Without this, real tokens attend to padding.
            stacked = jnp.repeat(mask.astype(bool), 3, axis=1)
            attn_mask = nn.combine_masks(attn_mask, nn.make_attention_mask(stacked, stacked))

        cross_mask = None
        if context is not None and context_mask is not None:
            q = jnp.ones((B, T * 3), dtype=bool)
            cross_mask = nn.make_attention_mask(q, context_mask.astype(bool))

        for _ in range(NUM_BLOCKS):
            x = Block(self.use_cross_attention, self.dropout)(
                x, context=context, causal_mask=attn_mask, cross_mask=cross_mask, train=train
            )

        # Actions are read off the o_t positions: index 1 of each triple.
        obs_hidden = x.reshape(B, T, 3, HIDDEN)[:, :, 1]
        logits = nn.Dense(self.action_dim)(obs_hidden)
        return logits, obs_hidden
