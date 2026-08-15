"""LIAM-offline, exactly as TAO's Appendix F specifies it.

> Backbone identical to ICD **minus the cross-attention layer**. Feed
> ``G¹_t, o¹_t, a¹_t``, predict ``a¹_t`` autoregressively under a causal mask.
> Add an **extra decoder** for the auxiliary task: reconstruct the opponent's
> observations ``o⁻¹_t`` and actions ``a⁻¹_{t-1}`` **from the ``o¹_t`` token
> embeddings** produced by the backbone. Extra decoder = 2 linear layers, 32
> nodes, no activation.

TAGET also converts LIAM, but its specification is a single sentence ("we train
the reconstruction loss on offline data"); TAO's is complete and unambiguous, so
this follows TAO.

Note what LIAM does *not* get: no conditioning path from the teammate into the
policy. The teammate only ever appears as a reconstruction *target*, so the
representation has to be forced into the ego embeddings by the auxiliary loss.
That is the whole hypothesis of the method, and it is why LIAM is the natural
floor for the trajectory-view family.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import optax

from oaht_bench.offline.backbone import HIDDEN, ControlDecoder


class LiamOffline(nn.Module):
    """Return-conditioned decoder with a teammate-reconstruction head."""

    action_dim: int
    obs_dim: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, rtg, obs, actions, *, timesteps, train: bool = False):
        logits, obs_hidden = ControlDecoder(
            action_dim=self.action_dim,
            use_cross_attention=False,  # Appendix F: ICD minus cross-attention.
            dropout=self.dropout,
        )(rtg, obs, actions, timesteps=timesteps, train=train)

        # Extra decoder: 2 linear layers, 32 nodes, no activation. One head per
        # reconstruction target, both read from the o_t embeddings.
        h = nn.Dense(HIDDEN)(obs_hidden)
        h = nn.Dense(HIDDEN)(h)
        mate_obs_hat = nn.Dense(self.obs_dim)(h)
        mate_action_logits = nn.Dense(self.action_dim)(h)
        return logits, mate_obs_hat, mate_action_logits


def liam_loss(
    params,
    model: LiamOffline,
    batch,
    *,
    rngs=None,
    reconstruction_weight: float = 1.0,
    train: bool = True,
):
    """Behaviour cloning on the ego, plus reconstruction of the teammate.

    The reconstruction target for actions is ``a⁻¹_{t-1}`` — the teammate's
    *previous* action — because Appendix F notes the ``o¹_t`` embedding already
    contains ``a¹_{t-1}``, so the aligned teammate quantity is one step back.
    Predicting ``a⁻¹_t`` instead would ask the model to see the teammate's
    simultaneous move, which it cannot observe at deployment.

    ``batch`` is a dict of arrays shaped like :class:`~oaht_bench.offline.dataset.Windows`.
    """
    logits, mate_obs_hat, mate_act_logits = model.apply(
        params,
        batch["ego_rtg"],
        batch["ego_obs"],
        batch["ego_actions"],
        timesteps=batch["timesteps"],
        train=train,
        rngs=rngs,
    )
    mask = batch["mask"].astype(jnp.float32)
    denom = jnp.maximum(mask.sum(), 1.0)

    bc = optax.softmax_cross_entropy_with_integer_labels(logits, batch["ego_actions"])
    bc = (bc * mask).sum() / denom

    obs_mse = ((mate_obs_hat - batch["mate_obs"]) ** 2).mean(axis=-1)
    obs_mse = (obs_mse * mask).sum() / denom

    # a^-1_{t-1}: shift the teammate's actions forward, dropping t=0 from the mask.
    prev_mate_actions = jnp.concatenate(
        [batch["mate_actions"][:, :1], batch["mate_actions"][:, :-1]], axis=1
    )
    shift_mask = mask.at[:, 0].set(0.0)
    act_ce = optax.softmax_cross_entropy_with_integer_labels(
        mate_act_logits, prev_mate_actions
    )
    act_ce = (act_ce * shift_mask).sum() / jnp.maximum(shift_mask.sum(), 1.0)

    total = bc + reconstruction_weight * (obs_mse + act_ce)
    return total, {
        "loss": total,
        "bc": bc,
        "recon_obs": obs_mse,
        "recon_action": act_ce,
    }
