"""LIAM — Local Information Agent Modelling, adapted to the offline setting.

Follows the original method (Papoudakis et al.; the online port in
:mod:`oaht_bench.algorithms.liam_agent`) rather than TAO's Appendix F sketch,
which drops LIAM's encoder and hangs a reconstruction head off the policy trunk.

**What LIAM is.** An encoder summarises the ego agent's *local* history into an
embedding; a decoder reconstructs the *teammate's* observation and action from
that embedding; the policy is conditioned on the embedding. The teammate is
never observed by the policy — only modelled — which is the method's hypothesis
and why it is the natural floor for the trajectory-view family.

**Two stages, not one.** The original trains encoder and policy together and
blocks the gradient with ``stop_gradient`` (``liam_agent.py:536``) because
everything is learned online in a single loop. Offline that constraint is gone:
stage 1 trains encoder and decoder on reconstruction, stage 2 trains the policy
against a frozen encoder. This is TAO's protocol; it *removes* the need for the
gradient block rather than reproducing it, and it makes LIAM and TAO two choices
of encoder and conditioning mode over one training procedure.

**The encoder is the backbone, read at the right position.** LIAM's encoder sees
``o¹_{0..t}`` and ``a¹_{0..t-1}`` — observations through ``t``, actions only
through ``t-1``, because at ``t`` the ego has not acted yet. In the interleaved
``(G_t, o_t, a_t)`` sequence ``o_t`` sits at index ``3t+1`` and ``a_t`` at
``3t+2``, so under the causal mask the hidden state at ``o_t`` attends to
``o_{≤t}`` and ``a_{<t}`` and *not* ``a_t``. That is exactly LIAM's information
set, so no separate encoder architecture is required.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import optax

from oaht_bench.offline.backbone import DEFAULT_HIDDEN_DIM, DecisionTransformer


class LiamEncoder(nn.Module):
    """Ego-history encoder: the backbone, read at the ``o_t`` positions."""

    action_dim: int
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    dropout: float = 0.1

    @nn.compact
    def __call__(self, rtg, obs, actions, *, timesteps, mask=None, train: bool = False):
        _, obs_hidden = DecisionTransformer(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            use_cross_attention=False,
            dropout=self.dropout,
        )(rtg, obs, actions, timesteps=timesteps, mask=mask, train=train)
        return obs_hidden


class LiamDecoder(nn.Module):
    """Reconstructs the teammate's observation and action at time ``t``.

    Two independent heads, each two hidden layers with ReLU, matching
    ``liam_agent.py:119-160``. The action head returns logits; the reference
    applies softmax then takes ``-log(sum(p * onehot))``, which is the same
    quantity as softmax cross-entropy but less numerically stable.
    """

    obs_dim: int
    action_dim: int
    hidden_dim: int = DEFAULT_HIDDEN_DIM

    @nn.compact
    def __call__(self, embedding):
        h = nn.relu(nn.Dense(self.hidden_dim)(embedding))
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        mate_obs_hat = nn.Dense(self.obs_dim)(h)

        g = nn.relu(nn.Dense(self.hidden_dim)(embedding))
        g = nn.relu(nn.Dense(self.hidden_dim)(g))
        mate_action_logits = nn.Dense(self.action_dim)(g)
        return mate_obs_hat, mate_action_logits


class LiamPolicy(nn.Module):
    """Stage 2: the backbone, conditioned on a frozen teammate embedding.

    LIAM concatenates the embedding to the observation (``liam_agent.py:536``)
    rather than cross-attending; that is the conditioning mode which
    distinguishes it from TAO.
    """

    action_dim: int
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    dropout: float = 0.1

    @nn.compact
    def __call__(self, rtg, obs, actions, *, timesteps, embedding, mask=None,
                 train: bool = False):
        logits, _ = DecisionTransformer(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            use_cross_attention=False,
            dropout=self.dropout,
        )(
            rtg,
            jnp.concatenate([obs, embedding], axis=-1),
            actions,
            timesteps=timesteps,
            mask=mask,
            train=train,
        )
        return logits


def liam_reconstruction_loss(params, encoder, decoder, batch, *, rngs=None,
                             train: bool = True):
    """Stage 1: reconstruct the teammate at time ``t`` from the ego's history.

    Both targets are at time ``t``, per the paper and ``liam_agent.py:302-303``.
    An earlier version used ``a⁻¹_{t-1}`` for the action, following TAO's
    Appendix F, which is a different task: it asks what the teammate did *last*
    step rather than what it is doing *now*.

    The observation term is ``0.5 * sum((y - ŷ)²)`` over the observation
    dimension, not a mean. That is the unit-variance Gaussian negative
    log-likelihood, which places it on the same footing as the categorical
    negative log-likelihood on actions. A mean rescales it by ``0.5 * obs_dim``
    — 12× on LBF — silently reweighting the two terms against each other.
    """
    z = encoder.apply(
        params["encoder"], batch["ego_rtg"], batch["ego_obs"], batch["ego_actions"],
        timesteps=batch["timesteps"], mask=batch["mask"], train=train, rngs=rngs,
    )
    mate_obs_hat, mate_act_logits = decoder.apply(params["decoder"], z)

    mask = batch["mask"].astype(jnp.float32)
    denom = jnp.maximum(mask.sum(), 1.0)

    recon_obs = 0.5 * ((batch["mate_obs"] - mate_obs_hat) ** 2).sum(axis=-1)
    recon_obs = (recon_obs * mask).sum() / denom

    recon_act = optax.softmax_cross_entropy_with_integer_labels(
        mate_act_logits, batch["mate_actions"]
    )
    recon_act = (recon_act * mask).sum() / denom

    total = recon_obs + recon_act
    return total, {"loss": total, "recon_obs": recon_obs, "recon_action": recon_act}


def liam_policy_loss(params, policy, encoder, encoder_params, batch, *, rngs=None,
                     train: bool = True):
    """Stage 2: behaviour cloning, conditioned on the frozen encoder.

    ``encoder_params`` come from stage 1 and are never differentiated, which is
    what makes the original's ``stop_gradient`` unnecessary rather than merely
    omitted.
    """
    z = encoder.apply(
        encoder_params, batch["ego_rtg"], batch["ego_obs"], batch["ego_actions"],
        timesteps=batch["timesteps"], mask=batch["mask"], train=False,
    )
    logits = policy.apply(
        params, batch["ego_rtg"], batch["ego_obs"], batch["ego_actions"],
        timesteps=batch["timesteps"], embedding=z, mask=batch["mask"],
        train=train, rngs=rngs,
    )
    mask = batch["mask"].astype(jnp.float32)
    bc = optax.softmax_cross_entropy_with_integer_labels(logits, batch["ego_actions"])
    bc = (bc * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    return bc, {"loss": bc, "bc": bc}
