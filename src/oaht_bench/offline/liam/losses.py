import jax.numpy as jnp
import optax

from oaht_bench.offline.utils import mask_logits, masked_accuracy


def liam_reconstruction_loss(params, encoder, decoder, batch, *, rngs=None, train: bool = True):
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
        params["encoder"],
        batch["ego_rtg"],
        batch["ego_obs"],
        batch["ego_actions"],
        timesteps=batch["timesteps"],
        mask=batch["mask"],
        train=train,
        rngs=rngs,
    )
    mate_obs_hat, mate_act_logits = decoder.apply(params["decoder"], z)
    # The teammate could only have taken a legal action, so the reconstruction
    # target lives in the masked distribution.
    mate_act_logits = mask_logits(mate_act_logits, batch["mate_avail"])

    mask = batch["mask"].astype(jnp.float32)
    denom = jnp.maximum(mask.sum(), 1.0)

    recon_obs = 0.5 * ((batch["mate_obs"] - mate_obs_hat) ** 2).sum(axis=-1)
    recon_obs = (recon_obs * mask).sum() / denom

    recon_act = optax.softmax_cross_entropy_with_integer_labels(
        mate_act_logits, batch["mate_actions"]
    )
    recon_act = (recon_act * mask).sum() / denom

    # Teammate action-prediction accuracy: does the embedding actually let the
    # decoder say what the teammate is doing? That is the auxiliary task's
    # purpose, and the loss alone does not say whether it succeeded.
    recon_acc = masked_accuracy(mate_act_logits, batch["mate_actions"], mask)

    total = recon_obs + recon_act
    return total, {
        "loss": total,
        "recon_obs": recon_obs,
        "recon_action": recon_act,
        "recon_action_accuracy": recon_acc,
    }


def liam_policy_loss(
    params, policy, encoder, encoder_params, batch, *, rngs=None, train: bool = True
):
    """Stage 2: behaviour cloning, conditioned on the frozen encoder.

    ``encoder_params`` come from stage 1 and are never differentiated, which is
    what makes the original's ``stop_gradient`` unnecessary rather than merely
    omitted.
    """
    z = encoder.apply(
        encoder_params,
        batch["ego_rtg"],
        batch["ego_obs"],
        batch["ego_actions"],
        timesteps=batch["timesteps"],
        mask=batch["mask"],
        train=False,
    )
    logits = mask_logits(
        policy.apply(
            params,
            batch["ego_rtg"],
            batch["ego_obs"],
            batch["ego_actions"],
            timesteps=batch["timesteps"],
            embedding=z,
            mask=batch["mask"],
            train=train,
            rngs=rngs,
        ),
        batch["ego_avail"],
    )
    mask = batch["mask"].astype(jnp.float32)
    bc = optax.softmax_cross_entropy_with_integer_labels(logits, batch["ego_actions"])
    acc = masked_accuracy(logits, batch["ego_actions"], mask)
    bc = (bc * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    return bc, {"loss": bc, "bc": bc, "action_accuracy": acc}
