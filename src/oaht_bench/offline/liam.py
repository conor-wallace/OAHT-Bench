"""LIAM offline training — the two-stage trainer and its losses.

The architecture and inference live in :mod:`oaht_bench.models.liam_agent`; this
module trains one and defines the reconstruction and policy losses.

**Two stages, not one.** The original trains encoder and policy together and
blocks the gradient with ``stop_gradient`` (``liam_agent.py:536``) because
everything is learned online in a single loop. Offline that constraint is gone:
stage 1 trains encoder and decoder on reconstruction, stage 2 trains the policy
against a frozen encoder. This is TAO's protocol; it *removes* the need for the
gradient block rather than reproducing it, and it makes LIAM and TAO two choices
of encoder and conditioning mode over one training procedure.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from oaht_bench.models.liam_agent import LiamAgent
from oaht_bench.offline.registry import BaseAhtTrainer
from oaht_bench.offline.utils import mask_logits, masked_accuracy, sample_window_batch


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


class LiamTrainer(BaseAhtTrainer):
    """LIAM on the two-stage contract: ego-history encoder, embedding concatenated
    to the observation.

    Stage 1 trains the encoder and reconstruction decoder; stage 2 trains the
    policy against the frozen encoder. The model and inference are a composed
    :class:`~oaht_bench.models.liam_agent.LiamAgent`, which does the acting.
    """

    name = "liam"

    def build_model(self) -> None:
        self.agent = LiamAgent(self.config)
        self.agent.build_model()

    def _sample_batch(self, _step):
        """Sample a batch of windows. LIAM's encoder reads the ego stream, so a
        batch is just windows -- no cross trajectory and no contrastive term. The
        step index is ignored; each call draws a fresh minibatch."""
        return sample_window_batch(self.dataset.windows, self.np_rng, self.config.stage2_batch_size)

    def train_stage_1(self):
        init_batch = self._sample_batch(0)
        self.rng, k1, k2 = jax.random.split(self.rng, 3)
        encoder_params = self.agent.encoder.init(
            k1,
            init_batch["ego_rtg"],
            init_batch["ego_obs"],
            init_batch["ego_actions"],
            timesteps=init_batch["timesteps"],
            mask=init_batch["mask"],
        )
        init_z = self.agent.encoder.apply(
            encoder_params,
            init_batch["ego_rtg"],
            init_batch["ego_obs"],
            init_batch["ego_actions"],
            timesteps=init_batch["timesteps"],
            mask=init_batch["mask"],
        )
        decoder_params = self.agent.decoder.init(k2, init_z)
        params = {"encoder": encoder_params, "decoder": decoder_params}

        def loss(p, b, rngs):
            return liam_reconstruction_loss(p, self.agent.encoder, self.agent.decoder, b, rngs=rngs)

        return self._run_stage(
            loss,
            params,
            self._sample_batch,
            learning_rate=self.config.stage1_learning_rate,
            steps=self.config.stage1_steps,
            prefix="Stage1",
        )

    def train_stage_2(self, stage1_params):
        init_batch = self._sample_batch(0)
        self.rng, k = jax.random.split(self.rng)
        init_z = self.agent.encoder.apply(
            stage1_params["encoder"],
            init_batch["ego_rtg"],
            init_batch["ego_obs"],
            init_batch["ego_actions"],
            timesteps=init_batch["timesteps"],
            mask=init_batch["mask"],
        )
        policy_params = self.agent.network.init(
            k,
            init_batch["ego_rtg"],
            init_batch["ego_obs"],
            init_batch["ego_actions"],
            timesteps=init_batch["timesteps"],
            embedding=init_z,
            mask=init_batch["mask"],
        )

        def loss(p, b, rngs):
            return liam_policy_loss(
                p, self.agent.network, self.agent.encoder, stage1_params["encoder"], b, rngs=rngs
            )

        return self._run_stage(
            loss,
            policy_params,
            self._sample_batch,
            learning_rate=self.config.stage2_learning_rate,
            steps=self.config.stage2_steps,
            prefix="Stage2",
        )
