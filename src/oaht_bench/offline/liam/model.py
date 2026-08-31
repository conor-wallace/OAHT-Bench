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
import jax
import jax.numpy as jnp

from oaht_bench.offline.backbone import DecisionTransformer
from oaht_bench.offline.liam.losses import liam_policy_loss, liam_reconstruction_loss
from oaht_bench.offline.registry import BaseAhtPolicy
from oaht_bench.offline.utils import sample_window_batch


class LiamEncoder(nn.Module):
    """Ego-history encoder: the backbone, read at the ``o_t`` positions."""

    action_dim: int
    hidden_dim: int = 32
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
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, embedding):
        h = nn.relu(nn.Dense(self.hidden_dim)(embedding))
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        mate_obs_hat = nn.Dense(self.obs_dim)(h)

        g = nn.relu(nn.Dense(self.hidden_dim)(embedding))
        g = nn.relu(nn.Dense(self.hidden_dim)(g))
        mate_action_logits = nn.Dense(self.action_dim)(g)
        return mate_obs_hat, mate_action_logits


class LiamNetwork(nn.Module):
    """Stage 2: the backbone, conditioned on a frozen teammate embedding.

    LIAM concatenates the embedding to the observation (``liam_agent.py:536``)
    rather than cross-attending; that is the conditioning mode which
    distinguishes it from TAO.
    """

    action_dim: int
    hidden_dim: int = 32
    dropout: float = 0.1

    @nn.compact
    def __call__(self, rtg, obs, actions, *, timesteps, embedding, mask=None, train: bool = False):
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


class LiamPolicy(BaseAhtPolicy):
    """LIAM on the two-stage contract: ego-history encoder, embedding concatenated
    to the observation.

    Stage 1 trains the encoder and reconstruction decoder; stage 2 trains the
    policy against the frozen encoder; ``act`` runs both for evaluation.
    """

    name = "liam"

    def build_model(self) -> None:
        net = self.config.network
        if net.obs_dim is None or net.action_dim is None:
            raise ValueError(
                "obs_dim/action_dim are unresolved on the network config; the "
                "runner must resolve them from the dataset before build_model()."
            )
        common = dict(hidden_dim=net.hidden_dim, dropout=net.dropout)
        self.encoder = LiamEncoder(action_dim=net.action_dim, **common)
        self.decoder = LiamDecoder(
            obs_dim=net.obs_dim, action_dim=net.action_dim, hidden_dim=net.hidden_dim
        )
        self.network = LiamNetwork(action_dim=net.action_dim, **common)

    def _sample_batch(self, _step):
        """Sample a batch of windows. LIAM's encoder reads the ego stream, so a
        batch is just windows -- no cross trajectory and no contrastive term. The
        step index is ignored; each call draws a fresh minibatch."""
        return sample_window_batch(self.dataset.windows, self.np_rng, self.config.stage2_batch_size)

    def train_stage_1(self):
        init_batch = self._sample_batch(0)
        self.rng, k1, k2 = jax.random.split(self.rng, 3)
        encoder_params = self.encoder.init(
            k1,
            init_batch["ego_rtg"],
            init_batch["ego_obs"],
            init_batch["ego_actions"],
            timesteps=init_batch["timesteps"],
            mask=init_batch["mask"],
        )
        init_z = self.encoder.apply(
            encoder_params,
            init_batch["ego_rtg"],
            init_batch["ego_obs"],
            init_batch["ego_actions"],
            timesteps=init_batch["timesteps"],
            mask=init_batch["mask"],
        )
        decoder_params = self.decoder.init(k2, init_z)
        params = {"encoder": encoder_params, "decoder": decoder_params}

        def loss(p, b, rngs):
            return liam_reconstruction_loss(p, self.encoder, self.decoder, b, rngs=rngs)

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
        init_z = self.encoder.apply(
            stage1_params["encoder"],
            init_batch["ego_rtg"],
            init_batch["ego_obs"],
            init_batch["ego_actions"],
            timesteps=init_batch["timesteps"],
            mask=init_batch["mask"],
        )
        policy_params = self.network.init(
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
                p, self.network, self.encoder, stage1_params["encoder"], b, rngs=rngs
            )

        return self._run_stage(
            loss,
            policy_params,
            self._sample_batch,
            learning_rate=self.config.stage2_learning_rate,
            steps=self.config.stage2_steps,
            prefix="Stage2",
        )

    def act(self, params, rtg, obs, actions, *, timesteps, mask):
        z = self.encoder.apply(
            params["stage1"]["encoder"],
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            mask=mask,
            train=False,
        )
        return self.network.apply(
            params["stage2"],
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            embedding=z,
            mask=mask,
            train=False,
        )
