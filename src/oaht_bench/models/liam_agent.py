"""LIAM's model — architecture and inference only (Local Information Agent Modelling).

The ego-history encoder, the teammate-reconstruction decoder, the
embedding-conditioned policy network, and :class:`LiamAgent`, the inference
wrapper. Given trained parameters, ``LiamAgent`` acts identically no matter how
they were produced -- online or offline -- so it is model-layer and carries no
dataset or training dependency. The offline two-stage training and the losses live
in :mod:`oaht_bench.offline.liam`.

**What LIAM is.** An encoder summarises the ego agent's *local* history into an
embedding; a decoder reconstructs the *teammate's* observation and action from
that embedding; the policy is conditioned on the embedding. The teammate is never
observed by the policy -- only modelled -- which is the method's hypothesis and
why it is the natural floor for the trajectory-view family.

**The encoder is the backbone, read at the right position.** LIAM's encoder sees
``o¹_{0..t}`` and ``a¹_{0..t-1}`` -- observations through ``t``, actions only
through ``t-1``, because at ``t`` the ego has not acted yet. In the interleaved
``(G_t, o_t, a_t)`` sequence ``o_t`` sits at index ``3t+1`` and ``a_t`` at
``3t+2``, so under the causal mask the hidden state at ``o_t`` attends to
``o_{≤t}`` and ``a_{<t}`` and *not* ``a_t``. That is exactly LIAM's information
set, so no separate encoder architecture is required.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from oaht_bench.models.backbone import DecisionTransformer
from oaht_bench.models.return_conditioned_agent import ReturnConditionedAgent


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


class LiamAgent(ReturnConditionedAgent):
    """LIAM's architecture and inference as a :class:`ReturnConditionedAgent`.

    Inference is identical no matter how the parameters were produced -- online or
    offline -- so LIAM is an acting policy like the actor-critics, driven by the
    shared :func:`~oaht_bench.common.run_episodes.run_episodes` loop. The base owns
    the rolling-window / return-to-go deployment; LIAM only supplies its modules and
    the forward: encode the ego history, condition the policy on the embedding. The
    offline two-stage training and the losses live in :mod:`oaht_bench.offline.liam`,
    which composes one of these.
    """

    def build_model(self) -> None:
        """Construct the flax modules from ``self.config`` (with resolved dims)."""
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

    def act(self, params, rtg, obs, actions, *, timesteps, mask):
        """Ego action logits: encode the ego history, condition the policy on it."""
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
