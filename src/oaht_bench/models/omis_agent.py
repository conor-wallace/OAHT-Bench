"""OMIS's model — architecture and inference only (Jing et al., NeurIPS 2024, adapted).

The shared representation encoder, the two search components (opponent imitator and
critic), the search-free actor policy, and :class:`OmisAgent`, the inference wrapper.
Given trained parameters, ``OmisAgent`` acts identically no matter how they were
produced, so it is model-layer and carries no dataset or training dependency. The
offline two-stage training, the losses, and the (unimplemented) search seam live in
:mod:`oaht_bench.offline.omis`.

Structurally OMIS is LIAM: a representation backbone read at the ``o_t`` positions and
an actor conditioned on it. What differs is the stage-1 objective (teammate imitation
and best-response value rather than reconstruction) and the two extra heads it trains
for a future decision-time search. Deployed today this is ``OMIS w/o S`` -- the actor
alone, on the same ego-only information set as the other forward baselines.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from oaht_bench.models.backbone import DecisionTransformer
from oaht_bench.models.return_conditioned_agent import ReturnConditionedAgent


class OmisEncoder(nn.Module):
    """Shared representation backbone, read at the ``o_t`` positions.

    Identical in form to :class:`~oaht_bench.models.liam_agent.LiamEncoder`; what
    differs is the stage-1 objective that trains it — teammate imitation and
    best-response value rather than teammate reconstruction.
    """

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


class OmisModel(nn.Module):
    """The two search components: opponent imitator and critic.

    Two heads off the representation, each two hidden layers with ReLU
    (mirroring :class:`~oaht_bench.models.liam_agent.LiamDecoder`). The imitator
    returns teammate-action logits (``μ_φ``); the critic returns a scalar value
    (``V_ω``) regressed to the ego return-to-go — the best response's RTG when the
    dataset carries best responses. Neither is used by the search-free actor; both
    are trained so a future search can be added without retraining.
    """

    action_dim: int
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, embedding):
        h = nn.relu(nn.Dense(self.hidden_dim)(embedding))
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        mate_action_logits = nn.Dense(self.action_dim)(h)

        g = nn.relu(nn.Dense(self.hidden_dim)(embedding))
        g = nn.relu(nn.Dense(self.hidden_dim)(g))
        value = nn.Dense(1)(g)[..., 0]
        return mate_action_logits, value


class OmisActor(nn.Module):
    """Stage 2: the actor policy, conditioned on the frozen representation.

    Structurally :class:`~oaht_bench.models.liam_agent.LiamNetwork` — a DT over the
    ego stream with the representation concatenated to the observation. This is OMIS's
    ``π_θ`` deployed feed-forward (``OMIS w/o S``); search would wrap it, not replace it.
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


class OmisAgent(ReturnConditionedAgent):
    """OMIS's architecture and inference as a :class:`ReturnConditionedAgent`.

    The base owns the rolling ego-window / return-to-go deployment; OMIS (without
    search) supplies its modules and the forward: encode the representation, condition
    the actor on it. The imitator and critic (:class:`OmisModel`) are trained and saved
    for a future search but are not read at inference. The offline two-stage training,
    the losses, and the search seam live in :mod:`oaht_bench.offline.omis`, which
    composes one of these.
    """

    def build_model(self) -> None:
        net = self.config.network
        if net.obs_dim is None or net.action_dim is None:
            raise ValueError(
                "obs_dim/action_dim are unresolved on the network config; the "
                "runner must resolve them from the dataset before build_model()."
            )
        common = dict(hidden_dim=net.hidden_dim, dropout=net.dropout)
        self.encoder = OmisEncoder(action_dim=net.action_dim, **common)
        self.model = OmisModel(action_dim=net.action_dim, hidden_dim=net.hidden_dim)
        self.actor = OmisActor(action_dim=net.action_dim, **common)

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
        return self.actor.apply(
            params["stage2"],
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            embedding=z,
            mask=mask,
            train=False,
        )
