"""OMIS's model — architecture and inference only (Jing et al., NeurIPS 2024, adapted).

The shared representation backbone, the three in-context heads (actor, opponent
imitator, critic), and :class:`OmisAgent`, the inference wrapper. Given trained
parameters, ``OmisAgent`` acts identically no matter how they were produced, so
it is model-layer and carries no dataset or training dependency. The offline
joint training, the loss, and the (unimplemented) search seam live in
:mod:`oaht_bench.offline.omis`.

**One backbone, three heads, no frozen boundary.** OMIS's own reference
(``pretraining/nets.py::GPTModel``) runs one GPT-2 trunk and reads its final
hidden state through three separately-owned linear heads --
``predict_action``, ``predict_value``, ``predict_oppo_action`` -- trained
end-to-end off one combined loss. That is what :class:`OmisBackbone` and
:class:`OmisHeads` reproduce here, unlike LIAM/MeLIBA/TAO's genuinely staged,
frozen-representation training. Deployed today this is ``OMIS w/o S`` -- the
actor head alone, on the same ego-only information set as the other forward
baselines.
"""

from __future__ import annotations

import flax.linen as nn

from oaht_bench.models.backbone import GPT2Model
from oaht_bench.models.return_conditioned_agent import ReturnConditionedAgent


class OmisBackbone(nn.Module):
    """Shared representation backbone, read at the ``o_t`` positions.

    Identical in form to :class:`~oaht_bench.models.liam_agent.LiamEncoder`; what
    differs is that every head reading it -- including the actor -- is trained
    jointly, with no frozen boundary anywhere.
    """

    action_dim: int
    hidden_dim: int = 32
    dropout: float = 0.1

    @nn.compact
    def __call__(self, rtg, obs, actions, *, timesteps, mask=None, train: bool = False):
        return GPT2Model(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            use_cross_attention=False,
            dropout=self.dropout,
        )(rtg, obs, actions, timesteps=timesteps, mask=mask, train=train)


class OmisHeads(nn.Module):
    """The three heads off the shared representation: actor, opponent imitator, critic.

    The actor head is a single dense layer, matching every other baseline's
    Network/Actor role and the reference's own ``predict_action`` (a single
    ``nn.Linear``). The imitator and critic keep the two-hidden-layer heads
    already established here (mirroring
    :class:`~oaht_bench.models.liam_agent.LiamDecoder`); the reference is
    single-layer for these too, but the extra depth is an existing,
    orthogonal choice this refactor does not revisit. The imitator returns
    teammate-action logits (``μ_φ``); the critic returns a scalar value
    (``V_ω``) regressed to the ego return-to-go — the best response's RTG
    when the dataset carries best responses.
    """

    action_dim: int
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, embedding):
        action_logits = nn.Dense(self.action_dim)(embedding)

        h = nn.relu(nn.Dense(self.hidden_dim)(embedding))
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        mate_action_logits = nn.Dense(self.action_dim)(h)

        g = nn.relu(nn.Dense(self.hidden_dim)(embedding))
        g = nn.relu(nn.Dense(self.hidden_dim)(g))
        value = nn.Dense(1)(g)[..., 0]
        return action_logits, mate_action_logits, value


class OmisAgent(ReturnConditionedAgent):
    """OMIS's architecture and inference as a :class:`ReturnConditionedAgent`.

    The base owns the rolling ego-window / return-to-go deployment; OMIS
    (without search) supplies its modules and the forward: one backbone pass,
    then the actor head. The imitator and critic (:class:`OmisHeads`) are
    trained alongside it for a future search but are not read by :meth:`act`.
    The offline joint training, the loss, and the search seam live in
    :mod:`oaht_bench.offline.omis`, which composes one of these.
    """

    def build_model(self) -> None:
        net = self.config.network
        if net.obs_dim is None or net.action_dim is None:
            raise ValueError(
                "obs_dim/action_dim are unresolved on the network config; the "
                "runner must resolve them from the dataset before build_model()."
            )
        self.backbone = OmisBackbone(
            action_dim=net.action_dim, hidden_dim=net.hidden_dim, dropout=net.dropout
        )
        self.heads = OmisHeads(action_dim=net.action_dim, hidden_dim=net.hidden_dim)

    def act(self, params, rtg, obs, actions, *, timesteps, mask):
        embedding = self.backbone.apply(
            params["stage2"]["backbone"],
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            mask=mask,
            train=False,
        )
        action_logits, _, _ = self.heads.apply(params["stage2"]["heads"], embedding)
        return action_logits

    def mate_action_logits(self, params, hstate):
        """The imitator head (``μ_φ``) reading the shared representation -- the
        same signal ``omis_joint_loss`` (``offline/omis.py``) trains, which is
        why OMIS trains an imitator at all even though the search-free actor
        never reads it. Raw logits; the caller masks illegal actions.
        """
        embedding = self.backbone.apply(
            params["stage2"]["backbone"],
            hstate.ctx_rtg[None],
            hstate.ctx_obs[None],
            hstate.ctx_act[None],
            timesteps=hstate.ctx_t[None],
            mask=hstate.ctx_mask[None],
            train=False,
        )
        _, mate_logits, _ = self.heads.apply(params["stage2"]["heads"], embedding[:, -1])
        return mate_logits[0]
