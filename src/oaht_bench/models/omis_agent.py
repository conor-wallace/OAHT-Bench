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
frozen-representation training. Deployed today this is ``OMIS w/o S`` --
search (:func:`~oaht_bench.offline.omis.omis_search`) stays the deliberately
open seam it always was.

**Opponent-conditioned, matching the paper's Eqs. 3-5, not simplified away.**
All three heads condition on ``D_t^k = (D_epi^k, D_step^k)`` (Sec. 4.1), not
just the ego's own ``(s_t, a^{1,k,*}_t, G^{1,k,*}_t)`` -- an earlier version
of this port dropped that conditioning entirely ("ego-stream-only, same
information set as LIAM/MeLIBA"), which was a real divergence from the
paper, not a faithful simplification; restored here. ``D_step`` (the current
episode's own opponent-action history) and ``D_epi`` (``C`` past episodes
against the same opponent, Appendix C) are the two new, off-by-default
:class:`~oaht_bench.models.backbone.GPT2Model` capabilities -- see that
module's own docstring for the token-layout design. :meth:`OmisAgent.act`'s
four new keyword args all default to ``None``, reproducing the old
ego-stream-only forward exactly when omitted: the generic
:class:`~oaht_bench.models.return_conditioned_agent.ReturnConditionedAgent`
deployment loop (shared with LIAM/MeLIBA/BC/TAO) has no field for the
opponent's own actions and calls ``act`` without them, so the real,
paper-faithful deployment goes through the bespoke
:mod:`oaht_bench.offline.incontext_eval` path instead -- the same reason TAO
needs a bespoke path (the generic loop can't express cross-episode context),
not a new category of exception.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from oaht_bench.models.backbone import GPT2Model
from oaht_bench.models.masking import mask_logits
from oaht_bench.models.return_conditioned_agent import ContextWindow, ReturnConditionedAgent


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
    def __call__(
        self,
        rtg,
        obs,
        actions,
        *,
        timesteps,
        mask=None,
        mate_actions=None,
        epi_state=None,
        epi_mate_actions=None,
        epi_mask=None,
        train: bool = False,
    ):
        return GPT2Model(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            use_cross_attention=False,
            dropout=self.dropout,
        )(
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            mask=mask,
            mate_actions=mate_actions,
            epi_state=epi_state,
            epi_mate_actions=epi_mate_actions,
            epi_mask=epi_mask,
            train=train,
        )


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

    def act(
        self,
        params,
        rtg,
        obs,
        actions,
        *,
        timesteps,
        mask,
        mate_actions=None,
        epi_state=None,
        epi_mate_actions=None,
        epi_mask=None,
    ):
        """The base ``ReturnConditionedAgent.get_action`` deployment loop (shared
        with LIAM/MeLIBA/BC/TAO) calls this without the four D_step/D_epi kwargs
        below, since ``ContextWindow`` has no field for the opponent's own
        action -- so they default to ``None``, giving the ego-stream-only
        forward exactly as before. The paper-faithful deployment, with real
        within-episode opponent actions and the cross-episode OCW, goes through
        the bespoke :mod:`oaht_bench.offline.incontext_eval` path instead, the
        same reason TAO needs a bespoke path rather than a change to the shared
        base.
        """
        embedding = self.backbone.apply(
            params["stage2"]["backbone"],
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            mask=mask,
            mate_actions=mate_actions,
            epi_state=epi_state,
            epi_mate_actions=epi_mate_actions,
            epi_mask=epi_mask,
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

    def encode_ocw(self, params, mate_next_obs, mate_actions, mate_rewards, mask, timesteps):
        """The Opponent Context Window into ``D_epi``'s three raw arrays.

        Same call signature as :meth:`~oaht_bench.models.tao_agent.TaoAgent.encode_ocw`
        (:mod:`oaht_bench.offline.incontext_eval` calls either one the same way), but
        there is no encoder pass: ``D_epi`` has no separate tower (Sec. 4.1/Fig. 1 is
        one shared backbone), so this just reshapes the OCW's ``(C, T, ...)`` arrays
        to the ``(1, C*T, ...)`` shape :class:`~oaht_bench.models.backbone.GPT2Model`
        reads directly as ``epi_state``/``epi_mate_actions``/``epi_mask``.
        ``mate_rewards`` is unused -- ``D_epi``'s pairs are ``(state, opponent-action)``
        only (Appendix C), not reward; the argument stays for the shared call site.
        """
        del mate_rewards, timesteps
        normalized_obs = (jnp.asarray(mate_next_obs) - self._obs_mean) / self._obs_std
        return {
            "epi_state": normalized_obs.reshape(1, -1, normalized_obs.shape[-1]),
            "epi_mate_actions": jnp.asarray(mate_actions).reshape(1, -1),
            "epi_mask": jnp.asarray(mask).reshape(1, -1),
        }

    def get_action_with_mate_history(
        self, params, obs, done, avail_actions, hstate, rng, mate_action_window, *, reward=None
    ):
        """OMIS's own deployment step, with ``D_step`` threaded in.

        Duplicates :meth:`ReturnConditionedAgent.get_action`'s window roll / return
        decrement / sampling exactly; the one difference is the extra
        ``mate_actions=mate_action_window`` passed into :meth:`act`, and that
        ``D_epi`` (``epi_state``/``epi_mate_actions``/``epi_mask``) is read off
        ``params["stage2"]``, where :mod:`oaht_bench.offline.incontext_eval` bakes
        it once per episode via :meth:`encode_ocw` -- the same place TAO's
        ``context`` lives, because both only change *between* episodes. ``D_step``
        changes every step, which is exactly why it can't live in ``stage2`` the
        same way and has to be threaded in here instead of through the shared,
        every-baseline :class:`ContextWindow` (see the module docstring).

        ``mate_action_window`` is ``(K,)`` int32, left-padded like ``hstate.ctx_act``:
        position ``K-1`` holds the opponent's action one step *before* the current
        one (``a^{-1}_{t-1}``), matching training's ``_shift_right`` convention
        (``offline/omis.py``) -- not the opponent's action at the current step, which
        is unknown until after this one is taken. Built and rolled by the caller
        (:mod:`oaht_bench.offline.incontext_eval`), which already collects the
        opponent's observed action stream for the OCW and is in the one position
        that knows both agents' actions each step.
        """
        K = self.context_length
        h = hstate
        r = (
            jnp.zeros((), jnp.float32)
            if reward is None
            else jnp.reshape(reward, (-1))[0].astype(jnp.float32)
        )
        rtg = h.rtg - r / self._rtg_scale

        ctx_obs = jnp.roll(h.ctx_obs, -1, axis=0)
        ctx_act = jnp.roll(h.ctx_act, -1)
        ctx_rtg = jnp.roll(h.ctx_rtg, -1)
        ctx_t = jnp.roll(h.ctx_t, -1)
        ctx_mask = jnp.roll(h.ctx_mask, -1)

        norm_obs = (jnp.reshape(obs, (-1)).astype(jnp.float32) - self._obs_mean) / self._obs_std
        ctx_obs = ctx_obs.at[-1].set(norm_obs)
        ctx_act = ctx_act.at[-1].set(jnp.int32(-10))
        ctx_rtg = ctx_rtg.at[-1].set(rtg)
        ctx_t = ctx_t.at[-1].set(jnp.minimum(h.step + 1, K * 64).astype(jnp.int32))
        ctx_mask = ctx_mask.at[-1].set(True)

        stage2 = params["stage2"]
        logits = self.act(
            params,
            ctx_rtg[None],
            ctx_obs[None],
            ctx_act[None],
            timesteps=ctx_t[None],
            mask=ctx_mask[None],
            mate_actions=jnp.asarray(mate_action_window, dtype=jnp.int32)[None],
            epi_state=stage2.get("epi_state"),
            epi_mate_actions=stage2.get("epi_mate_actions"),
            epi_mask=stage2.get("epi_mask"),
        )
        avail = jnp.reshape(avail_actions, (-1)).astype(jnp.float32)
        masked = mask_logits(logits[0, -1], avail)
        action = jax.random.categorical(rng, masked).astype(jnp.int32)
        ctx_act = ctx_act.at[-1].set(action)

        new_hstate = ContextWindow(ctx_obs, ctx_act, ctx_rtg, ctx_t, ctx_mask, rtg, h.step + 1)
        return action, new_hstate
