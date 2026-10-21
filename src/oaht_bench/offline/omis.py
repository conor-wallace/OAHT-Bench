"""OMIS — Opponent Modeling with In-context Search, the *search-free* actor plus
the components a later search would need, adapted to the offline setting.

Sources, in the priority this port was written against:

1. **The OMIS repository** (``/Users/conorwallace/Documents/Personal/Projects/OMIS``)
   is the source of truth for the components and objective. From it:
   - ``pretraining/nets.py`` builds one GPT-2 backbone with **three in-context
     heads** — an **actor** cloning the best-response self-action, an **opponent
     imitator** cloning the teammate action, and a **critic** regressing the
     best-response return-to-go;
   - ``pretraining/nn_trainer.py``'s ``train_step`` trains all three **jointly**:
     one combined loss, ``act_loss + vf_coef * value_loss + oppo_pi_coef *
     oppo_pi_loss``, one backward pass through the shared backbone. There is no
     frozen boundary anywhere in the reference, and the paper never claims one
     (§4.1 describes one input sequence through one shared backbone producing
     all three outputs);
   - ``testing/search.py`` runs decision-time search over a ``fake_env`` using all
     three heads. That search is deliberately **not** implemented here (see
     :func:`omis_search`); only the actor is deployed, which is the paper's
     ``OMIS w/o S`` ablation.
2. **The paper** (Jing et al., NeurIPS 2024; ``omis.pdf``): actor ``π_θ``, imitator
   ``μ_φ``, critic ``V_ω`` over shared in-context data ``D`` (Eqs. 3–5); search is
   the ``|A|×M×L`` rollout of Eqs. 6–10.

**All three heads condition on D_t^k, matching Eqs. 3-5 -- not simplified
away.** An earlier version of this port dropped OMIS's opponent-action
conditioning entirely ("ego-stream-only, same information set as LIAM and
MeLIBA"), on a fairness argument: the shared rollout in
:mod:`oaht_bench.offline.evaluate` is ego-stream only, and OMIS seeing what
the others cannot would be an unfair comparison. That was a real divergence
from the paper rather than an equivalent simplification, and was reverted
once a real experiment surfaced it -- the paper's actual design makes
*every* forward-baseline comparison TAO-shaped: OMIS adapts in-context the
same way TAO does, just through one shared backbone (Fig. 1) rather than
TAO's separate-encoder-plus-cross-attention, so it is evaluated through
:mod:`oaht_bench.offline.incontext_eval` alongside TAO, not through the
ego-stream-only rollout LIAM/MeLIBA use. ``D_t^k = (D_epi^k, D_step^k)``
(Sec. 4.1): ``D_step`` is the current episode's own opponent-action history,
threaded through training as the (label-shifted, see :func:`_shift_right`)
``mate_actions`` field already in every window; ``D_epi`` is ``C`` past
episodes against the same opponent (Appendix C), reusing
:func:`~oaht_bench.dataset.sampler.sample_stage2`'s ``context_mate_*``
fields -- built for TAO, teammate-agnostic, and directly applicable here.
Both are new, off-by-default capabilities on the shared
:class:`~oaht_bench.models.backbone.GPT2Model` (see that module's own
docstring for the token-layout design); :meth:`OmisAgent.act`'s docstring
explains how deployment supplies them where the ordinary, every-baseline
:class:`~oaht_bench.models.return_conditioned_agent.ContextWindow` cannot.

**One real training call, not two.** :class:`BaseAhtTrainer`'s
``train_stage_1() -> train_stage_2(stage1_params)`` contract is shared with
LIAM/MeLIBA/TAO, whose stage 2 trains a policy against a *frozen* stage-1
representation -- the reason that contract takes two calls at all. OMIS has
no such freeze (no `stop_gradient` anywhere in :func:`omis_joint_loss`), so
there is nothing for a first stage to produce. An earlier version of this
file still ran real gradient steps in both calls anyway -- continuing the
same parameters from `stage1_steps`/`stage1_learning_rate` into
`stage2_steps`/`stage2_learning_rate` -- but those are TAO's fields
(`configs/job.py`'s `OfflineTrainingConfig` docstring: "Defaults are TAO's"),
inherited from when OMIS genuinely *was* staged like LIAM/MeLIBA (a bug,
fixed and recorded in `docs/tuning_record.md`); keeping the two-call shape
after the loss became joint was leftover plumbing, not something the paper's
one-phase pretraining (Appendix H.2 gives one set of hyperparameters, no
stage split) asked for. ``train_stage_1`` now returns ``{}`` immediately,
mirroring :meth:`~oaht_bench.offline.bc.BcTrainer.train_stage_1`'s
established precedent for exactly this situation; all training happens in
``train_stage_2``, once, under ``stage2_learning_rate``/``stage2_steps``.
:meth:`OmisAgent.act` reads the resulting parameters from ``stage2``.

**Search is left open.** Both search components are trained and saved — the
imitator (opponent rollouts) and the critic (leaf values). Adding search later is
:func:`omis_search`, not a retrain. Deployed today, OMIS w/o S is the actor alone,
which — as its own authors note — is the version on equal footing with the
forward-only baselines.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from oaht_bench.dataset.sampler import sample_stage2
from oaht_bench.models.omis_agent import OmisAgent
from oaht_bench.offline.registry import BaseAhtTrainer
from oaht_bench.offline.utils import mask_logits, to_jax


def _masked_accuracy(logits, labels, mask) -> jnp.ndarray:
    """Top-1 accuracy over valid timesteps (see :func:`liam._masked_accuracy`)."""
    correct = (jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)
    m = mask.astype(jnp.float32)
    return (correct * m).sum() / jnp.maximum(m.sum(), 1.0)


def _shift_right(mate_actions):
    """``D_step``'s opponent-action history at position ``t`` is
    ``a^{-1,k}_{t-1}``, not ``a^{-1,k}_t`` -- the imitator's label at ``t`` is
    the raw, unshifted ``batch["mate_actions"][:, t]``, so feeding the same
    tensor as the backbone's conditioning input would let the label leak into
    its own input. Position 0 has no history yet: filled with ``-10``, the
    same "no action" sentinel ``ContextWindow.ctx_act`` already uses
    (``models/return_conditioned_agent.py``) -- not ``0``, which is a real
    action id and would read as "the opponent played action 0" rather than
    "unknown". ``jax.nn.one_hot`` maps any out-of-range index (negative
    included) to an all-zero row, so the backbone sees a clean "nothing here"
    token instead of a spurious action.
    """
    filler = jnp.full_like(mate_actions[:, :1], -10)
    return jnp.concatenate([filler, mate_actions[:, :-1]], axis=1)


def omis_joint_loss(
    params, backbone, heads, batch, *, vf_coef=0.5, oppo_pi_coef=0.8, rngs=None, train: bool = True
):
    """One backbone pass, three heads, one combined loss -- mirrors the
    reference's ``nn_trainer.py``: ``act_loss + vf_coef * value_loss +
    oppo_pi_coef * oppo_pi_loss``. All three heads read the *same* embedding
    and are differentiated together; nothing here is ``stop_gradient``-ed.

    The embedding conditions on ``D_t^k = (D_epi^k, D_step^k)`` (Eqs. 3-5),
    not just the ego stream: ``D_step`` is the batch's own (shifted)
    ``mate_actions``, and ``D_epi`` is the ``context_mate_*`` fields
    :func:`~oaht_bench.dataset.sampler.sample_stage2` already builds (``C``
    past episodes against the same teammate, Appendix C) -- the same
    cross-episode context TAO's encoder cross-attends to, fed here as a
    prepended block through the shared backbone instead.

    ``act_loss``/``oppo_pi_loss`` are the categorical negative log-likelihood
    of the ego action and the teammate action respectively; ``value_loss`` is
    the mean-squared error of the critic to the ego return-to-go (the best
    response's RTG on best-response data). All three are masked to valid
    timesteps and reported with an accuracy so a falling loss is interpretable.
    """
    embedding = backbone.apply(
        params["backbone"],
        batch["ego_rtg"],
        batch["ego_obs"],
        batch["ego_actions"],
        timesteps=batch["timesteps"],
        mask=batch["mask"],
        mate_actions=_shift_right(batch["mate_actions"]),
        epi_state=batch["context_mate_next_obs"],
        epi_mate_actions=batch["context_mate_actions"],
        epi_mask=batch["context_mask"],
        train=train,
        rngs=rngs,
    )
    action_logits, mate_logits, value = heads.apply(params["heads"], embedding)
    action_logits = mask_logits(action_logits, batch["ego_avail"])
    mate_logits = mask_logits(mate_logits, batch["mate_avail"])

    mask = batch["mask"].astype(jnp.float32)
    denom = jnp.maximum(mask.sum(), 1.0)

    act_loss = optax.softmax_cross_entropy_with_integer_labels(action_logits, batch["ego_actions"])
    act_loss = (act_loss * mask).sum() / denom
    action_acc = _masked_accuracy(action_logits, batch["ego_actions"], mask)

    oppo_pi_loss = optax.softmax_cross_entropy_with_integer_labels(
        mate_logits, batch["mate_actions"]
    )
    oppo_pi_loss = (oppo_pi_loss * mask).sum() / denom
    imitator_acc = _masked_accuracy(mate_logits, batch["mate_actions"], mask)

    value_loss = (value - batch["ego_rtg"]) ** 2
    value_loss = (value_loss * mask).sum() / denom

    total = act_loss + vf_coef * value_loss + oppo_pi_coef * oppo_pi_loss
    return total, {
        "loss": total,
        "bc": act_loss,
        "action_accuracy": action_acc,
        "imitator": oppo_pi_loss,
        "imitator_accuracy": imitator_acc,
        "critic": value_loss,
    }


def omis_search(*args, **kwargs):
    """Decision-time search over the environment model — **not implemented**.

    The seam is deliberately left open. Everything search needs is trained and
    saved: ``params["stage2"]`` carries ``backbone`` and ``heads`` -- the actor
    ``π_θ``, opponent imitator ``μ_φ``, and critic ``V_ω`` together.

    A search module (cf. ``OMIS/testing/search.py``) would, at each timestep,
    enumerate the legal ego actions and roll ``M`` trajectories of length ``L``
    through the environment as a ``fake_env`` — ego actions from the actor,
    teammate actions from the imitator, transitions from ``env.step``, leaf value
    from the critic — average to ``Q̂`` (Eq. 8), take ``argmax Q̂`` (Eq. 9), and
    fall back to sampling the actor when ``‖Q̂‖`` is below ``ε`` (Eq. 10). Adding
    it is a new evaluation path, not a retrain, and — per the OMIS paper — it must
    be reported as a distinct *test-time-simulator-access* entry rather than
    compared against the forward-only baselines.

    Deploying ``OMIS w/o S`` today uses the actor alone (see :meth:`OmisAgent.act`).
    """
    raise NotImplementedError(omis_search.__doc__)


class OmisTrainer(BaseAhtTrainer):
    """OMIS (without search), trained in one real pass.

    One ``{backbone, heads}`` parameter tree, one combined loss
    (:func:`omis_joint_loss`), unlike LIAM/MeLIBA/TAO's staged,
    frozen-representation training. ``train_stage_1`` is a no-op, matching
    :meth:`~oaht_bench.offline.bc.BcTrainer.train_stage_1`'s precedent for a
    baseline with nothing for a representation stage to do; all optimisation
    happens in ``train_stage_2``, under ``stage2_learning_rate``/
    ``stage2_steps``. ``vf_coef``/``oppo_pi_coef`` are OMIS-specific and read
    from the top-level config.
    """

    name = "omis"

    def build_model(self) -> None:
        # Inference is the composed OmisAgent's; training reads its
        # backbone/heads.
        self.agent = OmisAgent(self.config)
        self.agent.build_model()

    def _sample_batch(self, _step):
        return to_jax(
            sample_stage2(
                self.dataset.windows,
                self.dataset.index,
                self.np_rng,
                batch_size=self.config.batch_size,
                context_trajectories=self.config.context_trajectories,
            )
        )

    def _loss(self, p, b, rngs, frozen):
        return omis_joint_loss(
            p,
            self.agent.backbone,
            self.agent.heads,
            b,
            vf_coef=self.config.vf_coef,
            oppo_pi_coef=self.config.oppo_pi_coef,
            rngs=rngs,
        )

    def train_stage_1(self):
        """No separate representation stage -- OMIS trains one joint
        objective in a single pass (see :func:`omis_joint_loss`'s own
        docstring: "nothing here is `stop_gradient`-ed"), so there is nothing
        for a first stage to produce. Mirrors
        :meth:`~oaht_bench.offline.bc.BcTrainer.train_stage_1`'s established
        no-op rather than running the shared step loop for zero steps, which
        would leave an empty ``Stage1/`` block in the metrics that reads as a
        bug, not a design choice.
        """
        return {}

    def train_stage_2(self, stage1_params):
        del stage1_params  # unused: nothing from a first stage to continue.
        init_batch = self._sample_batch(0)
        self.rng, k = jax.random.split(self.rng)

        def init_one(key):
            k1, k2 = jax.random.split(key)
            backbone_params = self.agent.backbone.init(
                k1,
                init_batch["ego_rtg"],
                init_batch["ego_obs"],
                init_batch["ego_actions"],
                timesteps=init_batch["timesteps"],
                mask=init_batch["mask"],
                mate_actions=_shift_right(init_batch["mate_actions"]),
                epi_state=init_batch["context_mate_next_obs"],
                epi_mate_actions=init_batch["context_mate_actions"],
                epi_mask=init_batch["context_mask"],
            )
            embedding = self.agent.backbone.apply(
                backbone_params,
                init_batch["ego_rtg"],
                init_batch["ego_obs"],
                init_batch["ego_actions"],
                timesteps=init_batch["timesteps"],
                mask=init_batch["mask"],
                mate_actions=_shift_right(init_batch["mate_actions"]),
                epi_state=init_batch["context_mate_next_obs"],
                epi_mate_actions=init_batch["context_mate_actions"],
                epi_mask=init_batch["context_mask"],
            )
            heads_params = self.agent.heads.init(k2, embedding)
            return {"backbone": backbone_params, "heads": heads_params}

        params = self._init_params(init_one, k)

        return self._run_stage(
            self._loss,
            params,
            self._sample_batch,
            learning_rate=self.config.stage2_learning_rate,
            steps=self.config.stage2_steps,
            prefix="Stage2",
        )
