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

**What the offline adaptation changes, and why it is honest.** The actor
therefore conditions on the ego history only, not on live teammate actions:
OMIS's perfect-information opponent-action input is dropped so the baseline is
evaluated on the **same information set** as LIAM and MeLIBA (the shared
rollout in :mod:`oaht_bench.offline.evaluate` is ego-stream only, and fairness
requires OMIS not see what the others cannot). The opponent-conditioning
survives through the shared representation, which is trained -- jointly with
the actor, matching the reference -- to imitate the teammate and value the
best response.

**Two calls, one joint optimization.** :class:`BaseAhtTrainer`'s
``train_stage_1() -> train_stage_2(stage1_params)`` contract is shared with
LIAM/MeLIBA/TAO, whose stage 2 trains a policy against a *frozen* stage-1
representation. OMIS has no such freeze: both calls optimise the *same*
``{backbone, heads}`` parameter tree against the *same* joint loss, stage 2
simply continuing from stage 1's checkpoint at a different learning
rate/step budget (``stage1_learning_rate``/``steps`` vs.
``stage2_learning_rate``/``steps``) -- a two-phase schedule within one
training run, not two stages with different objectives. :meth:`OmisAgent.act`
reads the final, most-trained parameters from ``stage2``.

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

from oaht_bench.models.omis_agent import OmisAgent
from oaht_bench.offline.registry import BaseAhtTrainer
from oaht_bench.offline.utils import mask_logits, sample_window_batch


def _masked_accuracy(logits, labels, mask) -> jnp.ndarray:
    """Top-1 accuracy over valid timesteps (see :func:`liam._masked_accuracy`)."""
    correct = (jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)
    m = mask.astype(jnp.float32)
    return (correct * m).sum() / jnp.maximum(m.sum(), 1.0)


def omis_joint_loss(
    params, backbone, heads, batch, *, vf_coef=0.5, oppo_pi_coef=0.8, rngs=None, train: bool = True
):
    """One backbone pass, three heads, one combined loss -- mirrors the
    reference's ``nn_trainer.py``: ``act_loss + vf_coef * value_loss +
    oppo_pi_coef * oppo_pi_loss``. All three heads read the *same* embedding
    and are differentiated together; nothing here is ``stop_gradient``-ed.

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
    """OMIS (without search), trained jointly across the two-call contract.

    One ``{backbone, heads}`` parameter tree, one combined loss
    (:func:`omis_joint_loss`) -- unlike LIAM/MeLIBA/TAO's staged,
    frozen-representation training. ``train_stage_1`` initialises and runs the
    first optimisation phase; ``train_stage_2`` continues optimising the *same*
    parameters, not a fresh policy against a frozen encoder. ``vf_coef``/
    ``oppo_pi_coef`` are OMIS-specific and read from the top-level config.
    """

    name = "omis"

    def build_model(self) -> None:
        # Inference is the composed OmisAgent's; training reads its
        # backbone/heads.
        self.agent = OmisAgent(self.config)
        self.agent.build_model()

    def _sample_batch(self, _step):
        return sample_window_batch(self.dataset.windows, self.np_rng, self.config.batch_size)

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
            )
            embedding = self.agent.backbone.apply(
                backbone_params,
                init_batch["ego_rtg"],
                init_batch["ego_obs"],
                init_batch["ego_actions"],
                timesteps=init_batch["timesteps"],
                mask=init_batch["mask"],
            )
            heads_params = self.agent.heads.init(k2, embedding)
            return {"backbone": backbone_params, "heads": heads_params}

        params = self._init_params(init_one, k)

        return self._run_stage(
            self._loss,
            params,
            self._sample_batch,
            learning_rate=self.config.stage1_learning_rate,
            steps=self.config.stage1_steps,
            prefix="Stage1",
        )

    def train_stage_2(self, stage1_params):
        return self._run_stage(
            self._loss,
            stage1_params,
            self._sample_batch,
            learning_rate=self.config.stage2_learning_rate,
            steps=self.config.stage2_steps,
            prefix="Stage2",
        )
