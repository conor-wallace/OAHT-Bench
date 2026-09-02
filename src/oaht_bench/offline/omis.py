"""OMIS — Opponent Modeling with In-context Search, the *search-free* actor plus
the components a later search would need, adapted to the offline setting.

Sources, in the priority this port was written against:

1. **The OMIS repository** (``/Users/conorwallace/Documents/Personal/Projects/OMIS``)
   is the source of truth for the components and objective. From it:
   - ``pretraining/nets.py`` builds one GPT-2 backbone with **three in-context
     heads** — an **actor** cloning the best-response self-action, an **opponent
     imitator** cloning the teammate action, and a **critic** regressing the
     best-response return-to-go;
   - ``testing/search.py`` runs decision-time search over a ``fake_env`` using all
     three heads. That search is deliberately **not** implemented here (see
     :func:`omis_search`); only the actor is deployed, which is the paper's
     ``OMIS w/o S`` ablation.
2. **The paper** (Jing et al., NeurIPS 2024; ``omis.pdf``): actor ``π_θ``, imitator
   ``μ_φ``, critic ``V_ω`` over shared in-context data ``D`` (Eqs. 3–5); search is
   the ``|A|×M×L`` rollout of Eqs. 6–10.
3. **The shared offline protocol** (:mod:`oaht_bench.offline.liam`,
   :mod:`oaht_bench.offline.tao`): the DT backbone is the encoder read at ``o_t``,
   training is two-stage (representation, then a frozen-encoder policy), and losses
   are masked over valid timesteps.

**What the offline adaptation changes, and why it is honest.** OMIS shares one
backbone across the three heads; this pipeline separates a *representation*
backbone from a *policy* backbone (LIAM does the same). So here the imitator and
critic ride the **stage-1 encoder** — they *are* the teammate representation, the
analogue of LIAM's reconstruction decoder — and the **actor** is the stage-2
policy conditioned on that frozen representation, exactly like
:class:`liam.LiamPolicy`. The actor therefore conditions on the ego history only,
not on live teammate actions: OMIS's perfect-information opponent-action input is
dropped so the baseline is evaluated on the **same information set** as LIAM and
MeLIBA (the shared rollout in :mod:`oaht_bench.offline.evaluate` is ego-stream
only, and fairness requires OMIS not see what the others cannot). The
opponent-conditioning survives through the representation, which is trained to
imitate the teammate and value the best response.

**Search is left open.** Both search components are trained and saved — the
imitator (opponent rollouts) and the critic (leaf values). Adding search later is
:func:`omis_search`, not a retrain. Deployed today, OMIS w/o S is the actor alone,
which — as its own authors note — is the version on equal footing with the
forward-only baselines.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from oaht_bench.models.backbone import DecisionTransformer
from oaht_bench.offline.registry import BaseAhtPolicy
from oaht_bench.offline.utils import mask_logits, sample_window_batch


def _masked_accuracy(logits, labels, mask) -> jnp.ndarray:
    """Top-1 accuracy over valid timesteps (see :func:`liam._masked_accuracy`)."""
    correct = (jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)
    m = mask.astype(jnp.float32)
    return (correct * m).sum() / jnp.maximum(m.sum(), 1.0)


class OmisEncoder(nn.Module):
    """Shared representation backbone, read at the ``o_t`` positions.

    Identical in form to :class:`liam.LiamEncoder`; what differs is the stage-1
    objective that trains it — teammate imitation and best-response value rather
    than teammate reconstruction.
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
    (mirroring :class:`liam.LiamDecoder`). The imitator returns teammate-action
    logits (``μ_φ``); the critic returns a scalar value (``V_ω``) regressed to the
    ego return-to-go — the best response's RTG when the dataset carries best
    responses. Neither is used by the search-free actor; both are trained so
    :func:`omis_search` can be added without retraining.
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

    Structurally :class:`liam.LiamPolicy` — a DT over the ego stream with the
    representation concatenated to the observation. This is OMIS's ``π_θ``
    deployed feed-forward (``OMIS w/o S``); search would wrap it, not replace it.
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


def omis_representation_loss(
    params, encoder, model, batch, *, value_coef=1.0, rngs=None, train: bool = True
):
    """Stage 1: train the two search components on the shared representation.

    ``imitator_ce + value_coef · critic_mse``. The imitator is the categorical
    negative log-likelihood of the teammate action (``μ_φ``); the critic is the
    mean-squared error to the ego return-to-go (``V_ω`` regressed to ``G^1``, which
    is the best response's RTG on best-response data). Both are masked to valid
    timesteps and reported with an accuracy so a falling loss is interpretable.
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
    mate_logits, value = model.apply(params["model"], z)
    mate_logits = mask_logits(mate_logits, batch["mate_avail"])

    mask = batch["mask"].astype(jnp.float32)
    denom = jnp.maximum(mask.sum(), 1.0)

    imitator = optax.softmax_cross_entropy_with_integer_labels(mate_logits, batch["mate_actions"])
    imitator = (imitator * mask).sum() / denom
    imitator_acc = _masked_accuracy(mate_logits, batch["mate_actions"], mask)

    critic = (value - batch["ego_rtg"]) ** 2
    critic = (critic * mask).sum() / denom

    total = imitator + value_coef * critic
    return total, {
        "loss": total,
        "imitator": imitator,
        "imitator_accuracy": imitator_acc,
        "critic": critic,
    }


def omis_actor_loss(
    params, actor, encoder, encoder_params, batch, *, rngs=None, train: bool = True
):
    """Stage 2: behaviour cloning of the ego (best-response) action.

    Conditioned on the frozen representation; ``encoder_params`` are stage-1
    outputs and are never differentiated, so no ``stop_gradient`` is needed.
    Mirrors :func:`liam.liam_policy_loss`.
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
        actor.apply(
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
    acc = _masked_accuracy(logits, batch["ego_actions"], mask)
    bc = (bc * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    return bc, {"loss": bc, "bc": bc, "action_accuracy": acc}


def omis_search(*args, **kwargs):
    """Decision-time search over the environment model — **not implemented**.

    The seam is deliberately left open. Everything search needs is trained and
    saved: stage-1 params carry ``encoder`` and ``model`` (the opponent imitator
    ``μ_φ`` and critic ``V_ω``); stage-2 params carry the actor ``π_θ``.

    A search module (cf. ``OMIS/testing/search.py``) would, at each timestep,
    enumerate the legal ego actions and roll ``M`` trajectories of length ``L``
    through the environment as a ``fake_env`` — ego actions from the actor,
    teammate actions from the imitator, transitions from ``env.step``, leaf value
    from the critic — average to ``Q̂`` (Eq. 8), take ``argmax Q̂`` (Eq. 9), and
    fall back to sampling the actor when ``‖Q̂‖`` is below ``ε`` (Eq. 10). Adding
    it is a new evaluation path, not a retrain, and — per the OMIS paper — it must
    be reported as a distinct *test-time-simulator-access* entry rather than
    compared against the forward-only baselines.

    Deploying ``OMIS w/o S`` today uses the actor alone (see :meth:`OmisPolicy.act`).
    """
    raise NotImplementedError(omis_search.__doc__)


class OmisPolicy(BaseAhtPolicy):
    """OMIS (without search) on the two-stage contract.

    Same shape as :class:`~oaht_bench.offline.liam.model.LiamPolicy`, but stage 1
    trains an opponent imitator and a best-response critic off the shared
    representation (rather than a reconstruction decoder), and stage 2 clones the
    ego best-response action. The imitator and critic are trained and saved for a
    future :func:`omis_search`; the deployed actor is search-free. ``value_coef``
    is OMIS-specific and read from the top-level config.
    """

    name = "omis"

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

    def _sample_batch(self, _step):
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
        model_params = self.model.init(k2, init_z)
        params = {"encoder": encoder_params, "model": model_params}

        def loss(p, b, rngs):
            return omis_representation_loss(
                p, self.encoder, self.model, b, value_coef=self.config.value_coef, rngs=rngs
            )

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
        actor_params = self.actor.init(
            k,
            init_batch["ego_rtg"],
            init_batch["ego_obs"],
            init_batch["ego_actions"],
            timesteps=init_batch["timesteps"],
            embedding=init_z,
            mask=init_batch["mask"],
        )

        def loss(p, b, rngs):
            return omis_actor_loss(
                p, self.actor, self.encoder, stage1_params["encoder"], b, rngs=rngs
            )

        return self._run_stage(
            loss,
            actor_params,
            self._sample_batch,
            learning_rate=self.config.stage2_learning_rate,
            steps=self.config.stage2_steps,
            prefix="Stage2",
        )

    def act(self, params, rtg, obs, actions, *, timesteps, mask):
        z = self.encoder.apply(
            params["stage1"]["encoder"], rtg, obs, actions,
            timesteps=timesteps, mask=mask, train=False,
        )
        return self.actor.apply(
            params["stage2"], rtg, obs, actions,
            timesteps=timesteps, embedding=z, mask=mask, train=False,
        )
