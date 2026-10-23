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
(Sec. 4.1): ``D_step`` is the decoder window's own opponent *action* history
(Eq. 2 reads as a (state, action) pair, but the reference's
``GPTModel.forward`` has exactly one observation tensor -- the ego's own,
already in the backbone's standard stream -- and loads only opponent
*actions* for the main sequence, confirmed at the data layer by
``get_batch``/``load_dataset`` in ``OMIS/pretraining/utils.py``; see
``models/backbone.py``'s own docstring for the full citation trail), built
by :func:`build_d_step` from the batch's own (label-shifted, see
:func:`_shift_right`) ``mate_actions``; ``D_epi`` is ``C`` past episodes
against the same opponent (Appendix C), and here the state half is real --
the reference's ``prompt_states`` carries the opponent's own observation,
because D_epi reads *other* episodes with no current ego state to reuse --
built by :func:`build_d_epi` directly from the episode store via
:func:`~oaht_bench.dataset.windowing.teammate_episode_ids` -- its own
``H``/``C`` sizing, independent of TAO's GetOffD context even though both
select "C trajectories of the same teammate" (see ``tao.py``'s
``_get_off_d_context`` for the one that isn't).
Both are new, off-by-default capabilities on the shared
:class:`~oaht_bench.models.backbone.GPT2Model` (see that module's own
docstring for the token-layout design, including the reference's agent-role
and separate-D_epi-timestep embeddings, also ported here for the same
fidelity-to-the-released-model reason); :meth:`OmisAgent.act`'s docstring
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
import numpy as np

from oaht_bench.dataset.windowing import (
    enumerate_windows,
    identity_matched_episode_ids,
    teammate_episode_ids,
    window_slice_batch,
)
from oaht_bench.models.omis_agent import OmisAgent
from oaht_bench.offline.registry import BaseAhtTrainer
from oaht_bench.offline.utils import mask_logits, normalize_window_batch, safe_cross_entropy, to_jax

#: The decoder window: the ego stream D_step/D_epi condition a prediction on,
#: plus the decoder's OWN episode's teammate action stream -- D_step is built
#: from this by construction (Eq. 2 never samples elsewhere; see
#: build_d_step). No teammate *observation* field: the reference's main
#: sequence has no opponent-state channel (see module docstring).
_DECODER_FIELDS = (
    "ego_rtg",
    "ego_obs",
    "ego_actions",
    "timesteps",
    "mask",
    "ego_avail",
    "mate_actions",
    "mate_avail",
)
#: D_epi's per-fragment fields -- state and teammate action (Appendix C),
#: plus the mask the backbone's `epi_mask` needs for fragments shorter than
#: `H // C` (a short episode, or H not evenly divisible by C).
_D_EPI_FIELDS = ("mate_obs", "mate_actions", "mask")


def _masked_accuracy(logits, labels, mask) -> jnp.ndarray:
    """Top-1 accuracy over valid timesteps (see :func:`liam._masked_accuracy`)."""
    correct = (jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)
    m = mask.astype(jnp.float32)
    return (correct * m).sum() / jnp.maximum(m.sum(), 1.0)


def _random_t(batch, episode_id: int, rng: np.random.Generator) -> int:
    """A uniform random window end-position within ``episode_id`` -- D_epi's
    fragments are keyed by *teammate*, not drawn from "any window in the
    dataset", so there is no stride-pool/reproducibility argument here (see
    ``tao.py``'s identical helper, which this mirrors rather than imports --
    each baseline's sampling stays self-contained)."""
    length = int(batch.episode_lengths()[episode_id])
    return int(rng.integers(1, length + 1))


def build_d_epi(dataset, teammate_ids, exclude_episode_ids, rng, *, H: int, C: int):
    """D_epi^k = {(s_h, a^{-1,k}_h)}_{h=1}^H -- Appendix C, verbatim, for a
    whole batch of decoder rows at once: sample the ids of ``C`` trajectories
    "from all historical games involving" ``teammate_id`` (ANY ego policy,
    not necessarily the matching best response -- :func:`teammate_episode_ids`
    makes no ego-identity distinction), take a consecutive segment of length
    ``H // C`` from each, concatenate. ``H``/``C`` are D_epi's own
    hyperparameters (paper: H=15, C=3 on LBF) -- independent of the decoder
    window's own length, not reused from it.

    ``teammate_ids``/``exclude_episode_ids`` are arrays, one entry per
    decoder row. Falls back to including that row's own episode when its
    teammate appears in too few *other* episodes to fill ``C`` without it
    (common with 1-4 episodes per teammate on these datasets, see
    ``tao._anchor_and_cross_indices``'s identical fallback), and samples with
    replacement in that case so every fragment is always filled.

    D_epi's actual length is ``C * (H // C)``, not ``H`` itself, whenever
    ``H`` doesn't divide evenly by ``C`` -- the paper's own LBF values
    (``H=15, C=3``) do, but a config that doesn't still gets a consistent,
    non-crashing result rather than one silently rounded up past what was
    actually read.

    Picking *which* episodes/positions is cheap index bookkeeping and stays
    a Python loop, one row at a time; the actual reads are ONE
    :func:`~oaht_bench.dataset.windowing.window_slice_batch` call across
    every row's every fragment (``batch_size * C`` windows), not one call
    per row or per fragment -- see that function's own docstring for why a
    finer-grained loop here would silently undo ``VaultReader.read_slices``'s
    concurrency.
    """
    episode_ids_flat, ts_flat = [], []
    for teammate_id, exclude_episode_id in zip(teammate_ids, exclude_episode_ids, strict=True):
        all_episode_ids = teammate_episode_ids(dataset.batch, dataset.mate, int(teammate_id))
        other_episode_ids = all_episode_ids[all_episode_ids != exclude_episode_id]
        pool = other_episode_ids if other_episode_ids.size else all_episode_ids
        chosen = rng.choice(pool, size=C, replace=pool.size < C)
        episode_ids_flat.extend(int(e) for e in chosen)
        ts_flat.extend(_random_t(dataset.batch, int(e), rng) for e in chosen)
    h = H // C
    flat = window_slice_batch(
        dataset.batch, episode_ids_flat, ts_flat, h, _D_EPI_FIELDS, mate=dataset.mate
    )
    n = len(teammate_ids)
    return {k: v.reshape(n, C * h, *v.shape[2:]) for k, v in flat.items()}


def _shift_right(x, fill):
    """``D_step``'s opponent action history at position ``t`` is
    ``a^{-1,k}_{t-1}``, not ``a^{-1,k}_t`` -- the imitator's label at ``t``
    is the raw, unshifted ``batch["mate_actions"][:, t]``, so feeding the
    same tensor as the backbone's conditioning input would let the label
    leak into its own input. Position 0 has no history yet, filled with
    ``fill``: ``-10`` for ``mate_actions`` (the same "no action" sentinel
    ``ContextWindow.ctx_act`` already uses -- not ``0``, which is a real
    action id and would read as "the opponent played action 0" rather than
    "unknown"; ``jax.nn.one_hot`` maps any out-of-range index, negative
    included, to an all-zero row).
    """
    filler = jnp.full_like(x[:, :1], fill)
    return jnp.concatenate([filler, x[:, :-1]], axis=1)


def build_d_step(window: dict) -> dict:
    """D_step^k = (a^{-1,k}_0, ..., a^{-1,k}_{t-1}) -- Eq. 2's opponent
    action history. (Eq. 2 reads as a (state, action) pair, but the
    reference's main sequence has no opponent-observation channel; the
    ``s_t`` it writes is the ego's own state, already in the backbone's
    standard stream -- see ``models/backbone.py``'s docstring.)

    From the decoder window's OWN episode (the matching best-response
    episode, by construction -- D_step never samples elsewhere, unlike
    D_epi), shifted one step via :func:`_shift_right` so position ``t``
    never sees ``a^{-1,k}_t``.
    """
    return {"mate_actions": _shift_right(window["mate_actions"], fill=-10)}


def omis_joint_loss(
    params, backbone, heads, batch, *, vf_coef=0.5, oppo_pi_coef=0.8, rngs=None, train: bool = True
):
    """One backbone pass, three heads, one combined loss -- mirrors the
    reference's ``nn_trainer.py``: ``act_loss + vf_coef * value_loss +
    oppo_pi_coef * oppo_pi_loss``. All three heads read the *same* embedding
    and are differentiated together; nothing here is ``stop_gradient``-ed.

    The embedding conditions on ``D_t^k = (D_epi^k, D_step^k)`` (Eqs. 3-5),
    not just the ego stream: ``D_step`` is :func:`build_d_step` applied to
    this batch's own (decoder-episode) teammate stream, and ``D_epi`` is the
    ``epi_mate_*`` fields :func:`build_d_epi` already built (``C`` past
    episodes against the same teammate, Appendix C) -- the same cross-episode
    context TAO's encoder cross-attends to, fed here as a prepended block
    through the shared backbone instead.

    ``act_loss``/``oppo_pi_loss`` are the categorical negative log-likelihood
    of the ego action and the teammate action respectively; ``value_loss`` is
    the mean-squared error of the critic to the ego return-to-go (the best
    response's RTG on best-response data). All three are masked to valid
    timesteps and reported with an accuracy so a falling loss is interpretable.
    """
    d_step = build_d_step(batch)
    embedding = backbone.apply(
        params["backbone"],
        batch["ego_rtg"],
        batch["ego_obs"],
        batch["ego_actions"],
        timesteps=batch["timesteps"],
        mask=batch["mask"],
        mate_actions=d_step["mate_actions"],
        epi_state=batch["epi_mate_obs"],
        epi_mate_actions=batch["epi_mate_actions"],
        epi_mask=batch["epi_mask"],
        train=train,
        rngs=rngs,
    )
    action_logits, mate_logits, value = heads.apply(params["heads"], embedding)
    action_logits = mask_logits(action_logits, batch["ego_avail"])
    mate_logits = mask_logits(mate_logits, batch["mate_avail"])

    mask = batch["mask"].astype(jnp.float32)
    denom = jnp.maximum(mask.sum(), 1.0)

    act_loss = safe_cross_entropy(action_logits, batch["ego_actions"], mask)
    act_loss = (act_loss * mask).sum() / denom
    action_acc = _masked_accuracy(action_logits, batch["ego_actions"], mask)

    oppo_pi_loss = safe_cross_entropy(mate_logits, batch["mate_actions"], mask)
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

    def prepare(self, dataset, logger, *, rng, np_rng, num_seeds: int = 1) -> None:
        super().prepare(dataset, logger, rng=rng, np_rng=np_rng, num_seeds=num_seeds)
        # D_step is read from the decoder window's OWN episode by construction
        # (never sampled elsewhere -- see build_d_step), and that episode's ego
        # actions are the actor's BC imitation target. So the decoder window
        # pool must be restricted to episodes where the ego seat IS teammate
        # k's own dedicated best response -- roster-index identity equality,
        # not a quality/epsilon threshold -- or a mismatched pairing's ego
        # actions pollute the target. D_epi (below) stays unrestricted: it
        # needs teammate-behaviour diversity across ANY ego, the opposite
        # requirement -- see build_d_epi/teammate_episode_ids.
        matched_episodes = identity_matched_episode_ids(dataset.batch, dataset.ego, dataset.mate)
        episode_ids, ts = enumerate_windows(
            dataset.batch, h=self.config.context_length, stride=self.config.stride
        )
        keep = np.isin(episode_ids, matched_episodes)
        self._pool = (episode_ids[keep], ts[keep])

    def _sample_batch(self, _step):
        episode_ids, ts = self._pool
        h = self.config.context_length
        batch_size = self.config.batch_size
        rows = self.np_rng.choice(
            len(episode_ids), size=batch_size, replace=len(episode_ids) < batch_size
        )
        row_episode_ids, row_ts = episode_ids[rows], ts[rows]

        decoder = window_slice_batch(
            self.dataset.batch, row_episode_ids, row_ts, h, _DECODER_FIELDS, mate=self.dataset.mate
        )
        teammate_ids = self.dataset.batch.member_ids[row_episode_ids, self.dataset.mate]
        d_epi = build_d_epi(
            self.dataset,
            teammate_ids,
            row_episode_ids,
            self.np_rng,
            H=self.config.epi_length,
            C=self.config.context_trajectories,
        )

        batch = dict(decoder)
        batch["teammate_id"] = np.asarray(teammate_ids, dtype=np.int32)
        batch["epi_mate_obs"] = d_epi["mate_obs"]
        batch["epi_mate_actions"] = d_epi["mate_actions"]
        batch["epi_mask"] = d_epi["mask"]
        return to_jax(normalize_window_batch(batch, self.dataset.norm))

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

        d_step = build_d_step(init_batch)

        def init_one(key):
            k1, k2 = jax.random.split(key)
            backbone_params = self.agent.backbone.init(
                k1,
                init_batch["ego_rtg"],
                init_batch["ego_obs"],
                init_batch["ego_actions"],
                timesteps=init_batch["timesteps"],
                mask=init_batch["mask"],
                mate_actions=d_step["mate_actions"],
                epi_state=init_batch["epi_mate_obs"],
                epi_mate_actions=init_batch["epi_mate_actions"],
                epi_mask=init_batch["epi_mask"],
            )
            embedding = self.agent.backbone.apply(
                backbone_params,
                init_batch["ego_rtg"],
                init_batch["ego_obs"],
                init_batch["ego_actions"],
                timesteps=init_batch["timesteps"],
                mask=init_batch["mask"],
                mate_actions=d_step["mate_actions"],
                epi_state=init_batch["epi_mate_obs"],
                epi_mate_actions=init_batch["epi_mate_actions"],
                epi_mask=init_batch["epi_mask"],
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
