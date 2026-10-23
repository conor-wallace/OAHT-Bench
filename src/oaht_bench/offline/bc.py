"""BC — behaviour cloning, the "no modeling module" floor (§6).

Every other baseline in this package contributes a `history -> z` module: an
encoder, a set of policy embeddings, an in-context imitator. BC contributes
nothing — it is the shared backbone (:mod:`oaht_bench.models.backbone`,
§3.1) trained directly on the ego stream, so a comparison against it answers
"did modelling the teammate help at all," which none of the other nine
baselines in the inventory can answer on their own.

**An optional return filter.** BC can train on the top ``top_return_quantile``
fraction of episodes by ego return (`JAX-CORL`'s filtered-BC convention).
``top_return_quantile=1.0`` — the default — keeps every episode, which is plain
BC on the whole dataset; the filter is a data-selection knob, not part of the
method's definition. Always intersected with the identity-matched-episode
filter every baseline's BC/decoder target needs (:func:`~oaht_bench.dataset.
windowing.identity_matched_episode_ids`) -- the two are independent knobs, not
alternatives to each other.

**One stage, not two.** :func:`oaht_bench.offline.runner.run` always calls
``train_stage_1`` then ``train_stage_2`` — every other baseline needs both, so
the loop does not special-case a baseline that does not. :meth:`BcTrainer.
train_stage_1` returns immediately with nothing to show for it, rather than
running the shared step loop for zero steps to the same effect: an empty
``Stage1/`` block in the metrics would look like a bug, not a design choice.

**The caveat this baseline cannot dilute.** :mod:`oaht_bench.offline`'s module
docstring already flags it for LIAM and TAO: the ego stream in a collected
dataset is another population member's play, not a best response to the
teammate on the other end of it, so cloning it trains toward population-average
behaviour. LIAM and TAO at least condition the policy on *something* inferred
about the teammate; BC has no mechanism to condition on at all, so it clones
population-average play more directly than either. That is exactly why it is
the floor rather than a competitor — the gap between BC and everything else
in the inventory is the thing the other nine baselines exist to open up.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from oaht_bench.dataset.windowing import enumerate_windows, identity_matched_episode_ids
from oaht_bench.models.bc_agent import BcAgent
from oaht_bench.offline.registry import BaseAhtTrainer
from oaht_bench.offline.utils import (
    mask_logits,
    masked_accuracy,
    safe_cross_entropy,
    sample_window_batch,
)

#: BC reads only the ego stream -- no teammate fields.
_BATCH_FIELDS = ("ego_obs", "ego_actions", "ego_rtg", "ego_avail", "timesteps", "mask")


def _filter_by_return(batch, ego: int, top_return_quantile: float) -> np.ndarray:
    """Ids of the episodes in the top ``top_return_quantile`` of episodes by
    ego return -- NOT windows, episodes: the filter is a data-selection knob
    on which trajectories exist to be windowed, not on the windows themselves.
    """
    episode_return = batch.episode_returns()[:, ego]
    if top_return_quantile >= 1.0:
        return np.arange(len(episode_return))
    threshold = np.quantile(episode_return, 1.0 - top_return_quantile)
    keep = np.flatnonzero(episode_return >= threshold)
    if keep.size == 0:
        raise ValueError(
            f"top_return_quantile={top_return_quantile} kept 0 of "
            f"{len(episode_return)} episodes -- the threshold {threshold} "
            f"excluded every one."
        )
    return keep


def bc_loss(params, network, batch, *, rngs=None, train: bool = True):
    """Plain masked behaviour cloning on the ego stream.

    Identical in shape to :func:`~oaht_bench.offline.liam.losses.
    liam_policy_loss`'s stage-2 term, minus the encoder and embedding — which
    is the whole point: the only difference between this baseline and every
    modelling one is what, if anything, is concatenated in before this loss.
    """
    logits = mask_logits(
        network.apply(
            params,
            batch["ego_rtg"],
            batch["ego_obs"],
            batch["ego_actions"],
            timesteps=batch["timesteps"],
            mask=batch["mask"],
            train=train,
            rngs=rngs,
        ),
        batch["ego_avail"],
    )
    mask = batch["mask"].astype(jnp.float32)
    bc = safe_cross_entropy(logits, batch["ego_actions"], mask)
    acc = masked_accuracy(logits, batch["ego_actions"], mask)
    bc = (bc * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    return bc, {"loss": bc, "bc": bc, "action_accuracy": acc}


class BcTrainer(BaseAhtTrainer):
    """BC on the two-stage contract, with stage 1 empty by design."""

    name = "bc"

    def build_model(self) -> None:
        # Inference is the composed BcAgent's; training reads its network.
        self.agent = BcAgent(self.config)
        self.agent.build_model()

    def prepare(self, dataset, logger, *, rng, np_rng, num_seeds: int = 1) -> None:
        super().prepare(dataset, logger, rng=rng, np_rng=np_rng, num_seeds=num_seeds)
        # BC clones the ego's own actions as the near-optimal response target,
        # same as LIAM's/OMIS's D_step -- sound only when the ego seat IS the
        # mate's own dedicated best response, or a mismatched pairing's
        # suboptimal actions pollute the target. Intersected with the
        # existing return-quantile filter: the two are independent selection
        # knobs (identity match vs. realised return), not a replacement for
        # one another.
        keep_episodes = np.intersect1d(
            _filter_by_return(dataset.batch, dataset.ego, self.config.network.top_return_quantile),
            identity_matched_episode_ids(dataset.batch, dataset.ego, dataset.mate),
        )
        episode_ids, ts = enumerate_windows(
            dataset.batch, h=self.config.context_length, stride=self.config.stride
        )
        keep = np.isin(episode_ids, keep_episodes)
        self._episode_ids, self._ts = episode_ids[keep], ts[keep]
        if len(self._episode_ids) < self.config.batch_size:
            raise ValueError(
                f"top_return_quantile={self.config.network.top_return_quantile} combined "
                f"with the identity-matched filter keeps only {len(self._episode_ids)} "
                f"windows, fewer than batch_size={self.config.batch_size}. Raise the "
                f"quantile or lower the batch size."
            )

    def _sample_batch(self, _step):
        """Sample only from the return-filtered window pool. Each call draws a
        fresh minibatch, so the step index is ignored, matching every other
        ego-history baseline's sampler."""
        return sample_window_batch(
            self.dataset,
            (self._episode_ids, self._ts),
            self.config.context_length,
            self.np_rng,
            self.config.batch_size,
            _BATCH_FIELDS,
        )

    def train_stage_1(self):
        """No representation to learn — BC's entire point is having none."""
        return {}

    def train_stage_2(self, stage1_params):
        del stage1_params  # unused: nothing from stage 1 to condition on.
        init_batch = self._sample_batch(0)
        self.rng, k = jax.random.split(self.rng)

        def init_one(key):
            return self.agent.network.init(
                key,
                init_batch["ego_rtg"],
                init_batch["ego_obs"],
                init_batch["ego_actions"],
                timesteps=init_batch["timesteps"],
                mask=init_batch["mask"],
            )

        params = self._init_params(init_one, k)

        def loss(p, b, rngs, frozen):
            return bc_loss(p, self.agent.network, b, rngs=rngs)

        return self._run_stage(
            loss,
            params,
            self._sample_batch,
            learning_rate=self.config.stage2_learning_rate,
            steps=self.config.stage2_steps,
            prefix="Stage2",
        )
