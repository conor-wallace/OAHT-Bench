"""%BC — filtered behaviour cloning, the "no modeling module" floor (§6).

Every other baseline in this package contributes a `history -> z` module: an
encoder, a set of policy embeddings, an in-context imitator. %BC contributes
nothing — it is the shared backbone (:mod:`oaht_bench.offline.backbone`,
§3.1) trained directly on the ego stream, so a comparison against it answers
"did modelling the teammate help at all," which none of the other nine
baselines in the inventory can answer on their own.

**Filtered, not vanilla.** The "%" is a data-selection knob, not a modelling
one: train only on the top ``top_return_quantile`` fraction of episodes by
ego return (`JAX-CORL`'s convention). ``top_return_quantile=1.0`` — the
default — keeps every episode, which is plain BC on the whole dataset; the
filter is this baseline's one option, not its definition.

**One stage, not two.** :func:`oaht_bench.offline.runner.run` always calls
``train_stage_1`` then ``train_stage_2`` — every other baseline needs both, so
the loop does not special-case a baseline that does not. :meth:`PctBcPolicy.
train_stage_1` returns immediately with nothing to show for it, rather than
running the shared step loop for zero steps to the same effect: an empty
``Stage1/`` block in the metrics would look like a bug, not a design choice.

**The caveat this baseline cannot dilute.** :mod:`oaht_bench.offline`'s module
docstring already flags it for LIAM and TAO: the ego stream in a collected
dataset is another population member's play, not a best response to the
teammate on the other end of it, so cloning it trains toward population-average
behaviour. LIAM and TAO at least condition the policy on *something* inferred
about the teammate; %BC has no mechanism to condition on at all, so it clones
population-average play more directly than either. That is exactly why it is
the floor rather than a competitor — the gap between %BC and everything else
in the inventory is the thing the other nine baselines exist to open up.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from oaht_bench.dataset.windows import Windows
from oaht_bench.offline.backbone import DecisionTransformer
from oaht_bench.offline.registry import BaseAhtPolicy
from oaht_bench.offline.utils import mask_logits, masked_accuracy, to_jax

#: %BC reads only the ego stream — no teammate fields, unlike
#: ``utils.WINDOW_BATCH_KEYS``, which every modelling baseline needs.
_BATCH_KEYS = ("ego_obs", "ego_actions", "ego_rtg", "ego_avail", "timesteps", "mask")


def _filter_by_return(windows: Windows, top_return_quantile: float) -> np.ndarray:
    """Indices of windows whose *episode* is in the top ``top_return_quantile``
    of episodes by ego return.

    Filters by episode, not by window: an episode contributes several
    overlapping windows (``stride`` < ``context_length``), and ranking windows
    directly would let a long episode's redundant fragments dominate the
    quantile computation over a short one.
    """
    if top_return_quantile >= 1.0:
        return np.arange(len(windows))
    episode_ids, first = np.unique(windows.episode_id, return_index=True)
    per_episode_return = windows.episode_return[first]
    threshold = np.quantile(per_episode_return, 1.0 - top_return_quantile)
    keep = episode_ids[per_episode_return >= threshold]
    idx = np.flatnonzero(np.isin(windows.episode_id, keep))
    if idx.size == 0:
        raise ValueError(
            f"top_return_quantile={top_return_quantile} kept 0 windows out of "
            f"{len(windows)} -- the threshold {threshold} excluded every episode."
        )
    return idx


class PctBcNetwork(nn.Module):
    """The shared backbone, unmodified: no embedding, no context, no
    conditioning on anything about the teammate."""

    action_dim: int
    hidden_dim: int = 32
    dropout: float = 0.1

    @nn.compact
    def __call__(self, rtg, obs, actions, *, timesteps, mask=None, train: bool = False):
        logits, _ = DecisionTransformer(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            use_cross_attention=False,
            dropout=self.dropout,
        )(rtg, obs, actions, timesteps=timesteps, mask=mask, train=train)
        return logits


def pct_bc_loss(params, network, batch, *, rngs=None, train: bool = True):
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
    bc = optax.softmax_cross_entropy_with_integer_labels(logits, batch["ego_actions"])
    acc = masked_accuracy(logits, batch["ego_actions"], mask)
    bc = (bc * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    return bc, {"loss": bc, "bc": bc, "action_accuracy": acc}


class PctBcPolicy(BaseAhtPolicy):
    """%BC on the two-stage contract, with stage 1 empty by design."""

    name = "pct_bc"

    def build_model(self) -> None:
        net = self.config.network
        if net.obs_dim is None or net.action_dim is None:
            raise ValueError(
                "obs_dim/action_dim are unresolved on the network config; the "
                "runner must resolve them from the dataset before build_model()."
            )
        self.network = PctBcNetwork(
            action_dim=net.action_dim, hidden_dim=net.hidden_dim, dropout=net.dropout
        )

    def prepare(self, windows, index, logger, *, rng, np_rng) -> None:
        super().prepare(windows, index, logger, rng=rng, np_rng=np_rng)
        self._filtered_idx = _filter_by_return(windows, self.config.network.top_return_quantile)
        if len(self._filtered_idx) < self.config.stage2_batch_size:
            raise ValueError(
                f"top_return_quantile={self.config.network.top_return_quantile} keeps "
                f"only {len(self._filtered_idx)} windows, fewer than "
                f"stage2_batch_size={self.config.stage2_batch_size}. Raise the "
                f"quantile or lower the batch size."
            )

    def _sample_batch(self, _step):
        """Sample only from the return-filtered window set. Each call draws a
        fresh minibatch, so the step index is ignored, matching every other
        ego-history baseline's sampler."""
        idx = self.np_rng.choice(
            self._filtered_idx, size=self.config.stage2_batch_size, replace=False
        )
        return to_jax({k: getattr(self.windows, k)[idx] for k in _BATCH_KEYS})

    def train_stage_1(self):
        """No representation to learn — %BC's entire point is having none."""
        return {}

    def train_stage_2(self, stage1_params):
        del stage1_params  # unused: nothing from stage 1 to condition on.
        init_batch = self._sample_batch(0)
        self.rng, k = jax.random.split(self.rng)
        params = self.network.init(
            k,
            init_batch["ego_rtg"],
            init_batch["ego_obs"],
            init_batch["ego_actions"],
            timesteps=init_batch["timesteps"],
            mask=init_batch["mask"],
        )

        def loss(p, b, rngs):
            return pct_bc_loss(p, self.network, b, rngs=rngs)

        return self._run_stage(
            loss,
            params,
            self._sample_batch,
            learning_rate=self.config.stage2_learning_rate,
            steps=self.config.stage2_steps,
            prefix="Stage2",
        )

    def act(self, params, rtg, obs, actions, *, timesteps, mask):
        return self.network.apply(
            params["stage2"], rtg, obs, actions, timesteps=timesteps, mask=mask, train=False
        )
