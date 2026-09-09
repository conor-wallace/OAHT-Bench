"""The baseline-trainer contract and its registry.

Every offline baseline is a :class:`BaseAhtTrainer`: it owns its model, its
per-stage batch sampling, and its losses. Inference is not its concern -- each
trainer builds a :class:`~oaht_bench.models.return_conditioned_agent.ReturnConditionedAgent`
(via ``build_model``) that acts, and evaluation drives that agent directly. The
runner resolves the concrete trainer from the config's ``network.architecture`` via
:func:`get_trainer` and then drives training generically, so adding a baseline is
subclassing rather than extending an ``if/elif``.

``obs_dim`` and ``action_dim`` are resolved onto ``config.network`` from the
dataset before a trainer is constructed, so a trainer is built from the config
alone -- ``build_model`` never needs the environment.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from oaht_bench.configs.job import OfflineTrainingConfig
from oaht_bench.offline.training import get_optimizer, train


class BaseAhtTrainer:
    """Interface the runner drives to train a baseline.

    Lifecycle: ``build_model()`` (pure config, also constructs the acting agent)
    -> ``prepare(...)`` (inject data and logging) -> ``train_stage_1()`` ->
    ``train_stage_2(stage1_params)``, returning the parameters
    (``{"stage1": ..., "stage2": ...}``) the agent then acts with at evaluation.
    """

    #: The ``architecture`` discriminator this trainer answers to.
    name: str

    def __init__(self, config: OfflineTrainingConfig):
        self.config = config

    # --- construction -----------------------------------------------------

    def build_model(self) -> None:
        """Build the flax modules from ``self.config`` (including resolved dims)."""
        raise NotImplementedError

    def prepare(self, dataset, logger, *, rng, np_rng, num_seeds: int = 1) -> None:
        """Inject the training data and infrastructure used by both stages.

        Takes the whole :class:`~oaht_bench.dataset.dataset.Dataset`. The stages
        draw from it directly -- ``self.dataset.windows`` / ``self.dataset.index``
        fed to the samplers in :mod:`oaht_bench.dataset.sampler` -- rather than the
        policy holding its own copies.

        ``num_seeds`` trains that many independently-initialised seeds in parallel
        (vmapped over a leading seed axis); the returned parameter trees then carry
        that axis. 1 keeps the single-seed behaviour.
        """
        self.dataset = dataset
        self.logger = logger
        self.rng = rng
        self.np_rng = np_rng
        self.num_seeds = num_seeds

    def _init_params(self, init_one, key, *per_seed):
        """Build single- or N-seed initial params from a per-seed ``init_one``.

        ``init_one(key, *per_seed_slice)`` initialises one seed. With
        ``num_seeds > 1`` it is vmapped over ``num_seeds`` split keys; any
        ``per_seed`` arguments (e.g. a stage-1 representation the init depends on)
        must already carry a leading seed axis and are mapped in lockstep.
        """
        if self.num_seeds == 1:
            return init_one(key, *per_seed)
        keys = jax.random.split(key, self.num_seeds)
        return jax.vmap(init_one)(keys, *per_seed)

    # --- training (baseline-specific; implemented by subclasses) ----------

    def train_stage_1(self):
        """Train the teammate representation; returns the stage-1 parameters."""
        raise NotImplementedError

    def train_stage_2(self, stage1_params):
        """Train the policy against the frozen stage-1 representation."""
        raise NotImplementedError

    # --- shared machinery -------------------------------------------------

    def _run_stage(self, loss_fn, params, batch_fn, *, learning_rate, steps, prefix, frozen=None):
        """Optimise one stage with the shared AdamW-with-warmup loop.

        Splits a fresh key off ``self.rng`` so the two stages do not share
        randomness, mirroring the runner's ``s1_rng``/``s2_rng`` split.

        ``loss_fn(params, batch, rngs, frozen)`` -- ``frozen`` is a per-seed pytree
        the loss reads but does not train (a frozen stage-1 representation), or
        ``None``. With ``num_seeds > 1`` the per-step ``batch_fn`` is drawn once per
        seed and stacked, so each seed sees an independent minibatch stream.
        """
        num_seeds = getattr(self, "num_seeds", 1)
        self.rng, stage_rng = jax.random.split(self.rng)

        if num_seeds > 1:
            def seeded_batch(i):
                per_seed = [batch_fn(i) for _ in range(num_seeds)]
                return jax.tree.map(lambda *xs: jnp.stack(xs), *per_seed)
        else:
            seeded_batch = batch_fn

        return train(
            loss_fn,
            params,
            seeded_batch,
            optimizer=get_optimizer(self.config, learning_rate, steps),
            steps=steps,
            rng=stage_rng,
            logger=self.logger,
            prefix=prefix,
            log_every=self.config.log_every,
            num_seeds=num_seeds,
            frozen=frozen,
        )


def get_trainer(config: OfflineTrainingConfig) -> type[BaseAhtTrainer]:
    """Resolve the trainer class for a config's ``network.architecture``.

    Imports lazily so the registry does not depend on every baseline module (and
    so a baseline can import :class:`BaseAhtTrainer` from here without a cycle).
    """
    architecture = config.network.architecture
    if architecture == "liam":
        from oaht_bench.offline.liam import LiamTrainer

        return LiamTrainer
    if architecture == "meliba":
        from oaht_bench.offline.meliba import MelibaTrainer

        return MelibaTrainer
    if architecture == "omis":
        from oaht_bench.offline.omis import OmisTrainer

        return OmisTrainer
    if architecture == "tao":
        from oaht_bench.offline.tao import TaoTrainer

        return TaoTrainer
    if architecture == "bc":
        from oaht_bench.offline.bc import BcTrainer

        return BcTrainer
    raise NotImplementedError(f"no BaseAhtTrainer is registered for architecture {architecture!r}.")
