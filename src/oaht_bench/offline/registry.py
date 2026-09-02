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

    def prepare(self, dataset, logger, *, rng, np_rng) -> None:
        """Inject the training data and infrastructure used by both stages.

        Takes the whole :class:`~oaht_bench.dataset.dataset.Dataset`. The stages
        draw from it directly -- ``self.dataset.windows`` / ``self.dataset.index``
        fed to the samplers in :mod:`oaht_bench.dataset.sampler` -- rather than the
        policy holding its own copies.
        """
        self.dataset = dataset
        self.logger = logger
        self.rng = rng
        self.np_rng = np_rng

    # --- training (baseline-specific; implemented by subclasses) ----------

    def train_stage_1(self):
        """Train the teammate representation; returns the stage-1 parameters."""
        raise NotImplementedError

    def train_stage_2(self, stage1_params):
        """Train the policy against the frozen stage-1 representation."""
        raise NotImplementedError

    # --- shared machinery -------------------------------------------------

    def _run_stage(self, loss_fn, params, batch_fn, *, learning_rate, steps, prefix):
        """Optimise one stage with the shared AdamW-with-warmup loop.

        Splits a fresh key off ``self.rng`` so the two stages do not share
        randomness, mirroring the runner's ``s1_rng``/``s2_rng`` split.
        """
        self.rng, stage_rng = jax.random.split(self.rng)
        return train(
            loss_fn,
            params,
            batch_fn,
            optimizer=get_optimizer(self.config, learning_rate, steps),
            steps=steps,
            rng=stage_rng,
            logger=self.logger,
            prefix=prefix,
            log_every=self.config.log_every,
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
