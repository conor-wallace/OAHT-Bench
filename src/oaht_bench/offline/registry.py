"""The baseline-policy contract and its registry.

Every offline baseline is a :class:`BaseAhtPolicy`: it owns its model, its
per-stage batch sampling, its losses, and its inference (:meth:`act`). The runner
resolves the concrete class from the config's ``network.architecture`` via
:func:`get_policy` and then drives training and evaluation generically, so adding
a baseline is subclassing rather than extending an ``if/elif``.

``obs_dim`` and ``action_dim`` are resolved onto ``config.network`` from the
dataset before a policy is constructed, so a policy is built from the config
alone -- ``build_model`` never needs the environment.
"""

from __future__ import annotations

import jax

from oaht_bench.configs.job import OfflineTrainingConfig
from oaht_bench.offline.training import get_optimizer, train


class BaseAhtPolicy:
    """Interface the runner drives for training and evaluation.

    Lifecycle: ``build_model()`` (pure config) -> ``prepare(...)`` (inject data
    and logging) -> ``train_stage_1()`` -> ``train_stage_2(stage1_params)`` ->
    ``act(params, ...)`` at evaluation, where ``params`` is
    ``{"stage1": ..., "stage2": ...}``.
    """

    #: The ``architecture`` discriminator this policy answers to.
    name: str

    def __init__(self, config: OfflineTrainingConfig):
        self.config = config

    # --- construction -----------------------------------------------------

    def build_model(self) -> None:
        """Build the flax modules from ``self.config`` (including resolved dims)."""
        raise NotImplementedError

    def prepare(self, windows, index, logger, *, rng, np_rng) -> None:
        """Inject the training data and infrastructure used by both stages."""
        self.windows = windows
        self.index = index
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

    # --- inference --------------------------------------------------------

    def act(self, params, rtg, obs, actions, *, timesteps, mask):
        """Ego action logits for one batch of windows, used by the eval rollout.

        ``params`` carries both stages (``{"stage1", "stage2"}``). This is the
        forward pass the runner's evaluation loop calls once per environment step.
        """
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


def get_policy(config: OfflineTrainingConfig) -> type[BaseAhtPolicy]:
    """Resolve the policy class for a config's ``network.architecture``.

    Imports lazily so the registry does not depend on every baseline module (and
    so a baseline can import :class:`BaseAhtPolicy` from here without a cycle).
    """
    architecture = config.network.architecture
    if architecture == "liam":
        from oaht_bench.offline.liam.model import LiamPolicy

        return LiamPolicy
    if architecture == "meliba":
        from oaht_bench.offline.meliba import MelibaPolicy

        return MelibaPolicy
    if architecture == "omis":
        from oaht_bench.offline.omis import OmisPolicy

        return OmisPolicy
    if architecture == "tao":
        from oaht_bench.offline.tao import TaoPolicy

        return TaoPolicy
    raise NotImplementedError(f"no BaseAhtPolicy is registered for architecture {architecture!r}.")
