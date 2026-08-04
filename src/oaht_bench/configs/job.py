"""Job configs: the top-level object a single experiment JSON deserializes into.

One JSON file fully determines one run, and the CLI routes on ``job_type``. The
file is also the provenance record — its :meth:`~BaseConfig.content_hash` is what
gets written into the artifact the run produces, so a dataset or a checkpoint can
always be traced back to the exact configuration that made it.

Job models here declare *what* to run. The code that executes them lives in the
corresponding package (``oaht_bench.teammate_generation``, ``oaht_bench.data``,
...) so that importing a config never pulls in JAX.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from oaht_bench.configs.base import VersionedConfig
from oaht_bench.configs.env import EnvConfig

Generator = Literal["fcp", "comedi", "brdiv", "lbrdiv"]


class JobBase(VersionedConfig):
    """Fields every job carries."""

    label: str = Field(
        description="Human-readable run label. Appears in output paths alongside "
        "the config hash, which is what actually disambiguates runs."
    )
    seed: int = Field(default=0, description="Master seed for the run.")
    output_dir: str = Field(
        default="results",
        description="Root for run outputs. The run writes to "
        "<output_dir>/<job_type>/<label>-<config hash>/.",
    )

    def run_dir(self) -> str:
        """Output directory for this run, disambiguated by config hash."""
        return f"{self.output_dir}/{self.job_type}/{self.label}-{self.short_hash()}"


class TeammateGenerationJob(JobBase):
    """Train a teammate population with one of the four generators (§7.1)."""

    job_type: Literal["teammate_generation"] = "teammate_generation"
    env: EnvConfig
    generator: Generator
    population_size: int = Field(gt=0, description="PARTNER_POP_SIZE.")
    total_timesteps: float = Field(gt=0)
    num_envs: int = Field(gt=0)
    learning_rate: float = Field(gt=0, default=5e-4)
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Generator-specific overrides passed through to jax-aht's "
        "algorithm config (e.g. LAGRANGE_LR, XP_LOSS_WEIGHTS, COMEDI_ALPHA). "
        "Typed per-generator models supersede this once the sweeps are defined.",
    )


class DatasetCollectionJob(JobBase):
    """Roll out a trained population into the two dataset views (§4.1)."""

    job_type: Literal["dataset_collection"] = "dataset_collection"
    env: EnvConfig
    population_path: str = Field(description="Directory of teammate checkpoints.")
    variant: Literal["random", "medium", "expert", "replay_full", "mixed"] = Field(
        description="D4RL-style data regime (§4.3). 'replay_full' is deliberately "
        "not D4RL's 'medium-replay', which stops at medium performance."
    )
    num_episodes: int = Field(gt=0)
    mirror_trajectories: bool = Field(
        default=False,
        description="TAGET-style trajectory mirroring (§4.5). Only valid when the "
        "environment has symmetric roles; validated against env.symmetric_roles.",
    )


class TrainingJob(JobBase):
    """Train one baseline on one dataset."""

    job_type: Literal["training"] = "training"
    env: EnvConfig
    dataset_path: str
    baseline: str = Field(description="Baseline name, e.g. 'dt', 'liam', 'taget'.")
    backbone: Literal["dt", "iql", "pct_bc"] = Field(
        default="dt",
        description="Shared sequence-model backbone (§3.1). 'iql' is the "
        "backbone-sensitivity ablation.",
    )
    num_seeds: int = Field(default=3, gt=0)


class EvaluationJob(JobBase):
    """Evaluate trained baselines against held-out teammates (§8)."""

    job_type: Literal["evaluation"] = "evaluation"
    env: EnvConfig
    checkpoint_paths: list[str] = Field(min_length=1)
    heldout_population_path: str
    num_episodes: int = Field(
        default=1200,
        gt=0,
        description="Per teammate. The literature runs 50-2500; low budgets "
        "produce confidence intervals that overlap the baselines being beaten.",
    )
    seen_unseen_ratios: list[str] = Field(
        default_factory=lambda: ["10:0", "10:5", "10:10", "5:10", "0:10"],
        description="Graded distribution shift (§8), following OMIS.",
    )


#: Discriminated union. ``job_type`` selects the model, so one JSON file with one
#: extra key routes the entire CLI.
JobConfig = Annotated[
    TeammateGenerationJob | DatasetCollectionJob | TrainingJob | EvaluationJob,
    Field(discriminator="job_type"),
]
