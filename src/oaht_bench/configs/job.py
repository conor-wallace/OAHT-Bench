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

from pydantic import Field, model_validator

from oaht_bench.configs.base import BaseConfig, VersionedConfig
from oaht_bench.configs.env import EnvConfig
from oaht_bench.configs.teammate_gen import GeneratorConfig

#: The benchmark's baseline roster (§12.9). Thirteen entries in four groups:
#: floors, a reference row, a ceiling, the learning-history family, and the
#: trajectory-view family. Declaring it here means a typo fails at config load
#: rather than after a training run dispatches to nothing.
BaselineName = Literal[
    # floors and reference
    "random",
    "pct_bc",
    "prompt_dt",
    # ceiling: privileged teammate model, bounds how much headroom modeling has
    "oracle",
    # learning-history family
    "ad",
    "dpt",
    "amago_offline",
    "hybrid_ad",
    # trajectory-view family
    "liam",
    "meliba",
    "tao",
    "omis",
    "taget",
]


class LoggingConfig(BaseConfig):
    """Where run metrics go. Weights & Biases is opt-in.

    A benchmark must run identically for someone with no wandb account, and a
    config carrying someone else's ``entity`` must never publish there by
    accident. Metrics always land in ``<run_dir>/metrics.jsonl`` regardless.
    """

    use_wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    verbose: bool = False

    @model_validator(mode="after")
    def _wandb_needs_a_project(self) -> LoggingConfig:
        if self.use_wandb and not self.wandb_project:
            raise ValueError("use_wandb is set but wandb_project is empty.")
        return self


class JobBase(VersionedConfig):
    """Fields every job carries."""

    label: str = Field(
        description="Human-readable run label. Appears in output paths alongside "
        "the config hash, which is what actually disambiguates runs."
    )
    seed: int = Field(default=0, description="Master seed for the run.")
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
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
    generator: GeneratorConfig
    evaluation_episodes: int = Field(
        default=20,
        gt=0,
        description="Episodes per ordered pair in the post-training cross-play "
        "evaluation. Cost is population_size^2 x this, so it is the dial for the "
        "accuracy/time trade-off when scoring a sweep.",
    )
    evaluation_greedy: bool = Field(
        default=False,
        description="Take argmax actions in the cross-play evaluation instead of "
        "sampling. Defaults to sampling, which is both how training measures "
        "returns and how the released population is used downstream. Greedy is "
        "kept for diagnostics but is not the benchmark's measurement: in a "
        "symmetric coordination task two argmax policies are perfectly "
        "correlated and deadlock. On LBF 12x12 that held every episode to the "
        "100-step limit and cut food collected from 75% to 25%, reporting 0.11 "
        "for a population whose training curve read 0.40. It also discards the "
        "policy entropy, which makes entropy_coef unmeasurable in a sweep.",
    )

    def to_jax_aht_cfg(self) -> dict[str, Any]:
        """Build the nested dict the absorbed training code expects.

        jax-aht drops Hydra at the boundary — its runners call
        ``OmegaConf.to_container`` and then pass a plain dict to
        ``run_fcp``/``run_comedi``/``run_brdiv``/``run_lbrdiv`` — so this hands
        off directly, with no Hydra and no generated YAML.
        """
        task = self.env.task_config()
        algorithm = self.generator.to_algorithm_dict()
        # The generators read env fields from inside the algorithm block too,
        # because upstream Hydra interpolated them there.
        algorithm.update(
            {
                "ENV_NAME": task["ENV_NAME"],
                "ENV_KWARGS": task["ENV_KWARGS"],
                "ROLLOUT_LENGTH": task["ROLLOUT_LENGTH"],
            }
        )
        return {
            **task,
            "task": task,
            # Where the absorbed training code writes checkpoints. Upstream read
            # this from Hydra's global; we pass it explicitly.
            "run_dir": self.run_dir(),
            "algorithm": algorithm,
            "label": self.label,
            "name": f"{task['TASK_NAME']}/{algorithm['ALG']}/{self.label}",
            "train_ego": False,
            "run_heldout_eval": False,
            "logger": {
                "verbose": self.logging.verbose,
                "log_train_out": True,
                "log_eval_out": True,
            },
            "local_logger": {"save_train_out": True, "save_eval_out": True},
        }


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
    baseline: BaselineName = Field(description="Which baseline to train (§6).")
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


#: The union itself, for annotating anything that holds a concrete job.
AnyJob = Annotated[
    TeammateGenerationJob | DatasetCollectionJob | TrainingJob | EvaluationJob,
    Field(discriminator="job_type"),
]


class JobConfig(VersionedConfig):
    """An experiment config file: exactly one job.

    The union is a *field* rather than the model itself because
    ``Annotated[A | B, ...]`` is a typing alias — an assignment binding a name to
    a typing object, not a class — so it has no ``model_validate``. Wrapping it
    gives the normal pydantic interface, with ``job_type`` still selecting which
    member is validated::

        job = JobConfig.model_validate(payload).job

    Keeping the job under a named key also leaves the top level free for
    file-scoped metadata later without colliding with any job's own fields.

    Prefer :func:`~oaht_bench.configs.load_job` and
    :func:`~oaht_bench.configs.validate_job`, which unwrap ``.job`` and turn a
    malformed config into a message naming the file and the offending key.
    """

    job: AnyJob
