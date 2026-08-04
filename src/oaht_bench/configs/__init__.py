"""Config models for OAHT-Bench.

Every controllable parameter is a validated pydantic field. One JSON file
deserializes into one :data:`~oaht_bench.configs.job.JobConfig` and fully
determines one run.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from oaht_bench.configs.base import SCHEMA_VERSION, BaseConfig, VersionedConfig
from oaht_bench.configs.env import (
    PRESETS,
    EnvConfig,
    HanabiConfig,
    LbfConfig,
    OvercookedV1Config,
    Tier,
    get_preset,
    preset_names,
)
from oaht_bench.configs.job import (
    DatasetCollectionJob,
    EvaluationJob,
    JobConfig,
    TeammateGenerationJob,
    TrainingJob,
)

#: Validates and dispatches a raw payload to the right job model via ``job_type``.
JOB_ADAPTER: TypeAdapter[JobConfig] = TypeAdapter(JobConfig)

#: Same, for a bare environment config.
ENV_ADAPTER: TypeAdapter[EnvConfig] = TypeAdapter(EnvConfig)


def load_job(path: str | Path) -> JobConfig:
    """Load and validate an experiment config from a JSON file.

    Errors name the offending field and file rather than surfacing as a failure
    partway through a training run.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such config file: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not valid JSON — {e}") from e
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(payload).__name__}")
    if "job_type" not in payload:
        raise ValueError(
            f"{path}: missing 'job_type'. Expected one of: teammate_generation, "
            f"dataset_collection, training, evaluation."
        )
    return JOB_ADAPTER.validate_python(payload)


__all__ = [
    "SCHEMA_VERSION",
    "BaseConfig",
    "VersionedConfig",
    "EnvConfig",
    "LbfConfig",
    "OvercookedV1Config",
    "HanabiConfig",
    "Tier",
    "PRESETS",
    "get_preset",
    "preset_names",
    "JobConfig",
    "TeammateGenerationJob",
    "DatasetCollectionJob",
    "TrainingJob",
    "EvaluationJob",
    "JOB_ADAPTER",
    "ENV_ADAPTER",
    "load_job",
]
