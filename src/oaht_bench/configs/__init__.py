"""Config models for OAHT-Bench.

Every controllable parameter is a validated pydantic field. One JSON file
deserializes into one :data:`~oaht_bench.configs.job.JobConfig` and fully
determines one run.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    AnyJob,
    DatasetCollectionJob,
    EvaluationJob,
    JobConfig,
    TeammateGenerationJob,
    TrainingJob,
)




def load_job(path: str | Path) -> AnyJob:
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
    return validate_job(payload, source=str(path))


def validate_job(payload: dict, *, source: str = "<dict>") -> AnyJob:
    """Validate an in-memory payload and dispatch it to the right job model.

    Separate from :func:`load_job` so callers holding a payload already — a sweep
    generator emitting configs programmatically, for instance — get the same
    checks and the same error messages without writing a file first.
    """
    if "job" not in payload:
        if "job_type" in payload:
            # Without this, ``extra="forbid"`` reports every job field as an
            # unexpected key, which buries the actual mistake.
            raise ValueError(
                f"{source}: job fields are at the top level, but they belong under "
                f'a "job" key: {{"job": {{"job_type": "{payload["job_type"]}", ...}}}}'
            )
        raise ValueError(
            f"{source}: missing 'job'. A config file holds one job under that key, "
            f"with a 'job_type' of teammate_generation, dataset_collection, "
            f"training, or evaluation."
        )
    return JobConfig.model_validate(payload).job


def save_job(job: AnyJob, path: str | Path, *, indent: int = 2, minimal: bool = False) -> Path:
    """Write a job as a loadable config file.

    Wraps the job under the ``job`` key so the result round-trips through
    :func:`load_job`. Writing ``job.to_json_file`` directly would emit the job's
    own fields at the top level, which no longer loads.
    """
    return JobConfig(job=job).to_json_file(path, indent=indent, minimal=minimal)


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
    "AnyJob",
    "TeammateGenerationJob",
    "DatasetCollectionJob",
    "TrainingJob",
    "EvaluationJob",
    "validate_job",
    "save_job",
    "load_job",
]
