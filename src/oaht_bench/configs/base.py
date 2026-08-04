"""Foundations for every OAHT-Bench config model.

The benchmark's reproducibility claim rests on one idea: **a single JSON file
fully determines a run**, and that same file is the provenance record stored
alongside whatever the run produced. These base classes enforce the properties
that idea needs — configs are immutable, reject unknown fields, and hash to a
stable content digest that survives across processes and machines.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Bumped when a config model changes in a way that makes older JSON files
#: uninterpretable. Configs record the version they were written against so a
#: stale file fails loudly instead of loading with silently different meaning.
SCHEMA_VERSION = 1


class BaseConfig(BaseModel):
    """Immutable, strictly-validated config model.

    ``extra="forbid"`` is the important setting. jax-aht reads several kwargs with
    ``dict.get(key, default)``, so a misspelled field would otherwise fall through
    to a silent default — you would collect an entire dataset against the wrong
    environment and never see an error. Forbidding unknown fields turns that into
    a load-time failure naming the offending key.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        validate_default=True,
    )

    def canonical_json(self) -> str:
        """Deterministic JSON rendering, used for hashing and on-disk records."""
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def content_hash(self) -> str:
        """Stable SHA-256 of the config's content.

        Python's builtin ``hash`` is salted per process and undefined for models
        holding lists, so it cannot be used for provenance. This digest is written
        into dataset metadata (§4.2) so an artifact can be traced to the exact
        configuration that produced it.
        """
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def short_hash(self, length: int = 12) -> str:
        """First ``length`` characters of :meth:`content_hash`, for run directories."""
        return self.content_hash()[:length]


class VersionedConfig(BaseConfig):
    """A config that is serialized to disk and therefore needs a schema version."""

    schema_version: int = Field(
        default=SCHEMA_VERSION,
        description="Config schema version this file was written against.",
    )

    @field_validator("schema_version")
    @classmethod
    def _reject_future_versions(cls, v: int) -> int:
        if v > SCHEMA_VERSION:
            raise ValueError(
                f"Config declares schema_version {v}, but this build understands at "
                f"most {SCHEMA_VERSION}. Upgrade oaht-bench rather than editing the "
                f"version down — the file may use fields this build would silently "
                f"drop."
            )
        if v < 1:
            raise ValueError(f"schema_version must be >= 1, got {v}")
        return v

    @classmethod
    def from_json_file(cls, path: str | Path) -> Self:
        """Load and validate a config from a JSON file."""
        path = Path(path)
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: not valid JSON — {e}") from e
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected a JSON object, got {type(payload).__name__}")
        return cls.model_validate(payload)

    def to_json_file(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the config as human-editable JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = self.model_dump(mode="json")
        path.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n")
        return path
