from typing import Any

from pydantic import Field

from oaht_bench.configs.base import BaseConfig


class RLTeammateConfig(BaseConfig):
    algo: str
    ckpt_path: str
    use_log_wrapper: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)


class PopulationConfig(BaseConfig):
    algo: str = Field(
        description="Teammate generation algorithm used for generating the population"
    )
    ckpt_path: str = Field(description="Path to the population orbax checkpoint directory")
