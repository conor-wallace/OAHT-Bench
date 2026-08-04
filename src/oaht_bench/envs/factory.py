"""Config-driven environment construction.

Wraps the absorbed ``make_env`` dispatcher so environments are built from a
validated :data:`~oaht_bench.configs.env.EnvConfig` rather than a loose kwargs
dict. A misconfiguration then fails at config load, naming the field, instead of
surfacing as an opaque assertion from inside Jumanji on the first reset.
"""

from __future__ import annotations

from typing import Any

from oaht_bench.configs.env import EnvConfig, get_preset


def make(cfg: EnvConfig | str) -> Any:
    """Instantiate the environment described by ``cfg``.

    Accepts a config or the name of a canonical preset.
    """
    from oaht_bench.envs.make_env import make_env

    if isinstance(cfg, str):
        cfg = get_preset(cfg)
    return make_env(cfg.env_name, cfg.env_kwargs())
