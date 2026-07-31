"""Environment construction from validated configs.

Thin by design: jax-aht already exposes a uniform interface across LBF,
Overcooked-v1 and Hanabi (``reset``, ``step``, ``get_avail_actions``, ``agents``).
What this layer adds is that the environment is built from a validated
:data:`~oaht_bench.configs.env.EnvConfig` rather than a loose kwargs dict, so a
misconfiguration fails at config load with a message naming the field instead of
as an opaque assertion from inside Jumanji on the first reset.
"""

from __future__ import annotations

from typing import Any

from oaht_bench.configs.env import EnvConfig, get_preset, preset_names

__all__ = ["make", "get_preset", "preset_names"]


def make(cfg: EnvConfig | str) -> Any:
    """Instantiate the environment described by ``cfg``.

    Accepts a config or the name of a canonical preset. Imports jax-aht lazily so
    that config validation and metadata inspection do not pay JAX's import cost.
    """
    from envs import make_env  # jax-aht, top-level module

    if isinstance(cfg, str):
        cfg = get_preset(cfg)
    return make_env(cfg.env_name, cfg.env_kwargs())
