"""Structural guarantees about the package layout.

The absorbed upstreams (jax-aht, and ICRL4AHT later) both declare top-level
packages named ``agents``, ``common``, ``envs``, ``marl`` and
``teammate_generation``. Depending on both is impossible — whichever installs
second wins, silently. Absorbing them under ``oaht_bench.*`` is what makes the
two coexist, so these tests guard that boundary.
"""

from __future__ import annotations

import importlib

import pytest

COLLIDING_NAMES = ["envs", "agents", "common", "marl", "teammate_generation", "evaluation"]


@pytest.mark.parametrize("name", COLLIDING_NAMES)
def test_no_top_level_namespace_pollution(name: str):
    """These names must not be importable at top level.

    If one becomes importable, an upstream has been added as a dependency again
    and the collision is back.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


@pytest.mark.parametrize(
    "module",
    [
        "oaht_bench.configs",
        "oaht_bench.envs",
        "oaht_bench.agents",
        "oaht_bench.teammate_gen",
        "oaht_bench.marl",
        "oaht_bench.common",
        "oaht_bench.evaluation",
        "oaht_bench.algorithms",
        "oaht_bench.offline",
        "oaht_bench.data",
    ],
)
def test_package_modules_import(module: str):
    """Every declared subpackage imports cleanly under the single root."""
    importlib.import_module(module)


def test_configs_do_not_require_jax():
    """Config validation must not pay JAX's import cost.

    The CLI validates before dispatching, and --dry-run must work on a machine
    without a working accelerator.
    """
    import subprocess
    import sys

    code = (
        "import sys; import oaht_bench.configs as c; "
        "assert 'jax' not in sys.modules, 'importing configs pulled in JAX'"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
