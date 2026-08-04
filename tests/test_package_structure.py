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

COLLIDING_NAMES = ["envs", "agents", "common", "marl", "teammate_generation", "evaluation",
                   "ego_agent_training", "runners", "benchmarks", "teammate_wrapper"]


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
        "oaht_bench.teammate_gen.marl",
        "oaht_bench.common",
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


def test_baselines_live_in_algorithms_not_agents():
    """LIAM and MeLIBA are methods under evaluation, not agent infrastructure.

    Keeping them beside the actor-critic primitives conflated "a network we build
    things from" with "a baseline we are measuring" (§6).
    """
    import oaht_bench.algorithms.liam_agent  # noqa: F401
    import oaht_bench.algorithms.meliba_agent  # noqa: F401

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("oaht_bench.agents.liam_agent")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("oaht_bench.agents.meliba_agent")


def test_no_dangling_references_to_unabsorbed_upstream():
    """Nothing may import a jax-aht subtree we chose not to absorb."""
    import pathlib

    pkg = pathlib.Path(__file__).resolve().parents[1] / "src" / "oaht_bench"
    offenders = []
    for path in pkg.rglob("*.py"):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            for mod in ("ego_agent_training", "open_ended_training", "evaluation."):
                if stripped.startswith((f"from {mod}", f"import {mod}")):
                    offenders.append(f"{path.relative_to(pkg)}:{i}: {stripped}")
    assert not offenders, "dangling upstream imports:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    "module",
    [
        "oaht_bench.teammate_gen.fcp",
        "oaht_bench.teammate_gen.CoMeDi",
        "oaht_bench.teammate_gen.BRDiv",
        "oaht_bench.teammate_gen.LBRDiv",
        "oaht_bench.algorithms.liam_agent",
        "oaht_bench.algorithms.meliba_agent",
        "oaht_bench.algorithms.initialize_baselines",
    ],
)
def test_absorbed_modules_import(module: str):
    """Every absorbed module must import under its new path.

    The structural tests above only touch package ``__init__`` files, so a stale
    intra-package import inside a module went unnoticed until the first real run.
    Importing each one directly closes that gap.
    """
    importlib.import_module(module)


def test_all_generators_are_dispatchable():
    """Every generator named in the config union has a runner."""
    from oaht_bench.configs.teammate_gen import GeneratorConfig
    from oaht_bench.teammate_gen.runner import _generators

    import typing

    members = typing.get_args(typing.get_args(GeneratorConfig)[0])
    declared = {m.model_fields["generator"].default for m in members}
    assert declared == set(_generators())
