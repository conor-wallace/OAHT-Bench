"""Emit teammate-generation configs for each (generator, environment) pair.

Hyperparameters are ported from jax-aht's per-environment Hydra configs rather
than invented. Those encode real tuning — Hanabi wants `gamma=0.999` and a much
larger budget, Overcooked wants a larger `clip_eps` and entropy coefficient than
LBF — and discarding it to start from defaults would throw away working settings
and make the first runs uninformative.

**These are starting points, not the tuned configuration.** §7.2 of the project
plan makes the per-environment tuning record a contribution; this script produces
the baseline that record will be built against, and every value here should be
treated as provisional until a sweep says otherwise.

Regenerate with::

    uv run python scripts/gen_teammate_configs.py            # tier 1
    uv run python scripts/gen_teammate_configs.py --all-envs # all seven
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from oaht_bench.configs import get_preset, preset_names, save_job
from oaht_bench.configs.job import TeammateGenerationJob
from oaht_bench.configs.network import MlpNetwork
from oaht_bench.configs.teammate_gen import (
    BrDivConfig,
    CoMeDiConfig,
    FcpConfig,
    LBrDivConfig,
    PpoHyperparams,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "configs" / "teammate_gen"

#: Which environment family a preset belongs to, for looking up tuning below.
def _family(preset_name: str) -> str:
    if preset_name.startswith("overcooked"):
        return "overcooked"
    if "hanabi" in preset_name:
        return "hanabi"
    return "lbf"


# --------------------------------------------------------------------------
# PPO settings, per (generator, environment family), from jax-aht's configs.
# --------------------------------------------------------------------------
PPO: dict[str, dict[str, dict[str, Any]]] = {
    "fcp": {
        "lbf": dict(learning_rate=1e-4, update_epochs=15, num_minibatches=4,
                    clip_eps=0.03, entropy_coef=0.01),
        "overcooked": dict(learning_rate=1e-3, update_epochs=15, num_minibatches=16,
                           clip_eps=0.1, entropy_coef=0.05),
        "hanabi": dict(learning_rate=5e-4, update_epochs=4, num_minibatches=4,
                       clip_eps=0.2, entropy_coef=0.01, anneal_lr=True,
                       gamma=0.999, gae_lambda=0.95),
    },
    "comedi": {
        "lbf": dict(learning_rate=5e-4, update_epochs=15, num_minibatches=8,
                    clip_eps=0.05, entropy_coef=0.001),
        "overcooked": dict(learning_rate=1e-3, update_epochs=15, num_minibatches=8,
                           clip_eps=0.01, entropy_coef=0.05),
        "hanabi": dict(learning_rate=5e-4, update_epochs=4, num_minibatches=8,
                       clip_eps=0.2, entropy_coef=0.01, anneal_lr=True,
                       gamma=0.999, gae_lambda=0.95, max_grad_norm=0.5),
    },
    "brdiv": {
        "lbf": dict(learning_rate=5e-4, update_epochs=15, num_minibatches=2,
                    clip_eps=0.05, entropy_coef=0.01),
        "overcooked": dict(learning_rate=1e-3, update_epochs=15, num_minibatches=8,
                           clip_eps=0.01, entropy_coef=0.05),
        "hanabi": dict(learning_rate=5e-4, update_epochs=4, num_minibatches=4,
                       clip_eps=0.2, entropy_coef=0.01, anneal_lr=True,
                       gamma=0.999, gae_lambda=0.95),
    },
    "lbrdiv": {
        "lbf": dict(learning_rate=5e-4, update_epochs=15, num_minibatches=4,
                    clip_eps=0.05, entropy_coef=0.01),
        "overcooked": dict(learning_rate=1e-3, update_epochs=15, num_minibatches=8,
                           clip_eps=0.01, entropy_coef=0.05),
        "hanabi": dict(learning_rate=5e-4, update_epochs=4, num_minibatches=4,
                       clip_eps=0.2, entropy_coef=0.01, anneal_lr=True,
                       gamma=0.999, gae_lambda=0.95),
    },
}

#: Budget, population and environment count, per (generator, family).
#: ``pop`` is the authored PARTNER_POP_SIZE. Note it is *not* the resulting
#: population size for FCP, which yields ``pop * num_checkpoints`` members
#: because it snapshots during training — see the README.
SCALE: dict[str, dict[str, dict[str, Any]]] = {
    "fcp": {
        "lbf": dict(total_timesteps=1e6, num_envs=8, pop=5),
        "overcooked": dict(total_timesteps=4e6, num_envs=8, pop=5),
        "hanabi": dict(total_timesteps=1e9, num_envs=32, pop=3),
    },
    "comedi": {
        "lbf": dict(total_timesteps_per_iteration=6e6, num_envs=48, pop=10),
        "overcooked": dict(total_timesteps_per_iteration=1e7, num_envs=48, pop=10),
        "hanabi": dict(total_timesteps_per_iteration=2e7, num_envs=48, pop=5),
    },
    "brdiv": {
        "lbf": dict(total_timesteps=4.5e7, num_envs=64, pop=3),
        "overcooked": dict(total_timesteps=9e7, num_envs=128, pop=3),
        "hanabi": dict(total_timesteps=5e8, num_envs=128, pop=3),
    },
    "lbrdiv": {
        "lbf": dict(total_timesteps=4.5e7, num_envs=64, pop=3),
        "overcooked": dict(total_timesteps=9e7, num_envs=128, pop=3),
        "hanabi": dict(total_timesteps=5e8, num_envs=128, pop=3),
    },
}

#: Diversity weights that differ per environment.
CROSS_PLAY_WEIGHT = {
    "brdiv": {"lbf": 0.05, "overcooked": 0.005, "hanabi": 0.05},
    "comedi": {"lbf": 0.2, "overcooked": 1.0, "hanabi": 0.2},
}
MIXED_PLAY_WEIGHT = {"lbf": 0.4, "overcooked": 0.5, "hanabi": 0.5}
TOLERANCE_FACTOR = {"lbf": 0.1, "overcooked": 10.0, "hanabi": 0.1}

#: L-BRDiv's Lagrange multipliers receive gradient from an unnormalized sum over
#: ~n^2 pair terms, so the learning rate must be scaled by ~(n_ref/n)^2 relative
#: to the population it was tuned at. Upstream's 0.01 is tuned at n = 3 (§7.3).
LAGRANGE_LR_AT_N3 = 0.01


def _lagrange_lr(pop: int) -> float:
    return round(LAGRANGE_LR_AT_N3 * (3 / pop) ** 2, 6)


def build(generator: str, preset_name: str, num_checkpoints: int = 5):
    fam = _family(preset_name)
    ppo = PpoHyperparams(**PPO[generator][fam])
    scale = SCALE[generator][fam]
    pop = scale["pop"]
    common = dict(
        population_size=pop,
        num_checkpoints=num_checkpoints,
        num_envs=scale["num_envs"],
        ppo=ppo,
        network=MlpNetwork(),
    )

    if generator == "fcp":
        return FcpConfig(total_timesteps=scale["total_timesteps"], **common)
    if generator == "comedi":
        return CoMeDiConfig(
            total_timesteps_per_iteration=scale["total_timesteps_per_iteration"],
            cross_play_weight=CROSS_PLAY_WEIGHT["comedi"][fam],
            mixed_play_weight=MIXED_PLAY_WEIGHT[fam],
            **common,
        )
    if generator == "brdiv":
        return BrDivConfig(
            total_timesteps=scale["total_timesteps"],
            cross_play_weight=CROSS_PLAY_WEIGHT["brdiv"][fam],
            **common,
        )
    if generator == "lbrdiv":
        return LBrDivConfig(
            total_timesteps=scale["total_timesteps"],
            tolerance_factor=TOLERANCE_FACTOR[fam],
            lagrange_learning_rate=_lagrange_lr(pop),
            **common,
        )
    raise ValueError(f"unknown generator {generator!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all-envs", action="store_true",
                    help="Emit for all seven results configurations, not just tier 1.")
    args = ap.parse_args()

    envs = preset_names() if args.all_envs else preset_names("tier1")
    envs = [e for e in envs if e != "mini_hanabi"]

    written = []
    for env_name in envs:
        env = get_preset(env_name)
        for generator in ("fcp", "comedi", "brdiv", "lbrdiv"):
            gen = build(generator, env_name)
            job = TeammateGenerationJob(
                label=f"{generator}_{env_name}", env=env, generator=gen
            )
            path = OUT_ROOT / env_name / f"{generator}.json"
            save_job(job, path, minimal=True)
            written.append((env_name, generator, gen, job))

    print(f"{'environment':30s} {'gen':8s} {'pop':>4s} {'envs':>5s} {'budget':>10s}  hash")
    for env_name, generator, gen, job in written:
        budget = getattr(gen, "total_timesteps", None) or gen.total_timesteps_per_iteration
        print(f"{env_name:30s} {generator:8s} {gen.population_size:4d} "
              f"{gen.num_envs:5d} {budget:10.1e}  {job.short_hash()}")
    print(f"\n{len(written)} configs -> {OUT_ROOT.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
