"""Compute the pooled cross-population coordination-return matrix for one env.

Pools the released members of every generator under ``populations/<env>/`` into a
single roster of policies and scores every ordered ``(ego, teammate)`` pair, then
writes ``populations/<env>/pooled_crossplay.npz`` (matrix + roster manifest) plus
a human-readable ``.csv`` and ``roster.json``. This is the artifact the dataset
sampler reads to place each episode on the best-worst response spectrum
(``docs/dataset_design.md`` §3).

    uv run python scripts/pooled_crossplay.py populations/lbf_12x12 --num-episodes 20

Cost is ``K**2 * num_episodes`` episodes, so it is a run-once offline job; every
cell is a separate rollout and heterogeneous policies recompile, so expect it to
be slower than a within-population crossplay of the same members.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _population_dirs(env_dir: Path) -> list[Path]:
    """Released generator directories under an environment's populations root."""
    return sorted(d for d in env_dir.iterdir() if d.is_dir() and (d / "job.json").exists())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("env_dir", type=Path, help="populations/<env>/, holding one dir per generator.")
    parser.add_argument("--num-episodes", type=int, default=20, help="Episodes per (ego, teammate) pair.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="Output .npz (default: <env_dir>/pooled_crossplay.npz).")
    args = parser.parse_args()

    import jax
    import numpy as np

    from oaht_bench.configs import load_job
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.population.pooled_crossplay import (
        build_roster,
        evaluate_pooled,
        normalise_per_teammate,
        save_pooled,
    )

    pop_dirs = _population_dirs(args.env_dir)
    if not pop_dirs:
        raise SystemExit(f"no released populations (dirs with job.json) under {args.env_dir}")

    # Every population for this env shares one env config; read it from the first.
    first_job = load_job(pop_dirs[0] / "job.json")
    env = LogWrapper(make_env(first_job.env.env_name, first_job.env.env_kwargs()))

    roster = build_roster(pop_dirs, env)
    print(f"pooled roster: {len(roster)} policies from {len(pop_dirs)} populations")
    for i, e in enumerate(roster):
        print(f"  [{i:2d}] {e.generator:8s} member {e.member:2d} ({e.role})")

    matrix = evaluate_pooled(
        env,
        roster,
        rng=jax.random.PRNGKey(args.seed),
        max_episode_steps=first_job.env.rollout_length,
        num_episodes=args.num_episodes,
    )

    out = args.out or (args.env_dir / "pooled_crossplay.npz")
    save_pooled(
        matrix,
        roster,
        out,
        meta={
            "env": first_job.env.name,
            "populations": [str(p) for p in pop_dirs],
            "num_episodes": args.num_episodes,
            "seed": args.seed,
        },
    )
    # A readable copy of the matrix and roster alongside the npz.
    np.savetxt(out.with_suffix(".csv"), matrix, delimiter=",")
    (out.with_name("roster.json")).write_text(
        json.dumps(
            [{"index": i, "generator": e.generator, "member": e.member, "role": e.role} for i, e in enumerate(roster)],
            indent=2,
        )
        + "\n"
    )

    # Sanity read-out: for each teammate column, the best and worst ego.
    quality = normalise_per_teammate(matrix)
    print(f"\nwrote {out}  (K={len(roster)})")
    print("per-teammate best / worst response (by pooled return):")
    for j, mate in enumerate(roster):
        best_i = int(np.argmax(matrix[:, j]))
        worst_i = int(np.argmin(matrix[:, j]))
        b, w = roster[best_i], roster[worst_i]
        print(
            f"  teammate [{j:2d}] {mate.generator}/{mate.member}({mate.role}): "
            f"best={b.generator}/{b.member}({b.role}) R={matrix[best_i, j]:.3f}  "
            f"worst={w.generator}/{w.member}({w.role}) R={matrix[worst_i, j]:.3f}"
        )
    del quality
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
