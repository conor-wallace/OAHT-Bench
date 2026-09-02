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
from pathlib import Path


def _population_dirs(env_dir: Path) -> list[Path]:
    """Released generator directories under an environment's populations root."""
    return sorted(d for d in env_dir.iterdir() if d.is_dir() and (d / "job.json").exists())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "env_dir", type=Path, help="populations/<env>/, holding one dir per generator."
    )
    parser.add_argument(
        "--num-episodes", type=int, default=20, help="Episodes per (ego, teammate) pair."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .npz (default: <env_dir>/pooled_crossplay.npz).",
    )
    args = parser.parse_args()

    import numpy as np

    from oaht_bench.configs import load_job
    from oaht_bench.configs.job import PooledCrossplayJob
    from oaht_bench.population.pooled_crossplay import run

    pop_dirs = _population_dirs(args.env_dir)
    if not pop_dirs:
        raise SystemExit(f"no released populations (dirs with job.json) under {args.env_dir}")

    # Every population for this env shares one env config; read it from the first.
    first_job = load_job(pop_dirs[0] / "job.json")
    out = args.out or (args.env_dir / "pooled_crossplay.npz")

    # Delegate the compute and the writes to the pooled_crossplay job runner, so this
    # convenience wrapper and ``oaht-bench config=...`` share one code path. The
    # wrapper only adds directory discovery and the read-out below.
    job = PooledCrossplayJob(
        env=first_job.env,
        population_path=[str(p) for p in pop_dirs],
        num_episodes=args.num_episodes,
        seed=args.seed,
        output_path=str(out),
        label=f"pooled_{first_job.env.name}",
    )
    run(job)

    # Sanity read-out from what was written: for each teammate column, the best and
    # worst ego, straight off the roster arrays the npz carries.
    d = np.load(out, allow_pickle=True)
    matrix, gen, mem, role = (
        d["matrix"],
        d["roster_generator"],
        d["roster_member"],
        d["roster_role"],
    )
    print(f"\nwrote {out}  (K={matrix.shape[0]})")
    print("per-teammate best / worst response (by pooled return):")
    for j in range(matrix.shape[1]):
        best_i = int(np.argmax(matrix[:, j]))
        worst_i = int(np.argmin(matrix[:, j]))
        print(
            f"  teammate [{j:2d}] {gen[j]}/{mem[j]}({role[j]}): "
            f"best={gen[best_i]}/{mem[best_i]}({role[best_i]}) R={matrix[best_i, j]:.3f}  "
            f"worst={gen[worst_i]}/{mem[worst_i]}({role[worst_i]}) R={matrix[worst_i, j]:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
