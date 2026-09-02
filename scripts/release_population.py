"""Publish a finished teammate-generation run into populations/.

populations/ is the durable, git-tracked record of the benchmark's official
population checkpoints -- distinct from results/ (gitignored, every sweep
attempt) in that it holds only the one adopted checkpoint per
(environment, generator), organized to match configs/teammate_gen/'s layout::

    uv run python scripts/release_population.py \\
        results/teammate_generation/brdiv_lbf_budget2__..-3ddef8e7b353 \\
        --dest populations

    -> populations/lbf_12x12/brdiv/

The release keeps job.json alongside the checkpoint, in exactly the layout
artifact_dir()/rescore_run() already expect, so a released directory is itself
a valid run_dir: `sweep.py rescore populations/lbf_12x12/brdiv` works with no
new code, and the checkpoint can never be separated from the config that
produced it -- the problem that made the LBF FCP checkpoint unverifiable
before this script existed (docs/tuning_record.md).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from oaht_bench.configs import load_job
from oaht_bench.population.rescore import artifact_dir

#: Carried along for a human reading the release, if present. None of these
#: are required to rescore -- rescore_run only reads job.json and the
#: checkpoint -- so a missing one is not an error. metrics.jsonl is
#: deliberately not here: it scales with training length, not population
#: size, and a long run's can run into the hundreds of MB -- unbounded and
#: unnecessary in a git-tracked release. docs/tuning_record.md is the durable
#: summary of what a run's curve showed; the raw log stays in results/.
_PROVENANCE_FILES = (
    "config.json",
    "population_crossplay.csv",
    "Eval_LastXPMatrix.csv",
)


def release(run_dir: Path, dest_root: Path, *, force: bool = False) -> Path:
    """Copy one finished run's job.json and checkpoint into dest_root.

    Destination is dest_root/<env name>/<generator>/, e.g.
    populations/lbf_12x12/brdiv/ -- one slot per (environment, generator), so
    releasing again overwrites the previous release rather than accumulating
    every attempt the way results/ does. That is deliberate: this is meant to
    be the drop-in filled by scripts/gen_teammate_configs.py's adopted table
    (docs/tuning_record.md), not a history of every run.
    """
    run_dir = Path(run_dir)
    job_path = run_dir / "job.json"
    if not job_path.exists():
        raise FileNotFoundError(
            f"{job_path} does not exist -- {run_dir} does not look like a "
            f"finished run directory."
        )
    job = load_job(job_path)
    if job.job_type != "teammate_generation":
        raise ValueError(f"{run_dir} is a {job.job_type} run, not a population.")

    # Raises before anything is copied if training crashed before its
    # checkpoint write -- a run half-copied into populations/ would be worse
    # than one not attempted.
    ckpt_src = artifact_dir(run_dir)

    dest = dest_root / job.env.name / job.generator.generator
    if dest.exists():
        if not force:
            raise FileExistsError(
                f"{dest} already exists. Pass --force to replace it (e.g. "
                f"after re-tuning produced a better checkpoint)."
            )
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    shutil.copy2(job_path, dest / "job.json")
    shutil.copytree(ckpt_src, dest / "artifacts" / "saved_train_run")

    for name in _PROVENANCE_FILES:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)

    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="A finished results/teammate_generation/<run> directory.")
    parser.add_argument("--dest", type=Path, default=Path("populations"), help="Root to release into (default: populations).")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing release for this (environment, generator).")
    args = parser.parse_args()

    dest = release(args.run_dir, args.dest, force=args.force)
    print(f"released {args.run_dir} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
