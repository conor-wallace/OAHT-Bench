"""Single entry point. One config file determines one run.

    uv run oaht-bench config=configs/my_experiment.json

The CLI deliberately has almost no flags. Anything that changes what an
experiment *does* belongs in the config file, because the config file is the
artifact we release and the record a result is traced back to. A flag that
alters behaviour without appearing in the config would break that guarantee.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from oaht_bench.configs import load_job
from oaht_bench.configs.job import AnyJob


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="oaht-bench",
        description="Run an OAHT-Bench experiment from a JSON config.",
    )
    parser.add_argument(
        "config",
        help="Path to the experiment config. Accepts 'config=path.json' or a bare path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config, print the resolved run plan, and exit without running.",
    )
    return parser.parse_args(argv)


def _strip_key_prefix(value: str) -> str:
    """Accept ``config=path.json`` as well as a bare path."""
    prefix = "config="
    return value[len(prefix) :] if value.startswith(prefix) else value


def _describe(job: AnyJob) -> str:
    lines = [
        f"job_type   {job.job_type}",
        f"label      {job.label}",
        f"seed       {job.seed}",
        f"config     {job.content_hash()}",
        f"run_dir    {job.run_dir()}",
    ]
    if job.job_type == "teammate_generation":
        # Imported here so config validation stays free of JAX.
        from oaht_bench.teammate_gen.plan import training_plan

        try:
            lines += ["", *training_plan(job).describe().splitlines()]
        except ValueError as e:
            lines += ["", f"training plan   UNRUNNABLE: {e}"]

    env = getattr(job, "env", None)
    if env is not None:
        lines += [
            f"env        {env.name} ({env.env_name}, tier={env.tier})",
            f"           turn_based={env.turn_based} symmetric_roles={env.symmetric_roles}",
            f"           env_kwargs={env.env_kwargs()}",
        ]
    return "\n".join(lines)


def _dispatch(job: AnyJob) -> int:
    """Route a validated job to its runner.

    Runners are imported lazily so that ``--dry-run`` — and config validation
    generally — does not pay JAX's import cost or require a working accelerator.
    """
    if job.job_type == "teammate_generation":
        from oaht_bench.teammate_gen.runner import run as run_teammate_generation

        run_dir = run_teammate_generation(job)
        print(f"\nwrote {run_dir}")
        return 0
    if job.job_type == "dataset_collection":
        from oaht_bench.data.runner import run as run_data_generation

        run_dir = run_data_generation(job)
        print(f"\nwrote {run_dir}")
        return 0
    if job.job_type == "training":
        from oaht_bench.offline.runner import run as run_training

        run_dir = run_training(job)
        print(f"\nwrote {run_dir}")
        return 0
    if job.job_type == "evaluation":
        raise NotImplementedError("evaluation runner not yet implemented (§8).")
    raise ValueError(f"Unroutable job_type: {job.job_type!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    path = _strip_key_prefix(args.config)

    try:
        job = load_job(path)
    except ValidationError as e:
        print(f"Invalid config: {path}\n", file=sys.stderr)
        print(e, file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(_describe(job))
    if args.dry_run:
        return 0
    return _dispatch(job)


if __name__ == "__main__":
    raise SystemExit(main())
