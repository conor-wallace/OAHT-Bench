"""Generate and score a hyperparameter grid for one (generator, environment) cell.

Two modes.

**Generate** — expand a grid over a base config and write one JSON per cell::

    uv run python scripts/sweep.py generate \\
        --base configs/teammate_gen/lbf_12x12/brdiv.json \\
        --name brdiv_lbf_xpw \\
        --set generator.cross_play_weight=0.005,0.05,0.5 \\
        --set generator.ppo.learning_rate=1e-4,5e-4

Every cell is materialized as its own config file rather than expanded at run
time. Each therefore has its own content hash and its own run directory, so a
single cell can be re-run or cited in isolation — which is what §7.2 needs from
the released tuning record. It also means a cell that is invalid (a budget that
would train nothing, minibatches wider than the env axis) fails *here*, before
anything is queued.

**Collect** — tabulate finished runs::

    uv run python scripts/sweep.py collect --sweep configs/sweeps/brdiv_lbf_xpw

Reports self-play and cross-play return separately. Deliberately not a single
score: for the diversity-seeking generators these trade off, and collapsing them
hides the degenerate corner where cross-play is low because the population is
incompetent or because members have learned to sabotage each other (§7.4).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from oaht_bench.configs import load_job, save_job, validate_job
from oaht_bench.teammate_gen.plan import training_plan

REPO_ROOT = Path(__file__).resolve().parents[1]


def _relative_if_inside(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------


def _coerce(text: str) -> Any:
    """Parse a grid value, preserving ints, floats, bools and strings."""
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _set_path(payload: dict, dotted: str, value: Any) -> None:
    """Assign into a nested dict by dotted path, e.g. ``generator.ppo.gamma``."""
    node = payload
    parts = dotted.split(".")
    for key in parts[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[parts[-1]] = value


def generate(base: Path, name: str, axes: dict[str, list[Any]], out_root: Path) -> list[Path]:
    base_job = load_job(base)
    base_payload = json.loads(
        json.dumps(
            {"job": base_job.model_dump(mode="json"), "schema_version": 1}
        )
    )

    keys = list(axes)
    written: list[tuple[Path, dict, Any]] = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        payload = json.loads(json.dumps(base_payload))
        settings = dict(zip(keys, combo))
        for dotted, value in settings.items():
            _set_path(payload["job"], dotted, value)

        cell = "_".join(f"{k.split('.')[-1]}={v}" for k, v in settings.items())

        # Two validations, both before anything is written. The config check
        # catches field-level errors; the runtime check catches the ones that
        # only appear when the training shape is computed -- a budget that would
        # train nothing, minibatches wider than the batch axis. A sweep that
        # queues an unrunnable cell wastes a scheduler slot and reports nothing.
        try:
            job = validate_job(payload)
            plan = training_plan(job)
        except (ValueError, Exception) as e:  # noqa: B014 - ValidationError is a subclass
            raise SystemExit(
                f"cell {cell!r} is not runnable, so the sweep was not written:\n  {e}"
            ) from e

        job = job.model_copy(update={"label": f"{name}__{cell}"})
        written.append((f"{cell}.json", settings, job, plan))

    # Only now touch the filesystem: a rejected cell must not leave a partial
    # sweep behind that looks complete.
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        (save_job(j, out_dir / fname, minimal=True), s_, j, pl)
        for fname, s_, j, pl in written
    ]

    manifest = {
        "name": name,
        # Repo-relative where possible so the manifest is portable, but a base
        # outside the tree (a scratch config) is recorded as given.
        "base": _relative_if_inside(base),
        "axes": {k: axes[k] for k in keys},
        "cells": [
            {
                "config": p.name,
                "settings": s,
                "config_hash": j.content_hash(),
                "sequential_updates": pl.sequential_updates,
            }
            for p, s, j, pl in written
        ],
    }
    (out_dir / "sweep.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return [(p, pl) for p, _, _, pl in written]


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------


def _final_curve_value(metrics_path: Path, tag: str) -> float | None:
    best_step, value = -1, None
    for line in metrics_path.read_text().splitlines():
        rec = json.loads(line)
        if tag in rec and rec.get("train_step", -1) >= best_step:
            best_step, value = rec["train_step"], rec[tag]
    return value


def _population_scores(run_dir: Path) -> tuple[float | None, float | None]:
    """Self-play and cross-play from the post-training population evaluation.

    This is the one measurement computed identically for all four generators
    (see ``teammate_gen.crossplay``), so it is what a cross-generator sweep
    should rank on. The per-generator ``Eval/`` curves are reported alongside but
    are not comparable between methods.
    """
    csv = run_dir / "population_crossplay.csv"
    if not csv.exists():
        return None, None
    m = np.loadtxt(csv, delimiter=",", ndmin=2)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] < 2:
        return None, None
    diag = float(np.mean(np.diag(m)))
    off = float((m.sum() - np.trace(m)) / (m.size - m.shape[0]))
    return diag, off


def collect(sweep_dir: Path, results_root: Path, competence_tol: float) -> None:
    """Rank cells by competence first, then by separation.

    The ordering is the one that matches what a teammate population is *for*:
    members must be individually competent before their diversity means anything.
    Ranking on separation alone selects populations that are merely bad in
    different ways, since cross-play also falls when members cannot score at all.

    Competence is a band rather than a strict sort: cells within ``competence_tol``
    of the best self-play score are treated as equally competent and ordered by
    separation. A strict lexicographic sort on floats would let a 0.001 difference
    in self-play override a large difference in diversity.
    """
    manifest = json.loads((sweep_dir / "sweep.json").read_text())
    keys = list(manifest["axes"])

    rows = []
    for cell in manifest["cells"]:
        job = load_job(sweep_dir / cell["config"])
        run_dir = Path(job.run_dir())
        if not run_dir.is_absolute():
            run_dir = results_root / run_dir
        sp, xp = _population_scores(run_dir)
        rows.append({"settings": cell["settings"], "sp": sp, "xp": xp})

    scored = [r for r in rows if r["sp"] is not None and r["xp"] is not None]
    for r in scored:
        r["sep"] = r["sp"] - r["xp"]

    best_sp = max((r["sp"] for r in scored), default=None)
    if best_sp is not None and best_sp > 0:
        floor = best_sp * (1.0 - competence_tol)
        band = [r for r in scored if r["sp"] >= floor]
    else:
        floor, band = None, []
    band.sort(key=lambda r: r["sep"], reverse=True)
    winner = band[0] if band else None

    header = " ".join(f"{k.split('.')[-1]:>16s}" for k in keys)
    print(f"{header} {'SP':>9s} {'XP':>9s} {'SP-XP':>9s}  status")
    order = band + [r for r in scored if r not in band]
    for r in order:
        vals = " ".join(f"{str(r['settings'][k]):>16s}" for k in keys)
        mark = "  <- best" if r is winner else ("" if r in band else "  (below competence floor)")
        print(f"{vals} {r['sp']:9.4f} {r['xp']:9.4f} {r['sep']:9.4f}{mark}")
    for r in rows:
        if r["sp"] is None:
            vals = " ".join(f"{str(r['settings'][k]):>16s}" for k in keys)
            print(f"{vals} {'-':>9s} {'-':>9s} {'-':>9s}  not run")

    if winner is None:
        print("\nNo cell has a scored population yet.")
        return
    print(
        f"\nRanked by separation among cells within {competence_tol:.0%} of the best "
        f"self-play score ({best_sp:.4f}, floor {floor:.4f}).\n"
        f"Competence first: a population whose members cannot score when paired "
        f"with themselves\nis not useful training data however distinct they are, "
        f"and cross-play falls for that\nreason too -- so separation alone would "
        f"rank a collapsed population first."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("generate", help="Expand a grid into one config per cell.")
    g.add_argument("--base", required=True, type=Path)
    g.add_argument("--name", required=True)
    g.add_argument(
        "--set", action="append", default=[], metavar="PATH=V1,V2",
        help="Dotted config path and comma-separated values. Repeatable.",
    )
    g.add_argument("--out", type=Path, default=REPO_ROOT / "configs" / "sweeps")

    r = sub.add_parser(
        "rescore",
        help="Recompute population_crossplay.csv from saved checkpoints, no retraining.",
    )
    r.add_argument("runs", nargs="+", type=Path, help="Run directories.")
    r.add_argument("--episodes", type=int, default=None,
                   help="Override evaluation_episodes for this pass.")
    r.add_argument("--greedy", action="store_true", default=None,
                   help="Argmax actions. Diagnostic only -- deadlocks symmetric "
                        "coordination tasks; see TeammateGenerationJob.evaluation_greedy.")
    r.add_argument("--dry-run", action="store_true", help="Print without writing.")

    c = sub.add_parser("collect", help="Tabulate finished runs.")
    c.add_argument("--sweep", required=True, type=Path)
    c.add_argument("--results", type=Path, default=REPO_ROOT)
    c.add_argument(
        "--competence-tol", type=float, default=0.05,
        help="Cells within this fraction of the best self-play score are treated "
             "as equally competent and ranked by separation (default 0.05).",
    )

    args = ap.parse_args()

    if args.mode == "generate":
        axes: dict[str, list[Any]] = {}
        for spec in args.set:
            if "=" not in spec:
                raise SystemExit(f"--set expects PATH=V1,V2 but got {spec!r}")
            path, values = spec.split("=", 1)
            axes[path] = [_coerce(v) for v in values.split(",")]
        if not axes:
            raise SystemExit("at least one --set is required")

        paths = generate(args.base, args.name, axes, args.out)
        total = 0
        print(f"{'cell':52s} {'seq updates':>12s}")
        for p, plan in paths:
            total += plan.sequential_updates
            print(f"{p.name:52s} {plan.sequential_updates:12,d}")
        print(f"\n{len(paths)} cells -> {(args.out / args.name)}")
        print(f"total sequential updates across the sweep: {total:,}")
        return 0

    if args.mode == "rescore":
        from oaht_bench.teammate_gen.rescore import rescore_run

        print(f"{'run':64s} {'SP':>9s} {'XP':>9s} {'SP-XP':>9s}")
        failed = 0
        for run in args.runs:
            try:
                s = rescore_run(run, episodes=args.episodes, greedy=args.greedy,
                                write=not args.dry_run)
            except Exception as e:  # one unscorable run must not abort the batch
                failed += 1
                print(f"{run.name[:64]:64s}  {type(e).__name__}: {str(e).splitlines()[0][:60]}")
                continue
            print(f"{run.name[:64]:64s} {s.self_play:9.4f} {s.cross_play:9.4f} "
                  f"{s.separation:9.4f}")
        if args.dry_run:
            print("\ndry run: no population_crossplay.csv was written.")
        if failed:
            print(f"\n{failed} run(s) could not be scored.")
        return 1 if failed else 0

    collect(args.sweep, args.results, args.competence_tol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
