"""Compute and print the paper-reported statistics for one experiment life
cycle (teammate generation -> ppo_br -> dataset collection -> baseline
evaluation), so numbers going into the paper come from this script instead
of hand-transcription from JSON/CSV/npz files.

**The ceiling convention.** Every return in a life cycle is normalized by
that life cycle's own crossplay matrix's single peak cell
(``matrix.max()``) -- the ego x teammate matrix built by
``population/pooled_crossplay.py``, e.g. ``pooled_mep_crossplay.npz``. This
is the one formula used throughout the paper's tables and figures ("normalized
to [0,1] by the matrix's own maximum cell"); this script applies it
uniformly to everything in the life cycle -- baseline returns (pooled and
per-teammate), the trained population's own self-play/cross-play, the
ppo_br mean return, and the pooled dataset's mean episode return -- rather
than mixing that convention with an improvised one (e.g. population-mean
or min-max) per section, which is exactly the kind of inconsistency that
motivated this script.

Every input except ``--crossplay`` is optional; omit what a given life
cycle doesn't have and the corresponding block is skipped, not guessed.

    uv run python scripts/report_paper_stats.py \\
        --label "MEP on LBF-20x20" \\
        --crossplay lbf_20x20_experiment/pooled_mep_crossplay.npz \\
        --eval-report lbf_20x20_experiment/eval_mep_weighted-9bfe81212df1/evaluation_report.json \\
        --population-crossplay lbf_20x20_experiment/population_crossplay.csv \\
        --br-metrics lbf_20x20_experiment/ppo_mep_br_lbf_20x20-b2c1e6f71ffd/metrics.jsonl \\
        --dataset-summary lbf_20x20_experiment/pooled_mep_lbf_20x20_weighted-d8b496a9567b/dataset_summary.json \\
        --dataset-job lbf_20x20_experiment/pooled_mep_lbf_20x20_weighted-d8b496a9567b/job.json \\
        --teammate-gen-job lbf_20x20_experiment/job.json \\
        --tikz-out /tmp/lbf20x20_mep_bars.tex \\
        --dat-out /tmp/lbf20x20_mep_crossplay_data.dat \\
        --json-out /tmp/lbf20x20_mep_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_crossplay(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load an ego x teammate crossplay matrix (``.npz`` from
    ``population/pooled_crossplay.py``, or a plain ``.csv``).

    Returns ``(matrix, labels)`` where ``labels[i]`` is ``"<generator>:<member>:<role>"``
    for an ``.npz`` (falls back to bare indices for a ``.csv``, which carries
    no roster).
    """
    path = Path(path)
    if path.suffix == ".npz":
        d = np.load(path, allow_pickle=True)
        matrix = d["matrix"]
        gens = d["roster_generator"]
        mems = d["roster_member"]
        roles = d["roster_role"]
        labels = [f"{g}:{m}:{r}" for g, m, r in zip(gens, mems, roles, strict=True)]
        return matrix, labels
    matrix = np.loadtxt(path, delimiter=",")
    return matrix, [str(i) for i in range(matrix.shape[0])]


def load_population_crossplay(path: Path) -> tuple[float, float, float]:
    """Self-play / cross-play / separation from a population's own
    training-time crossplay matrix (``population_crossplay.csv``), matching
    ``population/crossplay.py::evaluate_population``'s exact formula.
    """
    m = np.loadtxt(path, delimiter=",")
    n = m.shape[0]
    self_play = float(np.mean(np.diag(m)))
    if n > 1:
        cross_play = float((m.sum() - np.trace(m)) / (m.size - n))
    else:
        cross_play = float("nan")
    return self_play, cross_play, self_play - cross_play


def load_br_mean_return(path: Path) -> float:
    """Final logged ``BR/mean_return`` from a ``ppo_br`` run's ``metrics.jsonl``."""
    last = None
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "BR/mean_return" in rec:
                last = rec["BR/mean_return"]
    if last is None:
        raise ValueError(f"no BR/mean_return logged in {path}")
    return float(last)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def crossplay_stats(matrix: np.ndarray, labels: list[str]) -> dict[str, Any]:
    n = matrix.shape[0]
    diag = np.diag(matrix)
    offdiag = matrix[~np.eye(n, dtype=bool)]
    peak_idx = int(np.argmax(matrix))
    peak_row, peak_col = divmod(peak_idx, n)
    ceiling = float(matrix.max())
    return {
        "n": n,
        "ceiling": ceiling,
        "ceiling_cell": (labels[peak_row], labels[peak_col]),
        "diag_mean": float(diag.mean()),
        "diag_min": float(diag.min()),
        "diag_min_label": labels[int(np.argmin(diag))],
        "diag_max": float(diag.max()),
        "diag_max_label": labels[int(np.argmax(diag))],
        "offdiag_mean": float(offdiag.mean()),
        "offdiag_min": float(offdiag.min()),
        "offdiag_max": float(offdiag.max()),
        "diag_raw": diag.tolist(),
        "diag_norm": (diag / ceiling).tolist(),
    }


def eval_report_stats(report: dict, ceiling: float) -> dict[str, Any]:
    out = {}
    for run_path, d in report["checkpoints"].items():
        baseline = d["baseline"]
        entry: dict[str, Any] = {
            "seen_pool_size": d["seen_pool_size"],
            "unseen_pool_size": d["unseen_pool_size"],
            "generalization_gap_raw": d["generalization_gap"],
        }
        for split in ("unseen", "seen"):
            raw = d[split]["mean_return"]
            entry[f"{split}_raw"] = raw
            entry[f"{split}_norm"] = raw / ceiling
            for acc_key, floor_key in (
                ("mate_action_acc", "mate_action_floor"),
                ("ancillary_mate_action_acc", "ancillary_mate_action_floor"),
            ):
                if acc_key in d[split]:
                    entry[f"{split}_mate_acc"] = d[split][acc_key]
                    entry[f"{split}_mate_floor"] = d[split][floor_key]
        out[baseline] = entry
    return out


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------


def emit_dat(matrix: np.ndarray, ceiling: float, path: Path) -> None:
    """Write the full matrix, normalized by ``ceiling``, as an ``x y C`` pgfplots
    table -- ready to drop in as a paper's crossplay-heatmap data file."""
    n = matrix.shape[0]
    lines = ["x y C"]
    for row in range(n):
        for col in range(n):
            lines.append(f"{col} {row} {matrix[row, col] / ceiling:.4f}")
        lines.append("")
    Path(path).write_text("\n".join(lines))


def emit_tikz_bars(diag_norm: np.ndarray, prefix: str, path: Path) -> None:
    """Write ``\\definecolor`` + ``\\addplot`` lines for the diagonal bar chart,
    colored on the same viridis scale as the heatmap -- ready to paste into the
    paper, no hand-transcription of 20 numbers or their colors."""
    import matplotlib.cm as cm

    cmap = cm.get_cmap("viridis")
    lines = []
    for i, v in enumerate(diag_norm):
        r, g, b, _ = cmap(float(v))
        lines.append(f"\\definecolor{{{prefix}{i}}}{{rgb}}{{{r:.4f},{g:.4f},{b:.4f}}}")
    for i, v in enumerate(diag_norm):
        lines.append(
            f"\\addplot[ybar, fill={prefix}{i}, draw=none, forget plot] "
            f"coordinates {{({i},{v:.4f})}};"
        )
    Path(path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--label", default="", help="Header label for this life cycle.")
    p.add_argument("--crossplay", required=True, type=Path, help="ego x teammate crossplay .npz/.csv.")
    p.add_argument("--eval-report", type=Path, help="evaluation_report.json")
    p.add_argument("--population-crossplay", type=Path, help="population_crossplay.csv (training-time self/cross-play)")
    p.add_argument("--br-metrics", type=Path, help="ppo_br run's metrics.jsonl")
    p.add_argument("--dataset-summary", type=Path, help="dataset_summary.json")
    p.add_argument("--dataset-job", type=Path, help="dataset_collection job.json")
    p.add_argument("--teammate-gen-job", type=Path, help="teammate_generation job.json")
    p.add_argument("--tikz-out", type=Path, help="Write bar-chart \\definecolor/\\addplot lines here.")
    p.add_argument("--tikz-prefix", default="bar", help="Color-name prefix for --tikz-out (must be unique per figure).")
    p.add_argument("--dat-out", type=Path, help="Write the normalized x y C matrix here.")
    p.add_argument("--json-out", type=Path, help="Also dump every computed number as JSON.")
    args = p.parse_args()

    report: dict[str, Any] = {"label": args.label}

    matrix, labels = load_crossplay(args.crossplay)
    cp = crossplay_stats(matrix, labels)
    report["crossplay"] = cp
    ceiling = cp["ceiling"]

    print(f"=== {args.label or args.crossplay} ===")
    print(f"crossplay matrix: {cp['n']}x{cp['n']}, ceiling = {ceiling:.4f} at {cp['ceiling_cell']}")
    print(f"  diagonal   mean={cp['diag_mean']:.4f}  min={cp['diag_min']:.4f} ({cp['diag_min_label']})"
          f"  max={cp['diag_max']:.4f} ({cp['diag_max_label']})")
    print(f"  offdiagonal mean={cp['offdiag_mean']:.4f}  min={cp['offdiag_min']:.4f}  max={cp['offdiag_max']:.4f}")

    if args.population_crossplay:
        sp, xp, sep = load_population_crossplay(args.population_crossplay)
        report["population"] = {
            "self_play_raw": sp, "self_play_norm": sp / ceiling,
            "cross_play_raw": xp, "cross_play_norm": xp / ceiling,
            "separation_raw": sep,
        }
        print(f"\npopulation self-play/cross-play (training-time, raw):")
        print(f"  self-play  = {sp:.4f}  (norm {sp/ceiling:.4f})")
        print(f"  cross-play = {xp:.4f}  (norm {xp/ceiling:.4f})")
        print(f"  separation = {sep:.4f}")

    if args.br_metrics:
        br = load_br_mean_return(args.br_metrics)
        report["ppo_br_mean_return"] = {"raw": br, "norm": br / ceiling}
        print(f"\nppo_br mean return (logged) = {br:.4f}  (norm {br/ceiling:.4f})")
        print(f"  [crossplay-diagonal mean was {cp['diag_mean']:.4f} -- should closely match]")

    if args.dataset_summary:
        ds = json.loads(args.dataset_summary.read_text())
        raw_ego = ds.get("mean_ego_return")
        report["dataset"] = {**ds}
        if raw_ego is not None:
            report["dataset"]["mean_ego_return_norm"] = raw_ego / ceiling
        print(f"\ndataset: {ds.get('episodes')} episodes, mean_length={ds.get('mean_length'):.2f}, "
              f"mean_ego_return={raw_ego:.4f} (norm {raw_ego/ceiling:.4f})" if raw_ego is not None
              else f"\ndataset: {ds}")

    if args.dataset_job:
        dj = json.loads(args.dataset_job.read_text())["job"]
        report["dataset_job"] = {
            k: dj.get(k) for k in ("num_episodes", "temperature", "holdout_per_generator", "variant")
        }
        print(f"\ndataset collection config: {report['dataset_job']}")

    if args.teammate_gen_job:
        tj = json.loads(args.teammate_gen_job.read_text())["job"]["generator"]
        keys = ("population_size", "num_envs", "total_timesteps", "population_entropy_coef")
        report["teammate_gen_job"] = {k: tj.get(k) for k in keys}
        report["teammate_gen_job"]["ppo"] = tj.get("ppo")
        print(f"\nteammate_gen config: {report['teammate_gen_job']}")

    if args.eval_report:
        er = json.loads(args.eval_report.read_text())
        stats = eval_report_stats(er, ceiling)
        report["baselines"] = stats
        print(f"\nbaseline returns (raw -> normalized by ceiling {ceiling:.4f}):")
        for baseline, e in stats.items():
            u_acc = f" mate_acc={e['unseen_mate_acc']:.4f}/floor={e['unseen_mate_floor']:.4f}" if "unseen_mate_acc" in e else ""
            s_acc = f" mate_acc={e['seen_mate_acc']:.4f}/floor={e['seen_mate_floor']:.4f}" if "seen_mate_acc" in e else ""
            print(f"  {baseline:8s} unseen raw={e['unseen_raw']:.4f} norm={e['unseen_norm']:.4f}{u_acc}")
            print(f"  {'':8s} seen   raw={e['seen_raw']:.4f} norm={e['seen_norm']:.4f}{s_acc}")
            print(f"  {'':8s} generalization_gap (raw) = {e['generalization_gap_raw']:.4f}")

    if args.dat_out:
        emit_dat(matrix, ceiling, args.dat_out)
        print(f"\nwrote normalized crossplay matrix -> {args.dat_out}")

    if args.tikz_out:
        emit_tikz_bars(np.array(cp["diag_norm"]), args.tikz_prefix, args.tikz_out)
        print(f"wrote bar-chart tikz -> {args.tikz_out}")

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote full report -> {args.json_out}")


if __name__ == "__main__":
    main()
