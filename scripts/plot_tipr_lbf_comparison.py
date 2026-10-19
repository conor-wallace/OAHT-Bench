#!/usr/bin/env python3
"""Recreate TIPR's Figure 3 LBF row (TAO vs. LIAM only) next to our own replication.

TIPR (ICML 2025) Fig. 3 has no backing numeric table in the paper -- the
``TIPR_REFERENCE`` values below are read directly off that figure's bar heights
(papers/tipr.pdf, page 7), not exact published numbers. They are the *original*
(pre-TIPR-refinement) bars only, since we don't implement TIPR's own online
refinement step -- we're replicating the Optimal-Ratio *dataset construction*,
not the refinement framework. Seen and Unseen were visually near-identical in
the figure for LBF, so the same read applies to both; Mixed is dropped since we
have no directly analogous number on our side.

Our side is discovered automatically: for every ``{tao,liam}_pooled_*/training_summary.json``
under --results, this reads the run's own job.json -> dataset_path -> that
dataset's own job.json -> ego_ratio, so it buckets by whatever rho a run was
actually trained on rather than guessing from directory names. Runs whose
dataset has no ego_ratio set (e.g. the earlier non-TIPR 'weighted'/'expert'
replications) are skipped -- they aren't part of this rho sweep.

Fills in whatever is available and leaves the rest blank, so this can be re-run
as more rho datasets finish training without editing anything.

    uv run python scripts/plot_tipr_lbf_comparison.py
    uv run python scripts/plot_tipr_lbf_comparison.py --results lbf_9x9_experiment
    uv run python scripts/plot_tipr_lbf_comparison.py --out figs/tipr_lbf.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASELINES = ["liam", "tao"]
RHOS = [0.2, 0.4, 0.6, 0.8]
MODES = ["seen", "unseen"]

# Colour encodes the baseline, matching TIPR's own Fig. 3 palette.
BASELINE_COLOR = {"liam": "#ED8B3A", "tao": "#C0392B"}

#: Read directly off papers/tipr.pdf page 7, Fig. 3, LBF row, original
#: (non-hatched, pre-refinement) bar heights -- NOT from a table (the paper has
#: none for this figure). Seen and Unseen looked the same within each rho panel,
#: so both modes use the same read here.
TIPR_REFERENCE = {
    ("liam", 0.2): 0.17,
    ("liam", 0.4): 0.25,
    ("liam", 0.6): 0.32,
    ("liam", 0.8): 0.35,
    ("tao", 0.2): 0.25,
    ("tao", 0.4): 0.33,
    ("tao", 0.6): 0.37,
    ("tao", 0.8): 0.42,
}


def _dataset_ego_ratio(dataset_path: str) -> float | None:
    """``ego_ratio`` of the dataset a training run consumed, or None (no rho set).

    ``dataset_path`` is recorded relative to the repo root (where the collection
    job wrote it), regardless of where the training run or --results root lives.
    """
    ds_dir = os.path.dirname(dataset_path)  # strip /dataset.vlt
    job_path = os.path.join(ds_dir, "job.json")
    if not os.path.exists(job_path):
        return None
    job = json.load(open(job_path)).get("job", {})
    return job.get("ego_ratio")


def _collect(root: str) -> dict:
    """``{(baseline, rho): {"seen": mean_return, "unseen": mean_return}}``."""
    data, runs_seen = {}, 0
    for baseline in BASELINES:
        pattern = os.path.join(root, f"{baseline}_pooled_*", "training_summary.json")
        for summary_path in glob.glob(pattern):
            run_dir = os.path.dirname(summary_path)
            job_path = os.path.join(run_dir, "job.json")
            if not os.path.exists(job_path):
                continue
            job = json.load(open(job_path)).get("job", {})
            dataset_path = job.get("dataset_path")
            if not dataset_path:
                continue
            rho = _dataset_ego_ratio(dataset_path)
            if rho is None:
                continue  # not part of the rho sweep (e.g. the earlier expert/weighted replications)
            summary = json.load(open(summary_path))
            ev = summary.get("eval")
            if not ev:
                continue
            runs_seen += 1
            data[(baseline, round(float(rho), 2))] = {
                "unseen": ev.get("mean_return"),
                "seen": (ev.get("train") or {}).get("mean_return"),
            }
    print(f"found {runs_seen} rho-sweep training run(s) under {root}")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", default="results/training", help="training results root to scan")
    ap.add_argument("--out", default="results/plots/tipr_lbf_comparison.png")
    args = ap.parse_args()

    ours = _collect(args.results)

    fig, axes = plt.subplots(1, len(RHOS), figsize=(16, 4.5), sharey=True)
    x = np.arange(len(MODES))
    w = 0.2
    # offsets: TIPR-liam, Ours-liam, TIPR-tao, Ours-tao
    offsets = {
        ("liam", "tipr"): -1.5 * w,
        ("liam", "ours"): -0.5 * w,
        ("tao", "tipr"): 0.5 * w,
        ("tao", "ours"): 1.5 * w,
    }

    for ax, rho in zip(axes, RHOS, strict=True):
        for baseline in BASELINES:
            color = BASELINE_COLOR[baseline]
            tipr_y = np.full(len(MODES), TIPR_REFERENCE.get((baseline, rho), np.nan))
            ax.bar(
                x + offsets[(baseline, "tipr")],
                tipr_y,
                w,
                color=color,
                alpha=0.45,
                hatch="//",
                edgecolor="black",
                linewidth=0.6,
                label=f"{baseline.upper()} (TIPR paper, read off Fig. 3)"
                if rho == RHOS[0]
                else None,
            )
            cell = ours.get((baseline, rho), {})
            ours_y = np.array([cell.get(m, np.nan) for m in MODES], dtype=float)
            ax.bar(
                x + offsets[(baseline, "ours")],
                np.nan_to_num(ours_y),
                w,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                label=f"{baseline.upper()} (ours)" if rho == RHOS[0] else None,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in MODES])
        ax.set_title(f"LBF - Optimal Ratio ρ: {rho}")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Average Return")
    handles = [
        mpatches.Patch(
            facecolor=BASELINE_COLOR["liam"],
            alpha=0.45,
            hatch="//",
            edgecolor="black",
            label="LIAM (TIPR paper)",
        ),
        mpatches.Patch(facecolor=BASELINE_COLOR["liam"], edgecolor="black", label="LIAM (ours)"),
        mpatches.Patch(
            facecolor=BASELINE_COLOR["tao"],
            alpha=0.45,
            hatch="//",
            edgecolor="black",
            label="TAO (TIPR paper)",
        ),
        mpatches.Patch(facecolor=BASELINE_COLOR["tao"], edgecolor="black", label="TAO (ours)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(
        "TAO vs. LIAM: TIPR's reported LBF results (approx., read off Fig. 3) vs. our replication",
        y=1.14,
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
