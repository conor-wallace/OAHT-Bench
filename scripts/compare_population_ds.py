"""Compare Diversity (D) and Sparseness (S) across populations or sub-populations.

``D`` and ``S`` are deterministic functions of the crossplay matrix (no dataset, no
baseline training), so you can slice any sub-population out of a pooled matrix -- by
generator and/or role -- and compare cells directly. The motivating case: does lowering
BRDiv's ``cross_play_weight`` (0.5 -> 0.05) move the *BRDiv* teammates off the adversarial
corner, isolated from the cross-generator mismatch that dominates the pooled roster?

Examples:
    # compare two runs, each sliced to just the BRDiv teammate policies:
    uv run python scripts/compare_population_ds.py \
        run_cpw05/pooled_crossplay.npz run_cpw50/pooled_crossplay.npz \
        --generators brdiv --roles conf

    # decompose one pooled matrix into per-generator blocks (shows the confound):
    uv run python scripts/compare_population_ds.py populations/hanabi/pooled_crossplay.npz --per-generator
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/ for the sibling import
from diagnose_population_suitability import (  # noqa: E402
    _ADV_DEAD_FRAC,
    _ADV_RATIO,
    _LOW_DIVERSITY,
    ds_from_matrix,
    load_crossplay,
)


def _regime(a) -> str:
    if np.isnan(a["diversity"]):
        return "-"
    if a["dead_fraction"] > _ADV_DEAD_FRAC or a["cross_over_self"] < _ADV_RATIO:
        return "STOP (adversarial)"
    if a["diversity"] < _LOW_DIVERSITY:
        return "TRIVIAL (low-D)"
    return "candidate GO"


def _idx(n, gens, roles, gsel, roleset):
    """Roster indices kept by the generator filter ``gsel`` and role filter ``roleset``."""
    mask = np.ones(n, bool)
    if gsel is not None:
        if gens is None:
            raise SystemExit("--generators needs an .npz with roster labels (csv has none)")
        mask &= np.array([g in gsel for g in gens])
    if roleset is not None:
        if roles is None:
            raise SystemExit("--roles / --ego-roles / --teammate-roles need an .npz with roster labels")
        mask &= np.array([r in roleset for r in roles])
    return np.flatnonzero(mask)


def _print_row(label, m, ego_idx, tm_idx, diag_selfplay):
    if len(tm_idx) < 2 or len(ego_idx) < 1:
        print(f"  {label:46s} {len(tm_idx):>3d}   (need >=1 ego, >=2 teammates)")
        return
    a = ds_from_matrix(m[np.ix_(ego_idx, tm_idx)], diagonal_is_selfplay=diag_selfplay)
    xself = "     --" if np.isnan(a["cross_over_self"]) else f"{a['cross_over_self']:>7.2f}"
    print(
        f"  {label:46s} {a['roster_size']:>3d} {a['diversity']:>6.2f} "
        f"{xself} {a['dead_fraction']:>6.2f}   {_regime(a)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("crossplay", nargs="+", help="one or more pooled_crossplay .npz (or .csv)")
    ap.add_argument("--generators", default=None, help="comma list to keep (e.g. brdiv or brdiv,lbrdiv)")
    ap.add_argument("--roles", default=None, help="symmetric role filter (conf,br,self)")
    ap.add_argument(
        "--ego-roles",
        default=None,
        help="asymmetric: roles to use as EGOs (e.g. br). With --teammate-roles this scores a "
        "BR-egos x teammates block -- the correct view for BR-paired generators like BRDiv, where "
        "confederates alone are degenerate. cross/self is then undefined.",
    )
    ap.add_argument("--teammate-roles", default=None, help="asymmetric: roles to use as TEAMMATEs (e.g. conf,self)")
    ap.add_argument(
        "--per-generator",
        action="store_true",
        help="also print each generator's own block per file (reveals within- vs cross-generator structure)",
    )
    args = ap.parse_args()
    gsel = set(args.generators.split(",")) if args.generators else None
    rsel = set(args.roles.split(",")) if args.roles else None
    ego_r = set(args.ego_roles.split(",")) if args.ego_roles else None
    tm_r = set(args.teammate_roles.split(",")) if args.teammate_roles else None
    asymmetric = ego_r is not None or tm_r is not None

    def emit(label, m, gens, roles, gfilter):
        if asymmetric:
            e = _idx(m.shape[0], gens, roles, gfilter, ego_r)
            t = _idx(m.shape[0], gens, roles, gfilter, tm_r)
            _print_row(label, m, e, t, diag_selfplay=False)
        else:
            i = _idx(m.shape[0], gens, roles, gfilter, rsel)
            _print_row(label, m, i, i, diag_selfplay=True)

    tag = ""
    if asymmetric:
        tag = f" [ego {args.ego_roles or '*'} x tm {args.teammate_roles or '*'}]"
    elif rsel:
        tag = f" [{args.roles}]"
    print(f"  {'population':46s} {'n':>3s} {'D':>6s} {'x/self':>7s} {'dead':>6s}   regime")
    for path in args.crossplay:
        m, gens, roles = load_crossplay(path)
        emit(path.split("/")[-1] + (f" [{args.generators}]" if gsel else "") + tag, m, gens, roles, gsel)
        if args.per_generator and gens is not None:
            for g in sorted(set(gens)):
                emit(f"  └ {g}{tag}", m, gens, roles, {g})


if __name__ == "__main__":
    main()
