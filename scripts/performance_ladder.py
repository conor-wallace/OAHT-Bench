"""The performance ladder: what a teammate-modeling method could and could not win.

Built to answer one question about ``lbf_20x20`` (and reusable for any ``ppo_br``
population): the five offline baselines (BC/LIAM/MeLIBA/OMIS/TAO) land within noise
of each other at ~0.09-0.10 return, and the question is whether that is a modeling
defect or a testbed ceiling. Two facts, measured directly from the trained BR egos
and teammates rather than inferred from the offline baselines, settle it:

1. **The right BR matters a lot.** ``matched BR`` vs ``mismatched BR`` on the same
   teammate is the return a perfect vs. a wrong teammate-identification would get.
2. **A deployed method can only reproduce what it saw.** Its ceiling against an
   *unseen* teammate is not the teammate's own BR (which the method never observed)
   but the best-scoring *train* BR replayed against it -- the retrieval oracle.

Four rows, in increasing order of "how good could this possibly be":

- ``mean random BR``      -- floor: pick a train BR that isn't tailored at all.
- ``best fixed generalist``-- ceiling for any teammate-*agnostic* policy (no ID).
- ``retrieval oracle``     -- ceiling for a method that identifies the teammate
                              perfectly *from among the train BRs* (transfers to
                              held-out teammates by construction -- this is the
                              oracle a trained method could actually approach).
- ``matched-BR oracle``    -- ceiling with the teammate's *own* trained BR (only
                              achievable for train teammates; a trained generalist
                              policy cannot reach this on held-out teammates no
                              matter how good its inference is).

From these, ``break-even identification rate`` is the fraction of episodes a
method would need to correctly identify the teammate (and play the retrieval-
oracle's choice) to beat the best fixed generalist -- solving
``fixed = p * retrieval + (1-p) * mean_random`` for ``p``. Report observed
identification against *this* number, not in isolation: 24% identification means
something different at a 19% bar than at a 62% one.

Usage:
    uv run python scripts/performance_ladder.py \\
        --populations populations/lbf_20x20/brdiv populations/lbf_20x20/comedi \\
                      populations/lbf_20x20/fcp populations/lbf_20x20/lbrdiv \\
        --br-run ppo_br_lbf_20x20-7db59378f7a9 \\
        --split pooled_lbf_20x20_expert-99e3ca80c5ab/teammate_split.json \\
        --episodes 200 --out populations/lbf_20x20/br_ladder.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from oaht_bench.common.run_episodes import run_episodes
from oaht_bench.configs import load_job
from oaht_bench.envs import make_env
from oaht_bench.population.loading import load_br_egos
from oaht_bench.population.pooled_crossplay import build_roster


def _label(entry) -> str:
    return f"{entry.generator}:{entry.member}:{entry.role}"


def build_br_matrix(populations: list[Path], br_run: Path, *, episodes: int, seed: int = 0):
    """Evaluate every trained BR ego against every teammate it could face.

    Returns ``(matrix, labels)`` where ``matrix[i, j]`` is BR ego ``i``'s mean
    return against teammate ``j`` -- the ``returned_episode_returns`` LogWrapper
    metric, matching what :mod:`population.pooled_crossplay` itself reports (a bare
    env without the wrapper reports a different, incompatible quantity).
    """
    job = load_job(populations[0] / "job.json")
    env = make_env(job.env.env_name, job.env.env_kwargs())
    roster = build_roster(populations, env)
    br = load_br_egos(br_run, env)

    mates = [e for e in roster if (e.generator, e.member, e.role) in br]
    if not mates:
        raise SystemExit(
            f"no roster entry matches a BR in {br_run} -- check --populations lists "
            f"the same generators the ppo_br run trained against."
        )
    labels = [_label(e) for e in mates]
    egos = [br[(e.generator, e.member, e.role)] for e in mates]

    k = len(mates)
    matrix = np.zeros((k, k))
    rng = jax.random.PRNGKey(seed)
    t0 = time.time()
    for i, (ego_params, ego_cls) in enumerate(egos):
        for j, mate in enumerate(mates):
            rng, key = jax.random.split(rng)
            out = run_episodes(
                key,
                env,
                agent_0_param=ego_params,
                agent_0_policy=ego_cls,
                agent_1_param=mate.params,
                agent_1_policy=mate.policy_cls,
                max_episode_steps=job.env.rollout_length,
                num_eps=episodes,
                agent_0_test_mode=False,
                agent_1_test_mode=False,
            )
            matrix[i, j] = float(np.asarray(out["returned_episode_returns"]).mean())
        print(f"  BR row {i + 1}/{k} ({labels[i]})  {time.time() - t0:.0f}s", flush=True)
    return matrix, labels


def ladder(matrix: np.ndarray, labels: list[str], train: set[str], held_out: set[str]) -> dict:
    """The four-row ceiling table, computed separately for train and held-out columns.

    ``train``/``held_out`` are label sets; every row/column of ``matrix`` must be
    labelled with a roster ``"{generator}:{member}:{role}"`` string that appears in
    exactly one of them (or neither, if it's excluded from both -- e.g. a ``br``
    role, which is an ego identity, never a teammate).
    """
    idx = {lab: i for i, lab in enumerate(labels)}
    train_i = [idx[lab] for lab in labels if lab in train]
    if not train_i:
        raise SystemExit(
            "no roster labels matched the 'train' set -- check the split file's keys/ids"
        )

    out = {}
    for split_name, cols in (
        ("train", train_i),
        ("held_out", [idx[lab] for lab in labels if lab in held_out]),
    ):
        if not cols:
            out[split_name] = None
            continue
        own = float(np.mean([matrix[j, j] for j in cols]))
        retrieval = float(np.mean([max(matrix[k, j] for k in train_i) for j in cols]))
        # best single fixed train BR, i.e. the best teammate-agnostic policy
        # available to a method that only ever saw the train BRs' behaviour.
        fixed = float(max(np.mean([matrix[k, j] for j in cols]) for k in train_i))
        rand = float(np.mean([np.mean([matrix[k, j] for k in train_i]) for j in cols]))
        headroom = retrieval - fixed
        # break-even: fixed = p*retrieval + (1-p)*rand  =>  p = (fixed-rand)/(retrieval-rand)
        denom = retrieval - rand
        break_even = float((fixed - rand) / denom) if denom > 1e-9 else float("nan")
        out[split_name] = {
            "n_teammates": len(cols),
            "matched_br_oracle": own,
            "retrieval_oracle": retrieval,
            "best_fixed_generalist": fixed,
            "mean_random_br": rand,
            "adaptive_headroom": headroom,
            "break_even_identification": break_even,
        }
    return out


def _print_ladder(report: dict) -> None:
    for split_name, row in report.items():
        if row is None:
            print(f"\n=== {split_name} : no teammates in this split ===")
            continue
        print(f"\n=== {split_name}  (n={row['n_teammates']}) ===")
        print(
            f"  matched-BR oracle (true ceiling, train-only reachable) {row['matched_br_oracle']:.4f}"
        )
        print(
            f"  retrieval oracle  (best TRAIN BR; transfers)           {row['retrieval_oracle']:.4f}"
        )
        print(
            f"  best fixed generalist (no identification)              {row['best_fixed_generalist']:.4f}"
        )
        print(
            f"  mean random train BR                                   {row['mean_random_br']:.4f}"
        )
        print(
            f"  adaptive headroom (retrieval - fixed)                  {row['adaptive_headroom']:+.4f}"
        )
        print(
            f"  break-even identification rate                         {100 * row['break_even_identification']:.1f}%"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--populations", nargs="+", required=True, type=Path)
    ap.add_argument("--br-run", required=True, type=Path, help="a ppo_br run directory")
    ap.add_argument(
        "--split", type=Path, default=None, help="a teammate_split.json (from dataset collection)"
    )
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", type=Path, default=None, help="save the raw BR x teammate matrix here (.npz)"
    )
    ap.add_argument(
        "--matrix-in", type=Path, default=None, help="skip evaluation, load a matrix saved by --out"
    )
    args = ap.parse_args()

    if args.matrix_in:
        d = np.load(args.matrix_in, allow_pickle=True)
        matrix, labels = d["matrix"], [str(x) for x in d["labels"]]
    else:
        matrix, labels = build_br_matrix(
            args.populations, args.br_run, episodes=args.episodes, seed=args.seed
        )
        if args.out:
            np.savez(args.out, matrix=matrix, labels=labels)
            print(f"\nsaved matrix to {args.out}")

    if args.split:
        split = json.loads(args.split.read_text())
        held = {f"{g}:{m}:conf" for g, ms in split["held_out"].items() for m in ms} | {
            f"{g}:{m}:self" for g, ms in split["held_out"].items() for m in ms
        }
        train = {lab for lab in labels if lab not in held}
        held = {lab for lab in labels if lab in held}
    else:
        # no split given: everything is "train", nothing is held out.
        train, held = set(labels), set()

    report = ladder(matrix, labels, train, held)
    _print_ladder(report)


if __name__ == "__main__":
    main()
