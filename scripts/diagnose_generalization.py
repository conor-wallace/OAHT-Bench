"""Generalization-gap probe with clean train/eval seed separation.

Measures the trained clone on two disjoint sets, using the *saved training
normalization* (not a recomputed one):

* IN-DIST: the exact deals training used (``--seed`` root, sampled teammate --
  same reset seeds *and* same action sampling reproduce the training episodes),
* HELD-OUT: a disjoint seed (``--eval-seed``), same collection method.

Reports, for each set, teacher-forced next-action accuracy (the generalization
gap in prediction) and closed-loop return of the BC ego vs the teammate (the gap
in behaviour). If the clone cannot even score on the deals it trained on, that is
a different, deeper problem than failing to generalize to new deals.

    uv run python scripts/diagnose_generalization.py \
        --config configs/hanabi/training/pooled_expert_scaled/bc.json \
        --params /tmp/bc_clone.pkl --pairing comedi:0:self \
        --seed 0 --eval-seed 987654321 --episodes 50
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--params", required=True, help="Policy pickle from --save-params.")
    ap.add_argument("--pairing", default="comedi:0:self")
    ap.add_argument(
        "--populations",
        nargs="+",
        default=[
            "populations/hanabi/brdiv",
            "populations/hanabi/comedi",
            "populations/hanabi/fcp",
            "populations/hanabi/lbrdiv",
        ],
    )
    ap.add_argument("--seed", type=int, default=0, help="Train seed (must match the saved clone).")
    ap.add_argument("--eval-seed", type=int, default=987654321, help="Disjoint held-out seed.")
    ap.add_argument("--episodes", type=int, default=50)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from oaht_bench.configs import load_job
    from oaht_bench.dataset.construction.collect import collect_episode
    from oaht_bench.dataset.dataset import Dataset
    from oaht_bench.dataset.vault import write_vault
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.models.bc_agent import BcAgent
    from oaht_bench.models.liam_agent import LiamAgent
    from oaht_bench.models.meliba_agent import MelibaAgent
    from oaht_bench.models.omis_agent import OmisAgent
    from oaht_bench.models.tao_agent import TaoAgent
    from oaht_bench.offline.runner import _resolve_dims
    from oaht_bench.population.pooled_crossplay import build_roster

    agents = {
        "bc": BcAgent,
        "liam": LiamAgent,
        "meliba": MelibaAgent,
        "omis": OmisAgent,
        "tao": TaoAgent,
    }

    saved = pickle.load(open(args.params, "rb"))
    norm = saved["normalization"]
    job = load_job(args.config)
    base_env = make_env(job.env.env_name, job.env.env_kwargs())
    env = LogWrapper(base_env)
    roster = build_roster([Path(p) for p in args.populations], env)
    gen, member, role = args.pairing.split(":")
    tm = next(
        e for e in roster if e.generator == gen and int(e.member) == int(member) and e.role == role
    )

    cfg = job.offline.model_copy(update={"context_length": saved["context_length"]})
    resolved = _resolve_dims(cfg, saved["obs_dim"], saved["action_dim"])
    agent = agents[saved["baseline"]](
        resolved,
        context_length=saved["context_length"],
        target_return=saved["cond_target"],
        normalization=norm,
    )
    agent.build_model()
    params = {"stage1": saved["stage1"], "stage2": saved["stage2"]}

    def collect_set(seed, seats, greedy):
        """Reproduce a seed's collection sequence exactly (matches the gate's loop)."""
        rng = jax.random.PRNGKey(seed)
        eps = []
        for _ in range(args.episodes):
            rng, ep_rng = jax.random.split(rng)
            eps.append(
                collect_episode(
                    ep_rng, base_env, seats, max_episode_steps=job.env.rollout_length, greedy=greedy
                )
            )
        return eps

    tm_seats = [(tm.params, tm.policy_cls), (tm.params, tm.policy_cls)]
    bc_seats = [(params, agent), (tm.params, tm.policy_cls)]

    def tf_accuracy(episodes):
        """Teacher-forced next-action accuracy, using the SAVED normalization."""
        d = Path(tempfile.mkdtemp(prefix="gen_")) / "p.vlt"
        write_vault(
            episodes,
            np.array([[int(tm.member), int(tm.member)]] * len(episodes)),
            d,
            ego_index=0,
            meta={"variant": "single"},
        )
        w = Dataset(
            str(d), context_length=cfg.context_length, stride=cfg.stride, normalize=False
        ).windows
        obs_n = (w.ego_obs - np.asarray(norm.obs_mean)) / np.asarray(norm.obs_std)
        rtg_n = w.ego_rtg / float(norm.rtg_scale)
        logits = np.asarray(
            agent.act(
                params,
                jnp.asarray(rtg_n),
                jnp.asarray(obs_n),
                jnp.asarray(w.ego_actions),
                timesteps=jnp.asarray(w.timesteps),
                mask=jnp.asarray(w.mask),
            )
        )
        pred = np.where(w.ego_avail > 0, logits, -1e9).argmax(-1)
        correct = (pred == w.ego_actions) & w.mask
        shutil.rmtree(d.parent, ignore_errors=True)
        by_t = {}
        for lo, hi in [(1, 5), (6, 10), (11, 20), (21, 40), (41, 200)]:
            sel = w.mask & (w.timesteps >= lo) & (w.timesteps <= hi)
            by_t[(lo, hi)] = float((correct & sel).sum() / sel.sum()) if sel.sum() else float("nan")
        return float(correct.sum() / w.mask.sum()), by_t

    # Same deals, teammate self-play: IN-DIST reproduces the training episodes exactly.
    indist_eps = collect_set(args.seed, tm_seats, greedy=False)
    heldout_eps = collect_set(args.eval_seed, tm_seats, greedy=False)
    indist_acc, _ = tf_accuracy(indist_eps)
    heldout_acc, heldout_by_t = tf_accuracy(heldout_eps)

    # Closed-loop return, BC ego (argmax) on the same deals (in-dist) vs held-out.
    # Two teammate modes: argmax teammate is the deterministic replication check;
    # sampled teammate (ego argmax, mate greedy=False) is the benchmark's eval regime.
    def mean_return(seed, seats, greedy):
        return float(np.mean([e.returns()[0] for e in collect_set(seed, seats, greedy)]))

    cl = {
        ("in", "argmax"): mean_return(args.seed, bc_seats, [True, True]),
        ("held", "argmax"): mean_return(args.eval_seed, bc_seats, [True, True]),
        ("in", "sampled"): mean_return(args.seed, bc_seats, [True, False]),
        ("held", "sampled"): mean_return(args.eval_seed, bc_seats, [True, False]),
    }
    ceil = {
        ("in", "argmax"): mean_return(args.seed, tm_seats, True),
        ("held", "argmax"): mean_return(args.eval_seed, tm_seats, True),
        ("in", "sampled"): mean_return(args.seed, tm_seats, False),
        ("held", "sampled"): mean_return(args.eval_seed, tm_seats, False),
    }

    print(
        "\n===== GENERALIZATION GAP (train seed vs eval seed, saved normalization) =====",
        flush=True,
    )
    print(
        f"  pairing={args.pairing}  train-seed={args.seed}  eval-seed={args.eval_seed}", flush=True
    )
    print("\n  teacher-forced next-action accuracy:", flush=True)
    print(f"    IN-DIST (training deals):  {indist_acc:.4f}", flush=True)
    print(f"    HELD-OUT (eval deals):     {heldout_acc:.4f}", flush=True)
    print(f"    gap:                       {indist_acc - heldout_acc:+.4f}", flush=True)
    print("    held-out accuracy by timestep:", flush=True)
    for (lo, hi), a in heldout_by_t.items():
        print(f"      {lo:>2}-{hi:<3}: {a:.4f}", flush=True)
    print("\n  closed-loop BC-ego return (argmax ego):", flush=True)
    for mate in ("argmax", "sampled"):
        note = " (benchmark regime)" if mate == "sampled" else " (deterministic replication)"
        print(f"    {mate}-teammate{note}:", flush=True)
        for where, label in (("in", "IN-DIST "), ("held", "HELD-OUT")):
            r, c = cl[(where, mate)], ceil[(where, mate)]
            print(
                f"      {label} deals:  BC {r:5.2f}  ({100 * r / c:.0f}% of ceiling {c:.2f})",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
