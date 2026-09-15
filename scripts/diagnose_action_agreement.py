"""Nuclear diagnostic: BC clone vs. the original comedi ego, in parallel.

Runs the trained BC clone and the policy it was cloned from (comedi:0) against the
same teammate, from *identical* env resets and under *argmax*, then compares the
actions they take and the returns they achieve. Same reset + argmax makes both
rollouts deterministic and bit-identical until the two ego policies first choose a
different action -- so the first-divergence step is attributable purely to the ego:

* diverges at step ~0-3          -> the clone cannot even track the expert early;
                                    a deploy/inference problem, not just drift.
* tracks for many steps then      -> genuine covariate shift: small deviations
  diverges, low return               push the recurrent partner off-distribution.
* never diverges but low return   -> the partner/return accounting is the issue.

Point it at a policy saved by ``diagnose_single_pairing.py --save-params`` so it
never retrains:

    uv run python scripts/diagnose_action_agreement.py \
        --config configs/hanabi/training/pooled_expert_scaled/bc.json \
        --params /tmp/bc_clone.pkl --pairing comedi:0:self --episodes 100
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, help="The training config used to train the clone.")
    ap.add_argument(
        "--params", required=True, help="Policy pickle from diagnose_single_pairing --save-params."
    )
    ap.add_argument("--pairing", default="comedi:0:self", help="generator:member:role of the ego.")
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
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax

    from oaht_bench.configs import load_job
    from oaht_bench.dataset.construction.collect import collect_episode
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
    job = load_job(args.config)
    base_env = make_env(job.env.env_name, job.env.env_kwargs())
    env = LogWrapper(base_env)
    roster = build_roster([Path(p) for p in args.populations], env)
    gen, member, role = args.pairing.split(":")
    orig = next(
        e for e in roster if e.generator == gen and int(e.member) == int(member) and e.role == role
    )

    cfg = job.offline.model_copy(update={"context_length": saved["context_length"]})
    resolved = _resolve_dims(cfg, saved["obs_dim"], saved["action_dim"])
    bc = agents[saved["baseline"]](
        resolved,
        context_length=saved["context_length"],
        target_return=saved["cond_target"],
        normalization=saved["normalization"],
    )
    bc.build_model()
    bc_params = {"stage1": saved["stage1"], "stage2": saved["stage2"]}

    # Both rollouts: ego vs the same teammate (the original comedi in seat 1),
    # argmax everywhere, identical reset key -> deterministic and comparable.
    ret_expert, ret_bc, first_div, identical = [], [], [], 0
    for ep in range(args.episodes):
        key = jax.random.PRNGKey(args.seed + ep)
        ep_e = collect_episode(
            key,
            base_env,
            [(orig.params, orig.policy_cls), (orig.params, orig.policy_cls)],
            max_episode_steps=job.env.rollout_length,
            greedy=True,
        )
        ep_b = collect_episode(
            key,
            base_env,
            [(bc_params, bc), (orig.params, orig.policy_cls)],
            max_episode_steps=job.env.rollout_length,
            greedy=True,
        )
        a_e = np.asarray(ep_e.actions[0])
        a_b = np.asarray(ep_b.actions[0])
        ret_expert.append(float(ep_e.returns()[0]))
        ret_bc.append(float(ep_b.returns()[0]))
        n = min(len(a_e), len(a_b))
        diverge = next((t for t in range(n) if int(a_e[t]) != int(a_b[t])), None)
        if diverge is None and len(a_e) == len(a_b):
            identical += 1
            first_div.append(len(a_e))
        else:
            first_div.append(diverge if diverge is not None else n)

    fd = np.asarray(first_div)
    print("\n===== BC-vs-ORIGINAL ACTION AGREEMENT (argmax, matched resets) =====", flush=True)
    print(f"  pairing:                     {args.pairing}", flush=True)
    print(f"  episodes:                    {args.episodes}", flush=True)
    print(f"  return  original ego:        {np.mean(ret_expert):.2f}", flush=True)
    print(f"  return  BC ego:              {np.mean(ret_bc):.2f}", flush=True)
    print(
        f"  first-divergence step:       mean {fd.mean():.1f}  median {np.median(fd):.0f}  "
        f"min {fd.min()}  max {fd.max()}",
        flush=True,
    )
    print(
        f"  episodes BC matched fully:   {identical}/{args.episodes} "
        f"({100 * identical / args.episodes:.0f}%)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
