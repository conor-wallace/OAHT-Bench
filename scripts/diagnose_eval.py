#!/usr/bin/env python3
"""Diagnostics for "why do all the baselines score the same?" -- run from saved
params.pkl, no retraining.

Two questions, per finished run:

1. **In-distribution vs held-out.** Roll each trained ego against the *train*
   teammates as well as the held-out *test* ones. A large train>>test gap is a real
   generalisation failure the methods should differ on; a near-zero gap (train ~=
   test ~= low) means nothing teammate-specific is being learned even on seen
   partners -- so the flatness is the task/conditioning, not generalisation.

2. **Teammate-representation ablation** (modeling methods only). Re-evaluate with the
   teammate representation zeroed (LIAM/OMIS embedding, MeLIBA belief, TAO context).
   If the held-out return barely drops, the policy is ignoring the partner -- i.e.
   it is BC in disguise, which is exactly why it scores like BC. This is the direct
   test of the partner-inference-shortcut hypothesis.

    uv run python scripts/diagnose_eval.py                       # all finished runs
    uv run python scripts/diagnose_eval.py --episodes 20 --all-seeds
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from oaht_bench.configs import load_job
from oaht_bench.dataset.dataset import Dataset
from oaht_bench.envs import make_env
from oaht_bench.envs.log_wrapper import LogWrapper
from oaht_bench.models.bc_agent import BcAgent
from oaht_bench.models.liam_agent import LiamAgent
from oaht_bench.models.meliba_agent import MelibaAgent
from oaht_bench.models.omis_agent import OmisAgent
from oaht_bench.models.tao_agent import TaoAgent
from oaht_bench.offline.evaluate import dataset_target_return, evaluate_agent_against
from oaht_bench.offline.runner import _resolve_dims
from oaht_bench.population.loading import build_roster

BASELINES = ["bc", "liam", "meliba", "omis", "tao"]
VARIANTS = ["worst", "mixed", "expert"]


# --- ablated agents: identical inference, teammate representation zeroed ----------
class LiamAgentAblate(LiamAgent):
    def act(self, params, rtg, obs, actions, *, timesteps, mask):
        z = self.encoder.apply(
            params["stage1"]["encoder"],
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            mask=mask,
            train=False,
        )
        z = jnp.zeros_like(z)
        return self.network.apply(
            params["stage2"],
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            embedding=z,
            mask=mask,
            train=False,
        )


class OmisAgentAblate(OmisAgent):
    def act(self, params, rtg, obs, actions, *, timesteps, mask):
        z = self.encoder.apply(
            params["stage1"]["encoder"],
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            mask=mask,
            train=False,
        )
        z = jnp.zeros_like(z)
        return self.actor.apply(
            params["stage2"],
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            embedding=z,
            mask=mask,
            train=False,
        )


class MelibaAgentAblate(MelibaAgent):
    def act(self, params, rtg, obs, actions, *, timesteps, mask):
        cm, cl, mm, ml = self.encoder.apply(
            params["stage1"]["encoder"],
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            mask=mask,
            train=False,
        )
        belief = jnp.zeros_like(jnp.concatenate([cm, cl, mm, ml], axis=-1))
        return self.network.apply(
            params["stage2"],
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            belief=belief,
            mask=mask,
            train=False,
        )


class TaoAgentAblate(TaoAgent):
    def act(self, params, rtg, obs, actions, *, timesteps, mask):
        stage2 = params["stage2"]
        context = jnp.zeros_like(stage2["context"])
        return self.network.apply(
            stage2["policy"],
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            context=context,
            mask=mask,
            context_mask=stage2["context_mask"],
            train=False,
        )


AGENT = {
    "bc": BcAgent,
    "liam": LiamAgent,
    "meliba": MelibaAgent,
    "omis": OmisAgent,
    "tao": TaoAgent,
}
ABLATE = {
    "liam": LiamAgentAblate,
    "omis": OmisAgentAblate,
    "meliba": MelibaAgentAblate,
    "tao": TaoAgentAblate,
}


def _teammates(batch, env, which):
    """Train or test teammate policies (self/conf roles only), from meta['held_out']."""
    meta = batch.meta
    held = {g: {int(m) for m in ms} for g, ms in meta["held_out"].items()}
    pop_paths = meta.get("populations") or [meta["population_run"]]
    roster = build_roster([Path(p) for p in pop_paths], env)
    out = []
    for e in roster:
        if e.role not in ("self", "conf"):
            continue
        is_test = int(e.member) in held.get(e.generator, set())
        if (which == "test") == is_test:
            out.append((f"{e.generator}:{int(e.member)}:{e.role}", e.params, e.policy_cls))
    return out


def _seed_slice(tree, s, ns):
    return tree if ns == 1 else jax.tree.map(lambda x: x[s], tree)


def _latest_run(baseline, variant, root):
    dirs = glob.glob(os.path.join(root, f"{baseline}_pooled_{variant}_lbf_12x12-*"))
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "params.pkl"))]
    return max(dirs, key=os.path.getmtime) if dirs else None


def diagnose(run_dir, *, episodes, seeds):
    job = load_job(Path(run_dir) / "job.json")
    cfg = job.offline
    with open(Path(run_dir) / "params.pkl", "rb") as fh:
        saved = pickle.load(fh)
    ns = int(job.num_seeds)

    env = LogWrapper(make_env(job.env.env_name, job.env.env_kwargs()))
    ds = Dataset(
        job.dataset_path,
        context_length=cfg.context_length,
        stride=cfg.stride,
        normalize=cfg.normalize_observations,
    )
    resolved = _resolve_dims(cfg, ds.obs_dim, ds.action_dim)
    target = dataset_target_return(ds.batch)
    cond = target if ds.windows.norm is None else ds.windows.norm.apply_rtg(target)
    tm_train = _teammates(ds.batch, env, "train")
    tm_test = _teammates(ds.batch, env, "test")

    def build(cls):
        a = cls(
            resolved,
            context_length=cfg.context_length,
            target_return=cond,
            normalization=ds.windows.norm,
        )
        a.build_model()
        return a

    def mean_return(agent, teammates):
        vals = []
        target_returns = {label: cond for label, *_ in teammates}
        for s in range(ns if seeds == "all" else 1):
            p = {
                "stage1": _seed_slice(saved["stage1"], s, ns),
                "stage2": _seed_slice(saved["stage2"], s, ns),
            }
            sc = evaluate_agent_against(
                agent,
                p,
                env,
                teammates,
                rng=jax.random.PRNGKey(100 + s),
                target_returns=target_returns,
                max_episode_steps=job.env.rollout_length,
                num_episodes=episodes,
            )
            vals.append(sc.mean_return)
        return float(np.mean(vals))

    agent = build(AGENT[job.baseline])
    row = {"train": mean_return(agent, tm_train), "test": mean_return(agent, tm_test)}
    row["gap"] = row["train"] - row["test"]
    if job.baseline in ABLATE:
        row["test_ablated"] = mean_return(build(ABLATE[job.baseline]), tm_test)
        row["repr_use"] = row["test"] - row["test_ablated"]
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/training")
    ap.add_argument("--episodes", type=int, default=10, help="rollouts per teammate")
    ap.add_argument(
        "--all-seeds",
        dest="seeds",
        action="store_const",
        const="all",
        default="one",
        help="average over all trained seeds (default: seed 0 only, faster)",
    )
    args = ap.parse_args()

    hdr = f"{'run':16s} {'train':>7s} {'test':>7s} {'gap':>7s} {'test_abl':>9s} {'repr_use':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for v in VARIANTS:
        for b in BASELINES:
            rd = _latest_run(b, v, args.results)
            if rd is None:
                continue
            r = diagnose(rd, episodes=args.episodes, seeds=args.seeds)
            abl = f"{r['test_ablated']:9.3f}" if "test_ablated" in r else "       --"
            use = f"{r['repr_use']:+9.3f}" if "repr_use" in r else "       --"
            print(
                f"{b + '/' + v:16s} {r['train']:7.3f} {r['test']:7.3f} {r['gap']:+7.3f} {abl} {use}"
            )

    print(
        "\ngap  = train - test  (generalisation gap; ~0 means nothing teammate-specific learned)\n"
        "repr_use = test - test_ablated  (drop when the teammate representation is zeroed;\n"
        "           ~0 means the policy ignores the partner -> BC in disguise, the shortcut)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
