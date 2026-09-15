"""Single-pairing offline diagnostic -- the gate before pooled/held-out AHT eval.

Collect one competent ``(ego, teammate)`` pairing, train an offline baseline on
just that data, and evaluate the trained ego against that same teammate. It
isolates *trainer capability* from the two things that otherwise confound a
pooled held-out number: convention mixing and zero-shot generalisation. If a
baseline cannot reach a competent fraction of the pairing's own return here --
on a single clean convention whose expert data it was handed -- then the offline
trainer/architecture is the bottleneck, and pooled/held-out numbers are not yet
interpretable.

This is how the Hanabi offline under-provisioning was found: at LBF's inherited
``context_length=20`` / ``hidden_dim=32`` a clean convention (comedi:0 self-play,
ceiling ~19.7) capped at ~4% of ceiling; extending the context toward the full
~70-step episode was a ~4x lever. See docs/tuning_record.md.

    uv run python scripts/diagnose_single_pairing.py \
        --config configs/hanabi/training/pooled_expert_scaled/bc.json \
        --pairing comedi:0:self --episodes 6000

``--pairing`` names the *teammate* as ``generator:member:role``. The ego is that
teammate's designed partner: itself for a ``self`` member, the matching ``br`` for
a ``conf``. ``--steps``/``--context``/``--episodes`` override the config for a quick
pass; omit them to run the config as written.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np


class _Logger:
    """Minimal logger the trainer expects; prints action accuracy periodically."""

    def __init__(self, every: int = 5000):
        self.every = every
        self.last_acc: float | None = None

    def log_item(self, name, value, train_step=None, **_):
        if name.endswith("action_accuracy"):
            self.last_acc = float(value)
            if train_step is not None and self.every and train_step % self.every == 0:
                print(f"    step {train_step}: action_acc={float(value):.3f}", flush=True)

    def commit(self, *a, **k):
        pass

    def log(self, *a, **k):
        pass


def _find_ego_and_teammate(roster, spec: str):
    """Resolve ``generator:member:role`` to (ego_entry, teammate_entry).

    The teammate is the named entry; the ego is its designed partner -- the same
    entry for a self-play member, the matching ``br`` for a confederate.
    """
    gen, member, role = spec.split(":")
    member = int(member)

    def get(g, m, r):
        return next(
            (e for e in roster if e.generator == g and int(e.member) == m and e.role == r),
            None,
        )

    mate = get(gen, member, role)
    if mate is None:
        raise SystemExit(f"teammate {spec!r} not found in roster")
    if role == "self":
        ego = mate
    elif role == "conf":
        ego = get(gen, member, "br")
        if ego is None:
            raise SystemExit(f"no matching br for confederate {spec!r}")
    else:
        raise SystemExit(f"--pairing role must be 'self' or 'conf', got {role!r}")
    return ego, mate


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config",
        required=True,
        help="A training-job config; supplies env, baseline, and offline hyperparameters.",
    )
    ap.add_argument(
        "--pairing",
        default="comedi:0:self",
        help="Teammate as generator:member:role (default comedi:0:self).",
    )
    ap.add_argument(
        "--populations",
        nargs="+",
        default=[
            "populations/hanabi/brdiv",
            "populations/hanabi/comedi",
            "populations/hanabi/fcp",
            "populations/hanabi/lbrdiv",
        ],
        help="Released population run directories to build the roster from.",
    )
    ap.add_argument(
        "--episodes", type=int, default=6000, help="Episodes of the pairing to collect."
    )
    ap.add_argument("--eval-episodes", type=int, default=50, help="Episodes for the final eval.")
    ap.add_argument(
        "--steps", type=int, default=None, help="Override stage2_steps (for a quick pass)."
    )
    ap.add_argument("--context", type=int, default=None, help="Override context_length.")
    ap.add_argument(
        "--vault",
        default=None,
        help="Where to write the collected vault (default: a temp dir, removed after).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--greedy",
        action="store_true",
        help="Eval the ego by argmax (teammate still samples). Diagnostic for whether "
        "a high-accuracy policy is sunk by sampling noise vs a train/deploy mismatch.",
    )
    args = ap.parse_args()

    import jax

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
    from oaht_bench.offline import get_trainer
    from oaht_bench.offline.evaluate import dataset_target_return, evaluate_agent_against
    from oaht_bench.offline.runner import _resolve_dims
    from oaht_bench.population.pooled_crossplay import build_roster

    agents = {
        "bc": BcAgent,
        "liam": LiamAgent,
        "meliba": MelibaAgent,
        "omis": OmisAgent,
        "tao": TaoAgent,
    }

    job = load_job(args.config)
    cfg = job.offline
    if args.context is not None:
        cfg = cfg.model_copy(update={"context_length": args.context})
    if args.steps is not None:
        cfg = cfg.model_copy(update={"stage2_steps": args.steps})

    base_env = make_env(job.env.env_name, job.env.env_kwargs())
    env = LogWrapper(base_env)
    roster = build_roster([Path(p) for p in args.populations], env)
    ego, mate = _find_ego_and_teammate(roster, args.pairing)
    print(
        f"pairing: teammate={args.pairing}  ego={ego.generator}:{ego.member}:{ego.role}", flush=True
    )

    vault_dir = (
        Path(args.vault)
        if args.vault
        else Path(tempfile.mkdtemp(prefix="single_pairing_")) / "pairing.vlt"
    )

    # 1. Collect the single pairing (sequential, like the real collector).
    rng = jax.random.PRNGKey(args.seed)
    episodes, member_ids = [], []
    for i in range(args.episodes):
        rng, ep_rng = jax.random.split(rng)
        episodes.append(
            collect_episode(
                ep_rng,
                base_env,
                [(ego.params, ego.policy_cls), (mate.params, mate.policy_cls)],
                max_episode_steps=job.env.rollout_length,
                greedy=False,
            )
        )
        member_ids.append([int(mate.member), int(mate.member)])
        if (i + 1) % 1000 == 0:
            print(f"  collected {i + 1}/{args.episodes}", flush=True)
    ego_mean = float(np.mean([e.returns()[0] for e in episodes]))
    if vault_dir.exists():
        shutil.rmtree(vault_dir)
    write_vault(episodes, np.array(member_ids), vault_dir, ego_index=0, meta={"variant": "single"})

    # 2. Train the baseline through the real pipeline (single seed).
    ds = Dataset(
        str(vault_dir),
        context_length=cfg.context_length,
        stride=cfg.stride,
        normalize=cfg.normalize_observations,
    )
    print(
        f"windows={len(ds.windows)} obs_dim={ds.obs_dim} ego_mean(ceiling)={ego_mean:.2f}",
        flush=True,
    )
    resolved = _resolve_dims(cfg, ds.obs_dim, ds.action_dim)
    logger = _Logger()
    trainer = get_trainer(resolved)(resolved)
    trainer.build_model()
    trainer.prepare(
        ds,
        logger,
        rng=jax.random.PRNGKey(args.seed),
        np_rng=np.random.default_rng(args.seed),
        num_seeds=1,
    )
    print(
        f"training {job.baseline} (stage2_steps={cfg.stage2_steps}, context={cfg.context_length})...",
        flush=True,
    )
    s1 = trainer.train_stage_1()
    s2 = trainer.train_stage_2(s1)

    # 3. Evaluate the trained ego against the same teammate.
    target = dataset_target_return(ds.batch)
    cond = target if ds.windows.norm is None else ds.windows.norm.apply_rtg(target)
    agent = agents[job.baseline](
        resolved,
        context_length=cfg.context_length,
        target_return=cond,
        normalization=ds.windows.norm,
    )
    agent.build_model()
    sc = evaluate_agent_against(
        agent,
        {"stage1": s1, "stage2": s2},
        env,
        [(args.pairing, mate.params, mate.policy_cls)],
        rng=jax.random.PRNGKey(args.seed + 123),
        target_return=cond,
        max_episode_steps=job.env.rollout_length,
        num_episodes=args.eval_episodes,
        greedy=args.greedy,
    )
    ret = float(next(iter(sc.per_teammate.values())))

    if not args.vault:
        shutil.rmtree(vault_dir.parent, ignore_errors=True)

    print("\n===== SINGLE-PAIRING DIAGNOSTIC =====", flush=True)
    print(f"  baseline:               {job.baseline}", flush=True)
    print(f"  pairing:                {args.pairing}", flush=True)
    print(
        f"  eval mode:              {'argmax (greedy ego)' if args.greedy else 'sampled'}",
        flush=True,
    )
    print(f"  ceiling (dataset mean): {ego_mean:.2f}", flush=True)
    print(
        f"  action accuracy:        {logger.last_acc if logger.last_acc is not None else float('nan'):.3f}",
        flush=True,
    )
    print(
        f"  trained return:         {ret:.2f}  ({100 * ret / ego_mean:.0f}% of ceiling)", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
