"""CLI wrapper around the production mate_action_acc path, for a quick check
against one checkpoint without writing an :class:`EvaluationJob` config.

The metric itself -- decode the teammate's predicted action from the same
representation the policy conditions on, compare to what the teammate really
did -- now lives in production code: ``ReturnConditionedAgent.mate_action_logits``
(overridden by LIAM/MeLIBA/OMIS) and ``offline.evaluate.evaluate_agent_against``,
wired into every training run's own final evaluation
(``offline.runner._evaluate``) and into standalone re-evaluation
(``offline.evaluation``, the ``EvaluationJob`` runner). This script used to
carry its own copy of the window-rollout logic; that duplicate drifted from
production twice (a missing illegal-action mask, a missing freeze-on-episode-
end guard that inflated returns 30-40%) before being replaced by the shared
path, which is what this script now calls. See docs/tuning_record.md.

For TAO, whose ancillary-decoder probe needs the teammate's own observation
stream (a different information set -- see ``incontext_eval._ancillary_mate_action_acc``),
use ``EvaluationJob`` or call ``offline.incontext_eval.evaluate_incontext``
directly; this script only covers the ego-only-information baselines.

Usage:
    uv run python scripts/mechanism_probes.py \\
        --run-dir results/training/liam_pooled_expert_lbf_12x12-e5565df35528 \\
        --episodes 100 --seed-index 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

from oaht_bench.envs import make_env
from oaht_bench.envs.log_wrapper import LogWrapper
from oaht_bench.offline.evaluate import evaluate_agent_against, resolve_target_returns
from oaht_bench.offline.evaluation import load_trained_agent
from oaht_bench.offline.runner import _teammate_policies


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument(
        "--dataset",
        default=None,
        help="override the config's dataset_path (a .vlt) -- for a checkpoint moved between "
        "machines, where job.json's recorded path no longer resolves",
    )
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument(
        "--seed-index", type=int, default=0, help="which seed to evaluate (num_seeds > 1)"
    )
    ap.add_argument("--seed", type=int, default=0, help="rng seed for the rollout")
    args = ap.parse_args()

    job, dataset, agent, all_params = load_trained_agent(args.run_dir, dataset_path=args.dataset)
    if job.offline.network.architecture == "tao":
        raise SystemExit(
            "TAO's mate-action probe needs the teammate's own observation stream and lives "
            "in offline.incontext_eval.evaluate_incontext (a different information set, see "
            "this script's docstring) -- use an EvaluationJob or call it directly."
        )
    ns = int(job.num_seeds)
    params = all_params if ns == 1 else jax.tree.map(lambda x: x[args.seed_index], all_params)

    env = LogWrapper(make_env(job.env.env_name, job.env.env_kwargs()))
    print(f"baseline={job.baseline}  env={job.env.name}  seed_index={args.seed_index}")
    for split in ("train", "held_out"):
        teammates = _teammate_policies(dataset.batch, env, which=split)
        if not teammates:
            print(f"  {split}: no teammates in this split")
            continue
        target_returns = resolve_target_returns(
            dataset.batch.meta, teammates, norm=dataset.windows.norm, fallback_batch=dataset.batch
        )
        scores = evaluate_agent_against(
            agent,
            params,
            env,
            teammates,
            rng=jax.random.PRNGKey(args.seed),
            target_returns=target_returns,
            max_episode_steps=job.env.rollout_length,
            num_episodes=args.episodes,
        )
        for label, v in scores.per_teammate.items():
            print(f"    {label}: online mean ego return={v:.4f}")
        if scores.mate_action_acc is not None:
            print(
                f"  {split:10s} online mate_action_acc={scores.mate_action_acc:.3f}  "
                f"(modal floor={scores.mate_action_floor:.3f})"
            )
        else:
            print(f"  {split:10s} mean_return={scores.mean_return:.4f}  (no mate model -- BC)")


if __name__ == "__main__":
    main()
