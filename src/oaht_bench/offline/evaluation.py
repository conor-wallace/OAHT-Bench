"""Standalone evaluation of already-trained checkpoints (:class:`EvaluationJob`, §8).

For when training already happened -- possibly hours ago, possibly on a
different machine -- and the only thing left is scoring it, or re-scoring it
against a *different* held-out population than the one baked into its own
dataset's split. Reuses the exact same :func:`~oaht_bench.offline.evaluate.
evaluate_agent_against` / :func:`~oaht_bench.offline.incontext_eval.
evaluate_incontext` paths training's own final evaluation calls
(:func:`oaht_bench.offline.runner._evaluate`), so a number from this job type
means the same thing as one from ``training_summary.json`` -- including
``mate_action_acc``, which needed no extra plumbing here: it comes for free
from ``evaluate_agent_against`` once the agent is rebuilt.

**Seen vs. unseen.** ``checkpoint_paths`` each carry their own training
dataset, so "seen" (in-distribution) teammates are that dataset's own
``train`` split (:func:`~oaht_bench.offline.runner._teammate_policies`).
``heldout_population_path`` is a *separate* population -- released members
never part of any checkpoint's collection -- so "unseen" is common across
every checkpoint evaluated in one job, letting several baselines be compared
against the identical distribution-shift target. ``seen_unseen_ratios``
(OMIS's graded protocol, §8) picks how many of each to sample per point on
the curve: ``"S:U"`` draws (up to) ``S`` seen + ``U`` unseen teammates,
without replacement, and scores the union with one ``evaluate_agent_against``
call -- a ratio of *teammate counts* in the evaluation roster, not of
episodes within a rollout.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _parse_ratio(spec: str) -> tuple[int, int]:
    seen_s, unseen_s = spec.split(":")
    return int(seen_s), int(unseen_s)


def _select_teammates(rng, pool: list, n: int) -> list:
    """Up to ``n`` teammates from ``pool``, without replacement -- fewer than
    ``n`` available is not an error (a small held-out set is still worth
    scoring), just fewer teammates than the ratio nominally asks for."""
    if n <= 0 or not pool:
        return []
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in idx]


def load_trained_agent(run_dir: Path, *, dataset_path: str | None = None):
    """Rebuild ``(job, dataset, agent, all_params)`` from a finished training run.

    ``all_params`` keeps its full seed axis (num_seeds > 1) -- callers slice per
    seed themselves, the same contract ``offline.runner._evaluate`` uses, so
    evaluation here is seed-averaged the same way training's own final eval is.
    ``dataset_path`` overrides the config's recorded path, for a checkpoint moved
    between machines whose ``job.json`` no longer resolves where it was trained.
    """
    from oaht_bench.configs import load_job
    from oaht_bench.dataset.dataset import Dataset
    from oaht_bench.offline.evaluate import dataset_target_return
    from oaht_bench.offline.runner import _resolve_dims, agent_classes

    job = load_job(run_dir / "job.json")
    cfg = job.offline
    dataset = Dataset(
        dataset_path or job.dataset_path,
        context_length=cfg.context_length,
        stride=cfg.stride,
        normalize=cfg.normalize_observations,
    )
    with (run_dir / "params.pkl").open("rb") as fh:
        raw = pickle.load(fh)
    all_params = {"stage1": raw["stage1"], "stage2": raw["stage2"]}

    resolved = _resolve_dims(cfg, dataset.obs_dim, dataset.action_dim)
    target = dataset_target_return(dataset.batch)
    cond_target = target if dataset.windows.norm is None else dataset.windows.norm.apply_rtg(target)
    agent = agent_classes()[resolved.network.architecture](
        resolved,
        context_length=cfg.context_length,
        target_return=cond_target,
        normalization=dataset.windows.norm,
    )
    agent.build_model()
    return job, dataset, agent, all_params, cond_target


def _unseen_roster(heldout_population_path: str, env) -> list:
    """The ``self``/``conf`` policies of one released population, as a teammate
    list -- the same restriction :func:`~oaht_bench.offline.runner._teammate_policies`
    uses (a ``br`` is a designed ego, never a partner)."""
    from oaht_bench.population.pooled_crossplay import build_roster

    roster = build_roster([Path(heldout_population_path)], env)
    return [
        (f"{e.generator}:{e.member}:{e.role}", e.params, e.policy_cls)
        for e in roster
        if e.role in ("self", "conf")
    ]


def _score(agent, all_params, env, teammates, *, job, ns, cond_target, rng_base, incontext):
    """Across-seed mean return (+ mate_action_acc where available), against one
    fixed teammate set -- the same aggregation ``offline.runner._evaluate``'s
    ``score``/``_score_incontext`` closures do, kept as a free function here so
    an :class:`EvaluationJob` can call it without importing training internals.
    """
    import jax

    if incontext:
        from oaht_bench.offline.incontext_eval import evaluate_incontext

        seed_pt, seed_means = [], []
        for s in range(ns):
            p = all_params if ns == 1 else jax.tree.map(lambda x: x[s], all_params)  # noqa: B023
            mean, _curve, _anc, _floor = evaluate_incontext(
                agent,
                p,
                env,
                teammates,
                max_episode_steps=job.env.rollout_length,
                num_episodes=job.num_episodes,
                ocw_size=job.offline.context_trajectories,
                obs_dim=agent.obs_dim,
                rng=jax.random.PRNGKey(rng_base + s),
            )
            seed_pt.append(mean)
            seed_means.append(float(np.mean(list(mean.values()))))
        labels = list(seed_pt[0])
        per_teammate = {t: float(np.mean([m[t] for m in seed_pt])) for t in labels}
        return {"mean_return": float(np.mean(seed_means)), "per_teammate": per_teammate}

    from oaht_bench.offline.evaluate import evaluate_agent_against

    per_seed = [
        evaluate_agent_against(
            agent,
            all_params if ns == 1 else jax.tree.map(lambda x: x[s], all_params),  # noqa: B023
            env,
            teammates,
            rng=jax.random.PRNGKey(rng_base + s),
            target_return=cond_target,
            max_episode_steps=job.env.rollout_length,
            num_episodes=job.num_episodes,
        )
        for s in range(ns)
    ]
    labels = list(per_seed[0].per_teammate)
    out = {
        "mean_return": float(np.mean([sc.mean_return for sc in per_seed])),
        "per_teammate": {
            t: float(np.mean([sc.per_teammate[t] for sc in per_seed])) for t in labels
        },
    }
    if per_seed[0].mate_action_acc is not None:
        out["mate_action_acc"] = float(np.mean([sc.mate_action_acc for sc in per_seed]))
        out["mate_action_floor"] = float(np.mean([sc.mate_action_floor for sc in per_seed]))
    return out


def run(job) -> Path:
    """Evaluate every ``checkpoint_paths`` entry against graded seen:unseen
    teammate mixes and write one JSON report."""
    from oaht_bench.configs import save_job
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.offline.runner import _teammate_policies

    run_dir = Path(job.run_dir())
    run_dir.mkdir(parents=True, exist_ok=True)
    save_job(job, run_dir / "job.json", minimal=False)

    env = LogWrapper(make_env(job.env.env_name, job.env.env_kwargs()))
    unseen_pool = _unseen_roster(job.heldout_population_path, env)
    if not unseen_pool:
        raise ValueError(
            f"heldout_population_path={job.heldout_population_path!r} has no self/conf "
            f"members -- nothing to seat as an unseen teammate."
        )

    rng = np.random.default_rng(job.seed)
    report: dict = {"checkpoints": {}}
    for ckpt in job.checkpoint_paths:
        run_path = Path(ckpt)
        log.info("evaluating %s", run_path)
        ckpt_job, dataset, agent, all_params, cond_target = load_trained_agent(run_path)
        seen_pool = _teammate_policies(dataset.batch, env, which="train")
        ns = int(ckpt_job.num_seeds)
        incontext = ckpt_job.offline.network.architecture == "tao"

        per_ratio = {}
        for spec in job.seen_unseen_ratios:
            n_seen, n_unseen = _parse_ratio(spec)
            teammates = _select_teammates(rng, seen_pool, n_seen) + _select_teammates(
                rng, unseen_pool, n_unseen
            )
            if not teammates:
                log.warning(
                    "ratio %s: no teammates available (seen=%d, unseen=%d) -- skipped",
                    spec,
                    len(seen_pool),
                    len(unseen_pool),
                )
                continue
            per_ratio[spec] = _score(
                agent,
                all_params,
                env,
                teammates,
                job=job,
                ns=ns,
                cond_target=cond_target,
                rng_base=job.seed,
                incontext=incontext,
            )
        report["checkpoints"][str(run_path)] = {
            "baseline": ckpt_job.baseline,
            "num_seeds": ns,
            "seen_pool_size": len(seen_pool),
            "unseen_pool_size": len(unseen_pool),
            "per_ratio": per_ratio,
        }

    out_path = run_dir / "evaluation_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    log.info("wrote %s", out_path)
    return run_dir
