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

**Seen vs. unseen.** Both come from the *same kind* of record: a dataset
collection's ``teammate_split.json`` (mirrored into its vault's metadata),
the authoritative record of which roster members were actually reserved as
held-out at collection time -- not "any population member not in my
training set," which says nothing about whether that member was a
deliberate generalisation target. ``checkpoint_paths`` each carry their own
training dataset, so "seen" is that dataset's own ``train`` split
(:func:`~oaht_bench.offline.runner._teammate_policies`). ``dataset_path`` is
one *dataset collection directory* (e.g.
``results/dataset_collection/pooled_lbf_20x20_expert-<hash>``, the directory
``teammate_split.json`` lives in, not a released ``populations/<env>/<gen>``
directory) whose recorded ``held_out`` roster is the "unseen" set, common
across every checkpoint evaluated in one job -- usually the checkpoint's own
dataset, for "did held-out generalisation change" as its only question. A
different dataset's split, for generalisation to a genuinely separate
collection, is a separate ``EvaluationJob`` rather than a second entry here:
merging two collections' held-out sets into one number would hide which
teammate came from which.

Every available seen and unseen teammate is scored, in one shot -- the same
"held-out primary, train contrast" shape as training's own final evaluation
(``offline.runner._evaluate``), not OMIS's graded seen:unseen ratio curve
(§8): that needs its own sampling design (a ratio of teammate *counts* is a
different question from a ratio of *episodes*, and either reading is a
larger, separate feature) and was cut rather than half-built.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def load_trained_agent(
    run_dir: Path, *, dataset_path: str | None = None, dataset_cache: dict | None = None
):
    """Rebuild ``(job, dataset, agent, all_params)`` from a finished training run.

    ``all_params`` keeps its full seed axis (num_seeds > 1) -- callers slice per
    seed themselves, the same contract ``offline.runner._evaluate`` uses, so
    evaluation here is seed-averaged the same way training's own final eval is.
    ``dataset_path`` overrides the config's recorded path, for a checkpoint moved
    between machines whose ``job.json`` no longer resolves where it was trained.

    ``dataset_cache``, keyed by ``(path, context_length, stride, normalize)`` --
    everything that actually determines the windowed result -- lets several
    checkpoints trained on the same dataset (the common case: one dataset, one
    baseline per architecture) share ONE load instead of re-reading and
    re-windowing the same vault once per checkpoint.
    """
    from oaht_bench.configs import load_job
    from oaht_bench.dataset.dataset import Dataset
    from oaht_bench.offline.evaluate import dataset_target_return
    from oaht_bench.offline.runner import _resolve_dims, agent_classes

    job = load_job(run_dir / "job.json")
    cfg = job.offline
    key = (
        dataset_path or job.dataset_path,
        cfg.context_length,
        cfg.stride,
        cfg.normalize_observations,
    )
    if dataset_cache is not None and key in dataset_cache:
        dataset = dataset_cache[key]
    else:
        dataset = Dataset(key[0], context_length=key[1], stride=key[2], normalize=key[3])
        if dataset_cache is not None:
            dataset_cache[key] = dataset
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


def _unseen_roster(dataset_path: str, env) -> list:
    """The "unseen" teammate list: the ``held_out`` roster ``dataset_path``'s
    own collection split recorded.

    Reads only the vault's metadata (:class:`~oaht_bench.dataset.vault.VaultReader`),
    not the full episode data -- the roster is all this needs, and a pooled
    dataset's vault can be large. ``_teammate_policies`` reads ``batch.meta``
    only, so a bare object exposing that attribute stands in for the
    :class:`~oaht_bench.dataset.schema.EpisodeBatch` it normally takes.
    """
    import types

    from oaht_bench.dataset.vault import VaultReader
    from oaht_bench.offline.runner import _teammate_policies

    reader = VaultReader(f"{dataset_path}/dataset.vlt")
    batch = types.SimpleNamespace(meta=reader.meta)
    return _teammate_policies(batch, env, which="held_out")


def _score(
    agent, all_params, env, teammates, *, job, ns, cond_target, rng_base, incontext, ocw_size=None
):
    """Across-seed mean return (+ mate_action_acc where available), against one
    fixed teammate set -- the same aggregation ``offline.runner._evaluate``'s
    ``score``/``_score_incontext`` closures do, kept as a free function here so
    an :class:`EvaluationJob` can call it without importing training internals.

    ``job`` is the :class:`EvaluationJob` (has ``.env``/``.num_episodes``, no
    ``.offline``) -- ``ocw_size`` (TAO's OCW capacity) is a property of how the
    *checkpoint* was trained, not of this evaluation, so the caller passes it
    explicitly from that checkpoint's own training config rather than this
    function reaching for ``job.offline``, which doesn't exist here.
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
                ocw_size=ocw_size,
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
    """Evaluate every ``checkpoint_paths`` entry against its unseen (and, if
    available, seen) teammate set and write one JSON report."""
    from oaht_bench.configs import save_job
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.offline.runner import _teammate_policies

    run_dir = Path(job.run_dir())
    run_dir.mkdir(parents=True, exist_ok=True)
    save_job(job, run_dir / "job.json", minimal=False)

    env = LogWrapper(make_env(job.env.env_name, job.env.env_kwargs()))
    unseen_pool = _unseen_roster(job.dataset_path, env)
    if not unseen_pool:
        raise ValueError(
            f"dataset_path={job.dataset_path!r} has no held_out self/conf "
            f"members -- nothing to seat as an unseen teammate."
        )

    dataset_cache: dict = {}
    report: dict = {"checkpoints": {}}
    for ckpt in job.checkpoint_paths:
        run_path = Path(ckpt)
        log.info("evaluating %s", run_path)
        ckpt_job, dataset, agent, all_params, cond_target = load_trained_agent(
            run_path, dataset_cache=dataset_cache
        )
        seen_pool = _teammate_policies(dataset.batch, env, which="train")
        ns = int(ckpt_job.num_seeds)
        incontext = ckpt_job.offline.network.architecture == "tao"
        score_kwargs = dict(
            agent=agent,
            all_params=all_params,
            env=env,
            job=job,
            ns=ns,
            cond_target=cond_target,
            rng_base=job.seed,
            incontext=incontext,
            ocw_size=ckpt_job.offline.context_trajectories if incontext else None,
        )

        # Unseen (primary) then seen (contrast), the same order and shape
        # _evaluate reports: a held-out score is only readable as
        # generalisation, not raw competence, next to the in-distribution one.
        entry = {
            "baseline": ckpt_job.baseline,
            "num_seeds": ns,
            "seen_pool_size": len(seen_pool),
            "unseen_pool_size": len(unseen_pool),
            "unseen": _score(teammates=unseen_pool, **score_kwargs),
        }
        if seen_pool:
            entry["seen"] = _score(teammates=seen_pool, **score_kwargs)
            entry["generalization_gap"] = float(
                entry["seen"]["mean_return"] - entry["unseen"]["mean_return"]
            )
        report["checkpoints"][str(run_path)] = entry

    out_path = run_dir / "evaluation_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    log.info("wrote %s", out_path)
    return run_dir
