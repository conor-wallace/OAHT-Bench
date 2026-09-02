"""Execute a :class:`~oaht_bench.configs.job.TrainingJob` (§3.1, §6).

Both trajectory-view baselines are two-stage: stage 1 learns a teammate
representation, stage 2 trains the policy against a frozen encoder. They differ
only in what the encoder reads and how its output reaches the policy, so one
loop drives both and the differences live in :mod:`~oaht_bench.offline.liam` and
:mod:`~oaht_bench.offline.tao`.

Follows the conventions teammate generation established: the config's content
hash names the run directory, the fully-resolved config is written into it, the
run refuses to start if it would overwrite an artifact, and **parameters are
saved before anything is reported** — a charting bug must not be able to discard
a finished run.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from oaht_bench.configs.job import TrainingJob

log = logging.getLogger(__name__)

#: Baselines this runner can train. The roster in ``BaselineName`` is larger;
#: the rest raise rather than silently training something else.
SUPPORTED = ("liam", "meliba", "omis", "tao", "bc")


def _resolve_dims(cfg, obs_dim: int, action_dim: int):
    """Return a copy of the offline config with dataset dims on the network config.

    A :class:`~oaht_bench.offline.registry.BaseAhtPolicy` is built from the config
    alone, so ``obs_dim``/``action_dim`` -- which come from the dataset -- are
    resolved onto ``config.network`` up front rather than threaded as arguments.
    """
    return cfg.model_copy(
        update={
            "network": cfg.network.model_copy(update={"obs_dim": obs_dim, "action_dim": action_dim})
        }
    )


def run(job: TrainingJob) -> Path:
    """Train one baseline and return the run directory."""
    import jax

    from oaht_bench.common.logging import RunLogger, nonfatal
    from oaht_bench.configs import save_job
    from oaht_bench.dataset.dataset import Dataset
    from oaht_bench.offline import get_policy

    if job.baseline not in SUPPORTED:
        raise NotImplementedError(
            f"baseline={job.baseline!r} has no runner yet; implemented: "
            f"{sorted(SUPPORTED)}. The roster in BaselineName is the plan, not "
            f"what exists."
        )

    run_dir = Path(job.run_dir())
    existing = run_dir / "params.pkl"
    if existing.exists():
        raise FileExistsError(
            f"{existing} already exists and would be overwritten. Delete "
            f"{run_dir} to retrain, or change the job's label. (The directory "
            f"name includes the config hash, so an identical config always "
            f"resolves here.)"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_job(job, run_dir / "job.json", minimal=False)

    cfg = job.offline
    dataset = Dataset(
        job.dataset_path,
        context_length=cfg.context_length,
        stride=cfg.stride,
        normalize=cfg.normalize_observations,
    )
    action_dim = dataset.action_dim
    log.info(
        "dataset %s -> %d windows, %d teammates, obs_dim %d, action_dim %d",
        job.dataset_path,
        len(dataset.windows),
        len(dataset.index.teammates),
        dataset.obs_dim,
        action_dim,
    )

    np_rng = np.random.default_rng(job.seed)
    rng = jax.random.PRNGKey(job.seed)

    with RunLogger(
        run_dir,
        use_wandb=job.logging.use_wandb,
        wandb_project=job.logging.wandb_project,
        wandb_entity=job.logging.wandb_entity,
        config=json.loads(job.canonical_json()),
        verbose=job.logging.verbose,
    ) as logger:
        resolved = _resolve_dims(cfg, dataset.obs_dim, action_dim)
        policy = get_policy(resolved)(resolved)
        policy.build_model()
        policy.prepare(dataset, logger, rng=rng, np_rng=np_rng)

        log.info("stage 1: %d steps", cfg.stage1_steps)
        stage1_params = policy.train_stage_1()
        log.info("stage 2: %d steps", cfg.stage2_steps)
        stage2_params = policy.train_stage_2(stage1_params)

        # Save before reporting. A charting failure after a long run must not
        # discard it -- the lesson from teammate generation.
        # The normalisation travels with the parameters: a policy trained on
        # standardised observations is wrong without it at rollout.
        out: dict[str, Any] = {
            "stage1": stage1_params,
            "stage2": stage2_params,
            "normalization": dataset.norm,
        }
        with (run_dir / "params.pkl").open("wb") as fh:
            pickle.dump(jax.device_get(out), fh)

        # Evaluation: the first number that says whether the policy plays, as
        # opposed to predicting dataset actions. Non-fatal because parameters are
        # already on disk -- a failure here costs a metric, not the run.
        eval_scores, eval_skipped = None, None
        if "population_run" not in dataset.meta:
            # Distinguish "no population to play against" from "evaluation
            # crashed": both leave eval null, and only one is a bug.
            eval_skipped = (
                "dataset metadata has no population_run, so there is no teammate "
                "population to roll out against"
            )
            log.warning("skipping evaluation: %s", eval_skipped)
        else:
            with nonfatal(f"{job.baseline} evaluation rollouts"):
                eval_scores = _evaluate(
                    job,
                    dataset.batch,
                    dataset.windows,
                    stage1_params,
                    stage2_params,
                    action_dim,
                    logger,
                )

        with nonfatal(f"{job.baseline} post-training summary"):
            (run_dir / "training_summary.json").write_text(
                json.dumps(
                    {
                        "baseline": job.baseline,
                        "windows": len(dataset.windows),
                        "teammates": len(dataset.index.teammates),
                        "obs_dim": dataset.obs_dim,
                        "action_dim": action_dim,
                        "stage1_steps": cfg.stage1_steps,
                        "stage2_steps": cfg.stage2_steps,
                        "eval_skipped": eval_skipped,
                        "eval": None
                        if eval_scores is None
                        else {
                            "mean_return": eval_scores.mean_return,
                            "worst_teammate_return": eval_scores.worst_teammate_return,
                            "per_teammate": eval_scores.per_teammate,
                            "target_return": eval_scores.target_return,
                        },
                    },
                    indent=2,
                )
                + "\n"
            )

    return run_dir


def _evaluate(job: TrainingJob, batch, windows, stage1_params, stage2_params, action_dim, logger):
    """Roll the trained policy against the population its dataset came from.

    Held-out teammates would be the stronger test (§8) and are not available
    yet, so this measures in-distribution coordination: the population the data
    was collected against. Recorded as such rather than presented as
    generalisation.
    """
    import jax

    from oaht_bench.common.save_load_utils import load_train_run
    from oaht_bench.configs import load_job
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.models.bc_agent import BcAgent
    from oaht_bench.models.liam_agent import LiamAgent
    from oaht_bench.models.meliba_agent import MelibaAgent
    from oaht_bench.models.omis_agent import OmisAgent
    from oaht_bench.models.tao_agent import TaoAgent
    from oaht_bench.offline.evaluate import dataset_target_return, evaluate_agent
    from oaht_bench.population import artifact_dir, population_from_run, released_members

    cfg = job.offline
    pop_run = Path(batch.meta["population_run"])
    run_dir = pop_run.parent.parent if pop_run.name == "saved_train_run" else pop_run
    gen_job = load_job(run_dir / "job.json")
    env = LogWrapper(make_env(job.env.env_name, job.env.env_kwargs()))
    loaded = population_from_run(gen_job, load_train_run(str(artifact_dir(run_dir))), env)
    members = released_members(gen_job, loaded.pop_size)

    resolved = _resolve_dims(cfg, windows.obs_dim, action_dim)
    params = {"stage1": stage1_params, "stage2": stage2_params}
    target = dataset_target_return(batch)
    cond_target = target if windows.norm is None else windows.norm.apply_rtg(target)

    # Every offline baseline is a ReturnConditionedAgent: the rolling window / RTG
    # bookkeeping lives in the agent, so all of them evaluate through the shared
    # vmapped run_episodes. The window transform and conditioning target are baked
    # into the agent, so evaluate_agent needs nothing baseline-specific.
    agent_classes = {
        "liam": LiamAgent,
        "tao": TaoAgent,
        "meliba": MelibaAgent,
        "omis": OmisAgent,
        "bc": BcAgent,
    }
    agent = agent_classes[resolved.network.architecture](
        resolved,
        context_length=cfg.context_length,
        target_return=cond_target,
        normalization=windows.norm,
    )
    agent.build_model()
    scores = evaluate_agent(
        agent,
        params,
        env,
        loaded,
        members,
        rng=jax.random.PRNGKey(job.seed + 1),
        target_return=cond_target,
        max_episode_steps=job.env.rollout_length,
        num_episodes=job.offline.eval_episodes,
    )
    for m, v in scores.per_teammate.items():
        logger.log_item(f"Eval/Return_teammate_{m}", v)
    logger.log_item("Eval/MeanReturn", scores.mean_return)
    logger.log_item("Eval/WorstTeammateReturn", scores.worst_teammate_return)
    logger.commit()
    log.info("Evaluation:\n%s", scores.describe())
    return scores
