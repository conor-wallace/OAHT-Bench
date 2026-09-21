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

    A :class:`~oaht_bench.offline.registry.BaseAhtTrainer` is built from the config
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
    from oaht_bench.offline import get_trainer

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
        trainer = get_trainer(resolved)(resolved)
        trainer.build_model()
        trainer.prepare(dataset, logger, rng=rng, np_rng=np_rng, num_seeds=job.num_seeds)

        log.info("stage 1: %d steps", cfg.stage1_steps)
        stage1_params = trainer.train_stage_1()
        log.info("stage 2: %d steps", cfg.stage2_steps)
        stage2_params = trainer.train_stage_2(stage1_params)

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
        if "held_out" not in dataset.meta and "population_run" not in dataset.meta:
            # Distinguish "no teammates to play against" from "evaluation crashed":
            # both leave eval null, and only one is a bug. A split dataset carries
            # 'held_out' (the test teammates); a legacy single-population one carries
            # 'population_run'; this fixture-style dataset has neither.
            eval_skipped = (
                "dataset metadata has neither a train/test split ('held_out') nor a "
                "'population_run', so there is no teammate population to roll out against"
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
                        "eval": eval_scores,
                    },
                    indent=2,
                )
                + "\n"
            )

    return run_dir


def _teammate_policies(batch, env, which: str = "held_out") -> list:
    """The teammate policies evaluation rolls the trained ego against.

    With a train/test split (§8), ``which`` selects which side of the split to
    seat: ``"held_out"`` (default) is the **test** set -- the teammates collection
    never trained on, drawn from every listed population, the generalisation
    measurement the benchmark is about -- and ``"train"`` is its complement, the
    in-distribution teammates whose trajectories the ego *did* learn from. Scoring
    both is what separates "modelled the training population" from "generalised to
    unseen partners". Without a split (legacy single-population datasets, no
    ``held_out`` in meta) it falls back to that one population's released members as
    an in-distribution measurement, and a ``"train"`` request returns ``[]`` because
    there is no held-out complement to contrast against. Only ``self``/``conf``
    policies are seated as teammates; a ``br`` is a designed ego, never a partner --
    the same restriction collection uses.

    Returns ``[(label, mate_params, policy_cls)]``.
    """
    from oaht_bench.population.pooled_crossplay import build_roster

    meta = batch.meta
    if "held_out" in meta:
        held = {g: {int(m) for m in ms} for g, ms in meta["held_out"].items()}
        pop_paths = meta.get("populations") or [meta["population_run"]]
        roster = build_roster([Path(p) for p in pop_paths], env)
        want_test = which != "train"
        return [
            (f"{e.generator}:{int(e.member)}:{e.role}", e.params, e.policy_cls)
            for e in roster
            if e.role in ("self", "conf")
            and (int(e.member) in held.get(e.generator, set())) == want_test
        ]

    # Legacy split-less datasets carry only the in-distribution population; there is
    # no held-out complement, so a "train" contrast has nothing to seat.
    if which == "train":
        return []

    from oaht_bench.common.save_load_utils import load_train_run
    from oaht_bench.configs import load_job
    from oaht_bench.population import artifact_dir, population_from_run, released_members
    from oaht_bench.population.members import get_member_params

    pop_run = Path(meta["population_run"])
    run_dir = pop_run.parent.parent if pop_run.name == "saved_train_run" else pop_run
    gen_job = load_job(run_dir / "job.json")
    loaded = population_from_run(gen_job, load_train_run(str(artifact_dir(run_dir))), env)
    return [
        (int(m), get_member_params(loaded.params, int(m)), loaded.policy_cls)
        for m in released_members(gen_job, loaded.pop_size)
    ]


def _evaluate(job: TrainingJob, batch, windows, stage1_params, stage2_params, action_dim, logger):
    """Roll the trained policy against its teammates and report per-teammate return.

    Against the **held-out** test teammates when the dataset carries a train/test
    split (§8) -- the generalisation test the benchmark exists for -- or, for legacy
    split-less datasets, the collection population (in-distribution). See
    :func:`_teammate_policies`.
    """
    import jax

    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.models.bc_agent import BcAgent
    from oaht_bench.models.liam_agent import LiamAgent
    from oaht_bench.models.meliba_agent import MelibaAgent
    from oaht_bench.models.omis_agent import OmisAgent
    from oaht_bench.models.tao_agent import TaoAgent
    from oaht_bench.offline.evaluate import dataset_target_return, evaluate_agent_against

    cfg = job.offline
    env = LogWrapper(make_env(job.env.env_name, job.env.env_kwargs()))
    heldout_teammates = _teammate_policies(batch, env, which="held_out")
    if not heldout_teammates:
        raise ValueError(
            "no teammate policies resolved for evaluation -- the held-out set is "
            "empty. Check the dataset's 'held_out' meta and its populations."
        )
    train_teammates = _teammate_policies(batch, env, which="train")

    resolved = _resolve_dims(cfg, windows.obs_dim, action_dim)
    all_params = {"stage1": stage1_params, "stage2": stage2_params}
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

    # One EvalScores per trained seed. With num_seeds > 1 the parameter trees carry
    # a leading seed axis, so each seed is a slice; the held-out returns are then
    # reported as the across-seed mean (and std), which is the significance a single
    # seed cannot give -- teammate-to-teammate variance already swamps the ~0.03
    # gaps between baselines, so seeds are what make an ordering trustworthy.
    ns = int(job.num_seeds)
    import numpy as np

    # TAO (and OMIS later) adapt across episodes via an Opponent Context Window, so
    # they evaluate through the sequential in-context path; the within-episode
    # baselines (BC/LIAM/MeLIBA) keep the parallel per-episode eval unchanged.
    incontext = resolved.network.architecture in ("tao",)

    def _score_incontext(teammates, *, rng_base):
        """TAO/OMIS scoring: sequential episodes per teammate, OCW re-encoded each one.

        Same shape as :func:`score`, plus ``adaptation_curve`` -- the per-episode return
        averaged over teammates and seeds. That curve is the point of cross-episode
        adaptation: a within-episode baseline would be flat, TAO should climb as its OCW
        fills. ``ocw_size`` is TAO's ``C`` (``context_trajectories``, the reference's
        OCW_SIZE), so deployment matches the context size stage 2 trained on.
        """
        from oaht_bench.offline.incontext_eval import evaluate_incontext

        seed_pt, seed_curves, seed_means = [], [], []
        for s in range(ns):
            p = all_params if ns == 1 else jax.tree.map(lambda x: x[s], all_params)  # noqa: B023
            mean, curve = evaluate_incontext(
                agent,
                p,
                env,
                teammates,
                max_episode_steps=job.env.rollout_length,
                num_episodes=job.offline.eval_episodes,
                ocw_size=job.offline.context_trajectories,
                obs_dim=windows.obs_dim,
                rng=jax.random.PRNGKey(rng_base + s),
            )
            seed_pt.append(mean)
            seed_curves.append(curve)
            seed_means.append(float(np.mean(list(mean.values()))))
        labels = list(seed_pt[0])
        per_teammate = {t: float(np.mean([m[t] for m in seed_pt])) for t in labels}
        n_eps = int(job.offline.eval_episodes)
        adaptation_curve = [
            float(np.mean([seed_curves[s][t][e] for s in range(ns) for t in labels]))
            for e in range(n_eps)
        ]
        out = {
            "num_seeds": ns,
            "mean_return": float(np.mean(seed_means)),
            "mean_return_std": float(np.std(seed_means)) if ns > 1 else 0.0,
            "worst_teammate_return": float(min(per_teammate.values())),
            "per_teammate": per_teammate,
            "adaptation_curve": adaptation_curve,
        }
        if ns > 1:
            out["per_seed_mean_return"] = [float(m) for m in seed_means]
        return out

    def score(teammates, *, rng_base):
        """Across-seed mean return against one teammate set.

        ``rng_base`` offsets the per-seed eval RNG so the held-out draw stays
        byte-identical to the pre-split runs while train gets independent episodes.
        """
        if incontext:
            return _score_incontext(teammates, rng_base=rng_base)
        per_seed = [
            evaluate_agent_against(
                agent,
                all_params if ns == 1 else jax.tree.map(lambda x: x[s], all_params),  # noqa: B023
                env,
                teammates,
                rng=jax.random.PRNGKey(rng_base + s),
                target_return=cond_target,
                max_episode_steps=job.env.rollout_length,
                num_episodes=job.offline.eval_episodes,
            )
            for s in range(ns)
        ]
        labels = list(per_seed[0].per_teammate)
        per_teammate = {t: float(np.mean([sc.per_teammate[t] for sc in per_seed])) for t in labels}
        seed_means = [sc.mean_return for sc in per_seed]
        out = {
            "num_seeds": ns,
            "mean_return": float(np.mean(seed_means)),
            "mean_return_std": float(np.std(seed_means)) if ns > 1 else 0.0,
            "worst_teammate_return": float(min(per_teammate.values())),
            "per_teammate": per_teammate,
        }
        if ns > 1:
            out["per_seed_mean_return"] = [float(m) for m in seed_means]
        return out

    # Held-out (test) is the primary metric; its rng_base is unchanged so existing
    # runs reproduce byte-for-byte.
    result = score(heldout_teammates, rng_base=job.seed + 1)
    result["target_return"] = float(cond_target)

    for label, v in result["per_teammate"].items():
        logger.log_item(f"Eval/Return_teammate_{label}", v)
    logger.log_item("Eval/MeanReturn", result["mean_return"])
    if ns > 1:
        logger.log_item("Eval/MeanReturnStd", result["mean_return_std"])
    logger.log_item("Eval/WorstTeammateReturn", result["worst_teammate_return"])
    # Adaptation curve (in-context baselines only): held-out return per episode index
    # as the OCW fills -- flat for a within-episode method, climbing if TAO adapts.
    for e, v in enumerate(result.get("adaptation_curve", [])):
        logger.log_item(f"Eval/AdaptationReturn_ep{e}", v)

    # In-distribution contrast: how well the ego coordinates with the *training*
    # teammates it learned from, so a held-out score is readable as generalisation
    # rather than raw competence. Absent for legacy split-less datasets.
    if train_teammates:
        train = score(train_teammates, rng_base=job.seed + 1001)
        result["train"] = train
        result["generalization_gap"] = float(train["mean_return"] - result["mean_return"])
        for label, v in train["per_teammate"].items():
            logger.log_item(f"Eval/Train_Return_teammate_{label}", v)
        logger.log_item("Eval/TrainMeanReturn", train["mean_return"])
        if ns > 1:
            logger.log_item("Eval/TrainMeanReturnStd", train["mean_return_std"])
        logger.log_item("Eval/GeneralizationGap", result["generalization_gap"])

    logger.commit()
    if "train" in result:
        log.info(
            "Evaluation (%d seed%s): held-out mean %.4f%s, train mean %.4f, gap %.4f, "
            "worst-teammate %.4f",
            ns,
            "" if ns == 1 else "s",
            result["mean_return"],
            "" if ns == 1 else f" ± {result['mean_return_std']:.4f}",
            result["train"]["mean_return"],
            result["generalization_gap"],
            result["worst_teammate_return"],
        )
    else:
        log.info(
            "Evaluation (%d seed%s): mean %.4f%s, worst-teammate %.4f",
            ns,
            "" if ns == 1 else "s",
            result["mean_return"],
            "" if ns == 1 else f" ± {result['mean_return_std']:.4f}",
            result["worst_teammate_return"],
        )
    return result
