"""Execute a :class:`~oaht_bench.configs.job.TeammateGenerationJob`.

Dispatches to the absorbed generator implementations. They take a plain nested
dict — jax-aht converted away from Hydra's ``DictConfig`` at its entry point and
we kept that boundary — so the job config projects straight onto them without
Hydra, YAML, or a subprocess.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from oaht_bench.common.logging import RunLogger
from oaht_bench.configs import save_job
from oaht_bench.configs.job import TeammateGenerationJob

log = logging.getLogger(__name__)


def _generators() -> dict[str, Callable[..., Any]]:
    """Import the generators lazily so the CLI can validate without loading JAX."""
    from oaht_bench.teammate_gen.BRDiv import run_brdiv
    from oaht_bench.teammate_gen.CoMeDi import run_comedi
    from oaht_bench.teammate_gen.fcp import run_fcp
    from oaht_bench.teammate_gen.LBRDiv import run_lbrdiv

    return {"fcp": run_fcp, "comedi": run_comedi, "brdiv": run_brdiv, "lbrdiv": run_lbrdiv}


def run(job: TeammateGenerationJob) -> Path:
    """Train a teammate population and return the run directory.

    The config's content hash names the directory and the config is written into
    it, so a population can always be traced to the settings that produced it —
    the provenance §7.1 requires for released checkpoints.
    """
    run_dir = Path(job.run_dir())

    # Orbax refuses to overwrite an existing checkpoint directory, and it only
    # finds out at the *save*, which is after training. On a multi-hour job that
    # discards the entire run. Fail in the first second instead.
    existing = run_dir / "saved_train_run"
    if existing.exists():
        raise FileExistsError(
            f"{existing} already exists, and the checkpoint writer will not "
            f"overwrite it -- the run would fail after training rather than now. "
            f"Delete {run_dir} to re-run, or change the job's label. (The "
            f"directory name includes the config hash, so an identical config "
            f"always resolves here.)"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = job.to_jax_aht_cfg()
    alg = job.generator.generator

    runners = _generators()
    if alg not in runners:
        raise ValueError(f"No runner for generator {alg!r}. Known: {sorted(runners)}")

    # Fully resolved, not the delta form the authored config uses: an artifact
    # must stay self-describing even if a default later moves.
    save_job(job, run_dir / "job.json", minimal=False)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(cfg, indent=2, sort_keys=True, default=str) + "\n"
    )

    with RunLogger(
        run_dir,
        use_wandb=job.logging.use_wandb,
        wandb_project=job.logging.wandb_project,
        wandb_entity=job.logging.wandb_entity,
        config=cfg,
        verbose=job.logging.verbose,
    ) as logger:
        # All four generators read the typed job directly.
        params, population = runners[alg](job, logger)

        # One cross-play evaluation for every generator, computed the same way,
        # so FCP -- which has no notion of cross-play during training -- is
        # measurable alongside the others.
        _evaluate_population(job, params, population, logger)

    return run_dir


def _best_response_params(job: TeammateGenerationJob):
    """The saved best-response set, for the generators that train one.

    Read back from the checkpoint the generator has already written rather than
    threading a second return value through all four training functions, which
    would change a contract three of them do not need.
    """
    from oaht_bench.common.save_load_utils import load_train_run

    out = load_train_run(str(Path(job.run_dir()) / "saved_train_run"))
    return out.get("final_params_br")


def _evaluate_population(
    job: TeammateGenerationJob, params: Any, population: Any, logger: RunLogger
) -> None:
    """Score the trained population against itself and record the result."""
    import jax

    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.teammate_gen.crossplay import evaluate_population, write_scores

    env = LogWrapper(make_env(job.env.env_name, job.env.env_kwargs()))

    # BRDiv and L-BRDiv train confederate/best-response *pairs* and save both
    # sets; their designed-optimal pairing is conf_i with br_i, so the matrix
    # must be conf x br. Pairing a confederate with a copy of itself would be an
    # out-of-distribution pairing that under-reports competence -- confederates
    # are never trained to play with themselves. FCP and CoMeDi release a single
    # set of self-play policies, so their column population is the row one.
    partner_params = _best_response_params(job)

    scores = evaluate_population(
        env,
        params,
        population,
        rng=jax.random.PRNGKey(job.seed),
        max_episode_steps=job.env.rollout_length,
        num_episodes=job.evaluation_episodes,
        partner_params=partner_params,
    )
    write_scores(scores, Path(job.run_dir()))
    logger.log_item("Population/SelfPlay", scores.self_play)
    logger.log_item("Population/CrossPlay", scores.cross_play)
    logger.log_item("Population/Separation", scores.separation)
    logger.commit()
    log.info("Population cross-play:\n%s", scores.describe())
