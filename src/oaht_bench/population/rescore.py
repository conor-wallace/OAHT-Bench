"""Recompute a finished run's cross-play scores from its saved checkpoints.

Scoring a population is separable from training it: ``saved_train_run`` holds
the parameters, so a change to how the population is *measured* does not require
re-running the training that produced it. That mattered the first time this was
needed — the evaluation had been running argmax actions, which deadlocks a
symmetric coordination task, and every finished run had to be re-scored without
spending the GPU hours again.

The measurement lives entirely in :func:`~.crossplay.evaluate_population`; this
module only rebuilds the two arguments it needs from disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from oaht_bench.configs import load_job
from oaht_bench.configs.job import TeammateGenerationJob

log = logging.getLogger(__name__)


def artifact_dir(run_dir: Path) -> Path:
    """Locate a run's Orbax checkpoint directory.

    Two layouts exist. The runner writes ``<run_dir>/artifacts/saved_train_run``,
    but the absorbed ``save_train_run`` resolves relative paths against
    ``REPO_PATH``, which after absorption points at ``src/oaht_bench`` rather
    than the repo root — so the same run also appears under
    ``src/oaht_bench/results/...``. The two are byte-identical; this prefers the
    runner's layout and falls back, rather than silently scoring nothing.
    """
    run_dir = Path(run_dir)
    candidates = [
        run_dir / "artifacts" / "saved_train_run",
        run_dir / "saved_train_run",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        f"no saved_train_run under {run_dir}; looked in "
        f"{[str(c) for c in candidates]}. A run that crashed before its "
        f"checkpoint write cannot be re-scored — it has to be re-run."
    )


def population_from_run(job: TeammateGenerationJob, out: Any, env: Any):
    """Rebuild ``(params, population)`` the way the generator itself does.

    Each generator ships a ``get_*_population(job, out, env)`` that turns a
    training output into the released population — FCP flattens its checkpoint
    grid, the others take ``final_params_conf``. Dispatching to those keeps one
    definition of what a generator's population *is*, instead of a second copy
    here that could drift.
    """
    from oaht_bench.teammate_gen.brdiv import get_brdiv_population
    from oaht_bench.teammate_gen.comedi import get_comedi_population
    from oaht_bench.population.loading import get_fcp_population
    from oaht_bench.teammate_gen.lbrdiv import get_lbrdiv_population

    builders = {
        "fcp": get_fcp_population,
        "comedi": get_comedi_population,
        "brdiv": get_brdiv_population,
        "lbrdiv": get_lbrdiv_population,
    }
    name = job.generator.generator
    if name not in builders:
        raise ValueError(f"no population builder for {name!r}. Known: {sorted(builders)}")
    return builders[name](job, out, env)


def rescore_run(
    run_dir: Path,
    *,
    episodes: int | None = None,
    greedy: bool | None = None,
    write: bool = True,
):
    """Re-evaluate one finished run and optionally overwrite its scores CSV.

    Args:
        episodes: Override ``job.evaluation_episodes``. Useful for a cheap pass
            over a sweep before re-scoring the winner at full fidelity.
        greedy: Override ``job.evaluation_greedy``. Left as ``None`` the run's
            own config decides, which is what makes a re-score reproducible.
        write: When false, compute and return without touching the run.

    Returns:
        The :class:`~.crossplay.CrossPlayScores` for the run.
    """
    import jax

    from oaht_bench.common.save_load_utils import load_train_run
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.population.crossplay import evaluate_population, write_scores
    from oaht_bench.population.members import released_members

    run_dir = Path(run_dir)
    job = load_job(run_dir / "job.json")
    if job.job_type != "teammate_generation":
        raise ValueError(f"{run_dir} is a {job.job_type} run, not a population.")

    # Absolute: load_train_run joins relative paths against the wrong root.
    out = load_train_run(str(artifact_dir(run_dir)))
    env = LogWrapper(make_env(job.env.env_name, job.env.env_kwargs()))
    params, population = population_from_run(job, out, env)

    # BRDiv/L-BRDiv save a best-response set alongside the confederates; FCP and
    # CoMeDi don't, so this is None for them, matching runner._best_response_params.
    partner_params = out.get("final_params_br")

    scores = evaluate_population(
        env,
        params,
        population,
        rng=jax.random.PRNGKey(job.seed),
        max_episode_steps=job.env.rollout_length,
        num_episodes=job.evaluation_episodes if episodes is None else episodes,
        partner_params=partner_params,
        member_indices=released_members(job, population.pop_size),
        greedy=job.evaluation_greedy if greedy is None else greedy,
    )
    if write:
        write_scores(scores, run_dir)
    return scores
