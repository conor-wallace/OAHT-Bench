"""Execute a :class:`~oaht_bench.configs.job.DatasetCollectionJob` (§4).

Seats a population member in every position and records full trajectories. The
population is rebuilt with the generator's own builder rather than by reading
the checkpoint directly, so "what a member is" has one definition shared with
scoring (see :func:`oaht_bench.population.rescore.population_from_run`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import jax
import numpy as np
from tqdm import tqdm

from oaht_bench.common.save_load_utils import load_train_run
from oaht_bench.configs import load_job, save_job
from oaht_bench.configs.job import DatasetCollectionJob
from oaht_bench.data.collect import collect_episode, pad_and_stack
from oaht_bench.data.schema import EpisodeBatch
from oaht_bench.envs import make_env
from oaht_bench.envs.log_wrapper import LogWrapper
from oaht_bench.population import artifact_dir, get_member_params, population_from_run, released_members

log = logging.getLogger(__name__)


def _load_population(job: DatasetCollectionJob, env):
    """Rebuild ``(params, population)`` from a teammate-generation run.

    ``load_train_run`` returns a dict of four keys, not a pair — turning it into
    a population requires the generator-specific builder, because FCP flattens a
    checkpoint grid while the others take ``final_params_conf``.
    """
    pop_run = Path(job.population_path)
    # Accept either the run directory or the checkpoint directory inside it.
    run_dir = pop_run.parent.parent if pop_run.name == "saved_train_run" else pop_run
    gen_job = load_job(run_dir / "job.json")

    out = load_train_run(str(artifact_dir(run_dir)))
    return population_from_run(gen_job, out, env), gen_job


# comment: I imagine that at the dataset generation phase that we will already have ALL of the populations
# comment: I think it makes sense to add option to the DatasetCollectionJob config that allows population_path to be a list of paths to [fcp, comedi, brdiv, and lbrdiv] populations
# comment: To take this further, it also makes sense to combine members of one population with members of another to REALLY mix performance
# comment: Anyway, this central run function should really be the entrypoint for dataset generation and should probably call different private runner functions depending on the dataset variant in the job config 
def run(job: DatasetCollectionJob) -> Path:
    """Collect a dataset and return the run directory."""
    run_dir = Path(job.run_dir())
    existing = run_dir / "dataset.npz"
    if existing.exists():
        raise FileExistsError(
            f"{existing} already exists and would be overwritten. Delete "
            f"{run_dir} to re-collect, or change the job's label. (The directory "
            f"name includes the config hash, so an identical config always "
            f"resolves here.)"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_job(job, run_dir / "job.json", minimal=False)

    env = LogWrapper(make_env(job.env.env_name, job.env.env_kwargs()))
    loaded, gen_job = _load_population(job, env)
    num_seats = len(env.agents)

    # Which members are eligible to be seated. FCP's population spans competence
    # by design, so the 'expert' variant must not draw from its early
    # checkpoints -- the same distinction scoring makes.
    eligible = released_members(gen_job, loaded.pop_size)
    if job.variant != "expert":
        # Other D4RL-style regimes (§4.3) draw from the wider ladder; not yet
        # implemented, so fail rather than silently collect 'expert' data.
        raise NotImplementedError(
            f"variant={job.variant!r} is not implemented yet; only 'expert' is. "
            f"The other regimes need the competence ladder (§4.3), which for FCP "
            f"is the checkpoint axis and for the others needs training snapshots "
            f"that are not currently saved."
        )

    rng = jax.random.PRNGKey(job.seed)
    episodes, member_ids = [], []
    for ep in tqdm(range(job.num_episodes), desc="Geneating dataset"):
        rng, seat_rng, ep_rng = jax.random.split(rng, 3)
        # comment: Should we be randomly sampling like this or should iterate over all combinations of teams?
        # comment: We tune teammage generation algorithms such that matched seats are expert level cooperative and mismatched seats are minimally cooperative
        seats = np.asarray(
            jax.random.choice(seat_rng, np.asarray(eligible), shape=(num_seats,))
        )
        episodes.append(
            collect_episode(
                ep_rng,
                env,
                loaded.seat([int(m) for m in seats]),
                max_episode_steps=job.env.rollout_length,
                greedy=False,  # sampled: matches training and deployment (see crossplay)
            )
        )
        member_ids.append(seats)
        if (ep + 1) % 10 == 0:
            log.info("collected %d/%d episodes", ep + 1, job.num_episodes)

    stacked = pad_and_stack(episodes)
    batch = EpisodeBatch(
        **stacked,
        member_ids=np.stack(member_ids),
        ego_index=0,
        meta={
            "config_hash": job.content_hash(),
            "env": job.env.name,
            "variant": job.variant,
            "generator": gen_job.generator.generator,
            "paired_roles": loaded.paired,
            "population_run": str(job.population_path),
            "population_config_hash": gen_job.content_hash(),
            "eligible_members": [int(m) for m in eligible],
        },
    )
    batch.save(run_dir / "dataset.npz")
    (run_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "episodes": batch.num_episodes,
                "agents": batch.num_agents,
                "mean_length": float(batch.episode_lengths().mean()),
                "mean_ego_return": float(batch.episode_returns()[:, 0].mean()),
            },
            indent=2,
        )
        + "\n"
    )
    log.info("Dataset:\n%s", batch.describe())
    return run_dir
