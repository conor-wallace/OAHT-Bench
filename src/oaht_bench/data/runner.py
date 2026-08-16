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


def _draw_cycling(pool: list, count: int, rng) -> list:
    """Take ``count`` entries from ``pool``, using every entry equally often.

    Repeatedly shuffles the whole pool rather than sampling with replacement, so
    with 5 members and 10 draws each member appears exactly twice. Sampling gives
    a multinomial spread instead, and uneven per-teammate coverage is what forces
    the stage-1 sampler to compensate when building contrastive batches.
    """
    out: list = []
    while len(out) < count:
        out.extend(pool[i] for i in rng.permutation(len(pool)))
    return out[:count]


def _seat_plan(eligible: list[int], num_episodes: int, mismatch_fraction: float, rng):
    """Which two members occupy the seats in each episode.

    The split is by *count*, not a coin flip per episode: with
    ``mismatch_fraction=0.5`` and 10 episodes exactly 5 are matched and 5 are
    mismatched. A per-episode Bernoulli only gives the fraction in expectation --
    at 12 episodes it produced 25% where 50% was asked -- and a dataset variant
    should be a stated property, not a draw.

    Matched episodes draw only from ``(i, i)`` and mismatched only from
    ``(i, j)`` with ``i != j``. Neither pool can produce the other, so the two
    counts mean exactly what they say.
    """
    n_mismatched = int(round(num_episodes * mismatch_fraction))
    n_matched = num_episodes - n_mismatched

    if n_mismatched and len(eligible) < 2:
        raise ValueError(
            f"mismatch_fraction={mismatch_fraction} needs at least two distinct "
            f"members, but the population releases {len(eligible)}."
        )

    matched = _draw_cycling([(m, m) for m in eligible], n_matched, rng)
    # Stratify the mismatched draws by the ego seat rather than sampling from
    # the n*(n-1) pool: drawing k pairs from that pool leaves ego coverage
    # lumpy for small k, and every teammate should appear equally often as the
    # one being modelled.
    primaries = _draw_cycling(list(eligible), n_mismatched, rng)
    mismatched = [
        (a, int(rng.choice([m for m in eligible if m != a]))) for a in primaries
    ]

    plan = matched + mismatched
    # Interleave, or any consumer that slices the dataset by index gets a
    # biased subset.
    return [plan[i] for i in rng.permutation(len(plan))]


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
    # The seating plan is decided up front so the matched/mismatched split is
    # exact rather than sampled, and so every teammate gets equal coverage.
    plan = _seat_plan(
        [int(m) for m in eligible], job.num_episodes, job.mismatch_fraction,
        np.random.default_rng(job.seed),
    )
    episodes, member_ids = [], []
    for ep in tqdm(range(job.num_episodes), desc="Geneating dataset"):
        rng, ep_rng = jax.random.split(rng)
        # Matched by default: member i opposite member i, which
        # LoadedPopulation.seat resolves to conf_i vs br_i for the paired
        # generators and self_i vs self_i for the homogeneous ones. That is the
        # designed pairing in every case, so the dataset stops carrying which
        # generator produced it -- one member index means "this teammate at its
        # intended competence" regardless of method.
        #
        # Independent draws per seat made this 1-in-population_size by accident:
        # at n=5, 80% of an "expert" dataset was mismatched play, which is what
        # the generators are tuned to make minimally cooperative.
        primary, partner = plan[ep]
        seats = np.asarray([primary] + [partner] * (num_seats - 1))
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
            "mismatch_fraction": job.mismatch_fraction,
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
