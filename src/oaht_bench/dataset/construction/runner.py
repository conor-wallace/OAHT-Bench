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
from oaht_bench.dataset.construction.collect import collect_episode
from oaht_bench.dataset.construction.epsilon_sampler import EPSILON_TARGETS, load_pooled, plan_for_variant
from oaht_bench.dataset.vault import write_vault
from oaht_bench.envs import make_env
from oaht_bench.envs.log_wrapper import LogWrapper
from oaht_bench.population import artifact_dir, population_from_run, released_members
from oaht_bench.population.pooled_crossplay import build_roster

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


def run(job: DatasetCollectionJob) -> Path:
    """Collect a dataset and return the run directory.

    Dispatches on the shape of ``population_path``: a single string keeps the
    within-one-generator path (:func:`_collect_single`), a *list* selects pooled
    mode (:func:`_collect_pooled`), where the released members of every listed
    generator are flattened into one roster and the ε sampler seats them across
    populations against the pooled cross-play matrix. Both return ragged episodes
    written straight to a flat Flashbax Vault -- no padding is ever materialised
    (:mod:`oaht_bench.dataset.vault`).
    """
    run_dir = Path(job.run_dir())
    artifact = run_dir / "dataset.vlt"
    if artifact.exists():
        raise FileExistsError(
            f"{artifact} already exists and would be overwritten. Delete "
            f"{run_dir} to re-collect, or change the job's label. (The directory "
            f"name includes the config hash, so an identical config always "
            f"resolves here.)"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_job(job, run_dir / "job.json", minimal=False)

    env = LogWrapper(make_env(job.env.env_name, job.env.env_kwargs()))
    if isinstance(job.population_path, (list, tuple)):
        episodes, member_ids, meta = _collect_pooled(job, env)
    else:
        episodes, member_ids, meta = _collect_single(job, env)

    write_vault(episodes, member_ids, artifact, ego_index=0, meta=meta)
    summary = _summary(episodes, member_ids, ego_index=0)
    (run_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info(
        "Dataset: %d episodes, %d agents, mean length %.1f, mean ego return %.4f",
        summary["episodes"], summary["agents"], summary["mean_length"], summary["mean_ego_return"],
    )
    return run_dir


def _summary(episodes: list[dict], member_ids: np.ndarray, *, ego_index: int) -> dict:
    """Describe a collection from the ragged episodes, without padding them."""
    lengths = [int(np.asarray(e["dones"]).shape[0]) for e in episodes]
    ego_returns = [float(np.asarray(e["rewards"])[ego_index].sum()) for e in episodes]
    return {
        "episodes": len(episodes),
        "agents": int(np.asarray(member_ids).shape[1]),
        "mean_length": float(np.mean(lengths)),
        "mean_ego_return": float(np.mean(ego_returns)),
    }


def _collect_single(job: DatasetCollectionJob, env) -> tuple[list[dict], np.ndarray, dict]:
    """The original path: seat one generator's designed pairing per episode.

    ``population_path`` is a single run directory. Only 'expert' is implemented
    here; the ε variants need the pooled roster and go through pooled mode.
    """
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
            f"variant={job.variant!r} is not implemented in single-population "
            f"mode; only 'expert' is. The ε variants ('br_vs_worst', 'mixed') "
            f"need pooled mode -- pass a list of population_paths and a "
            f"pooled_matrix_path. The τ variants ('medium', 'replay_full') need "
            f"the competence ladder (§4.3), not yet saved."
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

    meta = {
        "config_hash": job.content_hash(),
        "env": job.env.name,
        "variant": job.variant,
        "generator": gen_job.generator.generator,
        "paired_roles": loaded.paired,
        "mismatch_fraction": job.mismatch_fraction,
        "population_run": str(job.population_path),
        "population_config_hash": gen_job.content_hash(),
        "eligible_members": [int(m) for m in eligible],
    }
    return episodes, np.stack(member_ids), meta


def _pooled_matrix_hash(path: Path) -> str:
    """Content hash of the pooled matrix, so a dataset records which one it read."""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _collect_pooled(job: DatasetCollectionJob, env) -> tuple[list[dict], np.ndarray, dict]:
    """Pooled mode: seat the ε sampler's cross-population plan (§3, dataset_design).

    The released members of every ``population_path`` are flattened into one
    roster (:func:`~oaht_bench.population.pooled_crossplay.build_roster`, the same
    flattening the matrix was built with), and the ε sampler turns the variant's
    target quality distribution into concrete ``(ego, teammate)`` roster indices
    read off ``pooled_matrix_path``. Each episode seats ``roster[ego]`` in seat 0
    against ``roster[teammate]`` in the rest.

    ``member_ids`` here are *roster* indices, not per-population member indices;
    the roster manifest in ``meta`` maps each back to ``(generator, member,
    role)``, and ``ego_response_quality`` carries the per-episode ε -- which
    :func:`~oaht_bench.dataset.vault.write_vault` broadcasts into a flat vault field
    as well as keeping in ``meta`` (the stable descriptor the trajectory-view
    baselines read, ``dataset_design.md`` §2).
    """
    if job.variant not in EPSILON_TARGETS:
        raise NotImplementedError(
            f"variant={job.variant!r} has no ε target; pooled mode implements "
            f"{sorted(EPSILON_TARGETS)}. τ variants need the competence ladder (§4)."
        )
    if job.pooled_matrix_path is None:
        raise ValueError(
            "pooled mode (population_path is a list) needs pooled_matrix_path, "
            "the populations/<env>/pooled_crossplay.npz for these populations."
        )
    if job.mismatch_fraction:
        # Pairing correctness is orthogonal to ε and not yet layered onto the
        # pooled seating; fail rather than silently ignore a requested split.
        raise NotImplementedError(
            "mismatch_fraction is not yet supported in pooled mode; the ε bands "
            "define the seating. Leave it at 0."
        )

    pop_dirs = [Path(p) for p in job.population_path]
    roster = build_roster(pop_dirs, env)
    pooled = load_pooled(job.pooled_matrix_path)
    # Guard the emitted indices against a matrix computed for a different or
    # reordered roster -- otherwise a stale matrix silently seats the wrong pair.
    pooled.check_roster(roster)

    plan = plan_for_variant(
        pooled,
        job.variant,
        job.num_episodes,
        rng=np.random.default_rng(job.seed),
        allow_self_pairing=job.allow_self_pairing,
    )
    num_seats = len(env.agents)

    rng = jax.random.PRNGKey(job.seed)
    episodes, member_ids, epsilons, targets = [], [], [], []
    for ep, seating in enumerate(tqdm(plan, desc="Generating pooled dataset")):
        rng, ep_rng = jax.random.split(rng)
        ego, mate = roster[seating.ego], roster[seating.teammate]
        # Ego in seat 0, the teammate in every other seat (two-player today).
        seats = [(ego.params, ego.policy_cls)] + [
            (mate.params, mate.policy_cls) for _ in range(num_seats - 1)
        ]
        episodes.append(
            collect_episode(
                ep_rng,
                env,
                seats,
                max_episode_steps=job.env.rollout_length,
                greedy=False,  # sampled: matches training and deployment (see crossplay)
            )
        )
        member_ids.append(np.asarray([seating.ego] + [seating.teammate] * (num_seats - 1)))
        epsilons.append(seating.epsilon)
        targets.append(seating.target)
        if (ep + 1) % 10 == 0:
            log.info("collected %d/%d episodes", ep + 1, job.num_episodes)

    meta = {
        "config_hash": job.content_hash(),
        "env": job.env.name,
        "variant": job.variant,
        "mode": "pooled",
        "populations": [str(p) for p in pop_dirs],
        "pooled_matrix_path": str(job.pooled_matrix_path),
        "pooled_matrix_hash": _pooled_matrix_hash(job.pooled_matrix_path),
        "allow_self_pairing": job.allow_self_pairing,
        # member_ids are indices into this roster manifest.
        "roster": [
            {"generator": e.generator, "member": int(e.member), "role": e.role}
            for e in roster
        ],
        # Per-episode labels, aligned with the episode axis. write_vault also
        # broadcasts ego_response_quality into a flat vault field.
        "ego_response_quality": [float(x) for x in epsilons],
        "target_epsilon": [float(x) for x in targets],
    }
    return episodes, np.stack(member_ids), meta
