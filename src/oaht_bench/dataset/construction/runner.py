"""Execute a :class:`~oaht_bench.configs.job.DatasetCollectionJob` (§4).

Seats a population member in every position and records full trajectories. The
population is rebuilt with the generator's own builder rather than by reading
the checkpoint directly, so "what a member is" has one definition shared with
scoring (see :func:`oaht_bench.population.rescore.population_from_run`).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import jax
import numpy as np
from tqdm import tqdm

from oaht_bench.common.save_load_utils import load_train_run
from oaht_bench.configs import load_job, save_job
from oaht_bench.configs.job import DatasetCollectionJob
from oaht_bench.dataset.construction.collect import collect_episodes_batched
from oaht_bench.dataset.construction.epsilon_sampler import (
    EPSILON_TARGETS,
    load_pooled,
    plan_for_variant,
    plan_weighted_seatings,
)
from oaht_bench.dataset.construction.split import derive_split
from oaht_bench.dataset.vault import VaultWriter
from oaht_bench.envs import make_env
from oaht_bench.envs.log_wrapper import LogWrapper
from oaht_bench.population import artifact_dir, population_from_run, released_members
from oaht_bench.population.loading import load_br_egos
from oaht_bench.population.pooled_crossplay import teammate_roster

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
    mismatched = [(a, int(rng.choice([m for m in eligible if m != a]))) for a in primaries]

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
        meta, stream = _collect_pooled(job, env)
    else:
        meta, stream = _collect_single(job, env)

    # Stream episodes straight to the vault in chunks so collection never holds
    # them all in RAM -- a 150k-episode Hanabi dataset is tens of GiB. The summary
    # and the per-episode target labels are accumulated the same way.
    writer = VaultWriter(artifact, ego_index=0, meta=meta)
    chunk_eps, chunk_mids, chunk_erq, targets = [], [], [], []
    n = length_sum = 0
    ret_sum = 0.0
    agents = None

    def flush():
        if not chunk_eps:
            return
        writer.write(chunk_eps, np.stack(chunk_mids), ego_response_quality=(chunk_erq or None))
        chunk_eps.clear()
        chunk_mids.clear()
        chunk_erq.clear()

    for episode, seats, erq, target in stream:
        seats = np.asarray(seats)
        agents = int(seats.shape[0]) if agents is None else agents
        chunk_eps.append(episode)
        chunk_mids.append(seats)
        if erq is not None:
            chunk_erq.append(float(erq))
        if target is not None:
            targets.append(float(target))
        n += 1
        length_sum += int(episode.length)
        ret_sum += float(episode.returns()[0])
        if len(chunk_eps) >= _WRITE_CHUNK:
            flush()
    flush()
    # Finalise: writes the norm_stats.json sidecar (observation/rtg statistics
    # accumulated over the whole collection as it streamed) so training loads the
    # normalisation instead of recomputing it from disk.
    writer.close()

    summary = {
        "episodes": n,
        "agents": agents,
        "mean_length": length_sum / max(n, 1),
        "mean_ego_return": ret_sum / max(n, 1),
    }
    (run_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if targets:
        # Per-episode provenance labels ride beside the vault, not in its fixed
        # metadata (which incremental writes set once, at the first chunk).
        (run_dir / "collection_labels.json").write_text(
            json.dumps({"target_epsilon": targets}, indent=2) + "\n"
        )
    log.info(
        "Dataset: %d episodes, %d agents, mean length %.1f, mean ego return %.4f",
        summary["episodes"],
        summary["agents"],
        summary["mean_length"],
        summary["mean_ego_return"],
    )
    return run_dir


#: Episodes flat-packed and appended to the vault per write. Bounds collection RAM.
_WRITE_CHUNK = 2000

#: Episodes collected per batched call within a pairing group. Bounds the ragged
#: episodes held in RAM before they stream on to the vault writer.
_COLLECT_SUBGROUP = 2000


def _collect_single(job: DatasetCollectionJob, env) -> tuple[dict, Iterator]:
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

    # Hold out `holdout_per_generator` members as test teammates (§8) and seat only
    # the train members, exactly as pooled mode does -- here the single generator is
    # the whole roster. Deterministic in (members, split_seed, holdout_per_generator).
    gen_name = gen_job.generator.generator
    identities = [(gen_name, int(m), "member") for m in eligible]
    split = derive_split(
        identities,
        env=job.env.name,
        split_seed=job.split_seed,
        holdout_per_generator=job.holdout_per_generator,
    )
    manifest_path = Path(job.run_dir()) / "teammate_split.json"
    manifest_path.write_text(json.dumps(split.to_dict(), indent=2) + "\n")
    train_members = [int(eligible[i]) for i in split.train_indices]
    if len(train_members) < 2 and job.mismatch_fraction:
        raise ValueError(
            f"after holding out {job.holdout_per_generator} of {len(eligible)} "
            f"members, only {len(train_members)} train members remain, too few for "
            f"mismatch_fraction={job.mismatch_fraction}. Lower holdout_per_generator."
        )

    # The seating plan is decided up front so the matched/mismatched split is
    # exact rather than sampled, and so every teammate gets equal coverage.
    plan = _seat_plan(
        train_members,
        job.num_episodes,
        job.mismatch_fraction,
        np.random.default_rng(job.seed),
    )

    def stream():
        # Matched-by-default seating resolves each (primary, partner) to its designed
        # pairing (conf_i vs br_i for paired generators, self_i vs self_i otherwise).
        # Group the plan by that pairing so each collects in one batched device call;
        # yield order is by pairing rather than the plan's interleave (fine -- training
        # samples at random and the split is by member).
        by_pairing: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, (primary, partner) in enumerate(plan):
            by_pairing[(int(primary), int(partner))].append(idx)

        base = jax.random.PRNGKey(job.seed)
        with tqdm(total=len(plan), desc="Generating dataset", unit="ep") as bar:
            for gi, ((primary, partner), idxs) in enumerate(by_pairing.items()):
                seats_members = [primary] + [partner] * (num_seats - 1)
                seats = loaded.seat(seats_members)
                member_row = np.asarray(seats_members)
                for start in range(0, len(idxs), _COLLECT_SUBGROUP):
                    sub = idxs[start : start + _COLLECT_SUBGROUP]
                    sub_rng = jax.random.fold_in(jax.random.fold_in(base, gi), start)
                    eps = collect_episodes_batched(
                        sub_rng,
                        env,
                        seats,
                        max_episode_steps=job.env.rollout_length,
                        num_episodes=len(sub),
                        greedy=False,  # sampled: matches training and deployment
                    )
                    for e in eps:
                        yield e, member_row, None, None  # single mode: no ε label
                    bar.update(len(sub))

    meta = {
        "config_hash": job.content_hash(),
        "env": job.env.name,
        "variant": job.variant,
        "generator": gen_job.generator.generator,
        "paired_roles": loaded.paired,
        "mismatch_fraction": job.mismatch_fraction,
        "population_run": str(job.population_path),
        "population_config_hash": gen_job.content_hash(),
        # Train members only: the held-out (test) ones are excluded from collection.
        "eligible_members": train_members,
        "split_seed": job.split_seed,
        "holdout_per_generator": job.holdout_per_generator,
        "split_manifest_hash": split.manifest_hash,
        "held_out": {g: sorted(ms) for g, ms in split.held_out.items()},
        "test_teammates": [
            {"generator": gen_name, "member": int(eligible[i]), "role": "member"}
            for i in split.test_indices
        ],
    }
    return meta, stream()


def _pooled_matrix_hash(path: Path) -> str:
    """Content hash of the pooled matrix, so a dataset records which one it read."""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _collect_pooled(job: DatasetCollectionJob, env) -> tuple[dict, Iterator]:
    """Pooled mode: seat the ε sampler's cross-population plan (§3, dataset_design).

    The designed teammates of every ``population_path`` are flattened into one
    roster (:func:`~oaht_bench.population.pooled_crossplay.teammate_roster`, the
    same flattening the matrix was built with), and the ε sampler turns the
    variant's target quality distribution into concrete ``(ego, teammate)``
    roster indices read off ``pooled_matrix_path``. Ego and teammate share this
    same roster: an "ego" index is always that identity's own trained
    ``ppo_br`` best response (:mod:`oaht_bench.population.loading`'s
    ``load_br_egos``), never a reused population policy. Each episode seats
    that best response in seat 0 against ``roster[teammate]`` in the rest.

    ``member_ids`` here are *roster* indices, not per-population member indices;
    the roster manifest in ``meta`` maps each back to ``(generator, member,
    role)``, and ``ego_response_quality`` carries the per-episode ε -- which
    :func:`~oaht_bench.dataset.vault.write_vault` broadcasts into a flat vault field
    as well as keeping in ``meta`` (the stable descriptor the trajectory-view
    baselines read, ``dataset_design.md`` §2).
    """
    if job.variant not in EPSILON_TARGETS and job.variant != "weighted":
        raise NotImplementedError(
            f"variant={job.variant!r} has no ε target; pooled mode implements "
            f"{sorted({*EPSILON_TARGETS, 'weighted'})}. τ variants need the "
            f"competence ladder (§4)."
        )
    if job.pooled_matrix_path is None:
        raise ValueError(
            "pooled mode (population_path is a list) needs pooled_matrix_path, "
            "the populations/<env>/pooled_crossplay.npz for these populations."
        )
    if job.br_population_path is None:
        raise ValueError(
            "pooled mode needs br_population_path -- the ppo_br run the "
            "pooled_matrix_path matrix's ego axis was computed with. Every ego "
            "seated is that teammate's own dedicated best response, never a "
            "reused population policy."
        )
    if job.mismatch_fraction:
        # Pairing correctness is orthogonal to ε and not yet layered onto the
        # pooled seating; fail rather than silently ignore a requested split.
        raise NotImplementedError(
            "mismatch_fraction is not yet supported in pooled mode; the ε bands "
            "define the seating. Leave it at 0."
        )

    pop_dirs = [Path(p) for p in job.population_path]
    roster = teammate_roster(pop_dirs, env)
    br_egos = load_br_egos(job.br_population_path, env)
    pooled = load_pooled(job.pooled_matrix_path)
    # Guard the emitted indices against a matrix computed for a different or
    # reordered roster -- otherwise a stale matrix silently seats the wrong pair.
    pooled.check_roster(roster)

    # Ad-hoc-teamwork train/test split (§8): hold out `holdout_per_generator`
    # members of each generator as *test* teammates and restrict this collection
    # to the train partition, so test teammates never enter the training data in
    # either seat. Deterministic in (roster, split_seed, holdout_per_generator),
    # so every variant collected with the same values shares one held-out set.
    identities = [(e.generator, int(e.member), e.role) for e in roster]
    split = derive_split(
        identities,
        env=job.env.name,
        split_seed=job.split_seed,
        holdout_per_generator=job.holdout_per_generator,
    )
    manifest_path = Path(job.run_dir()) / "teammate_split.json"
    manifest_path.write_text(json.dumps(split.to_dict(), indent=2) + "\n")

    if job.variant == "weighted":
        plan = plan_weighted_seatings(
            pooled,
            job.num_episodes,
            temperature=job.temperature,
            rng=np.random.default_rng(job.seed),
            allow_self_pairing=job.allow_self_pairing,
            allowed=split.train_indices,
        )
    else:
        plan = plan_for_variant(
            pooled,
            job.variant,
            job.num_episodes,
            rng=np.random.default_rng(job.seed),
            allow_self_pairing=job.allow_self_pairing,
            allowed=split.train_indices,
        )
    num_seats = len(env.agents)

    def stream():
        # Group the plan by (ego, teammate) so each pairing's episodes collect in one
        # batched device call (vmap over episodes, scan over steps) instead of the
        # eager per-step Python loop -- 1-2 orders of magnitude faster on an
        # accelerator. Yield order is by pairing rather than the plan's interleave,
        # which is fine: training samples windows at random and the split is by member,
        # not index. Per-pairing ε/target ride out with each episode as before.
        by_pairing: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, seating in enumerate(plan):
            by_pairing[(seating.ego, seating.teammate)].append(idx)

        base = jax.random.PRNGKey(job.seed)
        with tqdm(total=len(plan), desc="Generating pooled dataset", unit="ep") as bar:
            for gi, ((ego_i, mate_i), idxs) in enumerate(by_pairing.items()):
                mate = roster[mate_i]
                # The ego is always ego_i's own trained best response -- ego and
                # teammate share one identity space, so this is a plain lookup,
                # never a reused roster policy.
                ego_identity = roster[ego_i]
                ego_params, ego_cls = br_egos[
                    (ego_identity.generator, int(ego_identity.member), ego_identity.role)
                ]
                ego_row_id = ego_i
                seats = [(ego_params, ego_cls)] + [
                    (mate.params, mate.policy_cls) for _ in range(num_seats - 1)
                ]
                for start in range(0, len(idxs), _COLLECT_SUBGROUP):
                    sub = idxs[start : start + _COLLECT_SUBGROUP]
                    sub_rng = jax.random.fold_in(jax.random.fold_in(base, gi), start)
                    eps = collect_episodes_batched(
                        sub_rng,
                        env,
                        seats,
                        max_episode_steps=job.env.rollout_length,
                        num_episodes=len(sub),
                        greedy=False,  # sampled: matches training and deployment
                    )
                    for k, idx in enumerate(sub):
                        seating = plan[idx]
                        member_row = np.asarray([ego_row_id] + [seating.teammate] * (num_seats - 1))
                        yield eps[k], member_row, seating.epsilon, seating.target
                    bar.update(len(sub))

    meta = {
        "config_hash": job.content_hash(),
        "env": job.env.name,
        "variant": job.variant,
        "mode": "pooled",
        "populations": [str(p) for p in pop_dirs],
        "pooled_matrix_path": str(job.pooled_matrix_path),
        "pooled_matrix_hash": _pooled_matrix_hash(job.pooled_matrix_path),
        "allow_self_pairing": job.allow_self_pairing,
        # Seat 0 is always the trained best-response for whichever teammate identity
        # the ε sampler drew as "ego", never a reused population policy.
        "br_population_path": job.br_population_path,
        "temperature": job.temperature if job.variant == "weighted" else None,
        # Train/test teammate split (§8). test_teammates are the held-out policies
        # online evaluation rolls the trained ego against; they never appear below.
        "split_seed": job.split_seed,
        "holdout_per_generator": job.holdout_per_generator,
        "split_manifest_hash": split.manifest_hash,
        "held_out": {g: sorted(ms) for g, ms in split.held_out.items()},
        "test_teammates": [
            {
                "generator": roster[i].generator,
                "member": int(roster[i].member),
                "role": roster[i].role,
            }
            for i in split.test_indices
        ],
        # member_ids are indices into this roster manifest.
        "roster": [
            {"generator": e.generator, "member": int(e.member), "role": e.role} for e in roster
        ],
    }
    return meta, stream()
