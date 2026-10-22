#!/usr/bin/env python3
"""Does TAO's trained encoder actually separate teammates in embedding space?

The direct test of "is representation quality the bottleneck" for an
already-trained checkpoint -- no retraining needed. The flat
``ancillary_mate_action_acc_curve`` seen on real held-out teammates
(scripts/report_paper_stats.py) says the opponent-recognition signal isn't
improving as more of a teammate's trajectory accumulates; this checks whether
the *embedding itself* even has the information to recognise a teammate from
in the first place, independent of the OCW/cross-attention/policy machinery
downstream of it. Equation-level unit tests (tests/unit/offline/tao/) already
confirmed that downstream wiring is correct, which is what makes this the
next falsifiable thing to check: if embeddings don't separate teammates here,
the bottleneck is representation quality, and nothing downstream (GetOffD,
the policy, deployment wiring) can fix a generalisation gap that starts here.

Encodes several trajectories per teammate through the trained
``OpponentPolicyEncoder`` + AP pooling -- the exact ``z_bar`` the ancillary
decoder and Stage 1's generative/discriminative losses all operate on -- for
both TRAIN teammates (pulled directly from the training dataset's own
windows) and HELD-OUT teammates. Held-out teammates never appear in the
training dataset at all (that is the entire point of the split), so their
trajectories are collected the only way they can be: a handful of real
rollouts against each, through the same env/agent machinery
``evaluate_incontext`` uses, padded with :class:`OpponentContextWindow`
purely as a convenient per-episode container (not for cross-episode
accumulation -- each episode is encoded independently here).

Reports two numbers per group:

* ``nn_same_teammate_acc`` -- for each trajectory, does its nearest OTHER one
  (by L2 distance in embedding space) belong to the same teammate? Chance is
  ``1/n_teammates``; well above chance means the embedding clusters by
  teammate identity, near chance means it doesn't.
* ``within/between_dist_ratio`` -- mean pairwise distance between trajectories
  of the *same* teammate, divided by mean pairwise distance between
  trajectories of *different* teammates. Near 1.0 means no separation at all;
  well below 1.0 means teammates form tight, distinguishable clusters.

Single-generator datasets only (teammate_id is the raw member id, with no
generator disambiguation -- fine for e.g. a pure-MEP roster; a pooled
multi-generator dataset could in principle alias member ids across
generators, not handled here).

    uv run python scripts/diagnose_tao_embedding_separability.py \\
        results/training/tao_pooled_lbf_9x9_tipr-908864fc6b03
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from oaht_bench.configs import load_job
from oaht_bench.dataset.dataset import Dataset
from oaht_bench.models.tao_agent import OpponentPolicyEncoder, TaoAgent
from oaht_bench.offline.evaluate import dataset_target_return
from oaht_bench.offline.runner import _resolve_dims, _teammate_policies


def _encode_from_windows(encoder, params, windows, row_idx: np.ndarray) -> np.ndarray:
    """``z_bar`` (AP-pooled) per window index, shape ``(n, hidden)`` -- the
    train-teammate path, pulled directly from the training dataset."""
    tok = encoder.apply(
        params,
        jnp.asarray(windows.mate_next_obs[row_idx]),
        jnp.asarray(windows.mate_actions[row_idx]),
        jnp.asarray(windows.mate_rewards[row_idx]),
        mask=jnp.asarray(windows.mask[row_idx]),
        timesteps=jnp.asarray(windows.timesteps[row_idx]),
        train=False,
    )
    return np.asarray(OpponentPolicyEncoder.pool(tok))


def _encode_held_out_teammates(
    agent,
    enc_params,
    params,
    env,
    teammates,
    *,
    episodes_per_teammate,
    max_episode_steps,
    obs_dim,
    rng,
):
    """``z_bar`` per rolled-out episode for each held-out teammate, collected via
    real rollouts (the only source of held-out trajectories that exists) and
    padded through :class:`OpponentContextWindow` purely as a per-episode
    container -- each episode is encoded independently, not accumulated."""
    from oaht_bench.offline.incontext_eval import OpponentContextWindow, rollout_one_episode

    zs, labels = [], []
    for raw_label, mate_params, mate_policy in teammates:
        teammate_id = int(raw_label.split(":")[1])
        ocw = OpponentContextWindow(episodes_per_teammate, max_episode_steps, obs_dim)
        for _ in range(episodes_per_teammate):
            rng, ep_rng = jax.random.split(rng)
            _ego_return, mate_stream, _mate_avail = rollout_one_episode(
                ep_rng,
                env,
                agent,
                params,
                mate_params,
                mate_policy,
                max_episode_steps=max_episode_steps,
            )
            ocw.append(*mate_stream)
        no, ac, rw, ts, mk = ocw.arrays()
        tok = agent.encoder.apply(
            enc_params,
            jnp.asarray(no),
            jnp.asarray(ac),
            jnp.asarray(rw),
            mask=jnp.asarray(mk),
            timesteps=jnp.asarray(ts),
            train=False,
        )
        z = np.asarray(OpponentPolicyEncoder.pool(tok, mask=jnp.asarray(mk)))
        zs.append(z)
        labels.extend([teammate_id] * len(z))
    return np.concatenate(zs), np.asarray(labels), rng


def _pairwise_dist(z: np.ndarray) -> np.ndarray:
    return np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)


def _nn_same_teammate_accuracy(z: np.ndarray, teammate_ids: np.ndarray) -> float:
    d = _pairwise_dist(z)
    np.fill_diagonal(d, np.inf)
    nn = np.argmin(d, axis=1)
    return float(np.mean(teammate_ids[nn] == teammate_ids))


def _within_between_ratio(z: np.ndarray, teammate_ids: np.ndarray) -> float:
    d = _pairwise_dist(z)
    same = teammate_ids[:, None] == teammate_ids[None, :]
    eye = np.eye(len(z), dtype=bool)
    within, between = d[same & ~eye], d[~same]
    if within.size == 0 or between.size == 0:
        return float("nan")
    return float(within.mean() / between.mean())


def _sample_train_rows(index, teammates, windows_per_teammate, rng):
    rows, labels = [], []
    for t in teammates:
        pool = index.by_teammate[t]
        n = min(windows_per_teammate, len(pool))
        rows.append(rng.choice(pool, size=n, replace=False))
        labels.extend([t] * n)
    return np.concatenate(rows), np.asarray(labels)


def _report(label, z, teammate_ids, n_teammates):
    nn_acc = _nn_same_teammate_accuracy(z, teammate_ids)
    ratio = _within_between_ratio(z, teammate_ids)
    chance = 1.0 / n_teammates
    print(
        f"{label:9s} n_teammates={n_teammates:3d} n_trajectories={len(z):4d}  "
        f"nn_same_teammate_acc={nn_acc:.4f} (chance={chance:.4f})  "
        f"within/between_dist_ratio={ratio:.4f} (1.0 = no separation)"
    )


def diagnose(
    run_dir: Path,
    *,
    windows_per_teammate: int,
    max_teammates: int,
    seed_index: int,
    held_out_episodes: int,
) -> None:
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper

    run_dir = Path(run_dir)
    job = load_job(run_dir / "job.json")
    if job.baseline != "tao":
        raise ValueError(f"{run_dir} is a {job.baseline!r} run, not tao -- nothing to diagnose.")
    cfg = job.offline
    with open(run_dir / "params.pkl", "rb") as fh:
        raw = pickle.load(fh)
    # raw also carries "normalization" (a plain Normalization dataclass, not a jax
    # pytree of arrays) alongside stage1/stage2 -- strip it before this dict is ever
    # used as a params argument to a jitted agent call, or jax chokes trying to
    # treat a Normalization object as a leaf inside the params tree.
    saved = {"stage1": raw["stage1"], "stage2": raw["stage2"]}
    if int(job.num_seeds) > 1:
        saved = jax.tree.map(lambda x: x[seed_index], saved)

    ds = Dataset(
        job.dataset_path,
        context_length=cfg.context_length,
        stride=cfg.stride,
        normalize=cfg.normalize_observations,
    )
    resolved = _resolve_dims(cfg, ds.obs_dim, ds.action_dim)
    net = resolved.network
    encoder = OpponentPolicyEncoder(
        action_dim=net.action_dim,
        hidden_dim=net.hidden_dim,
        ff_dim=net.ff_dim,
        num_blocks=net.num_blocks,
        dropout=net.dropout,
    )
    enc_params = saved["stage1"]["encoder"]

    held_out = {int(m) for ms in ds.batch.meta.get("held_out", {}).values() for m in ms}
    all_teammates = ds.index.teammates
    train_teammates = [t for t in all_teammates if t not in held_out][:max_teammates]

    print(f"=== {run_dir.name} ===")
    print(
        f"{len(all_teammates)} teammates in the training dataset, {len(held_out)} held out at collection time\n"
    )

    rng = np.random.default_rng(0)
    if len(train_teammates) >= 2:
        row_idx, train_ids = _sample_train_rows(
            ds.index, train_teammates, windows_per_teammate, rng
        )
        z_train = _encode_from_windows(encoder, enc_params, ds.windows, row_idx)
        _report("train", z_train, train_ids, len(train_teammates))
    else:
        print(f"{'train':9s} fewer than 2 teammates available -- skipping")

    target = dataset_target_return(ds.batch)
    target = target if ds.windows.norm is None else ds.windows.norm.apply_rtg(target)
    agent = TaoAgent(
        resolved,
        context_length=cfg.context_length,
        target_return=target,
        normalization=ds.windows.norm,
    )
    agent.build_model()
    env = LogWrapper(make_env(job.env.env_name, job.env.env_kwargs()))
    held_out_teammates = _teammate_policies(ds.batch, env, which="held_out")[:max_teammates]
    if len(held_out_teammates) >= 2:
        key = jax.random.PRNGKey(0)
        z_test, test_ids, _key = _encode_held_out_teammates(
            agent,
            enc_params,
            saved,
            env,
            held_out_teammates,
            episodes_per_teammate=held_out_episodes,
            max_episode_steps=job.env.rollout_length,
            obs_dim=ds.obs_dim,
            rng=key,
        )
        _report("held-out", z_test, test_ids, len(held_out_teammates))
    else:
        print(f"{'held-out':9s} fewer than 2 teammates available -- skipping")

    print(
        "\nnn_same_teammate_acc well above chance AND within/between_dist_ratio well "
        "below 1.0 on held-out => the embedding separates teammates fine; the gap is "
        "downstream (GetOffD, the policy, or deployment wiring).\n"
        "Both near chance/1.0 on held-out (even if train looks fine) => the encoder "
        "doesn't generalise what it learned to unseen teammates -- a representation "
        "problem, not a downstream one.\n"
        "Both near chance/1.0 on train too => Stage 1 didn't learn a "
        "teammate-discriminating representation at all."
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run", type=Path, help="A tao training run dir (job.json + params.pkl).")
    ap.add_argument(
        "--windows-per-teammate", type=int, default=20, help="train side, from the dataset"
    )
    ap.add_argument(
        "--held-out-episodes",
        type=int,
        default=20,
        help="held-out side, real rollouts per teammate",
    )
    ap.add_argument("--max-teammates", type=int, default=12)
    ap.add_argument(
        "--seed-index", type=int, default=0, help="which trained seed, if num_seeds > 1"
    )
    args = ap.parse_args()
    diagnose(
        args.run,
        windows_per_teammate=args.windows_per_teammate,
        max_teammates=args.max_teammates,
        seed_index=args.seed_index,
        held_out_episodes=args.held_out_episodes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
