"""The retrieval oracle, deployed: what a *realistic*, non-privileged identifier wins.

``scripts/performance_ladder.py`` computes the retrieval oracle's *ceiling* --
``max over train BRs of matrix[br, teammate]`` -- which assumes perfect teammate
identification. This script answers the next question: how much of that ceiling
is reachable by an identifier that only sees what a deployed ego actually sees
(its own observation/action stream), trained the way an offline method would be.

Pipeline (a "plug-in" estimator, not a live rollout of the swapped-in policy --
see the note on ``achieved_return`` below):

1. Roll out the **best fixed train BR** (a real, always-available policy -- no
   teammate identity needed to select it) against every teammate, capturing the
   ego's own ``(obs, action)`` stream. This is the same information LIAM/OMIS's
   encoder reads and the same distribution as "fixed ego" in the identification
   probes (tuning_record.md): a policy that hasn't identified anyone yet.
2. Fit a linear softmax classifier, from an episode-disjoint split, to predict
   *which TRAIN teammate* produced a given window of that stream. Only TRAIN
   labels are classes -- a deployed system has no held-out identity to name.
3. For each held-out evaluation episode, predict a label and look up the best
   TRAIN BR *for that predicted label* (precomputed from the BR matrix). The
   achieved return is read off the matrix for ``(that BR, the episode's TRUE
   teammate)`` -- a Monte Carlo plug-in using the already-validated 200-episode
   matrix, rather than re-simulating the swapped-in policy episode-by-episode.

Usage:
    uv run python scripts/retrieval_oracle_eval.py \\
        --populations populations/lbf_20x20/brdiv populations/lbf_20x20/comedi \\
                      populations/lbf_20x20/fcp populations/lbf_20x20/lbrdiv \\
        --br-run ppo_br_lbf_20x20-7db59378f7a9 \\
        --split pooled_lbf_20x20_expert-99e3ca80c5ab/teammate_split.json \\
        --matrix-in populations/lbf_20x20/br_ladder.npz \\
        --episodes 150 --context 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from oaht_bench.configs import load_job
from oaht_bench.envs import make_env
from oaht_bench.population.loading import load_br_egos
from oaht_bench.population.pooled_crossplay import build_roster

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/ for the sibling import
from performance_ladder import _label, build_br_matrix  # noqa: E402


def rollout_trajectories(env, ego_params, ego_cls, mate_params, mate_cls, *, steps, episodes, seed):
    """Roll ``episodes`` parallel episodes, capturing the ego's own (obs, action) stream.

    Not TAO/OMIS-faithful (it does not read the teammate's stream) -- this is exactly
    LIAM/OMIS's information set: the ego only ever sees its own local history. Returns
    ``(obs, actions)`` shaped ``(episodes, steps, obs_dim)`` / ``(episodes, steps)``.
    """
    action_dim = env.action_space(env.agents[0]).n

    def one(key):
        k, rk = jax.random.split(key)
        obs, state = env.reset(rk)
        h0 = ego_cls.init_hstate(1, aux_info={"agent_id": 0})
        h1 = mate_cls.init_hstate(1, aux_info={"agent_id": 1})
        ao0 = jnp.zeros((1, 1, action_dim))
        ao1 = jnp.zeros((1, 1, action_dim))
        rw = jnp.zeros((1, 1, 1))

        def step(carry, _):
            state, obs, h0, h1, ao0, ao1, rw, k = carry
            k, k0, k1, ks = jax.random.split(k, 4)
            av = jax.lax.stop_gradient(env.get_avail_actions(state))
            joint = jnp.concatenate((ao0, ao1), axis=-1)
            a0, h0n = ego_cls.get_action(
                params=ego_params,
                obs=obs["agent_0"].reshape(1, 1, -1),
                done=jnp.zeros((1, 1), bool),
                avail_actions=av["agent_0"].astype(jnp.float32),
                hstate=h0,
                rng=k0,
                aux_obs=(ao0, joint, rw),
                env_state=state,
                test_mode=False,
            )
            a1, h1n = mate_cls.get_action(
                params=mate_params,
                obs=obs["agent_1"].reshape(1, 1, -1),
                done=jnp.zeros((1, 1), bool),
                avail_actions=av["agent_1"].astype(jnp.float32),
                hstate=h1,
                rng=k1,
                aux_obs=None,
                env_state=state,
                test_mode=False,
            )
            a0, a1 = a0.squeeze(), a1.squeeze()
            ego_obs = obs["agent_0"].reshape(-1)
            act = {kk: [a0, a1][i] for i, kk in enumerate(env.agents)}
            obs2, state2, r, _, _ = env.step(ks, state, act)
            n0 = jax.nn.one_hot(a0, action_dim).reshape(1, 1, -1)
            n1 = jax.nn.one_hot(a1, action_dim).reshape(1, 1, -1)
            new_carry = (state2, obs2, h0n, h1n, n0, n1, r["agent_0"].reshape(1, 1, 1), k)
            return new_carry, (ego_obs, a0)

        _, (obs_seq, act_seq) = jax.lax.scan(
            step, (state, obs, h0, h1, ao0, ao1, rw, k), None, length=steps
        )
        return obs_seq, act_seq

    return jax.vmap(one)(jax.random.split(jax.random.PRNGKey(seed), episodes))


def fit_softmax(X, y, n_classes, *, steps=300, lr=0.5, l2=1e-4):
    """Full-batch softmax regression -- same footprint as diagnose_population_suitability's
    probes, kept local so this script has no private cross-script coupling."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    W = np.zeros((Xn.shape[1], n_classes))
    b = np.zeros(n_classes)
    Y = np.eye(n_classes)[y]
    n = len(Xn)
    for _ in range(steps):
        z = Xn @ W + b
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        g = (p - Y) / n
        W -= lr * (Xn.T @ g + l2 * W)
        b -= lr * g.sum(0)
    return {"W": W, "b": b, "mu": mu, "sd": sd}


def predict(clf, X):
    Xn = (X - clf["mu"]) / clf["sd"]
    return (Xn @ clf["W"] + clf["b"]).argmax(1)


def windows_and_features(obs, act, action_dim, *, context, seed):
    """One randomly-placed ``context``-step window per episode -> a permutation-
    invariant feature (mean obs, action-onehot histogram), matching the
    identification probes this script's estimate is built to be consistent with."""
    E, T, D = obs.shape
    rng = np.random.default_rng(seed)
    s = rng.integers(0, max(1, T - context), size=E)
    ow = np.stack([obs[i, s[i] : s[i] + context] for i in range(E)])
    aw = np.stack([act[i, s[i] : s[i] + context] for i in range(E)])
    onehot = np.eye(action_dim)[aw]
    return np.concatenate([ow.mean(1), onehot.mean(1)], -1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--populations", nargs="+", required=True, type=Path)
    ap.add_argument("--br-run", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument(
        "--matrix-in", type=Path, default=None, help="a matrix saved by performance_ladder.py --out"
    )
    ap.add_argument("--episodes", type=int, default=150, help="rollout episodes per teammate")
    ap.add_argument(
        "--context", type=int, default=20, help="window length for the identification feature"
    )
    ap.add_argument("--eval-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    job = load_job(args.populations[0] / "job.json")
    env = make_env(job.env.env_name, job.env.env_kwargs())
    roster = build_roster(args.populations, env)
    br = load_br_egos(args.br_run, env)
    mates = [e for e in roster if (e.generator, e.member, e.role) in br]
    labels = [_label(e) for e in mates]

    if args.matrix_in:
        d = np.load(args.matrix_in, allow_pickle=True)
        matrix, mat_labels = d["matrix"], [str(x) for x in d["labels"]]
        assert mat_labels == labels, "matrix labels don't match this roster's BR ordering"
    else:
        matrix, _ = build_br_matrix(args.populations, args.br_run, episodes=200, seed=args.seed)

    split = json.loads(args.split.read_text())
    held = {f"{g}:{m}:conf" for g, ms in split["held_out"].items() for m in ms} | {
        f"{g}:{m}:self" for g, ms in split["held_out"].items() for m in ms
    }
    train_idx = [i for i, lab in enumerate(labels) if lab not in held]
    held_idx = [i for i, lab in enumerate(labels) if lab in held]
    print(f"train teammates={len(train_idx)}  held-out teammates={len(held_idx)}")

    # the best fixed TRAIN BR -- selectable without knowing any teammate's identity.
    generalist_row = max(train_idx, key=lambda k: matrix[k, train_idx].mean())
    gm = mates[generalist_row]
    generalist_params, generalist_cls = br[(gm.generator, gm.member, gm.role)]
    print(f"calibration ego (best fixed train BR): {labels[generalist_row]}")

    action_dim = env.action_space(env.agents[0]).n
    X_all, y_all, teammate_of = [], [], []
    for j, mate in enumerate(mates):
        obs, act = rollout_trajectories(
            env,
            generalist_params,
            generalist_cls,
            mate.params,
            mate.policy_cls,
            steps=job.env.rollout_length,
            episodes=args.episodes,
            seed=args.seed + j,
        )
        feat = windows_and_features(
            np.asarray(obs), np.asarray(act), action_dim, context=args.context, seed=args.seed
        )
        X_all.append(feat)
        y_all.append(np.full(len(feat), j))
        teammate_of.append(np.full(len(feat), j))
        print(f"  rolled out generalist vs {labels[j]}  ({j + 1}/{len(mates)})", flush=True)
    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)

    # episode-disjoint fit/eval split, done independently per teammate.
    rng = np.random.default_rng(args.seed)
    fit_mask = np.zeros(len(X_all), bool)
    for j in range(len(mates)):
        idx = np.flatnonzero(y_all == j)
        rng.shuffle(idx)
        cut = int((1 - args.eval_frac) * len(idx))
        fit_mask[idx[:cut]] = True

    # classifier: TRAIN teammates only, relabelled to a dense 0..n_train-1 range.
    train_label_map = {orig: dense for dense, orig in enumerate(train_idx)}
    fit_on_train = fit_mask & np.isin(y_all, train_idx)
    clf = fit_softmax(
        X_all[fit_on_train],
        np.array([train_label_map[y] for y in y_all[fit_on_train]]),
        len(train_idx),
    )

    # best TRAIN BR for each (dense) predicted TRAIN label, from the matrix.
    best_br_for_label = np.array(
        [train_idx[int(np.argmax(matrix[train_idx, orig]))] for orig in train_idx]
    )

    def achieved(cols, split_name):
        eval_mask = (~fit_mask) & np.isin(y_all, cols)
        if not eval_mask.any():
            print(f"  {split_name}: no eval episodes")
            return
        pred_dense = predict(clf, X_all[eval_mask])
        chosen_br = best_br_for_label[pred_dense]
        true_teammate = y_all[eval_mask]
        achieved_return = matrix[chosen_br, true_teammate].mean()
        acc = None
        if split_name == "train":
            true_dense = np.array([train_label_map[y] for y in true_teammate])
            acc = float((pred_dense == true_dense).mean())
        print(
            f"  {split_name:10s} n_eval={eval_mask.sum():4d}  achieved={achieved_return:.4f}"
            + (f"  identification_acc={100 * acc:.1f}%" if acc is not None else "")
        )
        return achieved_return

    print("\n=== realistic (non-privileged) retrieval-based return ===")
    achieved(train_idx, "train")
    if held_idx:
        achieved(held_idx, "held_out")

    print("\n=== for comparison (from the BR matrix directly) ===")
    print(
        f"  best fixed generalist (train)     {matrix[np.ix_(train_idx, train_idx)].mean(1).max():.4f}"
    )
    print(
        f"  retrieval oracle, perfect ID (train)   {np.mean([matrix[train_idx, j].max() for j in train_idx]):.4f}"
    )
    if held_idx:
        print(
            f"  best fixed generalist (held-out)  {matrix[np.ix_(train_idx, held_idx)].mean(1).max():.4f}"
        )
        print(
            f"  retrieval oracle, perfect ID (held-out) "
            f"{np.mean([matrix[train_idx, j].max() for j in held_idx]):.4f}"
        )


if __name__ == "__main__":
    main()
