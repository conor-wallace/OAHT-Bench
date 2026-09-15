"""Inference-parity diagnostic: is the deploy path faithful to training, and is
early-episode prediction the weak spot?

The scaled BC clone hits 0.999 average teacher-forced accuracy but diverges from
the expert at deploy step ~6. Two explanations:

  (a) intrinsic -- early-episode actions are genuinely hard (hidden hand, few
      hints), so per-timestep accuracy is low early and the 0.999 *average* hides
      it. Sliding windows would not help: the causal mask already trains every
      timestep, and early actions are simply harder.
  (b) deploy bug -- the rolling-window ``get_action`` path produces different
      logits than the training forward on the same prefix.

Part 1 measures per-timestep teacher-forced accuracy (training forward, ``act``).
Low early / high late => (a). Uniformly high => the failure is a deploy issue.

Part 2 compares, on one expert episode, the training forward's argmax at each
timestep against the deploy ``get_action`` argmax fed the same teacher-forced
prefix (with the conditioning target set to the episode's realized return so the
return-to-go matches training). A mismatch localizes a deploy/windowing bug (b).

    uv run python scripts/diagnose_inference_parity.py \
        --config configs/hanabi/training/pooled_expert_scaled/bc.json \
        --params /tmp/bc_clone.pkl --pairing comedi:0:self --episodes 50
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--params", required=True, help="Policy pickle from --save-params.")
    ap.add_argument("--pairing", default="comedi:0:self")
    ap.add_argument(
        "--populations",
        nargs="+",
        default=[
            "populations/hanabi/brdiv",
            "populations/hanabi/comedi",
            "populations/hanabi/fcp",
            "populations/hanabi/lbrdiv",
        ],
    )
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from oaht_bench.configs import load_job
    from oaht_bench.dataset.construction.collect import collect_episode
    from oaht_bench.dataset.dataset import Dataset
    from oaht_bench.dataset.vault import write_vault
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.models.bc_agent import BcAgent
    from oaht_bench.models.liam_agent import LiamAgent
    from oaht_bench.models.meliba_agent import MelibaAgent
    from oaht_bench.models.omis_agent import OmisAgent
    from oaht_bench.models.tao_agent import TaoAgent
    from oaht_bench.offline.runner import _resolve_dims
    from oaht_bench.population.pooled_crossplay import build_roster

    agents = {
        "bc": BcAgent,
        "liam": LiamAgent,
        "meliba": MelibaAgent,
        "omis": OmisAgent,
        "tao": TaoAgent,
    }

    saved = pickle.load(open(args.params, "rb"))
    job = load_job(args.config)
    base_env = make_env(job.env.env_name, job.env.env_kwargs())
    env = LogWrapper(base_env)
    roster = build_roster([Path(p) for p in args.populations], env)
    gen, member, role = args.pairing.split(":")
    ego = next(
        e for e in roster if e.generator == gen and int(e.member) == int(member) and e.role == role
    )
    mate = ego  # self-play pairing

    cfg = job.offline.model_copy(update={"context_length": saved["context_length"]})
    resolved = _resolve_dims(cfg, saved["obs_dim"], saved["action_dim"])
    norm = saved["normalization"]
    params = {"stage1": saved["stage1"], "stage2": saved["stage2"]}

    def build_agent(cond):
        a = agents[saved["baseline"]](
            resolved, context_length=saved["context_length"], target_return=cond, normalization=norm
        )
        a.build_model()
        return a

    # Collect expert episodes (greedy = the argmax behaviour we compare against).
    rng = jax.random.PRNGKey(args.seed)
    episodes = []
    for _ in range(args.episodes):
        rng, ep_rng = jax.random.split(rng)
        episodes.append(
            collect_episode(
                ep_rng,
                base_env,
                [(ego.params, ego.policy_cls), (mate.params, mate.policy_cls)],
                max_episode_steps=job.env.rollout_length,
                greedy=True,
            )
        )
    tmp = Path(tempfile.mkdtemp(prefix="parity_")) / "p.vlt"
    write_vault(
        episodes,
        np.array([[int(mate.member), int(mate.member)]] * len(episodes)),
        tmp,
        ego_index=0,
        meta={"variant": "single"},
    )
    ds = Dataset(str(tmp), context_length=cfg.context_length, stride=cfg.stride, normalize=True)
    w = ds.windows
    agent = build_agent(saved["cond_target"])

    # ---- Part 1: per-timestep teacher-forced accuracy (training forward) ----
    logits = np.asarray(
        agent.act(
            params,
            jnp.asarray(w.ego_rtg),
            jnp.asarray(w.ego_obs),
            jnp.asarray(w.ego_actions),
            timesteps=jnp.asarray(w.timesteps),
            mask=jnp.asarray(w.mask),
        )
    )
    masked = np.where(w.ego_avail > 0, logits, -1e9)
    pred = masked.argmax(-1)  # (N, T)
    correct = (pred == w.ego_actions) & w.mask
    ts = w.timesteps
    print(
        "\n===== PART 1: per-timestep teacher-forced accuracy (training forward) =====", flush=True
    )
    print(f"  overall (masked positions): {correct.sum() / w.mask.sum():.4f}", flush=True)
    for lo, hi in [(1, 5), (6, 10), (11, 20), (21, 40), (41, 200)]:
        sel = w.mask & (ts >= lo) & (ts <= hi)
        n = int(sel.sum())
        acc = float((correct & sel).sum() / n) if n else float("nan")
        print(f"  timestep {lo:>2}-{hi:<3}: acc {acc:.4f}  (n={n})", flush=True)

    # ---- Part 2: training forward vs deploy get_action on one episode ----
    ep = episodes[0]
    ego_obs_raw = np.asarray(ep.obs[0])  # (L, obs_dim), unnormalised
    ego_avail_raw = np.asarray(ep.avail_actions[0])  # (L, A)
    ego_act_true = np.asarray(ep.actions[0])  # (L,)
    ego_rew = np.asarray(ep.rewards[0])  # (L,)
    L = ego_obs_raw.shape[0]
    total_return = float(ep.returns()[0])
    # Match training RTG: condition on the episode's realized return.
    cond_oracle = total_return if norm is None else norm.apply_rtg(total_return)
    dep_agent = build_agent(cond_oracle)

    # Path A on this one episode (find its window; episodes < context => one window).
    idx = int(np.flatnonzero(w.episode_id == 0)[0])
    a_logits = np.asarray(
        agent.act(
            params,
            jnp.asarray(w.ego_rtg[idx : idx + 1]),
            jnp.asarray(w.ego_obs[idx : idx + 1]),
            jnp.asarray(w.ego_actions[idx : idx + 1]),
            timesteps=jnp.asarray(w.timesteps[idx : idx + 1]),
            mask=jnp.asarray(w.mask[idx : idx + 1]),
        )
    )[0]
    a_masked = np.where(w.ego_avail[idx] > 0, a_logits, -1e9)
    a_pred = a_masked.argmax(-1)  # per window position
    # map window positions -> timestep; valid (real) positions in timestep order
    valid_pos = np.flatnonzero(w.mask[idx])
    a_pred_by_t = a_pred[valid_pos]  # aligned to timesteps 1..L

    # Path B: deploy get_action, teacher-forced with the true actions + rewards.
    h = dep_agent.init_hstate(1, aux_info={"agent_id": 0})
    b_pred = []
    rng = jax.random.PRNGKey(args.seed + 7)
    for t in range(L):
        rng, ak = jax.random.split(rng)
        r_prev = 0.0 if t == 0 else float(ego_rew[t - 1])
        act, h = dep_agent.get_action(
            params=params,
            obs=jnp.asarray(ego_obs_raw[t]).reshape(1, 1, -1),
            done=jnp.zeros((1, 1), dtype=bool),
            avail_actions=jnp.asarray(ego_avail_raw[t], dtype=jnp.float32),
            hstate=h,
            rng=ak,
            aux_obs=None,
            env_state=None,
            test_mode=True,  # argmax, to compare with Path A's argmax
            reward=jnp.asarray(r_prev).reshape(1, 1, 1),
        )
        b_pred.append(int(np.asarray(act).reshape(-1)[0]))
        # teacher-force: overwrite the just-written action with the true one
        h = h.replace(ctx_act=h.ctx_act.at[-1].set(int(ego_act_true[t])))
    b_pred = np.asarray(b_pred)

    n = min(len(a_pred_by_t), len(b_pred))
    agree = int((a_pred_by_t[:n] == b_pred[:n]).sum())
    first_mismatch = next((t for t in range(n) if a_pred_by_t[t] != b_pred[t]), None)
    print(
        "\n===== PART 2: training-forward vs deploy get_action (1 episode, matched RTG) =====",
        flush=True,
    )
    print(f"  steps compared:              {n}", flush=True)
    print(f"  argmax agreement A vs B:     {agree}/{n} ({100 * agree / n:.1f}%)", flush=True)
    print(
        f"  first mismatch step:         {first_mismatch if first_mismatch is not None else 'none'}",
        flush=True,
    )
    print(
        "  -> A==B everywhere: deploy path is faithful; the gap is intrinsic/early-difficulty.\n"
        "     A!=B: a deploy/windowing bug, first mismatch localizes it.",
        flush=True,
    )

    shutil.rmtree(tmp.parent, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
