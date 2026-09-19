"""Teammate-identity ORACLE for offline BC: an upper bound on pooled AHT.

BC on the pooled dataset must clone one best-response per teammate (TAO/OMIS build
the offline set exactly this way -- a different BR per opponent) with *no* teammate
signal, so it averages ~12 distinct experts into a policy coherent for none and
desyncs in closed loop (see docs/tuning_record.md, "Offline BC x Hanabi").

This trains the *same* BC backbone but adds the ground-truth teammate id -- a learned
per-teammate embedding added to every token. It is an **oracle**, not a method: it
consumes the true teammate identity, so it can only be evaluated against *train*
teammates (whose id we know). What it measures is the ceiling: if perfect teammate
identity lets one model reproduce the data's competence, the pooled failure is
teammate *inference* (the modelling baselines' job) and the data/egos are fine; if
the oracle also stays low, the egos themselves are not cloneable (a best-response
quality / collection problem) and no amount of teammate modelling will help.

Usage:
    uv run python scripts/diagnose_oracle_bc.py configs/hanabi/training/pooled_expert_scaled/bc.json \
        [--steps N] [--episodes E]
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="an offline BC training job JSON (for its hyperparameters + dataset)")
    ap.add_argument("--steps", type=int, default=None, help="override stage2_steps")
    ap.add_argument("--episodes", type=int, default=20, help="eval episodes per teammate")
    args = ap.parse_args()

    from oaht_bench.configs import load_job
    from oaht_bench.dataset.dataset import Dataset
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.models.bc_agent import ConditionedBcAgent
    from oaht_bench.models.masking import mask_logits
    from oaht_bench.offline.evaluate import dataset_target_return, evaluate_agent_against
    from oaht_bench.offline.runner import _resolve_dims
    from oaht_bench.offline.training import get_optimizer
    from oaht_bench.offline.utils import masked_accuracy, to_jax
    from oaht_bench.population.pooled_crossplay import build_roster

    job = load_job(args.config)
    cfg = job.offline
    ds = Dataset(
        job.dataset_path,
        context_length=cfg.context_length,
        stride=cfg.stride,
        normalize=cfg.normalize_observations,
    )
    w = ds.windows
    batch = ds.batch

    # Dense teammate index: the dataset's teammate_id is the global pooled-roster
    # index; map the ones present to a contiguous 0..N-1 for the embedding table.
    tids = np.unique(w.teammate_id)
    dense = {int(t): d for d, t in enumerate(tids.tolist())}
    N = len(tids)
    print(f"windows={len(w)}  train teammates={N}  roster ids={tids.tolist()}")

    resolved = _resolve_dims(cfg, w.obs_dim, ds.action_dim)
    net_cfg = resolved.network
    from oaht_bench.models.bc_agent import BcNetwork

    net = BcNetwork(
        action_dim=net_cfg.action_dim,
        hidden_dim=net_cfg.hidden_dim,
        dropout=net_cfg.dropout,
        num_teammates=N,
    )

    KEYS = ("ego_obs", "ego_actions", "ego_rtg", "ego_avail", "timesteps", "mask")
    np_rng = np.random.default_rng(job.seed)

    def sample():
        idx = np_rng.choice(len(w), size=cfg.stage2_batch_size, replace=False)
        b = to_jax({k: getattr(w, k)[idx] for k in KEYS})
        tid = jnp.asarray([dense[int(t)] for t in w.teammate_id[idx]], jnp.int32)
        return b, tid

    steps = int(args.steps or cfg.stage2_steps)
    rng = jax.random.PRNGKey(job.seed)
    b0, t0 = sample()
    params = net.init(
        rng, b0["ego_rtg"], b0["ego_obs"], b0["ego_actions"], timesteps=b0["timesteps"], mask=b0["mask"], teammate_id=t0
    )

    def loss_fn(p, b, tid, key):
        logits = mask_logits(
            net.apply(
                p, b["ego_rtg"], b["ego_obs"], b["ego_actions"],
                timesteps=b["timesteps"], mask=b["mask"], teammate_id=tid,
                train=True, rngs={"dropout": key},
            ),
            b["ego_avail"],
        )
        m = b["mask"].astype(jnp.float32)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, b["ego_actions"])
        ce = (ce * m).sum() / jnp.maximum(m.sum(), 1.0)
        return ce, masked_accuracy(logits, b["ego_actions"], m)

    opt = get_optimizer(cfg, cfg.stage2_learning_rate, steps)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, o, b, tid, key):
        (ce, acc), g = jax.value_and_grad(loss_fn, has_aux=True)(p, b, tid, key)
        u, o = opt.update(g, o, p)
        return optax.apply_updates(p, u), o, ce, acc

    bar = tqdm(range(steps), desc="oracle BC", unit="step", dynamic_ncols=True)
    for i in bar:
        rng, key = jax.random.split(rng)
        b, tid = sample()
        params, opt_state, ce, acc = step(params, opt_state, b, tid, key)
        if i % cfg.log_every == 0 or i == steps - 1:
            bar.set_postfix_str(f"loss={float(ce):.3f} acc={float(acc):.3f}")

    # --- Closed-loop eval against each TRAIN teammate, feeding its true id ---------
    env = LogWrapper(make_env(job.env.env_name, job.env.env_kwargs()))
    roster = build_roster([__import__("pathlib").Path(p) for p in batch.meta["populations"]], env)
    target = dataset_target_return(batch)
    cond_target = target if w.norm is None else w.norm.apply_rtg(target)

    returns = {}
    for j, e in enumerate(tqdm(roster, desc="eval teammates", unit="mate")):
        if j not in dense or e.role not in ("self", "conf"):
            continue  # only teammates the model was trained to respond to
        agent = ConditionedBcAgent(
            resolved,
            context_length=cfg.context_length,
            target_return=cond_target,
            normalization=w.norm,
            teammate_id=dense[j],
            num_teammates=N,
        )
        agent.build_model()
        scores = evaluate_agent_against(
            agent,
            {"stage1": {}, "stage2": params},
            env,
            [(f"{e.generator}:{int(e.member)}:{e.role}", e.params, e.policy_cls)],
            rng=jax.random.PRNGKey(job.seed + 7),
            target_return=cond_target,
            max_episode_steps=job.env.rollout_length,
            num_episodes=args.episodes,
        )
        returns.update(scores.per_teammate)

    print("\n===== Teammate-id ORACLE (train teammates, perfect identity) =====")
    for label, r in sorted(returns.items()):
        print(f"  {label:24s} {r:7.3f}")
    print(f"  ---- mean {np.mean(list(returns.values())):.3f}  (unconditioned pooled BC train ~1.06)")


if __name__ == "__main__":
    main()
