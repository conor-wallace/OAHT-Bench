"""Online mechanism probes: does the teammate-modeling signal reach the policy, live?

Stage-1 metrics (``recon_action_accuracy`` etc.) are measured on STATIC dataset
windows -- teacher-forced on the real ego history. This measures the same quantity
during an actual live rollout of the trained agent against a real teammate, which
is where the identification probes (docs/tuning_record.md) found a large offline
-> online gap on a comparable classification task (24% -> 10%): the ego's own
past actions are teacher-forced observations in the dataset but the model's own
(possibly wrong) interventions once it is acting, so a static-window number can
overstate what the deployed policy actually has to work with.

``mate_action_acc`` (online): at each step of a live episode, decode the
teammate's predicted action from the SAME embedding the policy conditions on
that step, and compare it to the teammate's REAL action taken that step. Report
against the modal-action floor -- an accuracy number alone is not evidence of
anything, a lesson from this investigation (tuning_record.md); a true
context-conditional entropy ceiling is not computed here and would need one.
Reported separately for TRAIN and HELD-OUT teammates.

**Implemented for LIAM only.** Its window bookkeeping is a line-for-line,
commented copy of ``ReturnConditionedAgent.get_action`` (necessary because that
method is jitted and does not expose its intermediate embedding) -- copied
rather than refactored so it stays visibly in sync with the production path it
mirrors. MeLIBA (belief mean/logvar, not a point embedding), OMIS (imitator head
on a different encoder), and TAO (cross-attention context, not a decoder) each
need their own decode step; they raise ``NotImplementedError`` naming exactly
what differs, rather than silently reporting a wrong number.

Usage:
    uv run python scripts/mechanism_probes.py \\
        --run-dir results/training/liam_pooled_expert_lbf_12x12-e5565df35528 \\
        --episodes 100 --seed-index 0
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
from oaht_bench.envs import make_env
from oaht_bench.models.masking import mask_logits
from oaht_bench.models.return_conditioned_agent import ContextWindow
from oaht_bench.offline.evaluate import dataset_target_return
from oaht_bench.offline.runner import _resolve_dims, _teammate_policies


def _agent_for(architecture: str):
    if architecture == "liam":
        from oaht_bench.models.liam_agent import LiamAgent

        return LiamAgent
    raise NotImplementedError(
        f"mate_action_acc is only wired for LIAM right now (architecture={architecture!r} "
        f"needs its own decode step -- MeLIBA decodes from (mean, logvar) belief params "
        f"not a point embedding, OMIS's imitator head lives on OmisModel not the encoder, "
        f"TAO has no decoder at all, its embedding is read via cross-attention). See this "
        f"module's docstring for what each needs."
    )


def load_run(run_dir: Path, *, seed_index: int):
    job = load_job(run_dir / "job.json")
    cfg = job.offline
    dataset = Dataset(
        job.dataset_path,
        context_length=cfg.context_length,
        stride=cfg.stride,
        normalize=cfg.normalize_observations,
    )
    with (run_dir / "params.pkl").open("rb") as fh:
        params = pickle.load(fh)
    num_seeds = int(job.num_seeds)
    if num_seeds > 1:
        # normalization has no seed axis -- only stage1/stage2 params do.
        params = {
            "stage1": jax.tree.map(lambda x: x[seed_index], params["stage1"]),
            "stage2": jax.tree.map(lambda x: x[seed_index], params["stage2"]),
        }

    resolved = _resolve_dims(cfg, dataset.obs_dim, dataset.action_dim)
    target = dataset_target_return(dataset.batch)
    cond_target = target if dataset.windows.norm is None else dataset.windows.norm.apply_rtg(target)
    agent_cls = _agent_for(resolved.network.architecture)
    agent = agent_cls(
        resolved,
        context_length=cfg.context_length,
        target_return=cond_target,
        normalization=dataset.windows.norm,
    )
    agent.build_model()
    return job, dataset, agent, params


def rollout_with_mate_probe(
    agent, params, env, mate_params, mate_cls, *, episodes, max_steps, seed
):
    """LIAM only. Returns ``(pred_mate_action, true_mate_action, valid, ego_return)``,
    the first three shaped ``(episodes, max_steps)`` and ``ego_return`` summed to
    ``(episodes,)``. ``valid`` is False from the step the episode ends onward --
    apply it before scoring ``mate_action_acc``, the same way ``mask`` gates every
    other accuracy in this codebase.

    The block marked REPLICATED is copied from
    :meth:`ReturnConditionedAgent.get_action` (obs normalisation, RTG decrement,
    left-padded window roll, causal write-then-sample) so the same window that
    conditions the policy can also be decoded for the teammate prediction --
    ``get_action`` itself is jitted and returns only the sampled action. The
    freeze-on-done guard is copied from :func:`run_episodes`'s ``_compiled_rollout``
    (``jax.lax.cond(done, freeze, take_step)``) -- a first pass of this probe
    omitted it and kept stepping (and accruing reward) past episode end, inflating
    the online return by ~30-40% versus the real ``get_action`` path.
    """
    stage1, stage2 = params["stage1"], params["stage2"]
    K = agent.context_length

    def one(key):
        k, rk = jax.random.split(key)
        obs0, state0 = env.reset(rk)
        h0 = agent.init_hstate(1)
        h1 = mate_cls.init_hstate(1, aux_info={"agent_id": 1})
        rw0 = jnp.zeros(())
        done0 = jnp.asarray(False)

        def take_step(carry):
            state, obs, h, h1, rw, k, _done = carry
            k, k0, k1, ks = jax.random.split(k, 4)
            av = jax.lax.stop_gradient(env.get_avail_actions(state))

            # --- REPLICATED: ReturnConditionedAgent.get_action's window roll ---
            rtg = h.rtg - rw / agent._rtg_scale
            ctx_obs = jnp.roll(h.ctx_obs, -1, axis=0)
            ctx_act = jnp.roll(h.ctx_act, -1)
            ctx_rtg = jnp.roll(h.ctx_rtg, -1)
            ctx_t = jnp.roll(h.ctx_t, -1)
            ctx_mask = jnp.roll(h.ctx_mask, -1)
            norm_obs = (
                jnp.reshape(obs["agent_0"], (-1)).astype(jnp.float32) - agent._obs_mean
            ) / agent._obs_std
            ctx_obs = ctx_obs.at[-1].set(norm_obs)
            ctx_act = ctx_act.at[-1].set(jnp.int32(-10))
            ctx_rtg = ctx_rtg.at[-1].set(rtg)
            ctx_t = ctx_t.at[-1].set(jnp.minimum(h.step + 1, K * 64).astype(jnp.int32))
            ctx_mask = ctx_mask.at[-1].set(True)
            # --- end of the copied portion; instrumentation starts here ---

            z = agent.encoder.apply(
                stage1["encoder"],
                ctx_rtg[None],
                ctx_obs[None],
                ctx_act[None],
                timesteps=ctx_t[None],
                mask=ctx_mask[None],
                train=False,
            )
            _, mate_act_logits = agent.decoder.apply(stage1["decoder"], z[:, -1])
            # The teammate could only have taken a legal action -- liam_reconstruction_loss
            # masks illegal actions before argmax (offline/liam.py); skipping this made a
            # first pass of this probe collapse onto whichever action is illegal least often.
            avail1 = jnp.reshape(av["agent_1"], (-1)).astype(jnp.float32)
            mate_act_logits_masked = mask_logits(mate_act_logits[0], avail1)
            pred_mate_action = jnp.argmax(mate_act_logits_masked).astype(jnp.int32)

            logits = agent.network.apply(
                stage2,
                ctx_rtg[None],
                ctx_obs[None],
                ctx_act[None],
                timesteps=ctx_t[None],
                embedding=z,
                mask=ctx_mask[None],
                train=False,
            )
            avail0 = jnp.reshape(av["agent_0"], (-1)).astype(jnp.float32)
            masked = mask_logits(logits[0, -1], avail0)
            a0 = jax.random.categorical(k0, masked).astype(jnp.int32)
            ctx_act = ctx_act.at[-1].set(a0)
            new_h = ContextWindow(ctx_obs, ctx_act, ctx_rtg, ctx_t, ctx_mask, rtg, h.step + 1)

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
            a1 = a1.squeeze()
            act = {"agent_0": a0, "agent_1": a1}
            obs2, state2, r2, done2, _ = env.step(ks, state, act)
            new_carry = (state2, obs2, new_h, h1n, r2["agent_0"].reshape(()), k, done2["__all__"])
            return new_carry, (pred_mate_action, a1, r2["agent_0"])

        def step(carry, _):
            _, _, _, _, _, _, done = carry
            new_carry, (pred, a1, r2) = jax.lax.cond(
                done, lambda c: (c, (jnp.int32(0), jnp.int32(0), jnp.zeros(()))), take_step, carry
            )
            return new_carry, (pred, a1, r2, ~done)

        _, (pred, true_mate, ego_r, valid) = jax.lax.scan(
            step, (state0, obs0, h0, h1, rw0, k, done0), None, length=max_steps
        )
        return pred, true_mate, valid, ego_r.sum()

    return jax.vmap(one)(jax.random.split(jax.random.PRNGKey(seed), episodes))


def _report(
    name: str, pred: np.ndarray, true: np.ndarray, valid: np.ndarray, n_actions: int
) -> None:
    v = valid.astype(bool)
    acc = float((pred == true)[v].mean())
    modal = float(np.bincount(true[v].reshape(-1), minlength=n_actions).max() / v.sum())
    print(
        f"  {name:10s} online mate_action_acc={acc:.3f}  (modal floor={modal:.3f})  "
        f"n_valid_steps={int(v.sum())}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    job, dataset, agent, params = load_run(args.run_dir, seed_index=args.seed_index)
    env = make_env(job.env.env_name, job.env.env_kwargs())

    print(f"baseline={job.baseline}  env={job.env.name}  seed_index={args.seed_index}")
    for split in ("train", "held_out"):
        teammates = _teammate_policies(dataset.batch, env, which=split)
        if not teammates:
            print(f"  {split}: no teammates in this split")
            continue
        preds, trues, valids = [], [], []
        for label, mate_params, mate_cls in teammates:
            pred, true_mate, valid, ego_ret = rollout_with_mate_probe(
                agent,
                params,
                env,
                mate_params,
                mate_cls,
                episodes=args.episodes,
                max_steps=job.env.rollout_length,
                seed=args.seed,
            )
            preds.append(np.asarray(pred))
            trues.append(np.asarray(true_mate))
            valids.append(np.asarray(valid))
            print(f"    {label}: online mean ego return={float(np.asarray(ego_ret).mean()):.4f}")
        _report(
            split,
            np.concatenate(preds),
            np.concatenate(trues),
            np.concatenate(valids),
            dataset.action_dim,
        )


if __name__ == "__main__":
    main()
