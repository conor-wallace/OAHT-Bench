"""Execute a :class:`~oaht_bench.configs.job.TrainingJob` (§3.1, §6).

Both trajectory-view baselines are two-stage: stage 1 learns a teammate
representation, stage 2 trains the policy against a frozen encoder. They differ
only in what the encoder reads and how its output reaches the policy, so one
loop drives both and the differences live in :mod:`~oaht_bench.offline.liam` and
:mod:`~oaht_bench.offline.tao`.

Follows the conventions teammate generation established: the config's content
hash names the run directory, the fully-resolved config is written into it, the
run refuses to start if it would overwrite an artifact, and **parameters are
saved before anything is reported** — a charting bug must not be able to discard
a finished run.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from oaht_bench.configs.job import TrainingJob

log = logging.getLogger(__name__)

#: Baselines this runner can train. The roster in ``BaselineName`` is larger;
#: the rest raise rather than silently training something else.
SUPPORTED = ("liam", "tao")


def _schedule(cfg):
    """Linear warmup then constant, as the reference schedules it.

    ``lambda steps: min((steps + 1) / warmup_steps, 1)`` on top of AdamW.
    """
    import jax.numpy as jnp
    import optax

    # jnp, not np: the step count is a traced array inside the jitted update.
    def scale(step):
        return jnp.minimum((step + 1) / cfg.warmup_steps, 1.0)

    return optax.scale_by_schedule(scale)


def _optimiser(cfg, learning_rate: float):
    import optax

    return optax.chain(
        optax.clip_by_global_norm(cfg.clip_grad),
        optax.adamw(learning_rate=learning_rate, weight_decay=cfg.weight_decay),
        _schedule(cfg),
    )


def _train_stage(loss_fn, params, batches, *, optimiser, steps, rng, logger, prefix,
                 log_every):
    """Run one stage, returning the trained parameters.

    ``batches`` is a callable taking a step index and returning a batch, so the
    sampler is re-invoked every step — TAO's batches are structured (positives
    per anchor, a GetOffD context per window) and cannot be precomputed once.
    """
    import jax
    import optax

    opt_state = optimiser.init(params)

    @jax.jit
    def step(params, opt_state, batch, key):
        (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, {"dropout": key}
        )
        updates, opt_state = optimiser.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, aux

    for i in range(steps):
        rng, key = jax.random.split(rng)
        params, opt_state, aux = step(params, opt_state, batches(i), key)
        if i % log_every == 0 or i == steps - 1:
            for name, value in aux.items():
                logger.log_item(f"{prefix}/{name}", float(value), train_step=i)
            logger.commit()
    return params


def run(job: TrainingJob) -> Path:
    """Train one baseline and return the run directory."""
    import jax
    import jax.numpy as jnp

    from oaht_bench.common.logging import RunLogger, nonfatal
    from oaht_bench.configs import save_job
    from oaht_bench.data.schema import EpisodeBatch
    from oaht_bench.offline import (
        AncillaryActionDecoder,
        LiamDecoder,
        LiamEncoder,
        LiamPolicy,
        OpponentPolicyEncoder,
        TaoPolicy,
        TeammateIndex,
        embedding_loss,
        liam_policy_loss,
        liam_reconstruction_loss,
        make_windows,
        sample_stage1,
        sample_stage2,
        tao_policy_loss,
    )

    if job.baseline not in SUPPORTED:
        raise NotImplementedError(
            f"baseline={job.baseline!r} has no runner yet; implemented: "
            f"{sorted(SUPPORTED)}. The roster in BaselineName is the plan, not "
            f"what exists."
        )

    run_dir = Path(job.run_dir())
    existing = run_dir / "params.pkl"
    if existing.exists():
        raise FileExistsError(
            f"{existing} already exists and would be overwritten. Delete "
            f"{run_dir} to retrain, or change the job's label. (The directory "
            f"name includes the config hash, so an identical config always "
            f"resolves here.)"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_job(job, run_dir / "job.json", minimal=False)

    cfg = job.offline
    batch = EpisodeBatch.load(Path(job.dataset_path))
    windows = make_windows(batch, context_length=cfg.context_length, stride=cfg.stride)
    index = TeammateIndex.build(windows)
    action_dim = int(batch.avail_actions.shape[-1])
    log.info(
        "dataset %s -> %d windows, %d teammates, obs_dim %d, action_dim %d",
        job.dataset_path, len(windows), len(index.teammates), windows.obs_dim, action_dim,
    )

    np_rng = np.random.default_rng(job.seed)
    rng = jax.random.PRNGKey(job.seed)
    to_jax = lambda d: {k: jnp.asarray(v) for k, v in d.items()}  # noqa: E731

    with RunLogger(
        run_dir,
        use_wandb=job.logging.use_wandb,
        wandb_project=job.logging.wandb_project,
        wandb_entity=job.logging.wandb_entity,
        config=json.loads(job.canonical_json()),
        verbose=job.logging.verbose,
    ) as logger:
        common = dict(hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)

        if job.baseline == "liam":
            encoder = LiamEncoder(action_dim=action_dim, **common)
            decoder = LiamDecoder(
                obs_dim=windows.obs_dim, action_dim=action_dim, hidden_dim=cfg.hidden_dim
            )
            policy = LiamPolicy(action_dim=action_dim, **common)

            # LIAM's encoder reads the ego stream, so a stage-1 batch is just
            # windows -- no cross trajectory and no contrastive term.
            def stage1_batch(_):
                idx = np_rng.choice(len(windows), size=cfg.stage2_batch_size, replace=False)
                return to_jax({
                    k: getattr(windows, k)[idx] for k in
                    ("ego_obs", "ego_actions", "ego_rtg", "mate_obs", "mate_actions",
                     "timesteps", "mask")
                })

            probe = stage1_batch(0)
            rng, k1, k2 = jax.random.split(rng, 3)
            enc_p = encoder.init(k1, probe["ego_rtg"], probe["ego_obs"],
                                 probe["ego_actions"], timesteps=probe["timesteps"],
                                 mask=probe["mask"])
            z = encoder.apply(enc_p, probe["ego_rtg"], probe["ego_obs"],
                              probe["ego_actions"], timesteps=probe["timesteps"],
                              mask=probe["mask"])
            dec_p = decoder.init(k2, z)
            stage1_params = {"encoder": enc_p, "decoder": dec_p}

            def stage1_loss(p, b, rngs):
                return liam_reconstruction_loss(p, encoder, decoder, b, rngs=rngs)

            def stage2_batch(_):
                return stage1_batch(0)

            def make_stage2(p1):
                rng_, k = jax.random.split(rng, 2)
                pol_p = policy.init(k, probe["ego_rtg"], probe["ego_obs"],
                                    probe["ego_actions"], timesteps=probe["timesteps"],
                                    embedding=z, mask=probe["mask"])

                def loss(p, b, rngs):
                    return liam_policy_loss(p, policy, encoder, p1["encoder"], b, rngs=rngs)

                return pol_p, loss

        else:  # tao
            encoder = OpponentPolicyEncoder(
                action_dim=action_dim, hidden_dim=cfg.hidden_dim, ff_dim=cfg.ff_dim,
                num_blocks=cfg.num_blocks, dropout=cfg.dropout,
            )
            decoder = AncillaryActionDecoder(
                action_dim=action_dim, hidden_dim=cfg.hidden_dim
            )
            policy = TaoPolicy(action_dim=action_dim, **common)

            def stage1_batch(_):
                return to_jax(sample_stage1(
                    windows, index, np_rng,
                    teammates_per_batch=cfg.teammates_per_batch,
                    windows_per_teammate=cfg.windows_per_teammate,
                ))

            def stage2_batch(_):
                return to_jax(sample_stage2(
                    windows, index, np_rng,
                    batch_size=cfg.stage2_batch_size,
                    context_trajectories=cfg.context_trajectories,
                ))

            probe = stage1_batch(0)
            rng, k1, k2 = jax.random.split(rng, 3)
            enc_p = encoder.init(k1, probe["mate_next_obs"], probe["mate_actions"],
                                 probe["mate_rewards"], mask=probe["mask"],
                                 timesteps=probe["timesteps"])
            tok = encoder.apply(enc_p, probe["mate_next_obs"], probe["mate_actions"],
                                probe["mate_rewards"], mask=probe["mask"],
                                timesteps=probe["timesteps"])
            dec_p = decoder.init(k2, probe["cross_mate_obs"],
                                 OpponentPolicyEncoder.pool(tok))
            stage1_params = {"encoder": enc_p, "decoder": dec_p}

            def stage1_loss(p, b, rngs):
                return embedding_loss(p, encoder, decoder, b, alpha=cfg.alpha,
                                      lam=cfg.lam, rngs=rngs)

            def make_stage2(p1):
                p2_probe = stage2_batch(0)
                ctx = encoder.apply(
                    p1["encoder"], p2_probe["context_mate_next_obs"],
                    p2_probe["context_mate_actions"], p2_probe["context_mate_rewards"],
                    mask=p2_probe["context_mask"], timesteps=p2_probe["context_timesteps"],
                )
                _, k = jax.random.split(rng, 2)
                pol_p = policy.init(
                    k, p2_probe["ego_rtg"], p2_probe["ego_obs"], p2_probe["ego_actions"],
                    timesteps=p2_probe["timesteps"], context=ctx,
                    mask=p2_probe["mask"], context_mask=p2_probe["context_mask"],
                )
                params = {"policy": pol_p, "encoder": p1["encoder"]}

                def loss(p, b, rngs):
                    return tao_policy_loss(p, policy, encoder, b,
                                           freeze_encoder=cfg.freeze_encoder, rngs=rngs)

                return params, loss

        rng, s1_rng, s2_rng = jax.random.split(rng, 3)
        log.info("stage 1: %d steps", cfg.stage1_steps)
        stage1_params = _train_stage(
            stage1_loss, stage1_params, stage1_batch,
            optimiser=_optimiser(cfg, cfg.stage1_learning_rate),
            steps=cfg.stage1_steps, rng=s1_rng, logger=logger,
            prefix="Stage1", log_every=cfg.log_every,
        )

        log.info("stage 2: %d steps", cfg.stage2_steps)
        stage2_params, stage2_loss = make_stage2(stage1_params)
        stage2_params = _train_stage(
            stage2_loss, stage2_params, stage2_batch,
            optimiser=_optimiser(cfg, cfg.stage2_learning_rate),
            steps=cfg.stage2_steps, rng=s2_rng, logger=logger,
            prefix="Stage2", log_every=cfg.log_every,
        )

        # Save before reporting. A charting failure after a long run must not
        # discard it -- the lesson from teammate generation.
        out: dict[str, Any] = {"stage1": stage1_params, "stage2": stage2_params}
        with (run_dir / "params.pkl").open("wb") as fh:
            pickle.dump(jax.device_get(out), fh)

        with nonfatal(f"{job.baseline} post-training summary"):
            (run_dir / "training_summary.json").write_text(
                json.dumps(
                    {
                        "baseline": job.baseline,
                        "windows": len(windows),
                        "teammates": len(index.teammates),
                        "obs_dim": windows.obs_dim,
                        "action_dim": action_dim,
                        "stage1_steps": cfg.stage1_steps,
                        "stage2_steps": cfg.stage2_steps,
                    },
                    indent=2,
                )
                + "\n"
            )

    return run_dir
