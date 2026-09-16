"""The generic two-stage training loop, shared by every baseline.

A baseline supplies *what* to optimise -- a loss, an initial parameter tree, and
a per-step batch sampler -- and this module supplies *how*: the AdamW-with-warmup
optimizer and the jitted gradient loop that logs each stage. It lives apart from
:mod:`oaht_bench.offline.runner` so that :class:`~oaht_bench.offline.registry.BaseAhtTrainer`
can drive a stage without importing the runner (which imports the policies).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm


def _loss_key(aux: dict) -> str | None:
    """The aux metric to surface on the progress bar: the loss if there is one,
    else the first metric, else nothing (an empty aux)."""
    if not aux:
        return None
    return next((k for k in aux if "loss" in k), next(iter(aux)))


def get_scheduler(cfg, total_steps: int):
    """Linear warmup then constant, as the reference schedules it.

    ``lambda steps: min((steps + 1) / warmup, 1)`` on top of AdamW, with warmup
    a fraction of *this* stage rather than a shared constant.
    """

    # jnp, not np: the step count is a traced array inside the jitted update.
    warmup = max(1.0, total_steps * cfg.warmup_fraction)

    def scale(step):
        return jnp.minimum((step + 1) / warmup, 1.0)

    return optax.scale_by_schedule(scale)


def get_optimizer(cfg, learning_rate: float, total_steps: int):
    return optax.chain(
        optax.clip_by_global_norm(cfg.clip_grad),
        optax.adamw(learning_rate=learning_rate, weight_decay=cfg.weight_decay),
        get_scheduler(cfg, total_steps),
    )


def train(
    loss_fn,
    params,
    batches,
    *,
    optimizer,
    steps,
    rng,
    logger,
    prefix,
    log_every,
    num_seeds: int = 1,
    frozen=None,
):
    """Run one stage, returning the trained parameters.

    ``loss_fn(params, batch, rngs, frozen)`` -- ``frozen`` is a per-seed pytree the
    loss reads but does not update (a frozen stage-1 representation for the methods
    that keep it out of ``params``), or ``None``.

    ``batches`` is a callable taking a step index and returning a batch, so the
    sampler is re-invoked every step -- TAO's batches are structured (positives per
    anchor, a GetOffD context per window) and cannot be precomputed once.

    With ``num_seeds > 1`` the update is vmapped over a leading seed axis: ``params``,
    ``opt_state``, the per-step batch and ``frozen`` all carry ``(num_seeds, ...)``,
    so N independently-initialised seeds train in parallel on one device and each aux
    metric is logged as its mean and std over seeds. ``num_seeds == 1`` is the
    original single-seed loop, byte-for-byte (no seed axis on the returned params).
    """

    if num_seeds == 1:
        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state, batch, key):
            (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, batch, {"dropout": key}, frozen
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, aux

        bar = tqdm(range(steps), desc=prefix, unit="step", dynamic_ncols=True)
        for i in bar:
            rng, key = jax.random.split(rng)
            params, opt_state, aux = step(params, opt_state, batches(i), key)
            if i % log_every == 0 or i == steps - 1:
                for name, value in aux.items():
                    logger.log_item(f"{prefix}/{name}", float(value), train_step=i)
                logger.commit()
                key_metric = _loss_key(aux)
                if key_metric is not None:
                    bar.set_postfix_str(f"{key_metric}={float(aux[key_metric]):.4f}")
        return params

    opt_state = jax.vmap(optimizer.init)(params)
    frozen_axis = None if frozen is None else 0

    @jax.jit
    def step(params, opt_state, batch, keys):
        def one(p, o, b, k, f):
            (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, b, {"dropout": k}, f)
            updates, o = optimizer.update(grads, o, p)
            return optax.apply_updates(p, updates), o, aux

        return jax.vmap(one, in_axes=(0, 0, 0, 0, frozen_axis))(
            params, opt_state, batch, keys, frozen
        )

    bar = tqdm(range(steps), desc=prefix, unit="step", dynamic_ncols=True)
    for i in bar:
        rng, sub = jax.random.split(rng)
        keys = jax.random.split(sub, num_seeds)
        params, opt_state, aux = step(params, opt_state, batches(i), keys)
        if i % log_every == 0 or i == steps - 1:
            means = {}
            for name, value in aux.items():
                v = jnp.asarray(value)  # (num_seeds,)
                means[name] = float(v.mean())
                logger.log_item(f"{prefix}/{name}", means[name], train_step=i)
                logger.log_item(f"{prefix}/{name}_std", float(v.std()), train_step=i)
            logger.commit()
            key_metric = _loss_key(means)
            if key_metric is not None:
                bar.set_postfix_str(f"{key_metric}={means[key_metric]:.4f} (mean/{num_seeds})")
    return params
