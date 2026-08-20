"""The generic two-stage training loop, shared by every baseline.

A baseline supplies *what* to optimise -- a loss, an initial parameter tree, and
a per-step batch sampler -- and this module supplies *how*: the AdamW-with-warmup
optimizer and the jitted gradient loop that logs each stage. It lives apart from
:mod:`oaht_bench.offline.runner` so that :class:`~oaht_bench.offline.registry.BaseAhtPolicy`
can drive a stage without importing the runner (which imports the policies).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


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


def train(loss_fn, params, batches, *, optimizer, steps, rng, logger, prefix, log_every):
    """Run one stage, returning the trained parameters.

    ``batches`` is a callable taking a step index and returning a batch, so the
    sampler is re-invoked every step -- TAO's batches are structured (positives
    per anchor, a GetOffD context per window) and cannot be precomputed once.
    """

    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, batch, key):
        (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, {"dropout": key})
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, aux

    for i in range(steps):
        rng, key = jax.random.split(rng)
        params, opt_state, aux = step(params, opt_state, batches(i), key)
        if i % log_every == 0 or i == steps - 1:
            for name, value in aux.items():
                logger.log_item(f"{prefix}/{name}", float(value), train_step=i)
            logger.commit()
    return params
