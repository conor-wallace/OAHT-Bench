"""Shared learning-rate schedule for the PPO generators.

The three PPO trainers (``marl/ippo.py``, ``brdiv.py``, ``lbrdiv.py``) all built the
same learning-rate value inline -- a linear 1->0 decay when ``anneal_lr`` is set,
else a constant. This factors that out and adds the OvercookedV2 schedule (Gessler
et al., ICLR 2025, App. D): a linear warmup from 0 to ``learning_rate`` over the
first ``lr_warmup`` fraction of updates, then a cosine decay to 0.

The optimiser steps optax once per minibatch per epoch, so a single "update" spans
``num_minibatches * update_epochs`` optax steps; the existing linear schedule divides
the step count by that to recover updates, and the warmup/decay bounds below are in
the same optax-step units so the two schedules share a clock.

``lr_warmup == 0`` returns exactly the previous value (the linear callable or the raw
float), so every already-tuned config is byte-for-byte unchanged.
"""

from __future__ import annotations

import optax


def make_lr_schedule(ppo, num_updates: int):
    """Return the learning-rate optax passes to ``adam`` -- a float or a schedule fn.

    ``ppo`` is a :class:`~oaht_bench.configs.teammate_gen.PpoHyperparams`.
    """
    steps_per_update = ppo.num_minibatches * ppo.update_epochs

    if ppo.lr_warmup > 0:
        total_steps = num_updates * steps_per_update
        warmup_steps = max(1, int(ppo.lr_warmup * total_steps))
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=ppo.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=max(warmup_steps + 1, total_steps),
            end_value=0.0,
        )

    if ppo.anneal_lr:

        def linear_schedule(count):
            frac = 1.0 - (count // steps_per_update) / num_updates
            return ppo.learning_rate * frac

        return linear_schedule

    return ppo.learning_rate
