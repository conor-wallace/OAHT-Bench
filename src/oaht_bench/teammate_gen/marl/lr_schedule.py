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

``accumulation_steps > 1`` (MEP's gradient accumulation across fresh micro-rollouts,
``teammate_gen/mep.py``) wraps the returned optimizer in ``optax.MultiSteps``, which
only calls through to the wrapped schedule's ``count`` on a *real* (accumulated)
update -- intermediate accumulation-only calls recompute the same ``count`` from the
same not-yet-advanced starting state and get discarded. So under accumulation,
``count`` already advances once per real update, not once per raw minibatch step, and
must not be divided by ``steps_per_update`` again -- that division is what recovered
"how many rollouts have completed" in the non-accumulated case, where every raw step
was itself a real update.
"""

from __future__ import annotations

import optax


def make_lr_schedule(ppo, num_updates: int, accumulation_steps: int = 1):
    """Return the learning-rate optax passes to ``adam`` -- a float or a schedule fn.

    ``ppo`` is a :class:`~oaht_bench.configs.teammate_gen.PpoHyperparams`.
    """
    steps_per_update = ppo.num_minibatches * ppo.update_epochs
    if accumulation_steps > 1:
        # count already counts real (accumulated) updates directly -- see module
        # docstring -- so no further division recovers anything.
        effective_num_updates = num_updates // accumulation_steps
        effective_steps_per_update = 1
    else:
        effective_num_updates = num_updates
        effective_steps_per_update = steps_per_update

    if ppo.lr_warmup > 0:
        total_steps = effective_num_updates * effective_steps_per_update
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
            frac = 1.0 - (count // effective_steps_per_update) / effective_num_updates
            return ppo.learning_rate * frac

        return linear_schedule

    return ppo.learning_rate
