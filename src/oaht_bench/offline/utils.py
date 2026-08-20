from typing import Any

import jax.numpy as jnp


def masked_accuracy(logits, labels, mask) -> jnp.ndarray:
    """Top-1 accuracy over valid timesteps.

    Reported alongside every cross-entropy term because a loss is not
    interpretable across baselines or datasets — 0.69 nats means one thing with
    two actions and another with six — while "fraction of actions predicted
    correctly" is comparable and has an obvious floor at chance.
    """
    correct = (jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)
    m = mask.astype(jnp.float32)
    return (correct * m).sum() / jnp.maximum(m.sum(), 1.0)


def mask_logits(logits, avail):
    """Suppress unavailable actions, as every absorbed policy does.

    ``logits - (1 - avail) * 1e10`` (``agents/mlp_actor_critic.py:36-37``) rather
    than ``-inf``, which would make a fully-masked row NaN after softmax.

    Collection already applies this: every seat's ``get_action`` receives
    ``avail_actions``, so a recorded action is always legal. Without it here the
    learned policy is trained and evaluated under a weaker constraint than the
    data was generated under -- on LBF that is 20.5% of (step, action) pairs, and
    action 5 is unavailable 67% of the time.
    """
    if avail is None:
        return logits
    return logits - (1.0 - avail) * 1e10


def to_jax(data: dict[str, Any]) -> dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v) for k, v in data.items()}


#: Window fields an ego-history stage batch needs: the ego stream, the teammate
#: targets a decoder reconstructs, availabilities, timesteps and the validity
#: mask. Shared by the baselines whose encoder reads the ego stream (LIAM, MeLIBA,
#: OMIS); TAO samples structured contrastive batches instead.
WINDOW_BATCH_KEYS = (
    "ego_obs",
    "ego_actions",
    "ego_rtg",
    "mate_obs",
    "mate_actions",
    "ego_avail",
    "mate_avail",
    "timesteps",
    "mask",
)


def sample_window_batch(windows, np_rng, size):
    """Draw a random minibatch of windows as jax arrays.

    The ego-history baselines all sample the same way: a uniform draw of window
    indices with no cross trajectory and no contrastive structure. Each call
    draws a fresh minibatch, so the caller ignores the training step index.
    """
    idx = np_rng.choice(len(windows), size=size, replace=False)
    return to_jax({k: getattr(windows, k)[idx] for k in WINDOW_BATCH_KEYS})
