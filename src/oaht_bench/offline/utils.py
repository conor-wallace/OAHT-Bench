from typing import Any

import jax.numpy as jnp

# mask_logits moved to the model layer (models/masking.py) so the return-conditioned
# agents can mask at inference without models importing offline; re-exported here so
# the baselines keep importing it from offline.utils.
from oaht_bench.models.masking import mask_logits

__all__ = [
    "mask_logits",
    "masked_accuracy",
    "to_jax",
    "sample_window_batch",
    "WINDOW_BATCH_KEYS",
]


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
