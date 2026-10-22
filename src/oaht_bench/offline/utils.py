from typing import Any

import jax.numpy as jnp
import numpy as np
import optax

from oaht_bench.dataset.windowing import window_slice

# mask_logits moved to the model layer (models/masking.py) so the return-conditioned
# agents can mask at inference without models importing offline; re-exported here so
# the baselines keep importing it from offline.utils.
from oaht_bench.models.masking import mask_logits

__all__ = [
    "mask_logits",
    "masked_accuracy",
    "normalize_window_batch",
    "safe_cross_entropy",
    "sample_window_batch",
    "to_jax",
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


def safe_cross_entropy(logits, labels, mask) -> jnp.ndarray:
    """``softmax_cross_entropy_with_integer_labels``, immune to the padding
    sentinel poisoning a masked-out mean to ``nan``.

    Every padded action field in this codebase is filled with ``-10`` (an
    out-of-range "no action" sentinel -- see ``windowing.py``'s ``_FIELDS``),
    and ``optax.softmax_cross_entropy_with_integer_labels`` returns ``nan``
    for an out-of-range label. Every caller multiplies the result by ``mask``
    afterward to exclude padded positions from the loss, but ``nan * 0.0``
    is still ``nan`` in IEEE float -- so a single padded window anywhere in a
    batch poisons the whole masked-mean loss, not just its own term. This
    substitutes a valid dummy label (``0``) wherever ``mask`` is false before
    computing the cross-entropy, which changes nothing the mask doesn't
    already exclude from the result.
    """
    safe_labels = jnp.where(mask.astype(bool), labels, 0)
    return optax.softmax_cross_entropy_with_integer_labels(logits, safe_labels)


def normalize_window_batch(batch: dict, norm) -> dict:
    """Apply the dataset's observation/return-to-go normalisation to a window
    batch built from raw :func:`~oaht_bench.dataset.windowing.window_slice` reads.

    ``window_slice`` itself returns unnormalized data (it has no notion of a
    particular dataset's statistics -- see ``windowing.py``'s module
    docstring), so every baseline's ``_sample_batch`` applies this once, after
    stacking. Keys are matched by suffix (``*_obs`` -> ``norm.apply_obs``,
    ``*_rtg`` -> ``norm.apply_rtg``), which covers every observation/
    return-to-go field any baseline builds, prefixed or not (``mate_obs``,
    ``cross_mate_obs``, ``context_mate_next_obs``, ``ego_rtg``, ...). A no-op
    when ``norm`` is ``None`` (``normalize=False``).
    """
    if norm is None:
        return batch
    out = dict(batch)
    for key, value in batch.items():
        if key.endswith("_obs"):
            out[key] = norm.apply_obs(value)
        elif key.endswith("_rtg"):
            out[key] = norm.apply_rtg(value)
    return out


def sample_window_batch(dataset, pool, h: int, np_rng, batch_size: int, fields) -> dict:
    """Draw ``batch_size`` random windows from ``pool`` -- an
    ``(episode_ids, ts)`` pair as :func:`~oaht_bench.dataset.windowing.
    enumerate_windows` returns, or a filtered subset of one -- as a stacked,
    normalized jax batch.

    The ego-history baselines (BC, LIAM, MeLIBA) all sample the same way: a
    uniform draw of windows with no cross trajectory and no contrastive
    structure. Each call draws a fresh minibatch, so a caller looping steps
    ignores the training step index.
    """
    episode_ids, ts = pool
    rows = np_rng.choice(len(episode_ids), size=batch_size, replace=False)
    windows = [
        window_slice(dataset.batch, int(episode_ids[i]), int(ts[i]), h, fields, mate=dataset.mate)
        for i in rows
    ]
    batch = {k: np.stack([w[k] for w in windows]) for k in fields}
    return to_jax(normalize_window_batch(batch, dataset.norm))
