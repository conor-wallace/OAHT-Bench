"""``offline/utils.py``'s shared primitives."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import optax

from oaht_bench.offline.utils import safe_cross_entropy


def test_safe_cross_entropy_does_not_let_a_padded_label_poison_the_batch():
    """``optax.softmax_cross_entropy_with_integer_labels`` returns ``nan`` for
    an out-of-range label (the ``-10`` padding sentinel every action field
    uses), and every caller multiplies by ``mask`` afterward -- but
    ``nan * 0.0 == nan``, so one padded row would otherwise poison the whole
    masked-mean loss, not just its own contribution.
    """
    logits = jnp.zeros((3, 6))
    labels = jnp.array([-10, 0, 3])
    mask = jnp.array([0.0, 1.0, 1.0])

    out = safe_cross_entropy(logits, labels, mask)
    assert not bool(jnp.any(jnp.isnan(out))), "a masked-out row must never be nan"

    masked_mean = (out * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    assert not bool(jnp.isnan(masked_mean))

    # Unmasked rows are byte-identical to the raw optax call -- this changes
    # nothing the mask doesn't already exclude from the result.
    raw = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    np.testing.assert_allclose(np.asarray(out[1:]), np.asarray(raw[1:]))
