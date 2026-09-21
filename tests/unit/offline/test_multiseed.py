"""Multi-seed training: the vmapped train() loop and its single-seed contract.

Pins the two things that matter: single-seed is byte-identical (no seed axis added),
and multi-seed trains N independently-initialised seeds in parallel, each with its own
frozen input, so the returned params carry a leading seed axis.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from oaht_bench.offline.training import train


class _NullLogger:
    def log_item(self, *a, **k):
        pass

    def commit(self):
        pass


def _loss(p, b, rngs, frozen):
    # drive p["w"] toward `frozen` if given, else toward the batch target
    target = frozen if frozen is not None else b["x"]
    mse = jnp.mean((p["w"] - target) ** 2)
    return mse, {"mse": mse}


def test_single_seed_has_no_seed_axis_and_converges():
    p0 = {"w": jnp.zeros(3)}
    out = train(
        _loss, p0, lambda i: {"x": jnp.ones(3)},
        optimizer=optax.sgd(0.5), steps=60, rng=jax.random.PRNGKey(0),
        logger=_NullLogger(), prefix="S", log_every=100, num_seeds=1, frozen=jnp.ones(3),
    )
    assert out["w"].shape == (3,)  # no seed axis
    assert np.allclose(np.asarray(out["w"]), 1.0, atol=1e-3)


def test_multi_seed_adds_axis_and_trains_each_seed_independently():
    ns = 3
    pN = {"w": jnp.zeros((ns, 3))}
    # per-seed frozen targets 1, 2, 3
    frozen = jnp.array([1.0, 2.0, 3.0])[:, None] * jnp.ones((ns, 3))
    out = train(
        _loss, pN, lambda i: {"x": jnp.ones((ns, 3))},
        optimizer=optax.sgd(0.5), steps=60, rng=jax.random.PRNGKey(0),
        logger=_NullLogger(), prefix="S", log_every=100, num_seeds=ns, frozen=frozen,
    )
    assert out["w"].shape == (ns, 3)  # leading seed axis
    per_seed = np.asarray(out["w"]).mean(axis=1)
    assert np.allclose(per_seed, [1.0, 2.0, 3.0], atol=1e-3)


def test_multi_seed_without_frozen_uses_shared_batch():
    ns = 4
    pN = {"w": jnp.zeros((ns, 2))}
    out = train(
        _loss, pN, lambda i: {"x": jnp.full((ns, 2), 5.0)},
        optimizer=optax.sgd(0.5), steps=50, rng=jax.random.PRNGKey(1),
        logger=_NullLogger(), prefix="S", log_every=100, num_seeds=ns, frozen=None,
    )
    assert out["w"].shape == (ns, 2)
    assert np.allclose(np.asarray(out["w"]), 5.0, atol=1e-3)
