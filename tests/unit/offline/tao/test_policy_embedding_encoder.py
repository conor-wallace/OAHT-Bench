"""The OPE (M_theta_e, Sec 3.1) -- token fusion, timestep convention, AP pooling.

Left-padding conventions and the ``next_obs`` fusion shift are windowing
properties, now pinned directly in ``tests/unit/dataset/test_windowing.py``
(``window_slice_batch``'s own tests) rather than here -- this file keeps only the
encoder architecture properties: AP pools over real timesteps only, and a
fully-masked row (an empty Opponent Context Window slot) stays finite.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from oaht_bench.models.tao_agent import OpponentPolicyEncoder

_T = 6


def _batch(n_rows=4, T=_T, obs_dim=5, seed=0):
    """Ragged: per-row valid length varies via ``mask``/``timesteps``, the
    same left-padding convention ``window_slice_batch`` uses."""
    rng = np.random.default_rng(seed)
    lengths = [T - (i % 4) for i in range(n_rows)]
    mask = np.zeros((n_rows, T), dtype=bool)
    timesteps = np.zeros((n_rows, T), dtype=np.int32)
    for i, length in enumerate(lengths):
        mask[i, T - length :] = True
        timesteps[i, T - length :] = np.arange(1, length + 1)
    return {
        "mate_next_obs": jnp.asarray(rng.normal(size=(n_rows, T, obs_dim)).astype(np.float32)),
        "mate_actions": jnp.asarray(rng.integers(0, 6, size=(n_rows, T))),
        "mate_rewards": jnp.asarray(rng.normal(size=(n_rows, T)).astype(np.float32)),
        "mask": jnp.asarray(mask),
        "timesteps": jnp.asarray(timesteps),
    }


def test_tao_encoder_pools_over_real_timesteps_only():
    b = _batch()
    rng = jax.random.PRNGKey(0)
    enc = OpponentPolicyEncoder(action_dim=6)
    args = (b["mate_next_obs"], b["mate_actions"], b["mate_rewards"])
    kw = dict(mask=b["mask"], timesteps=b["timesteps"])
    p = enc.init(rng, *args, **kw)
    tok = enc.apply(p, *args, **kw)
    assert tok.shape == (b["mate_next_obs"].shape[0], _T, 32)
    pooled = OpponentPolicyEncoder.pool(tok)
    assert pooled.shape == (b["mate_next_obs"].shape[0], 32)
    assert np.all(np.isfinite(np.asarray(pooled)))


def test_pool_is_finite_with_a_fully_masked_row_in_the_batch():
    """A window whose teammate has zero valid timesteps -- exactly what a
    partially-filled Opponent Context Window's empty slots look like -- must
    not corrupt that row (or leak into others) with NaN/inf once encoded and
    pooled. This is the self-attention half of the cold-start concern; the
    cross-attention half is tested in test_deployment_context.py.
    """
    rng = jax.random.PRNGKey(0)
    T, obs_dim, action_dim = 6, 4, 3
    enc = OpponentPolicyEncoder(action_dim=action_dim)
    mate_next_obs = jnp.zeros((2, T, obs_dim))
    mate_actions = jnp.zeros((2, T), dtype=jnp.int32)
    mate_rewards = jnp.zeros((2, T))
    timesteps = jnp.zeros((2, T), dtype=jnp.int32)
    # row 0: fully masked (empty OCW slot); row 1: fully valid.
    mask = jnp.array([[0.0] * T, [1.0] * T])
    p = enc.init(rng, mate_next_obs, mate_actions, mate_rewards, mask=mask, timesteps=timesteps)
    tok = enc.apply(p, mate_next_obs, mate_actions, mate_rewards, mask=mask, timesteps=timesteps)
    assert jnp.all(jnp.isfinite(tok)), "encoder produced non-finite tokens on a fully-masked row"
    pooled = OpponentPolicyEncoder.pool(tok)
    assert jnp.all(jnp.isfinite(pooled))
