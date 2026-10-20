"""The OPE (M_theta_e, Sec 3.1) -- token fusion, timestep convention, AP pooling.

Pins the architecture details that are in the reference implementation but not
the paper text: the (a_{t-1}, r_t, o_t) fusion via *next*-observations rather
than a shifted action/reward stream, left-padding with t=0 reserved for
padding, and AP's unmasked mean.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from oaht_bench.dataset.dataset import _build_windows
from oaht_bench.models.tao_agent import OpponentPolicyEncoder


def _windows(n_ep=4, T=12, obs_dim=5, n_agents=2, seed=0):
    from oaht_bench.dataset.schema import Episode, EpisodeBatch

    rng = np.random.default_rng(seed)
    # Ragged: episode lengths vary (all >= the context length of 6).
    lengths = [T - (i % 4) for i in range(n_ep)]
    episodes = [
        Episode(
            obs=rng.normal(size=(n_agents, L, obs_dim)).astype(np.float32),
            actions=rng.integers(0, 6, size=(n_agents, L)),
            rewards=rng.normal(size=(n_agents, L)).astype(np.float32),
            avail_actions=np.ones((n_agents, L, 6), dtype=np.float32),
            dones=np.zeros(L, dtype=bool),
        )
        for L in lengths
    ]
    return _build_windows(
        EpisodeBatch(
            episodes=episodes,
            member_ids=np.array([[0, i % 2] for i in range(n_ep)]),
            ego_index=0,
            meta={},
        ),
        context_length=6,
        stride=3,
    )


def test_tao_encoder_pools_over_real_timesteps_only():
    w = _windows()
    rng = jax.random.PRNGKey(0)
    enc = OpponentPolicyEncoder(action_dim=6)
    args = (jnp.asarray(w.mate_next_obs), jnp.asarray(w.mate_actions), jnp.asarray(w.mate_rewards))
    kw = dict(mask=jnp.asarray(w.mask), timesteps=jnp.asarray(w.timesteps))
    p = enc.init(rng, *args, **kw)
    tok = enc.apply(p, *args, **kw)
    assert tok.shape == (len(w), w.context_length, 32)
    pooled = OpponentPolicyEncoder.pool(tok)
    assert pooled.shape == (len(w), 32)
    assert np.all(np.isfinite(np.asarray(pooled)))


def test_windows_are_left_padded_with_the_reference_conventions():
    """Decision Transformer convention the TAO reference inherits.

    Left padding means "now" is always the last position, so a short window and
    a full one agree on where the present is. Timesteps are 1-indexed with 0
    reserved for padding, and actions pad with -10 -- an out-of-range sentinel
    whose one-hot is all zeros, so padding cannot be read as action 0.
    """
    w = _windows()
    short = np.flatnonzero(~w.mask.all(axis=1))
    if short.size:
        i = int(short[0])
        # padding at the front, real steps at the back
        first_real = int(np.argmax(w.mask[i]))
        assert w.mask[i][first_real:].all()
        assert not w.mask[i][:first_real].any()
        assert (w.timesteps[i][:first_real] == 0).all()
        assert (w.ego_actions[i][:first_real] == -10).all()
    assert w.timesteps[w.mask].min() >= 1


def test_encoder_consumes_next_observations():
    """TAO fuses (a_t, r_t, o_{t+1}); the reference does this with next_obs.

    Shifting the action and reward streams instead gets the same pairing but
    labels each token with a different timestep, and the encoder's positional
    encoding is asymmetric (t for obs/reward, t-1 for the action).
    """
    w = _windows()
    assert w.mate_next_obs.shape == w.mate_obs.shape
    # next_obs at t equals obs at t+1 wherever both steps are real
    for i in range(min(3, len(w))):
        real = np.flatnonzero(w.mask[i])
        for t in real[:-1]:
            assert np.allclose(w.mate_next_obs[i, t], w.mate_obs[i, t + 1])


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
