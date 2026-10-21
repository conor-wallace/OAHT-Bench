"""Eq. 6, deployment-stage in-context opponent adapting -- the two ways the
*context* fed to cross-attention at deployment can silently diverge from what
stage 2 trained it on.

``OpponentContextWindow``'s own buffer semantics (front-padding, capacity,
reset) are pinned in ``test_incontext_eval.py``, shared with OMIS. The two
tests here are TAO-specific: do the *network* and *encoder* survive what that
buffer can legitimately hand them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from oaht_bench.configs.job import OfflineTrainingConfig
from oaht_bench.configs.network import TaoNetworkConfig
from oaht_bench.dataset.dataset import Normalization
from oaht_bench.models.tao_agent import TaoAgent, TaoNetwork


def test_cross_attention_survives_an_all_zero_context_mask():
    """Episode 1 against a new/just-switched teammate: ``OpponentContextWindow.arrays()``
    returns an all-zero mask (pinned structurally in
    ``test_incontext_eval.py::test_empty_window_has_no_valid_context``). This is
    the other half -- whether the network's cross-attention produces finite
    logits when every context key is masked out, or an all-masked softmax row
    silently produces NaN that corrupts every action in that episode.
    """
    rng = jax.random.PRNGKey(0)
    B, T, C_T, hidden, action_dim, obs_dim = 2, 6, 15, 32, 6, 5
    pol = TaoNetwork(action_dim=action_dim, hidden_dim=hidden)
    ego_rtg = jnp.zeros((B, T))
    ego_obs = jax.random.normal(rng, (B, T, obs_dim))
    ego_actions = jnp.zeros((B, T), dtype=jnp.int32)
    timesteps = jnp.tile(jnp.arange(1, T + 1), (B, 1))
    mask = jnp.ones((B, T))
    context = jnp.zeros((B, C_T, hidden))
    context_mask = jnp.zeros((B, C_T))  # all-zero: cold-start OCW

    p = pol.init(
        rng,
        ego_rtg,
        ego_obs,
        ego_actions,
        timesteps=timesteps,
        context=context,
        mask=mask,
        context_mask=context_mask,
    )
    logits = pol.apply(
        p,
        ego_rtg,
        ego_obs,
        ego_actions,
        timesteps=timesteps,
        context=context,
        mask=mask,
        context_mask=context_mask,
    )
    assert jnp.all(jnp.isfinite(logits)), (
        "cross-attention over an all-masked context produced non-finite logits -- "
        "every cold-start episode's action distribution would be corrupted"
    )


def test_encode_ocw_normalizes_observations_the_same_way_training_does():
    """``encode_ocw`` must apply the agent's own ``self._obs_mean``/``self._obs_std``
    to the teammate stream, the same transform ``Dataset._obs_field`` bakes into
    every training window's ``mate_next_obs`` -- otherwise the encoder sees
    out-of-distribution input at deployment on every episode, not just cold-start
    ones. (Previously unfixed and pinned as xfail; fixed in models/tao_agent.py.)
    """
    rng = jax.random.PRNGKey(0)
    C, T, obs_dim, action_dim = 2, 5, 4, 3
    raw_obs = jax.random.normal(rng, (C, T, obs_dim))
    actions = jnp.zeros((C, T), dtype=jnp.int32)
    rewards = jnp.zeros((C, T))
    mask = jnp.ones((C, T))
    timesteps = jnp.tile(jnp.arange(1, T + 1), (C, 1))

    norm = Normalization(
        obs_mean=np.full(obs_dim, 5.0, np.float32),
        obs_std=np.full(obs_dim, 2.0, np.float32),
        rtg_scale=1.0,
    )
    cfg = OfflineTrainingConfig(
        network=TaoNetworkConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32)
    )
    agent = TaoAgent(cfg, normalization=norm)
    agent.build_model()

    enc_params = agent.encoder.init(rng, raw_obs, actions, rewards, mask=mask, timesteps=timesteps)
    params = {"stage2": {"encoder": enc_params}}

    # What training's encoder actually sees: the SAME trajectory, normalized the
    # way Dataset._obs_field normalizes every training window.
    normalized_obs = jnp.asarray(norm.apply_obs(np.asarray(raw_obs)))
    expected = agent.encoder.apply(
        enc_params, normalized_obs, actions, rewards, mask=mask, timesteps=timesteps, train=False
    )

    # What encode_ocw actually feeds it: the identical raw trajectory, unnormalized.
    got_context = agent.encode_ocw(params, raw_obs, actions, rewards, mask, timesteps)["context"]
    got = got_context.reshape(C, T, cfg.network.hidden_dim)

    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-5)
