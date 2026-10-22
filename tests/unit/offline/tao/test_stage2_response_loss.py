"""Stage 2, Opponent-aware Response Policy Training -- Eq. 5.

``L_res`` is cross-entropy over the ego's near-optimal action, predicted by the
ICD (``TaoNetwork``) cross-attending to the GetOffD context, with the encoder
frozen by default (stage 1's product must survive into stage 2 -- see
``offline/tao.py``'s module docstring on why TAO w/o PEL only means anything if
the encoder isn't silently retrained).
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from oaht_bench.dataset.windowing import enumerate_windows, window_slice
from oaht_bench.models.tao_agent import OpponentPolicyEncoder, TaoNetwork
from oaht_bench.offline import tao_policy_loss
from oaht_bench.offline.tao import _CONTEXT_FIELDS, _DECODER_FIELDS, _get_off_d_context

_T = 6


def _dataset(n_ep=4, T=12, obs_dim=5, n_agents=2, seed=0):
    from oaht_bench.dataset.schema import Episode, EpisodeBatch

    rng = np.random.default_rng(seed)
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
    batch = EpisodeBatch(
        episodes=episodes,
        member_ids=np.array([[0, i % 2] for i in range(n_ep)]),
        ego_index=0,
        meta={},
    )
    return SimpleNamespace(batch=batch, mate=1)


def test_tao_policy_consumes_the_embedding_sequence():
    """Appendix F: z^-1 enters as key/value, not concatenated to the state."""
    n_rows, obs_dim, action_dim = 4, 5, 6
    rng_np = np.random.default_rng(0)
    ego_rtg = jnp.asarray(rng_np.normal(size=(n_rows, _T)).astype(np.float32))
    ego_obs = jnp.asarray(rng_np.normal(size=(n_rows, _T, obs_dim)).astype(np.float32))
    ego_actions = jnp.asarray(rng_np.integers(0, action_dim, size=(n_rows, _T)))
    timesteps = jnp.asarray(np.tile(np.arange(1, _T + 1), (n_rows, 1)))

    rng = jax.random.PRNGKey(0)
    ctx = jnp.zeros((n_rows, _T, 32))
    pol = TaoNetwork(action_dim=action_dim)
    p = pol.init(rng, ego_rtg, ego_obs, ego_actions, timesteps=timesteps, context=ctx)
    logits = pol.apply(p, ego_rtg, ego_obs, ego_actions, timesteps=timesteps, context=ctx)
    assert logits.shape == (n_rows, _T, action_dim)


def _stage2_batch(ds, *, batch_size=6, context_trajectories=3, h=_T, seed=0):
    """The same decoder-window + GetOffD-context batch
    ``TaoTrainer._stage2_batch`` builds, standalone.

    The encoder context is GetOffD -- C fragments of the same teammate,
    concatenated -- so it is longer than the decoder window and cannot be faked
    by reusing the window's own teammate stream.
    """
    rng = np.random.default_rng(seed)
    episode_ids, ts = enumerate_windows(ds.batch, h=h, stride=h)
    rows = rng.choice(len(episode_ids), size=batch_size, replace=len(episode_ids) < batch_size)

    decoder = [
        window_slice(ds.batch, int(episode_ids[i]), int(ts[i]), h, _DECODER_FIELDS, mate=ds.mate)
        for i in rows
    ]
    teammate_ids = [int(ds.batch.member_ids[episode_ids[i], ds.mate]) for i in rows]
    contexts = [
        _get_off_d_context(
            ds, teammate_ids[j], int(episode_ids[rows[j]]), h, context_trajectories, rng
        )
        for j in range(len(rows))
    ]

    batch = {k: np.stack([d[k] for d in decoder]) for k in _DECODER_FIELDS}
    batch["teammate_id"] = np.asarray(teammate_ids, dtype=np.int32)
    for key in _CONTEXT_FIELDS:
        batch[f"context_{key}"] = np.stack([ctx[key] for ctx in contexts])
    return {k: jnp.asarray(v) for k, v in batch.items()}


def _tao_setup(ds):
    rng = jax.random.PRNGKey(0)
    batch = _stage2_batch(ds, batch_size=6, context_trajectories=3)

    enc = OpponentPolicyEncoder(action_dim=6)
    ep = enc.init(
        rng,
        batch["context_mate_next_obs"],
        batch["context_mate_actions"],
        batch["context_mate_rewards"],
        mask=batch["context_mask"],
        timesteps=batch["context_timesteps"],
    )
    tok = enc.apply(
        ep,
        batch["context_mate_next_obs"],
        batch["context_mate_actions"],
        batch["context_mate_rewards"],
        mask=batch["context_mask"],
        timesteps=batch["context_timesteps"],
    )
    pol = TaoNetwork(action_dim=6)
    pp = pol.init(
        rng,
        batch["ego_rtg"],
        batch["ego_obs"],
        batch["ego_actions"],
        timesteps=batch["timesteps"],
        context=tok,
        mask=batch["mask"],
        context_mask=batch["context_mask"],
    )
    return rng, batch, enc, pol, {"policy": pp, "encoder": ep}


def test_tao_stage_two_freezes_the_encoder_by_default():
    """Stage 1's embedding must survive stage 2, or TAO w/o PEL is meaningless.

    TAO w/o PEL is one of the paper's own baselines, and PEL is stage 1. If
    stage 2 retrained the encoder from scratch -- which the released code does,
    never loading stage 1's weights -- the two would be the same model. Freezing
    also protects what stage 1 builds: an embedding carrying policy identity,
    which stage 3 relies on to place an unseen teammate.
    """
    ds = _dataset()
    rng, batch, enc, pol, params = _tao_setup(ds)

    default = jax.grad(tao_policy_loss, has_aux=True)(
        params, pol, enc, batch, rngs={"dropout": rng}
    )[0]
    joint = jax.grad(tao_policy_loss, has_aux=True)(
        params, pol, enc, batch, freeze_encoder=False, rngs={"dropout": rng}
    )[0]

    # default: no gradient reaches the encoder at all
    assert float(optax.global_norm(default["encoder"])) == 0.0
    # opting out reproduces the released code's joint training
    assert float(optax.global_norm(joint["encoder"])) > 0.0
    # the policy is trained either way
    assert float(optax.global_norm(default["policy"])) > 0.0


def test_tao_stage_two_loss_is_a_masked_mean_over_valid_steps():
    """The reference indexes by the mask then calls CrossEntropyLoss.

    Its default reduction is a mean, so padding must not enter the denominator.
    """
    ds = _dataset()
    rng, batch, enc, pol, params = _tao_setup(ds)
    _, aux = tao_policy_loss(params, pol, enc, batch, rngs={"dropout": rng}, train=False)

    tok = enc.apply(
        params["encoder"],
        batch["context_mate_next_obs"],
        batch["context_mate_actions"],
        batch["context_mate_rewards"],
        mask=batch["context_mask"],
        timesteps=batch["context_timesteps"],
    )
    logits = pol.apply(
        params["policy"],
        batch["ego_rtg"],
        batch["ego_obs"],
        batch["ego_actions"],
        timesteps=batch["timesteps"],
        context=tok,
        mask=batch["mask"],
        context_mask=batch["context_mask"],
    )
    m = np.asarray(batch["mask"])
    ce = np.asarray(optax.softmax_cross_entropy_with_integer_labels(logits, batch["ego_actions"]))
    assert float(aux["bc"]) == pytest.approx(ce[m].mean(), rel=1e-4)
