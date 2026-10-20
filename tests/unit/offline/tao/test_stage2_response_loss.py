"""Stage 2, Opponent-aware Response Policy Training -- Eq. 5.

``L_res`` is cross-entropy over the ego's near-optimal action, predicted by the
ICD (``TaoNetwork``) cross-attending to the GetOffD context, with the encoder
frozen by default (stage 1's product must survive into stage 2 -- see
``offline/tao.py``'s module docstring on why TAO w/o PEL only means anything if
the encoder isn't silently retrained).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from oaht_bench.dataset.dataset import TeammateIndex, _build_windows
from oaht_bench.dataset.sampler import sample_stage2
from oaht_bench.models.tao_agent import OpponentPolicyEncoder, TaoNetwork
from oaht_bench.offline import tao_policy_loss


def _windows(n_ep=4, T=12, obs_dim=5, n_agents=2, seed=0):
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


def test_tao_policy_consumes_the_embedding_sequence():
    """Appendix F: z^-1 enters as key/value, not concatenated to the state."""
    w = _windows()
    rng = jax.random.PRNGKey(0)
    ctx = jnp.zeros((len(w), w.context_length, 32))
    pol = TaoNetwork(action_dim=6)
    p = pol.init(
        rng,
        jnp.asarray(w.ego_rtg),
        jnp.asarray(w.ego_obs),
        jnp.asarray(w.ego_actions),
        timesteps=jnp.asarray(w.timesteps),
        context=ctx,
    )
    logits = pol.apply(
        p,
        jnp.asarray(w.ego_rtg),
        jnp.asarray(w.ego_obs),
        jnp.asarray(w.ego_actions),
        timesteps=jnp.asarray(w.timesteps),
        context=ctx,
    )
    assert logits.shape == (len(w), w.context_length, 6)


def _tao_setup(w):
    """A stage-2 batch built the way training builds it, through the sampler.

    The encoder context is GetOffD -- C fragments of the same teammate,
    concatenated -- so it is longer than the decoder window and cannot be faked
    by reusing the window's own teammate stream.
    """
    rng = jax.random.PRNGKey(0)
    idx = TeammateIndex.build(w)
    raw = sample_stage2(w, idx, np.random.default_rng(0), batch_size=6, context_trajectories=3)
    batch = {k: jnp.asarray(v) for k, v in raw.items()}

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
    w = _windows()
    rng, batch, enc, pol, params = _tao_setup(w)

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
    w = _windows()
    rng, batch, enc, pol, params = _tao_setup(w)
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
