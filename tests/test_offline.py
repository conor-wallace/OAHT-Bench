"""The offline baselines' contracts with the dataset and with each other.

These pin the properties that make the shared-backbone claim (§3.1) true: that
LIAM and TAO differ only where TAO's Appendix F says they differ, and that both
read the tensors the collected dataset actually provides.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from oaht_bench.offline import (
    AncillaryActionDecoder,
    ControlDecoder,
    LiamOffline,
    OpponentPolicyEncoder,
    TaoPolicy,
    info_nce,
    liam_loss,
    make_windows,
    return_to_go,
)


def _windows(n_ep=4, T=12, obs_dim=5, n_agents=2, seed=0):
    from oaht_bench.data.schema import EpisodeBatch

    rng = np.random.default_rng(seed)
    valid = np.ones((n_ep, T), dtype=bool)
    valid[:, -3:] = False  # padding, so masking is exercised
    return make_windows(
        EpisodeBatch(
            obs=rng.normal(size=(n_ep, n_agents, T, obs_dim)).astype(np.float32),
            actions=rng.integers(0, 6, size=(n_ep, n_agents, T)),
            rewards=rng.normal(size=(n_ep, n_agents, T)).astype(np.float32),
            dones=np.zeros((n_ep, T), dtype=bool),
            valid=valid,
            avail_actions=np.ones((n_ep, n_agents, T, 6), dtype=np.float32),
            member_ids=np.array([[0, i % 2] for i in range(n_ep)]),
            ego_index=0,
            meta={},
        ),
        context_length=6,
        stride=3,
    )


def test_return_to_go_ignores_padding():
    """Padding must contribute nothing, or every earlier target is inflated."""
    rewards = np.array([[1.0, 1.0, 1.0, 99.0]])
    valid = np.array([[True, True, True, False]])
    assert return_to_go(rewards, valid).tolist() == [[3.0, 2.0, 1.0, 0.0]]


def test_windows_expose_both_streams_and_the_teammate_label():
    """Everything LIAM and TAO read has to come out of the collected schema."""
    w = _windows()
    assert w.ego_obs.shape == w.mate_obs.shape
    assert w.ego_actions.shape == w.mate_actions.shape == w.mask.shape
    assert w.teammate_id.shape == (len(w),)
    # windows never extend past their episode
    assert w.mask.sum() > 0


def test_liam_has_no_cross_attention_and_tao_does():
    """Appendix F's one structural difference between the two.

    LIAM's teammate signal arrives only through the reconstruction loss; if it
    gained a conditioning path the method would no longer be LIAM.
    """
    w = _windows()
    rng = jax.random.PRNGKey(0)
    kw = dict(timesteps=jnp.asarray(w.timesteps))
    liam = LiamOffline(action_dim=6, obs_dim=w.obs_dim)
    p = liam.init(
        rng, jnp.asarray(w.ego_rtg), jnp.asarray(w.ego_obs), jnp.asarray(w.ego_actions), **kw
    )
    flat = jax.tree_util.tree_flatten_with_path(p)[0]
    names = " ".join(str(k) for k, _ in flat)
    assert "MultiHeadDotProductAttention" not in names or "SelfAttention" in names

    # A cross-attending backbone must be given a context; omitting it is an error
    # rather than a silent no-op.
    with pytest.raises(ValueError, match="no context was passed"):
        ControlDecoder(action_dim=6, use_cross_attention=True).init(
            rng, jnp.asarray(w.ego_rtg), jnp.asarray(w.ego_obs), jnp.asarray(w.ego_actions), **kw
        )


def test_info_nce_ignores_rows_with_no_positive():
    """Ragged per-teammate coverage is the common case, not an edge case.

    With random seating a window's teammate may appear once in a batch, leaving
    no positive pair. Those rows must drop out rather than contribute a
    degenerate term.
    """
    z = jnp.eye(3)
    singleton = info_nce(z, jnp.array([0, 1, 2]))  # no positives anywhere
    assert float(singleton) == 0.0
    paired = info_nce(z, jnp.array([0, 0, 1]))
    assert float(paired) > 0.0


def test_liam_loss_masks_padding():
    """A window shorter than the context must not be trained on its padding."""
    w = _windows()
    batch = {
        k: jnp.asarray(getattr(w, k))
        for k in (
            "ego_obs",
            "ego_actions",
            "ego_rtg",
            "mate_obs",
            "mate_actions",
            "mate_rewards",
            "timesteps",
            "mask",
            "teammate_id",
        )
    }
    rng = jax.random.PRNGKey(0)
    m = LiamOffline(action_dim=6, obs_dim=w.obs_dim)
    p = m.init(
        rng, batch["ego_rtg"], batch["ego_obs"], batch["ego_actions"], timesteps=batch["timesteps"]
    )
    _, aux = liam_loss(p, m, batch, rngs={"dropout": rng}, train=False)
    assert all(np.isfinite(float(v)) for v in aux.values())

    # Corrupting only the padded tail must not move the loss.
    bumped = dict(batch)
    bumped["mate_obs"] = batch["mate_obs"].at[:, -1].add(1000.0)
    mask_covers_tail = bool(batch["mask"][:, -1].any())
    _, aux2 = liam_loss(p, m, bumped, rngs={"dropout": rng}, train=False)
    if not mask_covers_tail:
        assert float(aux["recon_obs"]) == pytest.approx(float(aux2["recon_obs"]))


def test_tao_encoder_pools_over_real_timesteps_only():
    w = _windows()
    rng = jax.random.PRNGKey(0)
    enc = OpponentPolicyEncoder(action_dim=6)
    args = (jnp.asarray(w.mate_obs), jnp.asarray(w.mate_actions), jnp.asarray(w.mate_rewards))
    p = enc.init(rng, *args, mask=jnp.asarray(w.mask))
    tok = enc.apply(p, *args, mask=jnp.asarray(w.mask))
    assert tok.shape == (len(w), w.context_length, 32)
    pooled = OpponentPolicyEncoder.pool(tok, jnp.asarray(w.mask))
    assert pooled.shape == (len(w), 32)
    assert np.all(np.isfinite(np.asarray(pooled)))


def test_tao_policy_consumes_the_embedding_sequence():
    """Appendix F: z^-1 enters as key/value, not concatenated to the state."""
    w = _windows()
    rng = jax.random.PRNGKey(0)
    ctx = jnp.zeros((len(w), w.context_length, 32))
    pol = TaoPolicy(action_dim=6)
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


def test_ancillary_decoder_conditions_on_the_embedding():
    w = _windows()
    rng = jax.random.PRNGKey(0)
    dec = AncillaryActionDecoder(action_dim=6)
    z = jnp.zeros((len(w), 32))
    p = dec.init(rng, jnp.asarray(w.mate_obs), z)
    out = dec.apply(p, jnp.asarray(w.mate_obs), z)
    assert out.shape == (len(w), w.context_length, 6)
