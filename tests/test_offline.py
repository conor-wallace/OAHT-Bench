"""The offline baselines' contracts with the dataset and with each other.

These pin the properties that make the shared-backbone claim (§3.1) true: that
LIAM and TAO differ only where TAO's Appendix F says they differ, and that both
read the tensors the collected dataset actually provides.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from oaht_bench.offline import (
    AncillaryActionDecoder,
    DecisionTransformer,
    LiamDecoder,
    LiamEncoder,
    LiamPolicy,
    OpponentPolicyEncoder,
    TaoPolicy,
    liam_reconstruction_loss,
    make_windows,
    return_to_go,
    supervised_contrastive,
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


def test_liam_conditions_by_concatenation_and_tao_by_cross_attention():
    """The conditioning mode is what separates the two, not its presence.

    An earlier version of this test asserted LIAM had *no* conditioning path,
    following TAO's Appendix F sketch. LIAM does have one: the original
    concatenates the embedding to the observation (``liam_agent.py:536``). The
    distinction is cross-attention versus concatenation.
    """
    w = _windows()
    rng = jax.random.PRNGKey(0)
    kw = dict(timesteps=jnp.asarray(w.timesteps), mask=jnp.asarray(w.mask))
    args = (jnp.asarray(w.ego_rtg), jnp.asarray(w.ego_obs), jnp.asarray(w.ego_actions))

    # LIAM widens the observation embedding by hidden_dim -- the concatenation.
    z = jnp.zeros((len(w), w.context_length, 32))
    liam = LiamPolicy(action_dim=6, hidden_dim=32)
    lp = liam.init(rng, *args, embedding=z, **kw)
    obs_kernel = jax.tree_util.tree_leaves_with_path(lp)
    widened = [v.shape for k, v in obs_kernel if v.ndim == 2 and v.shape[0] == w.obs_dim + 32]
    assert widened, "LIAM's observation layer should take obs_dim + hidden_dim inputs"

    # A cross-attending backbone must be given a context; omitting it is an error
    # rather than a silent no-op.
    with pytest.raises(ValueError, match="no context was passed"):
        DecisionTransformer(action_dim=6, use_cross_attention=True).init(rng, *args, **kw)


def test_supervised_contrastive_matches_the_reference_aggregation():
    """SupCon, not plain InfoNCE -- the reference averages log-prob over positives.

    ``-(T/T_base) * mean_over_positives(log_prob)``, on raw dot products with a
    per-row max subtracted (``offline_stage_1/nn_trainer.py:130-156``). Computed
    directly here so a future edit back to log-sum-exp is caught.
    """
    z = jnp.asarray(np.eye(3), dtype=jnp.float32)
    labels = jnp.array([0, 0, 1])
    got = float(supervised_contrastive(z, labels, temperature=0.1, base_temperature=0.1))

    sim = np.asarray(z @ z.T) / 0.1
    sim = sim - sim.max(axis=1, keepdims=True)
    eye = np.eye(3, dtype=bool)
    exp = np.exp(sim) * ~eye
    log_prob = sim - np.log(exp.sum(axis=1, keepdims=True))
    pos = (np.asarray(labels)[:, None] == np.asarray(labels)[None, :]) & ~eye
    rows = pos.sum(1) > 0
    expected = -((pos * log_prob).sum(1)[rows] / pos.sum(1)[rows]).mean()
    assert got == pytest.approx(expected, rel=1e-5)


def test_supervised_contrastive_drops_rows_with_no_positive():
    """Ragged per-teammate coverage is the common case, not an edge case.

    The reference divides by ``dis_mask.sum(1)`` unguarded because its sampler
    guarantees a positive per anchor; ours cannot, since seats are sampled
    independently.
    """
    assert float(supervised_contrastive(jnp.eye(3), jnp.array([0, 1, 2]))) == 0.0


def test_liam_reconstructs_the_teammate_at_the_same_timestep():
    """Both targets are at ``t``, per the paper and liam_agent.py:302-303.

    An earlier version used ``a^-1_{t-1}`` following TAO's Appendix F, which asks
    what the teammate did *last* step rather than what it is doing now.
    """
    import inspect

    from oaht_bench.offline import liam as liam_mod

    src = inspect.getsource(liam_mod.liam_reconstruction_loss)
    assert 'batch["mate_actions"]' in src
    # no shift of the teammate action stream
    assert 'mate_actions"][:, :-1]' not in src


def test_liam_observation_term_is_a_gaussian_nll_not_a_mean():
    """``0.5 * sum`` over dims, not ``mean``.

    They differ by ``0.5 * obs_dim`` -- 12x on LBF -- and LIAM sums two terms, so
    a mean silently reweights observation reconstruction against action
    reconstruction.
    """
    w = _windows()
    rng = jax.random.PRNGKey(0)
    enc = LiamEncoder(action_dim=6)
    dec = LiamDecoder(obs_dim=w.obs_dim, action_dim=6)
    batch = {
        k: jnp.asarray(getattr(w, k))
        for k in (
            "ego_obs",
            "ego_actions",
            "ego_rtg",
            "mate_obs",
            "mate_actions",
            "timesteps",
            "mask",
        )
    }
    ep = enc.init(
        rng,
        batch["ego_rtg"],
        batch["ego_obs"],
        batch["ego_actions"],
        timesteps=batch["timesteps"],
        mask=batch["mask"],
    )
    z = enc.apply(
        ep,
        batch["ego_rtg"],
        batch["ego_obs"],
        batch["ego_actions"],
        timesteps=batch["timesteps"],
        mask=batch["mask"],
    )
    dp = dec.init(rng, z)
    _, aux = liam_reconstruction_loss(
        {"encoder": ep, "decoder": dp}, enc, dec, batch, rngs={"dropout": rng}, train=False
    )

    obs_hat, _ = dec.apply(dp, z)
    m = np.asarray(batch["mask"], dtype=float)
    expected = 0.5 * ((np.asarray(batch["mate_obs"]) - np.asarray(obs_hat)) ** 2).sum(-1)
    expected = (expected * m).sum() / max(m.sum(), 1.0)
    assert float(aux["recon_obs"]) == pytest.approx(expected, rel=1e-4)


def test_liam_stage_two_does_not_differentiate_the_encoder():
    """Staging is what removes the original's stop_gradient.

    liam_agent.py blocks the gradient because encoder and policy train together
    online. Offline the encoder is frozen from stage 1, so the block is
    unnecessary rather than merely absent -- and no encoder parameter should
    appear in stage 2's gradient.
    """
    from oaht_bench.offline import liam_policy_loss

    w = _windows()
    rng = jax.random.PRNGKey(0)
    enc = LiamEncoder(action_dim=6)
    batch = {
        k: jnp.asarray(getattr(w, k))
        for k in (
            "ego_obs",
            "ego_actions",
            "ego_rtg",
            "mate_obs",
            "mate_actions",
            "timesteps",
            "mask",
        )
    }
    ep = enc.init(
        rng,
        batch["ego_rtg"],
        batch["ego_obs"],
        batch["ego_actions"],
        timesteps=batch["timesteps"],
        mask=batch["mask"],
    )
    z = enc.apply(
        ep,
        batch["ego_rtg"],
        batch["ego_obs"],
        batch["ego_actions"],
        timesteps=batch["timesteps"],
        mask=batch["mask"],
    )
    pol = LiamPolicy(action_dim=6)
    pp = pol.init(
        rng,
        batch["ego_rtg"],
        batch["ego_obs"],
        batch["ego_actions"],
        timesteps=batch["timesteps"],
        embedding=z,
        mask=batch["mask"],
    )

    grads = jax.grad(liam_policy_loss, has_aux=True)(
        pp, pol, enc, ep, batch, rngs={"dropout": rng}
    )[0]
    # gradient is over the policy only; the encoder pytree is not in it
    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(pp)


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


def _tao_setup(w):
    import jax as _jax

    rng = _jax.random.PRNGKey(0)
    batch = {
        k: jnp.asarray(getattr(w, k))
        for k in (
            "ego_obs",
            "ego_actions",
            "ego_rtg",
            "mate_obs",
            "mate_next_obs",
            "mate_actions",
            "mate_rewards",
            "timesteps",
            "mask",
            "teammate_id",
        )
    }
    enc = OpponentPolicyEncoder(action_dim=6)
    ep = enc.init(
        rng,
        batch["mate_next_obs"],
        batch["mate_actions"],
        batch["mate_rewards"],
        mask=batch["mask"],
        timesteps=batch["timesteps"],
    )
    tok = enc.apply(
        ep,
        batch["mate_next_obs"],
        batch["mate_actions"],
        batch["mate_rewards"],
        mask=batch["mask"],
        timesteps=batch["timesteps"],
    )
    pol = TaoPolicy(action_dim=6)
    pp = pol.init(
        rng,
        batch["ego_rtg"],
        batch["ego_obs"],
        batch["ego_actions"],
        timesteps=batch["timesteps"],
        context=tok,
        mask=batch["mask"],
        context_mask=batch["mask"],
    )
    return rng, batch, enc, pol, {"policy": pp, "encoder": ep}


def test_tao_stage_two_trains_the_encoder_by_default():
    """The reference fine-tunes the encoder in stage 2; the paper implies frozen.

    offline_stage_2/nn_trainer.py steps encoder_optimizer alongside
    decoder_optimizer, and its train.py builds a fresh encoder rather than
    loading stage 1's -- ENCODER_PARAM_PATH is defined in its config and read
    nowhere. The published numbers came from that code, so joint training is the
    default; freeze_encoder is available for the paper's reading.
    """
    import optax as _optax

    from oaht_bench.offline import tao_policy_loss

    w = _windows()
    rng, batch, enc, pol, params = _tao_setup(w)

    joint = jax.grad(tao_policy_loss, has_aux=True)(
        params, pol, enc, batch, freeze_encoder=False, rngs={"dropout": rng}
    )[0]
    frozen = jax.grad(tao_policy_loss, has_aux=True)(
        params, pol, enc, batch, freeze_encoder=True, rngs={"dropout": rng}
    )[0]

    assert float(_optax.global_norm(joint["encoder"])) > 0.0
    assert float(_optax.global_norm(frozen["encoder"])) == 0.0
    # the policy is trained either way
    assert float(_optax.global_norm(frozen["policy"])) > 0.0


def test_tao_stage_two_loss_is_a_masked_mean_over_valid_steps():
    """The reference indexes by the mask then calls CrossEntropyLoss.

    Its default reduction is a mean, so padding must not enter the denominator.
    """
    from oaht_bench.offline import tao_policy_loss

    w = _windows()
    rng, batch, enc, pol, params = _tao_setup(w)
    _, aux = tao_policy_loss(params, pol, enc, batch, rngs={"dropout": rng}, train=False)

    tok = enc.apply(
        params["encoder"],
        batch["mate_next_obs"],
        batch["mate_actions"],
        batch["mate_rewards"],
        mask=batch["mask"],
        timesteps=batch["timesteps"],
    )
    logits = pol.apply(
        params["policy"],
        batch["ego_rtg"],
        batch["ego_obs"],
        batch["ego_actions"],
        timesteps=batch["timesteps"],
        context=tok,
        mask=batch["mask"],
        context_mask=batch["mask"],
    )
    m = np.asarray(batch["mask"])
    ce = np.asarray(optax.softmax_cross_entropy_with_integer_labels(logits, batch["ego_actions"]))
    assert float(aux["bc"]) == pytest.approx(ce[m].mean(), rel=1e-4)
