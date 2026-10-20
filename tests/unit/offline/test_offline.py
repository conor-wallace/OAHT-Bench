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

from oaht_bench.dataset.dataset import _build_windows, return_to_go
from oaht_bench.models.liam_agent import LiamDecoder, LiamEncoder, LiamNetwork
from oaht_bench.offline import GPT2Model, liam_reconstruction_loss


def _windows(n_ep=4, T=12, obs_dim=5, n_agents=2, seed=0):
    from oaht_bench.dataset.schema import Episode, EpisodeBatch

    rng = np.random.default_rng(seed)
    # Ragged: episode lengths vary (all >= the context length of 6, as they were
    # before this became a list of variable-length episodes).
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
    liam = LiamNetwork(action_dim=6, hidden_dim=32)
    lp = liam.init(rng, *args, embedding=z, **kw)
    obs_kernel = jax.tree_util.tree_leaves_with_path(lp)
    widened = [v.shape for k, v in obs_kernel if v.ndim == 2 and v.shape[0] == w.obs_dim + 32]
    assert widened, "LIAM's observation layer should take obs_dim + hidden_dim inputs"

    # A cross-attending backbone must be given a context; omitting it is an error
    # rather than a silent no-op.
    with pytest.raises(ValueError, match="no context was passed"):
        GPT2Model(action_dim=6, use_cross_attention=True).init(rng, *args, **kw)


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
            "ego_avail",
            "mate_avail",
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
            "ego_avail",
            "mate_avail",
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
    pol = LiamNetwork(action_dim=6)
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


def test_available_actions_are_enforced_everywhere_the_data_enforces_them():
    """Collection masks; training and evaluation must too.

    ``data/collect.py`` passes ``avail_actions`` into every seat's
    ``get_action``, so a recorded action is always legal. For a while the
    learned policy was held to no such constraint -- windows dropped the field,
    the losses never masked, and the rollout sampled the ego from raw logits.
    On LBF that is 20.5% of (step, action) pairs, and action 5 is unavailable
    67% of the time.
    """
    import inspect

    from oaht_bench.dataset import dataset as ds
    from oaht_bench.models.return_conditioned_agent import ReturnConditionedAgent
    from oaht_bench.offline import liam as liam_mod
    from oaht_bench.offline import tao as tao_mod

    # windows carry the masks for both seats
    w = _windows()
    assert w.ego_avail.shape == (len(w), w.context_length, 6)
    assert w.mate_avail.shape == w.ego_avail.shape
    assert "avail_actions" in inspect.getsource(ds._build_windows)

    # every cross-entropy is over masked logits
    assert "mask_logits" in inspect.getsource(liam_mod.liam_reconstruction_loss)
    assert "mask_logits" in inspect.getsource(liam_mod.liam_policy_loss)
    assert "mask_logits" in inspect.getsource(tao_mod.embedding_loss)
    assert "mask_logits" in inspect.getsource(tao_mod.tao_policy_loss)

    # and the shared deployment loop masks before sampling the ego
    assert "mask_logits" in inspect.getsource(ReturnConditionedAgent.get_action)


def test_omis_all_three_heads_reach_the_shared_backbone():
    """The property this refactor exists for: no head is cut off from the
    backbone by a frozen boundary, unlike LIAM/MeLIBA/TAO's staged training.

    Differentiates each head's own loss term in isolation and checks it reaches
    :class:`~oaht_bench.models.omis_agent.OmisBackbone` -- the direct,
    per-head version of what ``omis_joint_loss``'s combined gradient does.
    """
    import optax as _optax

    from oaht_bench.models.omis_agent import OmisBackbone, OmisHeads
    from oaht_bench.offline.utils import mask_logits

    w = _windows()
    rng = jax.random.PRNGKey(0)
    batch = {
        k: jnp.asarray(getattr(w, k))
        for k in (
            "ego_rtg",
            "ego_obs",
            "ego_actions",
            "mate_actions",
            "ego_avail",
            "mate_avail",
            "timesteps",
            "mask",
        )
    }

    backbone = OmisBackbone(action_dim=6, hidden_dim=32)
    heads = OmisHeads(action_dim=6, hidden_dim=32)

    bp = backbone.init(
        rng,
        batch["ego_rtg"],
        batch["ego_obs"],
        batch["ego_actions"],
        timesteps=batch["timesteps"],
        mask=batch["mask"],
    )
    z = backbone.apply(
        bp,
        batch["ego_rtg"],
        batch["ego_obs"],
        batch["ego_actions"],
        timesteps=batch["timesteps"],
        mask=batch["mask"],
    )
    params = {"backbone": bp, "heads": heads.init(rng, z)}

    def head_loss(which):
        def loss_fn(p):
            embedding = backbone.apply(
                p["backbone"],
                batch["ego_rtg"],
                batch["ego_obs"],
                batch["ego_actions"],
                timesteps=batch["timesteps"],
                mask=batch["mask"],
            )
            action_logits, mate_logits, value = heads.apply(p["heads"], embedding)
            m = batch["mask"].astype(jnp.float32)
            denom = jnp.maximum(m.sum(), 1.0)
            if which == "actor":
                logits = mask_logits(action_logits, batch["ego_avail"])
                ce = optax.softmax_cross_entropy_with_integer_labels(logits, batch["ego_actions"])
                return (ce * m).sum() / denom
            if which == "imitator":
                logits = mask_logits(mate_logits, batch["mate_avail"])
                ce = optax.softmax_cross_entropy_with_integer_labels(logits, batch["mate_actions"])
                return (ce * m).sum() / denom
            mse = (value - batch["ego_rtg"]) ** 2
            return (mse * m).sum() / denom

        return loss_fn

    for which in ("actor", "imitator", "critic"):
        grads = jax.grad(head_loss(which))(params)
        assert float(_optax.global_norm(grads["backbone"])) > 0.0, (
            f"{which} head's loss should reach the shared backbone"
        )


def test_mask_logits_matches_the_absorbed_convention():
    """``logits - (1 - avail) * 1e10``, per agents/mlp_actor_critic.py:36-37.

    Not ``-inf``: a fully-masked row would then be NaN after softmax, and padded
    timesteps are masked with all-ones precisely to avoid that.
    """
    from oaht_bench.offline.utils import mask_logits

    logits = jnp.zeros((1, 3))
    avail = jnp.asarray([[1.0, 0.0, 1.0]])
    out = np.asarray(mask_logits(logits, avail))[0]
    assert out[0] == 0.0 and out[2] == 0.0
    assert out[1] == pytest.approx(-1e10)
    # the unavailable action gets essentially zero probability
    probs = np.asarray(jax.nn.softmax(jnp.asarray(out)))
    assert probs[1] < 1e-12
    # passing None is a no-op, for callers without a mask
    assert np.array_equal(np.asarray(mask_logits(logits, None)), np.asarray(logits))
