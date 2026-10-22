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

from oaht_bench.dataset.dataset import return_to_go
from oaht_bench.models.liam_agent import LiamDecoder, LiamEncoder, LiamNetwork
from oaht_bench.offline import GPT2Model, liam_reconstruction_loss

_OBS_DIM = 5
_T = 6
_ACTION_DIM = 6


def _batch(n_ep=4, T=_T, obs_dim=_OBS_DIM, action_dim=_ACTION_DIM, seed=0):
    """A window batch dict of the shape every baseline's ``_sample_batch``
    produces -- built directly rather than through
    :func:`~oaht_bench.dataset.windowing.window_slice`, since these tests
    check model/loss properties, not windowing itself (see
    ``tests/unit/dataset/test_windowing.py`` for that).

    Ragged: per-row valid length varies via ``mask``/``timesteps``, the same
    left-padding convention ``window_slice`` uses.
    """
    rng = np.random.default_rng(seed)
    lengths = [T - (i % 4) for i in range(n_ep)]
    mask = np.zeros((n_ep, T), dtype=bool)
    timesteps = np.zeros((n_ep, T), dtype=np.int32)
    for i, length in enumerate(lengths):
        mask[i, T - length :] = True
        timesteps[i, T - length :] = np.arange(1, length + 1)
    return {
        "ego_obs": rng.normal(size=(n_ep, T, obs_dim)).astype(np.float32),
        "mate_obs": rng.normal(size=(n_ep, T, obs_dim)).astype(np.float32),
        "ego_actions": rng.integers(0, action_dim, size=(n_ep, T)),
        "mate_actions": rng.integers(0, action_dim, size=(n_ep, T)),
        "ego_rtg": rng.normal(size=(n_ep, T)).astype(np.float32),
        "ego_avail": np.ones((n_ep, T, action_dim), dtype=np.float32),
        "mate_avail": np.ones((n_ep, T, action_dim), dtype=np.float32),
        "timesteps": timesteps,
        "mask": mask,
    }


def test_return_to_go_ignores_padding():
    """Padding must contribute nothing, or every earlier target is inflated."""
    rewards = np.array([[1.0, 1.0, 1.0, 99.0]])
    valid = np.array([[True, True, True, False]])
    assert return_to_go(rewards, valid).tolist() == [[3.0, 2.0, 1.0, 0.0]]


def test_liam_conditions_by_concatenation_and_tao_by_cross_attention():
    """The conditioning mode is what separates the two, not its presence.

    An earlier version of this test asserted LIAM had *no* conditioning path,
    following TAO's Appendix F sketch. LIAM does have one: the original
    concatenates the embedding to the observation (``liam_agent.py:536``). The
    distinction is cross-attention versus concatenation.
    """
    b = _batch()
    rng = jax.random.PRNGKey(0)
    kw = dict(timesteps=jnp.asarray(b["timesteps"]), mask=jnp.asarray(b["mask"]))
    args = (jnp.asarray(b["ego_rtg"]), jnp.asarray(b["ego_obs"]), jnp.asarray(b["ego_actions"]))

    # LIAM widens the observation embedding by hidden_dim -- the concatenation.
    z = jnp.zeros((b["ego_obs"].shape[0], _T, 32))
    liam = LiamNetwork(action_dim=_ACTION_DIM, hidden_dim=32)
    lp = liam.init(rng, *args, embedding=z, **kw)
    obs_kernel = jax.tree_util.tree_leaves_with_path(lp)
    widened = [v.shape for k, v in obs_kernel if v.ndim == 2 and v.shape[0] == _OBS_DIM + 32]
    assert widened, "LIAM's observation layer should take obs_dim + hidden_dim inputs"

    # A cross-attending backbone must be given a context; omitting it is an error
    # rather than a silent no-op.
    with pytest.raises(ValueError, match="no context was passed"):
        GPT2Model(action_dim=_ACTION_DIM, use_cross_attention=True).init(rng, *args, **kw)


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
    batch = {k: jnp.asarray(v) for k, v in _batch().items()}
    rng = jax.random.PRNGKey(0)
    enc = LiamEncoder(action_dim=_ACTION_DIM)
    dec = LiamDecoder(obs_dim=_OBS_DIM, action_dim=_ACTION_DIM)
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

    batch = {k: jnp.asarray(v) for k, v in _batch().items()}
    rng = jax.random.PRNGKey(0)
    enc = LiamEncoder(action_dim=_ACTION_DIM)
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
    pol = LiamNetwork(action_dim=_ACTION_DIM)
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

    from oaht_bench.dataset import windowing
    from oaht_bench.models.return_conditioned_agent import ReturnConditionedAgent
    from oaht_bench.offline import liam as liam_mod
    from oaht_bench.offline import tao as tao_mod

    # windows carry the masks for both seats
    b = _batch()
    assert b["ego_avail"].shape == (b["ego_obs"].shape[0], _T, _ACTION_DIM)
    assert b["mate_avail"].shape == b["ego_avail"].shape
    assert "avail_actions" in inspect.getsource(windowing)

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

    batch = {k: jnp.asarray(v) for k, v in _batch().items()}
    rng = jax.random.PRNGKey(0)

    backbone = OmisBackbone(action_dim=_ACTION_DIM, hidden_dim=32)
    heads = OmisHeads(action_dim=_ACTION_DIM, hidden_dim=32)

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
