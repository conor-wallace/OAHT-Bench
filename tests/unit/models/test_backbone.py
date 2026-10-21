"""The shared GPT2Model backbone -- the regression guarantee OMIS's D_step/D_epi
extension rests on, plus sanity checks that the new capabilities are actually wired in.

LIAM, MeLIBA, and TAO all build on this one module (see backbone.py's own
docstring). Adding OMIS's opponent-conditioning must not perturb any of them --
pinned here via parameter-structure and output-value equality against the
pre-extension behavior (the four new keyword args all at their off-defaults),
not just "it still runs".
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from oaht_bench.models.backbone import GPT2Model


def _inputs(B=3, T=5, action_dim=4, obs_dim=6, seed=0):
    rng = np.random.default_rng(seed)
    rtg = jnp.asarray(rng.normal(size=(B, T)), dtype=jnp.float32)
    obs = jnp.asarray(rng.normal(size=(B, T, obs_dim)), dtype=jnp.float32)
    actions = jnp.asarray(rng.integers(0, action_dim, size=(B, T)), dtype=jnp.int32)
    timesteps = jnp.asarray(np.tile(np.arange(T), (B, 1)), dtype=jnp.int32)
    mask = jnp.ones((B, T), dtype=bool).at[0, :2].set(False)
    return rtg, obs, actions, timesteps, mask


def _reference_forward(**model_kwargs):
    """The pre-extension computation, inlined: exactly what GPT2Model computed
    before D_step/D_epi existed (3 tokens/timestep, no prepended block,
    readout at a fixed index-1-of-3). Not imported from backbone.py, since the
    whole point is to catch backbone.py itself drifting from this.
    """
    import flax.linen as nn

    class Block(nn.Module):
        hidden_dim: int
        ff_dim: int
        use_cross_attention: bool
        dropout: float

        @nn.compact
        def __call__(self, x, *, context=None, causal_mask, cross_mask=None, train):
            attn = nn.SelfAttention(
                num_heads=1,
                qkv_features=self.hidden_dim,
                dropout_rate=self.dropout,
                deterministic=not train,
            )(x, mask=causal_mask)
            x = nn.LayerNorm()(x + nn.Dropout(self.dropout, deterministic=not train)(attn))
            if self.use_cross_attention:
                cross = nn.MultiHeadDotProductAttention(
                    num_heads=1,
                    qkv_features=self.hidden_dim,
                    dropout_rate=self.dropout,
                    deterministic=not train,
                )(x, context, mask=cross_mask)
                x = nn.LayerNorm()(x + nn.Dropout(self.dropout, deterministic=not train)(cross))
            ff = nn.Dense(self.ff_dim)(x)
            ff = nn.relu(ff)
            ff = nn.Dense(self.hidden_dim)(ff)
            return nn.LayerNorm()(x + nn.Dropout(self.dropout, deterministic=not train)(ff))

    class GPT2Model(nn.Module):
        action_dim: int
        hidden_dim: int = 32
        ff_dim: int = 128
        num_blocks: int = 3
        use_cross_attention: bool = False
        dropout: float = 0.1
        max_timesteps: int = 4096
        num_teammates: int = 0

        @nn.compact
        def __call__(
            self,
            rtg,
            obs,
            actions,
            *,
            timesteps,
            mask=None,
            context=None,
            context_mask=None,
            teammate_id=None,
            train=False,
        ):
            B, T = obs.shape[0], obs.shape[1]
            g_tok = nn.Dense(self.hidden_dim)(rtg[..., None])
            o_tok = nn.Dense(self.hidden_dim)(obs)
            a_tok = nn.Dense(self.hidden_dim)(jax.nn.one_hot(actions, self.action_dim))
            pos = nn.Embed(self.max_timesteps, self.hidden_dim)(timesteps)
            g_tok, o_tok, a_tok = g_tok + pos, o_tok + pos, a_tok + pos
            if self.num_teammates > 0 and teammate_id is not None:
                team = nn.Embed(self.num_teammates, self.hidden_dim)(teammate_id)[:, None, :]
                g_tok, o_tok, a_tok = g_tok + team, o_tok + team, a_tok + team
            x = jnp.stack([g_tok, o_tok, a_tok], axis=2).reshape(B, T * 3, self.hidden_dim)
            x = nn.LayerNorm()(x)
            attn_mask = nn.make_causal_mask(jnp.ones((B, T * 3)))
            if mask is not None:
                stacked = jnp.repeat(mask.astype(bool), 3, axis=1)
                attn_mask = nn.combine_masks(attn_mask, nn.make_attention_mask(stacked, stacked))
            cross_mask = None
            if context is not None and context_mask is not None:
                q = jnp.ones((B, T * 3), dtype=bool)
                cross_mask = nn.make_attention_mask(q, context_mask.astype(bool))
            for _ in range(self.num_blocks):
                x = Block(self.hidden_dim, self.ff_dim, self.use_cross_attention, self.dropout)(
                    x, context=context, causal_mask=attn_mask, cross_mask=cross_mask, train=train
                )
            return x.reshape(B, T, 3, self.hidden_dim)[:, :, 1]

    return GPT2Model(**model_kwargs)


@pytest.mark.parametrize(
    "extra_kwargs",
    [
        {},
        {"use_cross_attention": True, "with_context": True},
        {"num_teammates": 5, "with_team": True},
        {"use_cross_attention": True, "with_context": True, "num_teammates": 5, "with_team": True},
    ],
    ids=["plain", "cross_attention", "teammate_oracle", "cross_attention+oracle"],
)
def test_off_defaults_are_byte_identical_to_the_pre_extension_backbone(extra_kwargs):
    """D_step/D_epi's four new kwargs, all at their None off-default, must
    reproduce the exact pre-extension computation -- both the parameter
    pytree structure (so an already-trained checkpoint still loads) and the
    output values (so it still acts the same way). This is the regression
    guarantee LIAM/MeLIBA/TAO depend on; it is checked against an inlined
    copy of the old computation, not re-derived from the new code, since the
    whole point is to catch backbone.py itself drifting.
    """
    with_context = extra_kwargs.pop("with_context", False)
    with_team = extra_kwargs.pop("with_team", False)
    B, T, action_dim, obs_dim, hidden = 3, 5, 4, 6, 8
    rtg, obs, actions, timesteps, mask = _inputs(B, T, action_dim, obs_dim)

    kwargs = dict(
        action_dim=action_dim,
        hidden_dim=hidden,
        ff_dim=16,
        num_blocks=2,
        dropout=0.0,
        **extra_kwargs,
    )
    old = _reference_forward(**kwargs)
    new = GPT2Model(**kwargs)

    call_kwargs = dict(timesteps=timesteps, mask=mask, train=False)
    if with_context:
        rng = np.random.default_rng(1)
        context = jnp.asarray(rng.normal(size=(B, 7, hidden)), dtype=jnp.float32)
        call_kwargs["context"] = context
        call_kwargs["context_mask"] = jnp.ones((B, 7), dtype=bool)
    if with_team:
        call_kwargs["teammate_id"] = jnp.asarray([0, 1, 0], dtype=jnp.int32)

    init_rng = jax.random.PRNGKey(42)
    params_old = old.init(init_rng, rtg, obs, actions, **call_kwargs)
    params_new = new.init(init_rng, rtg, obs, actions, **call_kwargs)

    old_struct = [
        (jax.tree_util.keystr(p), x.shape)
        for p, x in jax.tree_util.tree_leaves_with_path(params_old)
    ]
    new_struct = [
        (jax.tree_util.keystr(p), x.shape)
        for p, x in jax.tree_util.tree_leaves_with_path(params_new)
    ]
    assert old_struct == new_struct

    out_old = old.apply(params_old, rtg, obs, actions, **call_kwargs)
    out_new = new.apply(params_new, rtg, obs, actions, **call_kwargs)
    np.testing.assert_allclose(np.asarray(out_old), np.asarray(out_new), atol=1e-6)


def test_d_step_and_d_epi_change_the_output_and_preserve_shape():
    """The new conditioning must actually reach the computation (not a
    silently-ignored kwarg) and must not change the output's shape -- the
    readout still slices off the D_epi prefix and returns one hidden state
    per real timestep, matching what every downstream head (actor/imitator/
    critic for OMIS, policy head for everyone else) expects.
    """
    B, T, action_dim, obs_dim, hidden, H = 3, 5, 4, 6, 8, 6
    rng = np.random.default_rng(0)
    rtg, obs, actions, timesteps, mask = _inputs(B, T, action_dim, obs_dim)
    mate_actions = jnp.asarray(rng.integers(0, action_dim, size=(B, T)), dtype=jnp.int32)
    epi_state = jnp.asarray(rng.normal(size=(B, H, obs_dim)), dtype=jnp.float32)
    epi_mate_actions = jnp.asarray(rng.integers(0, action_dim, size=(B, H)), dtype=jnp.int32)
    epi_mask = jnp.ones((B, H), dtype=bool)

    model = GPT2Model(
        action_dim=action_dim, hidden_dim=hidden, ff_dim=16, num_blocks=2, dropout=0.0
    )
    init_rng = jax.random.PRNGKey(0)
    off_shape = (B, T, hidden)

    # D_step alone.
    p = model.init(
        init_rng,
        rtg,
        obs,
        actions,
        timesteps=timesteps,
        mask=mask,
        mate_actions=mate_actions,
        train=False,
    )
    out1 = model.apply(
        p, rtg, obs, actions, timesteps=timesteps, mask=mask, mate_actions=mate_actions, train=False
    )
    assert out1.shape == off_shape
    mate_actions2 = jnp.asarray(rng.integers(0, action_dim, size=(B, T)), dtype=jnp.int32)
    out1b = model.apply(
        p,
        rtg,
        obs,
        actions,
        timesteps=timesteps,
        mask=mask,
        mate_actions=mate_actions2,
        train=False,
    )
    assert not np.allclose(out1, out1b, atol=1e-6)

    # D_epi alone.
    p2 = model.init(
        init_rng,
        rtg,
        obs,
        actions,
        timesteps=timesteps,
        mask=mask,
        epi_state=epi_state,
        epi_mate_actions=epi_mate_actions,
        epi_mask=epi_mask,
        train=False,
    )
    out2 = model.apply(
        p2,
        rtg,
        obs,
        actions,
        timesteps=timesteps,
        mask=mask,
        epi_state=epi_state,
        epi_mate_actions=epi_mate_actions,
        epi_mask=epi_mask,
        train=False,
    )
    assert out2.shape == off_shape
    epi_state2 = jnp.asarray(rng.normal(size=(B, H, obs_dim)), dtype=jnp.float32)
    out2b = model.apply(
        p2,
        rtg,
        obs,
        actions,
        timesteps=timesteps,
        mask=mask,
        epi_state=epi_state2,
        epi_mate_actions=epi_mate_actions,
        epi_mask=epi_mask,
        train=False,
    )
    assert not np.allclose(out2, out2b, atol=1e-6)

    # Both together, shape still preserved.
    p3 = model.init(
        init_rng,
        rtg,
        obs,
        actions,
        timesteps=timesteps,
        mask=mask,
        mate_actions=mate_actions,
        epi_state=epi_state,
        epi_mate_actions=epi_mate_actions,
        epi_mask=epi_mask,
        train=False,
    )
    out3 = model.apply(
        p3,
        rtg,
        obs,
        actions,
        timesteps=timesteps,
        mask=mask,
        mate_actions=mate_actions,
        epi_state=epi_state,
        epi_mate_actions=epi_mate_actions,
        epi_mask=epi_mask,
        train=False,
    )
    assert out3.shape == off_shape
