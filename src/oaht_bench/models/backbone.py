"""The shared sequence backbone every trajectory-view baseline is built on (§3.1).

This is TAO's In-context Control Decoder, specified in its Appendix F, which is
the published precedent for §3.1's design: one architecture, with each baseline
stating exactly what it adds. Reproducing that structure rather than authoring
our own means differences between baselines are differences the papers describe,
not differences we introduced.

From Appendix F, verbatim in effect:

* GPT-2 decoder, **3 self-attention blocks**; each block is single-head
  self-attention + single-head cross-attention + feed-forward, residual
  connections and LayerNorm after each layer, dropout on both the residual and
  the attention weights.
* ``G_t, o_t, a_t`` pass through **modality-specific linear layers** plus an
  *episodic timestep* positional encoding.
* Actions are predicted autoregressively under a causal mask, from the hidden
  states **at the ``o_t`` token positions**.

Three details come from the authors' reference implementation rather than the
paper, which does not mention them:

* a **LayerNorm on the stacked token sequence** before the blocks
  (``embed_ln``, ``offline_stage_2/net.py:77``);
* the causal mask is **combined with the padding mask**, so real tokens never
  attend to padding — the paper says only "causal mask";
* sequences are **left-padded**, the Decision Transformer convention the
  reference inherits, so the most recent timestep is always last.
* Feed-forward **128 nodes, ReLU**; every other hidden layer **32 nodes, no
  activation**; modality-specific layers **32 nodes, no activation**.

``use_cross_attention`` is one switch: TAO keeps it and feeds the opponent
embedding in as key/value; LIAM and MeLIBA drop it and attach an auxiliary head
instead. That is precisely how Appendix F describes them.

OMIS needs two more, from its own reference (``OMIS/pretraining/nets.py``,
``GPTModel.forward``) rather than its paper text (Sec. 4.1, Eqs. 3-5; Appendix
C, F), which under-specifies this relative to what the released code actually
does. Its actor, opponent imitator, and critic all condition on in-context
data ``D_t^k = (D_epi^k, D_step^k)``, not just ``(G_t, o_t, a_t)``:

``D_step^k = (s_0, a^{-1,k}_0, ..., s_{t-1}, a^{-1,k}_{t-1})`` (Eq. 2) reads
as an opponent **(state, action)** pair on the page, but the reference's
``GPTModel.forward`` takes exactly one observation tensor, ``obs``, sized by
``my_obs_dim_dict`` -- the *ego's own* state, already present as ``o_tok``.
``get_batch`` (``pretraining/utils.py``) confirms this at the data layer: it
reads ``my_traj['obs']`` for the main stream and only ``traj[idx]['act']`` for
every opponent index, never an opponent *observation*. So Eq. 2's ``s_t`` is
notating the backbone's existing ego state, not asking for a second channel --
``D_step`` is **one** per-timestep token, ``mate_actions`` alone, appended
after ``a_t`` so ``o_tok`` stays at a fixed relative offset (index 1) within
its group regardless of group size. (An earlier version of this file carried
``mate_obs`` alongside it, on a literal reading of Eq. 2's notation before the
reference code was checked; that was the wrong direction to resolve the
ambiguity in.)

``D_epi^k`` is ``{(s̃_h, ã^{-1,k}_h)}_{h=1}^H``, built from ``C`` *past*
episodes against the same opponent (Appendix C: sample ``C`` trajectories,
concatenate consecutive segments) -- and here the state half is real: the
reference's ``prompt_states`` is sized by ``oppo_obs_dim_dict``, the
opponent's *own* observation, because D_epi reads *other* episodes where
there is no current ego state to reuse. A block of ``H`` ``(state,
opponent-action)`` pairs *prepended* to the whole sequence (Appendix F's
literal token order), fed through the same causal self-attention rather than
a second encoder-plus-cross-attention tower the way TAO's context is (OMIS's
own Fig. 1 is one shared backbone, not two).

Two more details are purely in the reference's code, not its paper text, and
are included anyway on the same "match the released model" basis as the
``embed_ln``/left-padding/feed-forward-width details above:

* **A separate timestep embedding table for D_epi.** ``GPTModel`` has
  ``self.embed_timestep`` for the main sequence and a *distinct*
  ``self.prompt_embed_timestep`` for the prompt (``nets.py:34,70``) --
  different learned parameters for the same raw timestep index, not the one
  table this file used before. ``epi_embed_t`` below mirrors that.
* **An agent-role embedding**, ``embed_agent_idx`` (``nets.py:68``), added on
  top of the per-modality linear layers: ``my_idx_embeddings`` onto the ego's
  ``o_tok``/``a_tok`` (not ``g_tok`` -- ``returns_embeddings`` only ever gets
  the timestep term, ``nets.py:187``), and the *same* ``oppo_idx_embeddings``
  value onto both ``D_step``'s ``m_action_tok`` and ``D_epi``'s
  ``epi_o_tok``/``epi_a_tok`` (``nets.py:175,218-219`` -- one shared opponent
  bias, not three separate ones). ``embed_role`` below is a 2-row table (ego,
  opponent) doing the same job without the reference's full per-literal-agent-
  index table, since this codebase only ever has two seats.

Both D_step and D_epi (and the role embedding they carry) are off by ``None``
defaults, checked with plain ``is None`` (a trace-time, not data-dependent,
branch) -- every existing consumer (LIAM, MeLIBA, TAO) gets the identical
graph it gets today. Combining ``use_cross_attention`` with ``D_epi`` is not
forbidden (the two are orthogonal by construction -- one adds cross-attention
K/V, the other lengthens the self-attention sequence) but is untested: no
baseline needs both, the same "not a deployable method" caveat
``num_teammates`` already carries in prose rather than code.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

#: Rows of the agent-role embedding table (mirrors the reference's
#: `embed_agent_idx`, collapsed to the two seats this codebase has).
_ROLE_EGO, _ROLE_MATE = 0, 1


class Block(nn.Module):
    """One transformer block: self-attention, optional cross-attention, feed-forward."""

    hidden_dim: int
    ff_dim: int
    use_cross_attention: bool
    dropout: float
    ff_activation: str = "relu"

    @nn.compact
    def __call__(self, x, *, context=None, causal_mask, cross_mask=None, train: bool):
        # Single-head throughout, per Appendix F.
        attn = nn.SelfAttention(
            num_heads=1,
            qkv_features=self.hidden_dim,
            dropout_rate=self.dropout,
            deterministic=not train,
        )(x, mask=causal_mask)
        x = nn.LayerNorm()(x + nn.Dropout(self.dropout, deterministic=not train)(attn))

        if self.use_cross_attention:
            if context is None:
                raise ValueError(
                    "use_cross_attention is set but no context was passed. TAO "
                    "feeds the opponent embedding in as key/value here; LIAM and "
                    "MeLIBA should construct the backbone with it disabled."
                )
            cross = nn.MultiHeadDotProductAttention(
                num_heads=1,
                qkv_features=self.hidden_dim,
                dropout_rate=self.dropout,
                deterministic=not train,
            )(x, context, mask=cross_mask)
            x = nn.LayerNorm()(x + nn.Dropout(self.dropout, deterministic=not train)(cross))

        ff = nn.Dense(self.ff_dim)(x)
        ff = nn.gelu(ff) if self.ff_activation == "gelu" else nn.relu(ff)
        ff = nn.Dense(self.hidden_dim)(ff)
        return nn.LayerNorm()(x + nn.Dropout(self.dropout, deterministic=not train)(ff))


class GPT2Model(nn.Module):
    """Shared GPT-2 backbone over ``(G_t, o_t, a_t)`` triples (§3.1).

    TAO calls this the "In-context Control Decoder"; OMIS's own reference code
    names the equivalent trunk ``GPTModel``. It carries no action head of its
    own -- every baseline reads the hidden states at the ``o_t`` positions off
    this backbone and attaches whatever heads its role needs: an encoder role
    (LIAM, MeLIBA, OMIS) uses them as-is to summarise teammate behaviour or
    context; a decoder/actor role attaches its own ``nn.Dense(action_dim)`` (and,
    for OMIS, imitator/critic heads) on top. Appendix F is specific that these
    hidden states already contain ``o_t`` and ``a_{t-1}``, which is what makes
    them enough for LIAM's auxiliary decoder to reconstruct the teammate from.
    """

    action_dim: int
    hidden_dim: int = 32
    ff_dim: int = 128
    num_blocks: int = 3
    use_cross_attention: bool = False
    dropout: float = 0.1
    max_timesteps: int = 4096
    #: Each block's feed-forward activation. "relu" (default) is every existing
    #: baseline's byte-identical behavior; OMIS's own reference uses "gelu"
    #: (App. F / GPT2Config(activation_function="gelu_new")).
    ff_activation: str = "relu"
    #: Dense layers building the observation token (o_tok), activated after
    #: every layer but the last. 1 (default) is today's single, unactivated
    #: Dense; OMIS's own embed_state is 3 layers with obs_embed_activation
    #: "leaky_relu" (App. F / OMIS/pretraining/nets.py). action/RTG embeddings
    #: stay a single unactivated Dense regardless -- only o_tok's depth here
    #: differs in the reference.
    obs_embed_layers: int = 1
    obs_embed_activation: str = "none"
    #: >0 turns on the teammate-identity oracle: a learned per-teammate embedding
    #: added to every token (like the positional encoding). Off (0) by default, so
    #: every existing baseline is byte-identical. Not a deployable method -- it
    #: consumes the ground-truth teammate id, so it only bounds what a model could do
    #: given perfect teammate identification.
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
        mate_actions=None,
        epi_state=None,
        epi_mate_actions=None,
        epi_mask=None,
        train: bool = False,
    ):
        B, T = obs.shape[0], obs.shape[1]
        embed_t = nn.Embed(self.max_timesteps, self.hidden_dim)

        # Modality-specific linear layers, 32 nodes, no activation -- except
        # o_tok, whose depth/activation are configurable (obs_embed_layers/
        # obs_embed_activation): 1/"none" (default) is this same single,
        # unactivated Dense; OMIS's own embed_state is a deeper, activated MLP
        # (see the class docstring above).
        g_tok = nn.Dense(self.hidden_dim)(rtg[..., None])
        o_tok = obs
        for i in range(self.obs_embed_layers):
            o_tok = nn.Dense(self.hidden_dim)(o_tok)
            if i < self.obs_embed_layers - 1 and self.obs_embed_activation == "leaky_relu":
                o_tok = nn.leaky_relu(o_tok)
        a_tok = nn.Dense(self.hidden_dim)(jax.nn.one_hot(actions, self.action_dim))

        # Episodic timestep encoding, added to every modality (Chen et al. 2021).
        pos = embed_t(timesteps)
        g_tok, o_tok, a_tok = g_tok + pos, o_tok + pos, a_tok + pos

        # Teammate-identity oracle: a per-teammate bias added to every token,
        # broadcast over the window. Present only when enabled; the ground-truth id
        # makes this an upper bound, not a method.
        if self.num_teammates > 0 and teammate_id is not None:
            team = nn.Embed(self.num_teammates, self.hidden_dim)(teammate_id)  # (B, hidden)
            team = team[:, None, :]
            g_tok, o_tok, a_tok = g_tok + team, o_tok + team, a_tok + team

        # Agent-role embedding (reference's `embed_agent_idx`): added to the
        # ego's o_tok/a_tok whenever D_step or D_epi puts an opponent token in
        # the same sequence -- g_tok/returns_embeddings never gets one, per
        # the reference (nets.py:187). Off entirely for LIAM/MeLIBA/TAO calls --
        # constructed only in that branch, not merely called conditionally: an
        # nn.Embed built but never invoked still consumes an auto-generated
        # submodule name and would shift every later submodule's name, which
        # breaks loading an already-trained LIAM/MeLIBA/TAO checkpoint.
        using_role = mate_actions is not None or epi_state is not None
        if using_role:
            embed_role = nn.Embed(2, self.hidden_dim)
            ego_role = embed_role(jnp.asarray(_ROLE_EGO))
            o_tok, a_tok = o_tok + ego_role, a_tok + ego_role

        # OMIS's D_step (Eq. 2): the opponent's action history up to the
        # current step, one per-timestep token appended after a_tok so o_tok
        # stays at a fixed relative offset (index 1) within its group
        # regardless of group size -- the readout below does not need to
        # special-case this. Eq. 2 reads as an opponent (state, action) pair,
        # but the reference's main stream has no opponent-observation channel
        # at all (see module docstring) -- action-only matches what is
        # actually trained, not a simplification of it.
        group = 3
        per_t_toks = [g_tok, o_tok, a_tok]
        if mate_actions is not None:
            m_action_tok = (
                nn.Dense(self.hidden_dim)(jax.nn.one_hot(mate_actions, self.action_dim)) + pos
            )
            m_action_tok = m_action_tok + embed_role(jnp.asarray(_ROLE_MATE))
            if self.num_teammates > 0 and teammate_id is not None:
                m_action_tok = m_action_tok + team
            per_t_toks.append(m_action_tok)
            group = 4

        # Interleave to (G_0, o_0, a_0, [m_0], G_1, o_1, a_1, [m_1], ...).
        x = jnp.stack(per_t_toks, axis=2).reshape(B, T * group, self.hidden_dim)

        # OMIS's D_epi (Appendix C/F): H (state, opponent-action) pairs from
        # C past episodes against the same opponent, prepended as their own
        # block ahead of the main sequence -- Appendix F's literal token
        # order, fed through the same causal blocks rather than a second
        # encoder (unlike TAO's context, which cross-attends instead). Its own
        # timestep table (`epi_embed_t`), not `embed_t` -- the reference's
        # `prompt_embed_timestep` is a separate learned table from
        # `embed_timestep` (nets.py:34,70), not the same positions reused.
        epi_len = 0
        if epi_state is not None:
            epi_embed_t = nn.Embed(self.max_timesteps, self.hidden_dim)
            H = epi_state.shape[1]
            epi_pos = epi_embed_t(jnp.arange(H))[None]  # block-relative position
            mate_role = embed_role(jnp.asarray(_ROLE_MATE))
            epi_o_tok = nn.Dense(self.hidden_dim)(epi_state) + epi_pos + mate_role
            epi_a_tok = (
                nn.Dense(self.hidden_dim)(jax.nn.one_hot(epi_mate_actions, self.action_dim))
                + epi_pos
                + mate_role
            )
            epi_tok = jnp.stack([epi_o_tok, epi_a_tok], axis=2).reshape(B, H * 2, self.hidden_dim)
            x = jnp.concatenate([epi_tok, x], axis=1)
            epi_len = H * 2

        # Reference `embed_ln`: LayerNorm on the stacked sequence, not per token
        # stream. Absent from the paper.
        x = nn.LayerNorm()(x)
        N = x.shape[1]

        attn_mask = nn.make_causal_mask(jnp.ones((B, N)))
        pad_parts = []
        if epi_state is not None:
            epi_pad = (
                jnp.ones((B, epi_len), dtype=bool)
                if epi_mask is None
                else jnp.repeat(epi_mask.astype(bool), 2, axis=1)
            )
            pad_parts.append(epi_pad)
        if mask is not None:
            # Each timestep contributes `group` tokens, so the padding mask has
            # to be repeated to line up. Without this, real tokens attend to padding.
            pad_parts.append(jnp.repeat(mask.astype(bool), group, axis=1))
        elif epi_state is not None:
            # D_epi has a real mask but the main window doesn't -- still need a
            # same-length all-valid fill so the two concatenate.
            pad_parts.append(jnp.ones((B, T * group), dtype=bool))
        if pad_parts:
            stacked = jnp.concatenate(pad_parts, axis=1)
            attn_mask = nn.combine_masks(attn_mask, nn.make_attention_mask(stacked, stacked))

        cross_mask = None
        if context is not None and context_mask is not None:
            q = jnp.ones((B, N), dtype=bool)
            cross_mask = nn.make_attention_mask(q, context_mask.astype(bool))

        for _ in range(self.num_blocks):
            x = Block(
                self.hidden_dim,
                self.ff_dim,
                self.use_cross_attention,
                self.dropout,
                ff_activation=self.ff_activation,
            )(x, context=context, causal_mask=attn_mask, cross_mask=cross_mask, train=train)

        # Hidden states are read off the o_t positions: index 1 of each group,
        # after slicing off the D_epi prefix (always 0 when D_epi is unused, so
        # this reduces to today's `x.reshape(B, T, 3, hidden)[:, :, 1]` exactly).
        obs_hidden = x[:, epi_len:].reshape(B, T, group, self.hidden_dim)[:, :, 1]
        return obs_hidden
