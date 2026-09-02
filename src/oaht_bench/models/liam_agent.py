"""LIAM's model — architecture and inference only (Local Information Agent Modelling).

The ego-history encoder, the teammate-reconstruction decoder, the
embedding-conditioned policy network, and :class:`LiamAgent`, the inference
wrapper. Given trained parameters, ``LiamAgent`` acts identically no matter how
they were produced -- online or offline -- so it is model-layer and carries no
dataset or training dependency. The offline two-stage training and the losses live
in :mod:`oaht_bench.offline.liam`.

**What LIAM is.** An encoder summarises the ego agent's *local* history into an
embedding; a decoder reconstructs the *teammate's* observation and action from
that embedding; the policy is conditioned on the embedding. The teammate is never
observed by the policy -- only modelled -- which is the method's hypothesis and
why it is the natural floor for the trajectory-view family.

**The encoder is the backbone, read at the right position.** LIAM's encoder sees
``o¹_{0..t}`` and ``a¹_{0..t-1}`` -- observations through ``t``, actions only
through ``t-1``, because at ``t`` the ego has not acted yet. In the interleaved
``(G_t, o_t, a_t)`` sequence ``o_t`` sits at index ``3t+1`` and ``a_t`` at
``3t+2``, so under the causal mask the hidden state at ``o_t`` attends to
``o_{≤t}`` and ``a_{<t}`` and *not* ``a_t``. That is exactly LIAM's information
set, so no separate encoder architecture is required.
"""

from __future__ import annotations

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct

from oaht_bench.models.agent_interface import AgentPolicy
from oaht_bench.models.backbone import DecisionTransformer
from oaht_bench.models.masking import mask_logits


class LiamEncoder(nn.Module):
    """Ego-history encoder: the backbone, read at the ``o_t`` positions."""

    action_dim: int
    hidden_dim: int = 32
    dropout: float = 0.1

    @nn.compact
    def __call__(self, rtg, obs, actions, *, timesteps, mask=None, train: bool = False):
        _, obs_hidden = DecisionTransformer(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            use_cross_attention=False,
            dropout=self.dropout,
        )(rtg, obs, actions, timesteps=timesteps, mask=mask, train=train)
        return obs_hidden


class LiamDecoder(nn.Module):
    """Reconstructs the teammate's observation and action at time ``t``.

    Two independent heads, each two hidden layers with ReLU, matching
    ``liam_agent.py:119-160``. The action head returns logits; the reference
    applies softmax then takes ``-log(sum(p * onehot))``, which is the same
    quantity as softmax cross-entropy but less numerically stable.
    """

    obs_dim: int
    action_dim: int
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, embedding):
        h = nn.relu(nn.Dense(self.hidden_dim)(embedding))
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        mate_obs_hat = nn.Dense(self.obs_dim)(h)

        g = nn.relu(nn.Dense(self.hidden_dim)(embedding))
        g = nn.relu(nn.Dense(self.hidden_dim)(g))
        mate_action_logits = nn.Dense(self.action_dim)(g)
        return mate_obs_hat, mate_action_logits


class LiamNetwork(nn.Module):
    """Stage 2: the backbone, conditioned on a frozen teammate embedding.

    LIAM concatenates the embedding to the observation (``liam_agent.py:536``)
    rather than cross-attending; that is the conditioning mode which
    distinguishes it from TAO.
    """

    action_dim: int
    hidden_dim: int = 32
    dropout: float = 0.1

    @nn.compact
    def __call__(self, rtg, obs, actions, *, timesteps, embedding, mask=None, train: bool = False):
        logits, _ = DecisionTransformer(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            use_cross_attention=False,
            dropout=self.dropout,
        )(
            rtg,
            jnp.concatenate([obs, embedding], axis=-1),
            actions,
            timesteps=timesteps,
            mask=mask,
            train=train,
        )
        return logits


@struct.dataclass
class LiamHState:
    """The rolling ``K``-window LIAM carries between steps at inference.

    The same context the model was trained on -- a left-padded window of the last
    ``K`` steps -- plus the running return-to-go and a step counter. ``rtg`` is the
    scalar target that tracks what is *left* (decremented each step); ``ctx_rtg``
    is that scalar written into the window at each position.
    """

    ctx_obs: jnp.ndarray  # (K, obs_dim) float32, normalised
    ctx_act: jnp.ndarray  # (K,) int32, -10 where the ego has not acted
    ctx_rtg: jnp.ndarray  # (K,) float32
    ctx_t: jnp.ndarray  # (K,) int32 timesteps
    ctx_mask: jnp.ndarray  # (K,) bool validity
    rtg: jnp.ndarray  # scalar float32, the running target
    step: jnp.ndarray  # scalar int32, steps taken this episode


class LiamAgent(AgentPolicy):
    """LIAM's architecture and inference as a first-class :class:`AgentPolicy`.

    Inference is identical no matter how the parameters were produced -- online or
    offline -- so LIAM is a reactive acting policy like the actor-critics, driven by
    the shared :func:`~oaht_bench.common.run_episodes.run_episodes` loop. The
    difference is that it is *return-conditioned*: :meth:`get_action` maintains a
    rolling context window and a target return-to-go in its hidden state, decrements
    the target by the reward from the previous step, runs the decision-transformer
    forward, and samples a masked action. The offline two-stage training and the
    losses live in :mod:`oaht_bench.offline.liam`, which composes one of these.

    The window transform (observation standardisation, target scale) and the
    conditioning target are fixed at construction from the dataset the policy was
    trained on, so ``get_action`` is pure-jax and needs nothing but its params and
    the current observation. ``context_length``/``target_return``/``normalization``
    are only needed for acting; the trainer builds a :class:`LiamAgent` without them.
    """

    def __init__(self, config, *, context_length=None, target_return=None, normalization=None):
        net = config.network
        super().__init__(action_dim=net.action_dim, obs_dim=net.obs_dim)
        self.config = config
        self.context_length = context_length
        self._target_return = 0.0 if target_return is None else float(target_return)
        # Bake the dataset's transform as jax constants so acting repeats exactly what
        # training saw. Scalars stand in for the identity when the data was unnormalised,
        # which broadcast against any observation.
        if normalization is None:
            self._obs_mean = jnp.asarray(0.0, dtype=jnp.float32)
            self._obs_std = jnp.asarray(1.0, dtype=jnp.float32)
            self._rtg_scale = jnp.asarray(1.0, dtype=jnp.float32)
        else:
            self._obs_mean = jnp.asarray(normalization.obs_mean, dtype=jnp.float32)
            self._obs_std = jnp.asarray(normalization.obs_std, dtype=jnp.float32)
            self._rtg_scale = jnp.asarray(float(normalization.rtg_scale), dtype=jnp.float32)

    def init_hstate(self, batch_size, aux_info: dict = None) -> LiamHState:
        """A fresh left-padded window with the target return-to-go primed.

        ``batch_size`` is ignored: ``run_episodes`` vmaps whole episodes, so the
        window is per-episode and the episode axis is added by the vmap.
        """
        K = self.context_length
        return LiamHState(
            ctx_obs=jnp.zeros((K, self.obs_dim), dtype=jnp.float32),
            ctx_act=jnp.full((K,), -10, dtype=jnp.int32),
            ctx_rtg=jnp.zeros((K,), dtype=jnp.float32),
            ctx_t=jnp.zeros((K,), dtype=jnp.int32),
            ctx_mask=jnp.zeros((K,), dtype=bool),
            rtg=jnp.asarray(self._target_return, dtype=jnp.float32),
            step=jnp.asarray(0, dtype=jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_action(
        self,
        params,
        obs,
        done,
        avail_actions,
        hstate,
        rng,
        aux_obs=None,
        env_state=None,
        test_mode=False,
        reward=None,
    ):
        """One acting step, mirroring the reference deployment loop.

        Decrement the target by the reward actually received last step (zero on
        the first step, so the target starts intact), roll the window and write the
        current observation/target/timestep at the end with the action left blank
        (``-10``: the ego has not acted yet at ``t``), run the forward, mask
        unavailable actions, sample, then write the sampled action back into the
        window so the next step sees ``a_t`` at position ``t``.
        """
        K = self.context_length
        h = hstate
        r = (
            jnp.zeros((), jnp.float32)
            if reward is None
            else jnp.reshape(reward, (-1))[0].astype(jnp.float32)
        )
        rtg = h.rtg - r / self._rtg_scale

        ctx_obs = jnp.roll(h.ctx_obs, -1, axis=0)
        ctx_act = jnp.roll(h.ctx_act, -1)
        ctx_rtg = jnp.roll(h.ctx_rtg, -1)
        ctx_t = jnp.roll(h.ctx_t, -1)
        ctx_mask = jnp.roll(h.ctx_mask, -1)

        norm_obs = (jnp.reshape(obs, (-1)).astype(jnp.float32) - self._obs_mean) / self._obs_std
        ctx_obs = ctx_obs.at[-1].set(norm_obs)
        ctx_act = ctx_act.at[-1].set(jnp.int32(-10))
        ctx_rtg = ctx_rtg.at[-1].set(rtg)
        ctx_t = ctx_t.at[-1].set(jnp.minimum(h.step + 1, K * 64).astype(jnp.int32))
        ctx_mask = ctx_mask.at[-1].set(True)

        logits = self.act(
            params,
            ctx_rtg[None],
            ctx_obs[None],
            ctx_act[None],
            timesteps=ctx_t[None],
            mask=ctx_mask[None],
        )
        avail = jnp.reshape(avail_actions, (-1)).astype(jnp.float32)
        masked = mask_logits(logits[0, -1], avail)
        action = jax.lax.cond(
            test_mode,
            lambda: jnp.argmax(masked).astype(jnp.int32),
            lambda: jax.random.categorical(rng, masked).astype(jnp.int32),
        )
        ctx_act = ctx_act.at[-1].set(action)

        new_hstate = LiamHState(ctx_obs, ctx_act, ctx_rtg, ctx_t, ctx_mask, rtg, h.step + 1)
        return action, new_hstate

    def build_model(self) -> None:
        """Construct the flax modules from ``self.config`` (with resolved dims)."""
        net = self.config.network
        if net.obs_dim is None or net.action_dim is None:
            raise ValueError(
                "obs_dim/action_dim are unresolved on the network config; the "
                "runner must resolve them from the dataset before build_model()."
            )
        common = dict(hidden_dim=net.hidden_dim, dropout=net.dropout)
        self.encoder = LiamEncoder(action_dim=net.action_dim, **common)
        self.decoder = LiamDecoder(
            obs_dim=net.obs_dim, action_dim=net.action_dim, hidden_dim=net.hidden_dim
        )
        self.network = LiamNetwork(action_dim=net.action_dim, **common)

    def act(self, params, rtg, obs, actions, *, timesteps, mask):
        """Ego action logits: encode the ego history, condition the policy on it."""
        z = self.encoder.apply(
            params["stage1"]["encoder"],
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            mask=mask,
            train=False,
        )
        return self.network.apply(
            params["stage2"],
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            embedding=z,
            mask=mask,
            train=False,
        )
