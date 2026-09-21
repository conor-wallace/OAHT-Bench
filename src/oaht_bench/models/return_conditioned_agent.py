"""The shared deployment loop for the offline decision-transformer agents.

Every offline baseline built on the sequence backbone deploys the same way: a
left-padded rolling ``K``-window of ``(return-to-go, observation, action)`` and a
target return that is decremented by the reward actually received. They differ
only in what their forward pass reads *besides* the ego window -- LIAM
concatenates a teammate embedding, TAO cross-attends to a policy-embedding
context, and so on. So the window/return-to-go bookkeeping lives here once, moved
out of the old inline ``offline.evaluate._rollout``, and the forward is deferred
to :meth:`ReturnConditionedAgent.act`, which each subclass implements over its own
modules. This is why the offline agents are first-class :class:`AgentPolicy`
objects driven by the shared ``run_episodes`` loop rather than a bespoke rollout.
"""

from __future__ import annotations

import abc
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from oaht_bench.models.agent_interface import AgentPolicy
from oaht_bench.models.masking import mask_logits


@struct.dataclass
class ContextWindow:
    """The rolling ``K``-window a return-conditioned agent carries between steps.

    The context the model was trained on -- a left-padded window of the last ``K``
    steps -- plus the running return-to-go and a step counter. ``rtg`` is the
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


class ReturnConditionedAgent(AgentPolicy):
    """An :class:`AgentPolicy` that acts by return-conditioned sequence modelling.

    Subclasses implement :meth:`build_model` (construct their flax modules from the
    resolved config) and :meth:`act` (``(params, rtg, obs, actions, *, timesteps,
    mask) -> logits`` over their own modules). This base supplies :meth:`init_hstate`
    and :meth:`get_action`, so the shared, vmapped ``run_episodes`` loop can drive
    them exactly as it drives the reactive actor-critics.

    The dataset transform (observation standardisation, return scale) and the
    conditioning target are fixed at construction from the dataset the policy was
    trained on, so :meth:`get_action` is pure-jax and needs nothing but its params
    and the current observation. ``context_length``/``target_return``/
    ``normalization`` are only needed for acting; a trainer that only needs the
    modules builds a subclass without them.
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

    @abc.abstractmethod
    def build_model(self) -> None:
        """Construct the flax modules from ``self.config`` (with resolved dims)."""

    @abc.abstractmethod
    def act(self, params, rtg, obs, actions, *, timesteps, mask):
        """Ego action logits for one ``(1, K)`` window, over the subclass's modules."""

    def init_hstate(self, batch_size, aux_info: dict = None) -> ContextWindow:
        """A fresh left-padded window with the target return-to-go primed.

        ``batch_size`` is ignored: ``run_episodes`` vmaps whole episodes, so the
        window is per-episode and the episode axis is added by the vmap.
        """
        K = self.context_length
        return ContextWindow(
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

        new_hstate = ContextWindow(ctx_obs, ctx_act, ctx_rtg, ctx_t, ctx_mask, rtg, h.step + 1)
        return action, new_hstate
