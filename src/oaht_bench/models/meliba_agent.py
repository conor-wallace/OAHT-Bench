"""MeLIBA's model — architecture and inference only (Zintgraf et al. 2021, adapted).

The variational ego-history encoder emitting two Gaussian belief latents, the
partner-action decoder that shapes them, the belief-conditioned policy network, and
:class:`MelibaAgent`, the inference wrapper. Given trained parameters, ``MelibaAgent``
acts identically no matter how they were produced, so it is model-layer and carries
no dataset or training dependency. The offline two-stage training, the sequential-KL
ELBO, and the losses live in :mod:`oaht_bench.offline.meliba`.

MeLIBA is LIAM with a variational belief: the encoder emits ``(mean, logvar)`` for an
*agent character* and a *mental state* latent, and the policy conditions on those
belief *distribution parameters* concatenated to the observation rather than on a
point embedding. Sampling is deferred to the loss (where the reparameterisation rng
is available), so the encoder is deterministic given the backbone.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from oaht_bench.models.backbone import DecisionTransformer
from oaht_bench.models.return_conditioned_agent import ReturnConditionedAgent


class MelibaEncoder(nn.Module):
    """Ego-history encoder emitting two Gaussian belief latents.

    The DT backbone is read at the ``o_t`` positions (LIAM's information set),
    then four linear heads produce the *agent character* and *mental state*
    means and log-variances (``meliba_agent.py:138-145``). Sampling is deferred
    to the loss, where the reparameterisation rng is available, so this module is
    deterministic given the backbone.
    """

    action_dim: int
    latent_dim: int = 16
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

        char_mean = nn.Dense(self.latent_dim, name="char_mean")(obs_hidden)
        char_logvar = nn.Dense(self.latent_dim, name="char_logvar")(obs_hidden)
        mental_mean = nn.Dense(self.latent_dim, name="mental_mean")(obs_hidden)
        mental_logvar = nn.Dense(self.latent_dim, name="mental_logvar")(obs_hidden)
        return char_mean, char_logvar, mental_mean, mental_logvar


class MelibaDecoder(nn.Module):
    """Reconstructs the teammate's *action* from the belief samples.

    MeLIBA's decoder targets the partner action (``meliba_agent.py:283-300``),
    unlike LIAM which also reconstructs the partner observation. Conditioned on
    the two latent *samples* only — the belief must carry the partner information
    for the reconstruction to succeed, which is the point of the auxiliary task —
    with two hidden layers mirroring :class:`~oaht_bench.models.liam_agent.LiamDecoder`.
    """

    action_dim: int
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, latent_sample):
        h = nn.relu(nn.Dense(self.hidden_dim)(latent_sample))
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        return nn.Dense(self.action_dim)(h)


class MelibaNetwork(nn.Module):
    """Stage 2: the backbone conditioned on the frozen belief parameters.

    The policy sees ``(mean, logvar)`` of both latents concatenated to the
    observation (``meliba_agent.py:685``), i.e. the belief *distribution*, not a
    point embedding — the difference from LIAM. Offline the ``stop_gradient`` is
    unnecessary: stage 2 differentiates only the policy.
    """

    action_dim: int
    hidden_dim: int = 32
    dropout: float = 0.1

    @nn.compact
    def __call__(self, rtg, obs, actions, *, timesteps, belief, mask=None, train: bool = False):
        logits, _ = DecisionTransformer(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            use_cross_attention=False,
            dropout=self.dropout,
        )(
            rtg,
            jnp.concatenate([obs, belief], axis=-1),
            actions,
            timesteps=timesteps,
            mask=mask,
            train=train,
        )
        return logits


class MelibaAgent(ReturnConditionedAgent):
    """MeLIBA's architecture and inference as a :class:`ReturnConditionedAgent`.

    The base owns the rolling ego-window / return-to-go deployment; MeLIBA supplies
    its modules and the forward: encode the belief distribution parameters, condition
    the policy on them. The offline two-stage training and the losses live in
    :mod:`oaht_bench.offline.meliba`, which composes one of these. ``latent_dim`` is
    MeLIBA-specific and read from the top-level config.
    """

    def build_model(self) -> None:
        net = self.config.network
        if net.obs_dim is None or net.action_dim is None:
            raise ValueError(
                "obs_dim/action_dim are unresolved on the network config; the "
                "runner must resolve them from the dataset before build_model()."
            )
        common = dict(hidden_dim=net.hidden_dim, dropout=net.dropout)
        self.encoder = MelibaEncoder(
            action_dim=net.action_dim, latent_dim=self.config.latent_dim, **common
        )
        self.decoder = MelibaDecoder(action_dim=net.action_dim, hidden_dim=net.hidden_dim)
        self.network = MelibaNetwork(action_dim=net.action_dim, **common)

    def act(self, params, rtg, obs, actions, *, timesteps, mask):
        char_mean, char_logvar, mental_mean, mental_logvar = self.encoder.apply(
            params["stage1"]["encoder"],
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            mask=mask,
            train=False,
        )
        belief = jnp.concatenate([char_mean, char_logvar, mental_mean, mental_logvar], axis=-1)
        return self.network.apply(
            params["stage2"],
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            belief=belief,
            mask=mask,
            train=False,
        )
