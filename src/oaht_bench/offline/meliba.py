"""MeLIBA — Meta-Learned Interactive Bayesian Agent, adapted to the offline setting.

Sources, in the priority the port was written against:

1. **The jax-aht implementation** (its ``ego_agent_training/meliba_agent.py``, the
   MeLIBA online learner we did not absorb) is the source of truth for the
   objective and modelling design. From it:
   - the encoder emits **two Gaussian latents** — an *agent character* and a
     *mental state* — each ``(mean, logvar)`` (``meliba_agent.py:138-145``);
   - the decoder is trained by a **VariBAD-style sequential KL** in which the prior
     at ``t`` is the posterior at ``t-1`` (``meliba_agent.py:190-207``), plus a
     **partner-action** reconstruction (``meliba_agent.py:283-300``) — MeLIBA
     reconstructs the *teammate's action*, not its observation, which is the one
     modelling difference from LIAM that matters;
   - the policy conditions on the **belief distribution parameters**
     ``(mean, logvar)`` of both latents, not on the samples, and behind a
     ``stop_gradient`` (``meliba_agent.py:685``).
2. **The paper** (Zintgraf et al. 2021, *Deep Interactive Bayesian RL via
   Meta-Learning*; ``meliba.pdf``) is the second reference: MeLIBA meta-learns an
   approximate belief over the partner so the ego is Bayes-adaptive; the ELBO is
   ``recon + β·KL``.
3. **The shared offline protocol** (:mod:`oaht_bench.offline.liam`,
   :mod:`oaht_bench.offline.tao`) governs the offline *structure*: the DT backbone
   is the encoder read at the ``o_t`` positions, training is two-stage
   (encoder+decoder, then a frozen-encoder policy), and losses are masked over
   valid timesteps.

**What the offline setting changes.** Online, MeLIBA's belief drives exploration
during interaction; offline there is no interaction, so the belief is learned from
the fixed dataset and the policy is belief-conditioned behaviour cloning. Two of
the online decoder's mechanisms are simplifications here, per source (3): the
elaborate per-timestep sub-trajectory reconstruction (``transform_timestep_to_k_batch``)
is replaced by LIAM's single reconstruction head, and the ``stop_gradient`` is
unnecessary because stage 2 never differentiates the encoder. Everything that
distinguishes MeLIBA from LIAM — the variational two-latent belief, the sequential
KL, the partner-*action* target, and belief-parameter conditioning — is kept.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from oaht_bench.offline.backbone import DecisionTransformer
from oaht_bench.offline.registry import BaseAhtPolicy
from oaht_bench.offline.utils import mask_logits, sample_window_batch


def _masked_accuracy(logits, labels, mask) -> jnp.ndarray:
    """Top-1 accuracy over valid timesteps (see :func:`liam._masked_accuracy`)."""
    correct = (jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)
    m = mask.astype(jnp.float32)
    return (correct * m).sum() / jnp.maximum(m.sum(), 1.0)


def _sequential_kl(mean, logvar, mask) -> jnp.ndarray:
    """VariBAD sequential belief KL, from ``meliba_agent.py:190-207``.

    The prior for the belief at ``t`` is the belief at ``t-1``; the belief at
    ``t=0`` is regularised toward the unit Gaussian. ``mean``/``logvar`` are the
    two latents concatenated on the feature axis, shape ``(B, T, D)`` with
    ``D = 2·latent_dim``. Computed in the numerically stable ``exp(logE - logS)``
    form and averaged over valid timesteps so it sits on the same footing as the
    masked reconstruction term.
    """
    b, _, d = mean.shape
    prior = jnp.zeros((b, 1, d))  # N(0, I): the belief before any observation
    all_mean = jnp.concatenate([prior, mean], axis=1)
    all_logvar = jnp.concatenate([prior, logvar], axis=1)
    mu, m = all_mean[:, 1:], all_mean[:, :-1]  # posterior_t, posterior_{t-1}
    log_e, log_s = all_logvar[:, 1:], all_logvar[:, :-1]
    kl = 0.5 * (
        jnp.sum(log_s - log_e, axis=-1)
        - d
        + jnp.sum(jnp.exp(log_e - log_s), axis=-1)
        + jnp.sum((m - mu) ** 2 * jnp.exp(-log_s), axis=-1)
    )
    mask = mask.astype(jnp.float32)
    return (kl * mask).sum() / jnp.maximum(mask.sum(), 1.0)


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
    with two hidden layers mirroring :class:`liam.LiamDecoder`.
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


def _reparameterise(mean, logvar, key):
    """N(mean, exp(logvar)) via the reparameterisation trick (``meliba_agent.py:15-30``)."""
    return mean + jnp.exp(0.5 * logvar) * jax.random.normal(key, mean.shape)


def meliba_reconstruction_loss(
    params, encoder, decoder, batch, *, kl_weight, rngs=None, train: bool = True
):
    """Stage 1: the ELBO ``recon_action + β·KL`` (``meliba_agent.py:430-435``).

    Reconstruction is the negative log-likelihood of the teammate's action at
    time ``t`` from the belief samples; the KL is the sequential belief
    regulariser. The reparameterisation key is folded from the dropout rng the
    runner already provides, so no change to the shared training loop is needed.
    """
    char_mean, char_logvar, mental_mean, mental_logvar = encoder.apply(
        params["encoder"],
        batch["ego_rtg"],
        batch["ego_obs"],
        batch["ego_actions"],
        timesteps=batch["timesteps"],
        mask=batch["mask"],
        train=train,
        rngs=rngs,
    )

    key = rngs["dropout"] if rngs else jax.random.PRNGKey(0)
    k_char, k_mental = jax.random.split(jax.random.fold_in(key, 1))
    char_sample = _reparameterise(char_mean, char_logvar, k_char)
    mental_sample = _reparameterise(mental_mean, mental_logvar, k_mental)

    mate_logits = decoder.apply(
        params["decoder"], jnp.concatenate([char_sample, mental_sample], axis=-1)
    )
    # The teammate could only have taken a legal action.
    mate_logits = mask_logits(mate_logits, batch["mate_avail"])

    mask = batch["mask"].astype(jnp.float32)
    denom = jnp.maximum(mask.sum(), 1.0)

    recon_act = optax.softmax_cross_entropy_with_integer_labels(mate_logits, batch["mate_actions"])
    recon_act = (recon_act * mask).sum() / denom
    recon_acc = _masked_accuracy(mate_logits, batch["mate_actions"], mask)

    mean_all = jnp.concatenate([char_mean, mental_mean], axis=-1)
    logvar_all = jnp.concatenate([char_logvar, mental_logvar], axis=-1)
    kl = _sequential_kl(mean_all, logvar_all, batch["mask"])

    total = recon_act + kl_weight * kl
    return total, {
        "loss": total,
        "recon_action": recon_act,
        "recon_action_accuracy": recon_acc,
        "kl": kl,
    }


def meliba_belief(encoder, encoder_params, batch, *, train: bool = False):
    """The frozen-encoder belief parameters the policy conditions on.

    ``concat(char_mean, char_logvar, mental_mean, mental_logvar)`` — the same
    order the online policy uses (``meliba_agent.py:685``). Shared by stage 2 and
    evaluation so the conditioning cannot drift between them.
    """
    char_mean, char_logvar, mental_mean, mental_logvar = encoder.apply(
        encoder_params,
        batch["ego_rtg"],
        batch["ego_obs"],
        batch["ego_actions"],
        timesteps=batch["timesteps"],
        mask=batch["mask"],
        train=train,
    )
    return jnp.concatenate([char_mean, char_logvar, mental_mean, mental_logvar], axis=-1)


def meliba_policy_loss(
    params, policy, encoder, encoder_params, batch, *, rngs=None, train: bool = True
):
    """Stage 2: behaviour cloning conditioned on the frozen belief.

    Mirrors :func:`liam.liam_policy_loss`; ``encoder_params`` are stage-1 outputs
    and are never differentiated.
    """
    belief = meliba_belief(encoder, encoder_params, batch)
    logits = mask_logits(
        policy.apply(
            params,
            batch["ego_rtg"],
            batch["ego_obs"],
            batch["ego_actions"],
            timesteps=batch["timesteps"],
            belief=belief,
            mask=batch["mask"],
            train=train,
            rngs=rngs,
        ),
        batch["ego_avail"],
    )
    mask = batch["mask"].astype(jnp.float32)
    bc = optax.softmax_cross_entropy_with_integer_labels(logits, batch["ego_actions"])
    acc = _masked_accuracy(logits, batch["ego_actions"], mask)
    bc = (bc * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    return bc, {"loss": bc, "bc": bc, "action_accuracy": acc}


class MelibaPolicy(BaseAhtPolicy):
    """MeLIBA on the two-stage contract.

    Same shape as :class:`~oaht_bench.offline.liam.model.LiamPolicy` -- an
    ego-history encoder, window batches, two stages -- but the encoder is
    variational (two Gaussian latents) and the policy conditions on the belief
    *parameters* rather than a point embedding. ``latent_dim`` and ``kl_weight``
    are MeLIBA-specific and read from the top-level config; the shared model dims
    come from ``config.network``.
    """

    name = "meliba"

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

    def _sample_batch(self, _step):
        return sample_window_batch(self.dataset.windows, self.np_rng, self.config.stage2_batch_size)

    def train_stage_1(self):
        init_batch = self._sample_batch(0)
        self.rng, k1, k2 = jax.random.split(self.rng, 3)
        encoder_params = self.encoder.init(
            k1,
            init_batch["ego_rtg"],
            init_batch["ego_obs"],
            init_batch["ego_actions"],
            timesteps=init_batch["timesteps"],
            mask=init_batch["mask"],
        )
        char_mean, _, mental_mean, _ = self.encoder.apply(
            encoder_params,
            init_batch["ego_rtg"],
            init_batch["ego_obs"],
            init_batch["ego_actions"],
            timesteps=init_batch["timesteps"],
            mask=init_batch["mask"],
        )
        # The decoder reads the two latent samples concatenated; means stand in
        # for a sample at init, where only shapes matter.
        decoder_params = self.decoder.init(k2, jnp.concatenate([char_mean, mental_mean], axis=-1))
        params = {"encoder": encoder_params, "decoder": decoder_params}

        def loss(p, b, rngs):
            return meliba_reconstruction_loss(
                p, self.encoder, self.decoder, b, kl_weight=self.config.kl_weight, rngs=rngs
            )

        return self._run_stage(
            loss,
            params,
            self._sample_batch,
            learning_rate=self.config.stage1_learning_rate,
            steps=self.config.stage1_steps,
            prefix="Stage1",
        )

    def train_stage_2(self, stage1_params):
        init_batch = self._sample_batch(0)
        self.rng, k = jax.random.split(self.rng)
        belief = meliba_belief(self.encoder, stage1_params["encoder"], init_batch)
        policy_params = self.network.init(
            k,
            init_batch["ego_rtg"],
            init_batch["ego_obs"],
            init_batch["ego_actions"],
            timesteps=init_batch["timesteps"],
            belief=belief,
            mask=init_batch["mask"],
        )

        def loss(p, b, rngs):
            return meliba_policy_loss(
                p, self.network, self.encoder, stage1_params["encoder"], b, rngs=rngs
            )

        return self._run_stage(
            loss,
            policy_params,
            self._sample_batch,
            learning_rate=self.config.stage2_learning_rate,
            steps=self.config.stage2_steps,
            prefix="Stage2",
        )

    def act(self, params, rtg, obs, actions, *, timesteps, mask):
        char_mean, char_logvar, mental_mean, mental_logvar = self.encoder.apply(
            params["stage1"]["encoder"], rtg, obs, actions,
            timesteps=timesteps, mask=mask, train=False,
        )
        belief = jnp.concatenate([char_mean, char_logvar, mental_mean, mental_logvar], axis=-1)
        return self.network.apply(
            params["stage2"], rtg, obs, actions,
            timesteps=timesteps, belief=belief, mask=mask, train=False,
        )
