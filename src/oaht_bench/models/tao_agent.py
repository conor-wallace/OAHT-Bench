"""TAO's model — architecture and inference only (Wang et al., ICLR 2024).

The opponent-policy encoder ``M_θe`` over the teammate stream, the ancillary
action decoder that shapes its embedding, the cross-attending control network, and
:class:`TaoAgent`, the inference wrapper. Given trained parameters, ``TaoAgent``
acts identically no matter how they were produced, so it is model-layer and carries
no dataset or training dependency. The three-stage training, the losses, and the
deployment-context seeding live in :mod:`oaht_bench.offline.tao`.

Appendix F specifies the encoder as a GPT-2 *encoder*, 3 blocks of single-head
attention + feed-forward, with ELU modality layers and a fusion layer producing one
token per timestep; the control network is the shared backbone with cross-attention,
taking the teammate token sequence as key/value.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from oaht_bench.models.backbone import DecisionTransformer
from oaht_bench.models.return_conditioned_agent import ReturnConditionedAgent


class OpponentPolicyEncoder(nn.Module):
    """``M_θe``: the teammate's stream to a sequence of policy-embedding tokens.

    Returns the full token sequence. Stage 1 average-pools it into ``z̄⁻¹``;
    stages 2 and 3 feed the sequence itself as key/value into the decoder's
    cross-attention.
    """

    action_dim: int
    hidden_dim: int = 32
    ff_dim: int = 128
    num_blocks: int = 3
    dropout: float = 0.1
    max_timesteps: int = 4096

    @nn.compact
    def __call__(
        self, mate_next_obs, mate_actions, mate_rewards, *, mask, timesteps, train: bool = False
    ):
        """``mate_next_obs`` is deliberate.

        The reference feeds ``traj['next_observations']`` alongside ``actions``
        and ``rewards`` at the *same* index (``offline_stage_1/utils.py:109-111``),
        which is how the paper's ``(a_{t-1}, r_{t-1}, o_t)`` fusion is realised —
        by choosing next-observations, not by shifting the action and reward
        streams. An index shift gets the same pairing but labels each token with
        a different timestep, and the timestep embedding below is not symmetric.
        """
        # Modality-specific linear layers with ELU, 32 nodes (Appendix F).
        a = nn.elu(nn.Dense(self.hidden_dim)(jax.nn.one_hot(mate_actions, self.action_dim)))
        r = nn.elu(nn.Dense(self.hidden_dim)(mate_rewards[..., None]))
        o = nn.elu(nn.Dense(self.hidden_dim)(mate_next_obs))

        # Reference net.py:48-55 -- obs and reward take the timestep embedding at
        # t, the action takes t-1, clamped at 0. The paper mentions no positional
        # encoding in the encoder at all.
        embed_t = nn.Embed(self.max_timesteps, self.hidden_dim)
        pos = embed_t(timesteps)
        pos_m1 = embed_t(jnp.where(timesteps > 0, timesteps - 1, timesteps))
        a, r, o = a + pos_m1, r + pos, o + pos

        # LayerNorm over the concatenated 3*hidden vector, then fuse to one token
        # per timestep (reference `embed_ln` is LayerNorm(3 * hidden_size)).
        fused = nn.Dense(self.hidden_dim)(nn.LayerNorm()(jnp.concatenate([a, r, o], axis=-1)))

        # Encoder blocks: no causal mask -- the whole teammate trajectory is
        # available when building a policy embedding.
        pad = nn.make_attention_mask(mask, mask)
        x = fused
        for _ in range(self.num_blocks):
            attn = nn.SelfAttention(
                num_heads=1,
                qkv_features=self.hidden_dim,
                dropout_rate=self.dropout,
                deterministic=not train,
            )(x, mask=pad)
            x = nn.LayerNorm()(x + attn)
            ff = nn.Dense(self.hidden_dim)(nn.relu(nn.Dense(self.ff_dim)(x)))
            x = nn.LayerNorm()(x + ff)
        return x

    @staticmethod
    def pool(tokens, mask=None):
        """``AP``, giving ``z̄⁻¹``.

        The reference pools with a plain ``nn.AvgPool1d(kernel_size=NUM_STEPS)``
        (``nn_trainer.py:32,109``) — an unmasked mean over every position,
        padding included. That is what produced the published numbers, so it is
        the default here; ``mask`` is accepted for the masked variant but is not
        what TAO does.
        """
        if mask is None:
            return tokens.mean(axis=1)
        m = mask.astype(tokens.dtype)[..., None]
        return (tokens * m).sum(axis=1) / jnp.maximum(m.sum(axis=1), 1.0)


class AncillaryActionDecoder(nn.Module):
    """Predicts the teammate's actions from its own observations plus ``z̄⁻¹``.

    Stage 1's generative loss (Eq. 2). Deliberately weak — it exists to shape the
    embedding, not to be a good teammate model.
    """

    action_dim: int
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, mate_obs, embedding):
        # Reference MLPDecoder (offline_stage_1/net.py:116-133): the *raw*
        # observation is concatenated with the latent, then one hidden layer with
        # ReLU followed by LayerNorm. Embedding the observation first, or
        # dropping the activation, is a different function.
        z = jnp.broadcast_to(embedding[:, None, :], (*mate_obs.shape[:2], embedding.shape[-1]))
        h = nn.LayerNorm()(
            nn.relu(nn.Dense(self.hidden_dim)(jnp.concatenate([mate_obs, z], axis=-1)))
        )
        return nn.Dense(self.action_dim)(h)


class TaoNetwork(nn.Module):
    """Stage 2: the shared backbone, cross-attending to the policy embedding."""

    action_dim: int
    hidden_dim: int = 32
    dropout: float = 0.1

    @nn.compact
    def __call__(
        self,
        rtg,
        obs,
        actions,
        *,
        timesteps,
        context,
        mask=None,
        context_mask=None,
        train: bool = False,
    ):
        logits, _ = DecisionTransformer(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            use_cross_attention=True,  # Appendix F: z^-1 enters as key/value.
            dropout=self.dropout,
        )(
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            mask=mask,
            context=context,
            context_mask=context_mask,
            train=train,
        )
        return logits


class TaoAgent(ReturnConditionedAgent):
    """TAO's architecture and inference as a :class:`ReturnConditionedAgent`.

    The base owns the same rolling ego-window / return-to-go deployment LIAM uses;
    TAO differs only in the forward, which cross-attends to a *fixed* policy-embedding
    context rather than concatenating an embedding. That context -- the offline
    Opponent Context Window, ``C`` teammate trajectories already encoded -- is seeded
    at the end of training and travels in ``params["stage2"]``, so acting needs no
    encoder pass and no extra state. The three-stage training, the losses, and the
    context seeding live in :mod:`oaht_bench.offline.tao`, which composes one of these.
    """

    def build_model(self) -> None:
        net = self.config.network
        if net.obs_dim is None or net.action_dim is None:
            raise ValueError(
                "obs_dim/action_dim are unresolved on the network config; the "
                "runner must resolve them from the dataset before build_model()."
            )
        self.encoder = OpponentPolicyEncoder(
            action_dim=net.action_dim,
            hidden_dim=net.hidden_dim,
            ff_dim=net.ff_dim,
            num_blocks=net.num_blocks,
            dropout=net.dropout,
        )
        self.decoder = AncillaryActionDecoder(action_dim=net.action_dim, hidden_dim=net.hidden_dim)
        self.network = TaoNetwork(
            action_dim=net.action_dim, hidden_dim=net.hidden_dim, dropout=net.dropout
        )

    def act(self, params, rtg, obs, actions, *, timesteps, mask):
        """Ego action logits, cross-attending to the baked deployment context."""
        stage2 = params["stage2"]
        return self.network.apply(
            stage2["policy"],
            rtg,
            obs,
            actions,
            timesteps=timesteps,
            context=stage2["context"],
            mask=mask,
            context_mask=stage2["context_mask"],
            train=False,
        )
