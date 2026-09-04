"""CNN+GRU actor-critic for Overcooked-v2 (Gessler et al., ICLR 2025, App. C.1.1).

The plain ``RNNActorCritic`` runs a GRU over the *flattened* grid observation; the
source paper reports that architectures without a convolutional stem "did not learn
good policies" on Overcooked-v2. This mirrors ``rnn_actor_critic.py`` exactly --
same ``ScannedRNN``, same actor/critic head construction, same conditional-critic
variant for BRDiv/L-BRDiv -- with a CNN encoder swapped in for the input ``Dense``.

Encoder (App. C.1.1): reshape the flat obs to ``(width, height, channels)``, three
1x1 convs ``[128, 128, 8]`` then three 3x3 convs ``[16, 32, 32]`` (zero-pad, ReLU),
flatten, ``Dense(fc_hidden_dim)``, ``LayerNorm``. The encoder runs per timestep, its
output feeds the GRU, and the GRU embedding feeds the actor/critic heads.
"""

from collections.abc import Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from oaht_bench.models.rnn_actor_critic import ScannedRNN


class CNNEncoder(nn.Module):
    """Flat grid obs -> (fc_hidden_dim,) feature vector, per timestep.

    ``obs_shape`` is the unflattened ``(width, height, channels)`` the Overcooked-v2
    wrapper flattens away (``overcooked_v2_wrapper.py`` returns ``obs.flatten()``).
    Leading dims (seq_len, batch) are collapsed before the conv stack and restored
    after, so the encoder is agnostic to how many batch axes the caller carries.
    """

    obs_shape: Sequence[int]
    fc_hidden_dim: int = 128
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        lead = obs.shape[:-1]
        x = obs.reshape((-1, *self.obs_shape))  # (seq*batch, W, H, C)

        for features in (128, 128, 8):
            x = nn.Conv(
                features,
                kernel_size=(1, 1),
                padding="SAME",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)
        for features in (16, 32, 32):
            x = nn.Conv(
                features,
                kernel_size=(3, 3),
                padding="SAME",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))  # flatten spatial + channels
        x = nn.Dense(
            self.fc_hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = nn.LayerNorm()(x)
        x = activation(x)
        return x.reshape((*lead, self.fc_hidden_dim))


class CNNRNNActorCritic(nn.Module):
    """``RNNActorCritic`` with a ``CNNEncoder`` in place of the input ``Dense``.

    Shared CNN+GRU trunk, then an actor head (masked ``distrax.Categorical``) and a
    critic head, exactly as in ``RNNActorCritic``.
    """

    action_dim: Sequence[int]
    obs_shape: Sequence[int]
    fc_hidden_dim: int = 128
    gru_hidden_dim: int = 128
    activation: str = "relu"

    @nn.compact
    def __call__(self, hidden, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        obs, dones, avail_actions = x

        embedding = CNNEncoder(self.obs_shape, self.fc_hidden_dim, self.activation)(obs)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.gru_hidden_dim, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(self.fc_hidden_dim, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class CNNRNNActorWithConditionalCritic(nn.Module):
    """CNN+GRU actor with a teammate-id-conditioned critic.

    Mirrors ``RNNActorWithConditionalCritic``: the actor is recurrent (CNN+GRU), the
    critic is a non-recurrent MLP conditioned on ``teammate_id``. Here the critic
    reads the shared CNN features (not the raw flat grid the plain-RNN variant feeds
    its critic), so the value function sees the same convolutional representation the
    actor does -- the CNN is what makes the grid legible, and there is no reason to
    withhold it from the critic.
    """

    action_dim: Sequence[int]
    obs_shape: Sequence[int]
    fc_hidden_dim: int = 128
    gru_hidden_dim: int = 128
    activation: str = "relu"

    @nn.compact
    def __call__(self, hidden, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        obs, teammate_id, dones, avail_actions = x

        cnn_feat = CNNEncoder(self.obs_shape, self.fc_hidden_dim, self.activation)(obs)

        rnn_in = (cnn_feat, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.gru_hidden_dim, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        # Critic: non-recurrent MLP off the shared CNN features + teammate id.
        feat_with_teammate_id = jnp.concatenate([cnn_feat, teammate_id], axis=-1)
        critic = nn.Dense(
            self.fc_hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(feat_with_teammate_id)
        critic = activation(critic)
        critic = nn.Dense(
            self.fc_hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)
