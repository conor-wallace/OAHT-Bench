"""BC's model — architecture and inference only, the "no modeling module" floor.

The shared backbone trained directly on the ego stream, with no teammate module of
any kind, and :class:`BcAgent`, the inference wrapper. Given trained parameters,
``BcAgent`` acts identically no matter how they were produced, so it is model-layer
and carries no dataset or training dependency. The (single-stage) training, the loss,
and the optional return filter live in :mod:`oaht_bench.offline.bc`.
"""

from __future__ import annotations

import flax.linen as nn

from oaht_bench.models.backbone import DecisionTransformer
from oaht_bench.models.return_conditioned_agent import ReturnConditionedAgent


class BcNetwork(nn.Module):
    """The shared backbone, unmodified: no embedding, no context, no
    conditioning on anything about the teammate."""

    action_dim: int
    hidden_dim: int = 32
    dropout: float = 0.1

    @nn.compact
    def __call__(self, rtg, obs, actions, *, timesteps, mask=None, train: bool = False):
        logits, _ = DecisionTransformer(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            use_cross_attention=False,
            dropout=self.dropout,
        )(rtg, obs, actions, timesteps=timesteps, mask=mask, train=train)
        return logits


class BcAgent(ReturnConditionedAgent):
    """BC's architecture and inference as a :class:`ReturnConditionedAgent`.

    The base owns the rolling ego-window / return-to-go deployment; BC supplies only
    the backbone and a forward that runs it directly on the ego window -- no embedding,
    no context. The training and loss live in :mod:`oaht_bench.offline.bc`, which
    composes one of these.
    """

    def build_model(self) -> None:
        net = self.config.network
        if net.obs_dim is None or net.action_dim is None:
            raise ValueError(
                "obs_dim/action_dim are unresolved on the network config; the "
                "runner must resolve them from the dataset before build_model()."
            )
        self.network = BcNetwork(
            action_dim=net.action_dim, hidden_dim=net.hidden_dim, dropout=net.dropout
        )

    def act(self, params, rtg, obs, actions, *, timesteps, mask):
        return self.network.apply(
            params["stage2"], rtg, obs, actions, timesteps=timesteps, mask=mask, train=False
        )
