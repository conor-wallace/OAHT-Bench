"""Policy network architecture, made explicit.

The absorbed initializers read these with ``dict.get(key, default)`` — an MLP
policy silently becomes ``tanh`` with a 64-unit hidden layer if nothing says
otherwise. That means the architecture never appeared in any config file and so
never entered a run's content hash: two runs recorded as identical could differ
if a default moved.

Declaring them here puts the architecture in the JSON, in the hash, and in the
released artifact's provenance.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from oaht_bench.configs.base import BaseConfig

Activation = Literal["tanh", "relu"]


class MlpNetwork(BaseConfig):
    """Feed-forward actor-critic. Defaults match the absorbed code's fallbacks."""

    architecture: Literal["mlp"] = "mlp"
    activation: Activation = "tanh"
    hidden_dim: int = Field(default=64, gt=0)
    policy_input_dim: int | None = Field(
        default=None,
        description="Override the observation width the policy expects. None "
        "means use the environment's observation space.",
    )

    def to_agent_dict(self) -> dict[str, Any]:
        """The keys the absorbed agent initializers read.

        Confined to this one method: the training loop itself is typed, and only
        the handoff into ``initialize_agents`` still speaks dict.
        """
        out: dict[str, Any] = {
            "ACTIVATION": self.activation,
            "FC_HIDDEN_DIM": self.hidden_dim,
        }
        if self.policy_input_dim is not None:
            out["POLICY_INPUT_DIM"] = self.policy_input_dim
        return out
