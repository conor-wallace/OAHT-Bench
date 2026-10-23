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
    num_layers: int = Field(
        default=2,
        gt=0,
        description="Hidden Dense+activation layers before the output layer, "
        "mirrored on actor and critic. 2 (the default) matches the absorbed "
        "code's prior fixed depth; references vary (e.g. OMIS's own PPO net "
        "for LBF is 5 hidden layers wide at hidden_dim -- confirmed directly "
        "from OMIS/pretraining/ppo.py's PolicyNet/ValueNet, not the paper's "
        "own ambiguous table).",
    )
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
            "NUM_LAYERS": self.num_layers,
        }
        if self.policy_input_dim is not None:
            out["POLICY_INPUT_DIM"] = self.policy_input_dim
        return out


class BaseOfflineAhtNetworkConfig(BaseConfig):
    ff_dim: int = Field(default=128, gt=0)
    hidden_dim: int = Field(default=32, gt=0)
    num_blocks: int = Field(default=3, gt=0)

    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)

    ff_activation: Literal["relu", "gelu"] = Field(
        default="relu",
        description="Activation inside the transformer block's feed-forward "
        "sublayer (Dense(ff_dim) -> activation -> Dense(hidden_dim)). OMIS's "
        "own GPT2 backbone (App. F, confirmed against OMIS/pretraining/"
        "pretrain.py's GPT2Config(activation_function='gelu_new')) uses GELU. "
        "Default stays 'relu' -- this backbone is shared by every offline "
        "baseline (BC/LIAM/MeLIBA/TAO/OMIS), and changing the default would "
        "silently change already-validated results for baselines other than "
        "OMIS. An OMIS-fidelity config sets this to 'gelu' explicitly.",
    )
    obs_embed_layers: int = Field(
        default=1,
        gt=0,
        description="Dense layers building the observation/state token (o_tok), "
        "with obs_embed_activation applied after every layer except the last. "
        "OMIS's embed_state (App. F; confirmed against OMIS/pretraining/nets.py) "
        "is a 3-layer MLP with LeakyReLU -- unlike its action/RTG embeddings, "
        "which are a single Dense with no activation (already our default for "
        "those two, unchanged). Default (1 layer) preserves today's single-"
        "Dense, no-activation o_tok for every baseline; an OMIS-fidelity config "
        "sets this to 3.",
    )
    obs_embed_activation: Literal["none", "leaky_relu"] = Field(
        default="none",
        description="Activation applied after every obs_embed_layers Dense "
        "layer except the last. Paired with obs_embed_layers -- see its "
        "description. 'none' (default) matches today's unactivated o_tok.",
    )

    # Resolved from the dataset before a policy is built (see
    # ``offline.runner`` and ``BaseAhtTrainer``), so a policy is pure-config. Left
    # ``None`` in an authored config -- the dataset path already in the hash
    # determines them, so they are a derived convenience, not a tuning knob.
    obs_dim: int | None = Field(default=None, ge=1)
    action_dim: int | None = Field(default=None, ge=1)


class LiamNetworkConfig(BaseOfflineAhtNetworkConfig):
    architecture: Literal["liam"] = "liam"


class MelibaNetworkConfig(BaseOfflineAhtNetworkConfig):
    architecture: Literal["meliba"] = "meliba"


class OmisNetworkConfig(BaseOfflineAhtNetworkConfig):
    architecture: Literal["omis"] = "omis"


class TaoNetworkConfig(BaseOfflineAhtNetworkConfig):
    architecture: Literal["tao"] = "tao"


class BcNetworkConfig(BaseOfflineAhtNetworkConfig):
    """BC: the shared backbone, no modeling module — the floor every other
    trajectory-view baseline is measured against (§6)."""

    architecture: Literal["bc"] = "bc"
    top_return_quantile: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Train only on episodes in the top fraction by ego return "
        "(e.g. 0.1 keeps the best 10%). 1.0 is plain BC on the "
        "whole dataset, and the default: filtering is this baseline's one "
        "option, not its definition. A data-selection knob rather than a "
        "network shape one, but this is where every other baseline's own "
        "knobs live, so it stays here rather than on the shared "
        "OfflineTrainingConfig every baseline reads.",
    )
