"""Environment configs as a discriminated union, one model per environment family.

Replaces an untyped ``env_kwargs: dict[str, Any]``. The point is that LBF's
parameters and Hanabi's parameters are structurally different things, and a dict
cannot say so — nothing stops you writing ``num_colors`` into an LBF config and
having it ignored, or omitting a parameter whose default silently changes the
environment.

Two properties are *derived* rather than declared. ``turn_based`` and
``symmetric_roles`` are facts about an environment family, not choices a config
makes, so they are computed from the model type. An earlier version had them as
hand-set booleans, which meant they could disagree with the environment they
described.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from oaht_bench.configs.base import BaseConfig, VersionedConfig

Tier = Literal["tier1", "tier2", "debug"]

#: Overcooked-v1 layouts where the two agents do not share role semantics.
#: Trajectory mirroring (§4.5) is invalid for these.
_ASYMMETRIC_OVERCOOKED_LAYOUTS = frozenset({"asymm_advantages", "forced_coord"})

OvercookedLayout = Literal[
    "counter_circuit",
    "coord_ring",
    "cramped_room",
    "asymm_advantages",
    "forced_coord",
]

#: The five layout names v2 ships that share v1's names and core scenario --
#: deliberately not the full v2 roster (23 layouts, including recipe-variant
#: and demo layouts), so the first pass stays directly comparable to v1
#: rather than opening a second, larger layout-selection question at the same
#: time as the environment integration itself.
OvercookedV2Layout = Literal[
    "counter_circuit",
    "coord_ring",
    "cramped_room",
    "asymm_advantages",
    "forced_coord",
]

#: Same two layouts as v1's _ASYMMETRIC_OVERCOOKED_LAYOUTS -- not verified to
#: mean the same thing in v2 beyond sharing the name and general shape; worth
#: a real check before symmetric_roles is load-bearing for v2 (§4.5).
_ASYMMETRIC_OVERCOOKED_V2_LAYOUTS = frozenset({"asymm_advantages", "forced_coord"})


class EnvConfigBase(VersionedConfig):
    """Fields shared by every environment configuration."""

    name: str = Field(
        description="Registry name. Recorded in dataset metadata; must identify "
        "exactly one configuration."
    )
    tier: Tier = Field(
        default="tier2",
        description="Matrix density this configuration carries (§10.3). 'debug' "
        "never appears in results.",
    )
    rollout_length: int = Field(
        gt=0,
        description="Episode length used by jax-aht's runners. Shapes every "
        "collected dataset, so it belongs in the config rather than a default.",
    )
    notes: str = Field(default="", description="Why this configuration is in the benchmark.")

    # --- Derived environment properties -----------------------------------

    @property
    def turn_based(self) -> bool:
        """Whether agents act in alternation rather than simultaneously.

        Drives ``acting_agent`` in the dataset schema and episode segmentation in
        the learning-history view (§4.2).
        """
        raise NotImplementedError

    @property
    def symmetric_roles(self) -> bool:
        """Whether all agents share observation and action semantics (§4.5)."""
        return True

    # --- Projections to jax-aht -------------------------------------------

    def env_kwargs(self) -> dict[str, Any]:
        """Kwargs for ``jax_aht.envs.make_env``.

        Only keys the environment actually consumes. jax-aht passes unrecognised
        keys through to the underlying Jumanji/JaxMARL constructor, which rejects
        them, so this must be exact.
        """
        raise NotImplementedError


class LbfConfig(EnvConfigBase):
    """Level-Based Foraging (Jumanji), via jax-aht's wrapper."""

    env_name: Literal["lbf"] = "lbf"
    grid_size: int = Field(ge=5, description="Jumanji requires grid_size >= 5.")
    num_food: int = Field(gt=0)
    different_levels: bool = True
    num_agents: int = Field(default=2, gt=0)
    fov: int | None = Field(
        default=None,
        description="Field of view. None means full observability (jax-aht "
        "defaults fov to grid_size).",
    )

    @property
    def turn_based(self) -> bool:
        return False

    @model_validator(mode="after")
    def _food_must_fit(self) -> LbfConfig:
        """Enforce Jumanji's placement constraint at config load, not at reset().

        ``jumanji/environments/routing/lbf/generator.py`` asserts
        ``(grid_size - 2)**2 - num_agents > num_food * 5``, warning that otherwise
        "food will be incorrectly placed due to JAX's silent error handling". The
        native failure is an opaque AssertionError raised from inside Jumanji on
        the first reset; here it names the fields that conflict.
        """
        capacity = (self.grid_size - 2) ** 2 - self.num_agents
        required = self.num_food * 5
        if capacity <= required:
            raise ValueError(
                f"{self.name!r}: grid too small for this much food. Jumanji requires "
                f"(grid_size-2)^2 - num_agents > num_food*5, i.e. {capacity} > {required}. "
                f"Reduce num_food below {capacity // 5} or raise grid_size."
            )
        if self.fov is not None and not (1 <= self.fov <= self.grid_size):
            raise ValueError(
                f"{self.name!r}: fov must be between 1 and grid_size "
                f"({self.grid_size}), got {self.fov}."
            )
        return self

    def env_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "grid_size": self.grid_size,
            "num_food": self.num_food,
            "different_levels": self.different_levels,
            "num_agents": self.num_agents,
        }
        if self.fov is not None:
            kwargs["fov"] = self.fov
        return kwargs


class RewardShapingParams(BaseConfig):
    """Dense shaping bonuses layered on Overcooked's sparse delivery reward.

    Values are jax-aht's. Without them the task is sparse-reward and materially
    harder, so a population trained with shaping and one trained without are not
    comparable -- which is why these are explicit fields in the run's hash rather
    than defaults inside the environment.
    """

    PLACEMENT_IN_POT_REW: float = 0.5
    PLATE_PICKUP_REWARD: float = 0.1
    SOUP_PICKUP_REWARD: float = 1.0
    ONION_PICKUP_REWARD: float = 0.1
    COUNTER_PICKUP_REWARD: float = 0.0
    COUNTER_DROP_REWARD: float = 0.0


class OvercookedV1Config(EnvConfigBase):
    """Overcooked-v1 (JaxMARL), selected by layout."""

    env_name: Literal["overcooked-v1"] = "overcooked-v1"
    layout: OvercookedLayout
    random_obj_state: bool = Field(
        default=True,
        description="Randomize initial object state (pots part-filled, agents "
        "holding items). jax-aht's task configs enable this; the environment "
        "itself defaults to False.",
    )
    do_reward_shaping: bool = Field(
        default=True,
        description="Add dense shaping to the sparse delivery reward. The "
        "environment defaults this to False, so omitting it trains a different "
        "-- and much harder -- task than jax-aht's configs describe.",
    )
    reward_shaping: RewardShapingParams = Field(default_factory=RewardShapingParams)

    @property
    def turn_based(self) -> bool:
        return False

    @property
    def symmetric_roles(self) -> bool:
        """Derived from the layout rather than declared.

        Several layouts give the two agents structurally different roles, which
        makes trajectory mirroring (§4.5) invalid for them.
        """
        return self.layout not in _ASYMMETRIC_OVERCOOKED_LAYOUTS

    def env_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "layout": self.layout,
            "random_obj_state": self.random_obj_state,
            "do_reward_shaping": self.do_reward_shaping,
        }
        if self.do_reward_shaping:
            kwargs["reward_shaping_params"] = self.reward_shaping.model_dump()
        return kwargs


class OvercookedV2Config(EnvConfigBase):
    """Overcooked-v2 (JaxMARL, absorbed -- PROVENANCE.md), selected by layout.

    Covers the constructor args most likely to matter for teammate generation
    and dataset collection. Deliberately excludes ``observation_type``
    (FEATURIZED is a different obs encoding, out of scope for the first
    pass), ``start_cooking_interaction``, ``op_ingredient_permutations``,
    ``initial_state_buffer``, and ``force_path_planning`` -- all real
    ``OvercookedV2.__init__`` parameters, left at their environment defaults
    rather than exposed here until something needs them.
    """

    env_name: Literal["overcooked-v2"] = "overcooked-v2"
    layout: OvercookedV2Layout
    agent_view_size: int | None = Field(
        default=None,
        description="Partial observability radius; None is full-grid "
        "observability. The registered presets set this to 2, matching "
        "upstream's only validated reference config (baselines/IPPO/config/"
        "ippo_rnn_overcooked_v2.yaml) -- this is v2's headline new feature "
        "(README: 'configurable agent view radius') and the reason to use "
        "v2 at all rather than v1. Left settable to None here, not removed, "
        "since a full-observability run is still a legitimate comparison "
        "point later. Requires actor_type='rnn' or similar to be useful -- "
        "an MLP cannot make good use of a partial observation; see "
        "docs/tuning_record.md.",
    )
    random_reset: bool = Field(
        default=True,
        description="Randomize agent positions, inventories, and pot state on "
        "reset. v1's random_obj_state does the inventory/pot half of this; v2 "
        "folds both into one flag. Defaulted on to match v1's convention "
        "(random_obj_state=True in jax-aht's task configs) rather than the "
        "environment's own default (False).",
    )
    random_agent_positions: bool = Field(
        default=False,
        description="Randomize agent start positions independently of "
        "random_reset (v2-native; v1 has no equivalent). Off by default: "
        "changes the coordination problem itself, not just initial state "
        "diversity, and should be an explicit choice, not a silent default.",
    )
    negative_rewards: bool = Field(
        default=False,
        description="Penalize incorrect deliveries. Off by default, matching "
        "v1's sparse-plus-shaping reward having no penalty term either.",
    )
    sample_recipe_on_delivery: bool = Field(
        default=False,
        description="Resample the target recipe after each delivery instead "
        "of only on reset. Off by default: a fixed recipe per episode is the "
        "closer analogue to v1, which has exactly one recipe.",
    )
    indicate_successful_delivery: bool = Field(
        default=False,
        description="Add a successful-delivery indicator to the observation.",
    )

    @property
    def turn_based(self) -> bool:
        return False

    @property
    def symmetric_roles(self) -> bool:
        """See _ASYMMETRIC_OVERCOOKED_V2_LAYOUTS -- inherited from v1's
        layout list by name, not independently verified for v2 (§4.5)."""
        return self.layout not in _ASYMMETRIC_OVERCOOKED_V2_LAYOUTS

    def env_kwargs(self) -> dict[str, Any]:
        # max_steps deliberately not derived from rollout_length -- neither
        # LBF's nor Hanabi's env_kwargs() do that either. rollout_length is the
        # training loop's horizon, not an environment kwarg; it isn't passed to
        # the environment constructor for any environment family here.
        # OvercookedV2's own max_steps default (400) matches v1's effective
        # one, so it's left unset.
        kwargs: dict[str, Any] = {
            "layout": self.layout,
            "random_reset": self.random_reset,
            "random_agent_positions": self.random_agent_positions,
            "negative_rewards": self.negative_rewards,
            "sample_recipe_on_delivery": self.sample_recipe_on_delivery,
            "indicate_successful_delivery": self.indicate_successful_delivery,
        }
        if self.agent_view_size is not None:
            kwargs["agent_view_size"] = self.agent_view_size
        return kwargs


class HanabiConfig(EnvConfigBase):
    """Hanabi (JaxMARL), via jax-aht's wrapper. Turn-based and action-masked."""

    env_name: Literal["hanabi"] = "hanabi"
    num_agents: int = Field(default=2, ge=2)
    num_colors: int = Field(gt=0)
    num_ranks: int = Field(gt=0)
    hand_size: int = Field(gt=0)
    max_info_tokens: int = Field(gt=0)
    max_life_tokens: int = Field(gt=0)
    num_cards_of_rank: tuple[int, ...] = Field(
        description="Deck composition: copies of each rank, lowest rank first."
    )

    @property
    def turn_based(self) -> bool:
        return True

    @model_validator(mode="after")
    def _deck_must_be_consistent(self) -> HanabiConfig:
        if len(self.num_cards_of_rank) != self.num_ranks:
            raise ValueError(
                f"{self.name!r}: num_cards_of_rank has {len(self.num_cards_of_rank)} "
                f"entries but num_ranks is {self.num_ranks}; they must match."
            )
        if any(c <= 0 for c in self.num_cards_of_rank):
            raise ValueError(f"{self.name!r}: every entry of num_cards_of_rank must be positive.")
        deck_size = self.num_colors * sum(self.num_cards_of_rank)
        dealt = self.num_agents * self.hand_size
        if dealt >= deck_size:
            raise ValueError(
                f"{self.name!r}: dealing {dealt} cards from a {deck_size}-card deck "
                f"leaves nothing to draw."
            )
        return self

    def env_kwargs(self) -> dict[str, Any]:
        return {
            "num_agents": self.num_agents,
            "num_colors": self.num_colors,
            "num_ranks": self.num_ranks,
            "hand_size": self.hand_size,
            "max_info_tokens": self.max_info_tokens,
            "max_life_tokens": self.max_life_tokens,
            "num_cards_of_rank": list(self.num_cards_of_rank),
        }


#: Discriminated union. Pydantic selects the member by ``env_name``, so a JSON
#: config naming an environment gets that environment's validation rules.
EnvConfig = Annotated[
    LbfConfig | OvercookedV1Config | OvercookedV2Config | HanabiConfig,
    Field(discriminator="env_name"),
]


# --------------------------------------------------------------------------
# Canonical presets
#
# Experiment JSON files inline a full env config, so these are the canonical
# definitions used to construct and to test them -- not an indirection layer that
# experiment files reference by name.
# --------------------------------------------------------------------------

_PRESETS: dict[str, LbfConfig | OvercookedV1Config | OvercookedV2Config | HanabiConfig] = {}


def _register(cfg: LbfConfig | OvercookedV1Config | OvercookedV2Config | HanabiConfig):
    if cfg.name in _PRESETS:
        raise ValueError(
            f"Duplicate env preset {cfg.name!r}. Names appear in dataset metadata "
            f"and must identify exactly one configuration."
        )
    _PRESETS[cfg.name] = cfg
    return cfg


LBF_12X12 = _register(
    LbfConfig(
        name="lbf_12x12",
        grid_size=12,
        num_food=6,
        different_levels=True,
        rollout_length=128,
        tier="tier1",
        notes="Gridworld. Matches the existing checkpoints/lbf/lbf_12x12 populations.",
    )
)

HANABI = _register(
    HanabiConfig(
        name="hanabi",
        num_colors=5,
        num_ranks=5,
        hand_size=5,
        max_info_tokens=8,
        max_life_tokens=3,
        num_cards_of_rank=(3, 2, 2, 2, 1),
        rollout_length=128,
        tier="tier1",
        notes="Turn-based, action-masked, hidden own hand. The abstraction stress test.",
    )
)

MINI_HANABI = _register(
    HanabiConfig(
        name="mini_hanabi",
        num_colors=3,
        num_ranks=3,
        hand_size=3,
        max_info_tokens=5,
        max_life_tokens=3,
        num_cards_of_rank=(2, 2, 1),
        rollout_length=128,
        tier="debug",
        notes="Development/debug configuration only; never appears in results.",
    )
)

_OVERCOOKED_TIERS: dict[str, tuple[Tier, str]] = {
    "counter_circuit": ("tier1", "Full resource-sharing; discriminates between methods."),
    "coord_ring": ("tier2", "Full resource-sharing; Tier 1 fallback."),
    "cramped_room": ("tier2", "Standard easy reference layout."),
    "asymm_advantages": ("tier2", "ZSC-Eval: fails to differentiate algorithms."),
    "forced_coord": ("tier2", "ZSC-Eval: fails to differentiate algorithms."),
}

for _layout, (_tier, _notes) in _OVERCOOKED_TIERS.items():
    _register(
        OvercookedV1Config(
            name=f"overcooked_{_layout}",
            layout=_layout,  # type: ignore[arg-type]
            rollout_length=400,
            tier=_tier,
            notes=_notes,
        )
    )
del _layout, _tier, _notes

#: Same tier assignment as v1's -- not re-derived from v2-specific evidence
#: (no populations trained on any v2 layout yet), just carried over so the
#: two versions are comparable on the same layout at the same tier until
#: there's a reason to diverge.
_OVERCOOKED_V2_TIERS: dict[str, tuple[Tier, str]] = {
    "counter_circuit": ("tier1", "Full resource-sharing; discriminates between methods in v1."),
    "coord_ring": ("tier2", "Full resource-sharing; Tier 1 fallback."),
    "cramped_room": ("tier2", "Standard easy reference layout."),
    "asymm_advantages": ("tier2", "ZSC-Eval: fails to differentiate algorithms in v1."),
    "forced_coord": ("tier2", "ZSC-Eval: fails to differentiate algorithms in v1."),
}

for _layout, (_tier, _notes) in _OVERCOOKED_V2_TIERS.items():
    _register(
        OvercookedV2Config(
            name=f"overcooked_v2_{_layout}",
            layout=_layout,  # type: ignore[arg-type]
            rollout_length=400,
            tier=_tier,
            notes=_notes,
            # Partial observability is v2's headline feature over v1 and the
            # reason to use it -- agent_view_size=2 matches upstream's only
            # validated reference config. Requires an RNN policy
            # (actor_type='rnn' on the generator config) to be meaningful.
            agent_view_size=2,
        )
    )
del _layout, _tier, _notes

#: Read-only view of the canonical presets.
PRESETS: Mapping[str, LbfConfig | OvercookedV1Config | OvercookedV2Config | HanabiConfig] = MappingProxyType(_PRESETS)


def get_preset(name: str):
    """Look up a canonical environment preset by name."""
    try:
        return _PRESETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown environment preset {name!r}. Available: {sorted(_PRESETS)}"
        ) from None


def preset_names(tier: Tier | None = None) -> list[str]:
    """Preset names, optionally filtered to one tier."""
    names = sorted(_PRESETS)
    if tier is None:
        return names
    return [n for n in names if _PRESETS[n].tier == tier]
