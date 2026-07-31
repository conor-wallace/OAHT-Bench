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

from typing import Annotated, Any, Literal, Mapping

from pydantic import Field, model_validator
from types import MappingProxyType

from oaht_bench.configs.base import VersionedConfig

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

    def task_config(self) -> dict[str, Any]:
        """The ``task`` block jax-aht's runners expect.

        Mirrors ``teammate_generation/configs/task/*.yaml``. We build it rather
        than reading their YAML so there is one source of truth for what
        ``lbf_12x12`` means.
        """
        return {
            "ENV_NAME": self.env_name,
            "ENV_KWARGS": self.env_kwargs(),
            "ROLLOUT_LENGTH": self.rollout_length,
            "TASK_NAME": self.name,
        }


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


class OvercookedV1Config(EnvConfigBase):
    """Overcooked-v1 (JaxMARL), selected by layout."""

    env_name: Literal["overcooked-v1"] = "overcooked-v1"
    layout: OvercookedLayout

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
        return {"layout": self.layout}


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
    LbfConfig | OvercookedV1Config | HanabiConfig,
    Field(discriminator="env_name"),
]


# --------------------------------------------------------------------------
# Canonical presets
#
# Experiment JSON files inline a full env config, so these are the canonical
# definitions used to construct and to test them -- not an indirection layer that
# experiment files reference by name.
# --------------------------------------------------------------------------

_PRESETS: dict[str, LbfConfig | OvercookedV1Config | HanabiConfig] = {}


def _register(cfg: LbfConfig | OvercookedV1Config | HanabiConfig):
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

#: Read-only view of the canonical presets.
PRESETS: Mapping[str, LbfConfig | OvercookedV1Config | HanabiConfig] = MappingProxyType(_PRESETS)


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
