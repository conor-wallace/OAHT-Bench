"""The seven environment configurations of OAHT-Bench, as one addressable table.

Every experiment in the benchmark names a configuration from this registry rather
than passing ``env_kwargs`` around. That is what makes a run reproducible from a
config file (project plan §10.2, Phase 0a exit criterion) and what lets dataset
metadata record exactly which environment produced it (§4.2).

Environments come from ``jax-aht`` (MIT, UT Austin LARG), which already exposes a
uniform interface across all three: ``reset``, ``step``, ``get_avail_actions``,
and ``agents``. This module adds the benchmark's *choice* of configurations and
the tier assignment, not new environment code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Tier = Literal["tier1", "tier2", "debug"]


@dataclass(frozen=True)
class EnvConfig:
    """One named environment configuration.

    Attributes:
        name: Registry key. Stable; recorded in dataset metadata.
        env_name: The identifier ``jax_aht.envs.make_env`` dispatches on.
        env_kwargs: Arguments to ``make_env``. Recorded verbatim in metadata.
        tier: Matrix density this configuration carries (§10.3). ``tier1`` gets the
            full baseline x generator x split x variant cross product; ``tier2`` gets
            a reduced one; ``debug`` never appears in results.
        turn_based: Whether agents act in alternation rather than simultaneously.
            Drives ``acting_agent`` in the schema and episode segmentation in the
            learning-history view (§4.2).
        symmetric_roles: Whether all agents share observation and action semantics.
            Trajectory mirroring is only valid when this holds (§4.5).
        notes: Why this configuration is in the benchmark.
    """

    name: str
    env_name: str
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    tier: Tier = "tier2"
    turn_based: bool = False
    symmetric_roles: bool = True
    notes: str = ""


#: LBF at the configuration the existing bayes-tom populations were trained against
#: (§7.5), so Phase 0 data collection needs no retraining.
_LBF_12x12 = EnvConfig(
    name="lbf_12x12",
    env_name="lbf",
    env_kwargs={
        "num_agents": 2,
        "grid_size": 12,
        "fov": 12,
        "num_food": 6,
        "different_levels": True,
    },
    tier="tier1",
    notes="Gridworld. Matches existing checkpoints/lbf/lbf_12x12 populations.",
)

#: Overcooked-v1 (JaxMARL). Tier 1 is counter_circuit rather than cramped_room:
#: ZSC-Eval reports that the simpler layouts fail to differentiate algorithms, and
#: the "full resource-sharing" layouts discriminate better (§10.3).
_OVERCOOKED_LAYOUTS: dict[str, tuple[Tier, bool, str]] = {
    "counter_circuit": ("tier1", True, "Full resource-sharing; discriminates between methods."),
    "coord_ring": ("tier2", True, "Full resource-sharing; Tier 1 fallback."),
    "cramped_room": ("tier2", True, "Standard easy reference layout."),
    "asymm_advantages": ("tier2", False, "ZSC-Eval: fails to differentiate algorithms."),
    "forced_coord": ("tier2", False, "ZSC-Eval: fails to differentiate algorithms."),
}

#: Full Hanabi. Turn-based with legal-action masking and hidden own-hand — the
#: configuration that imposes the strictest requirements on every interface (§11).
_HANABI = EnvConfig(
    name="hanabi",
    env_name="hanabi",
    env_kwargs={
        "num_agents": 2,
        "num_colors": 5,
        "num_ranks": 5,
        "hand_size": 5,
        "max_info_tokens": 8,
        "max_life_tokens": 3,
        "num_cards_of_rank": [3, 2, 2, 2, 1],
    },
    tier="tier1",
    turn_based=True,
    notes="Turn-based, action-masked, hidden own hand. The abstraction stress test.",
)

#: Reduced Hanabi for fast iteration. Never appears in results (§12.5).
_MINI_HANABI = EnvConfig(
    name="mini_hanabi",
    env_name="hanabi",
    env_kwargs={
        "num_agents": 2,
        "num_colors": 3,
        "num_ranks": 3,
        "hand_size": 3,
        "max_info_tokens": 5,
        "max_life_tokens": 3,
        "num_cards_of_rank": [2, 2, 1],
    },
    tier="debug",
    turn_based=True,
    notes="Development/debug configuration only.",
)


def _build_registry() -> dict[str, EnvConfig]:
    configs = [_LBF_12x12, _HANABI, _MINI_HANABI]
    for layout, (tier, symmetric, notes) in _OVERCOOKED_LAYOUTS.items():
        configs.append(
            EnvConfig(
                name=f"overcooked_{layout}",
                env_name="overcooked-v1",
                env_kwargs={"layout": layout},
                tier=tier,
                symmetric_roles=symmetric,
                notes=notes,
            )
        )
    return {c.name: c for c in configs}


REGISTRY: dict[str, EnvConfig] = _build_registry()


def get_config(name: str) -> EnvConfig:
    """Look up a configuration by registry name."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown environment config {name!r}. Available: {sorted(REGISTRY)}"
        ) from None


def config_names(tier: Tier | None = None) -> list[str]:
    """Registry names, optionally filtered to one tier.

    Passing ``None`` returns every configuration including ``debug``; results code
    should filter to the tiers it reports on rather than relying on this default.
    """
    names = sorted(REGISTRY)
    if tier is None:
        return names
    return [n for n in names if REGISTRY[n].tier == tier]


def make(name: str):
    """Instantiate the environment for a registry configuration.

    Imports ``jax_aht`` lazily so that importing the registry — to read metadata,
    enumerate configurations, or build docs — does not require JAX.
    """
    from envs import make_env  # jax-aht, top-level module

    cfg = get_config(name)
    return make_env(cfg.env_name, dict(cfg.env_kwargs))
