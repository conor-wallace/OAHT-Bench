"""Phase 0a exit criterion: every registered environment instantiates and rolls out.

The project plan (§10.2) makes this the first gate deliberately — if the seven
configurations cannot be driven through a uniform interface, every later phase
inherits the defect.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from oaht_bench.envs.registry import (
    REGISTRY,
    EnvConfig,
    config_names,
    get_config,
    make,
    register,
)

ALL_CONFIGS = config_names()


def test_declaring_a_config_registers_it():
    """Registration is co-located with declaration, not a hand-maintained list.

    The module-level constants and the registry must not be able to disagree.
    """
    from oaht_bench.envs import registry as mod

    declared = {
        v.name
        for k, v in vars(mod).items()
        if isinstance(v, EnvConfig) and not k.startswith("_")
    }
    assert declared <= set(REGISTRY), "a declared config is missing from the registry"


def test_duplicate_names_are_rejected():
    """Names identify datasets (§4.2); two configs sharing one makes data ambiguous."""
    with pytest.raises(ValueError, match="Duplicate"):
        register(EnvConfig(name="hanabi", env_name="hanabi"))


def test_registry_is_not_mutable_in_place():
    """Callers must go through register(), which enforces the duplicate check."""
    with pytest.raises(TypeError):
        REGISTRY["injected"] = EnvConfig(name="injected", env_name="lbf")  # type: ignore[index]


def test_configs_compare_by_value():
    """Value equality lets us ask whether a dataset's recorded config still matches."""
    a = EnvConfig(name="x", env_name="lbf", env_kwargs={"grid_size": 12})
    b = EnvConfig(name="x", env_name="lbf", env_kwargs={"grid_size": 12})
    assert a == b


def test_registry_covers_the_committed_scope():
    """Seven results configurations plus one debug configuration (§12.1)."""
    assert set(config_names("tier1")) == {"lbf_12x12", "overcooked_counter_circuit", "hanabi"}
    assert len(config_names("tier2")) == 4  # the remaining Overcooked layouts
    assert config_names("debug") == ["mini_hanabi"]


def test_tier1_overcooked_is_not_a_non_discriminative_layout():
    """ZSC-Eval reports forced_coord and asymm_advantages fail to separate methods."""
    tier1_overcooked = [n for n in config_names("tier1") if n.startswith("overcooked")]
    assert tier1_overcooked == ["overcooked_counter_circuit"]


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_env_resets_and_steps(name: str):
    """Each configuration resets, exposes per-agent observations, and steps."""
    env = make(name)
    rng = jax.random.PRNGKey(0)

    obs, state = env.reset(rng)
    agents = list(env.agents)
    assert len(agents) >= 2
    assert set(obs) >= set(agents), f"{name}: obs missing agents"
    for agent in agents:
        assert jnp.asarray(obs[agent]).ndim == 1, f"{name}: expected flat observation"

    avail = env.get_avail_actions(state)
    assert set(avail) >= set(agents), f"{name}: avail_actions missing agents"

    # Act legally: Hanabi rejects illegal actions, so sample from the mask rather
    # than from the full action space.
    rng, step_rng = jax.random.split(rng)
    actions = {}
    for i, agent in enumerate(agents):
        mask = jnp.asarray(avail[agent])
        logits = jnp.where(mask > 0, 0.0, -jnp.inf)
        actions[agent] = jax.random.categorical(jax.random.fold_in(step_rng, i), logits)

    obs2, _, rewards, dones, _ = env.step(step_rng, state, actions)
    assert set(obs2) >= set(agents)
    assert set(rewards) >= set(agents)
    assert "__all__" in dones, f"{name}: expected an aggregate done flag"


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_action_masks_are_nonempty_and_consistent(name: str):
    """A legal action always exists, and the mask width matches the action space."""
    env = make(name)
    _, state = env.reset(jax.random.PRNGKey(0))
    avail = env.get_avail_actions(state)
    for agent in env.agents:
        mask = jnp.asarray(avail[agent])
        assert mask.ndim == 1
        assert mask.sum() > 0, f"{name}/{agent}: no legal actions at reset"
        assert mask.shape[0] == env.action_space(agent).n


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_declared_metadata_matches_behaviour(name: str):
    """turn_based configurations must actually mask down to one acting agent."""
    cfg = get_config(name)
    env = make(name)
    _, state = env.reset(jax.random.PRNGKey(0))
    avail = env.get_avail_actions(state)

    # In a turn-based game only the agent to move has a non-degenerate action set.
    acting = [a for a in env.agents if jnp.asarray(avail[a]).sum() > 1]
    if cfg.turn_based:
        assert len(acting) == 1, f"{name}: declared turn_based but {len(acting)} agents can act"
    else:
        assert len(acting) == len(list(env.agents)), f"{name}: declared simultaneous"


def test_env_kwargs_are_immutable_across_calls():
    """make() must not mutate the registry's stored kwargs (metadata integrity)."""
    before = dict(get_config("lbf_12x12").env_kwargs)
    make("lbf_12x12")
    assert get_config("lbf_12x12").env_kwargs == before
