"""Phase 0a exit criterion: every registered environment instantiates and rolls out.

The project plan (§10.2) makes this the first gate deliberately — if the seven
configurations cannot be driven through a uniform interface, every later phase
inherits the defect.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from oaht_bench.envs.registry import REGISTRY, config_names, get_config, make

ALL_CONFIGS = config_names()


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
    cfg = get_config(name)
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

    obs2, state2, rewards, dones, infos = env.step(step_rng, state, actions)
    assert set(obs2) >= set(agents)
    assert set(rewards) >= set(agents)
    assert "__all__" in dones, f"{name}: expected an aggregate done flag"


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_action_masks_are_nonempty_and_consistent(name: str):
    """A legal action always exists, and the mask width matches the action space."""
    env = make(name)
    obs, state = env.reset(jax.random.PRNGKey(0))
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
