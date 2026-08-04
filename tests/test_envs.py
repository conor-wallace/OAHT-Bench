"""Phase 0a exit criterion: every environment preset instantiates and rolls out.

The project plan (§10.2) makes this the first gate deliberately — if the seven
configurations cannot be driven through a uniform interface, every later phase
inherits the defect.

These tests check the config layer against *behaviour*, not against itself. The
important ones assert that a derived property (``turn_based``) matches what the
environment actually does, and that the kwargs we emit are exactly the ones
jax-aht accepts.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from oaht_bench.configs.env import PRESETS, get_preset, preset_names
from oaht_bench.envs import make

ALL_PRESETS = preset_names()


def test_presets_cover_the_committed_scope():
    """Seven results configurations plus one debug configuration (§12.1)."""
    assert set(preset_names("tier1")) == {"lbf_12x12", "overcooked_counter_circuit", "hanabi"}
    assert len(preset_names("tier2")) == 4  # the remaining Overcooked layouts
    assert preset_names("debug") == ["mini_hanabi"]


def test_tier1_overcooked_is_not_a_non_discriminative_layout():
    """ZSC-Eval reports forced_coord and asymm_advantages fail to separate methods."""
    tier1_overcooked = [n for n in preset_names("tier1") if n.startswith("overcooked")]
    assert tier1_overcooked == ["overcooked_counter_circuit"]


@pytest.mark.parametrize("name", ALL_PRESETS)
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


@pytest.mark.parametrize("name", ALL_PRESETS)
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


@pytest.mark.parametrize("name", ALL_PRESETS)
def test_turn_based_property_matches_behaviour(name: str):
    """The derived ``turn_based`` property must match what the environment does.

    Hanabi masks every agent but the one to move down to a single no-op; the
    simultaneous environments leave all agents with real choices. This is the
    check that would catch an upstream wrapper change silently breaking turn
    alternation — which would corrupt the learning-history view months later.
    """
    cfg = get_preset(name)
    env = make(name)
    _, state = env.reset(jax.random.PRNGKey(0))
    avail = env.get_avail_actions(state)

    acting = [a for a in env.agents if jnp.asarray(avail[a]).sum() > 1]
    if cfg.turn_based:
        assert len(acting) == 1, f"{name}: turn_based but {len(acting)} agents can act"
    else:
        assert len(acting) == len(list(env.agents)), f"{name}: expected simultaneous play"


@pytest.mark.parametrize("name", ALL_PRESETS)
def test_emitted_env_kwargs_are_exactly_accepted(name: str):
    """Every key we emit is one the environment accepts.

    jax-aht forwards unrecognised kwargs to the underlying Jumanji/JaxMARL
    constructor, which rejects them, so an over-broad ``env_kwargs()`` fails
    loudly here rather than at collection time.
    """
    cfg = get_preset(name)
    make(cfg)  # would raise TypeError on an unexpected kwarg


def test_symmetric_roles_is_derived_from_layout():
    """Mirroring validity (§4.5) follows from the layout, not a hand-set flag."""
    assert get_preset("overcooked_counter_circuit").symmetric_roles is True
    assert get_preset("overcooked_forced_coord").symmetric_roles is False
    assert get_preset("overcooked_asymm_advantages").symmetric_roles is False


def test_env_kwargs_are_not_shared_between_calls():
    """env_kwargs() must return a fresh dict; jax-aht mutates what it is given."""
    cfg = get_preset("lbf_12x12")
    first = cfg.env_kwargs()
    first["grid_size"] = 999
    assert cfg.env_kwargs()["grid_size"] == 12


def test_presets_are_immutable():
    """Configs are frozen so a preset cannot be mutated out from under a run."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        get_preset("lbf_12x12").grid_size = 7  # type: ignore[misc]


def test_preset_names_are_unique_and_registered():
    assert len(PRESETS) == len(set(PRESETS))
    assert set(PRESETS) == set(ALL_PRESETS)


# --- the training-facing environment interface -----------------------------


@pytest.mark.parametrize("name", ALL_PRESETS)
def test_envs_satisfy_the_training_protocol(name: str):
    """Both the raw environment and the wrapped one must be usable by training.

    ``LogWrapper`` inherits from JaxMARL's wrapper and shares no ancestor with
    ``BaseEnv``, so a nominal annotation would be wrong for one of them. The
    protocol is what the training loop actually requires.
    """
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.envs.protocols import TrainingEnv

    env = make(name)
    assert isinstance(env, TrainingEnv)
    assert isinstance(LogWrapper(env), TrainingEnv)
