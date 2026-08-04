"""The config layer's guarantees: strictness, provenance, and round-tripping.

These do not need JAX. They cover the properties the reproducibility claim rests
on — a config file fully determines a run, hashes stably, and rejects anything
it cannot interpret rather than silently defaulting.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from oaht_bench.configs import JOB_ADAPTER, SCHEMA_VERSION, get_preset, load_job
from oaht_bench.configs.env import HanabiConfig, LbfConfig
from oaht_bench.configs.job import TeammateGenerationJob
from oaht_bench.configs.teammate_gen import FcpConfig


def _job(**overrides) -> TeammateGenerationJob:
    kwargs = dict(
        label="t",
        env=get_preset("lbf_12x12"),
        generator=FcpConfig(PARTNER_POP_SIZE=5),
    )
    kwargs.update(overrides)
    return TeammateGenerationJob(**kwargs)


# --- strictness ------------------------------------------------------------


def test_unknown_fields_are_rejected():
    """The whole point of the typed layer.

    jax-aht reads kwargs with ``dict.get(key, default)``, so a misspelled field
    would otherwise fall through to a silent default and produce a *different
    environment* than intended — an entire dataset collected against the wrong
    config, with no error anywhere.
    """
    with pytest.raises(ValidationError, match="grid_sise"):
        LbfConfig(
            name="typo", grid_size=12, num_food=6, rollout_length=128, grid_sise=12
        )


def test_hanabi_kwargs_rejected_on_lbf():
    """The discriminated union keeps each environment's parameters separate."""
    with pytest.raises(ValidationError):
        LbfConfig(
            name="x", grid_size=12, num_food=6, rollout_length=128, num_colors=5
        )


def test_lbf_food_capacity_is_checked_at_config_load():
    """Jumanji asserts this at reset(); catching it here names the conflicting fields."""
    with pytest.raises(ValidationError, match="grid too small"):
        LbfConfig(name="x", grid_size=7, num_food=6, rollout_length=128)


def test_hanabi_deck_consistency_is_checked():
    with pytest.raises(ValidationError, match="num_cards_of_rank"):
        HanabiConfig(
            name="x",
            num_colors=5,
            num_ranks=5,
            hand_size=5,
            max_info_tokens=8,
            max_life_tokens=3,
            num_cards_of_rank=(3, 2, 2),  # 3 entries, num_ranks=5
            rollout_length=128,
        )


def test_hanabi_deck_must_outlast_the_deal():
    with pytest.raises(ValidationError, match="leaves nothing to draw"):
        HanabiConfig(
            name="x",
            num_colors=1,
            num_ranks=1,
            hand_size=5,
            max_info_tokens=8,
            max_life_tokens=3,
            num_cards_of_rank=(2,),
            rollout_length=128,
        )


# --- provenance ------------------------------------------------------------


def test_content_hash_is_stable_and_order_independent():
    """The hash is written into artifacts, so it must not depend on key order."""
    a = _job()
    b = TeammateGenerationJob.model_validate(json.loads(a.canonical_json()))
    assert a.content_hash() == b.content_hash()


def test_content_hash_changes_with_any_field():
    base = _job()
    assert base.content_hash() != _job(seed=1).content_hash()
    assert base.content_hash() != _job(generator=FcpConfig(PARTNER_POP_SIZE=6)).content_hash()
    assert base.content_hash() != _job(env=get_preset("hanabi")).content_hash()


def test_run_dir_is_disambiguated_by_hash():
    """Two runs sharing a label but differing in config must not collide."""
    a, b = _job(seed=0), _job(seed=1)
    assert a.label == b.label
    assert a.run_dir() != b.run_dir()


# --- serialization ---------------------------------------------------------


def test_job_round_trips_through_json(tmp_path):
    original = _job()
    path = original.to_json_file(tmp_path / "job.json")
    loaded = load_job(path)
    assert loaded == original
    assert loaded.content_hash() == original.content_hash()


def test_job_type_selects_the_model():
    payload = json.loads(_job().canonical_json())
    assert JOB_ADAPTER.validate_python(payload).job_type == "teammate_generation"


def test_missing_job_type_is_a_clear_error(tmp_path):
    p = tmp_path / "j.json"
    p.write_text(json.dumps({"label": "x"}))
    with pytest.raises(ValueError, match="missing 'job_type'"):
        load_job(p)


def test_malformed_json_names_the_file(tmp_path):
    p = tmp_path / "j.json"
    p.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_job(p)


# --- schema versioning -----------------------------------------------------


def test_future_schema_version_is_refused(tmp_path):
    """A newer config must fail loudly rather than load with fields dropped."""
    payload = json.loads(_job().canonical_json())
    payload["schema_version"] = SCHEMA_VERSION + 1
    p = tmp_path / "j.json"
    p.write_text(json.dumps(payload))
    with pytest.raises(ValidationError, match="Upgrade oaht-bench"):
        load_job(p)


def test_schema_version_defaults_to_current():
    assert _job().schema_version == SCHEMA_VERSION


# --- jax-aht interop -------------------------------------------------------


def test_task_config_matches_jax_aht_task_schema():
    """The shape jax-aht's runners read after OmegaConf.to_container."""
    task = get_preset("lbf_12x12").task_config()
    assert set(task) == {"ENV_NAME", "ENV_KWARGS", "ROLLOUT_LENGTH", "TASK_NAME"}
    assert task["ENV_NAME"] == "lbf"
    assert task["ROLLOUT_LENGTH"] == 128
    assert task["ENV_KWARGS"]["grid_size"] == 12
