"""The config layer's guarantees: strictness, provenance, and round-tripping.

These do not need JAX. They cover the properties the reproducibility claim rests
on — a config file fully determines a run, hashes stably, and rejects anything
it cannot interpret rather than silently defaulting.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from oaht_bench.configs import SCHEMA_VERSION, get_preset, load_job, save_job, validate_job
from oaht_bench.configs.env import HanabiConfig, LbfConfig
from oaht_bench.configs.job import JobConfig, TeammateGenerationJob
from oaht_bench.configs.teammate_gen import FcpConfig


def _job(**overrides) -> TeammateGenerationJob:
    kwargs = dict(
        label="t",
        env=get_preset("lbf_12x12"),
        generator=FcpConfig(population_size=5),
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
    assert base.content_hash() != _job(generator=FcpConfig(population_size=6)).content_hash()
    assert base.content_hash() != _job(env=get_preset("hanabi")).content_hash()


def test_run_dir_is_disambiguated_by_hash():
    """Two runs sharing a label but differing in config must not collide."""
    a, b = _job(seed=0), _job(seed=1)
    assert a.label == b.label
    assert a.run_dir() != b.run_dir()


# --- serialization ---------------------------------------------------------


def test_job_round_trips_through_json(tmp_path):
    original = _job()
    path = save_job(original, tmp_path / "job.json")
    loaded = load_job(path)
    assert loaded == original
    assert loaded.content_hash() == original.content_hash()


def test_job_type_selects_the_model():
    """JobConfig is a real class, so pydantic's usual interface works on it."""
    payload = {"job": json.loads(_job().canonical_json())}
    assert JobConfig.model_validate(payload).job.job_type == "teammate_generation"
    assert validate_job(payload).job_type == "teammate_generation"


def test_flat_payload_gets_a_targeted_error():
    """``extra="forbid"`` would otherwise report every job field as unexpected,
    burying the actual mistake."""
    with pytest.raises(ValueError, match='belong under a "job" key'):
        validate_job(json.loads(_job().canonical_json()))


def test_missing_job_key_is_a_clear_error(tmp_path):
    p = tmp_path / "j.json"
    p.write_text(json.dumps({"label": "x"}))
    with pytest.raises(ValueError, match="missing 'job'"):
        load_job(p)


def test_malformed_json_names_the_file(tmp_path):
    p = tmp_path / "j.json"
    p.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_job(p)


# --- schema versioning -----------------------------------------------------


def test_future_schema_version_is_refused(tmp_path):
    """A newer config must fail loudly rather than load with fields dropped."""
    payload = {"job": json.loads(_job().canonical_json())}
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


# --- naming boundary -------------------------------------------------------


def test_config_fields_are_snake_case():
    """Our public API must not leak the absorbed code's SCREAMING_CASE keys.

    Config authors write JSON against these field names; jax-aht's Hydra
    convention is an implementation detail of the boundary, translated in exactly
    one place (``to_algorithm_dict``).
    """
    from oaht_bench.configs.env import HanabiConfig, LbfConfig, OvercookedV1Config
    from oaht_bench.configs.job import JobConfig, TeammateGenerationJob
    from oaht_bench.configs.teammate_gen import (
        BrDivConfig,
        CoMeDiConfig,
        FcpConfig,
        LBrDivConfig,
        PpoHyperparams,
    )

    models = [
        LbfConfig, HanabiConfig, OvercookedV1Config, TeammateGenerationJob,
        PpoHyperparams, FcpConfig, CoMeDiConfig, BrDivConfig, LBrDivConfig,
    ]
    offenders = [
        f"{m.__name__}.{name}"
        for m in models
        for name in m.model_fields
        if name != name.lower()
    ]
    assert not offenders, f"non-snake_case config fields: {offenders}"


def test_generator_translates_to_upstream_keys():
    """The boundary translation produces the keys the absorbed code reads."""
    from oaht_bench.configs.teammate_gen import BrDivConfig, FcpConfig, LBrDivConfig

    fcp = FcpConfig(population_size=5).to_algorithm_dict()
    assert fcp["ALG"] == "fcp"
    assert fcp["PARTNER_POP_SIZE"] == 5
    assert fcp["LR"] == 1e-4  # from PpoHyperparams.learning_rate

    assert BrDivConfig(cross_play_weight=0.05).to_algorithm_dict()["XP_LOSS_WEIGHTS"] == 0.05
    assert LBrDivConfig(lagrange_learning_rate=0.0036).to_algorithm_dict()["LAGRANGE_LR"] == 0.0036


def test_comedi_defaults_to_eight_minibatches():
    """CoMeDi's base config differs from the shared PPO default; keep it."""
    from oaht_bench.configs.teammate_gen import CoMeDiConfig

    assert CoMeDiConfig().to_algorithm_dict()["NUM_MINIBATCHES"] == 8


def test_validate_job_shares_load_job_error_handling(tmp_path):
    """In-memory validation gives the same message as loading from a file."""
    with pytest.raises(ValueError, match="missing 'job'"):
        validate_job({"label": "x"})


def test_public_api_does_not_leak_the_adapter():
    """The adapter is an implementation detail of load_job/validate_job.

    Exposing it would let callers bypass the error handling that turns a bad
    config into a message naming the file and field.
    """
    import oaht_bench.configs as c

    assert "JOB_ADAPTER" not in c.__all__
    assert {"load_job", "validate_job"} <= set(c.__all__)


# --- branching fields must be constrained ----------------------------------

#: Field names whose value selects a code path. A plain ``str`` here reaches an
#: if/elif chain with no ``else``, so a typo either crashes deep inside training
#: or silently selects different behaviour.
_BRANCHING_FIELD_SUFFIXES = ("_type", "_name", "baseline", "backbone", "variant", "generator")


def _all_config_models():
    from oaht_bench.configs import env as env_mod
    from oaht_bench.configs import job as job_mod
    from oaht_bench.configs import teammate_gen as tg_mod
    from oaht_bench.configs.base import BaseConfig

    seen = {}
    for mod in (env_mod, job_mod, tg_mod):
        for name, obj in vars(mod).items():
            if isinstance(obj, type) and issubclass(obj, BaseConfig) and obj is not BaseConfig:
                seen[obj.__name__] = obj
    return list(seen.values())


def test_branching_fields_are_not_bare_strings():
    """Any field that selects a code path must be a Literal, not a str.

    ``job_type`` was already safe, but ``actor_type`` and ``baseline`` were not:
    both feed if/elif dispatch in the absorbed code, so 'mpl' or 'LIAM' would
    have passed validation and failed much later, or silently done the wrong
    thing.
    """
    import typing

    offenders = []
    for model in _all_config_models():
        for name, field in model.model_fields.items():
            if not any(name.endswith(s) or name == s for s in _BRANCHING_FIELD_SUFFIXES):
                continue
            ann = field.annotation
            # Unwrap Optional[...] and other unions to look for a Literal member.
            args = typing.get_args(ann)
            is_literal = typing.get_origin(ann) is typing.Literal or any(
                typing.get_origin(a) is typing.Literal for a in args
            )
            if ann is str and not is_literal:
                offenders.append(f"{model.__name__}.{name}")
    assert not offenders, (
        "branching fields typed as bare str; use Literal so a typo fails at "
        f"config load: {offenders}"
    )


def test_actor_type_typo_is_rejected():
    from oaht_bench.configs.teammate_gen import FcpConfig

    with pytest.raises(ValidationError, match="Input should be"):
        FcpConfig(actor_type="mpl")


def test_baseline_typo_is_rejected():
    from oaht_bench.configs.job import TrainingJob

    with pytest.raises(ValidationError, match="Input should be"):
        TrainingJob(
            label="t", env=get_preset("lbf_12x12"), dataset_path="d", baseline="LIAM"
        )


def test_baseline_roster_matches_the_plan():
    """Thirteen baselines in four groups (§12.9)."""
    import typing

    from oaht_bench.configs.job import BaselineName

    names = set(typing.get_args(BaselineName))
    assert len(names) == 13
    assert {"random", "pct_bc", "oracle"} <= names  # floors and ceiling
    assert {"ad", "dpt", "amago_offline", "hybrid_ad"} <= names  # learning-history
    assert {"liam", "meliba", "tao", "omis", "taget"} <= names  # trajectory-view


# --- derived runtime values ------------------------------------------------


def test_runtime_rejects_a_budget_that_trains_nothing():
    """Upstream computed num_updates with integer division and no check, so a
    too-small budget silently produced zero updates and a no-op training run."""
    from oaht_bench.configs.network import MlpNetwork
    from oaht_bench.configs.teammate_gen import PpoHyperparams
    from oaht_bench.teammate_gen.runtime import PpoRuntime

    with pytest.raises(ValueError, match="would be a no-op"):
        PpoRuntime.from_config(
            ppo=PpoHyperparams(), network=MlpNetwork(), actor_type="mlp",
            rollout_length=128, num_envs=8, total_timesteps=100,
            num_checkpoints=2, num_agents=2,
        )


def test_runtime_rejects_empty_minibatches():
    from oaht_bench.configs.network import MlpNetwork
    from oaht_bench.configs.teammate_gen import PpoHyperparams
    from oaht_bench.teammate_gen.runtime import PpoRuntime

    with pytest.raises(ValueError, match="minibatches would be empty"):
        PpoRuntime.from_config(
            ppo=PpoHyperparams(num_minibatches=100_000), network=MlpNetwork(),
            actor_type="mlp", rollout_length=4, num_envs=2,
            total_timesteps=1e5, num_checkpoints=2, num_agents=2,
        )


def test_runtime_derives_the_same_values_upstream_computed():
    from oaht_bench.configs.network import MlpNetwork
    from oaht_bench.configs.teammate_gen import PpoHyperparams
    from oaht_bench.teammate_gen.runtime import PpoRuntime

    rt = PpoRuntime.from_config(
        ppo=PpoHyperparams(num_minibatches=4), network=MlpNetwork(),
        actor_type="mlp", rollout_length=128, num_envs=8,
        total_timesteps=1e6, num_checkpoints=5, num_agents=2,
    )
    assert rt.num_actors == 2 * 8
    assert rt.num_updates == int(1e6 // 128 // 8)
    assert rt.minibatch_size == (2 * 8) * 128 // 4


def test_runtime_is_frozen():
    """Derived values must not drift mid-run any more than authored ones."""
    from oaht_bench.configs.network import MlpNetwork
    from oaht_bench.configs.teammate_gen import PpoHyperparams
    from oaht_bench.teammate_gen.runtime import PpoRuntime

    rt = PpoRuntime.from_config(
        ppo=PpoHyperparams(), network=MlpNetwork(), actor_type="mlp",
        rollout_length=128, num_envs=8, total_timesteps=1e6,
        num_checkpoints=5, num_agents=2,
    )
    with pytest.raises(ValidationError):
        rt.num_updates = 1  # type: ignore[misc]


def test_network_architecture_is_now_part_of_the_hash():
    """It was defaulted via dict.get upstream, so it never entered provenance."""
    from oaht_bench.configs.network import MlpNetwork
    from oaht_bench.configs.teammate_gen import FcpConfig

    a = _job(generator=FcpConfig(network=MlpNetwork(hidden_dim=64)))
    b = _job(generator=FcpConfig(network=MlpNetwork(hidden_dim=128)))
    assert a.content_hash() != b.content_hash()


def test_comedi_rejects_more_minibatches_than_envs():
    """CoMeDi minibatches over environments, not actors.

    Exceeding num_envs fails as an opaque reshape error many frames deep
    ("cannot reshape array of shape (128, 4) into [128, 8, -1]").
    """
    from oaht_bench.configs.teammate_gen import CoMeDiConfig, PpoHyperparams

    with pytest.raises(ValidationError, match="exceeds num_envs"):
        CoMeDiConfig(num_envs=4, ppo=PpoHyperparams(num_minibatches=8))


def test_conditional_critic_actor_requires_pop_size():
    """Omitting it surfaces as a bare KeyError('POP_SIZE') inside policy construction."""
    from oaht_bench.configs.network import MlpNetwork
    from oaht_bench.configs.teammate_gen import PpoHyperparams
    from oaht_bench.teammate_gen.runtime import PpoRuntime

    with pytest.raises(ValueError, match="pop_size is required"):
        PpoRuntime.from_config(
            ppo=PpoHyperparams(), network=MlpNetwork(),
            actor_type="actor_with_conditional_critic",
            rollout_length=128, num_envs=8, total_timesteps=1e6,
            num_checkpoints=2, num_agents=2,
        )


def test_logger_survives_unserializable_values(tmp_path):
    """A metric sink must never take a training run down with it.

    CoMeDi logs wandb chart objects through the same log_item used for scalars.
    """
    from oaht_bench.common.logging import RunLogger

    class Chart:
        pass

    with RunLogger(tmp_path / "run") as logger:
        logger.log_item("Losses/thing", Chart())
        logger.log_item("Train/ok", 1.5)

    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert any('"Train/ok": 1.5' in ln for ln in lines)
