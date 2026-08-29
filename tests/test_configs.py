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
        LbfConfig(name="typo", grid_size=12, num_food=6, rollout_length=128, grid_sise=12)


def test_hanabi_kwargs_rejected_on_lbf():
    """The discriminated union keeps each environment's parameters separate."""
    with pytest.raises(ValidationError):
        LbfConfig(name="x", grid_size=12, num_food=6, rollout_length=128, num_colors=5)


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
    from oaht_bench.configs.job import TeammateGenerationJob
    from oaht_bench.configs.teammate_gen import (
        BrDivConfig,
        CoMeDiConfig,
        FcpConfig,
        LBrDivConfig,
        PpoHyperparams,
    )

    models = [
        LbfConfig,
        HanabiConfig,
        OvercookedV1Config,
        TeammateGenerationJob,
        PpoHyperparams,
        FcpConfig,
        CoMeDiConfig,
        BrDivConfig,
        LBrDivConfig,
    ]
    offenders = [
        f"{m.__name__}.{name}" for m in models for name in m.model_fields if name != name.lower()
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
        for _name, obj in vars(mod).items():
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
        TrainingJob(label="t", env=get_preset("lbf_12x12"), dataset_path="d", baseline="LIAM")


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
            ppo=PpoHyperparams(),
            network=MlpNetwork(),
            actor_type="mlp",
            rollout_length=128,
            num_envs=8,
            total_timesteps=100,
            num_checkpoints=2,
            num_agents=2,
        )


def test_runtime_rejects_empty_minibatches():
    from oaht_bench.configs.network import MlpNetwork
    from oaht_bench.configs.teammate_gen import PpoHyperparams
    from oaht_bench.teammate_gen.runtime import PpoRuntime

    with pytest.raises(ValueError, match="minibatches would be empty"):
        PpoRuntime.from_config(
            ppo=PpoHyperparams(num_minibatches=100_000),
            network=MlpNetwork(),
            actor_type="mlp",
            rollout_length=4,
            num_envs=2,
            total_timesteps=1e5,
            num_checkpoints=2,
            num_agents=2,
        )


def test_runtime_derives_the_same_values_upstream_computed():
    from oaht_bench.configs.network import MlpNetwork
    from oaht_bench.configs.teammate_gen import PpoHyperparams
    from oaht_bench.teammate_gen.runtime import PpoRuntime

    rt = PpoRuntime.from_config(
        ppo=PpoHyperparams(num_minibatches=4),
        network=MlpNetwork(),
        actor_type="mlp",
        rollout_length=128,
        num_envs=8,
        total_timesteps=1e6,
        num_checkpoints=5,
        num_agents=2,
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
        ppo=PpoHyperparams(),
        network=MlpNetwork(),
        actor_type="mlp",
        rollout_length=128,
        num_envs=8,
        total_timesteps=1e6,
        num_checkpoints=5,
        num_agents=2,
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
            ppo=PpoHyperparams(),
            network=MlpNetwork(),
            actor_type="actor_with_conditional_critic",
            rollout_length=128,
            num_envs=8,
            total_timesteps=1e6,
            num_checkpoints=2,
            num_agents=2,
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


def test_comedi_runtime_accounts_for_selection_rollouts():
    """CoMeDi's update budget is not total_timesteps // rollout_length // num_envs.

    Each outer iteration also spends steps on population-selection rollouts, so
    the divisor is selection + training. Using the plain formula silently changes
    how many updates happen.
    """
    from oaht_bench.configs.teammate_gen import CoMeDiConfig
    from oaht_bench.teammate_gen.runtime import CoMeDiRuntime

    gen = CoMeDiConfig(
        population_size=2,
        num_envs=8,
        num_argmax_rollout_episodes=2,
        total_timesteps_per_iteration=8192,
    )
    rt = CoMeDiRuntime.from_config(gen, rollout_length=128, num_agents=2)

    selection = 2 * 2 * 128 // 2
    training = 4 * 128 * 8
    assert rt.num_updates == int(8192 // (selection + training))
    assert rt.num_updates != int(8192 // 128 // 8)  # the naive formula differs


def test_comedi_runtime_rejects_a_budget_below_one_update():
    from oaht_bench.configs.teammate_gen import CoMeDiConfig
    from oaht_bench.teammate_gen.runtime import CoMeDiRuntime

    gen = CoMeDiConfig(population_size=2, num_envs=8, total_timesteps_per_iteration=100)
    with pytest.raises(ValueError, match="population selection"):
        CoMeDiRuntime.from_config(gen, rollout_length=128, num_agents=2)


def test_comedi_warmup_differs_in_exactly_two_values():
    """The warmup replaced a mutated dict copy; state what it changes."""
    from oaht_bench.configs.teammate_gen import CoMeDiConfig
    from oaht_bench.teammate_gen.runtime import CoMeDiRuntime

    gen = CoMeDiConfig(population_size=2, num_envs=8, total_timesteps_per_iteration=1e5)
    rt = CoMeDiRuntime.from_config(gen, rollout_length=128, num_agents=2)
    warm = rt.warmup()

    assert warm.actor_type == "pseudo_actor_with_conditional_critic"
    assert warm.actor_type != rt.actor_type
    assert warm.total_timesteps == rt.total_timesteps_per_iteration
    assert warm.pop_size == rt.population_size
    assert warm.ppo == rt.ppo  # everything else carries over unchanged


def test_comedi_requires_two_agents():
    from oaht_bench.configs.teammate_gen import CoMeDiConfig
    from oaht_bench.teammate_gen.runtime import CoMeDiRuntime

    gen = CoMeDiConfig(population_size=2, num_envs=8)
    with pytest.raises(ValueError, match="exactly 2 agents"):
        CoMeDiRuntime.from_config(gen, rollout_length=128, num_agents=3)


def test_paired_runtime_serves_both_brdiv_and_lbrdiv():
    """BRDiv and L-BRDiv share their derived shape entirely.

    They differ only in how the cross-play term is weighted -- BRDiv fixes it,
    L-BRDiv learns it -- which is authored config, not anything derived.
    """
    from oaht_bench.configs.teammate_gen import BrDivConfig, LBrDivConfig
    from oaht_bench.teammate_gen.runtime import PairedDiversityRuntime

    shared = dict(population_size=2, num_envs=8, total_timesteps=8192)
    br = PairedDiversityRuntime.from_config(BrDivConfig(**shared), rollout_length=128, num_agents=2)
    lbr = PairedDiversityRuntime.from_config(
        LBrDivConfig(**shared), rollout_length=128, num_agents=2
    )

    assert (br.num_updates, br.num_conf_actors, br.num_br_actors) == (
        lbr.num_updates,
        lbr.num_conf_actors,
        lbr.num_br_actors,
    )
    assert br.cross_play_weight is not None and br.lagrange_learning_rate is None
    assert lbr.lagrange_learning_rate is not None and lbr.cross_play_weight is None


def test_paired_runtime_rejects_a_no_op_budget():
    from oaht_bench.configs.teammate_gen import BrDivConfig
    from oaht_bench.teammate_gen.runtime import PairedDiversityRuntime

    with pytest.raises(ValueError, match="would be a no-op"):
        PairedDiversityRuntime.from_config(
            BrDivConfig(population_size=2, num_envs=8, total_timesteps=100),
            rollout_length=128,
            num_agents=2,
        )


def test_paired_runtime_requires_two_agents():
    from oaht_bench.configs.teammate_gen import LBrDivConfig
    from oaht_bench.teammate_gen.runtime import PairedDiversityRuntime

    with pytest.raises(ValueError, match="exactly 2 agents"):
        PairedDiversityRuntime.from_config(
            LBrDivConfig(population_size=2, num_envs=8, total_timesteps=1e5),
            rollout_length=128,
            num_agents=3,
        )


def test_no_generator_reads_a_config_dict():
    """All four generators are converted; none should index a dict by string key.

    Guards the boundary: the typed config is the only way parameters reach the
    training code.
    """
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "oaht_bench" / "teammate_gen"
    # runtime.py quotes the old pattern in its module docstring to explain what
    # it replaced, so it is prose rather than a live read.
    offenders = []
    for path in sorted(p for p in src.glob("*.py") if p.name != "runtime.py"):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r'\b(config|algorithm_config|warmup_config)\["', line):
                offenders.append(f"{path.name}:{i}")
    assert not offenders, f"config dict reads remain: {offenders}"


# --- shipped teammate-generation configs ------------------------------------


def _shipped_configs():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "configs" / "teammate_gen"
    return sorted(root.rglob("*.json"))


def test_shipped_configs_exist_for_every_generator_and_tier1_env():
    from oaht_bench.configs import preset_names

    paths = _shipped_configs()
    envs = {p.parent.name for p in paths}
    gens = {p.stem for p in paths}
    tier1 = set(preset_names("tier1"))
    assert envs == tier1

    # The four cooperative generators cover every tier1 environment. AD-RPG is
    # shipped for LBF only: it is 2-player, ~n^2 in cost, and only validated on the
    # environment the others are tuned on (see gen_teammate_configs.py and
    # docs/tuning_record.md), so it is deliberately not emitted for the rest.
    cooperative = {"fcp", "comedi", "brdiv", "lbrdiv"}
    assert gens == cooperative | {"rpg"}
    rpg_envs = {p.parent.name for p in paths if p.stem == "rpg"}
    assert rpg_envs == {"lbf_12x12"}
    assert len(paths) == len(cooperative) * len(tier1) + 1


@pytest.mark.parametrize("path", _shipped_configs(), ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_shipped_config_builds_a_valid_runtime(path):
    """Every shipped config must survive runtime construction.

    That is where budget and minibatch constraints are checked, so this catches a
    config that would train nothing or fail deep inside a vmap -- before anyone
    queues it on a cluster.
    """
    from oaht_bench.configs import load_job
    from oaht_bench.teammate_gen.runtime import (
        CoMeDiRuntime,
        PairedDiversityRuntime,
        PpoRuntime,
    )

    job = load_job(path)
    gen, rl = job.generator, job.env.rollout_length

    if gen.generator == "fcp":
        rt = PpoRuntime.from_config(
            ppo=gen.ppo,
            network=gen.network,
            actor_type=gen.actor_type,
            rollout_length=rl,
            num_envs=gen.num_envs,
            total_timesteps=gen.total_timesteps,
            num_checkpoints=gen.num_checkpoints,
            num_agents=2,
        )
    elif gen.generator == "comedi":
        rt = CoMeDiRuntime.from_config(gen, rollout_length=rl, num_agents=2)
    else:
        rt = PairedDiversityRuntime.from_config(gen, rollout_length=rl, num_agents=2)

    assert rt.num_updates >= 1


def test_overcooked_configs_enable_reward_shaping():
    """The environment defaults it off; jax-aht's task configs turn it on.

    A population trained without shaping solves a materially harder sparse-reward
    task and is not comparable to one trained with it.
    """
    from oaht_bench.configs import load_job

    for path in _shipped_configs():
        # v1 only: OvercookedV2Config has no do_reward_shaping field.
        # Whether v2 populations should train against shaped or sparse
        # reward is a real, open question -- v2's own SHAPED_REWARDS
        # mechanism exists (settings.py) but isn't folded into the
        # returned reward by the wrapper (see overcooked_v2_wrapper.py),
        # deliberately left undecided rather than defaulted silently.
        if not path.parent.name.startswith("overcooked") or path.parent.name.startswith(
            "overcooked_v2"
        ):
            continue
        env = load_job(path).env
        assert env.do_reward_shaping is True
        assert "reward_shaping_params" in env.env_kwargs()


def test_lagrange_lr_is_scaled_for_population_size():
    """Upstream's 0.01 is tuned at n=3; it must scale by (3/n)^2 (§7.3)."""
    from scripts.gen_teammate_configs import _lagrange_lr

    assert _lagrange_lr(3) == pytest.approx(0.01)
    assert _lagrange_lr(5) == pytest.approx(0.0036, abs=1e-6)


# --- minimal (delta) config files -------------------------------------------


def test_minimal_dump_keeps_discriminator_tags():
    """exclude_defaults alone drops job_type, env_name and generator.

    Their values equal their defaults, but the discriminated unions need them to
    choose a model, so a plain sparse dump produces an unloadable file.
    """
    from oaht_bench.configs.job import JobConfig

    d = JobConfig(job=_job()).minimal_dump()
    assert d["job"]["job_type"] == "teammate_generation"
    assert d["job"]["env"]["env_name"] == "lbf"
    assert d["job"]["generator"]["generator"] == "fcp"


def test_minimal_dump_omits_defaults():
    from oaht_bench.configs.job import JobConfig

    d = JobConfig(job=_job()).minimal_dump()
    assert "seed" not in d["job"]  # default 0
    assert "logging" not in d["job"]  # all defaults
    assert "ppo" not in d["job"]["generator"]  # FCP's LBF PPO block is all defaults here


def test_minimal_dump_omits_all_default_nested_models():
    """An untouched nested model is dropped whole, tag included.

    Loading reconstructs it, so emitting a lone tag for it would be noise.
    """
    from oaht_bench.configs.job import JobConfig

    d = JobConfig(job=_job()).minimal_dump()
    assert "network" not in d["job"]["generator"]


def test_minimal_dump_always_states_schema_version():
    """A file without it is indistinguishable from one written against a schema
    this build cannot interpret."""
    from oaht_bench.configs.job import JobConfig

    assert JobConfig(job=_job()).minimal_dump()["schema_version"] == SCHEMA_VERSION


def test_minimal_and_full_forms_are_equivalent(tmp_path):
    """The delta file must load to the same object, and the same hash, as the
    full one -- otherwise provenance depends on which form was written."""
    from oaht_bench.configs.job import JobConfig

    original = _job()
    lean = JobConfig(job=original).to_json_file(tmp_path / "lean.json", minimal=True)
    fat = JobConfig(job=original).to_json_file(tmp_path / "fat.json", minimal=False)

    a, b = load_job(lean), load_job(fat)
    assert a == b == original
    assert a.content_hash() == b.content_hash() == original.content_hash()
    assert lean.read_text().count("\n") < fat.read_text().count("\n")


@pytest.mark.parametrize("path", _shipped_configs(), ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_shipped_configs_are_deltas(path):
    """Shipped configs state what the experiment changes, not every default."""
    import json

    payload = json.loads(path.read_text())
    gen = payload["job"]["generator"]
    # A generator block restating all ten PPO fields means the delta broke.
    assert len(gen.get("ppo", {})) < 10


def test_run_directories_record_the_full_config(tmp_path):
    """Authored configs are deltas; recorded ones are not.

    A run's job.json must remain self-describing even if a default later moves,
    otherwise a released artifact's meaning depends on the code version that
    reads it.
    """
    import json

    from oaht_bench.configs import save_job

    p = save_job(_job(), tmp_path / "job.json", minimal=False)
    payload = json.loads(p.read_text())
    ppo = payload["job"]["generator"]["ppo"]
    assert len(ppo) == 10  # every PPO field stated
    assert "logging" in payload["job"]


# --- cross-generator metric parity ------------------------------------------


def test_log_training_curves_emits_the_shared_tags(tmp_path):
    """All four generators must report the same episode statistics.

    FCP and CoMeDi get these from ippo's in-training callback; BRDiv and
    L-BRDiv have their own loops and collected but never logged them, so
    convergence could not be compared across methods.
    """
    import json

    import numpy as np

    from oaht_bench.common.logging import RunLogger, log_training_curves

    metrics = {
        "returned_episode_returns": np.arange(6.0).reshape(2, 3),
        "returned_episode_lengths": np.full((2, 3), 100.0),
        "percent_eaten": np.arange(6.0).reshape(2, 3) * 2,
        "pg_loss_conf_agent": np.zeros((2, 3, 4)),  # a loss, not an episode stat
    }
    with RunLogger(tmp_path / "run") as logger:
        log_training_curves(logger, metrics, "lbf")

    tags = set()
    for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines():
        tags |= set(json.loads(line))
    assert {
        "Train/returned_episode_returns",
        "Train/returned_episode_lengths",
        "Train/percent_eaten",
    } <= tags
    assert not any(t.startswith("Train/pg_loss") for t in tags)


def test_log_training_curves_averages_over_seeds(tmp_path):
    """Statistics arrive as (num_seeds, num_updates); only the update axis survives."""
    import json

    import numpy as np

    from oaht_bench.common.logging import RunLogger, log_training_curves

    metrics = {"returned_episode_returns": np.array([[0.0, 2.0], [2.0, 4.0]])}
    with RunLogger(tmp_path / "run") as logger:
        log_training_curves(logger, metrics, "hanabi")

    series = [
        json.loads(line)["Train/returned_episode_returns"]
        for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
        if "Train/returned_episode_returns" in json.loads(line)
    ]
    assert series == [1.0, 3.0]  # means over the seed axis, one per update


def test_all_generators_report_the_same_episode_statistics():
    """Every generator must emit Train/<stat>, by whichever route.

    FCP and CoMeDi inherit them from ippo's io_callback; BRDiv and L-BRDiv call
    log_update_metrics from their own loops. A statistic present for only some
    methods cannot be used to compare convergence.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "oaht_bench" / "teammate_gen"
    assert 'f"Train/{stat_name}"' in (src / "marl" / "ippo.py").read_text()
    for name in ("BRDiv.py", "LBRDiv.py"):
        assert "log_update_metrics" in (src / name).read_text()


def test_log_update_metrics_skips_non_scalars(tmp_path):
    """Called from inside a jit via io_callback, so it must tolerate what it gets.

    The paired generators' loss terms carry a population axis; a partially
    reduced array is not meaningful plotted against an update step.
    """
    import json

    import numpy as np

    from oaht_bench.common.logging import RunLogger, log_update_metrics

    with RunLogger(tmp_path / "run") as logger:
        log_update_metrics(
            {
                "returned_episode_returns": np.float32(0.5),
                "pg_loss_conf_agent": np.zeros(4),  # population axis
                "returned_episode": np.float32(1.0),  # bookkeeping
                "update_steps": np.int32(7),
            },
            logger,
        )

    tags = set()
    for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines():
        rec = json.loads(line)
        tags |= set(rec)
        assert rec.get("train_step") == 7
    assert "Train/returned_episode_returns" in tags
    assert not any(t.startswith("Train/pg_loss") for t in tags)
    assert "Train/returned_episode" not in tags


def test_log_update_metrics_needs_a_step():
    """Without update_steps there is nothing to plot against; do not guess."""
    import tempfile

    from oaht_bench.common.logging import RunLogger, log_update_metrics

    with tempfile.TemporaryDirectory() as d:
        with RunLogger(d) as logger:
            log_update_metrics({"returned_episode_returns": 1.0}, logger)
        import pathlib

        assert pathlib.Path(d, "metrics.jsonl").read_text() == ""


def test_paired_generators_stream_rather_than_batch():
    """BRDiv and L-BRDiv must call back out of the jit during training.

    Their whole run is one jit(vmap(...)); without a callback a multi-hour job
    reports nothing until it returns. They must also not additionally log the
    same curves post-hoc, which would double every point.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "oaht_bench" / "teammate_gen"
    for name in ("BRDiv.py", "LBRDiv.py"):
        text = (src / name).read_text()
        assert "io_callback" in text, f"{name} does not stream"
        assert "log_training_curves" not in text, f"{name} would double-log"


# --- training step accounting ------------------------------------------------


def test_training_plan_counts_fcp_members_as_parallel():
    """FCP vmaps over members, so sequential depth is one member's worth."""
    from oaht_bench.configs.teammate_gen import FcpConfig
    from oaht_bench.teammate_gen.plan import training_plan

    job = _job(generator=FcpConfig(population_size=5, total_timesteps=1e6, num_envs=8))
    plan = training_plan(job)
    assert plan.parallel_members == 5
    assert plan.sequential_units == 1
    assert plan.sequential_updates == plan.updates_per_unit
    assert plan.total_updates == plan.updates_per_unit * 5


def test_training_plan_counts_comedi_outer_iterations():
    """CoMeDi's num_updates is per outer iteration, and it scans
    arange(1, population_size) -- so population_size - 1 of them, after a warmup
    with its own budget and a different formula."""
    from oaht_bench.configs.teammate_gen import CoMeDiConfig
    from oaht_bench.teammate_gen.plan import training_plan

    gen = CoMeDiConfig(population_size=4, num_envs=8, total_timesteps_per_iteration=1e6)
    plan = training_plan(_job(generator=gen))
    assert plan.sequential_units == 3
    assert plan.warmup_updates > 0
    assert plan.sequential_updates == plan.warmup_updates + plan.updates_per_unit * 3


def test_training_plan_counts_paired_generators_as_a_single_run():
    """BRDiv and L-BRDiv train all pairs jointly, so num_updates is the total."""
    from oaht_bench.configs.teammate_gen import BrDivConfig
    from oaht_bench.teammate_gen.plan import training_plan

    gen = BrDivConfig(population_size=3, num_envs=8, total_timesteps=1e6)
    plan = training_plan(_job(generator=gen))
    assert (plan.sequential_units, plan.parallel_members) == (1, 1)
    assert plan.sequential_updates == plan.total_updates == plan.updates_per_unit


def test_training_plan_surfaces_an_unrunnable_budget():
    """The plan builds the same runtime the trainer does, so a budget that would
    train nothing fails here rather than after a job is queued."""
    from oaht_bench.configs.teammate_gen import BrDivConfig
    from oaht_bench.teammate_gen.plan import training_plan

    gen = BrDivConfig(population_size=3, num_envs=8, total_timesteps=100)
    with pytest.raises(ValueError, match="would be a no-op"):
        training_plan(_job(generator=gen))


@pytest.mark.parametrize("path", _shipped_configs(), ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_shipped_configs_have_a_computable_plan(path):
    from oaht_bench.configs import load_job
    from oaht_bench.teammate_gen.plan import training_plan

    assert training_plan(load_job(path)).sequential_updates >= 1


def test_comedi_streams_its_outer_loop_not_only_the_warmup():
    """CoMeDi's Train/ curve must cover the whole run.

    Before this, only the self-play warmup streamed -- through ippo's callback --
    so the phase that actually builds the diverse population was invisible. A
    2,416-update job showed 976 points and looked identical in length to FCP.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "oaht_bench" / "teammate_gen"
    text = (src / "CoMeDi.py").read_text()
    assert "io_callback" in text, "CoMeDi does not stream its outer loop"
    # The streamed step must continue past the warmup rather than restart at 0.
    assert "_warmup_updates" in text


def test_comedi_streams_self_play_for_continuity():
    """The streamed series must stay self-play across the warmup boundary.

    The outer loop's own metric dict is built from cross-play trajectories;
    splicing those onto a self-play prefix would read as a regression at the
    hand-off that is really a change of what is being measured.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "oaht_bench" / "teammate_gen"
    text = (src / "CoMeDi.py").read_text()
    stream_block = text[
        text.index("sp_metric = jax.tree.map") : text.index("jax.experimental.io_callback(_stream")
    ]
    assert "traj_batch_sp_agent0" in stream_block


def test_paired_generators_log_the_intended_pairing_only():
    """BRDiv and L-BRDiv must restrict the streamed metric to conf_i vs br_i.

    conf_ids and br_ids are sampled independently and uniformly, so only 1/n of
    rollout episodes are the designed-optimal pairing; the rest are cross-play,
    which these objectives actively minimize. Logging the raw mixture under the
    same tag FCP and CoMeDi use would make a successful run trend downwards while
    they trend up.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "oaht_bench" / "teammate_gen"
    for name in ("BRDiv.py", "LBRDiv.py"):
        text = (src / name).read_text()
        block = text[text.index("_paired = (") : text.index("jax.experimental.io_callback(_stream")]
        assert "self_onehot_id" in block and "oppo_onehot_id" in block
        assert "_sp_mask" in block


def test_runner_refuses_an_existing_run_directory(tmp_path):
    """Orbax only refuses to overwrite at the save, which is after training.

    On a multi-hour job that discards the whole run, so the check belongs before
    it starts.

    The path matters as much as the check: this asserted
    ``run_dir/saved_train_run`` while the checkpoint is written to
    ``run_dir/artifacts/saved_train_run``, so it passed against a guard that
    could never fire on a real run.
    """
    import pathlib

    from oaht_bench.teammate_gen.runner import run

    job = _job(output_dir=str(tmp_path))
    stale = pathlib.Path(job.run_dir()) / "artifacts" / "saved_train_run"
    stale.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        run(job)


def test_guard_watches_the_path_the_checkpoint_is_written_to(tmp_path):
    """Pin the guarded path to where the checkpoint actually lands.

    Without this the guard can drift back to a path nothing writes and still
    look tested, which is what happened.
    """
    import pathlib

    from oaht_bench.population import artifact_dir

    job = _job(output_dir=str(tmp_path))
    run_dir = pathlib.Path(job.run_dir())
    written = run_dir / "artifacts" / "saved_train_run"
    written.mkdir(parents=True)

    # The locator the re-scorer uses and the path the guard watches must agree.
    assert artifact_dir(run_dir) == written.resolve()


def test_evaluation_defaults_to_sampled_actions():
    """Greedy deadlocks symmetric coordination tasks; sampling is the default.

    Argmax also erases the policy entropy, so a sweep over ``entropy_coef``
    cannot observe what it varies.
    """
    assert _job().evaluation_greedy is False


# --- hyperparameter sweeps ---------------------------------------------------


def test_sweep_expands_the_grid_and_hashes_each_cell(tmp_path):
    """Every cell is a distinct config, so a single one can be re-run or cited."""
    from scripts.sweep import generate

    base = tmp_path / "base.json"
    save_job(_job(), base)
    out = generate(
        base,
        "s",
        {"generator.population_size": [3, 4], "generator.num_envs": [8, 16]},
        tmp_path / "sweeps",
    )
    assert len(out) == 4
    hashes = {load_job(p).content_hash() for p, _ in out}
    assert len(hashes) == 4


def test_sweep_rejects_an_unrunnable_cell_before_writing(tmp_path):
    """A cell that cannot train must fail at generation.

    Queueing it would burn a scheduler slot and report nothing -- and a partial
    sweep left on disk looks complete.
    """
    from oaht_bench.configs.teammate_gen import BrDivConfig
    from scripts.sweep import generate

    base = tmp_path / "base.json"
    save_job(_job(generator=BrDivConfig(population_size=3, num_envs=16, total_timesteps=1e6)), base)
    with pytest.raises(SystemExit, match="not runnable"):
        generate(base, "s", {"generator.total_timesteps": [1e6, 10]}, tmp_path / "sweeps")
    assert not (tmp_path / "sweeps" / "s").exists()


def test_sweep_manifest_records_cost_per_cell(tmp_path):
    """Sizing a sweep needs the update count, not just the cell count."""
    import json

    from scripts.sweep import generate

    base = tmp_path / "base.json"
    save_job(_job(), base)
    generate(base, "s", {"generator.population_size": [3, 4]}, tmp_path / "sweeps")
    manifest = json.loads((tmp_path / "sweeps" / "s" / "sweep.json").read_text())
    assert len(manifest["cells"]) == 2
    for cell in manifest["cells"]:
        assert cell["sequential_updates"] >= 1
        assert cell["config_hash"]


# --- population cross-play and sweep ranking --------------------------------


def test_crossplay_separation_is_sp_minus_xp():
    import numpy as np

    from oaht_bench.population import CrossPlayScores

    s = CrossPlayScores(matrix=np.eye(3), self_play=0.6, cross_play=0.1)
    assert s.separation == pytest.approx(0.5)


def test_crossplay_single_member_has_no_cross_play():
    """Reporting 0 would read as perfect separation for a population of one."""
    import math

    import jax
    import numpy as np

    from oaht_bench.population import evaluate_population

    class _Pop:
        pop_size = 1
        policy_cls = None

    calls = {}

    def fake_run_episodes(*a, **k):
        calls["n"] = calls.get("n", 0) + 1
        return {"returned_episode_returns": np.array([0.5])}

    import oaht_bench.population.crossplay as mod

    original, mod.run_episodes = mod.run_episodes, fake_run_episodes
    try:
        scores = evaluate_population(
            env=None,
            params={"w": np.zeros((1, 1, 2))},
            population=_Pop(),
            rng=jax.random.PRNGKey(0),
            max_episode_steps=8,
            num_episodes=1,
        )
    finally:
        mod.run_episodes = original

    assert scores.self_play == pytest.approx(0.5)
    assert math.isnan(scores.cross_play)


def test_sweep_ranks_competence_before_separation(tmp_path):
    """Separation alone would select a collapsed population.

    Cross-play falls both when members are genuinely distinct and when they
    cannot score at all, so competence has to gate the ranking.
    """
    import json

    import numpy as np

    import scripts.sweep as sw

    cases = {
        "good": (0.60, 0.10),  # competent and diverse
        "similar": (0.62, 0.55),  # competent, low separation
        "collapsed": (0.02, 0.00),  # incompetent, huge apparent separation
    }
    cells = []
    for name, (sp, xp) in cases.items():
        run = tmp_path / name
        run.mkdir()
        m = np.full((3, 3), xp)
        np.fill_diagonal(m, sp)
        np.savetxt(run / "population_crossplay.csv", m, delimiter=",")
        cells.append({"config": f"{name}.json", "settings": {"x": name}})
    (tmp_path / "sweep.json").write_text(
        json.dumps({"name": "t", "axes": {"x": []}, "cells": cells})
    )

    sp_scores = {n: sw._population_scores(tmp_path / n) for n in cases}
    assert sp_scores["collapsed"][0] < sp_scores["good"][0]
    # 'collapsed' has the largest separation but must not be selected.
    seps = {n: s - x for n, (s, x) in sp_scores.items()}
    best_sp = max(s for s, _ in sp_scores.values())
    band = [n for n, (s, _) in sp_scores.items() if s >= best_sp * 0.95]
    assert "collapsed" not in band
    assert max(band, key=lambda n: seps[n]) == "good"


def test_evaluation_episodes_is_configurable():
    """Cost is population_size^2 x this, so it must not be a magic number."""
    job = _job()
    assert job.evaluation_episodes > 0
    assert _job(evaluation_episodes=5).evaluation_episodes == 5


def test_crossplay_uses_the_partner_population_when_given():
    """BRDiv and L-BRDiv must be scored conf_i vs br_j, not conf_i vs conf_j.

    Their designed-optimal pairing is a confederate with *its own* best response.
    Pairing a confederate with a copy of itself is out-of-distribution -- they are
    never trained to play with themselves -- and under-reports competence.
    """
    import jax
    import numpy as np

    import oaht_bench.population.crossplay as mod

    rows = {"w": np.array([[[0.0], [1.0]]])}  # (seed, member, dim)
    cols = {"w": np.array([[[10.0], [11.0]]])}
    seen = []

    def fake_run_episodes(
        rng, env, *, agent_0_param, agent_0_policy, agent_1_param, agent_1_policy, **kw
    ):
        seen.append((float(agent_0_param["w"][0]), float(agent_1_param["w"][0])))
        return {"returned_episode_returns": np.array([0.0])}

    class _Pop:
        pop_size = 2
        policy_cls = None

    original, mod.run_episodes = mod.run_episodes, fake_run_episodes
    try:
        mod.evaluate_population(
            env=None,
            params=rows,
            population=_Pop(),
            rng=jax.random.PRNGKey(0),
            max_episode_steps=4,
            num_episodes=1,
            partner_params=cols,
        )
    finally:
        mod.run_episodes = original

    # Seat 0 always a row member, seat 1 always a column member.
    assert seen == [(0.0, 10.0), (0.0, 11.0), (1.0, 10.0), (1.0, 11.0)]


def test_crossplay_self_pairs_when_no_partner_given():
    """FCP and CoMeDi release one set of self-play policies; the diagonal is i vs i."""
    import jax
    import numpy as np

    import oaht_bench.population.crossplay as mod

    rows = {"w": np.array([[[0.0], [1.0]]])}
    seen = []

    def fake_run_episodes(rng, env, *, agent_0_param, agent_1_param, **kw):
        seen.append((float(agent_0_param["w"][0]), float(agent_1_param["w"][0])))
        return {"returned_episode_returns": np.array([0.0])}

    class _Pop:
        pop_size = 2
        policy_cls = None

    original, mod.run_episodes = mod.run_episodes, fake_run_episodes
    try:
        mod.evaluate_population(
            env=None,
            params=rows,
            population=_Pop(),
            rng=jax.random.PRNGKey(0),
            max_episode_steps=4,
            num_episodes=1,
        )
    finally:
        mod.run_episodes = original

    assert seen == [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]


# --- FCP is scored on converged checkpoints ---------------------------------


def test_fcp_scores_only_converged_checkpoints():
    """One member per run: the convention that run arrived at.

    FCP's flattening is (seeds, runs, ckpts, ...) -> (seeds, runs*ckpts, ...) in
    C order, so the converged member of run r is at r*num_checkpoints + ckpts-1.
    """
    from oaht_bench.configs.teammate_gen import FcpConfig
    from oaht_bench.population import released_members

    job = _job(generator=FcpConfig(population_size=4, num_checkpoints=3))
    # pop_size is the loaded population's width, 4 runs x 3 checkpoints.
    assert released_members(job, 12) == [2, 5, 8, 11]


def test_other_generators_score_every_member():
    """They release one member per convention already.

    Returned as an explicit list rather than ``None``-means-all, so no caller has
    to remember a sentinel.
    """
    from oaht_bench.configs.teammate_gen import BrDivConfig, CoMeDiConfig, LBrDivConfig
    from oaht_bench.population import released_members

    for gen in (CoMeDiConfig(), BrDivConfig(), LBrDivConfig()):
        assert released_members(_job(generator=gen), 5) == [0, 1, 2, 3, 4]


def test_fcp_scoring_does_not_reward_dropping_checkpoints():
    """The regression this guards against.

    Averaging self-play over all members makes num_checkpoints look harmful,
    because early checkpoints are deliberately weak -- so a sweep would drive it
    to 1 and reproduce the FCP-T ablation the paper shows is significantly worse.
    Scoring converged members only makes the count of scored members depend on
    population_size alone.
    """
    from oaht_bench.configs.teammate_gen import FcpConfig
    from oaht_bench.population import released_members

    few = released_members(_job(generator=FcpConfig(population_size=5, num_checkpoints=2)), 10)
    many = released_members(_job(generator=FcpConfig(population_size=5, num_checkpoints=8)), 40)
    assert len(few) == len(many) == 5


def test_member_indices_selects_the_submatrix():
    """Only the requested members are evaluated, and in the requested order."""
    import jax
    import numpy as np

    import oaht_bench.population.crossplay as mod

    params = {"w": np.array([[[0.0], [1.0], [2.0], [3.0]]])}
    seen = []

    def fake_run_episodes(rng, env, *, agent_0_param, agent_1_param, **kw):
        seen.append((float(agent_0_param["w"][0]), float(agent_1_param["w"][0])))
        return {"returned_episode_returns": np.array([0.0])}

    class _Pop:
        pop_size = 4
        policy_cls = None

    original, mod.run_episodes = mod.run_episodes, fake_run_episodes
    try:
        scores = mod.evaluate_population(
            env=None,
            params=params,
            population=_Pop(),
            rng=jax.random.PRNGKey(0),
            max_episode_steps=4,
            num_episodes=1,
            member_indices=[1, 3],
        )
    finally:
        mod.run_episodes = original

    assert scores.matrix.shape == (2, 2)
    assert seen == [(1.0, 1.0), (1.0, 3.0), (3.0, 1.0), (3.0, 3.0)]


# --- post-training reporting must not be able to destroy a finished run ------


def test_xp_matrix_columns_follow_the_matrix_width(tmp_path):
    """wandb.Table defaults to 3 columns, so any other width was rejected.

    This passed silently while populations were size 3 and became a hard failure
    the moment population_size moved to 5 -- after training had completed.
    """
    import numpy as np

    from oaht_bench.common.logging import RunLogger

    captured = {}

    class _FakeWandb:
        def Table(self, columns, data):  # noqa: N802 - mirrors wandb's API
            if len(data) and len(columns) != len(data[0]):
                raise ValueError(f"This table expects {len(columns)} columns, found {len(data[0])}")
            captured["columns"] = columns
            return object()

        def log(self, *a, **k):
            pass

    with RunLogger(tmp_path, use_wandb=False) as logger:
        logger.run = object()  # pretend wandb is active
        import sys

        sys.modules["wandb"] = _FakeWandb()
        try:
            for n in (3, 5, 8):
                logger.log_xp_matrix(f"Eval/M{n}", np.zeros((n, n)))
                assert len(captured["columns"]) == n
        finally:
            del sys.modules["wandb"]


def test_nonfatal_lets_a_finished_run_survive_a_reporting_failure():
    """A charting bug after hours of training must not lose the checkpoint."""
    from oaht_bench.common.logging import nonfatal

    reached = []
    with nonfatal("test reporting"):
        raise ValueError("this table expects 3 columns")
    reached.append("after")
    assert reached == ["after"]


def test_every_generator_saves_before_it_reports():
    """The checkpoint write must precede post-training logging in all four.

    BRDiv and L-BRDiv had save_train_run as the last statement of log_metrics, so
    a failure in any chart above it discarded a completed run. Pin the ordering.
    """
    import inspect

    from oaht_bench.teammate_gen import BRDiv, CoMeDi, LBRDiv, fcp

    for mod, runner in (
        (fcp, "run_fcp"),
        (CoMeDi, "run_comedi"),
        (BRDiv, "run_brdiv"),
        (LBRDiv, "run_lbrdiv"),
    ):
        src = inspect.getsource(getattr(mod, runner))
        assert "save_train_run(" in src, f"{runner} must save its own checkpoint"
        assert src.index("save_train_run(") < src.index("log_metrics("), (
            f"{runner} reports before it saves; a charting failure would lose the run"
        )
        assert "nonfatal(" in src, f"{runner} must not let reporting raise"

    # And log_metrics must no longer do the saving itself.
    for mod in (fcp, CoMeDi, BRDiv, LBRDiv):
        assert "save_train_run(" not in inspect.getsource(mod.log_metrics), (
            f"{mod.__name__}.log_metrics still writes the checkpoint"
        )


def test_paired_generators_scale_envs_with_population_squared():
    """BRDiv/L-BRDiv sample conf and br ids independently, so a specific
    ``(conf_i, br_j)`` pairing gets ``num_envs / n**2`` samples per rollout.

    Upstream tuned at n=3 with 64 envs, i.e. 7.1 environments per pairing. Moving
    to n=5 while leaving num_envs alone gave 2.6 and BRDiv collapsed: the final
    cross-play matrix was uniform to within noise and self-play fell *below*
    cross-play, inverting the quantity the method maximizes.

    The loss weighting genuinely is population-size invariant, which is what made
    this easy to miss -- it is the data behind each pairing that binds.
    """
    from oaht_bench.configs import load_job

    reference = 64 / 3**2  # the n=3 setting that produced a working population
    for env in ("lbf_12x12", "overcooked_counter_circuit", "hanabi"):
        for name in ("brdiv", "lbrdiv"):
            gen = load_job(f"configs/teammate_gen/{env}/{name}.json").generator
            per_pairing = gen.num_envs / gen.population_size**2
            assert per_pairing >= reference, (
                f"{env}/{name}: {per_pairing:.1f} envs per (conf, br) pairing is "
                f"below the {reference:.1f} that worked at n=3; raise num_envs "
                f"with the square of population_size"
            )


def test_paired_env_scaling_does_not_cost_gradient_steps():
    """Scaling num_envs without the budget would trade one failure for another.

    num_updates = total_timesteps // rollout_length // num_envs, so tripling the
    environments at a fixed budget cuts the update count to a third -- which is
    separately known to wreck these populations (see docs/tuning_record.md).
    """
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "gen_cfgs", pathlib.Path("scripts/gen_teammate_configs.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    base_envs, base_ts = 64, 4.5e7
    scaled = mod._paired_scale(base_envs, base_ts)
    # updates ~ timesteps / envs, so the ratio must be preserved exactly
    assert scaled["total_timesteps"] / scaled["num_envs"] == base_ts / base_envs
    assert scaled["pop"] == mod.POPULATION_SIZE


def test_paired_populations_seat_a_best_response_opposite_a_confederate():
    """BRDiv/L-BRDiv seats have roles; filling both from confederates is OOD.

    A confederate is never trained to play with another confederate. Before
    LoadedPopulation carried both halves, population_from_run returned only
    final_params_conf, so dataset collection seated conf-vs-conf and lost 25-40%
    of the population's designed return on LBF.
    """
    from oaht_bench.population.loading import LoadedPopulation

    conf = {"w": __import__("numpy").arange(12).reshape(1, 3, 4)}
    br = {"w": __import__("numpy").arange(100, 112).reshape(1, 3, 4)}
    policy = object()

    paired = LoadedPopulation(
        params=conf, policy_cls=policy, pop_size=3, generator="brdiv", partner_params=br
    )
    assert paired.paired
    seats = paired.seat([0, 1])
    # seat 0 from confederates, seat 1 from best responses
    assert seats[0][0]["w"].tolist() == [0, 1, 2, 3]
    assert seats[1][0]["w"].tolist() == [104, 105, 106, 107]

    homogeneous = LoadedPopulation(params=conf, policy_cls=policy, pop_size=3, generator="fcp")
    assert not homogeneous.paired
    seats = homogeneous.seat([0, 1])
    assert seats[0][0]["w"].tolist() == [0, 1, 2, 3]
    assert seats[1][0]["w"].tolist() == [4, 5, 6, 7]


def test_paired_population_refuses_a_third_seat():
    """The paired generators assert num_agents == 2 during training too."""
    import numpy as np
    import pytest

    from oaht_bench.population.loading import LoadedPopulation

    paired = LoadedPopulation(
        params={"w": np.zeros((1, 3, 2))},
        policy_cls=object(),
        pop_size=3,
        generator="brdiv",
        partner_params={"w": np.zeros((1, 3, 2))},
    )
    with pytest.raises(ValueError, match="no role for a third seat"):
        paired.seat([0, 1, 2])


# --- dataset seating -------------------------------------------------------


def test_seat_plan_splits_by_count_not_by_coin_flip():
    """mismatch_fraction is a property of the dataset, not a draw.

    A per-episode Bernoulli gives the fraction only in expectation: at 12
    episodes and p=0.5 it produced 25%, because 9 of that seed's 12 uniforms
    happened to fall below the threshold. Allocating counts up front makes the
    number exact.
    """
    import numpy as np

    from oaht_bench.dataset.construction.runner import _seat_plan

    for n, frac in ((10, 0.5), (10, 0.3), (100, 0.25), (7, 0.0), (7, 1.0)):
        plan = _seat_plan([0, 1, 2, 3, 4], n, frac, np.random.default_rng(0))
        assert len(plan) == n
        assert sum(a != b for a, b in plan) == round(n * frac)


def test_matched_and_mismatched_pools_are_disjoint():
    """Each fraction draws only from its own pool.

    A matched episode can never come out mismatched and vice versa, which is
    what makes the two counts mean exactly what they say.
    """
    import numpy as np

    from oaht_bench.dataset.construction.runner import _seat_plan

    plan = _seat_plan([0, 1, 2, 3, 4], 10, 0.5, np.random.default_rng(0))
    matched = [p for p in plan if p[0] == p[1]]
    mismatched = [p for p in plan if p[0] != p[1]]
    assert sorted(matched) == [(i, i) for i in range(5)]
    assert all(a != b for a, b in mismatched)


def test_seat_plan_covers_every_teammate_equally():
    """Uneven coverage is what forces the contrastive sampler to compensate.

    Cycling a shuffled pool rather than sampling with replacement means 10
    episodes over 5 members give each exactly two turns as the modelled agent.
    """
    import collections

    import numpy as np

    from oaht_bench.dataset.construction.runner import _seat_plan

    for n in (10, 20, 100):
        plan = _seat_plan([0, 1, 2, 3, 4], n, 0.5, np.random.default_rng(1))
        counts = collections.Counter(a for a, _ in plan)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_mismatch_needs_two_members():
    import numpy as np
    import pytest as _pytest

    from oaht_bench.dataset.construction.runner import _seat_plan

    with _pytest.raises(ValueError, match="at least two distinct"):
        _seat_plan([3], 10, 0.5, np.random.default_rng(0))


# --- AD-RPG (clean-room reimplementation) -----------------------------------


def test_rpg_config_to_algorithm_dict_carries_diversity_knobs():
    """The resolved-config dump must record the RPG-specific hyperparameters."""
    from oaht_bench.configs.teammate_gen import RpgConfig

    d = RpgConfig(
        population_size=3,
        partnerplay_ratio=0.1,
        off_diag_factor=0.25,
        n_lookahead=2,
        dice_lambda=0.95,
    ).to_algorithm_dict()
    assert d["ALG"] == "rpg"
    assert d["N_LOOKAHEAD"] == 2
    assert d["DICE_LAMBDA"] == 0.95
    assert d["PARTNERPLAY_RATIO"] == 0.1
    assert d["OFF_DIAG_FACTOR"] == 0.25


def test_rpg_runtime_derives_outer_updates():
    from oaht_bench.configs.teammate_gen import RpgConfig
    from oaht_bench.teammate_gen.runtime import RpgRuntime

    gen = RpgConfig(population_size=2, num_envs=8, total_timesteps=8192)
    rt = RpgRuntime.from_config(gen, rollout_length=128, num_agents=2)
    assert rt.num_updates == int(8192 // (128 * 8))
    assert rt.num_actors == 2 * 8


def test_rpg_runtime_needs_at_least_two_particles():
    """A single particle has no cross-play term -- that is FCP, not diversity."""
    from oaht_bench.configs.teammate_gen import RpgConfig
    from oaht_bench.teammate_gen.runtime import RpgRuntime

    gen = RpgConfig(population_size=1, num_envs=8, total_timesteps=1e6)
    with pytest.raises(ValueError, match="at least 2 particles"):
        RpgRuntime.from_config(gen, rollout_length=128, num_agents=2)


def test_rpg_runtime_rejects_a_negative_self_play_weight():
    """N * partnerplay_ratio must stay below 1 or the base self-play weight flips."""
    from oaht_bench.configs.teammate_gen import RpgConfig
    from oaht_bench.teammate_gen.runtime import RpgRuntime

    gen = RpgConfig(population_size=5, partnerplay_ratio=0.3, num_envs=8, total_timesteps=1e6)
    with pytest.raises(ValueError, match="self-play weight"):
        RpgRuntime.from_config(gen, rollout_length=128, num_agents=2)


def test_rpg_runtime_requires_two_agents():
    from oaht_bench.configs.teammate_gen import RpgConfig
    from oaht_bench.teammate_gen.runtime import RpgRuntime

    gen = RpgConfig(population_size=2, num_envs=8, total_timesteps=1e6)
    with pytest.raises(ValueError, match="exactly 2 agents"):
        RpgRuntime.from_config(gen, rollout_length=128, num_agents=3)


def test_training_plan_counts_rpg_lookahead_and_particles():
    """One outer step is n_lookahead base updates + one manipulator update; the N
    particles are vmapped, so they are parallel members."""
    from oaht_bench.configs.teammate_gen import RpgConfig
    from oaht_bench.teammate_gen.plan import training_plan

    gen = RpgConfig(population_size=3, num_envs=8, total_timesteps=1e6, n_lookahead=2)
    plan = training_plan(_job(generator=gen))
    assert plan.parallel_members == 3
    assert plan.updates_per_unit == 3  # n_lookahead + 1
    assert plan.sequential_updates == 3 * plan.sequential_units
