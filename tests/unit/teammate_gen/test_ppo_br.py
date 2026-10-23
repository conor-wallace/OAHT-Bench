"""`ppo_br` checkpoint-saving and TIPR Optimal Ratio (rho) unit tests.

Pins the two things the rho mechanism depends on: `make_br_train` actually
populates its checkpoint array at the expected update steps (and the last one
always equals `final_params`), and `select_ratio_checkpoint` breaks ties
toward the latest checkpoint -- otherwise ratio=1.0 does not reliably
degenerate to `load_ego_policies`'s own selection whenever two checkpoints
tie on return (as they do verbatim on a curve flat enough that no episode
completes within it, the exact case the end-to-end smoke run below hits).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest


def _mep_src_gen():
    from oaht_bench.configs.teammate_gen import MepConfig

    return MepConfig(
        population_size=2,
        num_envs=4,
        total_timesteps=32,
        num_seeds=1,
        population_entropy_coef=0.01,
        train_seed=0,
    )


def _train_tiny_ppo_br(tmp_path, *, num_checkpoints, total_timesteps=192, ego_init="warm_start"):
    """Trains a 2-member MEP source, releases it, then trains ppo_br against it.

    Returns ``(br_run_dir, env, released_dir)``. Shared by the checkpoint and
    end-to-end tests below -- each training step here is real (no mocking).
    """
    from oaht_bench.configs import get_preset
    from oaht_bench.configs.job import TeammateGenerationJob
    from oaht_bench.configs.teammate_gen import PpoBrConfig
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.teammate_gen.runner import run as run_teammate_gen
    from scripts.release_population import release

    env_cfg = get_preset("lbf_12x12").model_copy(update={"rollout_length": 8})
    env = LogWrapper(make_env(env_cfg.env_name, env_cfg.env_kwargs()))

    mep_job = TeammateGenerationJob(
        label="mep_src", env=env_cfg, generator=_mep_src_gen(), output_dir=str(tmp_path / "results")
    )
    mep_run_dir = run_teammate_gen(mep_job)
    released = release(mep_run_dir, tmp_path / "populations")

    br_gen = PpoBrConfig(
        source_population_path=str(released),
        num_envs=4,
        total_timesteps=total_timesteps,
        num_checkpoints=num_checkpoints,
        train_seed=0,
        ego_init=ego_init,
    )
    br_job = TeammateGenerationJob(
        label="ppo_br_test", env=env_cfg, generator=br_gen, output_dir=str(tmp_path / "results")
    )
    br_run_dir = run_teammate_gen(br_job)
    return br_run_dir, env, released, env_cfg


def test_make_br_train_populates_checkpoints_and_last_matches_final(tmp_path):
    """`checkpoints[:, -1]` must equal `final_params` exactly -- the same invariant
    MEP's own checkpointing pins (same `_is_checkpoint_step`/`_maybe_store_checkpoint`
    helpers, now shared via `teammate_gen/marl/ppo_utils.py`)."""
    from oaht_bench.common.save_load_utils import load_train_run
    from oaht_bench.population import artifact_dir

    br_run_dir, _env, _released, _env_cfg = _train_tiny_ppo_br(tmp_path, num_checkpoints=3)

    out = load_train_run(str(artifact_dir(br_run_dir)))
    assert "checkpoints_by_source" in out
    assert "ego_return_curve_by_source" in out

    checkpoints = out["checkpoints_by_source"]["0"]
    final_params = out["br_by_source"]["0"]
    num_checkpoints = jax.tree_util.tree_leaves(checkpoints)[0].shape[1]
    assert num_checkpoints == 3

    last_checkpoint = jax.tree.map(lambda x: x[:, -1], checkpoints)
    for a, b in zip(
        jax.tree_util.tree_leaves(last_checkpoint),
        jax.tree_util.tree_leaves(final_params),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize(
    "curve,num_checkpoints,ratio,expected_idx",
    [
        # Monotonic increase: ratio=0.5 should land near the middle checkpoint.
        (np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]), 3, 0.5, 1),
        # ratio=1.0 always resolves to the last checkpoint, even on a strictly
        # increasing curve where it is also the unique closest match.
        (np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]), 3, 1.0, 2),
        # Non-monotonic/noisy curve: nearest-return checkpoint still wins -- the
        # checkpointed steps are 1, 4, 6 (returns 0, 6, 10), and 0.6 * 10 = 6
        # matches the middle one exactly.
        (np.array([0.0, 5.0, 1.0, 6.0, 3.0, 10.0]), 3, 0.6, 1),
        # Flat curve: every checkpoint return ties exactly, so ratio=1.0 must
        # still resolve to the last checkpoint (the tie-break this test pins).
        (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 3, 1.0, 2),
    ],
)
def test_select_ratio_checkpoint_synthetic_curves(curve, num_checkpoints, ratio, expected_idx):
    from oaht_bench.population.loading import select_ratio_checkpoint

    # One scalar "param" per checkpoint, valued by its own index, so the
    # selected index is directly observable from the returned "params".
    checkpoints = jax.numpy.arange(num_checkpoints, dtype=jax.numpy.float32)
    idx = int(select_ratio_checkpoint(checkpoints, curve, ratio=ratio))
    assert idx == expected_idx


def test_load_ego_policies_at_ratio_one_matches_load_ego_policies(tmp_path):
    """ratio=1.0 must select parameters identical to `load_ego_policies`'s own BR --
    the degenerate case the ρ mechanism must never change existing behavior for."""
    from oaht_bench.population.loading import load_ego_policies, load_ego_policies_at_ratio

    br_run_dir, env, _released, _env_cfg = _train_tiny_ppo_br(tmp_path, num_checkpoints=3)

    converged = load_ego_policies(br_run_dir, env)
    at_ratio_one = load_ego_policies_at_ratio(br_run_dir, env, ratio=1.0)
    assert len(converged) == len(at_ratio_one)
    for a, b in zip(converged, at_ratio_one, strict=True):
        assert (a.generator, a.member, a.role) == (b.generator, b.member, b.role)
        for la, lb in zip(
            jax.tree_util.tree_leaves(a.params), jax.tree_util.tree_leaves(b.params), strict=True
        ):
            np.testing.assert_allclose(np.asarray(la), np.asarray(lb))


def test_load_ego_policies_at_ratio_raises_without_checkpoints(tmp_path):
    """A ppo_br artifact trained with num_checkpoints=1 (the old default behavior)
    has no checkpoints_by_source/ego_return_curve_by_source -- requesting a ratio
    against it must raise clearly, not silently fall back to the converged BR."""
    import json

    import jax.numpy as jnp

    from oaht_bench.common.save_load_utils import save_train_run
    from oaht_bench.population.loading import load_ego_policies_at_ratio

    run_dir = tmp_path / "old_style_run"
    run_dir.mkdir()
    (run_dir / "br_manifest.json").write_text(json.dumps([]))
    save_train_run(
        {"br_by_source": {"0": {"w": jnp.zeros((1, 4))}}},
        str(run_dir),
        savename="saved_train_run",
    )

    with pytest.raises(ValueError, match="checkpoints_by_source"):
        load_ego_policies_at_ratio(run_dir, env=None, ratio=0.5)


def test_pooled_dataset_collection_with_ego_ratio_end_to_end(tmp_path):
    """A tiny ppo_br (num_checkpoints=3) feeding a tiny pooled dataset-collection
    job with ego_ratio=0.5 runs without error and produces a dataset + a
    teammate_split-consistent manifest, mirroring collect_batched's smoke coverage."""
    from oaht_bench.configs.job import DatasetCollectionJob, PooledCrossplayJob
    from oaht_bench.dataset.construction import runner as dc_runner
    from oaht_bench.population import pooled_crossplay

    br_run_dir, _env, released, env_cfg = _train_tiny_ppo_br(tmp_path, num_checkpoints=3)

    pcj = PooledCrossplayJob(
        label="pooled_xp_test",
        env=env_cfg,
        population_path=[str(released)],
        br_population_path=str(br_run_dir),
        num_episodes=2,
        seed=0,
        output_dir=str(tmp_path / "results"),
    )
    matrix_dir = pooled_crossplay.run(pcj)
    matrix_path = matrix_dir / "pooled_crossplay.npz"
    assert matrix_path.exists()

    dcj = DatasetCollectionJob(
        label="pooled_dc_test",
        env=env_cfg,
        population_path=[str(released)],
        pooled_matrix_path=str(matrix_path),
        br_population_path=str(br_run_dir),
        variant="expert",
        num_episodes=4,
        holdout_per_generator=0,
        ego_ratio=0.5,
        output_dir=str(tmp_path / "results"),
    )
    run_dir = dc_runner.run(dcj)
    assert (run_dir / "dataset.vlt").exists()
    assert (run_dir / "teammate_split.json").exists()


def test_initialize_agent_gives_each_member_a_different_random_init():
    """The core correctness property ego_init='scratch' depends on: splitting
    one base key into per-member rngs (as _train_source's scratch_rngs does)
    and feeding each to initialize_agent must NOT hand every member the same
    draw -- that would silently defeat the whole point (independent BRs),
    reproducing one random init instead of len(members) of them."""
    from oaht_bench.configs import get_preset
    from oaht_bench.envs import make_env
    from oaht_bench.teammate_gen.marl.ippo import initialize_agent
    from oaht_bench.teammate_gen.runtime import PpoRuntime

    env_cfg = get_preset("lbf_12x12")
    env = make_env(env_cfg.env_name, env_cfg.env_kwargs())
    rt = PpoRuntime.from_config(
        ppo=_mep_src_gen().ppo,
        network=_mep_src_gen().network,
        actor_type="mlp",
        rollout_length=8,
        num_envs=4,
        total_timesteps=32,
        num_checkpoints=1,
        num_agents=env.num_agents,
    )
    scratch_rngs = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(0), 0), 2)

    _, p0 = initialize_agent(rt.actor_type, rt.to_agent_dict(), env, scratch_rngs[0])
    _, p1 = initialize_agent(rt.actor_type, rt.to_agent_dict(), env, scratch_rngs[1])

    leaves0 = jax.tree_util.tree_leaves(p0)
    leaves1 = jax.tree_util.tree_leaves(p1)
    assert len(leaves0) == len(leaves1)
    assert any(
        not np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(leaves0, leaves1, strict=True)
    )


def test_ppo_br_scratch_init_runs_and_is_recorded_in_the_manifest(tmp_path):
    """ego_init='scratch' runs end to end (no warm-start params ever touched)
    and the manifest records which path produced each BR, for provenance."""
    import json

    from oaht_bench.population import artifact_dir

    br_run_dir, _env, _released, _env_cfg = _train_tiny_ppo_br(
        tmp_path, num_checkpoints=1, ego_init="scratch"
    )

    manifest = json.loads((br_run_dir / "br_manifest.json").read_text())
    assert len(manifest) == 2  # the 2-member MEP source
    assert all(entry["ego_init"] == "scratch" for entry in manifest)

    # Still a loadable artifact, same shape as the warm_start path.
    from oaht_bench.common.save_load_utils import load_train_run

    out = load_train_run(str(artifact_dir(br_run_dir)))
    assert "br_by_source" in out
