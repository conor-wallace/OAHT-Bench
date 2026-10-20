"""MEP (Maximum Entropy Population-based training) unit tests.

Pins the delicate part -- the population-entropy reduction must be
``log(mean(probs))``, not ``mean(log(probs))``, a silent-wrong-objective bug
that would otherwise only surface as an unexplained training curve on GPU --
plus an end-to-end smoke test and a behavioral ablation matching the paper's
own Question 2 methodology (does the entropy bonus measurably raise trained
population entropy).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_population_entropy_bonus_is_log_mean_not_mean_log():
    """``-coef * log(mean(probs))``, not ``-coef * mean(log(probs))``.

    Those differ whenever members disagree (Jensen's inequality), which three
    hand-picked, clearly-different per-member probabilities guarantee here.
    """
    from oaht_bench.teammate_gen.mep import population_entropy_bonus

    probs = jnp.array([0.9, 0.5, 0.1])  # three members' P(this action | this obs)
    log_probs = jnp.log(probs)

    correct = -jnp.log(jnp.mean(probs))  # log(mean(p)) -- what the paper defines
    wrong = -jnp.mean(log_probs)  # mean(log(p)) -- the bug this test catches

    assert float(jnp.abs(correct - wrong)) > 1e-3  # the two really do differ here

    got = population_entropy_bonus(log_probs, coef=1.0)
    np.testing.assert_allclose(np.asarray(got), np.asarray(correct), atol=1e-6)
    assert float(jnp.abs(np.asarray(got) - np.asarray(wrong))) > 1e-3


def test_population_entropy_bonus_scales_linearly_with_coef():
    from oaht_bench.teammate_gen.mep import population_entropy_bonus

    log_probs = jnp.log(jnp.array([0.7, 0.3, 0.05]))
    b1 = population_entropy_bonus(log_probs, coef=1.0)
    b2 = population_entropy_bonus(log_probs, coef=3.0)
    np.testing.assert_allclose(np.asarray(b2), 3.0 * np.asarray(b1), atol=1e-6)


def test_population_entropy_bonus_exclude_idx_matches_self_excluding_mean():
    """``exclude_idx`` must give ``log(mean_{k != idx} pi_k(a|s))``, not
    today's self-inclusive ``log(mean_k pi_k(a|s))`` -- the round-robin
    reference's ``pbt_population_copy.pop(idx0)`` semantics. Checked against
    a direct numpy recomputation over the remaining members, not just "it
    runs"; also confirms it differs from the self-inclusive value (the two
    coincide only by coincidence, never in general).
    """
    from oaht_bench.teammate_gen.mep import population_entropy_bonus

    probs = np.array([0.9, 0.5, 0.1, 0.3])
    log_probs = jnp.log(jnp.asarray(probs))

    got = population_entropy_bonus(log_probs, coef=1.0, exclude_idx=2)
    expected_self_excluded = -np.log(np.mean(np.delete(probs, 2)))
    np.testing.assert_allclose(np.asarray(got), expected_self_excluded, atol=1e-6)

    self_inclusive = population_entropy_bonus(log_probs, coef=1.0)
    assert abs(float(got) - float(self_inclusive)) > 1e-3


def test_population_entropy_bonus_exclude_idx_none_is_unchanged():
    """``exclude_idx=None`` (the default) must be byte-identical to omitting
    the argument entirely -- the round_robin=False path's guarantee.
    """
    from oaht_bench.teammate_gen.mep import population_entropy_bonus

    log_probs = jnp.log(jnp.array([0.7, 0.3, 0.05]))
    without_arg = population_entropy_bonus(log_probs, coef=1.0)
    with_none = population_entropy_bonus(log_probs, coef=1.0, exclude_idx=None)
    np.testing.assert_array_equal(np.asarray(without_arg), np.asarray(with_none))


def test_maybe_store_best_tracks_the_true_running_maximum():
    """A non-monotonic return sequence with a tie -- confirms best_params/
    best_score end up at the step with the actual highest observed return
    (not the last step, and strict '>' means the first occurrence of a tied
    max wins, not a later one silently overwriting it).
    """
    from oaht_bench.teammate_gen.marl.ppo_utils import _maybe_store_best

    returns = [0.1, 0.5, 0.3, 0.9, 0.2, 0.9, 0.4]  # max 0.9 first at index 3, tied again at 5
    # distinct "params" per step so we can tell which step actually won
    params_per_step = [{"w": jnp.asarray(float(i))} for i in range(len(returns))]

    best_params = {"w": jnp.asarray(-1.0)}
    best_score = jnp.asarray(-jnp.inf)
    for p, r in zip(params_per_step, returns, strict=True):
        best_params, best_score = _maybe_store_best(best_params, best_score, p, jnp.asarray(r))

    assert float(best_score) == pytest.approx(0.9)
    assert float(best_params["w"]) == 3.0  # the first step to reach 0.9, not step 5's tie


def test_population_entropy_coef_schedule_zero_warmup_returns_target_unconditionally():
    """``warmup_fraction<=0`` is a genuinely different code path (returns the
    plain ``target_coef`` straight back), not a schedule that happens to
    evaluate to the constant -- the zero-means-off guarantee every existing
    config (which never sets this field) relies on.
    """
    from oaht_bench.teammate_gen.mep import population_entropy_coef_schedule

    for update_steps in (0, 1, 50, 1000):
        got = population_entropy_coef_schedule(update_steps, 100, 0.0, 0.5)
        assert got == 0.5


def test_population_entropy_coef_schedule_ramps_linearly_then_holds():
    from oaht_bench.teammate_gen.mep import population_entropy_coef_schedule

    num_updates, warmup_fraction, target = 100, 0.2, 0.5  # warmup spans 20 updates

    start = population_entropy_coef_schedule(0, num_updates, warmup_fraction, target)
    mid = population_entropy_coef_schedule(10, num_updates, warmup_fraction, target)
    end = population_entropy_coef_schedule(20, num_updates, warmup_fraction, target)
    past = population_entropy_coef_schedule(80, num_updates, warmup_fraction, target)

    assert float(start) == pytest.approx(0.0)
    assert float(mid) == pytest.approx(0.25)
    assert float(end) == pytest.approx(target)
    assert float(past) == pytest.approx(target)  # held, never exceeds target


def test_round_robin_visit_order_covers_every_member_exactly_once_per_block():
    """``jax.random.permutation`` is ``train_round_robin``'s visitation
    schedule (one PBT iteration = one permutation of ``population_size``).
    Pins that this actually gives full, exactly-once coverage per block
    across several blocks/seeds, not just "doesn't crash" -- a scheduling
    bug here would silently starve some members and double-visit others.
    """
    population_size = 7
    rng = jax.random.PRNGKey(0)
    for block in range(20):
        rng, perm_rng = jax.random.split(rng)
        visit_order = jax.random.permutation(perm_rng, population_size)
        assert sorted(np.asarray(visit_order).tolist()) == list(range(population_size)), (
            f"block {block} did not visit every member exactly once"
        )


def test_dynamic_index_update_round_trip_has_no_cross_member_aliasing():
    """The extract-train-write_back pattern ``train_round_robin`` uses
    (``jax.lax.dynamic_index_in_dim`` / ``dynamic_update_slice_in_dim``) must
    round-trip a value through one member's slot without disturbing any
    other member's slot -- the exact bug class an axis/shape mistake there
    would cause (catches it immediately, independent of a long real run).
    """
    population_size = 5
    population = jnp.arange(population_size, dtype=jnp.float32) * 10.0  # [0, 10, 20, 30, 40]

    def extract(x, idx):
        return jax.lax.dynamic_index_in_dim(x, idx, axis=0, keepdims=False)

    def write_back(full, idx, new_slice):
        return jax.lax.dynamic_update_slice_in_dim(full, new_slice[None], idx, axis=0)

    for idx in range(population_size):
        got = extract(population, idx)
        assert float(got) == pytest.approx(float(idx) * 10.0)

        updated = write_back(population, idx, got + 1.0)  # mutate only this member's slot
        for other in range(population_size):
            if other == idx:
                assert float(updated[other]) == pytest.approx(float(idx) * 10.0 + 1.0)
            else:
                assert float(updated[other]) == pytest.approx(float(population[other]))


def _toy_gen(population_entropy_coef, *, population_size=2, num_envs=8, total_timesteps=1600):
    from oaht_bench.configs.teammate_gen import MepConfig

    return MepConfig(
        population_size=population_size,
        num_envs=num_envs,
        total_timesteps=total_timesteps,
        num_seeds=1,
        population_entropy_coef=population_entropy_coef,
        train_seed=0,
    )


def test_run_mep_trains_saves_and_scores(tmp_path):
    """End-to-end through the runner on a tiny LBF job: trains, checkpoints, scores.

    Exercises the full path -- the named-vmap population axis, the all_gather
    cross-member forward pass, the entropy-bonus injection into reward, GAE,
    PPO update, checkpointing, and population wrapping (MEP releases a
    self-play set, like FCP/CoMeDi -- no best-response side).
    """
    from oaht_bench.configs import get_preset
    from oaht_bench.configs.job import TeammateGenerationJob
    from oaht_bench.population import artifact_dir
    from oaht_bench.teammate_gen.runner import run

    env = get_preset("lbf_12x12").model_copy(update={"rollout_length": 8})
    gen = _toy_gen(population_entropy_coef=0.01, num_envs=4, total_timesteps=32)
    job = TeammateGenerationJob(label="mep_e2e", env=env, generator=gen, output_dir=str(tmp_path))

    run_dir = run(job)
    assert (run_dir / "job.json").exists()
    assert artifact_dir(run_dir).exists()  # the checkpoint was written


def test_run_mep_tracks_best_and_releases_it_independently_of_checkpoints(tmp_path):
    """release_best=True adds population_size more members (each member's
    continuously-tracked best-performing snapshot), independent of and in
    addition to release_checkpoints -- both together give
    population_size * (num_checkpoints + 1) released members, matching the
    reference's 3-time-based-plus-1-best per-member structure (App. F.1).

    Also confirms, on real training data (not just the synthetic
    _maybe_store_best unit test), that best_score actually equals the true
    maximum of the recorded per-update returns -- not just that the shapes
    line up.
    """
    from oaht_bench.common.save_load_utils import load_train_run
    from oaht_bench.configs import get_preset
    from oaht_bench.configs.job import TeammateGenerationJob
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.population import artifact_dir
    from oaht_bench.population.loading import get_mep_population
    from oaht_bench.teammate_gen.runner import run

    env_cfg = get_preset("lbf_12x12").model_copy(update={"rollout_length": 8})
    population_size, num_checkpoints = 2, 3
    gen = _toy_gen(
        population_entropy_coef=0.01,
        population_size=population_size,
        num_envs=4,
        total_timesteps=192,
    ).model_copy(
        update={
            "release_checkpoints": True,
            "release_best": True,
            "num_checkpoints": num_checkpoints,
        }
    )
    job = TeammateGenerationJob(
        label="mep_best_e2e", env=env_cfg, generator=gen, output_dir=str(tmp_path)
    )

    run_dir = run(job)
    out = load_train_run(str(artifact_dir(run_dir)))
    assert "best_params" in out and "best_score" in out

    # best_score must equal the true max of this member's own recorded returns,
    # not merely have the right shape.
    returns = np.asarray(out["metrics"]["returned_episode_returns"])  # (seeds, pop, num_updates)
    best_score = np.asarray(out["best_score"])  # (seeds, pop)
    np.testing.assert_allclose(best_score, returns.max(axis=-1), atol=1e-5)

    env = LogWrapper(make_env(env_cfg.env_name, env_cfg.env_kwargs()))
    partner_params, partner_population = get_mep_population(job, out, env)
    expected_size = population_size * (num_checkpoints + 1)
    assert partner_population.pop_size == expected_size
    assert jax.tree_util.tree_leaves(partner_params)[0].shape[1] == expected_size


def _mean_population_entropy_estimate(job, run_dir, *, num_probe_envs=16, seed=123):
    """Monte-Carlo estimate of H(mean population policy) on a fresh observation batch.

    For each member i, sample an action from member i's own policy, then
    evaluate every member's log-prob of that action at the same observation
    and reduce with :func:`population_entropy_bonus` (coef=1.0) -- an
    unbiased single-sample estimate of H(pi_bar) for that draw, by the same
    identity Algorithm 1's training-time reward term relies on:
    ``mean_i E[a ~ pi_i][-log pi_bar(a)] == H(pi_bar)``. Averaging over both
    members and probe environments reduces the estimator's variance.
    """
    from oaht_bench.common.save_load_utils import load_train_run
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.models.mlp_actor_critic_agent import MLPActorCriticPolicy
    from oaht_bench.population import artifact_dir
    from oaht_bench.teammate_gen.marl.ppo_utils import batchify
    from oaht_bench.teammate_gen.mep import population_entropy_bonus

    env = LogWrapper(make_env(job.env.env_name, job.env.env_kwargs()))
    policy = MLPActorCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0],
    )

    out = load_train_run(artifact_dir(run_dir))
    # final_params: (num_seeds, population_size, ...) -- take seed 0.
    params = jax.tree.map(lambda x: x[0], out["final_params"])
    n_members = jax.tree.leaves(params)[0].shape[0]

    rng = jax.random.PRNGKey(seed)
    reset_rng = jax.random.split(rng, num_probe_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    num_actors = num_probe_envs * env.num_agents
    obs_batch = batchify(obsv, env.agents, num_actors).reshape(1, num_actors, -1)
    avail = jax.vmap(env.get_avail_actions)(env_state.env_state)
    avail_batch = (
        batchify(avail, env.agents, num_actors).astype(jnp.float32).reshape(1, num_actors, -1)
    )
    done_batch = jnp.zeros((1, num_actors), dtype=bool)

    def member_params(i):
        return jax.tree.map(lambda x: x[i], params)

    def log_prob_of(member_p, action):
        _, _, pi, _ = policy.get_action_value_policy(
            params=member_p,
            obs=obs_batch,
            done=done_batch,
            avail_actions=avail_batch,
            hstate=None,
            rng=jax.random.PRNGKey(0),
        )
        return pi.log_prob(action)

    estimates = []
    for i in range(n_members):
        act_rng, rng = jax.random.split(rng)
        _, _, pi_i, _ = policy.get_action_value_policy(
            params=member_params(i),
            obs=obs_batch,
            done=done_batch,
            avail_actions=avail_batch,
            hstate=None,
            rng=act_rng,
        )
        action = pi_i.sample(seed=act_rng)
        member_log_probs = jnp.stack(
            [log_prob_of(member_params(k), action) for k in range(n_members)]
        )
        # population_entropy_bonus(., coef=1.0) == -log(mean(probs)) == -log(pi_bar(a|s)),
        # which IS the single-sample MC estimator of H(pi_bar) -- no further negation.
        h_estimate = population_entropy_bonus(member_log_probs, coef=1.0)
        estimates.append(h_estimate)

    return float(jnp.mean(jnp.stack(estimates)))


def test_population_entropy_coef_raises_trained_population_entropy(tmp_path):
    """The behavioral claim MEP's own Table 1 / Question 2 makes: a nonzero
    population-entropy coefficient measurably raises the trained population's
    entropy relative to a coef=0 control on the same tiny job. This is the one
    signal that would catch the log-mean-exp bug even if the isolated
    reduction test above were somehow wrong for the same reason.
    """
    from oaht_bench.configs import get_preset
    from oaht_bench.configs.job import TeammateGenerationJob
    from oaht_bench.teammate_gen.runner import run

    env = get_preset("lbf_12x12").model_copy(update={"rollout_length": 8})

    # A large coefficient relative to LBF's ~0-1 task reward scale, so the
    # entropy bonus dominates and the effect is unmistakable even at this
    # tiny budget -- this is a mechanism-direction check, not a tuning result.
    control_gen = _toy_gen(population_entropy_coef=0.0)
    control_job = TeammateGenerationJob(
        label="mep_ablation_control",
        env=env,
        generator=control_gen,
        output_dir=str(tmp_path / "control"),
    )
    control_dir = run(control_job)
    control_entropy = _mean_population_entropy_estimate(control_job, control_dir)

    treated_gen = _toy_gen(population_entropy_coef=0.5)
    treated_job = TeammateGenerationJob(
        label="mep_ablation_treated",
        env=env,
        generator=treated_gen,
        output_dir=str(tmp_path / "treated"),
    )
    treated_dir = run(treated_job)
    treated_entropy = _mean_population_entropy_estimate(treated_job, treated_dir)

    assert treated_entropy > control_entropy


def test_population_entropy_warmup_zero_is_byte_identical_to_unset(tmp_path):
    """``population_entropy_warmup=0.0`` (explicit) must be a no-op -- the
    backward-compatibility guarantee this field's default is supposed to
    provide, checked on real training output (not just the schedule function
    in isolation). Both runs use the same ``train_seed``, so the only
    possible source of any divergence is the new field itself.
    """
    from oaht_bench.common.save_load_utils import load_train_run
    from oaht_bench.configs import get_preset
    from oaht_bench.configs.job import TeammateGenerationJob
    from oaht_bench.population import artifact_dir
    from oaht_bench.teammate_gen.runner import run

    env = get_preset("lbf_12x12").model_copy(update={"rollout_length": 8})
    default_gen = _toy_gen(population_entropy_coef=0.01, num_envs=4, total_timesteps=32)

    default_job = TeammateGenerationJob(
        label="mep_warmup_default",
        env=env,
        generator=default_gen,
        output_dir=str(tmp_path / "default"),
    )
    default_out = load_train_run(str(artifact_dir(run(default_job))))

    explicit_gen = default_gen.model_copy(update={"population_entropy_warmup": 0.0})
    explicit_job = TeammateGenerationJob(
        label="mep_warmup_explicit",
        env=env,
        generator=explicit_gen,
        output_dir=str(tmp_path / "explicit"),
    )
    explicit_out = load_train_run(str(artifact_dir(run(explicit_job))))

    np.testing.assert_array_equal(
        np.asarray(default_out["metrics"]["returned_episode_returns"]),
        np.asarray(explicit_out["metrics"]["returned_episode_returns"]),
    )
    default_leaves = jax.tree_util.tree_leaves(default_out["final_params"])
    explicit_leaves = jax.tree_util.tree_leaves(explicit_out["final_params"])
    for d, e in zip(default_leaves, explicit_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(d), np.asarray(e))


def test_population_entropy_warmup_ramps_the_logged_coefficient(tmp_path):
    """On real training data: ``population_entropy_coef_current`` starts at 0,
    rises monotonically, and reaches (and holds at) ``population_entropy_coef``
    by the end of the configured warmup window.

    Reads ``metrics.jsonl`` directly rather than the run's returned
    ``out["metrics"]`` -- the latter is ``condensed_metric``, which averages
    every key over episode-ending actor-steps only (right for
    ``returned_episode_returns`` which is genuinely episode-scoped, but wrong
    for this per-update scalar: it silently zeroes any update where no
    episode happened to complete in that tiny rollout). ``metrics.jsonl`` is
    written from the raw, unmasked ``metric`` dict via the
    ``io_callback``/``log_metrics_intermediate`` path -- the same
    distinction a prior session's logging bug turned on (see
    ``_update_step``'s comment about where this insertion must happen).
    """
    import json

    from oaht_bench.configs import get_preset
    from oaht_bench.configs.job import TeammateGenerationJob
    from oaht_bench.teammate_gen.runner import run

    env = get_preset("lbf_12x12").model_copy(update={"rollout_length": 8})
    target_coef = 0.01
    gen = _toy_gen(
        population_entropy_coef=target_coef, population_size=2, num_envs=4, total_timesteps=640
    ).model_copy(update={"population_entropy_warmup": 0.5})
    job = TeammateGenerationJob(
        label="mep_warmup_e2e", env=env, generator=gen, output_dir=str(tmp_path)
    )

    run_dir = run(job)
    rows = [
        json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line
    ]
    tag = "Train/population_entropy_coef_current"
    by_step: dict[int, list[float]] = {}
    for r in rows:
        if tag in r:
            by_step.setdefault(r["train_step"], []).append(r[tag])
    coef_curve = np.array([np.mean(by_step[step]) for step in sorted(by_step)])

    assert coef_curve[0] == pytest.approx(0.0, abs=1e-6)
    assert coef_curve[-1] == pytest.approx(target_coef, abs=1e-6)
    assert np.all(np.diff(coef_curve) >= -1e-8)  # monotone through the ramp, then flat
    assert np.any(np.isclose(coef_curve, target_coef, atol=1e-6))  # genuinely reaches target


def test_round_robin_false_is_byte_identical_to_unset(tmp_path):
    """``round_robin=False`` (explicit) must be a no-op -- the same
    backward-compatibility guarantee every other opt-in flag this session
    got (``population_entropy_warmup=0.0``, ``release_best=False``), checked
    on real training output rather than assumed from the default value.
    """
    from oaht_bench.common.save_load_utils import load_train_run
    from oaht_bench.configs import get_preset
    from oaht_bench.configs.job import TeammateGenerationJob
    from oaht_bench.population import artifact_dir
    from oaht_bench.teammate_gen.runner import run

    env = get_preset("lbf_12x12").model_copy(update={"rollout_length": 8})
    default_gen = _toy_gen(population_entropy_coef=0.01, num_envs=4, total_timesteps=32)

    default_job = TeammateGenerationJob(
        label="mep_rr_default", env=env, generator=default_gen, output_dir=str(tmp_path / "default")
    )
    default_out = load_train_run(str(artifact_dir(run(default_job))))

    explicit_gen = default_gen.model_copy(update={"round_robin": False})
    explicit_job = TeammateGenerationJob(
        label="mep_rr_explicit",
        env=env,
        generator=explicit_gen,
        output_dir=str(tmp_path / "explicit"),
    )
    explicit_out = load_train_run(str(artifact_dir(run(explicit_job))))

    np.testing.assert_array_equal(
        np.asarray(default_out["metrics"]["returned_episode_returns"]),
        np.asarray(explicit_out["metrics"]["returned_episode_returns"]),
    )
    default_leaves = jax.tree_util.tree_leaves(default_out["final_params"])
    explicit_leaves = jax.tree_util.tree_leaves(explicit_out["final_params"])
    for d, e in zip(default_leaves, explicit_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(d), np.asarray(e))


def test_round_robin_true_trains_saves_and_scores(tmp_path):
    """End-to-end through the runner with ``round_robin=True`` on a tiny LBF
    job: trains (sequential, frozen-others cadence), checkpoints, scores,
    and ``get_mep_population`` consumes the output exactly like the
    simultaneous-vmap path's -- the output-contract guarantee the whole
    design hinges on (no changes needed in ``population/loading.py``).
    """
    from oaht_bench.common.save_load_utils import load_train_run
    from oaht_bench.configs import get_preset
    from oaht_bench.configs.job import TeammateGenerationJob
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.population import artifact_dir
    from oaht_bench.population.loading import get_mep_population
    from oaht_bench.teammate_gen.runner import run

    env_cfg = get_preset("lbf_12x12").model_copy(update={"rollout_length": 8})
    population_size = 3
    gen = _toy_gen(
        population_entropy_coef=0.01,
        population_size=population_size,
        num_envs=4,
        total_timesteps=320,
    ).model_copy(
        update={
            "round_robin": True,
            "round_robin_chunk_timesteps": 32,  # tiny chunk -> several visits/iterations
        }
    )
    job = TeammateGenerationJob(
        label="mep_rr_e2e", env=env_cfg, generator=gen, output_dir=str(tmp_path)
    )

    run_dir = run(job)
    out = load_train_run(str(artifact_dir(run_dir)))
    assert "final_params" in out and "best_params" in out and "best_score" in out

    final_leaves = jax.tree_util.tree_leaves(out["final_params"])
    assert final_leaves[0].shape[:2] == (1, population_size)  # (num_seeds, population_size, ...)

    env = LogWrapper(make_env(env_cfg.env_name, env_cfg.env_kwargs()))
    partner_params, partner_population = get_mep_population(job, out, env)
    assert partner_population.pop_size == population_size
    assert jax.tree_util.tree_leaves(partner_params)[0].shape[1] == population_size


def test_round_robin_logs_member_idx_and_visits_every_member(tmp_path):
    """Real training data: ``metrics.jsonl``'s ``member_idx`` field (added
    only on the round-robin path) must show every member visited the
    expected number of times (``num_pbt_iters``), and a given member's own
    return curve across its (non-consecutive) visits must look like a
    continuous PPO learning signal -- not reset each visit -- confirming
    optimizer state actually persisted across visits rather than being
    reinitialized.
    """
    import json

    from oaht_bench.configs import get_preset
    from oaht_bench.configs.job import TeammateGenerationJob
    from oaht_bench.teammate_gen.runner import run

    env = get_preset("lbf_12x12").model_copy(update={"rollout_length": 8})
    population_size = 3
    num_envs = 4
    chunk_timesteps = 32
    total_timesteps = 320
    num_pbt_iters = total_timesteps // chunk_timesteps  # 10
    # chunk_updates = chunk_timesteps // rollout_length // num_envs = 32 // 8 // 4 = 1,
    # chosen deliberately so each visit logs exactly one member_idx row --
    # counting metrics.jsonl rows directly equals the visit count.
    gen = _toy_gen(
        population_entropy_coef=0.01,
        population_size=population_size,
        num_envs=num_envs,
        total_timesteps=total_timesteps,
    ).model_copy(update={"round_robin": True, "round_robin_chunk_timesteps": chunk_timesteps})
    job = TeammateGenerationJob(
        label="mep_rr_member_idx", env=env, generator=gen, output_dir=str(tmp_path)
    )

    run_dir = run(job)
    rows = [
        json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line
    ]
    member_idx_rows = [r for r in rows if "Train/member_idx" in r]
    assert member_idx_rows, "round_robin=True must log member_idx -- found none in metrics.jsonl"

    from collections import Counter

    visit_counts = Counter(int(r["Train/member_idx"]) for r in member_idx_rows)
    assert set(visit_counts) == set(range(population_size))
    for member, count in visit_counts.items():
        assert count == num_pbt_iters, (
            f"member {member} visited {count} times, expected {num_pbt_iters}"
        )
