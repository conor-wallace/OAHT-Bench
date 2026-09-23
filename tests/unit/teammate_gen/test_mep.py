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
