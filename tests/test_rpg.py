"""AD-RPG (clean-room reimplementation) unit tests.

These pin the delicate parts -- the DiCE surrogate and the diversity objective --
in isolation, on CPU-sized inputs, so a regression surfaces here rather than as a
silently-wrong training curve on GPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def test_magic_box_is_one_in_value_and_carries_gradient():
    """magic_box(x) == 1 everywhere, but d/dx magic_box(a*x) == a."""
    from oaht_bench.teammate_gen.RPG import magic_box

    x = jnp.array([-2.0, 0.0, 3.5])
    np.testing.assert_allclose(np.asarray(magic_box(x)), 1.0, atol=1e-6)

    a = 1.7
    g = jax.grad(lambda t: magic_box(a * t).sum())(jnp.array([0.3]))
    np.testing.assert_allclose(np.asarray(g), a, atol=1e-6)


def _logp(theta, actions):
    """Categorical log-probs of `actions` under logits `theta`, shape (batch, time)."""
    logits = jax.nn.log_softmax(theta)  # (n_actions,)
    return logits[actions]


def test_dice_ratio_is_zero_in_value():
    """The DiCE surrogate is a pure gradient object: exactly 0 in the forward pass."""
    from oaht_bench.teammate_gen.RPG import dice_ratio

    rng = np.random.default_rng(0)
    log_p = jnp.asarray(rng.normal(size=(4, 6)))
    partner = jnp.asarray(rng.normal(size=(4, 6)))
    starts = jnp.zeros((4, 6), dtype=bool)
    r = dice_ratio(log_p, partner, starts, lam=0.99)
    np.testing.assert_allclose(np.asarray(r), 0.0, atol=1e-6)


def test_dice_ratio_couples_gradient_to_both_learner_and_partner():
    """The surrogate must carry gradient to the learner AND its shaping partner.

    The manipulator's higher-order gradient flows through exactly this coupling;
    if the partner gradient were zero, opponent shaping could not work.
    """
    from oaht_bench.teammate_gen.RPG import dice_ratio

    rng = np.random.default_rng(1)
    a_base = jnp.asarray(rng.integers(0, 3, size=(4, 6)))
    a_partner = jnp.asarray(rng.integers(0, 3, size=(4, 6)))
    starts = jnp.zeros((4, 6), dtype=bool)
    gae = jnp.asarray(rng.normal(size=(4, 6)))
    theta_base = jnp.asarray(rng.normal(size=(3,)))
    theta_partner = jnp.asarray(rng.normal(size=(3,)))

    def surrogate(tb, tp):
        lp = _logp(tb, a_base)
        pp = _logp(tp, a_partner)
        return (dice_ratio(lp, pp, starts, lam=1.0) * gae).sum()

    g_base = jax.grad(surrogate, argnums=0)(theta_base, theta_partner)
    g_partner = jax.grad(surrogate, argnums=1)(theta_base, theta_partner)
    assert float(jnp.linalg.norm(g_base)) > 1e-6
    assert float(jnp.linalg.norm(g_partner)) > 1e-6


def test_dice_ratio_first_order_is_the_advantage_policy_gradient():
    """The learner's first-order gradient is the per-timestep advantage PG.

    With the loaded (two-magic_box) surrogate the first-order learner gradient is
    ``sum_t gae_t * dlogp_t`` -- the GAE advantage already carries the causal
    future return, so `lam` and `starts` affect only *higher* orders (the meta
    gradient), tested separately. Ground truth: grad of ``(logp * gae).sum()``.
    """
    from oaht_bench.teammate_gen.RPG import dice_ratio

    rng = np.random.default_rng(2)
    a_base = jnp.asarray(rng.integers(0, 3, size=(2, 5)))
    a_partner = jnp.asarray(rng.integers(0, 3, size=(2, 5)))
    starts = jnp.zeros((2, 5), dtype=bool)
    gae = jnp.asarray(rng.normal(size=(2, 5)))
    theta = jnp.asarray(rng.normal(size=(3,)))
    partner = jnp.asarray(rng.normal(size=(3,)))

    def dice_obj(t):
        return (
            dice_ratio(_logp(t, a_base), _logp(partner, a_partner), starts, lam=1.0) * gae
        ).sum()

    def advantage_pg(t):
        return (_logp(t, a_base) * gae).sum()

    np.testing.assert_allclose(
        np.asarray(jax.grad(dice_obj)(theta)),
        np.asarray(jax.grad(advantage_pg)(theta)),
        atol=1e-5,
    )


def test_dice_ratio_higher_order_is_nonzero_and_depends_on_lam_and_starts():
    """The mixed base/partner second derivative is the opponent-shaping signal.

    d^2 obj / (dtheta_base dtheta_partner) is where the manipulator's higher-order
    gradient lives. It must be nonzero (a plain surrogate would give zero) and must
    actually depend on the loaded discount `lam` and on episode boundaries `starts`.
    """
    from oaht_bench.teammate_gen.RPG import dice_ratio

    rng = np.random.default_rng(3)
    a_base = jnp.asarray(rng.integers(0, 3, size=(1, 6)))
    a_partner = jnp.asarray(rng.integers(0, 3, size=(1, 6)))
    gae = jnp.asarray(rng.normal(size=(1, 6)))
    tb = jnp.asarray(rng.normal(size=(3,)))
    tp = jnp.asarray(rng.normal(size=(3,)))
    no_reset = jnp.zeros((1, 6), dtype=bool)
    with_reset = jnp.asarray([[False, False, False, True, False, False]])

    def mixed(lam, starts):
        def obj(b, p):
            return (dice_ratio(_logp(b, a_base), _logp(p, a_partner), starts, lam=lam) * gae).sum()

        return jax.jacfwd(jax.grad(obj, argnums=0), argnums=1)(tb, tp)

    h1 = mixed(1.0, no_reset)
    h_lam = mixed(0.5, no_reset)
    h_reset = mixed(1.0, with_reset)

    assert float(jnp.linalg.norm(h1)) > 1e-6  # opponent-shaping signal exists
    assert float(jnp.linalg.norm(h1 - h_lam)) > 1e-6  # lam changes it
    assert float(jnp.linalg.norm(h1 - h_reset)) > 1e-6  # episode boundary changes it


def _toy_runtime(pop_size=2, n_lookahead=1, off_diag_factor=0.25):
    from oaht_bench.configs.teammate_gen import RpgConfig
    from oaht_bench.teammate_gen.runtime import RpgRuntime

    gen = RpgConfig(
        population_size=pop_size,
        num_envs=4,
        total_timesteps=32,
        n_lookahead=n_lookahead,
        off_diag_factor=off_diag_factor,
    )
    return RpgRuntime.from_config(gen, rollout_length=8, num_agents=2)


def _toy_traj(rng, T=4, E=3, obs_dim=6, act_dim=6):
    return {
        "obs0": jnp.asarray(rng.normal(size=(T, E, obs_dim))),
        "av0": jnp.ones((T, E, act_dim)),
        "a0": jnp.asarray(rng.integers(0, act_dim, size=(T, E))),
        "lp0": jnp.asarray(rng.normal(size=(T, E))),
        "adv": jnp.asarray(rng.normal(size=(T, E))),
        "done": jnp.zeros((T, E)),
        "obs1": jnp.asarray(rng.normal(size=(T, E, obs_dim))),
        "av1": jnp.ones((T, E, act_dim)),
        "a1": jnp.asarray(rng.integers(0, act_dim, size=(T, E))),
    }


def _meta_grad(off_diag_factor, base_lr=0.05):
    """Meta-gradient of the diversity objective w.r.t. the manipulator, on a toy."""
    from oaht_bench.models.mlp_actor_critic_agent import MLPActorCriticPolicy
    from oaht_bench.teammate_gen.RPG import _manipulator_meta_loss

    rng = np.random.default_rng(7)
    runtime = _toy_runtime(pop_size=2, n_lookahead=1, off_diag_factor=off_diag_factor)
    network = MLPActorCriticPolicy(action_dim=6, obs_dim=6).network
    base = network.init(jax.random.PRNGKey(0), (jnp.zeros((6,)), jnp.ones((6,))))
    manip = network.init(jax.random.PRNGKey(1), (jnp.zeros((6,)), jnp.ones((6,))))
    sp, div_self, div_cross = _toy_traj(rng), _toy_traj(rng), [_toy_traj(rng)]

    def meta(m):
        return _manipulator_meta_loss(
            m, base, network, sp, div_self, div_cross, runtime, base_lr=base_lr
        )

    return jax.grad(meta)(manip)


def test_manipulator_meta_gradient_exists_and_tracks_the_diversity_objective():
    """The manipulator meta-gradient must exist and be driven by the objective.

    A DiCE meta-loss is manipulator-invariant *in value* (the trajectories are
    sampled constants); the shaping signal lives entirely in the gradient. So a
    finite-step-decrease test is ill-posed. Instead assert the meta-gradient is
    (a) nonzero and finite, and (b) genuinely a function of the diversity
    objective: changing `off_diag_factor` -- the weight balancing self-play
    maximization against cross-play minimization -- changes it. The mechanism
    itself (the higher-order DiCE channel) is pinned by
    ``test_dice_ratio_higher_order_is_nonzero_and_depends_on_lam_and_starts``.
    """
    g_ref = _meta_grad(off_diag_factor=0.25)
    leaves = jax.tree_util.tree_leaves(g_ref)
    gnorm = float(jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves)))
    assert gnorm > 1e-8
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)

    g_hi = _meta_grad(off_diag_factor=2.0)
    diff = float(
        jnp.sqrt(
            sum(
                jnp.vdot(a - b, a - b)
                for a, b in zip(leaves, jax.tree_util.tree_leaves(g_hi), strict=True)
            )
        )
    )
    assert diff > 1e-8  # the gradient reflects the diversity trade-off, not noise


def test_run_rpg_trains_saves_and_scores(tmp_path):
    """End-to-end through the runner on a tiny LBF job: trains, checkpoints, scores.

    Exercises the full path -- rollouts, DiCE base update, manipulator meta-update,
    population wrapping, save-before-report, and self-play evaluation (RPG releases
    a self-play set, so there is no best-response side).
    """
    from oaht_bench.configs import get_preset
    from oaht_bench.configs.job import TeammateGenerationJob
    from oaht_bench.configs.teammate_gen import RpgConfig
    from oaht_bench.population import artifact_dir
    from oaht_bench.teammate_gen.runner import run

    env = get_preset("lbf_12x12").model_copy(update={"rollout_length": 8})
    gen = RpgConfig(population_size=2, num_envs=4, total_timesteps=32, num_seeds=1, n_lookahead=1)
    job = TeammateGenerationJob(label="rpg_e2e", env=env, generator=gen, output_dir=str(tmp_path))

    run_dir = run(job)
    assert (run_dir / "job.json").exists()
    assert artifact_dir(run_dir).exists()  # the checkpoint was written
