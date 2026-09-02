"""AD-RPG (Rational Adversarial Diversity) teammate generation.

A clean-room reimplementation of the ``doublesided_RAD`` algorithm from Lauffer,
Shah, Carroll, Seshia, Russell & Dennis, *Robust and Diverse Multi-Agent Learning
via Rational Policy Gradient* (NeurIPS 2025). The upstream repository
(``github.com/niklaslauffer/rational-policy-gradient``, commit ``0f9b863``) is
**unlicensed** and targets an incompatible JAX stack, so no code is absorbed from
it; this module is authored from the paper and the repo is a reference only. See
``PROVENANCE.md``. Unlike the four absorbed generators, this file is linted and
unit-tested.

Method (specialized to ``doublesided_RAD``, the adversarial-diversity variant):

* ``N = population_size`` diversity particles. Each particle ``i`` is a **base**
  policy shaped by a paired **manipulator** policy.
* **Base update** — PPO on task return, but with a Loaded-DiCE surrogate
  (:func:`dice_ratio`) that couples the base and manipulator log-probs so the
  base gradient stays differentiable through the manipulator. The base objective
  weights self-play ``(base_i, manipulator_i)`` by ``1 - N*partnerplay_ratio`` and
  each cross-play ``(base_i, base_j)`` by ``partnerplay_ratio``.
* **Manipulator update** — a higher-order (opponent-shaping) gradient of the
  diversity objective taken *through* ``n_lookahead`` inner base updates: maximize
  the shaped base's self-play return (weight ``+1``) and minimize its cross-play
  with the other bases (weight ``-off_diag_factor / (N-1)`` per pairing).
* **Released population** — the ``N`` converged base policies, a self-play set
  scored like the CoMeDi/BRDiv releases. Manipulators are discarded, per the paper.

The algorithm core lands in Phase B; :func:`run_rpg` is the harness entry point.
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import io_callback

from oaht_bench.common.logging import RunLogger, nonfatal
from oaht_bench.common.save_load_utils import save_train_run
from oaht_bench.configs.job import TeammateGenerationJob
from oaht_bench.envs import make_env
from oaht_bench.envs.log_wrapper import LogWrapper
from oaht_bench.models.mlp_actor_critic_agent import MLPActorCriticPolicy
from oaht_bench.models.population_interface import AgentPopulation
from oaht_bench.teammate_gen.runtime import RpgRuntime

log = logging.getLogger(__name__)


# --- Loaded DiCE ------------------------------------------------------------
#
# The "magic box" (Foerster et al. 2018) evaluates to 1 in the forward pass but
# carries the score-function gradient of its argument, so wrapping a stochastic
# objective in it makes the objective differentiable through the sampling that
# produced it -- and, unlike a plain surrogate, differentiable to *any* order,
# which is what the manipulator's higher-order (opponent-shaping) gradient needs.
# Loaded DiCE (Farquhar et al. 2019) adds a per-timestep discount ``lam`` on past
# dependencies to trade a little bias for much lower variance.


def magic_box(x: jnp.ndarray) -> jnp.ndarray:
    """DiCE operator: ``1`` in value, ``grad = x``'s gradient. ``exp(x - stop(x))``."""
    return jnp.exp(x - jax.lax.stop_gradient(x))


def dice_ratio(
    log_p: jnp.ndarray,
    partner_log_p: jnp.ndarray,
    starts: jnp.ndarray,
    lam: float,
) -> jnp.ndarray:
    """Loaded-DiCE surrogate ratio for one trajectory batch.

    ``log_p`` and ``partner_log_p`` are ``(batch, time)`` log-probabilities of the
    learner and the agent shaping it (its manipulator), and ``starts`` marks the
    first timestep of each episode so credit does not leak across resets. Returns a
    ``(batch, time)`` array that is 1 in value but whose gradient reproduces the
    (loaded) policy-gradient estimator, coupling the learner's update to the
    partner -- the differentiable channel the manipulator meta-gradient flows
    through. Mirrors the reference implementation's `dice_ratio`.
    """
    stochastic_nodes = partner_log_p + log_p
    starts = starts.at[:, 0].set(True)

    def _step(carry, inp):
        start_t, node_t = inp
        nxt = lam * carry + node_t
        cumsum_t = jnp.where(start_t, node_t, nxt)
        return cumsum_t, cumsum_t

    initial = stochastic_nodes[:, 0]
    inputs = (
        jnp.swapaxes(starts[:, 1:], 0, 1),
        jnp.swapaxes(stochastic_nodes[:, 1:], 0, 1),
    )
    _, body = jax.lax.scan(_step, initial, inputs)
    body = jnp.swapaxes(body, 0, 1)
    weighted_cumsum = jnp.concatenate([initial[:, None], body], axis=1)
    deps_exclusive = weighted_cumsum - stochastic_nodes
    return magic_box(weighted_cumsum) - magic_box(deps_exclusive)


# --- rollouts and GAE -------------------------------------------------------
#
# A rollout pairs two *specific* particles -- seat 0 the "learner" whose stream
# is scored, seat 1 its partner -- which is why IPPO's homogeneous-population
# rollout is not reused: each seat may carry different parameters. The learner's
# observations, actions, availabilities, old log-probs, values, rewards and
# episode boundaries are kept (the update recomputes log-probs from live params),
# plus the partner's observations/actions/availabilities so the base and
# manipulator losses can recompute the partner log-prob the DiCE coupling needs.


def _gae(rew, val, done, last_val, gamma, lam):
    """Generalised advantage estimation over a ``(time, env)`` batch."""

    def _step(carry, x):
        gae, next_val = carry
        rew_t, val_t, done_t = x
        delta = rew_t + gamma * next_val * (1.0 - done_t) - val_t
        gae = delta + gamma * lam * (1.0 - done_t) * gae
        return (gae, val_t), gae

    _, adv = jax.lax.scan(
        _step, (jnp.zeros_like(last_val), last_val), (rew, val, done), reverse=True
    )
    return adv, adv + val


def _rollout(network, params0, params1, env, rng, runtime):
    """Roll ``params0`` (seat 0) against ``params1`` (seat 1) for one horizon.

    Returns a dict of the learner's trajectory plus advantages and the partner's
    ``(obs, action, avail)``. Both seats share ``network`` (an ``ActorCritic``);
    the partner's value head is ignored.
    """
    a0, a1 = env.agents[0], env.agents[1]
    n = runtime.num_envs
    reset_rng = jax.random.split(rng, n)
    obs, state = jax.vmap(env.reset)(reset_rng)

    def _step(carry, _unused):
        obs, state, rng = carry
        avail = jax.vmap(env.get_avail_actions)(state.env_state)
        av0 = jax.lax.stop_gradient(avail[a0].astype(jnp.float32))
        av1 = jax.lax.stop_gradient(avail[a1].astype(jnp.float32))
        pi0, val0 = network.apply(params0, (obs[a0], av0))
        pi1, _ = network.apply(params1, (obs[a1], av1))
        rng, k0, k1, ks = jax.random.split(rng, 4)
        act0 = pi0.sample(seed=k0)
        act1 = pi1.sample(seed=k1)
        trans = {
            "obs0": obs[a0],
            "av0": av0,
            "a0": act0,
            "lp0": pi0.log_prob(act0),
            "val0": jnp.ravel(val0),
            "obs1": obs[a1],
            "av1": av1,
            "a1": act1,
            "lp1": pi1.log_prob(act1),
        }
        step_rng = jax.random.split(ks, n)
        nobs, nstate, rew, done, _info = jax.vmap(env.step)(step_rng, state, {a0: act0, a1: act1})
        trans["rew0"] = rew[a0]
        trans["done"] = done["__all__"].astype(jnp.float32)
        return (nobs, nstate, rng), trans

    (obs, state, _rng), traj = jax.lax.scan(_step, (obs, state, rng), None, runtime.rollout_length)

    avail = jax.vmap(env.get_avail_actions)(state.env_state)
    _, last_val = network.apply(params0, (obs[a0], avail[a0].astype(jnp.float32)))
    adv, tgt = _gae(
        traj["rew0"],
        traj["val0"],
        traj["done"],
        jnp.ravel(last_val),
        runtime.ppo.gamma,
        runtime.ppo.gae_lambda,
    )
    traj["adv"] = adv
    traj["tgt"] = tgt
    # Mean episodic return of the learner, for SP/XP curves. rew per (T, env);
    # summing over time then averaging over envs approximates return per episode
    # for the single-episode-per-rollout regime and is a monotone proxy otherwise.
    traj["ret"] = traj["rew0"].sum(axis=0).mean()
    return traj


# --- losses -----------------------------------------------------------------


def _actor_dice_loss(base_params, partner_params, network, traj, dice_lambda):
    """DiCE policy-gradient actor loss for the learner on one rollout.

    Recomputes both log-probs from live parameters so that (a) the gradient w.r.t.
    ``base_params`` is the advantage policy gradient and (b) the gradient of an
    *inner update* built from this loss w.r.t. ``partner_params`` is the
    higher-order opponent-shaping term. Returns the scalar actor loss and the mean
    policy entropy.
    """
    pi0, _ = network.apply(base_params, (traj["obs0"], traj["av0"]))
    lp0 = pi0.log_prob(traj["a0"])  # (T, env)
    pi1, _ = network.apply(partner_params, (traj["obs1"], traj["av1"]))
    lp1 = pi1.log_prob(traj["a1"])
    done_b = traj["done"].T  # (env, T)
    starts = jnp.roll(done_b, 1, axis=1)
    adv = traj["adv"].T
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ratio = dice_ratio(lp0.T, lp1.T, starts, dice_lambda)
    actor_loss = -(ratio * adv).mean()
    return actor_loss, pi0.entropy().mean()


def _base_loss(base_params, partner_list, weights, network, traj_list, runtime):
    """Combined base objective: weighted DiCE actor loss + critic MSE.

    ``partner_list``/``traj_list``/``weights`` are aligned: the first entry is the
    self-play pairing against the particle's manipulator (weight ``1 - N*pp``),
    the rest are cross-play pairings against every base (weight ``pp`` each). Only
    ``base_params`` is differentiated; partners are captured constants, so the
    self-copy cross-play term does not double-count.
    """
    actor = 0.0
    entropy = 0.0
    for partner, w, traj in zip(partner_list, weights, traj_list, strict=True):
        a, e = _actor_dice_loss(base_params, partner, network, traj, runtime.dice_lambda)
        actor = actor + w * a
        entropy = entropy + w * e
    # Critic trained on the self-play (first) rollout's targets.
    self_traj = traj_list[0]
    _, val = network.apply(base_params, (self_traj["obs0"], self_traj["av0"]))
    val = jnp.ravel(val).reshape(self_traj["val0"].shape)
    critic = jnp.mean((val - self_traj["tgt"]) ** 2)
    return actor + runtime.ppo.value_coef * critic - runtime.ppo.entropy_coef * entropy


def _diversity_surrogate(base_params, base_old_lp, network, traj):
    """Importance-weighted return surrogate for a looked-ahead base on a rollout.

    ``exp(logp(base') - logp(base_collected)) * advantage`` -- differentiable in
    ``base'`` (hence, through the lookahead, in the manipulator).
    """
    pi0, _ = network.apply(base_params, (traj["obs0"], traj["av0"]))
    lp = pi0.log_prob(traj["a0"])
    ratio = jnp.exp(lp - base_old_lp)
    return (ratio * traj["adv"]).mean()


def _manipulator_meta_loss(
    manip_params, base_params, network, self_traj, div_self_traj, div_cross_trajs, runtime, base_lr
):
    """Negative diversity objective, differentiated through the base lookahead.

    An inner SGD lookahead advances a copy of the base under the DiCE self-play
    loss coupled to ``manip_params`` (so ``base'`` is a function of the
    manipulator). The diversity objective is then evaluated at ``base'``: maximize
    its self-play return (weight ``+1``) and minimize its cross-play return
    (weight ``-off_diag_factor/(N-1)`` per other base). Returns the negative so a
    gradient *descent* step maximizes diversity.
    """
    bp = base_params
    for _ in range(runtime.n_lookahead):
        g = jax.grad(
            lambda p: _actor_dice_loss(p, manip_params, network, self_traj, runtime.dice_lambda)[0]
        )(bp)
        bp = jax.tree.map(lambda w, gr: w - base_lr * gr, bp, g)

    j = _diversity_surrogate(bp, div_self_traj["lp0"], network, div_self_traj)
    scale = runtime.off_diag_factor / max(runtime.population_size - 1, 1)
    for traj in div_cross_trajs:
        j = j - scale * _diversity_surrogate(bp, traj["lp0"], network, traj)
    return -j


# --- training loop ----------------------------------------------------------


def _log_rpg_update(metric, logger):
    """Stream one outer update's SP/XP returns (called through ``io_callback``).

    Must be a module-level function. Values arrive with a leading seed axis under
    the ``vmap`` over seeds, so reduce over it; the step is shared across seeds.
    """
    step = int(np.asarray(metric["update_steps"]).flat[0])
    logger.log_item(
        "Train/SelfPlayReturn", float(np.asarray(metric["sp_return"]).mean()), train_step=step
    )
    logger.log_item(
        "Train/CrossPlayReturn", float(np.asarray(metric["xp_return"]).mean()), train_step=step
    )
    logger.commit()


def _init_particle(network, obs_dim, action_dim, rng):
    return network.init(rng, (jnp.zeros((obs_dim,)), jnp.ones((action_dim,))))


def make_rpg_train(runtime: RpgRuntime, env, logger=None):
    """Build the AD-RPG training function ``train(rng) -> out``.

    ``out`` carries ``final_params`` (leading axis ``N``), the released base
    policies, and per-update SP/XP return metrics. Particles are unrolled in
    Python (``N`` is small and static); each outer step collects the self-play and
    all-pairs cross-play rollouts, meta-updates every manipulator through the base
    lookahead, then advances every base with the DiCE base objective.

    When ``logger`` is given, each outer update streams its SP/XP returns live
    through :func:`jax.experimental.io_callback` -- the same in-loop logging the
    other generators use -- so wandb shows a curve during training rather than
    only at the end.
    """
    n = runtime.population_size
    action_dim = env.action_space(env.agents[1]).n
    obs_dim = env.observation_space(env.agents[1]).shape[0]
    policy = MLPActorCriticPolicy(
        action_dim=action_dim,
        obs_dim=obs_dim,
        activation=runtime.network.activation,
        fc_hidden_dim=runtime.network.hidden_dim,
    )
    network = policy.network
    base_lr = runtime.ppo.learning_rate

    base_tx = optax.chain(
        optax.clip_by_global_norm(runtime.ppo.max_grad_norm), optax.adam(base_lr, eps=1e-5)
    )
    manip_tx = optax.chain(
        optax.clip_by_global_norm(runtime.ppo.max_grad_norm),
        optax.adam(runtime.manipulator_lr, eps=1e-5),
    )
    pp = runtime.partnerplay_ratio
    base_weights = [1.0 - n * pp] + [pp] * n  # self-play then one per base (incl. self-copy)

    def train(rng):
        rng, k = jax.random.split(rng)
        init_keys = jax.random.split(k, 2 * n)
        base_params = [_init_particle(network, obs_dim, action_dim, init_keys[i]) for i in range(n)]
        manip_params = [
            _init_particle(network, obs_dim, action_dim, init_keys[n + i]) for i in range(n)
        ]
        base_opt = [base_tx.init(p) for p in base_params]
        manip_opt = [manip_tx.init(p) for p in manip_params]

        def _update(carry, _unused):
            base_params, manip_params, base_opt, manip_opt, update_step, rng = carry

            # Collect rollouts: self-play (base_i, manip_i) and all-pairs
            # cross-play (base_i, base_j). One rng key per rollout.
            rng, kr = jax.random.split(rng)
            keys = jax.random.split(kr, n + n * n)
            sp = [
                _rollout(network, base_params[i], manip_params[i], env, keys[i], runtime)
                for i in range(n)
            ]
            cross = [
                [
                    _rollout(
                        network, base_params[i], base_params[j], env, keys[n + i * n + j], runtime
                    )
                    for j in range(n)
                ]
                for i in range(n)
            ]

            # Manipulator meta-update (shapes each base through the lookahead).
            for i in range(n):
                div_cross = [cross[i][j] for j in range(n) if j != i]
                loss, g = jax.value_and_grad(_manipulator_meta_loss)(
                    manip_params[i],
                    base_params[i],
                    network,
                    sp[i],
                    cross[i][i],
                    div_cross,
                    runtime,
                    base_lr,
                )
                upd, manip_opt[i] = manip_tx.update(g, manip_opt[i], manip_params[i])
                manip_params[i] = optax.apply_updates(manip_params[i], upd)

            # Base update (DiCE self-play + all-pairs cross-play + critic).
            for i in range(n):
                partner_list = [manip_params[i]] + [base_params[j] for j in range(n)]
                traj_list = [sp[i]] + [cross[i][j] for j in range(n)]
                loss, g = jax.value_and_grad(_base_loss)(
                    base_params[i], partner_list, base_weights, network, traj_list, runtime
                )
                upd, base_opt[i] = base_tx.update(g, base_opt[i], base_params[i])
                base_params[i] = optax.apply_updates(base_params[i], upd)

            sp_ret = jnp.mean(jnp.stack([sp[i]["ret"] for i in range(n)]))
            xp_ret = jnp.mean(
                jnp.stack([cross[i][j]["ret"] for i in range(n) for j in range(n) if i != j])
            )
            metrics = {"sp_return": sp_ret, "xp_return": xp_ret}

            # Stream this update's returns live, matching the other generators.
            if logger is not None:
                io_callback(
                    lambda m: _log_rpg_update(m, logger),
                    None,
                    {**metrics, "update_steps": update_step},
                )

            carry = (base_params, manip_params, base_opt, manip_opt, update_step + 1, rng)
            return carry, metrics

        carry = (base_params, manip_params, base_opt, manip_opt, jnp.int32(0), rng)
        carry, metrics = jax.lax.scan(_update, carry, None, runtime.num_updates)
        base_params = carry[0]

        # Stack particles onto a leading axis N to match the release convention.
        stacked = jax.tree.map(lambda *ps: jnp.stack(ps), *base_params)
        return {"final_params": stacked, "metrics": metrics}

    return train


def get_rpg_population(job: TeammateGenerationJob, out: dict, env):
    """Wrap AD-RPG's N converged base policies as a scorable population.

    Mirrors :func:`oaht_bench.population.loading.get_fcp_population`: a self-play
    set of ``MLPActorCriticPolicy`` members, so evaluation pairs the column
    population with the row one (no best-response set).
    """
    gen = job.generator
    policy = MLPActorCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0],
        activation=gen.network.activation,
    )
    population = AgentPopulation(pop_size=gen.population_size, policy_cls=policy)
    return out["final_params"], population


def run_rpg(job: TeammateGenerationJob, wandb_logger: RunLogger):
    """Train an AD-RPG population from a validated job config.

    Returns ``(flattened_base_params, AgentPopulation)`` -- the same contract the
    other four generators satisfy (see :func:`oaht_bench.teammate_gen.fcp.run_fcp`).
    """
    gen = job.generator
    env = LogWrapper(make_env(job.env.env_name, job.env.env_kwargs()))
    runtime = RpgRuntime.from_config(
        gen, rollout_length=job.env.rollout_length, num_agents=env.num_agents
    )

    # The logger is threaded into training so SP/XP returns stream live per update
    # via io_callback, rather than only after the whole run.
    train = make_rpg_train(runtime, env, logger=wandb_logger)
    rngs = jax.random.split(jax.random.PRNGKey(gen.train_seed), gen.num_seeds)

    start = time.time()
    out = jax.jit(jax.vmap(train))(rngs)
    log.info("Training AD-RPG population took %.2f seconds.", time.time() - start)

    flattened_params, population = get_rpg_population(job, out, env)

    # Save FIRST so a reporting failure cannot discard a finished run (invariant #1).
    savepath = save_train_run(out, job.run_dir(), savename="saved_train_run")
    with nonfatal("RPG artifact logging"):
        wandb_logger.log_artifact(name="saved_train_run", path=savepath, type_name="train_run")

    return flattened_params, population
