"""Maximum Entropy Population-based training (Zhao et al., AAAI-23) -- Stage 1 only.

OMIS uses MEP for its own teammate generation; this is a clean-room
implementation from the paper (no upstream code absorbed -- the reference
repo, https://github.com/ruizhaogit/maximum_entropy_population_based_training,
is on an incompatible stack). See ``PROVENANCE.md``.

N population members each run ordinary self-play PPO (paired with itself),
while every rollout step's reward is augmented with the *population-entropy*
bonus (Eq. 6/9 of the paper): the log-probability of the already-sampled
action under the population's *mean* policy, an unbiased Monte-Carlo estimate
of population entropy. This must be reduced as
``logsumexp(log_probs_across_members) - log(N)`` -- i.e. ``log(mean(probs))``
-- **not** ``mean(log_probs))``; those are different quantities, and picking
the wrong one silently trains a different (and wrong) objective. See
``tests/unit/teammate_gen/test_mep.py``'s isolated reduction test.

This is a *non-adversarial* diversity mechanism: unlike BRDiv/L-BRDiv/CoMeDi/
RPG, nothing here optimizes to minimize another member's reward or cross-play
score, so it is structurally immune to the self-sabotage failure mode those
methods must be engineered around (see ``papers/rpg.pdf``, Fig. 4). MEP's own
Stage 2 (training a single shared robust-generalist ego against the population
via prioritized/hardness-ranked sampling) is deliberately not built here --
every other generator in this repo only releases a *population* (BRDiv/
L-BRDiv discard their internal ``br``, RPG discards its manipulators) and
leaves ego training to ``ppo_br.py``; MEP follows that precedent.

Architecture note (why this isn't just ``fcp.py`` with an extra reward term):
FCP's ``train_fcp_partners`` vmaps ``population_size`` fully independent
self-play trains with zero communication between vmap lanes -- correct for
FCP, but MEP's entropy bonus needs every member's *current* policy visible
from every other member's rollout at each step. BRDiv/L-BRDiv solve a similar
visibility problem by threading one ``TrainState`` with a population leading
axis through a single scan, with ``gather_params`` pulling specific members'
params for *environment* pairings -- machinery built for their cross-play
*environment* interactions (conf-vs-br rollouts), which MEP has none of (only
a forward-pass-only statistic, no shared environment steps). Instead, the
population vmap axis here is *named* (``axis_name="population"``), and
``jax.lax.all_gather`` is used as a collective inside the training loop to
gather every lane's current params into every lane for that one forward pass
-- no restructuring of the self-play rollout/GAE/PPO-update body, which stays
close to ``marl/ippo.py``'s ``make_train``.

Recurrent-actor caveat (Hanabi / Overcooked-v2): MEP's objective is written as
``pi(a|s)`` -- state-conditioned, no history/hstate in its formalism, because
the paper's own environment (Overcooked) is fully observed. Hanabi and
Overcooked-v2 need recurrent actors here, which the paper's math doesn't
address. For the cross-member forward pass *only* (never each member's own
rollout, which keeps its own evolving hstate as normal), hstate is reset to
``policy.init_hstate(...)`` rather than gathering and reusing each member's
own trajectory-hstate against another lane's observation -- the closest
reading of the paper's literal state-only conditioning. For the "mlp"/
non-recurrent actor types (LBF), ``init_hstate`` is already the no-op hstate
ippo.py's own rollout uses, so this assumption changes nothing there; it only
bites for Hanabi ("rnn") and Overcooked-v2 ("cnn_rnn"). Flag in
``docs/tuning_record.md`` once those runs exist -- an extension the paper
doesn't cover, tracked openly rather than silently resolved.
"""

from __future__ import annotations

import logging
import time
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from oaht_bench.common.logging import RunLogger, nonfatal
from oaht_bench.common.plot_utils import get_metric_names
from oaht_bench.common.save_load_utils import save_train_run
from oaht_bench.configs.job import TeammateGenerationJob
from oaht_bench.envs import make_env
from oaht_bench.envs.log_wrapper import LogWrapper
from oaht_bench.envs.protocols import TrainingEnv
from oaht_bench.models.population_interface import AgentPopulation
from oaht_bench.population.loading import TrainOutput, get_mep_population
from oaht_bench.teammate_gen.marl.ippo import initialize_agent, log_metrics_intermediate
from oaht_bench.teammate_gen.marl.lr_schedule import make_lr_schedule
from oaht_bench.teammate_gen.marl.ppo_utils import (
    Transition,
    _create_minibatches,
    batchify,
    unbatchify,
)
from oaht_bench.teammate_gen.marl.reward_shaping import add_shaped_reward
from oaht_bench.teammate_gen.runtime import PpoRuntime

#: A trained MEP population: stacked parameters plus the policy class that
#: reads them. Leading axes of the parameters are ``(num_seeds, population_size)``.
MepPopulation = tuple[chex.ArrayTree, AgentPopulation]

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def population_entropy_bonus(member_log_probs: jnp.ndarray, coef: float) -> jnp.ndarray:
    """MEP's Eq. 6/9 reward bonus: ``-coef * log(mean_k pi_k(a|s))``.

    ``member_log_probs`` is every population member's log-probability of the
    same already-sampled action at the same observation (leading axis =
    population size). The population's mean-policy log-probability must be
    reduced as ``logsumexp(member_log_probs, axis=0) - log(N)`` --
    ``log(mean(probs))`` -- **not** ``mean(member_log_probs, axis=0)`` --
    ``mean(log(probs))``. Those are different quantities (Jensen's inequality
    makes the latter a strictly lower, biased estimate whenever members
    disagree), and training against the wrong one silently optimizes a
    different objective than the paper's. Kept as a standalone function so
    this reduction is unit-testable independent of the training loop.
    """
    n = member_log_probs.shape[0]
    pop_log_prob = jax.scipy.special.logsumexp(member_log_probs, axis=0) - jnp.log(n)
    return -coef * pop_log_prob


def make_mep_train(runtime, env, logger, population_entropy_coef, progress_callback=None):
    """Build the MEP training function. Must be ``jax.vmap``'d with
    ``axis_name="population"`` over ``population_size`` -- the population-
    entropy bonus reads that axis via ``jax.lax.all_gather``.
    """
    config = runtime

    def train(rng):
        rng, init_rng = jax.random.split(rng)
        policy, init_params = initialize_agent(
            config.actor_type, config.to_agent_dict(), env, init_rng
        )

        tx = optax.chain(
            optax.clip_by_global_norm(config.ppo.max_grad_norm),
            optax.adam(learning_rate=make_lr_schedule(config.ppo, config.num_updates), eps=1e-5),
        )
        train_state = TrainState.create(apply_fn=policy.network.apply, params=init_params, tx=tx)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, last_hstate, rng = runner_state

                rng, act_rng = jax.random.split(rng, 2)

                last_obs_batch = batchify(last_obs, env.agents, config.num_actors)
                last_done_batch = batchify(last_done, env.agents, config.num_actors)

                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config.num_actors).astype(jnp.float32)
                )

                obs_in = last_obs_batch.reshape(1, config.num_actors, -1)
                done_in = last_done_batch.reshape(1, config.num_actors)
                avail_in = avail_actions.reshape(1, config.num_actors, -1)

                action, value, pi, new_hstate = policy.get_action_value_policy(
                    params=train_state.params,
                    obs=obs_in,
                    done=done_in,
                    avail_actions=avail_in,
                    hstate=last_hstate,
                    rng=act_rng,
                )
                log_prob = pi.log_prob(action)

                # --- MEP population-entropy bonus (Eq. 6/9) ---
                # Gather every member's CURRENT params (forward-pass only, no
                # extra environment interaction), evaluate each member's
                # log-prob of THIS already-sampled action at THIS observation,
                # reduce via log-mean-exp. See module docstring for why this
                # must not be mean-of-log-probs, and for the hstate reset.
                gathered_params = jax.tree.map(
                    lambda x: jax.lax.all_gather(x, axis_name="population"), train_state.params
                )
                zero_hstate = policy.init_hstate(config.num_actors)

                def _member_log_prob(member_params):
                    _, _, member_pi, _ = policy.get_action_value_policy(
                        params=member_params,
                        obs=obs_in,
                        done=done_in,
                        avail_actions=avail_in,
                        hstate=zero_hstate,
                        rng=jax.random.PRNGKey(0),  # unused: only pi.log_prob is read
                    )
                    return member_pi.log_prob(action)

                member_log_probs = jax.vmap(_member_log_prob)(gathered_params)
                entropy_bonus = population_entropy_bonus(
                    member_log_probs, population_entropy_coef
                ).squeeze()

                action = action.squeeze()
                log_prob = log_prob.squeeze()
                value = value.squeeze()

                env_act = unbatchify(action, env.agents, config.num_envs, env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.num_envs)

                new_obs, new_env_state, reward, new_done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                reward = add_shaped_reward(
                    reward,
                    info,
                    env.agents,
                    horizon=config.ppo.reward_shaping_horizon,
                    global_env_step=update_steps * config.rollout_length * config.num_envs,
                )

                info = jax.tree.map(lambda x: x.reshape(config.num_actors), info)

                transition = Transition(
                    batchify(new_done, env.agents, config.num_actors).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config.num_actors).squeeze() + entropy_bonus,
                    log_prob,
                    last_obs_batch,
                    info,
                    avail_actions,
                )
                runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.rollout_length
            )

            train_state, env_state, last_obs, last_done, last_hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config.num_actors).reshape(
                1, config.num_actors, -1
            )
            last_done_batch = batchify(last_done, env.agents, config.num_actors).reshape(
                1, config.num_actors
            )
            last_avail_batch = jax.vmap(env.get_avail_actions)(env_state.env_state)
            last_avail_batch = jax.lax.stop_gradient(
                batchify(last_avail_batch, env.agents, config.num_actors).astype(jnp.float32)
            )

            _, last_val, _, _ = policy.get_action_value_policy(
                params=train_state.params,
                obs=last_obs_batch,
                done=last_done_batch,
                avail_actions=last_avail_batch,
                hstate=last_hstate,
                rng=jax.random.PRNGKey(0),
            )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config.ppo.gamma * next_value * (1 - done) - value
                    gae = delta + config.ppo.gamma * config.ppo.gae_lambda * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        _, value, pi, _ = policy.get_action_value_policy(
                            params=params,
                            obs=traj_batch.obs,
                            done=traj_batch.done,
                            avail_actions=traj_batch.avail_actions,
                            hstate=init_hstate,
                            rng=jax.random.PRNGKey(0),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config.ppo.clip_eps, config.ppo.clip_eps
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(ratio, 1.0 - config.ppo.clip_eps, 1.0 + config.ppo.clip_eps)
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config.ppo.value_coef * value_loss
                            - config.ppo.entropy_coef * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_minibatches(
                    traj_batch,
                    advantages,
                    targets,
                    init_hstate,
                    config.num_actors,
                    config.ppo.num_minibatches,
                    perm_rng,
                )
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            init_hstate = policy.init_hstate(config.num_actors)
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.ppo.update_epochs
            )
            train_state = update_state[0]

            def mask_and_mean(x, mask):
                return jnp.where(mask, x, 0).sum() / jnp.maximum(1, mask.sum())

            mask = traj_batch.info.get("returned_episode", jnp.ones_like(traj_batch.reward))
            metric = jax.tree.map(lambda x: mask_and_mean(x, mask), traj_batch.info)
            metric["update_steps"] = update_steps

            def callback(metrics):
                log_metrics_intermediate(metrics, logger)
                if progress_callback is not None:
                    progress_callback()

            jax.experimental.io_callback(callback, None, metric)

            rng = update_state[-1]
            update_steps += 1
            runner_state = (train_state, env_state, last_obs, last_done, last_hstate, rng)

            mask = metric["returned_episode"]
            n_episodes = mask.sum()
            condensed_metric = {}
            for key, val in metric.items():
                if key == "update_steps":
                    condensed_metric[key] = val
                elif key == "returned_episode":
                    condensed_metric[key] = n_episodes.astype(jnp.float32)
                else:
                    condensed_metric[key] = jnp.where(
                        n_episodes > 0,
                        jnp.where(mask, val, 0.0).sum() / jnp.maximum(n_episodes, 1),
                        0.0,
                    )
            condensed_metric["value_loss"] = loss_info[1][0].mean()
            condensed_metric["actor_loss"] = loss_info[1][1].mean()
            condensed_metric["entropy_loss"] = loss_info[1][2].mean()

            return (runner_state, update_steps), condensed_metric

        ckpt_and_eval_interval = config.num_updates // max(1, config.num_checkpoints - 1)
        num_ckpts = config.num_checkpoints

        def init_ckpt_array(params_pytree):
            return jax.tree.map(lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype), params_pytree)

        def _update_step_with_checkpoint(update_with_ckpt_runner_state, unused):
            (update_runner_state, checkpoint_array, ckpt_idx) = update_with_ckpt_runner_state
            update_runner_state, metric = _update_step(update_runner_state, None)
            _, update_steps = update_runner_state
            to_store = jnp.logical_or(
                jnp.equal(jnp.mod(update_steps - 1, ckpt_and_eval_interval), 0),
                jnp.equal(update_steps, config.num_updates),
            )

            def store_ckpt_fn(args):
                _checkpoint_array, _ckpt_idx = args
                new_checkpoint_array = jax.tree.map(
                    lambda c_arr, p: c_arr.at[_ckpt_idx].set(p),
                    _checkpoint_array,
                    update_runner_state[0][0].params,
                )
                return new_checkpoint_array, _ckpt_idx + 1

            def skip_ckpt_fn(args):
                return args

            checkpoint_array, ckpt_idx = jax.lax.cond(
                to_store, store_ckpt_fn, skip_ckpt_fn, (checkpoint_array, ckpt_idx)
            )
            runner_state = (update_runner_state, checkpoint_array, ckpt_idx)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        update_steps = 0
        init_hstate = policy.init_hstate(config.num_actors)
        init_done = {k: jnp.zeros((config.num_envs), dtype=bool) for k in env.agents + ["__all__"]}
        update_runner_state = (
            (train_state, env_state, obsv, init_done, init_hstate, _rng),
            update_steps,
        )
        checkpoint_array = init_ckpt_array(train_state.params)
        ckpt_idx = 0
        update_with_ckpt_runner_state = (update_runner_state, checkpoint_array, ckpt_idx)

        runner_state, metrics = jax.lax.scan(
            _update_step_with_checkpoint,
            update_with_ckpt_runner_state,
            xs=None,
            length=config.num_updates,
        )

        update_runner_state, checkpoint_array, final_ckpt_idx = runner_state

        return {
            "final_params": update_runner_state[0][0].params,
            "metrics": metrics,
            "checkpoints": checkpoint_array,
            "final_ckpt_idx": final_ckpt_idx,
        }

    return train


def train_mep_members(
    rng: chex.PRNGKey,
    env: TrainingEnv,
    population_size: int,
    runtime: PpoRuntime,
    population_entropy_coef: float,
    wandb_logger: RunLogger,
) -> TrainOutput:
    """Single seed of training an MEP population. The population axis is
    *named* (``axis_name="population"``) so the entropy bonus's
    ``jax.lax.all_gather`` can synchronize every member's current params at
    every step -- see module docstring."""
    rngs = jax.random.split(rng, population_size)
    train_jit = jax.jit(
        jax.vmap(
            make_mep_train(
                runtime, env, logger=wandb_logger, population_entropy_coef=population_entropy_coef
            ),
            axis_name="population",
        )
    )
    out = train_jit(rngs)
    return out


def run_mep(job: TeammateGenerationJob, wandb_logger: RunLogger) -> MepPopulation:
    """Train an MEP population from a validated job config.

    Stage 1 only -- see module docstring. Reads a
    :class:`~oaht_bench.configs.job.TeammateGenerationJob` directly, same
    contract as ``run_fcp``.
    """
    gen = job.generator
    rng = jax.random.PRNGKey(gen.train_seed)
    rngs = jax.random.split(rng, gen.num_seeds)

    env = make_env(job.env.env_name, job.env.env_kwargs())
    env = LogWrapper(env)

    runtime = PpoRuntime.from_config(
        ppo=gen.ppo,
        network=gen.network,
        actor_type=gen.actor_type,
        rollout_length=job.env.rollout_length,
        num_envs=gen.num_envs,
        total_timesteps=gen.total_timesteps,
        num_checkpoints=gen.num_checkpoints,
        num_agents=env.num_agents,
    )

    start_time = time.time()
    with jax.disable_jit(False):
        vmapped_train_fn = jax.jit(
            jax.vmap(
                partial(
                    train_mep_members,
                    env=env,
                    population_size=gen.population_size,
                    runtime=runtime,
                    population_entropy_coef=gen.population_entropy_coef,
                    wandb_logger=wandb_logger,
                )
            )
        )
        out = vmapped_train_fn(rngs)
    end_time = time.time()
    log.info(f"Training MEP population took {end_time - start_time:.2f} seconds.")

    flattened_partner_params, partner_population = get_mep_population(job, out, env)

    # Save FIRST so the checkpoint survives even if metric logging OOMs.
    out_savepath = save_train_run(out, job.run_dir(), savename="saved_train_run")
    with nonfatal("MEP post-training metrics"):
        log_metrics(job, out, wandb_logger, out_savepath)

    return flattened_partner_params, partner_population


def log_metrics(
    job: TeammateGenerationJob, out: TrainOutput, logger: RunLogger, out_savepath: str
) -> None:
    """Log statistics and record the saved train run as an artifact."""
    metric_names = get_metric_names(job.env.env_name)
    # metrics shape after mask_and_mean: (num_seeds, population_size, num_updates)
    member_metrics = out["metrics"]
    num_updates = member_metrics["returned_episode_returns"].shape[2]

    member_stat_means = {
        stat_name: np.mean(np.asarray(member_metrics[stat_name]), axis=(0, 1))
        for stat_name in metric_names
        if stat_name in member_metrics
    }

    for step in range(num_updates):
        for stat_name, stat_data in member_stat_means.items():
            logger.log_item(f"Train/Member_{stat_name}", stat_data[step], train_step=step)

    logger.commit()

    logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
