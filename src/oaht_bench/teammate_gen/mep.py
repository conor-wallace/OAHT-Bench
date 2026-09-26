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


def _mask_and_mean(x, mask):
    """Mean of ``x`` over the entries where ``mask`` is true, else 0."""
    return jnp.where(mask, x, 0).sum() / jnp.maximum(1, mask.sum())


def _is_checkpoint_step(update_steps, num_updates, num_checkpoints):
    """True on update steps whose params should be snapshotted: ``num_checkpoints``
    evenly spaced steps across training, plus always the final step (which may
    not land exactly on that spacing)."""
    ckpt_interval = num_updates // max(1, num_checkpoints - 1)
    return jnp.logical_or(
        jnp.equal(jnp.mod(update_steps - 1, ckpt_interval), 0),
        jnp.equal(update_steps, num_updates),
    )


def _maybe_store_checkpoint(checkpoint_array, ckpt_idx, params, should_store):
    """Write ``params`` into ``checkpoint_array[ckpt_idx]`` and advance ``ckpt_idx``
    if ``should_store``, else leave both unchanged."""

    def store(args):
        _checkpoint_array, _ckpt_idx, _params = args
        new_checkpoint_array = jax.tree.map(
            lambda c_arr, p: c_arr.at[_ckpt_idx].set(p), _checkpoint_array, _params
        )
        return new_checkpoint_array, _ckpt_idx + 1

    def skip(args):
        _checkpoint_array, _ckpt_idx, _ = args
        return _checkpoint_array, _ckpt_idx

    return jax.lax.cond(should_store, store, skip, (checkpoint_array, ckpt_idx, params))


class MepTrainer:
    """Builds and runs one MEP population member's training loop.

    Use as ``jax.vmap(trainer.train, axis_name="population")`` over
    ``population_size`` -- the population-entropy bonus reads that axis via
    ``jax.lax.all_gather``.

    ``gradient_accumulation_steps > 1`` wraps the optimizer in
    ``optax.MultiSteps``: every ``_env_step``/rollout collection below still
    runs at ``config.num_envs``, but the optimizer only actually updates
    params once every ``gradient_accumulation_steps`` rollouts, having
    averaged their gradients. Params stay frozen for the whole accumulation
    window (``MultiSteps`` returns a zero update on intermediate calls), so
    every rollout and every PPO minibatch/epoch pass inside that window sees
    the exact same policy snapshot -- the on-policy ratio in ``_loss_fn``
    stays valid throughout, and nothing else in this class needs to know
    accumulation is happening. See
    ``docs/tuning_record.md``'s MEP-on-Hanabi section for why (GPU memory) and
    ``marl/lr_schedule.py`` for how the LR schedule stays correctly paced.
    """

    def __init__(
        self,
        config,
        env,
        logger,
        population_entropy_coef,
        progress_callback=None,
        gradient_accumulation_steps=1,
    ):
        self.config = config
        self.env = env
        self.logger = logger
        self.population_entropy_coef = population_entropy_coef
        self.progress_callback = progress_callback
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.policy, _ = initialize_agent(
            config.actor_type, config.to_agent_dict(), env, jax.random.PRNGKey(0)
        )

    def train(self, rng):
        rng, init_rng = jax.random.split(rng)
        _, init_rng = jax.random.split(init_rng)
        init_params = self.policy.init_params(init_rng)

        tx = optax.chain(
            optax.clip_by_global_norm(self.config.ppo.max_grad_norm),
            optax.adam(
                learning_rate=make_lr_schedule(
                    self.config.ppo,
                    self.config.num_updates,
                    accumulation_steps=self.gradient_accumulation_steps,
                ),
                eps=1e-5,
            ),
        )
        if self.gradient_accumulation_steps > 1:
            every_k = (
                self.gradient_accumulation_steps
                * self.config.ppo.update_epochs
                * self.config.ppo.num_minibatches
            )
            tx = optax.MultiSteps(tx, every_k_schedule=every_k)
        train_state = TrainState.create(
            apply_fn=self.policy.network.apply, params=init_params, tx=tx
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, self.config.num_envs)
        obsv, env_state = jax.vmap(self.env.reset, in_axes=(0,))(reset_rng)

        rng, _rng = jax.random.split(rng)
        update_steps = 0
        init_hstate = self.policy.init_hstate(self.config.num_actors)
        init_done = {
            k: jnp.zeros((self.config.num_envs), dtype=bool) for k in self.env.agents + ["__all__"]
        }
        update_runner_state = (
            (train_state, env_state, obsv, init_done, init_hstate, _rng),
            update_steps,
        )
        checkpoint_array = self._init_ckpt_array(train_state.params)
        ckpt_idx = 0
        update_with_ckpt_runner_state = (update_runner_state, checkpoint_array, ckpt_idx)

        runner_state, metrics = jax.lax.scan(
            self._update_step_with_checkpoint,
            update_with_ckpt_runner_state,
            xs=None,
            length=self.config.num_updates,
        )

        update_runner_state, checkpoint_array, final_ckpt_idx = runner_state

        return {
            "final_params": update_runner_state[0][0].params,
            "metrics": metrics,
            "checkpoints": checkpoint_array,
            "final_ckpt_idx": final_ckpt_idx,
        }

    def _init_ckpt_array(self, params_pytree):
        num_ckpts = self.config.num_checkpoints
        return jax.tree.map(lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype), params_pytree)

    def _update_step_with_checkpoint(self, update_with_ckpt_runner_state, unused):
        (update_runner_state, checkpoint_array, ckpt_idx) = update_with_ckpt_runner_state
        update_runner_state, metric = self._update_step(update_runner_state, None)
        _, update_steps = update_runner_state
        should_store = _is_checkpoint_step(
            update_steps, self.config.num_updates, self.config.num_checkpoints
        )
        checkpoint_array, ckpt_idx = _maybe_store_checkpoint(
            checkpoint_array, ckpt_idx, update_runner_state[0][0].params, should_store
        )
        runner_state = (update_runner_state, checkpoint_array, ckpt_idx)
        return runner_state, metric

    def _update_step(self, update_runner_state, unused):
        runner_state, update_steps = update_runner_state

        env_step = partial(self._env_step, update_steps=update_steps)
        runner_state, traj_batch = jax.lax.scan(
            env_step, runner_state, None, self.config.rollout_length
        )

        train_state, env_state, last_obs, last_done, last_hstate, rng = runner_state
        last_obs_batch = batchify(last_obs, self.env.agents, self.config.num_actors).reshape(
            1, self.config.num_actors, -1
        )
        last_done_batch = batchify(last_done, self.env.agents, self.config.num_actors).reshape(
            1, self.config.num_actors
        )
        last_avail_batch = jax.vmap(self.env.get_avail_actions)(env_state.env_state)
        last_avail_batch = jax.lax.stop_gradient(
            batchify(last_avail_batch, self.env.agents, self.config.num_actors).astype(jnp.float32)
        )

        _, last_val, _, _ = self.policy.get_action_value_policy(
            params=train_state.params,
            obs=last_obs_batch,
            done=last_done_batch,
            avail_actions=last_avail_batch,
            hstate=last_hstate,
            rng=jax.random.PRNGKey(0),
        )
        last_val = last_val.squeeze()

        advantages, targets = self._calculate_gae(traj_batch, last_val)

        init_hstate = self.policy.init_hstate(self.config.num_actors)
        update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(
            self._update_epoch, update_state, None, self.config.ppo.update_epochs
        )
        train_state = update_state[0]

        mask = traj_batch.info.get("returned_episode", jnp.ones_like(traj_batch.reward))
        metric = jax.tree.map(lambda x: _mask_and_mean(x, mask), traj_batch.info)
        metric["update_steps"] = update_steps

        jax.experimental.io_callback(self._callback, None, metric)

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

    def _callback(self, metrics):
        log_metrics_intermediate(metrics, self.logger)
        if self.progress_callback is not None:
            self.progress_callback()

    def _env_step(self, runner_state, unused, *, update_steps):
        train_state, env_state, last_obs, last_done, last_hstate, rng = runner_state

        rng, act_rng = jax.random.split(rng, 2)

        last_obs_batch = batchify(last_obs, self.env.agents, self.config.num_actors)
        last_done_batch = batchify(last_done, self.env.agents, self.config.num_actors)

        avail_actions = jax.vmap(self.env.get_avail_actions)(env_state.env_state)
        avail_actions = jax.lax.stop_gradient(
            batchify(avail_actions, self.env.agents, self.config.num_actors).astype(jnp.float32)
        )

        obs_in = last_obs_batch.reshape(1, self.config.num_actors, -1)
        done_in = last_done_batch.reshape(1, self.config.num_actors)
        avail_in = avail_actions.reshape(1, self.config.num_actors, -1)

        action, value, pi, new_hstate = self.policy.get_action_value_policy(
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
        zero_hstate = self.policy.init_hstate(self.config.num_actors)

        def _member_log_prob(member_params):
            _, _, member_pi, _ = self.policy.get_action_value_policy(
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
            member_log_probs, self.population_entropy_coef
        ).squeeze()

        action = action.squeeze()
        log_prob = log_prob.squeeze()
        value = value.squeeze()

        env_act = unbatchify(action, self.env.agents, self.config.num_envs, self.env.num_agents)
        env_act = {k: v.flatten() for k, v in env_act.items()}

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, self.config.num_envs)

        new_obs, new_env_state, reward, new_done, info = jax.vmap(self.env.step, in_axes=(0, 0, 0))(
            rng_step, env_state, env_act
        )

        reward = add_shaped_reward(
            reward,
            info,
            self.env.agents,
            horizon=self.config.ppo.reward_shaping_horizon,
            global_env_step=update_steps * self.config.rollout_length * self.config.num_envs,
        )

        info = jax.tree.map(lambda x: x.reshape(self.config.num_actors), info)

        transition = Transition(
            batchify(new_done, self.env.agents, self.config.num_actors).squeeze(),
            action,
            value,
            batchify(reward, self.env.agents, self.config.num_actors).squeeze() + entropy_bonus,
            log_prob,
            last_obs_batch,
            info,
            avail_actions,
        )
        runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng)
        return runner_state, transition

    def _get_advantages(self, gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = transition.done, transition.value, transition.reward
        delta = reward + self.config.ppo.gamma * next_value * (1 - done) - value
        gae = delta + self.config.ppo.gamma * self.config.ppo.gae_lambda * (1 - done) * gae
        return (gae, value), gae

    def _calculate_gae(self, traj_batch, last_val):
        _, advantages = jax.lax.scan(
            self._get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch.value

    def _loss_fn(self, params, traj_batch, gae, targets, *, init_hstate):
        _, value, pi, _ = self.policy.get_action_value_policy(
            params=params,
            obs=traj_batch.obs,
            done=traj_batch.done,
            avail_actions=traj_batch.avail_actions,
            hstate=init_hstate,
            rng=jax.random.PRNGKey(0),
        )
        log_prob = pi.log_prob(traj_batch.action)

        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
            -self.config.ppo.clip_eps, self.config.ppo.clip_eps
        )
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(ratio, 1.0 - self.config.ppo.clip_eps, 1.0 + self.config.ppo.clip_eps) * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
        entropy = pi.entropy().mean()

        total_loss = (
            loss_actor
            + self.config.ppo.value_coef * value_loss
            - self.config.ppo.entropy_coef * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy)

    def _update_minbatch(self, train_state, batch_info):
        init_hstate, traj_batch, advantages, targets = batch_info

        loss_fn = partial(self._loss_fn, init_hstate=init_hstate)
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, total_loss

    def _update_epoch(self, update_state, unused):
        train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
        rng, perm_rng = jax.random.split(rng)
        minibatches = _create_minibatches(
            traj_batch,
            advantages,
            targets,
            init_hstate,
            self.config.num_actors,
            self.config.ppo.num_minibatches,
            perm_rng,
        )
        train_state, total_loss = jax.lax.scan(self._update_minbatch, train_state, minibatches)
        update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
        return update_state, total_loss


def train_mep_members(
    rng: chex.PRNGKey,
    env: TrainingEnv,
    population_size: int,
    runtime: PpoRuntime,
    population_entropy_coef: float,
    wandb_logger: RunLogger,
    gradient_accumulation_steps: int = 1,
) -> TrainOutput:
    """Single seed of training an MEP population. The population axis is
    *named* (``axis_name="population"``) so the entropy bonus's
    ``jax.lax.all_gather`` can synchronize every member's current params at
    every step -- see module docstring."""
    rngs = jax.random.split(rng, population_size)
    trainer = MepTrainer(
        runtime,
        env,
        logger=wandb_logger,
        population_entropy_coef=population_entropy_coef,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    train_jit = jax.jit(jax.vmap(trainer.train, axis_name="population"))
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
                    gradient_accumulation_steps=gen.gradient_accumulation_steps,
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
