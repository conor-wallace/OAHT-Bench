'''Implementation of the Fictitious Co-Play teammate generation algorithm (Strouse et al. NeurIPS 2021)
https://proceedings.neurips.cc/paper/2021/hash/797134c3e42371bb4979a462eb2f042a-Abstract.html
'''
from __future__ import annotations

import logging
import time
from functools import partial

import chex
import jax
import numpy as np

from oaht_bench.agents.mlp_actor_critic_agent import MLPActorCriticPolicy
from oaht_bench.agents.population_interface import AgentPopulation
from oaht_bench.envs import make_env
from oaht_bench.envs.log_wrapper import LogWrapper
from oaht_bench.teammate_gen.marl.ippo import make_train as make_ppo_train
from oaht_bench.common.plot_utils import get_metric_names
from oaht_bench.common.save_load_utils import save_train_run
from oaht_bench.common.logging import RunLogger, nonfatal
from oaht_bench.configs.job import TeammateGenerationJob
from oaht_bench.envs.protocols import TrainingEnv
from oaht_bench.teammate_gen.runtime import PpoRuntime, TrainOutput

#: A trained population: stacked parameters plus the policy class that reads them.
#: Leading axes of the parameters are ``(num_seeds, population_size * num_checkpoints)``.
FcpPopulation = tuple[chex.ArrayTree, AgentPopulation]

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_fcp_population(
    job: TeammateGenerationJob, out: TrainOutput, env: TrainingEnv
) -> FcpPopulation:
    '''Flatten each seed's partner pool for downstream use.'''
    gen = job.generator
    num_seeds = gen.num_seeds
    fcp_pop_size = gen.population_size * gen.num_checkpoints

    partner_params = out['checkpoints'] # shape is (num_seeds, partner_pop_size, num_ckpts, ...)
    flattened_partner_params = jax.tree.map(lambda x: x.reshape(num_seeds, fcp_pop_size, *x.shape[3:]), partner_params)

    partner_policy = MLPActorCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0],
        activation=gen.network.activation,
    )

    # Create partner population
    partner_population = AgentPopulation(
        pop_size=fcp_pop_size,
        policy_cls=partner_policy
    )

    return flattened_partner_params, partner_population

def train_fcp_partners(
    rng: chex.PRNGKey,
    env: TrainingEnv,
    population_size: int,
    runtime: PpoRuntime,
    wandb_logger: RunLogger,
) -> TrainOutput:
    '''Single seed of training an FCP pool.'''
    rngs = jax.random.split(rng, population_size)
    train_jit = jax.jit(jax.vmap(make_ppo_train(runtime, env, logger=wandb_logger)))
    out = train_jit(rngs)
    return out

def run_fcp(job: TeammateGenerationJob, wandb_logger: RunLogger) -> FcpPopulation:
    '''Train a pool of FCP partners from a validated job config.

    OAHT-Bench: reads a :class:`~oaht_bench.configs.job.TeammateGenerationJob`
    directly rather than a config dict. Parameters are reached by attribute
    (``job.generator.population_size``), so a typo is a load-time error and every
    value is the one recorded in the run's content hash.
    '''
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
            partial(train_fcp_partners,
                    env=env,
                    population_size=gen.population_size,
                    runtime=runtime,
                    wandb_logger=wandb_logger)
            )
        )
        out = vmapped_train_fn(rngs)
    end_time = time.time()
    log.info(f"Training FCP partners took {end_time - start_time:.2f} seconds.")

    flattened_partner_params, partner_population = get_fcp_population(job, out, env)

    # Save FIRST so the checkpoint survives even if metric logging OOMs
    # on long runs. Same pattern as teammate_generation/train_ego.py.
    # OAHT-Bench: read the output directory from the config instead of Hydra's
    # global. The benchmark drives these functions directly from a validated
    # job config, so there is no HydraConfig to reach into, and an implicit
    # global is the wrong place for something that determines where a released
    # artifact lands.
    out_savepath = save_train_run(out, job.run_dir(), savename="saved_train_run")
    with nonfatal("FCP post-training metrics"):
        log_metrics(job, out, wandb_logger, out_savepath)

    return flattened_partner_params, partner_population

def log_metrics(
    job: TeammateGenerationJob,
    out: TrainOutput,
    logger: RunLogger,
    out_savepath: str,
) -> None:
    '''Log statistics and record the saved train run as an artifact.'''
    metric_names = get_metric_names(job.env.env_name)
    # After mask_and_mean in ippo, metrics have shape
    # (num_seeds, partner_pop_size, num_partner_updates)
    partner_metrics = out["metrics"]
    num_partner_updates = partner_metrics["returned_episode_returns"].shape[2]

    # Average over seeds and pop members → (num_partner_updates,)
    partner_stat_means = {
        stat_name: np.mean(np.asarray(partner_metrics[stat_name]), axis=(0, 1))
        for stat_name in metric_names
        if stat_name in partner_metrics
    }

    for step in range(num_partner_updates):
        for stat_name, stat_data in partner_stat_means.items():
            logger.log_item(f"Train/Partner_{stat_name}", stat_data[step], train_step=step)

    logger.commit()

    logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
