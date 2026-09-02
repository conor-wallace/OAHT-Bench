"""Rebuild a released population from a finished teammate-generation run.

Each generator writes the same four-key checkpoint but means something different
by it: FCP flattens a ``(runs, checkpoints)`` grid into one member axis, while
the others release ``final_params_conf``. Turning a checkpoint into "the
population" therefore needs a per-generator builder, and those builders live
here rather than beside the training code so that reading an artifact does not
require importing the trainer that produced it.

The builders take only ``(job, out, env)`` and touch only ``agents`` and
``configs``, which is what makes this package free of any dependency on
:mod:`oaht_bench.teammate_gen`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import chex
import jax

from oaht_bench.models.mlp_actor_critic_agent import (
    ActorWithConditionalCriticPolicy,
    MLPActorCriticPolicy,
)
from oaht_bench.models.population_interface import AgentPopulation
from oaht_bench.models.rnn_actor_critic_agent import (
    RNNActorCriticPolicy,
    RNNActorWithConditionalCriticPolicy,
)
from oaht_bench.configs.job import TeammateGenerationJob
from oaht_bench.envs.protocols import TrainingEnv
from oaht_bench.population.members import get_member_params

log = logging.getLogger(__name__)


class TrainOutput(TypedDict):
    r"""What the PPO training function returns.

    Written down because three functions index into this dict and none of them
    said what was in it. Leading axes are added by the ``vmap``\ s the callers
    apply, so shapes are described relative to a single training run.
    """

    #: Parameters at the end of training.
    final_params: chex.ArrayTree
    #: Per-update statistics, keyed by metric name.
    metrics: dict[str, chex.Array]
    #: Snapshots taken during training; leading axis is ``num_checkpoints``.
    checkpoints: chex.ArrayTree
    #: Index of the checkpoint selected as each member's final policy.
    final_ckpt_idx: chex.Array


def artifact_dir(run_dir: Path) -> Path:
    """Locate a run's Orbax checkpoint directory.

    Two layouts exist. The runner writes ``<run_dir>/artifacts/saved_train_run``,
    but the absorbed ``save_train_run`` resolves relative paths against
    ``REPO_PATH``, which after absorption points at ``src/oaht_bench`` rather
    than the repo root -- so the same run also appears under
    ``src/oaht_bench/results/...``. The two are byte-identical; this prefers the
    runner's layout and falls back, rather than silently scoring nothing.
    """
    run_dir = Path(run_dir)
    candidates = [
        run_dir / "artifacts" / "saved_train_run",
        run_dir / "saved_train_run",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        f"no saved_train_run under {run_dir}; looked in "
        f"{[str(c) for c in candidates]}. A run that crashed before its "
        f"checkpoint write cannot be re-scored -- it has to be re-run."
    )


def get_fcp_population(
    job: TeammateGenerationJob, out: TrainOutput, env: TrainingEnv
) -> FcpPopulation:
    '''Flatten each seed's partner pool for downstream use.'''
    gen = job.generator
    num_seeds = gen.num_seeds
    fcp_pop_size = gen.population_size * gen.num_checkpoints

    partner_params = out['checkpoints'] # shape is (num_seeds, partner_pop_size, num_ckpts, ...)
    flattened_partner_params = jax.tree.map(lambda x: x.reshape(num_seeds, fcp_pop_size, *x.shape[3:]), partner_params)

    # Dispatch matches initialize_agents.py's EGO_ACTOR_TYPE branching --
    # this is the same choice made at training time, reconstructed here so a
    # saved checkpoint can be scored without re-importing the trainer. RNN's
    # gru_hidden_dim isn't a network config field (MlpNetwork.to_agent_dict()
    # doesn't emit GRU_HIDDEN_DIM), so training used initialize_rnn_agent's
    # default of 64; matched here for the same reason, not independently
    # chosen. CoMeDi/BRDiv/L-BRDiv don't get this dispatch: there's no RNN
    # variant of ActorWithConditionalCriticPolicy, so actor_type="rnn" isn't
    # a valid choice for them yet (see docs/tuning_record.md).
    if gen.actor_type == "rnn":
        partner_policy = RNNActorCriticPolicy(
            action_dim=env.action_space(env.agents[1]).n,
            obs_dim=env.observation_space(env.agents[1]).shape[0],
            activation=gen.network.activation,
            gru_hidden_dim=64,
        )
    else:
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


def get_comedi_population(
    job: TeammateGenerationJob, out: TrainOutput, env: TrainingEnv
) -> CoMeDiPopulation:
    '''Build the partner population from a completed CoMeDi run.'''
    comedi_pop_size = job.generator.population_size

    # partner_params has shape (num_seeds, comedi_pop_size, ...)
    partner_params = out['final_params_conf']

    # Same dispatch as get_fcp_population/get_brdiv_population above, and the
    # same construction CoMeDi.py itself now uses -- see docs/tuning_record.md.
    policy_cls = (
        RNNActorWithConditionalCriticPolicy
        if job.generator.actor_type == "rnn_actor_with_conditional_critic"
        else ActorWithConditionalCriticPolicy
    )
    partner_policy = policy_cls(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0],
        pop_size=comedi_pop_size, # used to create onehot agent id
        activation=job.generator.network.activation,
    )

    # Create partner population
    partner_population = AgentPopulation(
        pop_size=comedi_pop_size,
        policy_cls=partner_policy
    )

    return partner_params, partner_population


# Returns confederates only; population_from_run pairs them with
# final_params_br so seats get their designed roles.
def get_brdiv_population(
    job: TeammateGenerationJob, out: TrainOutput, env: TrainingEnv
) -> PairedPopulation:
    '''
    Get the partner params and partner population for ego training.
    '''
    brdiv_pop_size = job.generator.population_size

    # partner_params has shape (num_seeds, brdiv_pop_size, ...)
    partner_params = out['final_params_conf']

    # Same dispatch as get_fcp_population above, and the same construction
    # BRDiv.py itself now uses -- see docs/tuning_record.md.
    policy_cls = (
        RNNActorWithConditionalCriticPolicy
        if job.generator.actor_type == "rnn_actor_with_conditional_critic"
        else ActorWithConditionalCriticPolicy
    )
    partner_policy = policy_cls(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0],
        pop_size=brdiv_pop_size, # used to create onehot agent id
        activation=job.generator.network.activation
    )

    # Create partner population
    partner_population = AgentPopulation(
        pop_size=brdiv_pop_size,
        policy_cls=partner_policy
    )

    return partner_params, partner_population


# Returns confederates only; population_from_run pairs them with
# final_params_br so seats get their designed roles.
def get_lbrdiv_population(
    job: TeammateGenerationJob, out: TrainOutput, env: TrainingEnv
) -> PairedPopulation:
    '''
    Get the partner params and partner population for ego training.
    '''
    pop_size = job.generator.population_size

    # partner_params has shape (num_seeds, pop_size, ...)
    partner_params = out['final_params_conf']

    # Same dispatch as get_fcp_population/get_brdiv_population above, and the
    # same construction LBRDiv.py itself now uses -- see docs/tuning_record.md.
    policy_cls = (
        RNNActorWithConditionalCriticPolicy
        if job.generator.actor_type == "rnn_actor_with_conditional_critic"
        else ActorWithConditionalCriticPolicy
    )
    partner_policy = policy_cls(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0],
        pop_size=pop_size, # used to create onehot agent id
        activation=job.generator.network.activation
    )

    # Create partner population
    partner_population = AgentPopulation(
        pop_size=pop_size,
        policy_cls=partner_policy
    )

    return partner_params, partner_population


@dataclass(frozen=True)
class LoadedPopulation:
    """A population read back off disk, with its seating roles intact.

    BRDiv and L-BRDiv train confederate/best-response *pairs* and save both sets.
    Their designed pairing is confederate ``i`` against best response ``j`` — a
    confederate is never trained to play with another confederate, so seating two
    of them is out of distribution and under-reports competence by 25-40% on LBF.
    Returning only ``final_params_conf``, as this used to, made that the default
    for anything that did not know to reach into the checkpoint for the other
    half.

    FCP and CoMeDi release a single set of self-play policies, so every seat
    draws from the same set and ``partner_params`` is ``None``.
    """

    params: Any
    #: Named to match AgentPopulation so crossplay accepts either.
    policy_cls: Any
    pop_size: int
    generator: str
    #: Best responses, for the paired generators only.
    partner_params: Any | None = None

    @property
    def paired(self) -> bool:
        """Whether seats have distinct roles rather than being interchangeable."""
        return self.partner_params is not None

    def seat(self, member_indices, *, seed_index: int = 0) -> list[tuple[Any, Any]]:
        """``(params, policy)`` per seat, honouring the generator's roles.

        For a paired population seat 0 is a confederate and seat 1 its best
        response, matching the diagonal that ``crossplay`` scores. For a
        homogeneous one every seat draws from the same set.
        """
        idx = list(member_indices)
        if not self.paired:
            return [(get_member_params(self.params, i, seed_index=seed_index), self.policy_cls)
                    for i in idx]
        if len(idx) != 2:
            raise ValueError(
                f"{self.generator} trains confederate/best-response pairs and has "
                f"no role for a third seat, but {len(idx)} seats were requested. "
                f"The paired generators assert num_agents == 2 during training too."
            )
        conf, br = idx
        return [
            (get_member_params(self.params, conf, seed_index=seed_index), self.policy_cls),
            (get_member_params(self.partner_params, br, seed_index=seed_index), self.policy_cls),
        ]


#: Which builder reads which generator's checkpoint.
_BUILDERS = {
    "fcp": get_fcp_population,
    "comedi": get_comedi_population,
    "brdiv": get_brdiv_population,
    "lbrdiv": get_lbrdiv_population,
}


def population_from_run(
    job: TeammateGenerationJob, out: Any, env: TrainingEnv
) -> LoadedPopulation:
    """Rebuild the released population for whichever generator produced ``out``.

    One definition of what a generator's population *is*, shared by scoring and
    by dataset collection. A second copy would let the two disagree about which
    policy a member index refers to — and, before this returned roles, they did:
    scoring paired confederates against best responses while collection paired
    confederates against each other.
    """
    name = job.generator.generator
    if name not in _BUILDERS:
        raise ValueError(f"no population builder for {name!r}. Known: {sorted(_BUILDERS)}")
    params, population = _BUILDERS[name](job, out, env)
    return LoadedPopulation(
        params=params,
        policy_cls=population.policy_cls,
        pop_size=population.pop_size,
        generator=name,
        # Paired generators save both halves; keeping only the confederates is
        # what made every downstream seat out-of-distribution.
        partner_params=out.get("final_params_br"),
    )
