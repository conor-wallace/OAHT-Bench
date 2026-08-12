"""Values derived from a config and an environment, computed once per run.

The absorbed training code stashed these back into its config dict::

    config["NUM_ACTORS"]     = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"]    = config["TOTAL_TIMESTEPS"] // ...
    config["MINIBATCH_SIZE"] = ...

That made one object serve two jobs: the settings a human authored, and a
scratchpad for quantities the code computed. Our configs are frozen because
their hash is a run's provenance record — a config that mutates mid-run cannot
identify what produced an artifact — so the two roles have to separate.

:class:`PpoRuntime` is the second role. It is built from a config plus an
environment, is itself frozen, and never appears in a config file.
"""

from __future__ import annotations

from typing import Any, TypedDict

import chex
from pydantic import Field

from oaht_bench.configs.base import BaseConfig
from oaht_bench.configs.network import MlpNetwork
from oaht_bench.configs.teammate_gen import ActorType, PpoHyperparams


class PpoRuntime(BaseConfig):
    """Everything the PPO training loop needs, authored and derived.

    Built by :meth:`from_config`; do not construct directly unless a test wants
    a specific shape.
    """

    # --- authored, carried through from the job config ---
    ppo: PpoHyperparams
    network: MlpNetwork
    actor_type: ActorType
    rollout_length: int = Field(gt=0)
    num_envs: int = Field(gt=0)
    total_timesteps: float = Field(gt=0)
    num_checkpoints: int = Field(gt=0)
    pop_size: int | None = Field(
        default=None,
        description="Population width the conditional-critic policies condition "
        "on. Required by the *_conditional_critic actor types and unused by the "
        "others, so it lives here rather than on MlpNetwork.",
    )

    # --- derived from the above plus the environment ---
    num_actors: int = Field(gt=0, description="env.num_agents * num_envs.")
    num_updates: int = Field(gt=0, description="Gradient updates over the run.")
    minibatch_size: int = Field(gt=0)

    @classmethod
    def from_config(
        cls,
        *,
        ppo: PpoHyperparams,
        network: MlpNetwork,
        actor_type: ActorType,
        rollout_length: int,
        num_envs: int,
        total_timesteps: float,
        num_checkpoints: int,
        num_agents: int,
        pop_size: int | None = None,
    ) -> PpoRuntime:
        """Compute the derived quantities, validating that they are usable.

        The absorbed code computed these inline with integer division and no
        checks, so a budget too small for even one update produced
        ``num_updates == 0`` and a training loop that silently did nothing.
        """
        num_actors = num_agents * num_envs
        num_updates = int(total_timesteps // rollout_length // num_envs)
        minibatch_size = num_actors * rollout_length // ppo.num_minibatches

        if num_updates < 1:
            raise ValueError(
                f"total_timesteps={total_timesteps:g} gives {num_updates} updates at "
                f"rollout_length={rollout_length} and num_envs={num_envs}; training "
                f"would be a no-op. Raise total_timesteps above "
                f"{rollout_length * num_envs}."
            )
        if minibatch_size < 1:
            raise ValueError(
                f"num_minibatches={ppo.num_minibatches} exceeds the batch of "
                f"{num_actors * rollout_length} transitions "
                f"(num_actors={num_actors} x rollout_length={rollout_length}); "
                f"minibatches would be empty."
            )

        if "conditional_critic" in actor_type and pop_size is None:
            raise ValueError(
                f"actor_type={actor_type!r} conditions its critic on a population "
                f"index, so pop_size is required. Omitting it surfaces as a bare "
                f"KeyError('POP_SIZE') from inside policy construction."
            )

        return cls(
            ppo=ppo,
            network=network,
            actor_type=actor_type,
            rollout_length=rollout_length,
            num_envs=num_envs,
            total_timesteps=total_timesteps,
            num_checkpoints=num_checkpoints,
            pop_size=pop_size,
            num_actors=num_actors,
            num_updates=num_updates,
            minibatch_size=minibatch_size,
        )

    def to_agent_dict(self) -> dict[str, Any]:
        """Keys the absorbed agent initializers read."""
        out = self.network.to_agent_dict()
        if self.pop_size is not None:
            out["POP_SIZE"] = self.pop_size
        return out


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


class CoMeDiRuntime(BaseConfig):
    """Derived values for a CoMeDi run.

    CoMeDi's update budget is not ``total_timesteps // rollout_length //
    num_envs``. Each outer iteration spends steps on *population selection*
    rollouts as well as training, so the divisor is the sum of both. Getting this
    wrong changes how many updates happen without any error.
    """

    # --- authored ---
    ppo: PpoHyperparams
    network: MlpNetwork
    actor_type: ActorType
    rollout_length: int = Field(gt=0)
    num_envs: int = Field(gt=0)
    total_timesteps_per_iteration: float = Field(gt=0)
    num_checkpoints: int = Field(gt=0)
    population_size: int = Field(gt=0)
    num_eval_episodes: int = Field(gt=0)
    num_argmax_rollout_episodes: int = Field(gt=0)
    cross_play_weight: float
    mixed_play_weight: float

    # --- derived ---
    num_game_agents: int = Field(gt=0)
    num_actors: int = Field(gt=0)
    num_controlled_actors: int = Field(gt=0)
    num_updates: int = Field(gt=0)

    @classmethod
    def from_config(cls, gen: Any, rollout_length: int, num_agents: int) -> CoMeDiRuntime:
        """Build from a :class:`~oaht_bench.configs.teammate_gen.CoMeDiConfig`."""
        if num_agents != 2:
            raise ValueError(f"CoMeDi assumes exactly 2 agents; this environment has {num_agents}.")
        num_actors = num_agents * gen.num_envs

        # Population-selection rollouts are halved because the implementation
        # vmaps over the whole population, evaluating against members a
        # sequential implementation would not revisit.
        selection_steps = (
            gen.population_size * gen.num_argmax_rollout_episodes * rollout_length // 2
        )
        # Four rollout phases per update: SP, XP, MP, MP2.
        training_steps = 4 * rollout_length * gen.num_envs
        steps_per_update = selection_steps + training_steps
        num_updates = int(gen.total_timesteps_per_iteration // steps_per_update)

        if num_updates < 1:
            raise ValueError(
                f"total_timesteps_per_iteration="
                f"{gen.total_timesteps_per_iteration:g} gives {num_updates} updates: "
                f"each costs {steps_per_update} steps ({selection_steps} for "
                f"population selection plus {training_steps} for the four rollout "
                f"phases). Raise it above {steps_per_update}."
            )

        return cls(
            ppo=gen.ppo,
            network=gen.network,
            actor_type=gen.actor_type,
            rollout_length=rollout_length,
            num_envs=gen.num_envs,
            total_timesteps_per_iteration=gen.total_timesteps_per_iteration,
            num_checkpoints=gen.num_checkpoints,
            population_size=gen.population_size,
            num_eval_episodes=gen.num_eval_episodes,
            num_argmax_rollout_episodes=gen.num_argmax_rollout_episodes,
            cross_play_weight=gen.cross_play_weight,
            mixed_play_weight=gen.mixed_play_weight,
            num_game_agents=num_agents,
            num_actors=num_actors,
            num_controlled_actors=num_actors,
            num_updates=num_updates,
        )

    def warmup(self) -> PpoRuntime:
        """The PPO runtime for CoMeDi's self-play warmup phase.

        Replaces the ``warmup_config = dict(config)`` copy that overrode
        ``TOTAL_TIMESTEPS`` and ``ACTOR_TYPE``. Making it a method states which
        two values differ, instead of leaving a mutated copy for a reader to
        diff against the original.
        """
        return PpoRuntime.from_config(
            ppo=self.ppo,
            network=self.network,
            actor_type="pseudo_actor_with_conditional_critic",
            rollout_length=self.rollout_length,
            num_envs=self.num_envs,
            total_timesteps=self.total_timesteps_per_iteration,
            num_checkpoints=self.num_checkpoints,
            num_agents=self.num_game_agents,
            pop_size=self.population_size,
        )

    def to_agent_dict(self) -> dict[str, Any]:
        """Keys the absorbed agent initializers read."""
        return {**self.network.to_agent_dict(), "POP_SIZE": self.population_size}


class PairedDiversityRuntime(BaseConfig):
    """Derived values for BRDiv and L-BRDiv.

    Both train *paired* populations — a confederate and its best response — so
    they carry separate actor counts for each side, and both minibatch over those
    counts rather than over a combined actor axis.

    The two algorithms share this shape entirely; they differ only in how the
    cross-play term is weighted (BRDiv fixes it, L-BRDiv learns it), which is
    authored config rather than anything derived.
    """

    # --- authored ---
    ppo: PpoHyperparams
    network: MlpNetwork
    rollout_length: int = Field(gt=0)
    num_envs: int = Field(gt=0)
    total_timesteps: float = Field(gt=0)
    num_checkpoints: int = Field(gt=0)
    population_size: int = Field(gt=0)
    num_eval_episodes: int = Field(gt=0)

    #: BRDiv only. Fixed cross-play weight; see the config field for why it must
    #: not be rescaled with population size.
    cross_play_weight: float | None = None
    #: L-BRDiv only. Learned multipliers, which *do* need (n_ref/n)^2 scaling.
    lagrange_learning_rate: float | None = None
    tolerance_factor: float | None = None

    # --- derived ---
    num_game_agents: int = Field(gt=0)
    num_conf_actors: int = Field(gt=0)
    num_br_actors: int = Field(gt=0)
    num_updates: int = Field(gt=0)

    @classmethod
    def from_config(cls, gen: Any, rollout_length: int, num_agents: int) -> PairedDiversityRuntime:
        if num_agents != 2:
            raise ValueError(
                f"{gen.generator} trains confederate/best-response pairs and assumes "
                f"exactly 2 agents; this environment has {num_agents}."
            )
        num_updates = int(gen.total_timesteps // (rollout_length * gen.num_envs))
        if num_updates < 1:
            raise ValueError(
                f"total_timesteps={gen.total_timesteps:g} gives {num_updates} updates "
                f"at rollout_length={rollout_length} and num_envs={gen.num_envs}; "
                f"training would be a no-op. Raise it above "
                f"{rollout_length * gen.num_envs}."
            )
        if gen.ppo.num_minibatches > gen.num_envs:
            raise ValueError(
                f"{gen.generator} minibatches over each side's actors, which is "
                f"num_envs={gen.num_envs} wide; num_minibatches="
                f"{gen.ppo.num_minibatches} exceeds it."
            )

        return cls(
            ppo=gen.ppo,
            network=gen.network,
            rollout_length=rollout_length,
            num_envs=gen.num_envs,
            total_timesteps=gen.total_timesteps,
            num_checkpoints=gen.num_checkpoints,
            population_size=gen.population_size,
            num_eval_episodes=gen.num_eval_episodes,
            cross_play_weight=getattr(gen, "cross_play_weight", None),
            lagrange_learning_rate=getattr(gen, "lagrange_learning_rate", None),
            tolerance_factor=getattr(gen, "tolerance_factor", None),
            num_game_agents=num_agents,
            num_conf_actors=gen.num_envs,
            num_br_actors=gen.num_envs,
            num_updates=num_updates,
        )

    def to_agent_dict(self) -> dict[str, Any]:
        """Keys the absorbed agent initializers read."""
        return {**self.network.to_agent_dict(), "POP_SIZE": self.population_size}
