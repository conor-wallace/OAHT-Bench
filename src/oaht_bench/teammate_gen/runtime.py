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

from pydantic import Field

import chex

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
    """What the PPO training function returns.

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
