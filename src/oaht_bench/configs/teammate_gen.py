"""Typed configs for the four teammate-generation algorithms (§7).

Replaces the untyped ``extra: dict[str, Any]`` escape hatch. Each generator has
structurally different hyperparameters — CoMeDi weights cross-play and mixed-play,
L-BRDiv learns Lagrange multipliers, BRDiv has a fixed cross-play weight, FCP has
none of these — and a dict cannot say so.

Fields are snake_case and named for what they mean. The absorbed training code
reads SCREAMING_CASE keys inherited from jax-aht's Hydra configs, but that is an
implementation detail of the boundary, not something a config author should have
to know; ``to_algorithm_dict`` is the single place the translation happens, the
same way :meth:`~oaht_bench.configs.env.EnvConfigBase.env_kwargs` handles it for
environments.

Defaults reproduce jax-aht's, so a config that sets nothing behaves as upstream
does. Population-size scaling guidance from §7.3 is documented on the fields it
governs, because the failure it prevents — a silently collapsed population — is
not visible until cross-play matrices are inspected much later.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from oaht_bench.configs.base import BaseConfig

#: Policy architectures the absorbed ``initialize_agents`` dispatches on. A plain
#: ``str`` here would let a typo through to an ``if/elif`` chain with no ``else``,
#: which either crashes deep inside training or silently builds a different
#: architecture than intended.
ActorType = Literal[
    "mlp",
    "rnn",
    "s5",
    "actor_with_double_critic",
    "pseudo_actor_with_double_critic",
    "actor_with_conditional_critic",
    "pseudo_actor_with_conditional_critic",
]


class PpoHyperparams(BaseConfig):
    """PPO settings shared by all four generators."""

    learning_rate: float = Field(default=1e-4, gt=0)
    update_epochs: int = Field(default=15, gt=0)
    num_minibatches: int = Field(default=4, gt=0)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, gt=0, le=1)
    clip_eps: float = Field(default=0.05, gt=0)
    entropy_coef: float = Field(default=0.01, ge=0)
    value_coef: float = Field(default=0.5, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    anneal_lr: bool = False

    def to_algorithm_dict(self) -> dict[str, Any]:
        return {
            "LR": self.learning_rate,
            "UPDATE_EPOCHS": self.update_epochs,
            "NUM_MINIBATCHES": self.num_minibatches,
            "GAMMA": self.gamma,
            "GAE_LAMBDA": self.gae_lambda,
            "CLIP_EPS": self.clip_eps,
            "ENT_COEF": self.entropy_coef,
            "VF_COEF": self.value_coef,
            "MAX_GRAD_NORM": self.max_grad_norm,
            "ANNEAL_LR": self.anneal_lr,
        }


class GeneratorBase(BaseConfig):
    """Fields common to every generator."""

    population_size: int = Field(
        default=4,
        gt=0,
        description="Members trained. Growing this dilutes per-policy self-play "
        "data quadratically — partners are sampled independently, so a given "
        "policy's self-play draw probability is 1/n^2, not 1/n. Scale "
        "`num_envs` and the timestep budget with it (§7.3).",
    )
    num_checkpoints: int = Field(
        default=5,
        gt=0,
        description="Snapshots per training run. FCP's own ablation shows "
        "checkpoint diversity is what carries the method, and that architectural "
        "diversity adds nothing — so do not trim this to save time (§7.3).",
    )
    num_envs: int = Field(default=64, gt=0)
    train_seed: int = 20374
    num_seeds: int = Field(default=1, gt=0)
    ppo: PpoHyperparams = Field(default_factory=PpoHyperparams)

    def _base_algorithm_dict(self) -> dict[str, Any]:
        """The SCREAMING_CASE keys the absorbed training code reads."""
        return {
            "ALG": self.generator,
            "NUM_CHECKPOINTS": self.num_checkpoints,
            "PARTNER_POP_SIZE": self.population_size,
            "NUM_ENVS": self.num_envs,
            "TRAIN_SEED": self.train_seed,
            "NUM_SEEDS": self.num_seeds,
            **self.ppo.to_algorithm_dict(),
        }


class FcpConfig(GeneratorBase):
    """Fictitious Co-Play: independent self-play runs, snapshotted during training.

    Diversity is *of competence*, not of convention. The paper defines the
    mid-training checkpoint as the point where the agent reaches 50% of its final
    reward — not an arbitrary step-schedule snapshot (§7.3).
    """

    generator: Literal["fcp"] = "fcp"
    actor_type: ActorType = "mlp"
    total_timesteps: float = Field(default=1e6, gt=0, description="Per member trained.")
    num_envs: int = Field(default=8, gt=0)

    def to_algorithm_dict(self) -> dict[str, Any]:
        return {
            **self._base_algorithm_dict(),
            "ACTOR_TYPE": self.actor_type,
            "TOTAL_TIMESTEPS": self.total_timesteps,
        }


def _comedi_ppo() -> PpoHyperparams:
    """CoMeDi's base config uses 8 minibatches rather than the shared default of 4."""
    return PpoHyperparams(num_minibatches=8)


class CoMeDiConfig(GeneratorBase):
    """CoMeDi: maximize self-play, minimize cross-play against the *most compatible*
    existing convention, and maximize mixed-play.

    Construction is greedy and sequential — members are added one at a time, and
    cross-play is minimized only against the single best-matching existing member.
    """

    generator: Literal["comedi"] = "comedi"
    actor_type: ActorType = "actor_with_conditional_critic"
    total_timesteps_per_iteration: float = Field(default=1.2e7, gt=0)
    num_envs: int = Field(default=48, gt=0)
    num_argmax_rollout_episodes: int = Field(default=20, gt=0)
    cross_play_weight: float = Field(
        default=1.0, description="CoMeDi's alpha: weight on cross-play minimization."
    )
    mixed_play_weight: float = Field(
        default=0.5,
        description="CoMeDi's beta. Load-bearing, not a nuisance parameter: "
        "mixed-play is what prevents handshake degeneracy, where members learn to "
        "signal identity and then deliberately sabotage cross-play. That inflates "
        "apparent diversity and makes the cross-play matrix diagnostic meaningless "
        "(§7.4).",
    )
    ppo: PpoHyperparams = Field(default_factory=_comedi_ppo)

    def to_algorithm_dict(self) -> dict[str, Any]:
        return {
            **self._base_algorithm_dict(),
            "ACTOR_TYPE": self.actor_type,
            "TOTAL_TIMESTEPS_PER_ITERATION": self.total_timesteps_per_iteration,
            "NUM_ARGMAX_ROLLOUT_EPS": self.num_argmax_rollout_episodes,
            "COMEDI_ALPHA": self.cross_play_weight,
            "COMEDI_BETA": self.mixed_play_weight,
        }


class BrDivConfig(GeneratorBase):
    """BRDiv: maximize best-response diversity over the conf x br cross-play matrix."""

    generator: Literal["brdiv"] = "brdiv"
    total_timesteps: float = Field(default=4.5e7, gt=0)
    cross_play_weight: float = Field(
        default=1.0,
        description="BRDiv's XP_LOSS_WEIGHTS. **Do not rescale with population "
        "size.** BRDiv.py builds sp_weight=(1+2w)*(n/2) and xp_weight=w*(n/(2(n-1))), "
        "whose n factors exactly cancel the sampling probabilities P(SP)=1/n and "
        "P(XP)=(n-1)/n — expected per-sample contributions are independent of n. A "
        "collapsed population at larger n is a sample-count problem; raise "
        "`num_envs` and `total_timesteps` instead (§7.3).",
    )

    def to_algorithm_dict(self) -> dict[str, Any]:
        return {
            **self._base_algorithm_dict(),
            "TOTAL_TIMESTEPS": self.total_timesteps,
            "XP_LOSS_WEIGHTS": self.cross_play_weight,
        }


class LBrDivConfig(GeneratorBase):
    """L-BRDiv: BRDiv's objective with the weights learned as Lagrange multipliers."""

    generator: Literal["lbrdiv"] = "lbrdiv"
    total_timesteps: float = Field(default=4.5e7, gt=0)
    tolerance_factor: float = Field(
        default=0.1, description="Require self-play minus cross-play > this."
    )
    lagrange_learning_rate: float = Field(
        default=0.01,
        gt=0,
        description="**Scale with population size**, unlike BRDiv's fixed weight. "
        "These multipliers are learned by SGD on an unnormalized sum over ~n^2 pair "
        "terms, so the same value produces a proportionally larger update at larger "
        "n. Scale by ~(n_ref/n)^2 relative to the population it was tuned at "
        "(n_ref=3 for the 0.01 default). Left unscaled at n=5 this produced entropy "
        "runaway to ~49 and pg_loss to -25 (§7.3).",
    )

    def to_algorithm_dict(self) -> dict[str, Any]:
        return {
            **self._base_algorithm_dict(),
            "TOTAL_TIMESTEPS": self.total_timesteps,
            "TOLERANCE_FACTOR": self.tolerance_factor,
            "LAGRANGE_LR": self.lagrange_learning_rate,
        }


#: Discriminated union, selected by ``generator``.
GeneratorConfig = Annotated[
    FcpConfig | CoMeDiConfig | BrDivConfig | LBrDivConfig,
    Field(discriminator="generator"),
]
