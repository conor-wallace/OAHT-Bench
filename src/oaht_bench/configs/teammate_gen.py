"""Typed configs for the four teammate-generation algorithms (§7).

Replaces the untyped ``extra: dict[str, Any]`` escape hatch. Each generator has
structurally different hyperparameters — CoMeDi has ``COMEDI_ALPHA`` and
``COMEDI_BETA``, L-BRDiv has ``LAGRANGE_LR`` and ``TOLERANCE_FACTOR``, BRDiv has
``XP_LOSS_WEIGHTS``, FCP has none of these — and a dict cannot say so.

Defaults are jax-aht's, so a config that sets nothing reproduces upstream
behaviour. Population-size scaling guidance from §7.3 is documented on the
fields it applies to, because the failure it prevents (a silently collapsed
population) is not visible until cross-play matrices are inspected much later.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from oaht_bench.configs.base import BaseConfig


class PpoHyperparams(BaseConfig):
    """PPO settings shared by all four generators."""

    LR: float = Field(default=1e-4, gt=0)
    UPDATE_EPOCHS: int = Field(default=15, gt=0)
    NUM_MINIBATCHES: int = Field(default=4, gt=0)
    GAMMA: float = Field(default=0.99, gt=0, le=1)
    GAE_LAMBDA: float = Field(default=0.95, gt=0, le=1)
    CLIP_EPS: float = Field(default=0.05, gt=0)
    ENT_COEF: float = Field(default=0.01, ge=0)
    VF_COEF: float = Field(default=0.5, ge=0)
    MAX_GRAD_NORM: float = Field(default=1.0, gt=0)
    ANNEAL_LR: bool = False


class GeneratorBase(BaseConfig):
    """Fields common to every generator."""

    NUM_CHECKPOINTS: int = Field(
        default=5,
        gt=0,
        description="Snapshots per training run. FCP's ablation shows checkpoint "
        "diversity is what matters, so do not reduce this to save time (§7.3).",
    )
    PARTNER_POP_SIZE: int = Field(default=4, gt=0)
    NUM_ENVS: int = Field(default=64, gt=0)
    TRAIN_SEED: int = 20374
    NUM_SEEDS: int = Field(default=1, gt=0)
    ppo: PpoHyperparams = Field(default_factory=PpoHyperparams)

    def _algorithm_dict(self) -> dict[str, Any]:
        """The ``algorithm`` block jax-aht's runners read."""
        out: dict[str, Any] = {
            "ALG": self.generator,
            "NUM_CHECKPOINTS": self.NUM_CHECKPOINTS,
            "PARTNER_POP_SIZE": self.PARTNER_POP_SIZE,
            "NUM_ENVS": self.NUM_ENVS,
            "TRAIN_SEED": self.TRAIN_SEED,
            "NUM_SEEDS": self.NUM_SEEDS,
        }
        out.update(self.ppo.model_dump())
        return out


class FcpConfig(GeneratorBase):
    """Fictitious Co-Play: independent self-play runs, snapshotted during training.

    Diversity is *of competence*, not of convention. The mid-training checkpoint
    is defined by the paper as the point where the agent reaches 50% of its final
    reward — not an arbitrary step-schedule snapshot (§7.3).
    """

    generator: Literal["fcp"] = "fcp"
    ACTOR_TYPE: str = "mlp"
    TOTAL_TIMESTEPS: float = Field(default=1e6, gt=0, description="Per member trained.")
    NUM_ENVS: int = Field(default=8, gt=0)

    def to_algorithm_dict(self) -> dict[str, Any]:
        d = self._algorithm_dict()
        d.update({"ACTOR_TYPE": self.ACTOR_TYPE, "TOTAL_TIMESTEPS": self.TOTAL_TIMESTEPS})
        return d


class CoMeDiConfig(GeneratorBase):
    """CoMeDi: maximize self-play, minimize cross-play against the most compatible
    existing convention, maximize mixed-play.

    Construction is greedy and sequential — members are added one at a time.
    """

    generator: Literal["comedi"] = "comedi"
    ACTOR_TYPE: str = "actor_with_conditional_critic"
    TOTAL_TIMESTEPS_PER_ITERATION: float = Field(default=1.2e7, gt=0)
    NUM_ENVS: int = Field(default=48, gt=0)
    NUM_MINIBATCHES_OVERRIDE: int | None = Field(
        default=8, description="CoMeDi's base config uses 8 rather than the shared default of 4."
    )
    NUM_ARGMAX_ROLLOUT_EPS: int = Field(default=20, gt=0)
    COMEDI_ALPHA: float = Field(default=1.0, description="Cross-play minimization weight.")
    COMEDI_BETA: float = Field(
        default=0.5,
        description="Mixed-play weight. Load-bearing, not a nuisance parameter: "
        "mixed-play is what prevents handshake degeneracy, where agents signal "
        "identity and then deliberately sabotage cross-play, which would make the "
        "cross-play matrix diagnostic meaningless (§7.4).",
    )

    def to_algorithm_dict(self) -> dict[str, Any]:
        d = self._algorithm_dict()
        d.update(
            {
                "ACTOR_TYPE": self.ACTOR_TYPE,
                "TOTAL_TIMESTEPS_PER_ITERATION": self.TOTAL_TIMESTEPS_PER_ITERATION,
                "NUM_ARGMAX_ROLLOUT_EPS": self.NUM_ARGMAX_ROLLOUT_EPS,
                "COMEDI_ALPHA": self.COMEDI_ALPHA,
                "COMEDI_BETA": self.COMEDI_BETA,
            }
        )
        if self.NUM_MINIBATCHES_OVERRIDE is not None:
            d["NUM_MINIBATCHES"] = self.NUM_MINIBATCHES_OVERRIDE
        return d


class BrDivConfig(GeneratorBase):
    """BRDiv: maximize best-response diversity over the conf x br cross-play matrix."""

    generator: Literal["brdiv"] = "brdiv"
    TOTAL_TIMESTEPS: float = Field(default=4.5e7, gt=0)
    XP_LOSS_WEIGHTS: float = Field(
        default=1.0,
        description="Cross-play loss weight. **Do not rescale with population "
        "size.** BRDiv.py:389-391 builds sp_weight=(1+2*XP)*(n/2) and "
        "xp_weight=XP*(n/(2(n-1))), whose n factors exactly cancel the sampling "
        "probabilities P(SP)=1/n and P(XP)=(n-1)/n — the expected per-sample "
        "contributions are independent of n (§7.3).",
    )

    def to_algorithm_dict(self) -> dict[str, Any]:
        d = self._algorithm_dict()
        d.update({"TOTAL_TIMESTEPS": self.TOTAL_TIMESTEPS, "XP_LOSS_WEIGHTS": self.XP_LOSS_WEIGHTS})
        return d


class LBrDivConfig(GeneratorBase):
    """L-BRDiv: BRDiv's objective with the weights learned as Lagrange multipliers."""

    generator: Literal["lbrdiv"] = "lbrdiv"
    TOTAL_TIMESTEPS: float = Field(default=4.5e7, gt=0)
    TOLERANCE_FACTOR: float = Field(
        default=0.1, description="Require self-play minus cross-play > this."
    )
    LAGRANGE_LR: float = Field(
        default=0.01,
        gt=0,
        description="**Scale with population size.** Unlike BRDiv's fixed weights, "
        "these multipliers are learned by SGD on an unnormalized sum over ~n^2 pair "
        "terms, so the same value produces a larger update at larger n. Scale by "
        "~(n_ref/n)^2 relative to the population it was tuned at (n_ref=3 for the "
        "0.01 default). Leaving it unscaled at n=5 produced entropy runaway to ~49 "
        "and pg_loss to -25 (§7.3).",
    )

    def to_algorithm_dict(self) -> dict[str, Any]:
        d = self._algorithm_dict()
        d.update(
            {
                "TOTAL_TIMESTEPS": self.TOTAL_TIMESTEPS,
                "TOLERANCE_FACTOR": self.TOLERANCE_FACTOR,
                "LAGRANGE_LR": self.LAGRANGE_LR,
            }
        )
        return d


#: Discriminated union, selected by ``generator``.
GeneratorConfig = Annotated[
    FcpConfig | CoMeDiConfig | BrDivConfig | LBrDivConfig,
    Field(discriminator="generator"),
]
