"""Typed configs for the four teammate-generation algorithms (§7).

Replaces the untyped ``extra: dict[str, Any]`` escape hatch. Each generator has
structurally different hyperparameters — CoMeDi weights cross-play and mixed-play,
L-BRDiv learns Lagrange multipliers, BRDiv has a fixed cross-play weight, FCP has
none of these — and a dict cannot say so.

Fields are snake_case and named for what they mean. The absorbed training code
reads SCREAMING_CASE keys inherited from jax-aht's Hydra configs, but that is an
implementation detail of the boundary, not something a config author should have
to know; the translation happens once at the runtime layer
(:mod:`oaht_bench.teammate_gen.runtime`'s ``from_config`` / ``to_agent_dict``),
which reads these typed fields directly.

Defaults reproduce jax-aht's, so a config that sets nothing behaves as upstream
does. Population-size scaling guidance from §7.3 is documented on the fields it
governs, because the failure it prevents — a silently collapsed population — is
not visible until cross-play matrices are inspected much later.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from oaht_bench.configs.base import BaseConfig
from oaht_bench.configs.network import MlpNetwork

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
    # RNNActorWithConditionalCritic (agents/rnn_actor_critic.py): partial
    # observability on Overcooked-v2 for BRDiv/L-BRDiv. See
    # docs/tuning_record.md.
    "rnn_actor_with_conditional_critic",
    # CoMeDiRuntime.warmup()'s RNN analogue of pseudo_actor_with_conditional_critic
    # -- the network-shape-only variant a plain self-play IPPO trainer can drive
    # before any real population exists to condition on.
    "pseudo_rnn_actor_with_conditional_critic",
    # CNN+GRU variants for Overcooked-v2 (Gessler et al., ICLR 2025, App. C.1.1):
    # a convolutional stem over the grid observation, which the paper found
    # necessary to learn good policies there. "cnn_rnn" is FCP's shared-trunk
    # actor-critic; "cnn_rnn_actor_with_conditional_critic" is the BRDiv/L-BRDiv/
    # CoMeDi conditional-critic variant; the pseudo is CoMeDi's warmup shape.
    # See models/cnn_rnn_actor_critic.py and docs/tuning_record.md.
    "cnn_rnn",
    "cnn_rnn_actor_with_conditional_critic",
    "pseudo_cnn_rnn_actor_with_conditional_critic",
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
    lr_warmup: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Fraction of the total updates spent warming the learning rate "
        "linearly from 0 to `learning_rate`, after which it cosine-decays to 0 "
        "(OvercookedV2 App. D). 0 disables warmup -- the default -- leaving the "
        "existing linear (`anneal_lr`) or constant schedule byte-for-byte unchanged, "
        "so every previously tuned config is unaffected.",
    )
    reward_shaping_horizon: float = Field(
        default=0.0,
        ge=0,
        description="Environment steps over which a linear 1->0 anneal is applied to "
        "the environment's shaped reward before it is added to the base reward "
        "(OvercookedV2 App. C/D). 0 disables shaping -- the default -- so sparse-reward "
        "environments (LBF, Hanabi, Overcooked-v1) are unaffected. Only environments "
        "that surface info['shaped_reward'] (the Overcooked-v2 wrapper) use it.",
    )


class GeneratorBase(BaseConfig):
    """Fields common to every generator."""

    population_size: int = Field(
        default=5,
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
    num_eval_episodes: int = Field(
        default=20,
        gt=0,
        description="Episodes per in-training evaluation. CoMeDi, BRDiv and "
        "L-BRDiv use these to estimate the cross-play returns their diversity "
        "objectives are computed from, so it affects the population produced, "
        "not just reporting.",
    )
    train_seed: int = 20374
    num_seeds: int = Field(default=1, gt=0)
    ppo: PpoHyperparams = Field(default_factory=PpoHyperparams)
    network: MlpNetwork = Field(
        default_factory=MlpNetwork,
        description="Policy architecture. Previously implicit -- the absorbed "
        "initializers defaulted it via dict.get, so it never entered a run's "
        "content hash.",
    )


class FcpConfig(GeneratorBase):
    """Fictitious Co-Play: independent self-play runs, snapshotted during training.

    Diversity is *of competence*, not of convention. The paper defines the
    mid-training checkpoint as the point where the agent reaches 50% of its final
    reward — not an arbitrary step-schedule snapshot (§7.3).
    """

    generator: Literal["fcp"] = "fcp"
    actor_type: ActorType = "mlp"
    total_timesteps: float = Field(default=1e6, gt=0, description="Per member trained.")


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

    @model_validator(mode="after")
    def _minibatches_must_fit_the_env_axis(self) -> CoMeDiConfig:
        """CoMeDi minibatches over environments, not over actors.

        Its ``_create_minibatches`` calls pass ``NUM_ENVS`` where the other
        generators pass ``NUM_ACTORS``, so the batch axis is ``num_envs`` wide.
        Exceeding it fails as an opaque reshape error many frames deep
        ("cannot reshape array of shape (128, 4) into [128, 8, -1]").
        """
        if self.ppo.num_minibatches > self.num_envs:
            raise ValueError(
                f"CoMeDi minibatches over environments: num_minibatches="
                f"{self.ppo.num_minibatches} exceeds num_envs={self.num_envs}. "
                f"Raise num_envs to at least {self.ppo.num_minibatches}, or lower "
                f"num_minibatches."
            )
        return self


class BrDivConfig(GeneratorBase):
    """BRDiv: maximize best-response diversity over the conf x br cross-play matrix."""

    generator: Literal["brdiv"] = "brdiv"
    # No actor_type axis existed for BRDiv/L-BRDiv before overcooked_v2's RNN
    # support -- BRDiv.py/LBRDiv.py hardcoded ActorWithConditionalCriticPolicy
    # directly at the construction site, not driven by ACTOR_TYPE the way FCP
    # and CoMeDi already were. "rnn_actor_with_conditional_critic" is the only
    # other value either file's construction site currently understands.
    actor_type: ActorType = "actor_with_conditional_critic"
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


class LBrDivConfig(GeneratorBase):
    """L-BRDiv: BRDiv's objective with the weights learned as Lagrange multipliers."""

    generator: Literal["lbrdiv"] = "lbrdiv"
    # See BrDivConfig.actor_type -- same gap, same fix.
    actor_type: ActorType = "actor_with_conditional_critic"
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


class RpgConfig(GeneratorBase):
    """AD-RPG (Rational Adversarial Diversity): general-sum diversity via manipulators.

    Clean-room reimplementation of the ``doublesided_RAD`` algorithm from Lauffer
    et al. (NeurIPS 2025); the upstream repo is unlicensed and on an incompatible
    stack, so nothing is absorbed — this is authored from the paper (see
    ``PROVENANCE.md``). ``population_size`` is the number of diversity particles N
    (the paper's ``NUM_PARTICLES``, which it runs at 2); each particle is a base
    policy shaped by a paired manipulator, and the released population is the N
    converged base policies (a self-play set, like CoMeDi/BRDiv release members).

    ``ppo`` supplies the *base* agent's PPO hyperparameters; the manipulator has
    its own learning rate and entropy, since it optimizes the diversity objective
    rather than task return.
    """

    generator: Literal["rpg"] = "rpg"
    total_timesteps: float = Field(default=1e7, gt=0)
    n_lookahead: int = Field(
        default=1,
        gt=0,
        description="Inner base-update steps the manipulator differentiates "
        "through (the opponent-shaping horizon). Upstream default 1.",
    )
    dice_lambda: float = Field(
        default=0.99,
        gt=0,
        le=1,
        description="Loaded-DiCE past-dependency discount in the base surrogate "
        "loss; controls the bias/variance of the higher-order gradient estimate.",
    )
    partnerplay_ratio: float = Field(
        default=0.1,
        ge=0,
        description="Weight moved from self-play to cross-play in the base "
        "objective (the paper's PARTNERPLAY_RATIO): base self-play carries "
        "(1 - N*ratio), each cross-play pairing carries `ratio`. Keeps a base "
        "policy in-distribution against the partners it is scored with.",
    )
    off_diag_factor: float = Field(
        default=0.25,
        ge=0,
        description="Scales the manipulator's cross-play *minimization* term "
        "(weight -off_diag_factor/(N-1) per off-diagonal pairing) against its "
        "self-play *maximization* term (weight +1). Higher pushes harder for "
        "mutual incompatibility.",
    )
    manipulator_lr: float = Field(default=2.5e-4, gt=0)
    manipulator_entropy_coef: float = Field(default=0.0, ge=0)


#: Discriminated union, selected by ``generator``.
GeneratorConfig = Annotated[
    FcpConfig | CoMeDiConfig | BrDivConfig | LBrDivConfig | RpgConfig,
    Field(discriminator="generator"),
]
