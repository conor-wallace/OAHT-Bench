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
from typing import Any, NotRequired, TypedDict

import chex
import jax
import jax.numpy as jnp
import numpy as np

from oaht_bench.models.mlp_actor_critic_agent import (
    ActorWithConditionalCriticPolicy,
    MLPActorCriticPolicy,
)
from oaht_bench.models.population_interface import AgentPopulation
from oaht_bench.models.rnn_actor_critic_agent import (
    RNNActorCriticPolicy,
    RNNActorWithConditionalCriticPolicy,
)
from oaht_bench.models.cnn_rnn_actor_critic_agent import (
    CNNRNNActorCriticPolicy,
    CNNRNNActorWithConditionalCriticPolicy,
)
from oaht_bench.models.initialize_agents import _unwrap_obs_shape
from oaht_bench.configs.job import TeammateGenerationJob
from oaht_bench.envs.protocols import TrainingEnv
from oaht_bench.population.members import Member, get_member_params, released_members

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
    #: MEP-specific: the continuously-tracked best-performing params per member
    #: (MEP reference's best_sparse_rew_avg mechanism), not populated by every
    #: generator sharing this TypedDict.
    best_params: NotRequired[chex.ArrayTree]
    #: MEP-specific: the measured return best_params was selected at.
    best_score: NotRequired[chex.Array]


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
    elif gen.actor_type == "cnn_rnn":
        # GRU_HIDDEN_DIM isn't a network config field, so training used
        # initialize_cnn_rnn_agent's default of 128 (the paper's value);
        # matched here, like the rnn branch matches its own default.
        partner_policy = CNNRNNActorCriticPolicy(
            action_dim=env.action_space(env.agents[1]).n,
            obs_dim=env.observation_space(env.agents[1]).shape[0],
            obs_shape=_unwrap_obs_shape(env),
            activation=gen.network.activation,
            fc_hidden_dim=gen.network.hidden_dim,
            gru_hidden_dim=128,
        )
    else:
        partner_policy = MLPActorCriticPolicy(
            action_dim=env.action_space(env.agents[1]).n,
            obs_dim=env.observation_space(env.agents[1]).shape[0],
            activation=gen.network.activation,
            fc_hidden_dim=gen.network.hidden_dim,
            num_layers=gen.network.num_layers,
        )

    # Create partner population
    partner_population = AgentPopulation(
        pop_size=fcp_pop_size,
        policy_cls=partner_policy
    )

    return flattened_partner_params, partner_population


def get_mep_population(
    job: TeammateGenerationJob, out: TrainOutput, env: TrainingEnv
) -> MepPopulation:
    '''Build the population from a completed MEP run.

    Stage 1 only (see ``teammate_gen/mep.py``): by default this reads
    ``final_params`` directly, one member per population slot -- the same
    shape ``get_comedi_population`` produces. MEP does not condition on
    population index, so its actor-type dispatch mirrors
    ``get_fcp_population``'s (rnn/cnn_rnn/mlp), not the conditional-critic
    policies CoMeDi/BRDiv/L-BRDiv use.

    ``gen.release_checkpoints`` (default ``False``) switches to FCP's
    convention instead: flatten every saved (evenly-spaced, time-based)
    checkpoint into its own population member, matching MEP's own paper's
    practice ("we use multiple checkpoints ... to form the final
    population"). MEP's checkpoint array comes out of ``teammate_gen/mep.py``
    shaped ``(num_checkpoints, num_seeds, population_size, ...)`` --
    checkpoints *outermost*, since a single ``train()`` call already vmaps
    jointly over ``(num_seeds, population_size)`` before checkpointing adds
    its own axis in front -- the opposite leading-axis order from FCP's
    ``(num_seeds, population_size, num_checkpoints, ...)``, so this moves the
    checkpoint axis in before reshaping, rather than reusing FCP's reshape
    directly. Flat index is ``member * num_checkpoints + checkpoint``, the same
    convention ``released_members`` uses for FCP.

    ``gen.release_best`` (default ``False``, independent of
    ``release_checkpoints``) additionally releases each member's
    continuously-tracked best-performing snapshot (``out['best_params']``,
    the MEP reference's own ``best_sparse_rew_avg`` mechanism -- see
    ``teammate_gen/mep.py``'s module docstring) as ``population_size`` more
    members, appended after whatever ``release_checkpoints`` already
    produced. Already shaped ``(num_seeds, population_size, ...)``, so no
    reorder is needed the way ``checkpoints`` needs one. Both toggles
    together give ``population_size * (num_checkpoints + 1)`` released
    members -- the reference's exact per-member checkpoint-type count (3
    time-based + 1 best) that turns MEP's population size 4 into TIPR's
    reported 12-20 distinct opponent policies (App. F.1).
    '''
    gen = job.generator
    num_seeds = gen.num_seeds

    if gen.release_checkpoints:
        mep_pop_size = gen.population_size * gen.num_checkpoints
        # (num_checkpoints, num_seeds, population_size, ...) ->
        # (num_seeds, population_size, num_checkpoints, ...)
        reordered = jax.tree.map(lambda x: jnp.moveaxis(x, 0, 2), out['checkpoints'])
        partner_params = jax.tree.map(
            lambda x: x.reshape(num_seeds, mep_pop_size, *x.shape[3:]), reordered
        )
    else:
        mep_pop_size = gen.population_size
        # partner_params has shape (num_seeds, mep_pop_size, ...)
        partner_params = out['final_params']

    if gen.release_best:
        partner_params = jax.tree.map(
            lambda a, b: jnp.concatenate([a, b], axis=1), partner_params, out['best_params']
        )
        mep_pop_size += gen.population_size

    if gen.actor_type == "rnn":
        partner_policy = RNNActorCriticPolicy(
            action_dim=env.action_space(env.agents[1]).n,
            obs_dim=env.observation_space(env.agents[1]).shape[0],
            activation=gen.network.activation,
            gru_hidden_dim=64,
        )
    elif gen.actor_type == "cnn_rnn":
        partner_policy = CNNRNNActorCriticPolicy(
            action_dim=env.action_space(env.agents[1]).n,
            obs_dim=env.observation_space(env.agents[1]).shape[0],
            obs_shape=_unwrap_obs_shape(env),
            activation=gen.network.activation,
            fc_hidden_dim=gen.network.hidden_dim,
            gru_hidden_dim=128,
        )
    else:
        partner_policy = MLPActorCriticPolicy(
            action_dim=env.action_space(env.agents[1]).n,
            obs_dim=env.observation_space(env.agents[1]).shape[0],
            activation=gen.network.activation,
            fc_hidden_dim=gen.network.hidden_dim,
            num_layers=gen.network.num_layers,
        )

    partner_population = AgentPopulation(
        pop_size=mep_pop_size,
        policy_cls=partner_policy
    )

    return partner_params, partner_population


def _conditional_critic_policy(job: TeammateGenerationJob, env: TrainingEnv, pop_size: int):
    """Reconstruct the conditional-critic policy a CoMeDi/BRDiv/L-BRDiv checkpoint
    was trained with, for scoring. Mirrors the construction each generator and
    ``initialize_actor_with_conditional_critic`` make: the recurrent flat-obs
    variant, the CNN+GRU grid variant (Overcooked-v2), or the MLP default. GRU/FC
    dims match the training-time defaults for the same reason the fcp/rnn branch
    does -- they are not independently chosen. See docs/tuning_record.md.
    """
    actor_type = job.generator.actor_type
    kwargs = dict(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0],
        pop_size=pop_size,  # used to create onehot agent id
        activation=job.generator.network.activation,
    )
    if actor_type == "rnn_actor_with_conditional_critic":
        return RNNActorWithConditionalCriticPolicy(**kwargs)
    if actor_type == "cnn_rnn_actor_with_conditional_critic":
        return CNNRNNActorWithConditionalCriticPolicy(
            obs_shape=_unwrap_obs_shape(env),
            fc_hidden_dim=job.generator.network.hidden_dim,
            gru_hidden_dim=128,
            **kwargs,
        )
    return ActorWithConditionalCriticPolicy(
        fc_hidden_dim=job.generator.network.hidden_dim,
        num_layers=job.generator.network.num_layers,
        **kwargs,
    )


def get_comedi_population(
    job: TeammateGenerationJob, out: TrainOutput, env: TrainingEnv
) -> CoMeDiPopulation:
    '''Build the partner population from a completed CoMeDi run.'''
    comedi_pop_size = job.generator.population_size

    # partner_params has shape (num_seeds, comedi_pop_size, ...)
    partner_params = out['final_params_conf']

    partner_policy = _conditional_critic_policy(job, env, comedi_pop_size)

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

    partner_policy = _conditional_critic_policy(job, env, brdiv_pop_size)

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

    partner_policy = _conditional_critic_policy(job, env, pop_size)

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


#: Which builder reads which *generated* population's checkpoint. Scripted OOD
#: teammates (§7.6) would be the other population kind, but they are not built yet:
#: they have no checkpoint and are heterogeneous (each member a different policy
#: class), so they do not fit ``LoadedPopulation``'s "one ``policy_cls`` + a
#: member-indexed param tree" shape. When added, they belong on the heterogeneous
#: roster (:mod:`oaht_bench.population.members`'s ``Member``, which already
#: carries a per-member ``policy_cls``), not on this checkpoint dispatch.
_BUILDERS = {
    "fcp": get_fcp_population,
    "comedi": get_comedi_population,
    "brdiv": get_brdiv_population,
    "lbrdiv": get_lbrdiv_population,
    "mep": get_mep_population,
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


def _load_population(pop_dir: Path, env: TrainingEnv):
    """Rebuild a released population the same way scoring and collection do."""
    from oaht_bench.common.save_load_utils import load_train_run
    from oaht_bench.configs import load_job

    pop_dir = Path(pop_dir)
    run_dir = pop_dir.parent.parent if pop_dir.name == "saved_train_run" else pop_dir
    job = load_job(run_dir / "job.json")
    out = load_train_run(str(artifact_dir(run_dir)))
    return population_from_run(job, out, env), job


def build_roster(population_dirs: list[Path], env: TrainingEnv, *, seed_index: int = 0) -> list[Member]:
    """Flatten released populations into one roster of individual policies.

    ``population_dirs`` are released run directories (``populations/<env>/<gen>/``),
    one per generator. Each contributes its *released* members -- the converged
    checkpoints for FCP, one per convention for the others (see
    :func:`~oaht_bench.population.members.released_members`) -- as self / conf / br
    policies.
    """
    roster: list[Member] = []
    for pop_dir in population_dirs:
        loaded, job = _load_population(pop_dir, env)
        generator = job.generator.generator
        for m in released_members(job, loaded.pop_size):
            m = int(m)
            if loaded.paired:
                roster.append(
                    Member(
                        generator,
                        m,
                        "conf",
                        get_member_params(loaded.params, m, seed_index=seed_index),
                        loaded.policy_cls,
                    )
                )
                roster.append(
                    Member(
                        generator,
                        m,
                        "br",
                        get_member_params(loaded.partner_params, m, seed_index=seed_index),
                        loaded.policy_cls,
                    )
                )
            else:
                roster.append(
                    Member(
                        generator,
                        m,
                        "self",
                        get_member_params(loaded.params, m, seed_index=seed_index),
                        loaded.policy_cls,
                    )
                )
    return roster


#: Roles that are designed teammates. A paired generator's own ``br`` role is
#: excluded -- it was always excluded from being seated as a teammate, and now
#: that ``ppo_br`` supplies the ego axis, it plays no other part in this matrix.
_TEAMMATE_ROLES = ("self", "conf")


def load_teammate_policies(
    population_dirs: list[Path], env: TrainingEnv, *, seed_index: int = 0
) -> list[Member]:
    """The designed-teammate subset of :func:`build_roster` -- ``self``/``conf`` only.

    This is the ``K``-sized identity list both the crossplay matrix and pooled
    dataset collection index by: every teammate has exactly one entry here, and
    (once ``ppo_br`` has been trained against it) exactly one dedicated best
    response, so ego and teammate share this same index space.
    """
    return [
        e
        for e in build_roster(population_dirs, env, seed_index=seed_index)
        if e.role in _TEAMMATE_ROLES
    ]


def load_ego_policies(br_dir: Path | str, env: TrainingEnv) -> list[Member]:
    """Load every trained best-response ego, tagged with the teammate identity it
    was trained for.

    Reads a ``ppo_br`` run (``br_manifest.json`` + the ``br_by_source`` artifact) and
    returns one :class:`~oaht_bench.population.members.Member` per manifest entry, in
    manifest order -- not reordered or position-aligned with any teammate roster. Each
    returned ``Member``'s ``generator``/``member``/``role`` is the *teammate's own*
    identity (the one this ego is a dedicated best response for), while ``params``/
    ``policy_cls`` are the ego's own. Callers needing identity-keyed lookup (e.g. "the
    ego for teammate X") should build one via
    :func:`~oaht_bench.population.members.index_by_identity` rather than assume order.
    The artifact stores one member-axis param stack per source population (they carry
    different architectures in pooled mode); each manifest entry names its
    ``source_index`` and its ``pos`` on that source's member axis, and ``policy_cls`` is
    that source population's (the BR is warm-started from and shares its architecture).
    One ``ppo_br`` run can cover the whole released roster, so collection needs a single
    ``br_population_path`` entry.
    """
    import json

    from oaht_bench.common.save_load_utils import load_train_run
    from oaht_bench.configs import load_job

    br_dir = Path(br_dir)
    run_dir = br_dir.parent.parent if br_dir.name == "saved_train_run" else br_dir
    manifest = json.loads((run_dir / "br_manifest.json").read_text())
    by_source = load_train_run(str(artifact_dir(run_dir)))["br_by_source"]

    cls_cache: dict[str, Any] = {}

    def _policy_cls(src_path: str):
        if src_path not in cls_cache:
            sr = Path(src_path)
            sr = sr.parent.parent if sr.name == "saved_train_run" else sr
            sj = load_job(sr / "job.json")
            cls_cache[src_path] = population_from_run(
                sj, load_train_run(str(artifact_dir(sr))), env
            ).policy_cls
        return cls_cache[src_path]

    egos = []
    for e in manifest:
        params = jax.tree.map(lambda leaf, p=e["pos"]: leaf[p], by_source[str(e["source_index"])])
        egos.append(
            Member(
                e["generator"],
                int(e["member"]),
                e["role"],
                params,
                _policy_cls(e["source_population_path"]),
            )
        )
    return egos


def _checkpoint_update_steps(num_updates: int, num_checkpoints: int) -> np.ndarray:
    """The 1-indexed update steps ``make_br_train``'s ``_is_checkpoint_step`` stores a
    checkpoint on, in storage order (ascending) -- mirrors that JAX condition in plain
    numpy so a saved ``ego_return_curve`` can be read back without re-running training.
    """
    ckpt_interval = num_updates // max(1, num_checkpoints - 1)
    steps = np.arange(1, num_updates + 1)
    is_ckpt = ((steps - 1) % ckpt_interval == 0) | (steps == num_updates)
    return steps[is_ckpt]


def select_ratio_checkpoint(checkpoints, return_curve, *, ratio: float):
    """Pick the checkpoint whose return against the fixed teammate is closest to
    ``ratio`` of the fully-converged (final) return -- TIPR's Optimal Ratio ``ρ``.

    ``checkpoints`` is one member's ``(num_checkpoints,) + params_shape`` pytree (the
    leading axis in storage order, oldest first); ``return_curve`` is that same
    member's ``(num_updates,)`` per-update return curve, as saved alongside it. At
    ``ratio=1.0`` the target is the final return itself, which the last stored
    checkpoint always matches exactly by construction (its update step is always
    ``num_updates``) -- so this degenerates to selecting the last checkpoint, the same
    one :func:`load_ego_policies` returns.
    """
    return_curve = np.asarray(return_curve)
    num_updates = return_curve.shape[0]
    num_checkpoints = jax.tree_util.tree_leaves(checkpoints)[0].shape[0]
    steps = _checkpoint_update_steps(num_updates, num_checkpoints)
    checkpoint_returns = return_curve[steps - 1]
    target = ratio * return_curve[-1]
    diffs = np.abs(checkpoint_returns - target)
    # Ties broken toward the latest checkpoint (np.argmin keeps the first match
    # otherwise), so ratio=1.0 -- whose target is return_curve[-1] exactly, a value
    # every checkpoint can tie on verbatim when the curve is flat -- always resolves
    # to the last checkpoint, not whichever flat run happens to come first.
    idx = len(diffs) - 1 - int(np.argmin(diffs[::-1]))
    return jax.tree.map(lambda c: c[idx], checkpoints)


def load_ego_policies_at_ratio(br_dir: Path | str, env: TrainingEnv, *, ratio: float) -> list[Member]:
    """Like :func:`load_ego_policies`, but seat a ρ-matched intermediate checkpoint
    instead of the fully-converged best response -- TIPR's Optimal Ratio dataset
    construction, where each dataset is built from a single, uniformly undertrained ego
    rather than the converged BR. Requires the source ``ppo_br`` artifact to have been
    trained with ``num_checkpoints > 1`` (it must carry ``checkpoints_by_source`` and
    ``ego_return_curve_by_source``); raises clearly rather than silently falling back to
    the converged BR if it wasn't.
    """
    import json

    from oaht_bench.common.save_load_utils import load_train_run
    from oaht_bench.configs import load_job

    br_dir = Path(br_dir)
    run_dir = br_dir.parent.parent if br_dir.name == "saved_train_run" else br_dir
    manifest = json.loads((run_dir / "br_manifest.json").read_text())
    saved = load_train_run(str(artifact_dir(run_dir)))
    if "checkpoints_by_source" not in saved or "ego_return_curve_by_source" not in saved:
        raise ValueError(
            f"{run_dir} has no per-checkpoint artifact (checkpoints_by_source / "
            "ego_return_curve_by_source) -- it was trained with num_checkpoints=1, so "
            f"there is no intermediate checkpoint to select ratio={ratio} from. "
            "Retrain this ppo_br run with num_checkpoints > 1 to use load_ego_policies_at_ratio."
        )
    checkpoints_by_source = saved["checkpoints_by_source"]
    curve_by_source = saved["ego_return_curve_by_source"]

    cls_cache: dict[str, Any] = {}

    def _policy_cls(src_path: str):
        if src_path not in cls_cache:
            sr = Path(src_path)
            sr = sr.parent.parent if sr.name == "saved_train_run" else sr
            sj = load_job(sr / "job.json")
            cls_cache[src_path] = population_from_run(
                sj, load_train_run(str(artifact_dir(sr))), env
            ).policy_cls
        return cls_cache[src_path]

    egos = []
    for e in manifest:
        src_idx, pos = str(e["source_index"]), e["pos"]
        member_checkpoints = jax.tree.map(lambda leaf, p=pos: leaf[p], checkpoints_by_source[src_idx])
        member_curve = np.asarray(curve_by_source[src_idx])[pos]
        params = select_ratio_checkpoint(member_checkpoints, member_curve, ratio=ratio)
        egos.append(
            Member(
                e["generator"],
                int(e["member"]),
                e["role"],
                params,
                _policy_cls(e["source_population_path"]),
            )
        )
    return egos
