"""How many gradient updates a teammate-generation job will actually perform.

The number is not ``total_timesteps // rollout_length // num_envs`` for three of
the four generators, and the differences are large enough to matter when sizing a
cluster job:

* **FCP** trains ``population_size`` members, each for ``num_updates``. They are
  ``vmap``\\ ed, so the *sequential* depth is one member's worth even though the
  total work is the product.
* **CoMeDi** builds its population one member at a time. ``num_updates`` is per
  *outer iteration*, and there are ``population_size - 1`` of them — the first
  member comes from a self-play warmup, which has its own budget and a different
  formula.
* **BRDiv / L-BRDiv** train all confederate/best-response pairs jointly, so
  ``num_updates`` really is the total.

``sequential_updates`` is how many streamed points to expect, so it is the number
to compare a progress log against. ``train_step`` is 0-indexed, so its highest
value is one less.
"""

from __future__ import annotations

from dataclasses import dataclass

from oaht_bench.configs.job import TeammateGenerationJob
from oaht_bench.teammate_gen.runtime import (
    CoMeDiRuntime,
    PairedDiversityRuntime,
    PpoRuntime,
    RpgRuntime,
)


@dataclass(frozen=True)
class TrainingPlan:
    """Update accounting for one job."""

    generator: str
    #: Updates in one unit of sequential work (one FCP member, one CoMeDi outer
    #: iteration, or the whole BRDiv/L-BRDiv run).
    updates_per_unit: int
    #: How many such units run one after another.
    sequential_units: int
    #: Members trained in parallel under a ``vmap`` within a unit.
    parallel_members: int
    #: Extra updates before the main loop (CoMeDi's self-play warmup).
    warmup_updates: int = 0

    @property
    def sequential_updates(self) -> int:
        """Depth of the training loop: how many streamed points to expect.

        ``train_step`` is 0-indexed, so the highest value logged is one less than
        this.
        """
        return self.warmup_updates + self.updates_per_unit * self.sequential_units

    @property
    def total_updates(self) -> int:
        """Gradient updates across all members, parallel work included."""
        return self.sequential_updates * self.parallel_members

    def describe(self) -> str:
        lines = [
            f"generator            {self.generator}",
            f"updates per unit     {self.updates_per_unit:,}",
            f"sequential units     {self.sequential_units:,}",
        ]
        if self.warmup_updates:
            lines.append(f"warmup updates       {self.warmup_updates:,}")
        if self.parallel_members > 1:
            lines.append(f"parallel members     {self.parallel_members:,}  (vmapped)")
        lines += [
            f"sequential updates   {self.sequential_updates:,}   <- streamed points "
            f"(train_step 0..{self.sequential_updates - 1:,})",
            f"total updates        {self.total_updates:,}",
        ]
        return "\n".join(lines)


def training_plan(job: TeammateGenerationJob, *, num_agents: int = 2) -> TrainingPlan:
    """Compute the update accounting without running anything.

    Builds the same runtime the trainer does, so this also surfaces a budget that
    would train nothing — the runtime rejects it — before a job is queued.
    """
    gen = job.generator
    rollout = job.env.rollout_length

    if gen.generator == "fcp":
        rt = PpoRuntime.from_config(
            ppo=gen.ppo,
            network=gen.network,
            actor_type=gen.actor_type,
            rollout_length=rollout,
            num_envs=gen.num_envs,
            total_timesteps=gen.total_timesteps,
            num_checkpoints=gen.num_checkpoints,
            num_agents=num_agents,
        )
        return TrainingPlan(
            generator="fcp",
            updates_per_unit=rt.num_updates,
            sequential_units=1,
            parallel_members=gen.population_size,
        )

    if gen.generator == "comedi":
        rt = CoMeDiRuntime.from_config(gen, rollout_length=rollout, num_agents=num_agents)
        return TrainingPlan(
            generator="comedi",
            updates_per_unit=rt.num_updates,
            # The scan runs over arange(1, population_size).
            sequential_units=max(0, gen.population_size - 1),
            parallel_members=1,
            warmup_updates=rt.warmup().num_updates,
        )

    if gen.generator == "rpg":
        rt = RpgRuntime.from_config(gen, rollout_length=rollout, num_agents=num_agents)
        # One outer step = n_lookahead base updates + one manipulator update, run
        # sequentially over num_updates steps. The N particles (base+manipulator
        # pairs) are vmapped, so they are parallel members rather than sequential.
        return TrainingPlan(
            generator="rpg",
            updates_per_unit=rt.n_lookahead + 1,
            sequential_units=rt.num_updates,
            parallel_members=rt.population_size,
        )

    rt = PairedDiversityRuntime.from_config(gen, rollout_length=rollout, num_agents=num_agents)
    return TrainingPlan(
        generator=gen.generator,
        updates_per_unit=rt.num_updates,
        sequential_units=1,
        parallel_members=1,
    )
