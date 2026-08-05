"""Cross-play evaluation of a trained teammate population.

Every generator produces a population, but only some of them measure cross-play,
and the three that do measure it *differently* and *during* training:

* FCP has no notion of cross-play at all — it trains independent self-play runs
  and never pairs members with each other.
* CoMeDi reports self- and cross-play from its own training rollouts.
* BRDiv and L-BRDiv report a confederate against its *best response*.

That makes the numbers incomparable across generators, and leaves FCP with
nothing. This module evaluates the released population against itself, after
training, in one way for all four: pair member ``i`` with member ``j`` for every
``(i, j)`` and record the mean episode return.

The diagonal is a member paired with a copy of itself and the off-diagonal is
mismatched members, so:

* **SP** (diagonal mean) — competence. A population whose members cannot score
  when paired with themselves is not useful as training data regardless of how
  distinct they are.
* **XP** (off-diagonal mean) — overlap. Low means members require different
  responses.
* **separation** ``SP - XP`` — diversity that is not bought by incompetence.

.. warning::

   The diagonal means slightly different things per generator, and this is worth
   stating rather than smoothing over. FCP and CoMeDi members are trained *by*
   self-play, so ``i`` vs ``i`` is their designed pairing. BRDiv and L-BRDiv
   release only the confederate set, whose designed partner is a separately
   trained best response, not another copy of itself — so their diagonal here
   under-reports the competence their own ``Eval/AvgSPReturnCurve`` measures.
   Use this matrix to compare *within* a generator across hyperparameters, which
   is what tuning needs; treat cross-generator comparison of the absolute
   diagonal with care.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np

from oaht_bench.common.run_episodes import run_episodes


@dataclass(frozen=True)
class CrossPlayScores:
    """Summary of a population's self-evaluation."""

    matrix: np.ndarray
    #: Mean of the diagonal — members paired with themselves.
    self_play: float
    #: Mean of the off-diagonal — members paired with each other.
    cross_play: float

    @property
    def separation(self) -> float:
        """``self_play - cross_play``. Diversity, net of competence."""
        return self.self_play - self.cross_play

    def describe(self) -> str:
        return (
            f"self-play    {self.self_play:.4f}\n"
            f"cross-play   {self.cross_play:.4f}\n"
            f"separation   {self.separation:.4f}"
        )


def evaluate_population(
    env,
    params,
    population,
    *,
    rng: jax.Array,
    max_episode_steps: int,
    num_episodes: int = 20,
    seed_index: int = 0,
) -> CrossPlayScores:
    """Pair every population member with every other and score the result.

    Args:
        params: Stacked population parameters, leading axes
            ``(num_seeds, pop_size, ...)``.
        population: The :class:`AgentPopulation` describing how to read them.
        num_episodes: Episodes per ordered pair. The estimate is noisy at small
            values and the cost is ``pop_size**2 * num_episodes`` episodes, so
            this is the dial for the accuracy/time trade-off.
        seed_index: Which training seed's population to evaluate.

    Returns:
        Scores whose ``matrix[i, j]`` is the mean return of member ``i`` in seat 0
        with member ``j`` in seat 1.
    """
    pop_size = population.pop_size
    policy = population.policy_cls

    def member(idx: int):
        return jax.tree.map(lambda leaf: leaf[seed_index][idx], params)

    matrix = np.zeros((pop_size, pop_size), dtype=float)
    for i in range(pop_size):
        for j in range(pop_size):
            rng, pair_rng = jax.random.split(rng)
            out = run_episodes(
                pair_rng, env,
                agent_0_param=member(i), agent_0_policy=policy,
                agent_1_param=member(j), agent_1_policy=policy,
                max_episode_steps=max_episode_steps,
                num_eps=num_episodes,
                agent_0_test_mode=True, agent_1_test_mode=True,
            )
            returns = np.asarray(out["returned_episode_returns"])
            matrix[i, j] = float(returns.mean())

    diag = float(np.mean(np.diag(matrix)))
    if pop_size > 1:
        off = float((matrix.sum() - np.trace(matrix)) / (matrix.size - pop_size))
    else:
        # A single-member population has no cross-play to measure; reporting 0
        # would read as perfect separation.
        off = float("nan")
    return CrossPlayScores(matrix=matrix, self_play=diag, cross_play=off)


def write_scores(scores: CrossPlayScores, run_dir: Path) -> Path:
    """Write the matrix as CSV beside the run's other artifacts."""
    path = Path(run_dir) / "population_crossplay.csv"
    np.savetxt(path, scores.matrix, delimiter=",")
    return path
