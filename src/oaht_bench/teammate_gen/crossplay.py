"""Cross-play evaluation of a trained teammate population.

Every generator produces a population, but only some of them measure cross-play,
and the three that do measure it *differently* and *during* training:

* FCP has no notion of cross-play at all — it trains independent self-play runs
  and never pairs members with each other.
* CoMeDi reports self- and cross-play from its own training rollouts.
* BRDiv and L-BRDiv report a confederate against its *best response*.

That makes the numbers incomparable across generators, and leaves FCP with
nothing. This module scores every trained population after training, one way for
all four: pair member ``i`` with member ``j`` for every ``(i, j)`` and record the
mean episode return.

* **SP** (diagonal mean) — competence with the *designed* partner. A population
  whose members cannot score with the partner they were trained for is not
  useful as training data regardless of how distinct they are.
* **XP** (off-diagonal mean) — overlap. Low means members require different
  responses.
* **separation** ``SP - XP`` — diversity that is not bought by incompetence.

The **diagonal is each generator's own designed pairing**, which differs by
population structure:

* *Homogeneous* populations — FCP's checkpoints and CoMeDi's ``final_params_conf``
  are sets of self-play-trained policies, so the matrix is ``pop x pop`` and the
  diagonal is member ``i`` with a copy of itself.
* *Paired* populations — BRDiv and L-BRDiv save ``final_params_conf`` **and**
  ``final_params_br``, and their designed-optimal pairing is confederate ``i``
  with best response ``i``. The matrix is ``conf x br`` and the diagonal is that
  pairing, matching what their own ``Eval/AvgSPReturnCurve`` reports.

Pairing a confederate with a *copy of itself* would be an out-of-distribution
pairing that under-reports competence, since confederates are never trained to
play with themselves. Using each generator's intended pairing is what makes the
self-play column mean one thing across all four.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    partner_params=None,
    member_indices: Sequence[int] | None = None,
    greedy: bool = False,
) -> CrossPlayScores:
    """Pair every row member with every column member and score the result.

    Args:
        params: Stacked row-population parameters, leading axes
            ``(num_seeds, pop_size, ...)``. For a paired generator these are the
            confederates.
        population: The :class:`AgentPopulation` describing how to read them.
        partner_params: Column population, when it differs from the row one —
            BRDiv and L-BRDiv pass their ``final_params_br`` here so the diagonal
            is confederate ``i`` with *its own* best response. Omit for the
            homogeneous generators, where the column population is the row one
            and the diagonal is genuine self-play.
        num_episodes: Episodes per ordered pair. Cost is
            ``pop_size**2 * num_episodes`` episodes, so this is the dial for the
            accuracy/time trade-off.
        seed_index: Which training seed's population to evaluate.
        member_indices: Score only these members. FCP passes its *converged*
            checkpoints here; see :func:`scored_members` for why. ``None`` scores
            the whole population.
        greedy: Take argmax actions rather than sampling. Off by default, and it
            should stay off for anything reported. Argmax makes two members of a
            symmetric population perfectly correlated, and in a coordination task
            that is a deadlock rather than a strong pairing -- on LBF 12x12 it
            held every episode to the time limit at 25% of the food collected,
            scoring 0.11 where sampling scored 0.37 and training read 0.40. It
            also erases the policy entropy, so a sweep over ``entropy_coef``
            cannot see what it is tuning.

    Returns:
        Scores whose ``matrix[i, j]`` is the mean return with row member ``i`` in
        seat 0 and column member ``j`` in seat 1.
    """
    policy = population.policy_cls
    cols = params if partner_params is None else partner_params
    idx = list(range(population.pop_size)) if member_indices is None else list(member_indices)
    pop_size = len(idx)

    def member(source, i: int):
        return jax.tree.map(lambda leaf: leaf[seed_index][i], source)

    matrix = np.zeros((pop_size, pop_size), dtype=float)
    for a, i in enumerate(idx):
        for b, j in enumerate(idx):
            rng, pair_rng = jax.random.split(rng)
            out = run_episodes(
                pair_rng,
                env,
                agent_0_param=member(params, i),
                agent_0_policy=policy,
                agent_1_param=member(cols, j),
                agent_1_policy=policy,
                max_episode_steps=max_episode_steps,
                num_eps=num_episodes,
                agent_0_test_mode=greedy,
                agent_1_test_mode=greedy,
            )
            returns = np.asarray(out["returned_episode_returns"])
            matrix[a, b] = float(returns.mean())

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


def scored_members(job) -> list[int] | None:
    """Which population members the SP/XP scores should be computed over.

    FCP is the exception, and getting it wrong inverts the tuning signal. Its
    population deliberately spans *competence*: ``ippo`` stores checkpoints at
    ``num_updates // (num_checkpoints - 1)`` intervals from step 1 onward, so
    members range from barely-trained to converged. Averaging self-play across
    all of them penalises exactly what makes the method work — and the paper's
    own ``FCP-T`` ablation, which keeps only converged checkpoints, is
    *significantly worse* downstream. Ranking a sweep on that mean would push
    ``num_checkpoints`` toward 1 and reproduce the ablation.

    So FCP is scored on the converged checkpoint of each independent run: one
    member per training seed, which is the "convention" that run arrived at. The
    competence spread is retained in the population and is not a defect to
    optimize away.

    The other three release one member per convention already, so all of their
    members are scored.

    Returns:
        Flat member indices, or ``None`` to score everything.
    """
    gen = job.generator
    if gen.generator != "fcp":
        return None
    # get_fcp_population reshapes (seeds, runs, ckpts, ...) -> (seeds, runs*ckpts, ...)
    # in C order, so flat index == run * num_checkpoints + checkpoint.
    ckpts = gen.num_checkpoints
    return [run * ckpts + (ckpts - 1) for run in range(gen.population_size)]
