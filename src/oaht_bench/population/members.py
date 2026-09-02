"""Which members a population releases, and how to read one out of it.

Both questions are conventions of the *artifact*, not of training: a member
index means the same thing to the code that scores a population and to the code
that seats it in a dataset rollout. One definition is what stops those two from
disagreeing about which policy an index refers to.
"""

from __future__ import annotations

import jax


def get_member_params(params, index: int, *, seed_index: int = 0):
    """Slice one member's parameters out of a stacked population tree.

    Generators return parameters with leading axes ``(num_seeds, pop_size, ...)``
    — see :func:`~oaht_bench.population.loading.population_from_run` for how each
    one arrives at that shape. This is the one place that knows the layout, so
    scoring and dataset collection cannot drift apart on what "member i" means.

    Args:
        params: Stacked parameters for a whole population.
        index: Flat member index. For FCP this is ``run * num_checkpoints +
            checkpoint``; see :func:`released_members`.
        seed_index: Which training seed's population to read.
    """
    return jax.tree.map(lambda leaf: leaf[seed_index][index], params)


def released_members(job, pop_size: int) -> list[int]:
    """Which members a generator actually releases as its population.

    Everything a population is *used* for — scoring it, seating it in a dataset
    — should draw from these, so the two cannot disagree about what the
    population is.

    FCP is the exception, and getting it wrong inverts the tuning signal. Its
    population deliberately spans *competence*: ``ippo`` stores checkpoints at
    ``num_updates // (num_checkpoints - 1)`` intervals from step 1 onward, so
    members range from barely-trained to converged. Averaging self-play across
    all of them penalises exactly what makes the method work — and the paper's
    own ``FCP-T`` ablation, which keeps only converged checkpoints, is
    *significantly worse* downstream. Ranking a sweep on that mean would push
    ``num_checkpoints`` toward 1 and reproduce the ablation.

    So FCP releases the converged checkpoint of each independent run: one member
    per training run, which is the convention that run arrived at. The competence
    spread stays in the stored population and is not a defect to optimize away.
    The other three release one member per convention already.

    Args:
        job: The teammate-generation job that produced the population.
        pop_size: Members in the loaded population, used for the non-FCP case.

    Returns:
        Flat member indices. Always a list — an earlier version returned ``None``
        to mean "all", which made every caller responsible for remembering the
        sentinel.
    """
    gen = job.generator
    if gen.generator != "fcp":
        return list(range(pop_size))
    # get_fcp_population reshapes (seeds, runs, ckpts, ...) -> (seeds, runs*ckpts, ...)
    # in C order, so flat index == run * num_checkpoints + checkpoint.
    ckpts = gen.num_checkpoints
    return [run * ckpts + (ckpts - 1) for run in range(gen.population_size)]
