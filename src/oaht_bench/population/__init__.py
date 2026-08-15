"""A trained teammate population as an artifact: loading, indexing, scoring.

Deliberately independent of :mod:`oaht_bench.teammate_gen`, which houses only
the training algorithms that *produce* populations. Both dataset collection and
post-training evaluation consume populations, so the code they share lives here
rather than inside the trainer — otherwise ``data`` would depend on
``teammate_gen`` to read an artifact it does not train.

Dependencies run one way::

    population   -> agents, configs, common
    teammate_gen -> population      (a run scores the population it produced)
    data         -> population      (collection seats members)
"""

from oaht_bench.population.crossplay import (
    CrossPlayScores,
    evaluate_population,
    write_scores,
)
from oaht_bench.population.loading import (
    artifact_dir,
    population_from_run,
)
from oaht_bench.population.members import get_member_params, released_members
from oaht_bench.population.rescore import rescore_run

__all__ = [
    "CrossPlayScores",
    "artifact_dir",
    "evaluate_population",
    "get_member_params",
    "population_from_run",
    "rescore_run",
    "released_members",
    "write_scores",
]
