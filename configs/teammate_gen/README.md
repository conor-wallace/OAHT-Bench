# Teammate-generation configs

Twelve configs: four generators × the three Tier 1 environments (§10.3). Regenerate
with `uv run python scripts/gen_teammate_configs.py`, or `--all-envs` for all seven
results configurations.

```
uv run oaht-bench config=configs/teammate_gen/lbf_12x12/fcp.json
```

## Where the numbers come from

Hyperparameters are **ported from jax-aht's per-environment Hydra configs**, not
invented. That tuning is real and environment-specific — Hanabi wants
`gamma=0.999`, a much larger budget, and fewer update epochs; Overcooked wants a
larger entropy coefficient than LBF — and starting from library defaults instead
would have made the first runs uninformative.

Two values are ours rather than upstream's:

- **`lagrange_learning_rate`** is scaled by `(3/n)²` from upstream's `0.01`, which
  is tuned at `n = 3`. L-BRDiv's multipliers receive gradient from an unnormalized
  sum over ~n² pair terms, so an unscaled value overcorrects at larger populations
  — observed as entropy runaway to ~49 and `pg_loss` to −25 (§7.3). At the
  populations used here (`n = 3`) the factor is 1, so this only bites when the
  population is raised.
- **Overcooked reward shaping** is explicit in the environment config rather than
  left to a default. The environment defaults `do_reward_shaping=False`, but
  jax-aht's task configs enable it; a population trained without shaping solves a
  materially harder, sparse-reward task and is not comparable to one trained with.

## These are starting points, not the tuned configuration

§7.2 of the project plan makes the per-environment tuning record a contribution.
This is the baseline that record gets built against. Every value here is
provisional, and the ones most likely to move are the diversity weights
(`cross_play_weight`, `mixed_play_weight`, `tolerance_factor`) and the budgets.

## Known comparability problem: population sizes differ

The generators do not produce populations of the same size, because upstream tuned
each at a different `population_size` and because "population size" does not mean
the same thing across methods:

| generator | authored `population_size` (LBF) | resulting members |
|---|---|---|
| FCP | 5 | **25** — snapshots `num_checkpoints` times per run |
| CoMeDi | 10 | 10 |
| BRDiv | 3 | 3 confederate/best-response pairs |
| L-BRDiv | 3 | 3 pairs |

So a benchmark run over these configs compares *(generator, population size)*
pairs, not generators. That is a real threat to §1's head-to-head claim and it has
to be resolved deliberately — either by fixing a target member count and deriving
each generator's `population_size` from it (FCP's would become
`target // num_checkpoints`), or by reporting population size as an axis and
sweeping it.

Fixing the target count is not free: §7.3 documents that raising `n` dilutes
per-policy self-play data quadratically, so budgets need scaling with it, and the
BRDiv collapse at `n = 5` was a sample-count problem rather than a diversity-weight
one. Retuning at a common `n` is exactly the work §7.2 describes.

**This is an open decision, recorded here rather than silently resolved.** The
configs as written reproduce upstream's tuning; they do not yet constitute a fair
comparison between generators.
