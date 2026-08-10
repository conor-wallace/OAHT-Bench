# Per-environment tuning record

§7.2 of the project plan makes this a contribution rather than an appendix. The
teammate-generation literature reports results at hyperparameters tuned on one
environment and rarely says what happened to the others; a benchmark that
inherits those numbers unexamined inherits their blind spots too.

Each entry records what was swept, what the sweep concluded, and — where it
matters more — what the sweep could *not* conclude.

Reproduce any entry with `scripts/sweep.py generate` using the grid quoted, then
`scripts/sweep.py collect`. Run directories are named by config hash, so a cell
in a table here resolves to exactly one directory.

---

## FCP × LBF 12×12

**Adopted:** `learning_rate=1e-3`, `entropy_coef=0.003`, `total_timesteps=24e6`,
`num_envs=64`. Everything else at the generator defaults.

**Upstream was:** `learning_rate=1e-4`, `entropy_coef=0.01`,
`total_timesteps=1e6`, `num_envs=8`.

### Grid

18 cells, one seed each:

```
generator.ppo.learning_rate  = 1e-4, 3e-4, 1e-3
generator.ppo.entropy_coef   = 0.003, 0.01, 0.03
generator.total_timesteps    = 8e6, 24e6        (num_envs = 64 throughout)
```

Scored by post-training cross-play (`teammate_gen/crossplay.py`) on the
converged checkpoint of each of the 5 runs: self-play is the diagonal mean,
cross-play the off-diagonal mean, separation their difference.

### What the sweep found

**Budget dominates competence; nothing else comes close.** Main effect on
self-play, averaged over the other two factors:

| factor | span of self-play |
|---|---:|
| `total_timesteps` | **0.078** |
| `entropy_coef` | 0.032 |
| `learning_rate` | 0.010 |

**And the budget is now spent.** At `24e6` the population collects **96.8–97.4%
of the food**, and self-play is flat across every other setting — spans of 0.010
for `entropy_coef` and 0.026 for `learning_rate`. Every configuration converges
to self-play ≈ 0.47–0.49 given enough steps. At `8e6` the same populations reach
only ~74% of the food and *every* cell fell below the competence floor. This is
the ceiling of the task, not of the method.

**Batch size mattered independently of update count.** Two cells identical in
every hyperparameter, both at 976 updates, differing only in `num_envs` (8 vs
64) with `total_timesteps` scaled 8× to hold updates fixed:

| num_envs | batch | updates | self-play |
|---:|---:|---:|---:|
| 8 | 2,048 | 976 | 0.272 |
| 64 | 16,384 | 976 | **0.425** |

Raising `num_envs` *without* scaling the budget does the opposite, and severely:
`num_envs=128` at `1e6` is 61 updates and scores 0.073. `num_envs` is not a
throughput knob — it trades gradient steps for batch size, and only pays when
the budget moves with it.

**Once competence saturates, `entropy_coef` drives separation.** Within `24e6`:

| | span of separation |
|---|---:|
| `entropy_coef` | **0.097** |
| `learning_rate` | 0.079 |

Lower entropy gives more separation, monotonically at every learning rate
(0.161 → 0.123 → 0.065 for `ec` = 0.003 → 0.01 → 0.03).

### What the sweep could not conclude

**The learning-rate effect is not credible at one seed.** Its effect on
separation is non-monotonic — 0.116 at `1e-4`, 0.060 at `3e-4`, 0.122 at `1e-3`.
A U-shape from single runs is more likely noise than structure. `1e-3` is
adopted because it is jointly best on both metrics, not because the ordering is
established. Do not read a `learning_rate` ranking off this table.

**The optimum is at the grid corner on both open axes.** The adopted cell has
the highest learning rate and the lowest entropy coefficient tested, so the
grid does not bracket it. Separation was still improving in both directions when
the sweep ran out of grid.

**Separation was still improving with budget, too.** Its main effect is small
(0.040) but interacts: at the adopted `lr`/`ec`, going `8e6 → 24e6` moved
separation 0.122 → 0.226, the largest single jump in the sweep. With two budget
points there is no way to tell whether that saturates at `24e6`.

**Separation is an unvalidated proxy.** The objective is downstream ego-agent
generalization, which cannot be measured until `ppo_br.py` is absorbed. Driving
self-play to the task ceiling is well founded and is what the adopted settings
do. Optimizing separation beyond that is a bet on the proxy — and a weaker bet
for FCP than for the others, since FCP's diversity is meant to come from
checkpoints spanning competence, not from an entropy knob it never claimed.

**Follow-up, not yet run.** Deliberately deferred rather than abandoned:

```
generator.ppo.learning_rate = 1e-3
generator.ppo.entropy_coef  = 0.001, 0.003
generator.total_timesteps   = 24e6, 48e6
seed                        = 0, 1, 2
```

Four configurations, three seeds each — enough to put error bars on the
separation gaps and to test whether either trend continues past the grid edge.

### Method notes that came out of this

Two measurement bugs were found while running this sweep, both of which would
have invalidated it:

- Evaluation ran **argmax** actions. Two argmax policies in a symmetric
  coordination task are perfectly correlated and deadlock: every episode ran to
  the 100-step limit at 25% of the food, reporting self-play 0.11 for a
  population whose training curve read 0.41. It also erases the policy entropy,
  which would have made `entropy_coef` — an axis of this very sweep —
  unmeasurable. Fixed in `690b209`; evaluation now samples, and
  `evaluation_greedy` records the choice.
- FCP was scored across **all** population members. Its checkpoints span
  competence by design, so the mean penalised the mechanism and would have
  driven `num_checkpoints → 1`. Fixed in `b4359cf`; scoring uses the converged
  checkpoint of each run.

---

## Not yet tuned

CoMeDi, BRDiv and L-BRDiv on LBF, and all four on Overcooked and Hanabi, still
run at hyperparameters ported from jax-aht's per-environment Hydra configs.
Those encode real tuning and are a reasonable starting point, but the FCP result
above shows what inheriting them can cost: upstream's LBF budget left the
population at 74% of the achievable food.
