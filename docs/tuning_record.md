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

## Carrying the FCP result across to the other three (LBF 12×12)

No sweep was run for these. What transferred is the *diagnostic* — measure the
population against the task ceiling (~97% of the food) and check whether the
return curve is still climbing — not the tuned numbers. Applying FCP's budget
uniformly would have been wrong for two of the three.

Baseline, from one run of each at the inherited settings:

| generator | updates | % food | final return | last-quarter slope /1k | verdict |
|---|---:|---:|---:|---:|---|
| FCP *(tuned)* | 2,929 | **96.8** | 0.483 | — | at ceiling |
| CoMeDi | 2,416 | 80.2 | 0.393 | **+0.232** | starved |
| BRDiv | 5,493 | 40.6 | 0.182 | +0.002 | converged |
| L-BRDiv | 5,493 | 21.9 | 0.096 | ~0 | converged |

**CoMeDi — budget raised.** `6e6 → 2.4e7`, `num_envs 48 → 64`. Its
`total_timesteps_per_iteration` is spent *per member*, and at the old setting
each of the 9 members got **160 updates** against FCP's 2,929. That single fact
explains the 80% plateau and the steeply-climbing curve. The new setting gives
526/member, which is a step and not a fix: parity would need ~1.2e8 and ~38k
sequential updates, and CoMeDi builds its population one member at a time, so
that is a GPU-scale run rather than a laptop one.

**BRDiv — unchanged.** It already receives 5,493 updates, nearly double tuned
FCP, and its curve is flat over the final quarter (+0.002/1k). It is converged
at 40% of the food, not starved. More budget buys time and nothing else. The
open knob is `cross_play_weight`, which sets how much competence the diversity
term is allowed to trade away — a grid for it already exists at
`configs/sweeps/brdiv_lbf_xpw` and has never been run.

**L-BRDiv — unchanged, and the least understood.** Flat as well, so also not
starved, but it reaches only 22% of the food *while producing the best
separation of any generator measured here* (SP 0.212, XP 0.019, separation
0.193). Low absolute competence may be what the Minimum Coverage Set objective
is buying rather than a defect, so it is not yet clear what "fixing" it would
even mean. `tolerance_factor` is the knob; it needs a sweep and a downstream
check before anyone concludes the number is too low.

The general lesson, worth restating because it nearly went the other way here:
**a low return does not imply a small budget.** Three generators all well short
of the ceiling had three different causes, separable in about a minute by
looking at the slope of the last quarter of training.

---

## Scaling law: BRDiv and L-BRDiv need `num_envs` ∝ n²

Found by raising `population_size` from 3 to 5 for comparability and watching
BRDiv collapse.

Both paired generators draw `conf_id` and `br_id` **independently per
environment**, so a specific `(conf_i, br_j)` pairing receives `num_envs / n²`
samples per rollout — not `num_envs / n`.

| n | num_envs | pairings | envs per pairing | outcome |
|---:|---:|---:|---:|---|
| 3 | 64 | 9 | 7.1 | SP 0.237, separation 0.080 |
| 5 | 64 | 25 | **2.6** | SP 0.067, separation **−0.006** |
| 5 | 192 | 25 | 7.7 | adopted |

At 2.6 environments per pairing no best response could specialize against its
confederate. The final cross-play matrix was uniform to within noise (every
entry between 0.047 and 0.096) and **self-play fell below cross-play**, which
inverts the quantity BRDiv maximizes.

**The loss weighting is genuinely population-size invariant, which is what made
this easy to miss.** `BRDiv.py` reweights by `sp_weight = (1 + 2α)(n/2)` and
`xp_weight = α·n/(2(n−1))` against sampling probabilities `P(SP) = 1/n`,
`P(XP) = (n−1)/n`, giving `E[SP weight] = 0.55` and `E[XP weight] = 0.025` at
every n. That invariance is real and was verified. It is simply not the binding
quantity — the *data* behind each pairing is.

`total_timesteps` scales with `num_envs` too, holding the update count at 5,493.
Without that, tripling the environments would cut gradient steps to a third,
trading this failure for the one the FCP entry documents.

Encoded as `_paired_scale()` in `gen_teammate_configs.py` so it is derived from
`POPULATION_SIZE` rather than restated, and pinned by a test asserting envs per
pairing never drops below the n=3 reference.

**Cost:** 3× on both paired generators. LBF goes to `num_envs=192`,
`total_timesteps=1.35e8` — roughly 50 hr each on an M1 CPU.

**Not yet confirmed.** The mechanism is clear and the evidence fits, but the
hypothesis is only verified once a BRDiv run at `num_envs=192` shows structure
returning to the cross-play matrix. Check `Eval/AvgSPReturnCurve` exceeds
`Eval/AvgXPReturnCurve` before trusting the population.

---

## Not yet tuned

All four on Overcooked and Hanabi still run at hyperparameters ported from
jax-aht's per-environment Hydra configs. Those encode real tuning and are a
reasonable starting point, but the FCP result above shows what inheriting them
can cost: upstream's LBF budget left the population at 74% of the achievable
food, and CoMeDi's left each member with 160 updates.

## AD-RPG × LBF 12×12 (untuned; a falsification, not a tuned baseline)

AD-RPG is a clean-room reimplementation of the paper's `doublesided_RAD`
(`src/oaht_bench/teammate_gen/RPG.py`, see `PROVENANCE.md`). Its LBF config is a
**deliberately modest, untuned starting point** (`total_timesteps=1e7`,
`num_envs=64`, `pop=5`, defaults for `partnerplay_ratio`, `off_diag_factor`,
`dice_lambda`, `n_lookahead`). It exists to answer one question first: **does an
algorithm sold as general-purpose produce a diverse, non-sabotaging LBF
population, on the same environment the other four are tuned on?**

Two things this config cannot yet tell us, both open adoption gates:

- **Scale.** The paper only demonstrates `NUM_PARTICLES=2`. Cost here grows ~`n²`
  (each outer update collects `n` self-play + `n²` cross-play rollouts and runs an
  inner `n_lookahead` per particle), so `pop=5` is far heavier than any other
  generator and may not hold coverage — or may destabilise. Whether SP−XP
  separation survives past `n=2` is the first thing a GPU sweep must establish.
- **Held-out usability.** The paper evaluates its population by in-population
  cross-play, never as a held-out training population for a separately-trained
  ego. Our runner scores it exactly like the other self-play releases, but whether
  that population is *useful* as AHT teammates is unmeasured until `ppo_br.py`.

The correctness of the algorithm itself (the DiCE surrogate and the higher-order
manipulator meta-gradient) is pinned by `tests/test_rpg.py` on CPU-sized inputs;
what those tests do **not** establish is that training converges to a good LBF
population, which only a real run can show.
