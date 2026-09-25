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

## CoMeDi × LBF 12×12 — budget follow-up

Continues the "budget raised" line from the table above. `total_timesteps_per_iteration=2.4e7`
fixed the 80%-food plateau but not fully — the sweep below asks how much further
budget buys, and stops short of a confirmed answer.

**Best so far, not yet adopted:** `learning_rate=5e-4` (unchanged from upstream),
`total_timesteps_per_iteration=9.6e7`, `num_envs=64` unchanged. Not written into
`gen_teammate_configs.py` because the training curve is still climbing at this
setting — see below.

**Upstream was:** `learning_rate=5e-4`, `entropy_coef=0.001`,
`total_timesteps_per_iteration=6e6`, `num_envs=48` — 160 updates/member, the
starved baseline the table above already diagnosed.

### Grid

Two sweeps, one seed each:

```
comedi_lbf_competence:
  generator.ppo.learning_rate               = 5e-4, 1e-3
  generator.total_timesteps_per_iteration    = 2.4e7, 4.8e7

comedi_lbf_budget2:
  generator.ppo.learning_rate               = 5e-4
  generator.total_timesteps_per_iteration    = 2.4e7, 4.8e7, 9.6e7
```

Scored by post-training cross-play at the default `evaluation_episodes=20`; the
`48e6` and `96e6` cells were re-scored at `--episodes 100` (via `sweep.py
rescore`, no retraining) once the 20-episode numbers turned out not to be
trustworthy at this resolution — see Method notes.

### What the sweep found

**Budget still dominates, and is not yet spent.** At 100-episode fidelity:

| `total_timesteps_per_iteration` | SP | XP | SP−XP |
|---:|---:|---:|---:|
| 2.4e7 | 0.395 *(20 ep)* | 0.207 *(20 ep)* | 0.188 |
| 4.8e7 | 0.453 | 0.208 | 0.245 |
| 9.6e7 | **0.465** | 0.193 | **0.272** |

Each doubling still gains SP; the gain from `4.8e7 → 9.6e7` (+0.011) is real but
roughly half the gain from `2.4e7 → 4.8e7`, consistent with the training curve:
last-quarter slope (same diagnostic as the table above) falls from **+0.038/1k**
at `4.8e7` to **+0.020/1k** at `9.6e7`, and % food climbs 84% → 93%, still short
of FCP's ~97% ceiling.

**`learning_rate=5e-4` beat `1e-3` at both budgets tested**, direction-consistent
though single-seed: SP 0.393 vs 0.362 at `2.4e7`, 0.455 vs 0.441 at `4.8e7`. This
also happens to be the upstream value — no change adopted on this axis.

### What the sweep could not conclude

**Not converged.** `9.6e7`'s last-quarter slope (+0.020/1k) is still ~10x
BRDiv's converged rate (+0.002/1k, see below). Whether `1.92e8` flattens it
further is unrun and would roughly double wall-clock again.

**The learning-rate ordering is not established**, for the same single-seed
reason the FCP entry flags for its own `learning_rate` axis.

**Separation is the same unvalidated proxy** noted in the FCP entry — the
`9.6e7` gain here is on both SP and SP−XP together, which is the more
defensible direction, but downstream validation is still gated on `ppo_br.py`.

### Method notes that came out of this

- **`sweep.py rescore` was broken for all four generators, not just CoMeDi.**
  `population/rescore.py` called `.params` / `.partner_params` / `.pop_size` on
  the return of `population_from_run`, which is a plain `(params, population)`
  tuple for every generator — confirmed against all four `get_*_population`
  builders and against how `runner.py`'s live-evaluation path unpacks the same
  call. No test exercised the rescore CLI path. Fixed by unpacking the tuple and
  reading `partner_params` from the already-loaded checkpoint dict directly.
- **`label` is folded into the content hash**, so renaming a sweep
  (`comedi_lbf_competence → comedi_lbf_budget2`) gave the `2.4e7`/`4.8e7`,
  `lr=5e-4` cells a new hash and silently retrained them instead of reusing the
  already-trained artifacts — the same category of gap already noted under
  "Known-open" for `LoggingConfig`/`evaluation_episodes`, just via `label`
  instead. Keep `--name` stable when extending an existing grid rather than
  starting a new one, or expect to repay the compute.
- **That accidental duplicate quantified an eval-noise floor.** Two
  independently-trained checkpoints at identical hyperparameters
  (`train_seed=20374` fixed, not varied by either sweep) scored SP 0.4550 vs
  0.4433 at the sweep default of 20 episodes/pair — a 0.0117 gap that looks like
  real signal in a `collect` table. Re-scoring both at 100 episodes/pair shrank
  it to 0.0050 (0.4484 vs 0.4534): most of the original gap was evaluation
  sampling noise, not training-time nondeterminism. **`collect`'s numbers at the
  20-episode default should not be trusted below roughly this resolution** —
  re-score any finalist at higher `--episodes` before reading a ranking off it.

### Budget chase, concluded: `1.92e8`

One more doubling, mirroring BRDiv's own budget chase: hold `learning_rate=5e-4`
fixed (confirmed, not re-tested), double `total_timesteps_per_iteration` again.

| `total_timesteps_per_iteration` | SP | XP | SP−XP | slope /1k |
|---:|---:|---:|---:|---:|
| 9.6e7 | 0.465 | 0.193 | 0.272 | +0.0204 |
| 1.92e8 | **0.472** | 0.256 | 0.217 | **+0.0046** |

(SP/XP at matched 100-episode rescoring fidelity, per the eval-noise finding
above.) Slope fell ~4.4x, into the same range BRDiv's and L-BRDiv's converged
runs landed in (+0.001–0.004/1k) — this is CoMeDi's converged budget.
**Adopted, pushed into `gen_teammate_configs.py`.**

**Unlike every other budget doubling in this file, separation fell rather than
rose or held.** SP's move (+0.008) is within the measured noise floor, but XP
rose meaningfully (0.193→0.256), so separation dropped by 0.055. Read: once
competence approaches the task ceiling, CoMeDi's members apparently converge
toward similar competent behavior rather than staying differentiated, and
`cross_play_weight=0.2` — set once at the original, far shorter budget and
never revisited this session — may no longer be strong enough to hold
specialization now that competence isn't the binding constraint. This is the
same shape of problem BRDiv had (its diversity knob needed retuning once its
own budget question was settled), just not yet addressed for CoMeDi. Follow-up
sweep queued.

### `cross_play_weight` follow-up: confirmed the hypothesis, rejected the trade

Single point, `cross_play_weight=0.4` (double the adopted `0.2`), at the
converged `1.92e8` budget. Not fully converged itself (slope +0.0084/1k vs
`0.2`'s +0.0046), so treat the SP gap below as an upper bound on the true cost.

| `cross_play_weight` | SP | XP | SP−XP |
|---:|---:|---:|---:|
| 0.2 *(adopted)* | **0.472** | 0.256 | 0.217 |
| 0.4 | 0.430 | **0.088** | **0.342** |

(100-episode rescoring, matching fidelity.) The hypothesis was right —
doubling the weight recovers separation dramatically (+0.125, or +58%) by
suppressing cross-play (0.256→0.088) — but at a real SP cost (-0.042, ~9%,
well past the ~0.005–0.012 noise floor established earlier in this file), and
`0.4302` sits below the competence floor set by `0.2` (`0.4488` at 5%
tolerance). **Rejected; `cross_play_weight=0.2` stays adopted.** Kept
consistent with the competence-first rule applied everywhere else in this
file, even though the trade ratio here (9% SP for 58% separation) is far more
favorable than BRDiv's `cross_play_weight=0.20` ever offered (a 42% SP
collapse for a 16% separation gain) — worth a second look if the downstream
`ppo_br.py` validation this whole file is gated on ever suggests separation is
underweighted relative to competence in the adoption rule itself, but that's a
change to the rule, not a case for bending it here.

**Not explored:** an intermediate `0.3`, or re-testing `0.4` once it's
actually converged rather than extrapolating from a still-climbing curve.

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

**Confirmed.** `results/teammate_generation/brdiv_lbf_12x12-72cf94790359`
(`num_envs=192`, `cross_play_weight=0.05`, the adopted config) recovers
structure: SP 0.286, XP 0.181, separation 0.105 — self-play clearly above
cross-play, versus SP *below* XP at `n=5, num_envs=64`. The n² diagnosis holds.
Also notably above the n=3 reference's SP 0.237, so the fix did not just repair
the collapse, it improved on the population BRDiv was originally tuned at.

**Not converged, despite an earlier note in this file claiming otherwise.**
The `+0.002/1k` "flat" slope quoted in an earlier draft of this entry was
misattributed — it belongs to the *old* `n=5, num_envs=64` collapsed run in the
table above, not to this `num_envs=192` run. The actual last-quarter slope here
is **+0.027/1k**, comparable to CoMeDi's still-climbing `4.8e7` checkpoint
(+0.038/1k, see above), at only 58% of the food against FCP's ~97% ceiling.
`total_timesteps` was scaled with `num_envs` to hold the update count fixed at
5,493 (the point of `_paired_scale`), but that count was tuned to be adequate at
`n=3, num_envs=64` — nothing re-derives it for `n=5, num_envs=192`, and the
curve says it isn't. `cross_play_weight` is still worth sweeping, but not yet
at a budget where the result is trustworthy — see below.

### Budget check: doubling `total_timesteps` at `cross_play_weight=0.05`

`total_timesteps=1.35e8 → 2.7e8` (5,493 → 10,986 updates), `cross_play_weight`
and `num_envs=192` held fixed, to separate the budget question from the
diversity-weight question before sweeping the latter.

| `total_timesteps` | SP | XP | SP−XP | % food | last-quarter slope /1k |
|---:|---:|---:|---:|---:|---:|
| 1.35e8 | 0.286 | 0.181 | 0.105 | 58.3 | +0.027 |
| 2.7e8 | **0.313** | 0.196 | **0.117** | 58.0 | **+0.010** |

Doubling gained +0.026 SP and +0.012 separation, and the slope fell to roughly a
third of its previous value — a sharper deceleration than CoMeDi showed over
its own first doubling (which roughly halved). `% food` stayed flat while SP
rose, which doesn't fully square with a food-eaten proxy tracking the same
signal 1:1 and is unexplained; not investigated further here. `+0.010/1k` is
still well above BRDiv's originally-reported converged rate (+0.002/1k, at the
old collapsed `n=5, num_envs=64` setting) but far closer to it than +0.027/1k
was, so treated as adequate for the purpose of unblocking `cross_play_weight` —
not as a converged number in its own right. One more doubling was not run.

### `cross_play_weight` sweep at the old (undertrained) budget — do not read a ranking off this

Run at `total_timesteps=1.35e8`, before the budget check above. Included for the
record, since the doc's convention is to keep what a sweep could not conclude,
not just what it could.

| `cross_play_weight` | SP | XP | SP−XP |
|---:|---:|---:|---:|
| 0.02 | 0.260 | 0.210 | 0.050 |
| 0.05 *(adopted)* | 0.286 | 0.181 | 0.105 |
| 0.10 | 0.252 | 0.121 | 0.131 |
| 0.20 | 0.171 | 0.049 | 0.121 *(below competence floor)* |

`collect`, run only over the three new cells, reported `0.10` as best — but that
ranking excludes the `0.05` baseline (different sweep name, different config
hash, not in the same manifest) and is an artifact of that exclusion: against
all four points together, `0.05` has the highest SP of any cell and clears its
own competence floor; none of the other three do. More importantly, all four
cells share the single budget just shown to be inadequate, and the fastest-
climbing cell (`0.05`) is the current baseline — cells sitting lower on their
own curve are not distinguishable from cells that are genuinely worse at the
tradeoff. **Re-run at `total_timesteps=2.7e8` before trusting any ordering
here.**

### `cross_play_weight` sweep, re-run at `total_timesteps=2.7e8` — this one is trustworthy

Same three cells plus the `0.05` budget-check baseline, all now at matched
budget and all past the worst of the deceleration: last-quarter slopes sit
between +0.003/1k and +0.010/1k, versus the −0.001 to +0.027/1k spread at the
old budget — close enough to flat that a ranking is meaningful.

| `cross_play_weight` | SP | XP | SP−XP | % food | slope /1k |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.323 | 0.208 | 0.114 | 57.5 | +0.0035 |
| 0.05 | 0.313 | 0.196 | 0.117 | 58.0 | +0.0100 |
| **0.10** | **0.335** | 0.130 | **0.205** | 49.7 | +0.0033 |
| 0.20 | 0.207 | 0.053 | 0.154 *(below floor)* | 39.0 | +0.0030 |

**Adopted: `cross_play_weight=0.10`.** Highest SP of any cell (floor at 95% of
0.335 is 0.318; only `0.02` and `0.10` clear it) and, among those two, nearly
double the separation of `0.02` (0.205 vs 0.114). Under the doc's own
competence-first-then-separation rule this isn't a close call: `0.10` wins on
both axes at once, which the `0.05` baseline never did even at the corrected
budget. Supersedes the `0.05` adoption from the scaling-law section above.

**One loose end, not chased further.** `0.10` has the best SP-based self-play
score but the *worst* `% food` of the top three (49.7% vs 57–58%). The two
metrics come from different measurement paths — `% food` is a training-time
rollout statistic, SP is the post-training population evaluation on sampled
actions — and they agreed everywhere else in this file (FCP, the budget-check
table above). Here they point in different directions for the winning cell.
Plausibly the return function isn't purely food count, or the two rollouts
sample differently enough to diverge at this specific setting; not resolved,
and worth a look before this population is used downstream.

**Single seed, same caveat as everywhere else in this file.** `0.10`'s margin
(both SP and separation, over both other in-band and out-of-band cells) is
large enough that it's a more defensible read than the FCP entry's
single-seed `learning_rate` ordering, but it is still one run per cell.

### Chasing SP past 0.4

Motivated by a hunch that BRDiv was undertrained relative to FCP, which the
budget check above already partly vindicated. Six further full-budget runs, all
at `cross_play_weight=0.10` unless noted, asking two questions in sequence:
does BRDiv's underlying PPO need the same hyperparameters FCP's did, and does
one more budget doubling clear 0.4.

**PPO knobs, at `total_timesteps=2.7e8`, against the FCP-tuned LBF values
(`learning_rate=1e-3`, `entropy_coef=0.003`, `clip_eps=0.03`; BRDiv's inherited
defaults were `5e-4`, `0.01`, `0.05`):**

| change | SP | SP−XP | slope /1k | verdict |
|---|---:|---:|---:|---|
| `entropy_coef` 0.01→0.003 | 0.350 | 0.219 | +0.0042 | adopted |
| `entropy_coef` 0.003→0.001 | 0.331 | 0.216 | — | reverted — non-monotonic, `0.003` is a local peak |
| `learning_rate` 5e-4→1e-3 | 0.262 | 0.115 | +0.0061 | reverted — see below |
| `clip_eps` 0.05→0.03 | 0.338 | 0.229 | +0.0096 | reverted — roughly a wash, and less converged |

Only `entropy_coef=0.003` transferred from FCP; the other two didn't, and
`learning_rate` didn't just fail to help, it actively hurt. Its training curve
by quarter (mean return / slope-per-1k) shows why:

| quarter | mean return | slope /1k |
|---|---:|---:|
| 1 | 0.118 | +0.077 |
| 2 | 0.205 | +0.002 |
| 3 | 0.220 | +0.004 |
| 4 | 0.234 | +0.006 |

A fast start that stalls hard after the first quarter and barely recovers —
`1e-3` destabilized training rather than just taking longer, and settled into a
lower-competence regime. Matches the direction CoMeDi's own `learning_rate`
finding took, not FCP's; hyperparameters don't transfer cleanly across
generators here even on the same environment.

**Budget, at the winning PPO setting (`entropy_coef=0.003`, rest unchanged):**

| `total_timesteps` | SP | SP−XP | slope /1k |
|---:|---:|---:|---:|
| 2.7e8 | 0.350 | 0.219 | +0.0042 |
| 5.4e8 | **0.386** | **0.271** | **+0.0013** |

The big one. +0.037 SP and +0.052 separation from one doubling, landing at a
slope that finally matches BRDiv's originally-reported converged rate
(+0.002/1k) rather than approaching it asymptotically the way CoMeDi's budget
axis did. The undertrained-BRDiv hunch was right, and this is most of where the
overall gain came from.

**`cross_play_weight=0.07`, at the new best budget** (5.4e8, `entropy=0.003`):
SP 0.365, SP−XP 0.172, slope +0.0026/1k — both worse than `0.10`, and equally
converged, so not a budget artifact. This was proposed on a "lower weight
trades separation for competence" model that, in hindsight, the data already
contradicted: in the original `2.7e8`/`entropy=0.01` sweep, `0.10` had already
beaten both `0.05` (0.335 vs 0.313) and `0.02` (0.335 vs 0.323) on SP, not just
on separation. Confirmed twice now from both sides — `0.10` is a local optimum
on this axis, not a point on a monotonic slope, and going lower is not a lever
that works here.

**Result: SP 0.386, short of the 0.4 target by 0.014.** Adopted-for-the-chase
config: `cross_play_weight=0.10`, `entropy_coef=0.003`, `learning_rate=5e-4`
(unchanged), `clip_eps=0.05` (unchanged), `total_timesteps=5.4e8`,
`num_envs=192` (unchanged) — not pushed into `gen_teammate_configs.py`, per the
same "leave it recorded" call as the rest of this entry. Net gain from the
whole chase, against the confirmed `num_envs=192` baseline at the top of this
section: **+0.10 SP, +0.166 separation** (SP 0.286→0.386, separation
0.105→0.271) — real, but the last two moves tried (`clip_eps`, `xpw=0.07`) both
came back negative or flat, which is the signal that stopped this rather than
hitting the target number itself. `cross_play_weight=0.12` (the one direction
on that axis not yet tried) and a third budget doubling are both still
technically open, but the hit rate on single-point guesses was declining by the
end of this sequence and neither was pursued.

**Cost.** Six full-budget runs at `2.7e8`–`5.4e8` timesteps in this chase alone,
on top of the eight already spent on the scaling-law confirmation and the two
`cross_play_weight` sweeps above — BRDiv's LBF tuning is now the single most
expensive line item in this file. Worth knowing before proposing another
single-point guess on this generator without a stronger prior than "try it and
see."

---

## L-BRDiv × LBF 12×12 — `tolerance_factor` sweep

Never run at `num_envs=192` before this — the 22%-food, SP-0.212 number quoted
earlier in this file is the old collapsed `n=5, num_envs=64` setting, same
regime BRDiv collapsed in, not a tuned result. Rather than re-run BRDiv's whole
incremental discovery (scaling confirmation → budget check → PPO knobs →
budget again), this sweep started from what transferred there:
`entropy_coef=0.003` (the one PPO knob that helped) and `total_timesteps=5.4e8`
(the budget BRDiv actually needed to reach a flat curve), applied directly
rather than rediscovered. `learning_rate=5e-4` and `clip_eps=0.05` are already
L-BRDiv's defaults, matching what BRDiv confirmed rather than requiring a
change. `lagrange_learning_rate=0.0036` is L-BRDiv-specific machinery, already
scaled for `n=5` by `_lagrange_lr` — not a BRDiv-shared parameter, left alone.

### Grid

```
generator.total_timesteps    = 5.4e8
generator.ppo.entropy_coef   = 0.003
generator.tolerance_factor   = 0.03, 0.1 (default), 0.3
```

One seed each. `tolerance_factor=0.1` doubles as the scaling-fix confirmation,
since no separate baseline run existed to reuse.

### What the sweep found

**The transfer worked, and this generator responds very differently to its
diversity knob than BRDiv does to `cross_play_weight`.**

| `tolerance_factor` | SP | XP | SP−XP | % food | slope /1k |
|---:|---:|---:|---:|---:|---:|
| 0.03 | **0.396** | 0.185 | 0.211 | 71.0 | +0.0021 |
| 0.10 *(default)* | 0.331 | 0.042 | 0.289 | 59.8 | +0.0039 |
| 0.30 | 0.340 | 0.026 | **0.314** | 56.9 | +0.0018 |

All three converged (slopes +0.002–0.004/1k, matching BRDiv's converged rate),
so this ranking is trustworthy the same way the re-run `cross_play_weight`
sweep was.

`tolerance_factor` enforces a *minimum required margin* between self-play and
cross-play (`SP - XP > tolerance_factor`, via Lagrangian dual ascent) rather
than weighting a loss term, and the sweep shows that mechanism plainly: raising
it from `0.03` to `0.3` barely moves SP (0.396→0.340) but crushes XP
(0.185→0.026) — the constraint is satisfied by suppressing cross-play, not by
trading away self-play competence the way BRDiv's soft weight does. This is a
materially different failure/success mode between the two paired generators
despite sharing the same n² scaling problem.

**Adopted: `tolerance_factor=0.03`.** Clears its own competence floor (only
cell within 5% of the best SP); the other two don't. SP 0.396 edges out
BRDiv's own chase-best (0.386) — L-BRDiv is, at this budget, the more
competent of the two paired generators — while trailing on separation (0.211
vs BRDiv's 0.271). Not pushed into `gen_teammate_configs.py`, same "record
only" call as BRDiv.

### What the sweep could not conclude

**Whether SP keeps climbing below `0.03`.** The trend from `0.3→0.1→0.03` is
mostly monotonic on XP but not cleanly on SP (0.340→0.331→0.396 — a dip then a
jump), so a fourth point below `0.03` isn't guaranteed to continue improving
competence; it's also the point at which the constraint starts becoming
trivially satisfiable, converging toward plain self-play with no diversity
pressure at all. Not run.

**Single seed**, same caveat as every sweep in this file — the SP gap between
`0.03` and the other two is large enough to be a credible signal, but it's one
run per cell.

---

## FCP × Hanabi — budget

Never run before this. jax-aht's inherited config is `total_timesteps=1e9`,
`num_envs=32` (244,141 updates). `num_envs=64` adopted first, matching the
batch-size lesson from LBF — which immediately means `total_timesteps` numbers
don't carry over raw: `1e9` at the new batch size is only 122,070 updates, half
of jax-aht's actual reference. Bracketed the corrected reference point (`2e9`)
rather than reusing the raw upstream number, per the same "`num_envs` is not a
throughput knob" lesson the LBF FCP entry already established.

### Grid

```
generator.num_envs         = 64
generator.total_timesteps  = 1e9, 2e9, 5e9
```

One seed each.

### What the sweep found

| `total_timesteps` | updates | SP | XP | SP−XP | slope /1k |
|---:|---:|---:|---:|---:|---:|
| 1e9 | 122,070 | 19.02 | 3.68 | 15.34 | +0.0111 |
| 2e9 | 244,140 | 19.97 | 5.19 | 14.78 | +0.0039 |
| 5e9 | 610,351 | 19.37 | 2.69 | **16.68** | +0.0023 |

**Competence is flat past `1e9`**, within about a point on Hanabi's 25-point
scale across a 5x budget range — not a clean monotonic climb, but no further
gain either. Slope drops into a converged range (< +0.004/1k) by `2e9`.

**Separation is not monotonic, and reversed rather than continuing a trend.**
Going `1e9 → 2e9`, XP rose and separation fell — the same "competence
saturates, cross-play catches up" shape CoMeDi showed on its own LBF budget
doubling, and it looked like a trend worth flagging mid-sweep. It didn't
continue: at `5e9`, XP fell back below even the `1e9` value and separation hit
its highest point of the three. Single seed — treated as noise around a
roughly flat separation level (14.8–16.7), the same caution applied to every
other non-monotonic result in this file, not a real trend in either direction.

### Adopted: `total_timesteps=2e9`, `num_envs=64`

Not the best of the three on any single metric (`5e9` has the best separation,
`1e9` is cheapest) — chosen because it matches jax-aht's own reference update
count almost exactly (244,140 vs upstream's 244,141 at `num_envs=32`), is fully
converged (slope +0.0039/1k), and costs a quarter of `5e9` for differences that
are within noise rather than a signal favoring more spend. Pushed into
`gen_teammate_configs.py`.

### What the sweep could not conclude

**PPO hyperparameters untouched.** Unlike LBF, no `entropy_coef` /
`learning_rate` / `clip_eps` sweep was run here — this entry is budget only.
LBF's tuned PPO values were checked against Hanabi's inherited ones at smoke
scale before this sweep and not carried forward (see the discussion of why
LBF's hyperparameters don't transfer across environments); Hanabi's PPO stays
jax-aht's inherited settings.

**CoMeDi, BRDiv, and L-BRDiv have not been started on Hanabi.** Only FCP is
tuned here. Deliberately parked in favor of Overcooked-v2 integration — see
`CLAUDE.md`'s State section.

**Whether `5e9`'s separation is real** is exactly the kind of question this
file usually resolves with a second seed, not run here given the cost — `5e9`
alone is 610,351 updates, the largest single run in this file.

---

## FCP × Overcooked-v2 — budget, against an external reference

The first fully-completed result on Overcooked-v2, absorbed this session
(`PROVENANCE.md` — from `jaxmarl==0.1.0`, not a version bump, since that
release's `overcooked_v2` declares `jax<=0.4.38` and can't install alongside
this project's `jax==0.5.3`). Two decisions and one real infrastructure gap
came before any budget question was answerable at all:

- **Reward shaping**: v2's `rewards` (from `step_env`) is already the correct
  base task reward; `shaped_reward` is diagnostic-only, not folded into
  training. Independently confirmed against ICRL4AHT's own overcooked_v2
  wrapper (read for understanding only — their repo has no license, so
  nothing was copied; same clean-room principle as AD-RPG), whose own comment
  states the same thing about the same field.
- **Partial observability, not full**: `agent_view_size=2` on every
  registered `overcooked_v2_*` preset, matching upstream's only validated
  reference config (`baselines/IPPO/config/ippo_rnn_overcooked_v2.yaml`).
  This is v2's headline feature over v1 and the reason to use v2 at all
  rather than a full-observability run comparable to v1's — but it requires
  a policy with memory to be useful, which is why the next point mattered.
- **The crossplay evaluation harness could not score an RNN policy at all**,
  discovered by actually running one: `population/loading.py`'s
  `get_fcp_population` hardcoded `MLPActorCriticPolicy` regardless of
  `actor_type`. Fixed by dispatching on `actor_type`, scoped to FCP only —
  `run_episodes.py` and `AgentPopulation` were *already* fully polymorphic
  over policy type (hidden-state threading, generic `init_hstate`/
  `get_action` dispatch), designed for this from the start and simply never
  wired at the loading layer. CoMeDi/BRDiv/L-BRDiv still can't use
  `actor_type="rnn"`: there is no RNN variant of
  `ActorWithConditionalCriticPolicy`, so they stay on `"mlp"` and therefore
  cannot make good use of `agent_view_size=2` if pointed at these presets
  today.

### Grid

Two stages. First, a device-safe bracket translating upstream's own
reference update count (`NUM_ENVS=256`, `TOTAL_TIMESTEPS=3e7` → 293 updates)
down to `num_envs=64` (256 OOMs on this GPU — confirmed by an actual crash,
not just `check_device.py`'s estimate) while holding the update count fixed,
the same `num_envs`-isn't-a-throughput-knob rule as everywhere else in this
file:

```
generator.num_envs         = 64  (fixed for both stages)
generator.total_timesteps  = 3.75e6, 7.5e6, 1.5e7   # stage 1: 146/292/585 updates
generator.total_timesteps  = 3e7, 6e7, 1e8           # stage 2: 1,171/2,343/3,906 updates
```

`agent_view_size=2`, `actor_type="rnn"` fixed throughout. PPO hyperparameters
taken from upstream's reference config rather than v1's MLP-tuned ones,
since partial observability + RNN makes v1's values an irrelevant prior.

### What the sweep found

**Stage 1 was nowhere close, by an external measure this file doesn't
usually have.** The original Overcooked-v2 paper reports **~163** return on
`counter_circuit`. The largest stage-1 cell (585 updates) was still climbing
with no sign of leveling off across all four quarters of its own training
(47.55 → 58.50 → 64.61 → 71.69) — under half the reference at the top of the
bracket. This is a clearer "still starved" signal than the slope diagnostic
alone usually gives, and it's why stage 2 jumped straight to a much larger
bracket anchored on upstream's raw `total_timesteps` number rather than
another small increment.

**A run was lost, and the lesson from it was still worth keeping.** The
`1e8` cell was launched in the foreground of an SSH session that dropped
overnight — the process died with it, and since `save_train_run` only writes
once at the very end, no checkpoint exists (~7 hours of GPU time,
unrecoverable as a scored population). But `metrics.jsonl` streams
incrementally and survived, 92.7% through (3,620 of 3,906 updates) before
the connection dropped:

| quarter | mean return |
|---|---:|
| 1 | 106.29 |
| 2 | 193.42 |
| 3 | 199.82 |
| 4 | 201.11 |

Crossed the 163 reference at **train_step 716 — 18.3% of the target
budget** — and was clearly decelerating by quarter 3 (+1.29 from quarter 3
to 4, versus +87 from quarter 1 to 2). Lesson, separate from the tuning
result itself: **run anything that needs to survive a dropped connection
inside `tmux`/`screen`**, not a plain foreground process. Costly enough to
be worth stating plainly rather than filing away.

**`6e7`, rerun properly (inside `screen`), confirms the lost run's shape and
gives a real, scored result.**

| | SP | XP | SP−XP |
|---|---:|---:|---:|
| `6e7` (2,343 updates) | **205.20** | 90.25 | 114.95 |

SP is **126% of the paper's reference**. Its own training curve (quarters:
77.90 → 168.68 → 187.44 → 190.67, final 200.00) decelerates the same way the
lost `1e8` run's did, and lands within a few points of where that run's own
quarters 3–4 sat (199.82 / 201.11) despite 67% less budget — `1e8`'s entire
advantage over `6e7` is on the order of 5–10 points on a ~200 scale. Neither
`3e7` nor `1e8` was run to completion at this fidelity; the comparison
above is what the adoption call rests on.

### Adopted: `total_timesteps=6e7`, `num_envs=64`, `actor_type="rnn"`, `agent_view_size=2`

Not the largest budget tried, and not confirmed flat by its own slope
(+3.49/1k, still mildly positive) — adopted because it already clears an
external, published reference by a wide margin, and the lost `1e8` run's
curve shows the remaining headroom past this point is small relative to its
cost. Pushed into `gen_teammate_configs.py`.

### What the sweep could not conclude

**Whether `6e7` is actually converged, or just close enough.** `1e8` was
never scored at this fidelity — its checkpoint didn't survive — so the
comparison above is curve-shape evidence, not a second scored data point.

**The `num_envs=64` vs. upstream's `256` batch-size question, still open.**
Flagged before this bracket ran and not resolved by it: FCP × LBF found
batch size matters independently of update count, and this environment's
budget was chased entirely by adding more (smaller-batch) updates rather
than testing whether more batch would have gotten here cheaper.
`check_device.py` showed real headroom left at `num_envs=64` (16% of the
device budget) — `~128` is a plausible next value to test, deliberately not
conflated into this budget-only sweep.

**Separation (114.95) has no comparison point yet** — the first crossplay
reading FCP has ever produced on this environment. Whether that's large,
small, or typical for this generator/task is unknown.

**CoMeDi/BRDiv/L-BRDiv remain untouched on Overcooked-v2**, and can't
meaningfully use `agent_view_size=2` until an RNN-compatible
conditional/double-critic policy exists.

---

## BRDiv/L-BRDiv × Overcooked-v2 — RNN support and budget derivation

FCP's Overcooked-v2 result above left BRDiv/L-BRDiv stuck: `agent_view_size=2`
partial observability needs a policy with memory to be useful, and there was
no RNN variant of `ActorWithConditionalCriticPolicy` for either generator to
use. Two things had to happen before a budget question was even askable.

**RNN support.** Added `RNNActorWithConditionalCriticPolicy`
(`agents/rnn_actor_critic.py`, `rnn_actor_critic_agent.py`), matching
`ActorWithConditionalCriticPolicy`'s convention with a GRU actor path, and
wired `actor_type` dispatch through `configs/teammate_gen.py` and
`population/loading.py` the same way FCP's was. This exercised BRDiv.py's and
LBRDiv.py's `_env_step` hstate-threading code for the first time — both
carried a "not tested with recurrent actors" warning, and both actually broke
on first run: a `needs_resample` broadcast mismatch, and a `jax.vmap` axis
mismatch. BRDiv vmaps `forward_pass_conf`/`forward_pass_br` per-actor (each
of the `num_envs` actors can be paired with a different population member,
so params vary per actor), but the RNN policy's hstate carries its actor axis
at position 1, not 0 — fixed with an explicit `in_axes`/`out_axes` on that
vmap. Verified via an isolated smoke test for each generator (trains and
checkpoints end-to-end) and the full suite (309 passed, no regressions).

**GPU memory.** `_paired_scale`'s usual reference point gives `num_envs=192`
(7.7 envs/pairing, matching LBF's established-safe reference) — `check_device.py`
puts that at 99% of this 6GB GPU's memory, confirmed by an actual OOM.
`num_envs=96` (3.84 envs/pairing — between the known-collapse point 2.6 from
the LBF `n=5` incident and LBF's established-safe 7.7) fits at 49% and both
generators train cleanly at that size.

**Budget, derived rather than guessed.** On LBF, BRDiv/L-BRDiv's budget
(`_paired_scale(64, 1.8e8)`) is 7.5x FCP's at the same `num_envs=64`
(`1.8e8 / 24e6`) — a ratio independent of the pairing multiplier, which
scales `num_envs` and `total_timesteps` together and cancels out of
`num_updates`. Applying 7.5x to FCP's tuned Overcooked-v2 budget (`6e7` at
`num_envs=64`) gives a base of `4.5e8`, then ×1.5 to move from that 64-env
base to the actual `num_envs=96` (holding `num_updates` constant) gives
`6.75e8`. Pushed into `gen_teammate_configs.py`.

### What this could not conclude yet

**SP-vs-XP at `num_envs=96` is still an open empirical question, and the
first real attempt was inconclusive.** The smoke tests that verified the RNN
plumbing only ran 2e6 timesteps (52 updates) — far too little to say
anything about collapse risk. A real `6.75e8`-timestep BRDiv run was
launched and killed at 44% (7,737 of 17,578 updates, ~19.5 hours in) before
reaching its crossplay evaluation, once its own training-return curve
(`Train/base_return`) showed a hard plateau: decile means climbed
4.5 → 9.4 → 12.1 → 13.0 → 14.9 → 16.3 → 16.8 → 17.2 → 16.4 → 18.0 through the
44% mark, with the final ~2,000 updates flat within noise (15.8–18.3, no
consistent slope) and the single highest episode return across the whole run
only 40 — roughly an order of magnitude below FCP's own Overcooked-v2 curve
at a comparable fraction of training (which had already cleared 78 in its
*first* quarter). This is consistent with the collapse risk `num_envs=96`
(3.84 envs/pairing) was flagged for, but it's curve-shape evidence from an
incomplete run, not a scored SP/XP result — the run was never rescoreable
since BRDiv only checkpoints once at the very end. Whether `num_envs=96` is
actually unworkable for BRDiv/L-BRDiv on this task, or whether a longer
warm-up would still have turned the corner, is unresolved.

**"Train longer at `num_envs=96`" is not a fix for this, if it is a
collapse.** Envs-per-pairing is `num_envs / population_size²`, a property of
the rollout, not of how many updates run — doubling `total_timesteps` at a
fixed `num_envs=96` still gives only 3.84 environments' worth of data per
confederate/best-response pairing on every update. The two real levers are a
bigger GPU (to actually reach `num_envs=192`, 7.68 envs/pairing) or reducing
the per-env memory footprint on this GPU (smaller GRU hidden dim, or
stratified pairing sampling instead of independent uniform draws per env, per
the Overcooked-v1 Known-open note in `CLAUDE.md`) so more envs fit at once.
Neither has been tried yet.

**Whether `6.75e8` is the right budget, as opposed to a defensible one.**
The 7.5x ratio and the FCP Overcooked-v2 budget it's built on both carry
their own uncertainty (see FCP × Overcooked-v2's own "could not conclude"
above) — this derivation propagates that uncertainty rather than resolving
it, and has not itself been swept. Moot until the `num_envs=96` question
above is resolved one way or the other.

## CoMeDi × Overcooked-v2 — RNN support and budget derivation

The same gap BRDiv/L-BRDiv had: `agent_view_size=2` partial observability
needs a policy with memory, and CoMeDi had no RNN-compatible conditional
critic. Landed the same session as the above, reusing
`RNNActorWithConditionalCriticPolicy`.

**RNN support.** Wired `actor_type` dispatch through
`initialize_actor_with_conditional_critic` (`agents/initialize_agents.py`),
the shared helper both CoMeDi.py and `common/agent_loader_from_config.py`
call — a single fix-point rather than two. Unlike BRDiv/L-BRDiv, CoMeDi
never reassigns which population member plays a role mid-rollout (no
per-env independent id sampling), so none of BRDiv's per-actor `jax.vmap`
axis handling was needed; the existing shape conventions in CoMeDi.py's
rollout functions already matched what the RNN policy expects, and the real
work was threading a real hidden state through four rollout functions
(`_env_step_conf_ego`, `_env_step_conf_br` at two call sites,
`_env_step_mixed`) plus GAE bootstrapping and PPO minibatching/loss, all of
which already had the right parameter slots sitting unused as `None`.

Two bugs surfaced only by actually running training, not by reading:

- **The warmup phase needed its own RNN variant.** CoMeDi trains its first
  population member via a plain self-play IPPO trainer
  (`make_ppo_train`/`initialize_agent`) that predates any real population to
  condition on, hardcoded to `actor_type="pseudo_actor_with_conditional_critic"`
  regardless of the main phase's setting. An RNN main phase with an
  MLP-shaped warmup member fails when the warmup member's params go into the
  same `BufferedPopulation` as the RNN-shaped members added afterward. Fixed
  with a new `PseudoRNNActorWithConditionalCriticPolicy` and
  `CoMeDiRuntime.warmup()` now picks the matching pseudo type.
- **`CoMeDiRuntime.to_agent_dict()` didn't include `ACTOR_TYPE` at all.**
  `initialize_actor_with_conditional_critic`'s new dispatch reads
  `config.get("ACTOR_TYPE", ...)` from the dict it's handed; CoMeDi.py calls
  it with `config.to_agent_dict()` for two things — the population buffer's
  dummy policy and each new confederate's real policy — and that dict was
  silently falling back to the MLP default regardless of `actor_type`, while
  the warmup phase (which reads `actor_type` through a separate typed
  argument in `make_train`, not this dict) correctly built RNN-shaped
  params. `BufferedPopulation.add_agent` failed on the resulting dict-key
  mismatch on the very first real training run.

Verified via an LBF smoke test (small budget, `actor_type=
"rnn_actor_with_conditional_critic"`) — trained, checkpointed, and scored
successfully — the full 309-test suite, and a separate independent review
pass over the diff (shape/broadcast correctness, MLP-mode backward
compatibility, tuple-ordering consistency across every rollout function,
and the reset-branch semantics), which found no further issues.

**Budget, derived the same way as BRDiv/L-BRDiv's.** On LBF, CoMeDi's
`total_timesteps_per_iteration` (`1.92e8`) is 8x FCP's `total_timesteps`
(`24e6`) at the same `num_envs=64`. Applying 8x to FCP's tuned Overcooked-v2
budget (`6e7`) gives `4.8e8`. Unlike BRDiv/L-BRDiv, CoMeDi has no
per-env independent id sampling and therefore no n²-pairing memory
constraint, so `num_envs` didn't need to shrink to fit this GPU —
`num_envs=64` (`check_device.py`: 16% of the device budget) matches both
CoMeDi's own LBF value and FCP's Overcooked-v2 value, keeping the ratio
derivation consistent with how it was built. Pushed into
`gen_teammate_configs.py`.

### What this could not conclude yet

**Untested at the real budget.** Only the small LBF smoke test has actually
trained with this code path; the derived `4.8e8` Overcooked-v2 budget has
not been run.

**The 8x ratio and the FCP budget it's built on both carry the same
uncertainty already noted for BRDiv/L-BRDiv's derivation** — propagated,
not resolved, here either.

## All four × Overcooked-v2 — matched to the source paper (supersedes the three sections above)

The three Overcooked-v2 adoptions above (FCP's `actor_type="rnn"`, BRDiv/L-BRDiv
and CoMeDi's RNN conditional critic, all over *flattened* observations with
PPO hyperparameters inherited from v1 or JaxMARL's IPPO reference) were built
before we went back to the environment's own paper. A BRDiv Counter Circuit run
on that setup came back **SP 28.0 ≈ XP 26.55, separation ~1.4** — a competent-
looking number hiding a homogeneous population — and a fresh run stalled at
return ~1.0. That sent us to *OvercookedV2: Rethinking Overcooked for Zero-Shot
Coordination* (Gessler et al., ICLR 2025), the implementation this env is
absorbed from, which diverges from our setup on three things that gate learning.
All three are now matched; the earlier adoptions are **superseded**, not amended.

**1. Annealed dense-reward shaping (the gate).** The v2 wrapper computes
`info['shaped_reward']` but returns only the sparse delivery reward
(`overcooked_v2_wrapper.py`). Sparse Overcooked is a long-horizon exploration
problem; the paper adds the shaped reward with a linear 1→0 anneal over
`REW_SHAPING_HORIZON=5e6` env steps. Wired as `reward_shaping_horizon` on
`PpoHyperparams`, folded right after `env.step` in every generator's rollout
(`marl/reward_shaping.py`; `0` disables it, so LBF/Hanabi/v1 are untouched).

**2. CNN+GRU network (App. C.1.1).** The paper reports that architectures
without a convolutional stem "did not learn good policies" here. Our RNN ran a
GRU over the flattened grid with no CNN. New `models/cnn_rnn_actor_critic.py`
(three 1×1 convs `[128,128,8]` → three 3×3 convs `[16,32,32]`, ReLU, zero-pad →
flatten → Dense 128 → LayerNorm → GRU 128 → heads), in a shared-trunk variant
(`cnn_rnn`, FCP) and a teammate-id conditional-critic variant
(`cnn_rnn_actor_with_conditional_critic`, CoMeDi/BRDiv/L-BRDiv) plus the pseudo
shape CoMeDi's warmup needs.

**3. Table-4 PPO backbone + LR warmup (App. D.2.1).** The authoritative Counter
Circuit table is Table 4, not D.1.1 (which is the fully-observable "Limitations"
feed-forward setting). `_OVERCOOKED_V2_PPO` in `gen_teammate_configs.py`: `LR=5e-4`,
`CLIP_EPS=0.2`, `ENT_COEF=0.01`, `VF_COEF=0.5`, `MAX_GRAD_NORM=0.25`,
`NUM_MINIBATCHES=64`, `UPDATE_EPOCHS=4`, `GAMMA=0.99`, `GAE=0.95`, `ANNEAL_LR`,
`LR_WARMUP=0.05` (linear warmup then cosine decay, `marl/lr_schedule.py`),
`REW_SHAPING_HORIZON=5e6`. Network `FC_DIM=GRU=128`, ReLU. Env `negative_rewards=True`.

The paper does **not** train BRDiv, so `cross_play_weight` stays *our* knob, not
the paper's; started at `0.5` for BRDiv (a mid-range diversity value, since a
collapsed population was the symptom). L-BRDiv's `tolerance_factor` is still the
v1-inherited `10.0`, unswept.

### What this establishes, and what it cannot

- **Verified (CPU):** all four generators train, checkpoint, *and score* end-to-end
  on `overcooked_v2_counter_circuit` with the CNN actor types — the full path
  including the eval-time policy reconstruction in `population/loading.py`, which
  had to learn the CNN variants (an early smoke caught FCP silently falling back to
  an MLP-shaped policy at scoring time). Pinned by
  `tests/unit/teammate_gen/test_overcooked_v2_paper_matching.py` (shaping anneal,
  schedule shape, CNN forward pass from `env.obs_shape`, generated-config values,
  LBF non-regression).
- **Not GPU-validated.** No claim yet that this reaches the paper's ~150–205 return
  or that separation becomes non-trivial. That BRDiv Counter Circuit run — matched
  setup, sweeping `cross_play_weight` around `0.5` — is the real check, and the
  reason Part 1 (shaping on the old net) was landed first as the fast de-risk.
- **`num_envs=256` for all four** — the paper's value, on the H100 training target
  (the earlier 64/96 were 6GB-GPU limits). At `NUM_MINIBATCHES=64` that is the paper's
  exact 4-envs-per-minibatch; for BRDiv/L-BRDiv at n=5 it is 256/n²=10.2 envs/pairing,
  above LBF's established-safe 7.7 (invariant #4 holds with margin). `total_timesteps`
  was scaled with `num_envs` to hold `num_updates` fixed (FCP 6e7→2.4e8, CoMeDi
  4.8e8→1.92e9, BRDiv/L-BRDiv 6.75e8→1.8e9); bumping envs without this would have
  quartered the gradient-step count and undertrained the population — the failure the
  FCP/Hanabi budget chases and `_paired_scale` both guard against.
- **Budgets carried over from the pre-paper sections are unre-derived** against the
  new backbone; treat them as starting points. `rollout_length` is still the preset's
  400, not the paper's `NUM_STEPS=256` — a remaining, deliberate difference (episode
  length is an env-preset decision), noted here rather than changed.

### First GPU run: BRDiv Counter Circuit, `320e6` — not converged, keep training

First real (H100) training on the matched backbone. A **short, deliberately
under-budget probe** to see whether the setup learns at all before committing to
the full `1.8e9`. Config as committed except **`total_timesteps=320e6`** (3,125
updates at `num_envs=256`, `rollout_length=400`) and **`reward_shaping_horizon=160e6`**
(≈half the run — 32× the paper's fixed `5e6`, so dense shaping stayed on for the
first ~1,562 updates). `cross_play_weight=0.5`. Read from `metrics.jsonl`
(`Train/base_return`, the sparse delivery return).

**Verdict: it learns, and it has not levelled off — undertrained, not converged.**

- Binned `base_return` climbs **monotonically to the last bin**: ~48 (post-warmup)
  → ~56 (shaping fading) → **58 → 60 → 64 → 70 → 74** across the sparse-only second
  half. End state ≈ **72 mean / 95 peak**, well under the paper's ~163 self-play
  reference on this layout — large headroom.
- Late slope is decelerating but still clearly positive: ~+10.6 return / 1k updates
  over the last quarter, ~+3.7 / 1k over the last 10% (≈+5%/1k relative). Per-update
  reads are noisy (last-200 swings 55–92); the binned means are the signal.
- **Healthy exploration signal:** `base_return` kept *rising* after dense shaping
  annealed to zero at update ~1,562 (54 → 74 across the sparse-only half), so the
  policy learned the true sparse task rather than a shaping artifact — the whole
  reason shaping was added. Separation (SP−XP) at `cross_play_weight=0.5` was not
  read from this file and is still the open adoption gate.

**Action:** train ~5× longer. The committed `total_timesteps=1.8e9` (17,578 updates,
≈5.6×) is the right target; the mild deceleration means it may plateau somewhat
before the full 5×, but `320e6` is nowhere near converged. Extending to `1.8e9`
also drops the `160e6` shaping horizon to ~9% of the run (closer to the paper's
proportion), so most of training is the pure sparse task the curve already improves
on. Re-read separation at the higher budget before adopting.

### Full-budget run: BRDiv Counter Circuit, `1.8e9` — converged, competent *and* diverse

The full committed budget (`total_timesteps=1.8e9`, `num_envs=256`, 17,578 updates,
`cross_play_weight=0.5`), H100, ~3 days. **This is the run the probe above pointed to,
and it lands the paper-matching.**

**Converged.** `base_return` (sparse task) climbs monotonically and flattens: by
decile ~36 → 56 → 65 → 76 → 89 → 97 → 96 → **99**, peak 130; last-quarter slope
−0.04/1k (flat). Mostly converged by ~70% of the budget, so a shorter run would have
sufficed — but it is genuinely plateaued.

**Both failure modes of the pre-paper run are gone.** The final scored population:

| metric | this run (`1.8e9`, matched) | pre-paper-match (`§ above`) |
|---|---:|---:|
| SelfPlay | **87.0** | 28.0 |
| CrossPlay | **0.05** | 26.55 |
| Separation (SP−XP) | **86.95** | 1.45 |

The old run was *both* incompetent (SP 28) *and* homogeneous (XP≈SP → separation ~1.4,
no real diversity). This one is competent (SP 87) *and* genuinely diverse (members
coordinate with themselves but essentially not with each other, XP 0.05). BRDiv's
best-response-diversity objective is now actually satisfied — which validates the
CNN+GRU + annealed shaping + Table-4 backbone as the fix.

**What this still cannot conclude:**

- **SP 87 is well below the paper's ~163 single-policy self-play**, and the run peaked
  at 130 before settling to 99 — the signature of the diversity/competence tradeoff:
  `cross_play_weight=0.5` pushes members apart and pulls individual self-play down.
- **XP ≈ 0 may be *too* diverse.** Separation is still the unvalidated proxy (Known-open):
  the real objective is downstream ego generalization, unmeasurable until `ppo_br` is
  absorbed. A maximally-incompatible population (XP~0) maximises separation but could be
  a *worse* ego-training set than a moderately-diverse one (the ego never sees mutually
  coordinatable partners). **Do not treat `cross_play_weight=0.5` as adopted** — sweep it
  lower (0.1, 0.3) to see whether a little separation buys back a lot of SP, and validate
  against downstream ego return, not separation, before adopting.

### Correction: the SP-87 run trained on *inverted* shaping (a bug, since fixed)

A later BRDiv run at the reference budget (`3e7`, ~229 updates, matching ICRL4AHT's
`train_brdiv_overcooked_v2.py`) collapsed to a do-nothing policy — `base_return` stuck at
the random floor, `Train/shaped_reward` a flat 0. The cause was a real bug: `brdiv.py`
folded the annealed shaped reward into the reward **before** the conf/br diversity
transform (`_compute_rewards`), so in the ~80% of pairings that are cross-play the
`lambda x: -x` **negated the shaping** — the sub-tasks it exists to teach (pot placement,
dish pickup) were *punished*, driving the policy to never act. A random policy triggers
shaping (~6 reward over 600 steps in a standalone check); the trained policy triggered
none. Fixed by adding shaping **after** the transform, as an always-positive skill bonus
(only the sparse task return carries the diversity `+/-`). BRDiv-only: L-BRDiv/CoMeDi
apply diversity via loss weighting, not a reward-sign flip, so they don't share it (though
whether their loss weighting has an analogous effect on shaped advantages is unverified).

**This reframes the SP-87 result above.** That run reached SP 87 *despite* the inverted
shaping — it used stronger, longer shaping (`horizon=160e6`), higher LR (`5e-4`), and 60×
the budget, enough to grind through the adverse gradient. So SP 87 is not a clean
paper-matched result; it is what brute force bought around a bug. **Re-run at the reference
budget with the fix before trusting any Overcooked-v2 BRDiv number** — competence should
now come from the shaping doing its job, not from budget.

## BRDiv × Hanabi — the recurrent actor is the gate; the separation is (probably) the ZSC floor

First converged Hanabi run for any paired generator. Hand-written config (not yet in
`gen_teammate_configs.py`); Hanabi is otherwise untuned (see "Not yet tuned").

### The memoryless plateau, and the fix

On the inherited default actor (`actor_with_conditional_critic`, non-recurrent), BRDiv
Hanabi flatlines at **~3.5 / 25** and stays there for 6× the budget — a hard capability
ceiling, not under-training. Hanabi hides your own hand and carries coordination in the
history of hints/plays/discards, so a memoryless policy plays a few safe cards and no more.
Switching to **`actor_type="rnn_actor_with_conditional_critic"`** (the flat-obs GRU sibling
of the v2 CNN+GRU actor) lifts it to **~11.5** — the single biggest change, and the Hanabi
analogue of the v2 "wrong network" finding: partially-observable coordination needs memory.

### `cross_play_weight` is not the SP lever (and is entangled with it)

Dropping `cross_play_weight` 0.5 → 0.05 did **not** raise self-play — it came out slightly
*lower* (~10 vs ~11) and slower. Two reasons: (1) the ~11 ceiling is a capability/tuning
limit of the actor + inherited PPO, not diversity pressure, so relaxing diversity can't lift
it; (2) `sp_weight = (1 + 2·cross_play_weight)·(n/2)` (`brdiv.py:462`), so lowering cw also
**down-weights the self-play gradient** (`n → 0.55n` from 0.5 → 0.05). The knob sets both
the diversity pressure and how hard the matched pair trains; it is not a clean SP dial.
Higher SP is a PPO/architecture tuning problem, not a cw or budget one.

### Budget: converges by ~15–17k updates

`base_return` and the eval-SP curve both plateau by update **~15–17k**; at
`rollout_length=128 × num_envs=1024 = 131,072` steps/update that is **~2–2.6e9 timesteps**.
The `1e10` runs were ~4× past convergence and bought nothing (the last ~40k updates flat).
**Adopted `total_timesteps=3e9`** (~22.9k updates) — converged with margin. Express the
target as **~20k updates**, not raw timesteps: the schedules (LR cosine, eval cadence) key
off `num_updates`, so changing `num_envs`/`rollout_length` re-derives the timestep count.

### The converged run (`cw=0.5`, `rnn` actor, `3e9`)

`rnn_actor_with_conditional_critic`, `cross_play_weight=0.5`, `num_envs=1024`,
`rollout_length=128`, `lr=5e-4`, `num_minibatches=4`, `update_epochs=4`, `total_timesteps=3e9`:

| SP | XP | separation |
|---:|---:|---:|
| 11.55 | 0.003 | 11.55 |

Eval SP peaked 11.84 (update ~17k), settled 11.43. Competent-ish, maximally separated.

### What this cannot conclude — and the check that would

The separation (11.55, XP≈0) looks spectacular but **is not yet evidence BRDiv's diversity
mechanism did anything**: in Hanabi, independently-trained self-play agents generically
cross-play ~0 (the zero-shot-coordination problem — the reason Other-Play/OBL exist). So
XP≈0 may be free, not earned, inflating separation relative to what the same number means on
LBF/Overcooked. This is the sharpest instance of the "separation is an unvalidated proxy"
known-open.

**The control**: `configs/hanabi/teammate_gen/brdiv_selfplay_control.json` — the identical
config with **`cross_play_weight=0.0`** (diversity objective off, everything else fixed).
Train it and read `Population/CrossPlay`:

- **XP_control ≈ 0** (≈ the cw=0.5 XP): removing diversity pressure didn't change cross-play,
  so XP≈0 is the generic Hanabi ZSC floor and the 11.55 separation is mostly free. Report
  BRDiv Hanabi as "competent-ish teammates; separation is the ZSC gap, not a BRDiv effect."
- **XP_control ≫ 0**: without diversity pressure the independent self-play policies coordinate,
  so `cw=0.5` genuinely drove XP to 0 → the separation is real and attributable to BRDiv.

Prior from the ZSC literature is that XP_control ≈ 0 (generic), but it is untested here; the
control settles it with one single-variable comparison. A corroborating control is an FCP
Hanabi run (independent self-play by construction) with `actor_type="rnn"` — note the plain
`rnn`, **not** the conditional-critic variant, which FCP cannot build (no population index to
condition on). Competence (SP ~11.5) is the separate open lever (PPO/architecture tuning).

## Offline BC × Hanabi — not broken, data-limited (and the RAM wall that hid it)

This is an offline-baseline (§3.1) entry, not teammate generation, but it is the
sharpest "what the sweep could *not* conclude at first" case in the file, so it
belongs here.

**The scare.** Trained on the pooled `expert` Hanabi dataset (25k episodes), every
offline baseline scored ~0.2–0.4 return against held-out teammates (ceiling ~19.7),
and BC on a *single* clean convention (comedi:0 self-play) reached only ~24% of that
convention's own self-play return — while the identical trainer/config reached ~84%
of ceiling on LBF. It read like BC/DT fundamentally cannot do offline Hanabi.

**What it actually was, ruled out in order.** Reporting units (the summary mixed a
normalized RTG target with a raw return), RTG conditioning (flat across target
quantiles — the model ignores RTG), sampling vs argmax (argmax no better), action
leakage (causal mask; logits read off the `o_t` tokens), train/deploy window
misalignment (training left-pads with absolute timesteps, matching deploy — verified
by an inference-parity probe: training-forward and deploy `get_action` argmax agree),
and over-training (early-stopping made it *worse*). What remained, measured cleanly
with a train-seed/eval-seed split and the *saved* normalization:

- a **generalization gap** — 1.00 teacher-forced accuracy on the exact training
  deals vs ~0.80 on held-out deals (the model memorized 2k distinct trajectories),
  and
- a **closed-loop compounding** cost — even at 100% teacher-forced accuracy the ego
  reached only ~51% of ceiling in closed loop, because one wrong action desyncs the
  recurrent partner. Held-out closed-loop is the two stacked.

**Both scale away with data.** Doubling then quintupling the single-convention data
(the compounding runs in reverse — fewer per-step errors → less drift):

| episodes | held-out per-step acc | held-out closed-loop (sampled teammate) |
|---:|---:|---:|
| 2,000 | 0.80 | ~30% of ceiling |
| 4,000 | 0.84 | ~65% |
| 10,000 | 0.89 | **~95%** |

At 10k the in-dist/held-out gap collapses (21 → ~0 pts) and in-dist teacher-forced
*drops* 1.00 → 0.98 — memorization giving way to generalization. **Conclusion: the
offline trainer works; Hanabi (hidden hand, ~70-step horizon) is data-hungry, and
per-convention cloning needs ~10k trajectories.** The LBF-inherited offline config
(`context_length=20`, `hidden_dim=32`) caps at ~4%; the config that clears the gate
is `context_length=80` (full episode), `hidden_dim=128`, `num_blocks=3`, `ff_dim=256`,
`stage2_batch_size=32`, `stage2_steps=60000` — `configs/hanabi/training/pooled_expert_scaled/`.

**What it could not conclude, and the tooling the answer needed.** The 25k pooled
set gave only ~2.1k episodes per train pairing (12 pairings, `holdout_per_generator=2`
over 20 self/conf teammates) — 5× under the ~10k bar — so the pooled numbers were
*doubly* handicapped (undersampled per convention *and* un-conditioned BC's pooling
incoherence). Scaling the pooled collection to 150k (~12.5k/pairing) is the open
test, and it needed the pipeline to stop being RAM-bound: `LazyWindows`
(`stream_windows`) builds windows on demand, `DiskEpisodeSource` (`stream_from_disk`)
streams episodes from the vault, and `VaultWriter` collects them in chunks — so
collection *and* training run in bounded host RAM (all three verified byte-identical
to the eager path). Two ceilings will remain even at 150k: **un-conditioned BC is
capped by pooling incoherence** (it cannot be 12 conventions at once at deploy — the
teammate-modelling baselines are what turn per-convention coverage into behaviour),
and **held-out AHT is bounded by the confederate ZSC wall** (a BRDiv/L-BRDiv `conf`'s
only competent partner is its `br`, held out with it — best train-ego over held-out
was ~2.4–3.6). Data cannot fix either; they are protocol properties.

Reproduce the single-convention curve with `scripts/diagnose_single_pairing.py`
(`--episodes N`), `scripts/diagnose_generalization.py` (train/eval-seed
gap), and `scripts/diagnose_inference_parity.py` (deploy faithfulness).

## Offline BC × Hanabi (pooled) — the ego has no best-response, and BC *should* fail

A 150k-episode pooled `expert` retrain (the scale-up the section above called for)
left **both BC and LIAM at ~5% of ceiling** — train-partner return ~1.0, held-out
~0.5 (ceiling ~19.7), against 87% teacher-forced action accuracy — and scaling the
data 25k → 150k moved nothing. It read like a fundamental break. It is not, but the
"pooling incoherence, a protocol property" framing above was too vague; three checks
pin what it actually is.

**The dataset is K distinct per-teammate egos.** The pooled `expert` set is 12 fixed
`(ego, teammate)` pairings, ~2.1k episodes each: 6 self (`ego=mate`, an FCP/CoMeDi
member's self-play) and 6 cross (`ego=br, mate=conf` for BRDiv/L-BRDiv). The ego seat
— the stream BC clones — therefore holds **12 different networks**, one per teammate
(verified: `member_ids[:, ego_index]` has 12 distinct values). Un-conditioned BC has
no teammate signal, so it averages 12 experts into a policy coherent for none; over a
~66-step episode, 87% per-step accuracy is `0.87^66 ≈ 1e-4` chance of an error-free
episode, so it desyncs against every specific partner. Single-convention BC works
(84%) because there is exactly one ego to clone.

**This matches the papers — un-conditioned BC failing here is the expected result,
not a bug.** TAO (§3) generates each controlled-agent trajectory `T^{1,k}` "employing
its approximate best response policy `π^{1,k,*}`" — a *distinct BR per opponent*;
OMIS (Alg.) trains `{BR(π^{-1,k})}_K`, one per opponent. So the multi-ego
construction is *correct*. None of these methods clone un-conditioned; they condition
on the (inferred) opponent. And the field's own AHT-scale benchmark reports the same
failure: **ICRL4AHT's AD/DPT "struggle to consistently outperform a random baseline"**
(DPT 12.4 ± 11.0 vs Random 5.5, and that skewed by a no-coordination teammate
family). (An earlier pass this session claimed the papers use a *single* shared
best-response ego; reading TAO/OMIS refuted it — recorded so it isn't re-derived.)

**The collected data is competent.** Mean ego return in the pooled vault is **12.4**
(population self-play ≈ 11.5), pairings ranging 0.94 (a weak `br`, `ego=3/mate=2`) to
19.76. So the ~1.0 BC eval is the averaging, not bad data.

**Two structural gaps against ICRL4AHT stay open.** (1) Our egos are *not* dedicated
best-responses — they are reused self-play members / BRDiv-internal `br`s, a
non-uniform procedure with no competence guarantee (the 0.94 pairing is the tell).
ICRL4AHT §4.2 fixes the generated teammates and **trains an ego PPO best-response
against each** — the `ppo_br.py` step this benchmark still lacks (Known-open,
CLAUDE.md). (2) They collect *learning histories* (random→expert per teammate) with
quality filtering and optional expert-action relabeling; we collect static expert
only. Which matters depends on the method (opponent-modelling wants expert BR; AD
wants histories).

**The instrument.** `scripts/diagnose_oracle_bc.py` trains the BC backbone with a
per-teammate embedding of the *ground-truth* id (an oracle — it consumes the true
teammate identity, so it is an upper bound, not a deployable method; `num_teammates=0`
leaves plain BC byte-identical) and rolls it closed-loop against each train teammate.
It was meant to separate two hypotheses — oracle ~12 (egos cloneable, the wall is
teammate *inference*) vs oracle ~1 (egos not cloneable, the fix is `ppo_br`).

**The answer is *both*, and it lands in the middle.** Converged (60k steps, 0.886
teacher-forced accuracy, matching plain BC's 0.869), the oracle scores **mean 5.64**
vs un-conditioned pooled BC's **1.06** and the data's own **12.4** competence:

| | mean return | of data (12.4) |
|---|---:|---:|
| un-conditioned pooled BC | 1.06 | ~9% |
| teammate-id oracle (perfect id) | **5.64** | **~45%** |

So (1) teammate identity is a **large** lever — perfect id is **5.3× over BC**, i.e.
teammate modelling has real headroom and BC-on-pooled failing is genuinely mostly the
averaging; but (2) even perfect identity caps a single shared model at **~45% of
competence** — a second ceiling beyond inference. The per-teammate spread shows both:
some egos are cleanly cloneable given the id (lbrdiv:4 → 11.3, comedi:3 → 10.0,
lbrdiv:1 → 10.0, at the data), others resist even trained (fcp:24 → 2.1, comedi:0 →
2.2, fcp:19 → 2.4) — shared-model interference across 12 unrelated egos and/or the
reused egos being harder to clone. (A plausible contributor to the residual: the ~45%
tracks the single-convention **closed-loop compounding tax** — even 100% teacher-forced
reached only ~51% of ceiling above — so part of the gap may be generic drift, not
multi-ego interference.)

**Consequences.** The oracle is the ceiling for *any* teammate-conditioned method on
this dataset: TAO/OMIS cannot exceed ~5.6 here no matter how good their inference, so
running them measures how much of the 1.06 → 5.64 gap realistic inference recovers,
against a dataset ceiling that is itself <50% of competence. Raising that ceiling
toward 12 needs the dataset fixed — **`ppo_br`**: uniform, dedicated best-responses
(ICRL4AHT's fix-teammates-then-train-ego-PPO) address both the ego-quality and the
"12 unrelated networks" interference. Priority order: `ppo_br` to lift the ceiling,
then modelling baselines to approach it. Still unrun: whether TAO/OMIS-strength
conditioning approaches the 5.64 oracle where LIAM (weak, ego-history-only inference —
65% teammate-action reconstruction) sat at BC's floor.

## Not yet tuned

All four on Overcooked-v1 and Hanabi still run at hyperparameters ported
from jax-aht's per-environment Hydra configs. Those encode real tuning and
are a reasonable starting point, but the FCP result above shows what
inheriting them can cost: upstream's LBF budget left the population at 74%
of the achievable food, and CoMeDi's left each member with 160 updates.

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

### Recurrent-actor support added: RPG now covers all five wired environments

RPG was originally hardcoded to `MLPActorCriticPolicy` — no `actor_type`
dispatch at all, unlike every other generator — so it could only run on
plain-MLP environments (LBF, MPE). Adding Hanabi (`"rnn"`) and Overcooked-v2
(`"cnn_rnn"`) support meant more than a policy-class swap: `_rollout`,
`_actor_dice_loss`, `_diversity_surrogate`, and `_base_loss`'s critic term all
called `network.apply(params, (obs, avail))` directly -- the MLP network's
stateless, no-hstate signature, incompatible with a recurrent actor's hidden
state. Every one of those call sites now routes through
`policy.get_action_value_policy(params, obs, done, avail_actions, hstate, rng)`,
the same uniform interface `MLPActorCriticPolicy`/`RNNActorCriticPolicy`/
`CNNRNNActorCriticPolicy` all implement identically -- `RpgConfig`/`RpgRuntime`
gained an `actor_type` field, and `make_rpg_train`/`get_rpg_population` now go
through `initialize_agent`'s dispatch, matching FCP/MEP.

The one design fact that made this tractable: `_rollout` always starts with a
fresh `env.reset()` and is never carried across outer `_update` iterations, so
hidden state is always freshly zero-initialized both for the online rollout
*and* for the later whole-trajectory replay calls (`_actor_dice_loss` etc.) --
no initial hstate needed saving into the trajectory dict. Done/hstate semantics
were mirrored from `marl/ippo.py`'s already-proven convention verbatim rather
than re-derived, specifically to avoid an off-by-one that would silently
corrupt the higher-order (manipulator) gradient rather than crash.

**Verification, in order of how likely each was to catch a real bug:**
1. A new pinned test (`test_actor_dice_loss_replay_reproduces_rollout_log_probs_{mlp,rnn}`)
   asserts that recomputing log-probs via `_actor_dice_loss`'s own network call,
   at the *same* params used to collect, reproduces the trajectory's own
   `lp0`/`lp1` bit-for-bit -- RPG's DiCE objective has no importance-sampling
   ratio against a stored old policy, so this has no reason to differ unless
   the hstate/done threading between the online and replay call sites
   disagrees. Passes for both actor types.
2. Full `tests/` suite (275 tests) passes -- no regression in the mlp path.
3. Live smoke runs on both new actor types (`population_size=2`, tiny budget)
   -- Hanabi (`actor_type="rnn"`) and Overcooked-v2 (`actor_type="cnn_rnn"`) --
   confirm `initialize_agent`/`get_rpg_population`'s dispatch wires correctly
   end-to-end, not just the loss functions in isolation.

Configs for `hanabi`/`overcooked_v2_counter_circuit`/`lbf_20x20`/
`mpe_reference`/`mpe_spread` are now generated (`scripts/gen_teammate_configs.py`),
all still the same **deliberately modest, untuned** budgets as LBF 12×12 --
this only removes the structural blocker, it does not tune anything. The two
open adoption gates above (scale past `n=2`, held-out usability) are
unaffected and still stand for every environment, not just LBF.

## MEP × LBF 12×12 / Hanabi / Overcooked-v2 (wired, untuned; not yet run at scale)

MEP (Zhao et al., AAAI-23 — the teammate-generation method OMIS uses) is now a
fifth generator (`src/oaht_bench/teammate_gen/mep.py`, clean-room from the
paper, `MepConfig`). Stage 1 only: self-play PPO per member plus a
population-entropy reward bonus, `-alpha * log(mean_k pi_k(a_t|s_t))`. MEP's
own Stage 2 (a shared robust-generalist ego via prioritized sampling) is not
built — every other generator here only releases a population, and ego
training stays `ppo_br.py`'s job.

Architecturally distinct from every other generator here: computing the
entropy bonus needs every population member's *current* policy visible at
every rollout step, which neither FCP's fully-independent per-member vmap nor
BRDiv/L-BRDiv's dual-role gather/scatter machinery provide cleanly. The
population vmap axis is *named* (`axis_name="population"`), and
`jax.lax.all_gather` is used as a collective to gather every lane's current
params into every lane for a forward-pass-only cross-member log-prob query —
no shared environment interaction, unlike BRDiv/CoMeDi's cross-play rollouts.
The reduction itself must be `logsumexp(log_probs) - log(N)` (`log(mean(p))`),
not `mean(log(p))` — those differ by Jensen's inequality whenever members
disagree, and the wrong one silently trains a different objective. Pinned by
`tests/unit/teammate_gen/test_mep.py`'s isolated reduction test, plus an
end-to-end smoke test and a behavioral ablation (Question 2 from the paper: a
large `population_entropy_coef` measurably raises trained population entropy
relative to a `coef=0` control on the same tiny CPU-sized job — confirmed).

**Open, named assumption: the hstate reset for Hanabi/Overcooked-v2.** MEP's
objective is written as `pi(a|s)` — state-conditioned, no history in its
formalism, because the paper's own environment (Overcooked) is fully
observed. Hanabi ("rnn") and Overcooked-v2 ("cnn_rnn") need recurrent actors,
which the paper's math doesn't address. For the cross-member forward pass
*only* (never each member's own rollout), hstate is reset to
`policy.init_hstate(...)` rather than reusing each member's own
trajectory-hstate against another lane's observation — the closest reading of
the paper's literal state-only conditioning, but an extension the paper
doesn't specify. Doesn't affect LBF ("mlp": `init_hstate` is already the
no-op hstate ippo.py's own rollout uses there), so LBF's first sweep is
unaffected by this open question; revisit once a Hanabi/Overcooked-v2 run
exists to check whether the assumption produces sensible entropy curves.

Configs are **untuned starting points** on all three families (see
`scripts/gen_teammate_configs.py`'s `PPO["mep"]`/`SCALE["mep"]`): LBF starts
at `PpoHyperparams`' bare defaults (a blank slate, deliberately not inheriting
any other generator's tuned numbers) with a modest budget matching RPG's own
first-LBF-budget precedent (`total_timesteps=1e7`, `num_envs=64`); Hanabi and
Overcooked-v2 copy the shared recurrent-actor backbone every other generator
already inherits on those families verbatim (not MEP-specific tuning). Not
yet run at LBF's real budget — the first thing a real run should check is
whether the population-entropy bonus produces a genuinely diverse,
non-collapsed LBF population (self-play ≈ cross-play expected, since MEP is
non-adversarial — see the cross-play sanity check AD-RPG's own paper runs,
`papers/rpg.pdf` Fig. 5/6) before anything past LBF is worth attempting.

## Backbone cleanup, and fixing OMIS's training to match its reference

An investigation into why the offline baselines (LIAM, MeLIBA, TAO, OMIS)
cluster near BC on lbf_20x20 (see the section on the identification bottleneck
above) had a second, narrower thread: whether each baseline's *training
procedure* — staged vs. joint, frozen vs. not — actually matches its paper and
its reference code, since that would explain why OMIS/TAO don't show the
relative edge over LIAM/MeLIBA their papers claim, independent of the
identification-bottleneck story. Reading all four papers plus TAO's and OMIS's
released code (`papers/{liam,meliba,tao,omis}.pdf`,
`/Users/conorwallace/Documents/Personal/Projects/{TAO,OMIS}`) found:

- **LIAM, MeLIBA: correct as-is.** Both papers' online training uses a single
  combined loss differentiated w.r.t. policy and encoder/decoder together, but
  both apply `jax.lax.stop_gradient` on the embedding/latent right before it
  reaches the policy (`liam_agent.py:536`, `meliba_agent.py:941` in
  `LARG/jax-aht`) — mathematically identical to our offline two-stage,
  frozen-representation split. No change.
- **TAO: our implementation matches the paper; the released code does not.**
  TAO's Appendix C, Algorithm 1 is unambiguous that stage 2 backpropagates to
  the decoder only, encoder frozen — what we do. But
  `offline_stage_2/{train,nn_trainer}.py` never loads
  `ENCODER_PARAM_PATH` (defined, unused) and trains a *fresh* encoder jointly
  with the decoder — contradicting its own paper and Algorithm 1. Likely a
  release bug (missing `encoder.load_model(...)` call), not an intentional
  design change. Left alone: our TAO is the paper's TAO, which is what we want
  to claim, and `freeze_encoder=False` already exists as an explicit opt-out
  reproducing the released code's joint training if that's ever needed for
  comparison (`offline/tao.py`, pinned by
  `test_tao_stage_two_freezes_the_encoder_by_default`).
- **OMIS: our implementation genuinely diverges from its reference.**
  `pretraining/nets.py::GPTModel` is one GPT-2 backbone with three
  *separately-owned* linear heads (`predict_action`, `predict_value`,
  `predict_oppo_action`); `pretraining/nn_trainer.py`'s `train_step` sums them
  into one loss (`act_loss + vf_coef*value_loss + oppo_pi_coef*oppo_pi_loss`)
  and backpropagates through the whole thing in one pass — confirmed by ONNX
  inspection of a released checkpoint (one graph, three outputs, one trunk).
  The paper's §4.1 is consistent with this (one sequence through one shared
  backbone) and never claims a staged/frozen split. Our prior implementation
  split OMIS into a frozen-representation stage (imitator + critic) and a
  policy stage (actor) conditioned on the frozen output — LIAM/MeLIBA's
  pattern, not OMIS's. Fixed this session; see below.

**Fix, part 1 — the shared backbone loses its baked-in head.**
`models/backbone.py`'s `DecisionTransformer` returned `(logits, obs_hidden)`
unconditionally; every encoder role (`LiamEncoder`, `MelibaEncoder`,
`OmisEncoder`) discarded `logits`, every network/actor role discarded
`obs_hidden` and just forwarded the baked-in head — none of them owned an
action head of their own, unlike either reference (TAO's separate
`GPTEncoder`/`GPTDecoder`; OMIS's one trunk with owned heads). The backbone now
returns `obs_hidden` alone and is renamed `GPT2Model` (the "Decision
Transformer" framing was only true while it baked in the action head); LIAM,
MeLIBA, TAO, and BC's network/actor classes each now attach their own
`nn.Dense(action_dim)`. Behavior-preserving for all four — pinned by
`test_liam_stage_two_does_not_differentiate_the_encoder` and
`test_tao_stage_two_freezes_the_encoder_by_default` passing unchanged, plus the
full suite.

**Fix, part 2 — OMIS: one backbone, three heads, trained jointly.**
`models/omis_agent.py`'s `OmisEncoder` + `OmisActor` (two separate backbone
calls, one feeding the other, output frozen between them) is now
`OmisBackbone` + `OmisHeads` (one backbone call; three sibling heads — actor,
opponent imitator, critic — all reading the same `obs_hidden`, matching
`nets.py::GPTModel` structurally). `offline/omis.py`'s
`omis_representation_loss` + `omis_actor_loss` collapsed into one
`omis_joint_loss` (`act_loss + vf_coef*value_loss + oppo_pi_coef*oppo_pi_loss`,
naming and defaulting the coefficients to the reference's own
`args.vf_coef=0.5`/`args.oppo_pi_coef=0.8`, replacing the old
single-purpose `value_coef=1.0` field on `OfflineTrainingConfig`). No
`stop_gradient` anywhere in the loss.

`BaseAhtTrainer`'s `train_stage_1() -> train_stage_2(stage1_params)` contract
is shared with LIAM/MeLIBA/TAO and stayed untouched at the interface level,
but for OMIS both calls now optimise the *same* `{backbone, heads}` parameter
tree against the *same* joint loss — `train_stage_2` continues from
`train_stage_1`'s checkpoint at `stage2_learning_rate`/`stage2_steps` rather
than training a fresh policy against a frozen encoder. This was chosen over a
literal "stage 2 is a no-op pass-through" so that the generic runner tests
(`test_runner_trains_and_writes_parameters`,
`test_runner_logs_accuracies_and_evaluation_returns`, parametrized across all
four baselines) keep seeing real `Stage1/`/`Stage2/` metrics without special-
casing OMIS — and it is arguably more faithful to "one continuous joint
optimization" than an empty second call would have been. `OmisAgent.act` and
`mate_action_logits` read the final, most-trained parameters from `stage2`
exclusively.

**Verification.** A new targeted test,
`test_omis_all_three_heads_reach_the_shared_backbone`, differentiates each
head's own loss term in isolation and confirms all three reach
`OmisBackbone`'s parameters — the direct check for the property this fix
exists for, not just an end-to-end smoke test that could pass even if a head
were accidentally cut off. Full suite (276 tests) passes. A live smoke run
(20+20 steps on the real `pooled_expert_lbf_12x12` dataset) trains,
checkpoints, and evaluates end-to-end, and its `metrics.jsonl` shows
`Stage1/{action_accuracy,imitator,imitator_accuracy,critic,loss}` and the same
under `Stage2/` — all three heads' losses present and moving at every logged
step of both calls, not just the first.

**Still open, unchanged by this fix:** decision-time search
(`omis_search`, `testing/search.py`'s `fake_env` rollout) remains
unimplemented; deployed OMIS is still `OMIS w/o S`. This fix makes that
ablation's *training* match the reference exactly — it does not add search.

## Pooled crossplay redefined: `ppo_br` is now the only ego axis, plus a `weighted` variant

Follow-on from the training-procedure investigation above: OMIS/TAO's own
reference pipelines build a cross-play corpus (every ego against every
opponent) specifically so their in-context encoder learns to identify a
teammate independent of who it's paired with. Our pooled dataset-collection
variants didn't — `expert` seated each teammate against one fixed ego (its
best *existing* roster responder, or a separately-loaded dedicated best
response via a collection-time override), `mixed` against exactly two fixed
egos, both picked deterministically off a matrix whose ego axis was the
*original*, not-specifically-trained population policies — capping realistic
competence at ~45% (see the FCP× / BRDiv× sections above) and giving zero
cross-partner exposure per teammate.

**Redefinition, not an addition.** Per review direction, the pooled crossplay
matrix's ego axis is now *exclusively* the trained `ppo_br` population — the
original population's `self`/`conf`/`br` policies are never egos again, only
teammates (`self`/`conf`; a paired generator's own `br` role is now unused
entirely). Since `ppo_br` trains exactly one dedicated best response per
teammate identity, ego and teammate share one `K`-sized index space and the
matrix is a genuine square `K x K`, every cell measured (row `i`= teammate
`i`'s best response, column `j` = teammate `j`) — real cross-play data, not
just a diagonal. `expert`/`mixed`/`br_vs_worst` read off this matrix through
the *same* unmodified `plan_seatings` machinery as before (it was always
roster-index-oblivious to what a position means); the old collection-time BR
override (`_collect_pooled`'s `br_egos is not None` branch) is deleted as
dead weight — `expert`'s argmax now lands on the diagonal through the same
code path as everything else, not a special case.

**New `weighted` variant**: draws the ego *per episode*, i.i.d., from
`softmax(matrix[egos, j] / temperature)` over raw returns, instead of a fixed
discrete-band argmin pick. `temperature -> 0` recovers `expert`'s argmax;
`temperature -> large` approaches a uniform draw over every teammate's
dedicated best response. This is what actually gives a teammate broad,
continuous ego exposure across its episodes, closer to what OMIS/TAO's own
prompt corpus does — `expert`/`mixed` still only ever put a teammate in front
of one or two fixed egos even after this redefinition, since the *matrix* is
richer now but their sampling was never designed to spread across it.
`br_population_path` narrowed from `list[str]` to `str`: one `ppo_br` run
already covers the whole released roster (`load_br_egos`'s own docstring
already said so), so the list type and its per-path merge loop were solving
a problem that didn't exist.

**Migration cost, not yet paid.** `configs/lbf_12x12/{crossplay/pooled,
data_collection/pooled_{expert,mixed,worst,weighted}}.json` all now require
`br_population_path`, filled in with a `TODO_FILL_IN` placeholder pointing at
`configs/lbf_12x12/ppo_br.json`'s run directory — that `ppo_br` job has not
been run yet (no `results/teammate_generation/ppo_br_lbf_12x12-*` exists),
so neither has a matrix recomputation under the new schema. The existing
`populations/lbf_12x12/pooled_crossplay.npz` (the old, roster-as-ego, 30-wide
matrix behind the already-collected `pooled_expert_lbf_12x12-6a931eee466c`
dataset OMIS/LIAM/MeLIBA/TAO were trained on above) is untouched and still
loadable by old code paths, but is incompatible with the new schema's
required `br_egos`/square-roster invariant — the four configs above point at
a fresh `new_pooled_crossplay.npz` instead rather than colliding with it.
Unrun: training `ppo_br` for lbf_12x12, recomputing the matrix, and picking a
first real `temperature` for `weighted` (default `0.2`, unvalidated — pick a
value via the realised per-teammate ego-diversity spread once real data
exists, not asserted here).

Verification so far is unit-only: `plan_weighted_seatings`'s temperature
limits (`temperature to 0` concentrates on the argmax, `temperature` large
approaches uniform), `evaluate_pooled`'s new square-1:1 invariant (raises on
a `br_egos`/teammate mismatch), and `teammate_roster`'s role filter, plus the
full suite (283 tests). No live rollout has exercised this yet — that's the
next thing to run, not a claim being made here.

## Evaluation targets are now per teammate, from the crossplay matrix — every baseline, same mechanism

Follow-on from the same investigation. Every baseline (BC, LIAM, MeLIBA, OMIS,
TAO) conditioned on one dataset-wide return-to-go target
(`dataset_target_return`'s single best per-episode return, identical for
every teammate a policy was rolled against) — TAO's own reference computes
something more specific, `OPPO_TARGET[i] = max over egos of that ego's mean
return against teammate i`, a per-teammate value. That's exactly a
column-max over a crossplay-style matrix, and the pooled crossplay matrix
above already computes one for a different reason (dataset collection) —
its ego axis is now the trained `ppo_br` population, so a teammate's column
max there already *is* `OPPO_TARGET` for that teammate. Nothing new to
compute.

Resolved per direction from review: apply this uniformly to every baseline
(no TAO-specific branch) rather than trading faithfulness against
cross-baseline comparability. `offline.evaluate.resolve_target_returns` reads
a per-teammate target off `pooled_matrix_path` when a dataset was collected
in pooled mode, falling back to today's single `dataset_target_return` value
(broadcast to every teammate) for legacy/single-population datasets with no
matrix. `ReturnConditionedAgent.set_target_return` lets the eval loop change
the conditioning target between teammates (previously baked in once at
construction); `evaluate_agent_against`/`evaluate_incontext` call it right
before each teammate's rollout, so BC/LIAM/MeLIBA/OMIS/TAO all go through the
identical mechanism — TAO's `evaluate_incontext` path needed the exact same
one-line addition as the parallel path, no special-casing.

**Verified live** (not just unit tests): reran a small OMIS training job
against `pooled_expert_lbf_12x12-6a931eee466c` (whose meta already carries
`pooled_matrix_path`, from before this session's redefinition — an older,
reused-policy-ego matrix, not yet the `ppo_br` one) and confirmed
`training_summary.json`'s `eval.target_returns` now holds twenty *different*
values, one per teammate (range ~2.29–3.07 in normalised units), where every
prior run recorded one shared number. Full suite: 290 tests (283 + 7 new
covering `crossplay_target_returns`/`resolve_target_returns`'s two branches
and `set_target_return`).

**Still stale**: this ran against the *old* matrix (reused-policy egos,
~45%-competence-capped), since `ppo_br` for lbf_12x12 hasn't been trained yet
(see the section above). The per-teammate targets are real and distinct, but
their absolute values will shift once the matrix is recomputed with dedicated
best-response egos — expected to rise, since `OPPO_TARGET` is a ceiling that
mechanism is specifically meant to lift.

## A one-off MEP config approximately replicating OMIS's own LBF population (pop=20)

Follow-on from checking whether OMIS's paper/code specify MEP's own
hyperparameters (they don't — both defer entirely to
`ruizhaogit/maximum_entropy_population_based_training`'s own defaults). This
records the config built to try reproducing their population:
`scripts/gen_omis_lbf_mep_config.py` -> `configs/lbf_9x9/teammate_gen/mep.json`.

**Environment, not just hyperparameters, had to change.** OMIS's LBF (§4,
`omis.pdf`): "a mixed environment in a 9x9 grid world containing two
players... along with five apples," no sight/view restriction mentioned
(full observability), horizon 50 (their own
`horizon_per_ep_dict["lbf"]`). Neither `lbf_12x12` (12x12/6-food) nor
`lbf_20x20` (20x20/4-food, `fov=2`, `rollout_length=128`) matches on any of
grid size, food count, or horizon. `lbf_9x9` (`grid_size=9, num_food=5,
num_agents=2, fov=None, rollout_length=50`) is a new, deliberately
*unregistered* environment — not added to `configs/env.py`'s `_PRESETS`,
since every registered preset is iterated unconditionally by
`gen_teammate_configs.py --all-envs` for all six generators, and
FCP/CoMeDi/BRDiv/L-BRDiv have no tuning entries for a family only MEP uses
here (registering it would hard-`KeyError` that shared script). The config
is schema-valid and hash-stable via the real Pydantic classes + `save_job`,
just generated by its own standalone script rather than folded into the
shared per-environment tuning tables.

**Known, stated limitation**: our LBF is Jumanji's implementation (via
jax-aht), not the `lb-foraging` gym package OMIS's own code actually runs.
`LbfConfig` has no `force_coop`-style knob controlling the solo-vs-cooperative
eating rule the way the original package does, so every *configurable*
parameter matches (grid, food, agents, levels, observability, horizon) but
the underlying game mechanics are not guaranteed bit-identical. "Approximate"
is doing real work in "approximately replicate."

**MEP hyperparameter mapping**, reference repo defaults ->
`PpoHyperparams`/`MepConfig`: `LR=5e-3`, `ENTROPY=0.5` ->
`entropy_coef`, `ENTROPY_POOL=0.1` -> `population_entropy_coef` (was 0.010,
sourced from the MEP/OMIS paper's own Table-1 middle-of-sweep value — this
supersedes that for this one-off config specifically, not the shared
default), `VF_COEF=0.1` -> `value_coef`, `LAM=0.98` -> `gae_lambda`,
`MAX_GRAD_NORM=0.1`, `STEPS_PER_UPDATE=8` -> `update_epochs`,
`MINIBATCHES=5` -> `num_minibatches`, `TOTAL_STEPS_PER_AGENT=1.5e7` ->
`total_timesteps`. `GAMMA=0.99`, `CLIPPING=0.05` -> `clip_eps`, and
`SIZE_HIDDEN_LAYERS=64` -> `hidden_dim` already matched our existing
defaults, no change needed. `population_size=20` is OMIS's own reported
population count (not the reference repo's own default of 4) — deliberately
breaks `gen_teammate_configs.py`'s "population size held equal across
generators" convention (`POPULATION_SIZE=5`), since this population was
never meant to be pooled with the other four generators' 5-member ones.

**Not imported, and why**: the reference's PBT resample/mutate/select-the-
worst-out loop (`RESAMPLE_PROB`, `MUTATION_FACTORS`, `HYPERPARAMS_TO_MUTATE`,
`ITER_PER_SELECTION`, `NUM_SELECTION_GAMES`, `NUM_PBT_ITER`,
`PPO_RUN_TOT_TIMESTEPS`, `TOTAL_BATCH_SIZE`) — confirmed by rereading
`teammate_gen/mep.py` that our implementation is one continuous `vmap`'d
parallel-population PPO run per member with the population-entropy bonus,
no iterative resampling cycle at all, so none of these have a field to map
onto. Also not imported: the reference's Overcooked-specific CNN network
settings (LBF's flat observation uses `MlpNetwork`) and `sim_threads`/
`MINIBATCHES` (-> `num_envs`/`ppo.num_minibatches`, left at our own defaults
rather than force-matched — see the correction below for why `MINIBATCHES`
specifically could not be ported as a raw number).

**Bug found on the real run, fixed.** The first version of this config set
`ppo.num_minibatches=5` (the reference's `MINIBATCHES`), which crashed
immediately on the user's GPU machine:
`cannot reshape array of shape (50, 128) ... [50, 5, -1]` —
`marl/ppo_utils.py::_create_minibatches` requires `num_actors (= num_agents x
num_envs = 2 x 64 = 128)` to divide evenly by `num_minibatches`, and 128 is
not divisible by 5. `num_minibatches` is not an independently-portable
optimization hyperparameter like LR/entropy/GAE — it's a batch-structure
knob mechanically coupled to `num_envs`, and the reference's value of 5 was
sized against *their* batch structure (`TOTAL_BATCH_SIZE=20000`,
`sim_threads=50`), never a number that could transfer to ours. Same category
as `sim_threads`/`num_envs`, which were already correctly left unmatched —
this one should have been too, and wasn't. Fixed by leaving
`num_minibatches` at `PpoHyperparams`' own default (4), which divides 128
evenly.

**The smoke test that shipped with the first version didn't catch this.**
It used a hand-simplified config (`num_envs=8, num_minibatches=2`, its own
divisibility accidentally fine: `16 % 2 == 0`) rather than the real
generated config's own values (`num_envs=64, num_minibatches=5`) — so it
verified "the environment constructs and MEP trains" but not "this exact
config's hyperparameters are mutually compatible." Re-verified properly this
time: a smoke run using the *actual* `num_envs=64`/`num_minibatches=4` the
real config now has (population_size and total_timesteps trimmed for speed,
nothing else) completes cleanly. Lesson for next time: a smoke test's
purpose is to exercise the values that will actually ship, not a
structurally-different stand-in that happens to avoid the same reshape.

**Still not verified**: the real run — population_size=20 at
total_timesteps=1.5e7 each is a real budget decision, left for later, not
run here.

### Second bug: `rollout_length=50` never set the episode horizon at all

After the `num_minibatches` fix, the user ran real training and reported
`percent_eaten`/`returns` stagnant around 0.07–0.09 with
`Train/returned_episode_lengths` reading exactly `100.0` on every non-zero
log line — despite the config's `rollout_length=50`, meant to reproduce
OMIS's `horizon_per_ep_dict["lbf"]=50`. The logs showed a clean pattern:
alternating `train_step`s with all-zero completion metrics, then all-100.0
completions.

Root cause: `rollout_length` on `EnvConfigBase` only sizes the PPO
rollout-collection scan for one training update
(`teammate_gen/mep.py`/`runtime.py`) — it is never passed into
`LbfConfig.env_kwargs()`, so it never reaches the environment constructor at
all. Jumanji's `LevelBasedForaging` therefore always ran its own default,
`time_limit=100`, regardless of what `rollout_length` said. `lbf_20x20`'s own
preset notes already flagged this ("Episode horizon stays at Jumanji's 100
... time_limit is not a config knob here") from an earlier session, but that
was read as a Jumanji limitation rather than double-checked — it wasn't:
`LevelBasedForaging.__init__` accepts `time_limit` directly
(`jumanji/environments/routing/lbf/env.py:117`), and it isn't blocked by
`make_env`'s own kwarg filtering (`env_kwargs.py`'s `process_default_args`
only intercepts generator/viewer keys, and `time_limit` is neither) — it was
simply never wired up on our side. Since the real horizon (100) was exactly
2x the configured `rollout_length` (50), every other PPO rollout window fell
entirely inside one ongoing episode (no `done`, all-zero logged metrics) and
the other half straddled the episode boundary (`done` for every env at once,
`returned_episode_lengths=100.0` uniformly) — exactly the alternating
pattern reported. Not a training failure; a logging/config-plumbing artifact
on top of a config that silently wasn't doing what its own comment claimed.

Fixed by adding `LbfConfig.time_limit: int | None = None` (`configs/env.py`),
forwarded to Jumanji's constructor only when set — `None` preserves the
previous default exactly, so `lbf_12x12`/`lbf_20x20` are unaffected (verified
live: both still resolve to `time_limit=100`). `lbf_9x9`'s config now sets
`time_limit=50` explicitly; `rollout_length=50` is left as-is (a reasonable
PPO scan size on its own terms, not because it ever controlled episode
length). Verified live end-to-end: `make_env(..., {'time_limit': 50, ...})`
constructs an env whose episodes actually terminate at step 50, not 100.

**Separately, on "barely 0.07 returns after 2000 steps" being evidence of a
learning failure**: at `rollout_length=50`/`num_envs=64`, 2000 raw env steps
is ~40 PPO updates against a `total_timesteps=1.5e7` budget (~4,700 updates
per member) — under 1% of the run. Too early to read as non-learning on its
own; worth re-checking once a run at the corrected `time_limit=50` has
progressed substantially further, separately from the horizon bug above.

### Third bug: `population_entropy_coef=0.1` (the reference repo's raw default) makes MEP unable to learn LBF at all

To settle whether the flat-returns result above was a hyperparameter issue or
an implementation issue, the user re-ran MEP with its PPO block matched
exactly to the already-tuned FCP config (`entropy_coef=0.01, learning_rate=
0.001, max_grad_norm=0.5, update_epochs=15` — none of the reference repo's
own PPO values), same env, same seed, same population size and budget, so
`generator` was the only real difference. Result:
`Train/returned_episode_returns` for FCP climbed from ~0.04 at step 0 to
~0.47–0.49 (task ceiling) by ~2000 updates; MEP stayed at ~0.01–0.04 the
whole ~4650-update run and, if anything, drifted down. The population
cross-play matrix confirmed it at the population level: FCP's off-diagonal
entries are ~0.3–0.5, MEP's are ~0.003–0.04 — an order of magnitude apart at
identical PPO hyperparameters, generator held as the only difference.

**Root cause: not the log-sum-exp reduction (verified correct, matches the
paper, and is unit-tested), but the scale of `population_entropy_coef`.**
LBF's reward is normalized so a whole *episode's* maximum possible task
return is 1.0 (0.5 shared per agent, `normalize_reward=True` in Jumanji's
`LevelBasedForaging`). The population-entropy bonus (`population_entropy_
bonus()` in `teammate_gen/mep.py`) is added to *every single step's* reward
regardless of task performance. At `coef=0.1` (the reference repo's raw
`ENTROPY_POOL` default, which this script had explicitly set to match the
repo rather than the paper) and a near-uniform 6-action population
(representative of early training), the bonus computes to `~=0.179` per
step, `~=8.96` summed over a 50-step episode — **~18x the entire episode's
max achievable task reward.** Since PPO's advantage is computed from
`reward + entropy_bonus` (`_env_step` in `teammate_gen/mep.py`), the
gradient signal at that scale is almost entirely "look different from the
population's mean policy," not "forage well" — consistent with both the
flat/non-learning return curve and the near-zero self-play/cross-play
separation seen earlier (members degrading to similarly low-competence,
mutually-indistinguishable-in-task-terms policies chasing the same novelty
signal, rather than diverging into genuinely diverse *competent* ones).

At `MepConfig`'s own default (`0.010`, the MEP/OMIS paper's own Table-1
middle-of-sweep value, which this script had overridden away from), the same
calculation gives `~=1.79x` the max episode reward — a real nudge, not a
signal that erases the task. `ENTROPY_POOL=0.1` was tuned against
Overcooked's much denser, per-event, unnormalized reward (`SOUP_PICKUP_
REWARD=1.0` etc. firing many times per 400-step episode) and was never
validated against a reward this sparse — same "does not port" category as
`MINIBATCHES`/`sim_threads` (see above), just far more consequential when
mis-set: total learning failure rather than weaker diversity.

**Fix**: `scripts/gen_omis_lbf_mep_config.py` no longer overrides
`population_entropy_coef` away from `MepConfig`'s own default — the config
now trains at `0.010`. `configs/lbf_9x9/teammate_gen/mep.json` regenerated
(hash `8588a96bd16a`). A local smoke run (population_size=3,
total_timesteps=5e4) still trains/saves/scores cleanly at the new value.

**Not yet verified**: the theoretical ~1.79x ratio at `0.010` is a plausible
nudge, not a proof that MEP will now learn LBF well — the real test is a
full GPU run at the corrected value, not run here. If it still fails to
learn, the scale analysis above at least rules out "the reference's raw
default was fine" as an explanation and narrows where to look next.
