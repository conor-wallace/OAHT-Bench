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
