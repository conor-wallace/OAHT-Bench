# OAHT-Bench

A survey and benchmark for **offline ad-hoc teamwork**, targeting AAMAS 2027.
Four teammate generators (FCP, CoMeDi, BRDiv, L-BRDiv) across three environments
(LBF 12×12, Hanabi, Overcooked-v2), feeding
offline datasets and a shared decision-transformer backbone.

`docs/offline_aht_benchmark_project.md` is the plan. `docs/baseline_specs.md`
holds per-paper extractions. `docs/tuning_record.md` is a **contribution, not an
appendix** (§7.2) — record what a sweep concluded *and what it could not*.

## Layout

All code lives under `src/oaht_bench/`. Most of it is **absorbed** from
jax-aht (MIT, commit `0885df95c386121b9c94cb0fb516531895e29702`) rather than
depended on, because jax-aht and ICRL4AHT claim colliding top-level package
names. See `PROVENANCE.md`.

**Never reformat absorbed code.** Keeping the absorbed files untouched preserves
their diff against the recorded upstream commit (`PROVENANCE.md`); whitespace
churn destroys that. Ruff is configured with an *allowlist* of the ~22 files we
authored (`[tool.ruff] include` in `pyproject.toml`). An absorbed file is
untouched by default. The four generators (`fcp.py`, `comedi.py`, `brdiv.py`,
`lbrdiv.py`) are absorbed-but-heavily-modified and stay excluded for the same
reason.

## Conventions the user has established

- **Pinned dependencies only.** Never an editable local path — this is a
  benchmark, so it inherits from a pinned jax-aht.
- **Pydantic models for every config parameter.** One JSON per experiment,
  single CLI entry routing on `job_type`:
  `uv run oaht-bench config=configs/my_experiment.json`.
- **Configs are generated**, not hand-edited. `scripts/gen_teammate_configs.py`
  is the source of truth; tuned values live there with a comment saying why.
  Hand edits are lost on regeneration.
- **Never commit a personal wandb entity.** `--wandb` writes the project but
  never the entity; wandb takes that from `WANDB_ENTITY` or the local login.
- **Commit only when asked.** Branch first if on the default branch.
- `results/`, `wandb/`, `configs/sweeps/`, `papers/` are gitignored. The
  provenance record is the config whose hash names the run directory, not the
  output.

## Invariants that have already cost real runs

Each of these was a multi-hour loss. They are pinned by tests; do not relax them
without reading the test's docstring.

1. **Save the checkpoint before reporting.** BRDiv/L-BRDiv had
   `save_train_run` as the last statement of `log_metrics`, behind charting calls
   and a `raise ValueError` on NaN. A `wandb.Table` width mismatch destroyed a
   17-hour run. All four now save first, and reporting is wrapped in
   `common.logging.nonfatal`. Reporting must never be able to raise into
   training.
2. **Evaluate with sampled actions, never argmax.** Two argmax policies in a
   symmetric coordination task are perfectly correlated and deadlock — on LBF
   every episode ran to the 100-step limit at 25% of the food, reporting 0.11
   for a population whose training curve read 0.41. It also erases policy
   entropy, making `entropy_coef` unmeasurable in a sweep. `evaluation_greedy`
   defaults `False` and should stay there.
3. **FCP is scored on converged checkpoints only.** Its population spans
   competence *by design*; averaging self-play over all members inverts the
   tuning signal and drives `num_checkpoints → 1`, reproducing the `FCP₋T`
   ablation the paper reports as significantly worse.
4. **BRDiv/L-BRDiv need `num_envs` ∝ n².** They draw `conf_id` and `br_id`
   independently per environment, so a specific pairing gets `num_envs / n²`
   samples. The loss *weighting* is population-size invariant (verified) — the
   *data per pairing* is not, and that is what binds. Encoded as
   `_paired_scale()`.
5. **Capture a reference run before refactoring, then diff.** This caught
   several bugs that static checks and import tests missed.

## Commands

```bash
uv sync --extra dev
uv run python -m pytest tests/ -q                       # 270 tests, ~8min
uv run ruff format . && uv run ruff check . --fix

uv run python scripts/gen_teammate_configs.py --wandb   # regenerate configs
uv run python scripts/sweep.py run configs/lbf_12x12/teammate_gen/ --jobs 2
uv run python scripts/sweep.py generate --base ... --set path=v1,v2
uv run python scripts/sweep.py collect --sweep configs/sweeps/<name>
uv run python scripts/sweep.py rescore results/teammate_generation/<run>/
```

`rescore` recomputes SP/XP from saved checkpoints without retraining — use it
whenever the *measurement* changes rather than the training.

## Known-open, deliberately unfixed

- **Separation (SP−XP) is an unvalidated proxy.** The real objective is
  downstream ego-agent generalization, unmeasurable until `ppo_br.py` is
  absorbed. Driving self-play to the task ceiling is well founded; optimizing
  separation past that is a bet. Weakest for FCP, whose diversity is meant to
  come from checkpoint spread rather than an entropy knob.
- **Overcooked BRDiv/L-BRDiv do not fit any GPU.** The n² scaling gives 384
  envs × 1040-float obs × 400-step rollouts ≈ 11.9 GiB of observations alone.
  The rule is right; it cannot be paid for with environments on that task.
  Needs stratified id sampling or longer rollouts instead.
- **The content hash includes fields that do not affect the artifact** —
  `LoggingConfig`, `evaluation_episodes`, `evaluation_greedy`, and `label`.
  Toggling wandb therefore retrains a population into a new directory; so does
  renaming a sweep (`--name`) while re-sweeping cells you already have — this
  cost a full retrain of two CoMeDi cells before it was caught. Scoping the
  hash to training-determining fields is the fix; until then, keep `--name`
  stable when extending an existing grid.
- **Checkpoints are written twice.** `save_load_utils.REPO_PATH` resolves to
  `src/oaht_bench` rather than the repo root, so every run also lands in
  `src/oaht_bench/results/`. Verified byte-identical. `rescore.artifact_dir()`
  handles both layouts.
- **FCP releases 25 members where the others release 5.** `population_size=5`
  equalizes the *scored* population, not the *released* one, because FCP
  snapshots during training. Cutting `num_checkpoints` to 1 would equalize it
  and would be the `FCP₋T` ablation. Open (§7.3).
- **`sweep.py rescore` had zero test coverage and was silently broken for all
  four generators** until fixed this session — it called `.params` on a plain
  `(params, population)` tuple. No regression test exists for the CLI path;
  add one before trusting further changes to `population/rescore.py`.
- **`collect`'s default `evaluation_episodes=20` is too coarse to resolve SP
  differences below roughly 0.01.** Confirmed by re-scoring two independently-
  trained CoMeDi checkpoints at identical hyperparameters: their 20-episode
  gap (0.0117) shrank to 0.0050 at 100 episodes. Re-score sweep finalists at
  higher `--episodes` before reading a ranking off `collect`'s table.
- **BRDiv's adopted `cross_play_weight=0.10` cell has the best self-play score
  of any tested but the worst `% food`** of the top three in that sweep — the
  two metrics agree everywhere else in `docs/tuning_record.md` but disagree
  here, unexplained. Worth resolving before leaning on this population
  downstream.

## State

Branch `docs/project-plan-rev7`. **All four generators are now tuned on LBF
12×12**, the reference environment:

| generator | adopted | SP | separation |
|---|---|---:|---:|
| FCP | `learning_rate=1e-3`, `entropy_coef=0.003`, `total_timesteps=24e6`, `num_envs=64` | ~0.48 | at ceiling (~97% food) |
| CoMeDi | `learning_rate=5e-4`, `total_timesteps_per_iteration=1.92e8`, `cross_play_weight=0.2` | 0.472 | 0.217 |
| BRDiv | `cross_play_weight=0.10`, `entropy_coef=0.003`, `total_timesteps=5.4e8` | 0.386 | 0.271 |
| L-BRDiv | `tolerance_factor=0.03`, `entropy_coef=0.003`, `total_timesteps=5.4e8` (shared with BRDiv) | 0.396 | 0.211 |

Each row is the end of its own chase, with rejected alternatives worth reading
before re-deriving them — full detail in `docs/tuning_record.md`. Two things
worth knowing before touching these further:

- **BRDiv and L-BRDiv plateau well under FCP's SP** (~0.39 vs ~0.48). Six
  PPO/budget variations were tried on BRDiv alone before concluding this looks
  like a limit from the n²-divided training data and the diversity
  constraint itself, not an undertuned knob.
- **Every adoption call used competence-first-then-separation**, consistently
  — including turning down a CoMeDi `cross_play_weight` cell that traded 9%
  SP for 58% more separation, a far better ratio than BRDiv's own sweep ever
  offered. Separation is still an unvalidated proxy (see Known-open); if
  downstream validation ever suggests it's underweighted, that's a change to
  the adoption rule, not a case for bending it after the fact case-by-case.

Overcooked-v1 and Hanabi are untuned for all four generators (still on jax-aht's
inherited hyperparameters) — Overcooked-v1 BRDiv/L-BRDiv additionally don't fit
any GPU yet at all (see Known-open).

**Overcooked-v2 is now matched to its source paper** (Gessler et al., ICLR 2025)
for all four generators: annealed dense-reward shaping (`reward_shaping_horizon`),
a CNN+GRU actor (`cnn_rnn` / `cnn_rnn_actor_with_conditional_critic`, App. C.1.1),
LR warmup+cosine (`lr_warmup`), the Table-4 PPO backbone, and `negative_rewards=True`.
Wired across `models/cnn_rnn_actor_critic*.py`, `marl/{reward_shaping,lr_schedule}.py`,
`initialize_agents.py`, `population/loading.py`, all four generators, and the
generated v2 configs. **Verified end-to-end on CPU** (all four train, checkpoint
*and* score) and by `tests/unit/teammate_gen/test_overcooked_v2_paper_matching.py`;
**not yet GPU-validated** — the BRDiv Counter Circuit run sweeping `cross_play_weight`
around 0.5 is the real check. `cross_play_weight` stays our knob (the paper doesn't
train BRDiv). Full detail + what it can't yet conclude in `docs/tuning_record.md`.

Next: the dataset schema (§4) and the DT backbone (§3.1) — LBF tuning was the
gate for starting this, and it's now clear. Overcooked/Hanabi tuning can
follow the same playbook once the schema work starts consuming LBF
populations for real and there's a concrete reason to need it sooner.
