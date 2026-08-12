# OAHT-Bench

A survey and benchmark for **offline ad-hoc teamwork**, targeting ICML 2027
(deadline ~late January 2027). Four teammate generators (FCP, CoMeDi, BRDiv,
L-BRDiv) across three environments (LBF 12×12, Overcooked-v1, Hanabi), feeding
offline datasets and a shared decision-transformer backbone.

`docs/offline_aht_benchmark_project.md` is the plan. `docs/baseline_specs.md`
holds per-paper extractions. `docs/tuning_record.md` is a **contribution, not an
appendix** (§7.2) — record what a sweep concluded *and what it could not*.

## Layout

All code lives under `src/oaht_bench/`. Most of it is **absorbed** from
jax-aht (MIT, commit `0885df95c386121b9c94cb0fb516531895e29702`) rather than
depended on, because jax-aht and ICRL4AHT claim colliding top-level package
names. See `PROVENANCE.md` and `scripts/absorb_upstream.py`.

**Never reformat absorbed code.** `absorb_upstream.py` exists so that re-running
it against a newer upstream shows what drifted; whitespace churn destroys that.
Ruff is configured with an *allowlist* of the ~22 files we authored
(`[tool.ruff] include` in `pyproject.toml`). A newly absorbed file is untouched
by default. The four generators (`fcp.py`, `CoMeDi.py`, `BRDiv.py`, `LBRDiv.py`)
are absorbed-but-heavily-modified and stay excluded for the same reason.

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
uv run python -m pytest tests/ -q                       # 196 tests, ~50s
uv run ruff format . && uv run ruff check . --fix

uv run python scripts/check_device.py [config.json]     # GPU + memory preflight
uv run python scripts/gen_teammate_configs.py --wandb   # regenerate configs
uv run python scripts/sweep.py run configs/teammate_gen/lbf_12x12/ --jobs 2
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
  `LoggingConfig`, `evaluation_episodes`, `evaluation_greedy`. Toggling wandb
  therefore retrains a population into a new directory. Scoping the hash to
  training-determining fields is the fix.
- **Checkpoints are written twice.** `save_load_utils.REPO_PATH` resolves to
  `src/oaht_bench` rather than the repo root, so every run also lands in
  `src/oaht_bench/results/`. Verified byte-identical. `rescore.artifact_dir()`
  handles both layouts.
- **FCP releases 25 members where the others release 5.** `population_size=5`
  equalizes the *scored* population, not the *released* one, because FCP
  snapshots during training. Cutting `num_checkpoints` to 1 would equalize it
  and would be the `FCP₋T` ablation. Open (§7.3).

## State

Branch `docs/project-plan-rev7`. Only **FCP × LBF** is tuned; the other eleven
(generator, environment) pairs still run jax-aht's inherited hyperparameters,
and the FCP result shows what that can cost — upstream's LBF budget left the
population at 74% of achievable food, and CoMeDi's left each member 160 updates
against the 2,929 FCP needed.

Next: confirm BRDiv recovers at `num_envs=192` (check `Eval/AvgSPReturnCurve`
exceeds `Eval/AvgXPReturnCurve` — if not, the n² diagnosis is wrong and
`cross_play_weight` is the suspect; a grid for it sits unrun at
`configs/sweeps/brdiv_lbf_xpw`). Then the dataset schema (§4) and the DT
backbone (§3.1).
