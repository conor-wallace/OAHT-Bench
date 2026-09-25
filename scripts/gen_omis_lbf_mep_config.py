"""Emit the one-off MEP config that approximately replicates OMIS's own LBF
opponent population (Jing et al., NeurIPS 2024), for trying to reproduce
their results.

Two things this deliberately does NOT do:

* **Register a new environment preset.** `lbf_9x9` below is not added to
  `configs/env.py`'s `_PRESETS` registry. Every registered preset is iterated
  unconditionally by `gen_teammate_configs.py --all-envs` for all six
  generators, and FCP/CoMeDi/BRDiv/L-BRDiv have no PPO/SCALE table entries
  for a family nobody but MEP uses here -- registering it would land a bare
  `KeyError` in that shared script for a family it was never meant to cover.
  "Experiment JSON files inline a full env config" already (see
  `configs/env.py`'s own module docstring), so the job doesn't need a
  registered preset to be valid.
* **Touch `gen_teammate_configs.py`'s shared PPO/SCALE tables.** This config
  is a one-off replication target, not part of the maintained per-environment
  tuning matrix those tables encode.

Environment: OMIS's paper (Sec. 4, `papers/omis.pdf`) states LBF is "a mixed
environment in a 9x9 grid world containing two players... along with five
apples" -- no sight/view restriction is mentioned anywhere, so full
observability. Horizon comes from their own code
(`OMIS/pretraining/utils.py::horizon_per_ep_dict["lbf"] = 50`), set via
`LbfConfig.time_limit` (forwarded straight to Jumanji's own
`LevelBasedForaging(time_limit=...)`), NOT `rollout_length` -- the first
version of this script set `rollout_length=50` believing that controlled
episode length, which is wrong: `rollout_length` only sizes the PPO
rollout-collection scan window per training update and never reaches the
environment constructor at all (`configs/env.py`'s own `LBF_20X20` preset
notes already said as much -- "time_limit is not a config knob here" -- but
that referred to our wrapper never plumbing it through, not a real Jumanji
limitation; `LevelBasedForaging.__init__` accepts `time_limit` directly,
defaulting to 100). Left unfixed, every episode silently ran Jumanji's
default 100 steps instead of 50, and because 100 is exactly 2x the
`rollout_length=50` PPO scan window, every other rollout-collection window
straddled zero episode completions while the other half completed a full
100-step episode -- the alternating all-zero / all-100-length pattern seen
in a real training log is that artifact, not a training failure. Fixed by
adding `LbfConfig.time_limit` (new field, default `None` = Jumanji's
unchanged default, so `lbf_12x12`/`lbf_20x20` are untouched) and setting it
here explicitly. `rollout_length` is left at 50 too since it's a reasonable
PPO scan size on its own, not because it does anything to episode length.

Caveat, stated once rather than repeated: our LBF is Jumanji's implementation
(via jax-aht), not the original `lb-foraging` gym package OMIS actually uses.
`LbfConfig` has no `force_coop`-style knob to control the solo-vs-cooperative
eating rule the way the original package does, so this matches every
*configurable* parameter exactly, not necessarily the underlying game
mechanics bit-for-bit.

MEP hyperparameters are the reference implementation's own defaults
(`ruizhaogit/maximum_entropy_population_based_training`,
`human_aware_rl/pbt/pbt_model_pool_entropy_parallel.py`), mapped onto our
`PpoHyperparams`/`MepConfig` fields where an analogous field exists --
**except `population_entropy_coef`, see below.**

`population_entropy_coef` is deliberately NOT the reference's raw
`ENTROPY_POOL=0.1`, and (after a second finding below) not `MepConfig`'s own
`0.010` default either -- `0.001`, an empirically validated value, specific
to this config. A controlled side-by-side run (MEP vs FCP, identical PPO
hyperparameters, differing only in generator) showed MEP never learning the
task at all while FCP converged to ~0.49/0.5 (near the task ceiling) within
~2000 updates, at `coef=0.1`. The mechanism itself checks out against the
paper (log-sum-exp reduction, not mean-of-log, per
`population_entropy_bonus`'s own docstring and unit test) -- the bug looked
like scale, not logic. LBF's reward is normalized so a whole *episode's*
maximum possible task return is 1.0 (0.5 shared per agent); the entropy
bonus is added every single *step* regardless of task performance. At
`coef=0.1` and a near-uniform 6-action population (representative of early
training), the bonus is `-0.1 * log(1/6) ~= 0.179` per step, `~=8.96`
summed over a 50-step episode -- ~18x the entire episode's max achievable
task reward. At the paper's own `0.010`, the same snapshot calculation gives
`~=1.79x` the max episode reward, which read as a plausible nudge.

**That `0.010` estimate turned out to be unreliable.** A real run at
`0.010` (PPO block still the reference's own, not FCP's) showed no
improvement either. That wasn't yet a controlled test -- two variables
differed from FCP's known-working config at once. A local CPU diagnostic
held everything fixed to FCP's exact PPO block, varying only
`population_entropy_coef` (full run-by-run table in
`docs/tuning_record.md`): `coef=0` reproduces FCP's learning curve exactly
(numerically identical trajectories -- confirms MEP's training scaffolding,
gradient flow, and `TrainState` threading are all correct, ruling out a
"loss isn't propagating" bug); `coef=0.01` stays completely flat for a full
1000-update run; `coef=0.001` learns, tracking FCP's curve shape closely. A
shape/reduction/gradient-isolation audit of `population_entropy_bonus` and
its call site (`_env_step`) found no code bug. The `t=0`-snapshot magnitude
estimates above undersell the bonus's real, sustained in-training effect:
the mechanism is self-reinforcing (whichever member is last to commit to a
good action keeps earning more bonus for staying different from an
increasingly confident population), so it does not shrink toward
convergence the way an ordinary reward term does -- a closed-form estimate
at initialization is not a reliable predictor of its real magnitude;
`0.001` was found by direct empirical trial, not derived. `MepConfig`'s own
`0.010` default is left alone (a legitimate citation of the paper's value
for environments this hasn't been contradicted on); this script's override
is specific to LBF's reward scale. `ENTROPY_POOL=0.1` was tuned against
Overcooked's much denser, unnormalized, per-event reward
(`SOUP_PICKUP_REWARD=1.0` etc., many events per 400-step episode) and was
never validated against a reward this sparse; it belongs on the same "does
not port" list as `MINIBATCHES`/`sim_threads` below, just far more
consequential when mis-set (total learning failure, not just weaker
diversity).
Deliberately NOT imported: the reference's PBT resample/mutate/select-the-
worst-out loop (`RESAMPLE_PROB`, `MUTATION_FACTORS`, `HYPERPARAMS_TO_MUTATE`,
`ITER_PER_SELECTION`, `NUM_SELECTION_GAMES`, `NUM_PBT_ITER`,
`PPO_RUN_TOT_TIMESTEPS`, `TOTAL_BATCH_SIZE`) -- our `teammate_gen/mep.py` is
one continuous vmap'd parallel-population PPO run with the population-entropy
bonus, no iterative resampling cycle, so there is no field to map these onto.
Also not imported: the reference's Overcooked-specific CNN network settings
(LBF's flat observation uses `MlpNetwork`, already `hidden_dim=64` by
default, matching the reference's `SIZE_HIDDEN_LAYERS`) and `sim_threads`/
`MINIBATCHES` (-> our `num_envs`/`ppo.num_minibatches`, left at our own
defaults rather than force-matched). Both are batch-structure knobs, not
independent optimization hyperparameters: `num_minibatches` must evenly
divide `num_actors = num_agents * num_envs` (`marl/ppo_utils.py`'s
`_create_minibatches` reshapes on it), and the reference's own value of 5 was
sized against *their* batch structure (`TOTAL_BATCH_SIZE=20000`,
`sim_threads=50`), not ours -- copying the raw number crashed at
`num_actors=128 % num_minibatches=5 != 0` on a real run (caught after the
first version of this script shipped; see `docs/tuning_record.md`). Left at
`PpoHyperparams`' own default (4), which does divide 128 evenly, rather than
picked ad hoc to "look similar" to 5.

``population_size=20`` is OMIS's own reported population count (not the
reference repo's own default of 4), and deliberately breaks this project's
"population size held equal across generators" convention
(`gen_teammate_configs.py`'s ``POPULATION_SIZE = 5``) -- this population was
never meant to be pooled with the other generators' 5-member populations for
cross-generator comparison.

Usage::

    uv run python scripts/gen_omis_lbf_mep_config.py
"""

from __future__ import annotations

from pathlib import Path

from oaht_bench.configs import save_job
from oaht_bench.configs.env import LbfConfig
from oaht_bench.configs.job import TeammateGenerationJob
from oaht_bench.configs.network import MlpNetwork
from oaht_bench.configs.teammate_gen import MepConfig, PpoHyperparams

REPO_ROOT = Path(__file__).resolve().parents[1]


def build() -> TeammateGenerationJob:
    env = LbfConfig(
        name="lbf_9x9",
        grid_size=9,
        num_food=5,
        num_agents=2,
        different_levels=True,
        fov=None,  # full observability -- the paper mentions no sight restriction
        rollout_length=50,  # PPO scan window size -- does NOT set episode length, see module docstring
        time_limit=50,  # OMIS's own horizon_per_ep_dict["lbf"] -- this is what actually sets it
        tier="debug",  # one-off replication target, not part of the tiered benchmark matrix
        notes=(
            "Approximately replicates OMIS's (Jing et al., NeurIPS 2024) own LBF "
            "setup for a population-replication experiment: 9x9 grid, 2 agents, 5 "
            "food, full observability, horizon=50 (their own "
            "horizon_per_ep_dict['lbf']). Our LBF is Jumanji's implementation (via "
            "jax-aht), not the lb-foraging gym package OMIS actually uses -- no "
            "force_coop-style knob exists here to match the solo-vs-cooperative "
            "eating rule exactly, so this matches every configurable parameter, not "
            "necessarily the underlying game mechanics bit-for-bit. Not a "
            "registered preset (see scripts/gen_omis_lbf_mep_config.py); not part "
            "of the tiered benchmark matrix."
        ),
    )

    ppo = PpoHyperparams(
        learning_rate=5e-3,  # reference LR
        entropy_coef=0.5,  # reference ENTROPY (per-agent PPO entropy bonus)
        value_coef=0.1,  # reference VF_COEF
        gae_lambda=0.98,  # reference LAM
        max_grad_norm=0.1,  # reference MAX_GRAD_NORM
        update_epochs=8,  # reference STEPS_PER_UPDATE
        # num_minibatches: NOT the reference's MINIBATCHES=5 -- see module
        # docstring. Left at PpoHyperparams' own default (4), which divides
        # num_actors=128 (num_envs=64 x num_agents=2) evenly; 5 does not.
        # gamma=0.99, clip_eps=0.05 already match PpoHyperparams' own defaults.
    )

    generator = MepConfig(
        population_size=20,  # OMIS's own population count -- see module docstring
        total_timesteps=1.5e7,  # reference TOTAL_STEPS_PER_AGENT
        # population_entropy_coef: neither the reference's ENTROPY_POOL=0.1 nor
        # MepConfig's own 0.010 default -- see module docstring. Both looked
        # plausible from a t=0 magnitude estimate and both empirically blocked
        # all task learning in a controlled local run; 0.001 was the value
        # found, by direct trial, to actually let the population learn.
        population_entropy_coef=0.001,
        network=MlpNetwork(),  # hidden_dim=64 already matches SIZE_HIDDEN_LAYERS
        ppo=ppo,
    )

    return TeammateGenerationJob(
        label="mep_lbf_9x9_omis_replication",
        env=env,
        generator=generator,
    )


def main() -> int:
    job = build()
    path = REPO_ROOT / "configs" / "lbf_9x9" / "teammate_gen" / "mep.json"
    save_job(job, path, minimal=True)
    print(f"wrote {path}  (hash {job.short_hash()})")
    print(
        f"population_size={job.generator.population_size}  "
        f"total_timesteps={job.generator.total_timesteps:.1e}  "
        f"population_entropy_coef={job.generator.population_entropy_coef}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
