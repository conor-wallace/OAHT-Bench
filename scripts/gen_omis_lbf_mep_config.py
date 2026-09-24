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
(`OMIS/pretraining/utils.py::horizon_per_ep_dict["lbf"] = 50`), not either of
our existing LBF families' `rollout_length=128`.

Caveat, stated once rather than repeated: our LBF is Jumanji's implementation
(via jax-aht), not the original `lb-foraging` gym package OMIS actually uses.
`LbfConfig` has no `force_coop`-style knob to control the solo-vs-cooperative
eating rule the way the original package does, so this matches every
*configurable* parameter exactly, not necessarily the underlying game
mechanics bit-for-bit.

MEP hyperparameters are the reference implementation's own defaults
(`ruizhaogit/maximum_entropy_population_based_training`,
`human_aware_rl/pbt/pbt_model_pool_entropy_parallel.py`), mapped onto our
`PpoHyperparams`/`MepConfig` fields where an analogous field exists.
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
        rollout_length=50,  # OMIS's own horizon_per_ep_dict["lbf"]
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
        population_entropy_coef=0.1,  # reference ENTROPY_POOL
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
