"""One-off diagnostic config: MEP on lbf_9x9 with its PPO block set to exactly
match the already-tuned FCP config on the same environment, isolating
whether MEP's population-entropy mechanism itself is broken.

Two runs so far have each held only one variable fixed, never both together:

* PPO matched to FCP (entropy_coef=0.01, learning_rate=0.001,
  max_grad_norm=0.5, update_epochs=15), but population_entropy_coef=0.1 (the
  reference repo's raw default) -- MEP never learned; later analysis showed
  the bonus was ~18x LBF's whole per-episode max task reward at that
  coefficient (docs/tuning_record.md).
* population_entropy_coef corrected to 0.010 (the paper's own Table-1 value,
  now scripts/gen_omis_lbf_mep_config.py's default), but the PPO block left
  at the reference repo's own values (entropy_coef=0.5, learning_rate=5e-3,
  max_grad_norm=0.1, update_epochs=8) -- still didn't learn. entropy_coef=0.5
  is 50x FCP's 0.01 and is a completely standard PPO regularizer, unrelated
  to MEP's diversity mechanism; on its own it's a plausible explanation for a
  policy that never commits to a low-entropy forage-and-load strategy,
  independent of anything MEP-specific.

Neither run isolates "does MEP's population-entropy mechanism work at all,
holding every PPO hyperparameter equal to a generator (FCP) we already know
solves this task." This script builds exactly that: reuses
gen_omis_lbf_mep_config.py's environment (9x9, 5 food, time_limit=50) and
population_entropy_coef=0.010, but replaces the PPO block with FCP's own
tuned values verbatim (read from fcp_lbf_9x9_omis_replication-05a1f067d0e7/
job.json). If this still fails to learn, that's a real, load-bearing signal
of an actual implementation bug, not a hyperparameter-transfer problem --
worth then diffing mep.py's per-lane train_state/optimizer construction and
the all_gather-based entropy bonus much more closely. If it learns
comparably to FCP, the reference repo's raw PPO defaults (LR, entropy_coef,
max_grad_norm, update_epochs) simply don't transfer to LBF, same category as
num_minibatches and population_entropy_coef already found not to.

Not part of the tiered benchmark matrix; not registered in
configs/env.py's _PRESETS; written next to the canonical replication config
under a clearly diagnostic filename so it's never mistaken for it.

Usage::

    uv run python scripts/gen_omis_lbf_mep_ppo_matched_diagnostic.py
"""

from __future__ import annotations

from pathlib import Path

from oaht_bench.configs import save_job
from oaht_bench.configs.job import TeammateGenerationJob
from oaht_bench.configs.network import MlpNetwork
from oaht_bench.configs.teammate_gen import MepConfig, PpoHyperparams

from gen_omis_lbf_mep_config import build as build_replication_job

REPO_ROOT = Path(__file__).resolve().parents[1]


def build() -> TeammateGenerationJob:
    replication_job = build_replication_job()

    # FCP's own tuned PPO block on this exact environment, verbatim from
    # fcp_lbf_9x9_omis_replication-05a1f067d0e7/job.json -- not the reference
    # repo's values, and not this repo's LBF-12x12 FCP tuning either (a
    # different environment); this is FCP's actual behavior on lbf_9x9,
    # which is the only thing worth holding MEP equal to here.
    ppo = PpoHyperparams(
        learning_rate=0.001,
        entropy_coef=0.01,
        value_coef=0.1,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        update_epochs=15,
        # clip_eps=0.05, num_minibatches=4, gamma=0.99 already match.
    )

    generator = MepConfig(
        population_size=replication_job.generator.population_size,
        total_timesteps=replication_job.generator.total_timesteps,
        population_entropy_coef=0.010,  # the now-corrected default, held fixed
        network=MlpNetwork(),
        ppo=ppo,
    )

    return TeammateGenerationJob(
        label="mep_lbf_9x9_ppo_matched_to_fcp_diagnostic",
        env=replication_job.env,
        generator=generator,
    )


def main() -> int:
    job = build()
    path = REPO_ROOT / "configs" / "lbf_9x9" / "teammate_gen" / "mep_ppo_matched_diagnostic.json"
    save_job(job, path, minimal=True)
    print(f"wrote {path}  (hash {job.short_hash()})")
    print(
        f"population_entropy_coef={job.generator.population_entropy_coef}  "
        f"ppo={job.generator.ppo.model_dump()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
