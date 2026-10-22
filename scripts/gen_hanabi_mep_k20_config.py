"""Emit a one-off MEP config on Hanabi with population_size=20, so its
crossplay heatmap is directly comparable in scale to the existing pooled
20x20 BRDiv/CoMeDi/FCP/L-BRDiv figure already in the paper -- the same
"single generator, K=20" format used for the LBF-9x9 MEP/CoMeDi replication
(scripts/gen_omis_lbf_mep_config.py).

Motivation: CoMeDi's LBF-9x9 crossplay did NOT collapse to a sparse matrix
the way BRDiv/L-BRDiv's does on Hanabi (see fig:hanabi-crossplay's own
caption: "BRDiv and L-BRDiv in particular are essentially pure diagonal").
That means LBF-9x9 may simply be too easy/small a task for an adversarial
objective's cost to show up in downstream AHT difficulty -- every
population, adversarial or not, ends up "diverse enough." The converse
test is more informative: does MEP's own non-adversarial population-entropy
bonus fail to produce comparably sharp diversity on a task (Hanabi) where
the adversarial generators already do?

Everything here matches the existing `configs/hanabi/teammate_gen/mep.json`
exactly (env spec, and the shared recurrent-actor PPO backbone already
validated for BRDiv/CoMeDi/L-BRDiv/FCP on Hanabi: num_envs=1024,
learning_rate=5e-4, max_grad_norm=0.5, update_epochs=4,
total_timesteps=3e9) -- population_size is the only intentional change, from
GeneratorBase's default (5) to 20.

`population_entropy_coef` is NOT left at MepConfig's own default (0.010,
the paper's own Table-1 value, tuned for Overcooked's reward scale) --
LBF-9x9 already showed that value silently blocks all learning there, and
Hanabi's reward (a small integer score, ceiling 25, BRDiv's own validated
self-play at 11.55 as the realistic competence anchor) is yet another
different scale from both Overcooked and LBF. Ran a local, small-scale
(num_envs=16, population_size=3, total_timesteps=2e6) entropy-bonus-on-vs-
off comparison, the same methodology that found LBF's real value, before
trusting any number here:

  FCP baseline:        tail-avg return 3.44
  MEP, coef=0:          tail-avg return 3.47 (matches FCP, confirms scaffolding)
  MEP, coef=0.002:      tail-avg return 3.47, separation 0.21 -- full competence, real diversity
  MEP, coef=0.005:      tail-avg return 2.50 (-28%), separation 0.90 -- diversity at a real cost

`0.002` is used here rather than the larger, higher-separation `0.005`
deliberately: this experiment is testing whether MEP's diversity mechanism
underperforms Hanabi's adversarial generators (BRDiv/L-BRDiv, whose
crossplay is already near-pure-diagonal in the paper), and a coefficient
that pays for its separation with competence loss would confound that
question the same way the CoMeDi-vs-MEP LBF-9x9 comparison was confounded
by CoMeDi's population being individually less competent -- see that
section's correction in docs/tuning_record.md. A result at a competence-
preserving coefficient is unconfounded either way: if separation stays
weak at the real K=20/3e9-step scale too, that is evidence MEP's mechanism
underperforms here, not evidence it merely traded competence differently.

Usage::

    uv run python scripts/gen_hanabi_mep_k20_config.py
"""

from __future__ import annotations

from pathlib import Path

from oaht_bench.configs import save_job
from oaht_bench.configs.env import HanabiConfig
from oaht_bench.configs.job import TeammateGenerationJob
from oaht_bench.configs.teammate_gen import MepConfig, PpoHyperparams

REPO_ROOT = Path(__file__).resolve().parents[1]


def build() -> TeammateGenerationJob:
    env = HanabiConfig(
        name="hanabi",
        num_colors=5,
        num_ranks=5,
        hand_size=5,
        max_info_tokens=8,
        max_life_tokens=3,
        num_cards_of_rank=(3, 2, 2, 2, 1),
        rollout_length=128,
        tier="tier1",
        notes="Turn-based, action-masked, hidden own hand. The abstraction stress test.",
    )

    ppo = PpoHyperparams(
        learning_rate=0.0005,  # matches the shared Hanabi backbone (BRDiv/CoMeDi/L-BRDiv/FCP)
        max_grad_norm=0.5,
        update_epochs=4,
        anneal_lr=True,
        clip_eps=0.2,
        gamma=0.999,
        # entropy_coef, value_coef, gae_lambda, num_minibatches: unchanged from
        # PpoHyperparams' own defaults, matching the existing mep.json exactly.
    )

    generator = MepConfig(
        actor_type="rnn",
        num_envs=1024,  # matches the shared Hanabi backbone
        total_timesteps=3e9,  # matches the shared Hanabi backbone
        population_size=20,  # the one intentional change -- see module docstring
        # population_entropy_coef: NOT MepConfig's own 0.010 default -- see
        # module docstring. 0.002 is validated by a local smoke test to
        # preserve full competence (tail-avg return matches FCP/coef=0)
        # while still producing real, non-zero separation (0.21).
        population_entropy_coef=0.002,
        ppo=ppo,
    )

    return TeammateGenerationJob(
        label="mep_hanabi_k20",
        env=env,
        generator=generator,
    )


def main() -> int:
    job = build()
    path = REPO_ROOT / "configs" / "hanabi" / "teammate_gen" / "mep_k20.json"
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
