"""Emit teammate-generation configs for each (generator, environment) pair.

Hyperparameters are ported from jax-aht's per-environment Hydra configs rather
than invented. Those encode real tuning — Hanabi wants `gamma=0.999` and a much
larger budget, Overcooked wants a larger `clip_eps` and entropy coefficient than
LBF — and discarding it to start from defaults would throw away working settings
and make the first runs uninformative.

**These are starting points, not the tuned configuration.** §7.2 of the project
plan makes the per-environment tuning record a contribution; this script produces
the baseline that record will be built against, and every value here should be
treated as provisional until a sweep says otherwise.

Regenerate with::

    uv run python scripts/gen_teammate_configs.py            # tier 1
    uv run python scripts/gen_teammate_configs.py --all-envs # all seven
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from oaht_bench.configs import get_preset, preset_names, save_job
from oaht_bench.configs.job import LoggingConfig, TeammateGenerationJob
from oaht_bench.configs.network import MlpNetwork
from oaht_bench.configs.teammate_gen import (
    BrDivConfig,
    CoMeDiConfig,
    FcpConfig,
    LBrDivConfig,
    PpoHyperparams,
    RpgConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
# Configs are laid out environment-first: configs/<env>/<step>/<name>.json.
CONFIGS_ROOT = REPO_ROOT / "configs"


#: Which environment family a preset belongs to, for looking up tuning below.
def _family(preset_name: str) -> str:
    # Checked before the general "overcooked" prefix -- v1 and v2 are
    # different environments (partial observability, a generalized recipe
    # system, no shared absorbed code; see PROVENANCE.md) and must not
    # silently share PPO/budget tuning just because they share a name prefix.
    if preset_name.startswith("overcooked_v2"):
        return "overcooked_v2"
    if preset_name.startswith("overcooked"):
        return "overcooked"
    if "hanabi" in preset_name:
        return "hanabi"
    return "lbf"


# --------------------------------------------------------------------------
# PPO settings, per (generator, environment family), from jax-aht's configs.
# --------------------------------------------------------------------------
PPO: dict[str, dict[str, dict[str, Any]]] = {
    "fcp": {
        # Tuned, not inherited — see docs/tuning_record.md. Upstream's 1e-4/0.01
        # left the population well short of the task ceiling.
        "lbf": dict(
            learning_rate=1e-3,
            update_epochs=15,
            num_minibatches=4,
            clip_eps=0.03,
            entropy_coef=0.003,
        ),
        "overcooked": dict(
            learning_rate=1e-3,
            update_epochs=15,
            num_minibatches=16,
            clip_eps=0.1,
            entropy_coef=0.05,
        ),
        # UNTUNED, and deliberately NOT copied from v1's "overcooked" entry
        # above (unlike every other family here) -- v2 uses partial
        # observability (agent_view_size=2 on the registered presets) and
        # therefore an RNN policy (actor_type="rnn" in SCALE below), so v1's
        # MLP/full-observability-tuned values aren't a meaningful prior.
        # Starting from upstream's own reference instead: JaxMARL's only
        # validated overcooked_v2 config, baselines/IPPO/config/
        # ippo_rnn_overcooked_v2.yaml (LR, CLIP_EPS, ENT_COEF, ANNEAL_LR).
        # gamma/gae_lambda/max_grad_norm below already match its 0.99/0.95;
        # max_grad_norm differs (theirs is 0.25) and is left at this
        # generator's own default rather than copied blind. See
        # docs/tuning_record.md once a sweep exists.
        "overcooked_v2": dict(
            learning_rate=2.5e-4,
            update_epochs=4,
            num_minibatches=64,
            clip_eps=0.2,
            entropy_coef=0.01,
            anneal_lr=True,
        ),
        "hanabi": dict(
            learning_rate=5e-4,
            update_epochs=4,
            num_minibatches=4,
            clip_eps=0.2,
            entropy_coef=0.01,
            anneal_lr=True,
            gamma=0.999,
            gae_lambda=0.95,
        ),
    },
    "comedi": {
        "lbf": dict(
            learning_rate=5e-4,
            update_epochs=15,
            num_minibatches=8,
            clip_eps=0.05,
            entropy_coef=0.001,
        ),
        "overcooked": dict(
            learning_rate=1e-3,
            update_epochs=15,
            num_minibatches=8,
            clip_eps=0.01,
            entropy_coef=0.05,
        ),
        # UNTUNED. Copied from v1's "overcooked" entry above as a starting
        # point, not a validated choice for v2 -- see the fcp entry's note.
        "overcooked_v2": dict(
            learning_rate=1e-3,
            update_epochs=15,
            num_minibatches=8,
            clip_eps=0.01,
            entropy_coef=0.05,
        ),
        "hanabi": dict(
            learning_rate=5e-4,
            update_epochs=4,
            num_minibatches=8,
            clip_eps=0.2,
            entropy_coef=0.01,
            anneal_lr=True,
            gamma=0.999,
            gae_lambda=0.95,
            max_grad_norm=0.5,
        ),
    },
    "brdiv": {
        # entropy_coef tuned 0.01 -> 0.003 (FCP's value; the only one of FCP's
        # three PPO gaps that transferred -- learning_rate=1e-3 destabilized
        # training and clip_eps=0.03 was a wash). See docs/tuning_record.md.
        "lbf": dict(
            learning_rate=5e-4,
            update_epochs=15,
            num_minibatches=2,
            clip_eps=0.05,
            entropy_coef=0.003,
        ),
        "overcooked": dict(
            learning_rate=1e-3,
            update_epochs=15,
            num_minibatches=8,
            clip_eps=0.01,
            entropy_coef=0.05,
        ),
        # UNTUNED. Copied from v1's "overcooked" entry above as a starting
        # point, not a validated choice for v2 -- see the fcp entry's note.
        "overcooked_v2": dict(
            learning_rate=1e-3,
            update_epochs=15,
            num_minibatches=8,
            clip_eps=0.01,
            entropy_coef=0.05,
        ),
        "hanabi": dict(
            learning_rate=5e-4,
            update_epochs=4,
            num_minibatches=4,
            clip_eps=0.2,
            entropy_coef=0.01,
            anneal_lr=True,
            gamma=0.999,
            gae_lambda=0.95,
        ),
    },
    "lbrdiv": {
        # entropy_coef 0.01 -> 0.003, transferred directly from BRDiv's own
        # tuning rather than re-swept -- see docs/tuning_record.md.
        "lbf": dict(
            learning_rate=5e-4,
            update_epochs=15,
            num_minibatches=4,
            clip_eps=0.05,
            entropy_coef=0.003,
        ),
        "overcooked": dict(
            learning_rate=1e-3,
            update_epochs=15,
            num_minibatches=8,
            clip_eps=0.01,
            entropy_coef=0.05,
        ),
        # UNTUNED. Copied from v1's "overcooked" entry above as a starting
        # point, not a validated choice for v2 -- see the fcp entry's note.
        "overcooked_v2": dict(
            learning_rate=1e-3,
            update_epochs=15,
            num_minibatches=8,
            clip_eps=0.01,
            entropy_coef=0.05,
        ),
        "hanabi": dict(
            learning_rate=5e-4,
            update_epochs=4,
            num_minibatches=4,
            clip_eps=0.2,
            entropy_coef=0.01,
            anneal_lr=True,
            gamma=0.999,
            gae_lambda=0.95,
        ),
    },
    "rpg": {
        # UNTUNED starting point. RPG's base update is a single DiCE policy-gradient
        # step (no PPO clipping/epochs), so only learning_rate, entropy_coef, gamma,
        # gae_lambda, value_coef and max_grad_norm are read; the PPO-specific fields
        # are inert. Base LR follows the reference repo's Overcooked base actor
        # (2.5e-4); the manipulator LR lives on RpgConfig, not here.
        "lbf": dict(
            learning_rate=2.5e-4,
            entropy_coef=0.01,
        ),
    },
}

#: Budget, population and environment count, per (generator, family).
#: ``pop`` is the authored PARTNER_POP_SIZE. Note it is *not* the resulting
#: population size for FCP, which yields ``pop * num_checkpoints`` members
#: because it snapshots during training — see the README.
#:
#: Held equal across every generator and environment so that population size is
#: not a free variable when methods are compared. Upstream used 5 for FCP, 10 for
#: CoMeDi and 3 for BRDiv/L-BRDiv, which meant a difference in downstream results
#: could always be attributed to how many teammates a method happened to produce.
#:
#: This equalizes the number of *scored* members and the number a dataset is
#: collected against. It does not equalize the *released* population: FCP
#: snapshots during training, so it still yields ``POPULATION_SIZE ×
#: num_checkpoints`` = 25 members where the others yield 5. Cutting
#: ``num_checkpoints`` to 1 would equalize that too, but it is precisely the
#: ``FCP₋T`` ablation the paper reports as significantly worse — FCP's diversity
#: *is* the checkpoint spread. §7.3 of the plan tracks this as open.
POPULATION_SIZE = 5

#: Population size BRDiv and L-BRDiv's upstream settings were tuned at.
PAIRED_REFERENCE_POP = 3


def _paired_scale(base_envs: int, base_timesteps: float) -> dict[str, Any]:
    """Scale a paired generator's environments with the square of the population.

    BRDiv and L-BRDiv draw ``conf_id`` and ``br_id`` independently for each
    environment, so a *specific* ``(conf_i, br_j)`` pairing receives only
    ``num_envs / n²`` samples per rollout. The loss weighting is population-size
    invariant — ``E[SP weight]`` is 0.55 and ``E[XP weight]`` 0.025 at every n —
    but the data behind each pairing is not, and that is what actually binds.

    Upstream tuned these at n=3, where ``num_envs=64`` gives 7.1 environments per
    pairing. Moving to n=5 without scaling gave 2.6, and BRDiv collapsed: no
    pairing specialized, the final cross-play matrix was uniform to within noise,
    and self-play fell *below* cross-play — the opposite of what the method
    maximizes. A best response cannot be learned against a confederate it meets
    in two environments per rollout.

    ``total_timesteps`` scales with ``num_envs`` so the update count is
    unchanged; without that, more environments would buy fewer gradient steps and
    trade one failure for another.
    """
    mult = math.ceil((POPULATION_SIZE / PAIRED_REFERENCE_POP) ** 2)
    return dict(
        total_timesteps=base_timesteps * mult,
        num_envs=base_envs * mult,
        pop=POPULATION_SIZE,
    )


SCALE: dict[str, dict[str, dict[str, Any]]] = {
    "fcp": {
        # Tuned. 1e6 at num_envs=8 is 976 updates and stops at ~74% of the food
        # collected; 24e6 at num_envs=64 reaches ~97%, which is the task ceiling.
        # Both the budget and the batch mattered independently — see
        # docs/tuning_record.md.
        "lbf": dict(total_timesteps=24e6, num_envs=64, pop=POPULATION_SIZE),
        "overcooked": dict(total_timesteps=4e6, num_envs=8, pop=POPULATION_SIZE),
        # UNTUNED. Copied from v1's "overcooked" budget as a starting point.
        # Tuned against an external reference, not just an internal slope
        # reading: the original Overcooked-v2 paper reports ~163 return on
        # counter_circuit. num_envs=64 (not upstream's 256 -- OOMs on this
        # device, confirmed by an actual run, not just check_device.py's
        # estimate). total_timesteps=6e7 (2,343 updates) reaches SP=205.20,
        # 126% of the paper's number, and the training curve had already
        # decelerated hard by then (quarter 3->4: 187->191) -- a separate,
        # since-lost 1e8 run (died before checkpointing when an SSH
        # connection dropped, but its metrics.jsonl survived) landed at only
        # 200->207 over the same quarters despite 67% more budget, so 6e7
        # is judged close enough to where more budget stops paying for
        # itself rather than fully flat. actor_type="rnn" +
        # agent_view_size=2 (on the registered presets): partial
        # observability is v2's headline feature and requires memory to be
        # useful -- an MLP cannot make good use of a partial observation.
        # CoMeDi/BRDiv/L-BRDiv do NOT have this option yet:
        # initialize_agents.py's RNN path only covers the plain-actor case
        # FCP uses, not the conditional/double-critic architectures those
        # three need, so they stay on "mlp" pending that work -- meaning
        # they'd currently see agent_view_size=2 with no mechanism to cope
        # with it if run on this preset today. See docs/tuning_record.md.
        "overcooked_v2": dict(
            total_timesteps=6e7, num_envs=64, pop=POPULATION_SIZE, actor_type="rnn"
        ),
        # Tuned. num_envs 32 -> 64 (the LBF batch-size lesson), total_timesteps
        # 1e9 -> 2e9 to hold jax-aht's own reference update count (244,141)
        # fixed at the new batch size -- raw total_timesteps doesn't carry over
        # across a num_envs change. SP flat past 1e9; converged by 2e9 (slope
        # +0.004/1k); 5e9 bought no more competence and its separation edge is
        # unconfirmed at one seed. See docs/tuning_record.md.
        "hanabi": dict(total_timesteps=2e9, num_envs=64, pop=POPULATION_SIZE),
    },
    "comedi": {
        # Converged: 2.4e7 -> 1.92e8 at 64 envs (43,041 sequential updates --
        # CoMeDi trains members one at a time, so this is the single most
        # expensive LBF run in the file). Last-quarter slope fell from
        # +0.020/1k at 9.6e7 to +0.005/1k here, matching the other three
        # generators' converged range. Note the direction this cuts: SP barely
        # moved (0.465 -> 0.472, within the measurement noise floor) while
        # separation *fell* (0.272 -> 0.217) -- the opposite of every other
        # budget doubling in this file. cross_play_weight=0.2 (unchanged) may
        # no longer be enough now that competence isn't the binding
        # constraint; that's the open follow-up. See docs/tuning_record.md.
        "lbf": dict(total_timesteps_per_iteration=1.92e8, num_envs=64, pop=POPULATION_SIZE),
        "overcooked": dict(total_timesteps_per_iteration=1e7, num_envs=48, pop=POPULATION_SIZE),
        # UNTUNED budget, but actor_type is no longer a guess: CoMeDi's RNN
        # conditional critic (RNNActorWithConditionalCriticPolicy) landed this
        # session, mirroring BRDiv/L-BRDiv's own RNN support -- see
        # docs/tuning_record.md. CoMeDi never reassigns which population
        # member plays a role mid-rollout, so unlike BRDiv/L-BRDiv it has no
        # n^2-pairing memory constraint and needs no num_envs reduction;
        # num_envs=64 matches both CoMeDi's own LBF value and FCP's tuned
        # Overcooked-v2 value, keeping the ratio derivation below apples to
        # apples. total_timesteps_per_iteration derived the same way as
        # BRDiv/L-BRDiv's: on LBF, CoMeDi's total_timesteps_per_iteration
        # (1.92e8) is 8x FCP's total_timesteps (24e6) at the same num_envs=64.
        # Applying 8x to FCP's tuned Overcooked-v2 budget (6e7) gives 4.8e8.
        "overcooked_v2": dict(
            total_timesteps_per_iteration=4.8e8,
            num_envs=64,
            pop=POPULATION_SIZE,
            actor_type="rnn_actor_with_conditional_critic",
        ),
        "hanabi": dict(total_timesteps_per_iteration=2e7, num_envs=48, pop=POPULATION_SIZE),
    },
    "brdiv": {
        # LBF budget quadrupled (4.5e7 -> 1.8e8 base, still x3 for n=5 pairing
        # scale = 5.4e8 total): the "converged at 5,493 updates" read above was
        # wrong for num_envs=192 -- that +0.002/1k figure belonged to the old
        # collapsed num_envs=64 run. At 192, the curve was still climbing at
        # +0.027/1k; two more doublings (21,972 updates total) got it to
        # +0.001/1k, genuinely flat. See docs/tuning_record.md.
        "lbf": _paired_scale(64, 1.8e8),
        "overcooked": _paired_scale(128, 9e7),
        # num_envs is NOT _paired_scale(64, ...)'s default. _paired_scale(64,
        # ...) gives num_envs=192 (7.7 envs/pairing, matching LBF's
        # established-safe reference) but check_device.py puts that at 99%
        # of this 6GB GPU's memory -- confirmed OOM. 96 (3.84 envs/pairing,
        # between the known-collapse point 2.6 and LBF's 7.7) fits and both
        # BRDiv and L-BRDiv now train and checkpoint end-to-end at this size
        # with the RNN conditional critic (hstate-threading bugs in
        # _env_step's per-actor vmap fixed this session -- see
        # docs/tuning_record.md). SP-vs-XP has not yet been checked at this
        # budget scale (only a 2e6-timestep smoke test so far) -- still open.
        #
        # total_timesteps derived the same way FCP's overcooked_v2 budget was
        # derived from its LBF one, not guessed: on LBF, BRDiv/L-BRDiv's base
        # (pre-pairing-multiplier, num_envs=64) budget is total_timesteps=1.8e8
        # vs FCP's 24e6 at the same num_envs -- a 7.5x ratio, independent of
        # the pairing multiplier (which scales num_envs and total_timesteps
        # together and cancels out of num_updates). Applying 7.5x to FCP's
        # tuned overcooked_v2 budget (6e7 at num_envs=64) gives a base of
        # 4.5e8, then x1.5 to move from the 64-env base to our actual 96
        # (holding num_updates constant) gives 6.75e8.
        "overcooked_v2": {
            "num_envs": 96,
            "total_timesteps": 6.75e8,
            "pop": POPULATION_SIZE,
            "actor_type": "rnn_actor_with_conditional_critic",
        },
        "hanabi": _paired_scale(128, 5e8),
    },
    "lbrdiv": {
        # LBF budget matched to BRDiv's tuned value directly (4.5e7 -> 1.8e8
        # base = 5.4e8 total) rather than re-swept -- confirmed flat
        # (+0.002/1k) on the first run at this budget. See
        # docs/tuning_record.md.
        "lbf": _paired_scale(64, 1.8e8),
        "overcooked": _paired_scale(128, 9e7),
        # num_envs is NOT _paired_scale(64, ...)'s default. _paired_scale(64,
        # ...) gives num_envs=192 (7.7 envs/pairing, matching LBF's
        # established-safe reference) but check_device.py puts that at 99%
        # of this 6GB GPU's memory -- confirmed OOM. 96 (3.84 envs/pairing,
        # between the known-collapse point 2.6 and LBF's 7.7) fits and both
        # BRDiv and L-BRDiv now train and checkpoint end-to-end at this size
        # with the RNN conditional critic (hstate-threading bugs in
        # _env_step's per-actor vmap fixed this session -- see
        # docs/tuning_record.md). SP-vs-XP has not yet been checked at this
        # budget scale (only a 2e6-timestep smoke test so far) -- still open.
        #
        # total_timesteps derived the same way FCP's overcooked_v2 budget was
        # derived from its LBF one, not guessed: on LBF, BRDiv/L-BRDiv's base
        # (pre-pairing-multiplier, num_envs=64) budget is total_timesteps=1.8e8
        # vs FCP's 24e6 at the same num_envs -- a 7.5x ratio, independent of
        # the pairing multiplier (which scales num_envs and total_timesteps
        # together and cancels out of num_updates). Applying 7.5x to FCP's
        # tuned overcooked_v2 budget (6e7 at num_envs=64) gives a base of
        # 4.5e8, then x1.5 to move from the 64-env base to our actual 96
        # (holding num_updates constant) gives 6.75e8.
        "overcooked_v2": {
            "num_envs": 96,
            "total_timesteps": 6.75e8,
            "pop": POPULATION_SIZE,
            "actor_type": "rnn_actor_with_conditional_critic",
        },
        "hanabi": _paired_scale(128, 5e8),
    },
    "rpg": {
        # UNTUNED. RPG is the most expensive generator here: each outer update
        # collects N self-play + N**2 cross-play rollouts and runs an inner
        # n_lookahead per particle, so cost grows ~quadratically in pop. This is a
        # deliberately modest LBF starting budget (~1,220 updates at num_envs=64);
        # scaling is one of the two open adoption questions (does coverage hold past
        # the paper's N=2?). Tune on GPU before trusting the population.
        "lbf": dict(total_timesteps=1e7, num_envs=64, pop=POPULATION_SIZE),
    },
}

#: Diversity weights that differ per environment.
#: BRDiv's LBF value tuned 0.05 -> 0.10: confirmed a local optimum, beating
#: both a lower (0.07) and higher (0.20) retest at the tuned budget. See
#: docs/tuning_record.md.
#: overcooked_v2 entries below are UNTUNED, copied from v1's as starting
#: points -- same caveat as the PPO/SCALE dicts above.
CROSS_PLAY_WEIGHT = {
    "brdiv": {"lbf": 0.10, "overcooked": 0.005, "overcooked_v2": 0.005, "hanabi": 0.05},
    "comedi": {"lbf": 0.2, "overcooked": 1.0, "overcooked_v2": 1.0, "hanabi": 0.2},
}
MIXED_PLAY_WEIGHT = {"lbf": 0.4, "overcooked": 0.5, "overcooked_v2": 0.5, "hanabi": 0.5}
#: L-BRDiv's LBF value tuned 0.1 -> 0.03: raising it mostly suppresses
#: cross-play rather than trading away self-play competence (a different
#: mechanism from BRDiv's cross_play_weight), so the lower value wins on
#: competence without giving up much separation. See docs/tuning_record.md.
TOLERANCE_FACTOR = {"lbf": 0.03, "overcooked": 10.0, "overcooked_v2": 10.0, "hanabi": 0.1}

#: L-BRDiv's Lagrange multipliers receive gradient from an unnormalized sum over
#: ~n^2 pair terms, so the learning rate must be scaled by ~(n_ref/n)^2 relative
#: to the population it was tuned at. Upstream's 0.01 is tuned at n = 3 (§7.3).
LAGRANGE_LR_AT_N3 = 0.01


def _lagrange_lr(pop: int) -> float:
    return round(LAGRANGE_LR_AT_N3 * (3 / pop) ** 2, 6)


def build(generator: str, preset_name: str, num_checkpoints: int = 5):
    fam = _family(preset_name)
    ppo = PpoHyperparams(**PPO[generator][fam])
    scale = SCALE[generator][fam]
    pop = scale["pop"]
    common = dict(
        population_size=pop,
        num_checkpoints=num_checkpoints,
        num_envs=scale["num_envs"],
        ppo=ppo,
        network=MlpNetwork(),
    )
    # Only overridden when SCALE explicitly names one (currently just FCP x
    # overcooked_v2, for its RNN policy -- see docs/tuning_record.md). Every
    # other (generator, family) keeps that generator's own default
    # (CoMeDi/BRDiv/L-BRDiv default to their conditional/double-critic actor
    # types, not "mlp"), so this must not apply a blanket default here.
    if "actor_type" in scale:
        common["actor_type"] = scale["actor_type"]

    if generator == "fcp":
        return FcpConfig(total_timesteps=scale["total_timesteps"], **common)
    if generator == "comedi":
        return CoMeDiConfig(
            total_timesteps_per_iteration=scale["total_timesteps_per_iteration"],
            cross_play_weight=CROSS_PLAY_WEIGHT["comedi"][fam],
            mixed_play_weight=MIXED_PLAY_WEIGHT[fam],
            **common,
        )
    if generator == "brdiv":
        return BrDivConfig(
            total_timesteps=scale["total_timesteps"],
            cross_play_weight=CROSS_PLAY_WEIGHT["brdiv"][fam],
            **common,
        )
    if generator == "lbrdiv":
        return LBrDivConfig(
            total_timesteps=scale["total_timesteps"],
            tolerance_factor=TOLERANCE_FACTOR[fam],
            lagrange_learning_rate=_lagrange_lr(pop),
            **common,
        )
    if generator == "rpg":
        # Diversity knobs use RpgConfig's defaults (partnerplay_ratio=0.1,
        # off_diag_factor=0.25, dice_lambda=0.99, n_lookahead=1, manipulator_lr).
        # At pop=5 the base self-play weight is 1 - 5*0.1 = 0.5 (stays positive).
        return RpgConfig(total_timesteps=scale["total_timesteps"], **common)
    raise ValueError(f"unknown generator {generator!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--all-envs",
        action="store_true",
        help="Emit for all seven results configurations, not just tier 1.",
    )
    ap.add_argument(
        "--wandb",
        metavar="PROJECT",
        nargs="?",
        const="oaht-bench",
        default=None,
        help="Enable wandb logging in the emitted configs, under PROJECT "
        "(default 'oaht-bench'). The entity is deliberately never written: "
        "wandb takes it from WANDB_ENTITY or your login, so a config that "
        "someone else runs does not publish into your account.",
    )
    args = ap.parse_args()

    envs = preset_names() if args.all_envs else preset_names("tier1")
    envs = [e for e in envs if e != "mini_hanabi"]

    written = []
    for env_name in envs:
        env = get_preset(env_name)
        for generator in ("fcp", "comedi", "brdiv", "lbrdiv", "rpg"):
            # RPG is only tuned/supported on LBF so far (see SCALE/PPO tables).
            if generator == "rpg" and _family(env_name) != "lbf":
                continue
            gen = build(generator, env_name)
            kwargs: dict[str, Any] = {}
            if args.wandb:
                kwargs["logging"] = LoggingConfig(use_wandb=True, wandb_project=args.wandb)
            job = TeammateGenerationJob(
                label=f"{generator}_{env_name}", env=env, generator=gen, **kwargs
            )
            path = CONFIGS_ROOT / env_name / "teammate_gen" / f"{generator}.json"
            save_job(job, path, minimal=True)
            written.append((env_name, generator, gen, job))

    print(f"{'environment':30s} {'gen':8s} {'pop':>4s} {'envs':>5s} {'budget':>10s}  hash")
    for env_name, generator, gen, job in written:
        budget = getattr(gen, "total_timesteps", None) or gen.total_timesteps_per_iteration
        print(
            f"{env_name:30s} {generator:8s} {gen.population_size:4d} "
            f"{gen.num_envs:5d} {budget:10.1e}  {job.short_hash()}"
        )
    print(f"\n{len(written)} configs -> configs/<env>/teammate_gen/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
