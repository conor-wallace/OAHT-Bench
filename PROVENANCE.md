# Provenance of absorbed upstream code

Parts of `src/oaht_bench/` originate in other projects and were absorbed
rather than depended on, because the upstreams claim colliding top-level
package names (see `scripts/absorb_upstream.py` for the reasoning).

Absorbed code is **owned and modified** here. To see our changes relative to
upstream, diff against the recorded commit. Regenerate this file by re-running
the absorption script.

## jax-aht

- Upstream: `https://github.com/LARG/jax-aht`
- Commit: `0885df95c386121b9c94cb0fb516531895e29702`
- License: MIT (see `LICENSES/jax-aht-LICENSE`)

| upstream path | local path | contents |
|---|---|---|
| `envs/` | `src/oaht_bench/envs/` | LBF, Overcooked-v1 and Hanabi wrappers over Jumanji/JaxMARL. |
| `agents/` | `src/oaht_bench/agents/` | Policy architectures, population interfaces, scripted teammates (§7.6). |
| `teammate_generation/` | `src/oaht_bench/teammate_gen/` | FCP, CoMeDi, BRDiv, L-BRDiv (§7). |
| `marl/` | `src/oaht_bench/teammate_gen/marl/` | IPPO and PPO utilities; teammate generation is the only consumer. |
| `common/` | `src/oaht_bench/common/` | Rollout helpers, checkpoint save/load, plotting. |

Import rewrites applied:

```
envs                   -> oaht_bench.envs
agents                 -> oaht_bench.agents
teammate_generation    -> oaht_bench.teammate_gen
marl                   -> oaht_bench.teammate_gen.marl
common                 -> oaht_bench.common
```

The one exception is jax-aht's `ego_agent_training/` (its online PPO LIAM and
MeLIBA learners). Its MeLIBA network components were absorbed once into
`src/oaht_bench/algorithms/` but were never wired into this pipeline -- §3.1
replaces the online ego trainers with the shared decision-transformer backbone --
so that copy was removed. LIAM and MeLIBA are instead reimplemented offline in
`src/oaht_bench/offline/{liam,meliba}` (clean-room, from the papers).

## JaxMARL (`overcooked_v2`)

- Upstream: `https://github.com/FLAIROx/JaxMARL`
- Version / commit: `v0.1.0`, `66f41e5a36131d86bf5791d6bbe501275ed2cd30`
- License: Apache-2.0 (see `LICENSES/jaxmarl-LICENSE`)

Not absorbed via `jax-aht`, and not from the same tree as jax-aht's Overcooked-v1
wrapper, which imports the pinned `jaxmarl==0.0.7` package directly rather than
absorbing its source. `overcooked_v2` doesn't exist in `0.0.7`; it first appears
in `0.1.0`, which declares `jax<=0.4.38` and therefore cannot be installed
alongside this project's `jax==0.5.3` pin (see `pyproject.toml`). So its source is
absorbed at the `0.1.0` tag instead of bumping the package, leaving `jaxmarl` at
`0.0.7` — and therefore Overcooked-v1 and Hanabi, both of which import it
directly — untouched.

| upstream path | local path | contents |
|---|---|---|
| `jaxmarl/environments/overcooked_v2/{__init__,common,layouts,overcooked,settings,utils}.py` | `src/oaht_bench/envs/overcooked_v2/` | The OvercookedV2 environment, absorbed whole. |

Not absorbed:
- `interactive.py` — a human-play CLI (`jaxmarl/viz/overcooked_v2_visualizer.py` and
  pygame), no consumer in this training pipeline.

`OvercookedV2(MultiAgentEnv)` subclasses the pinned `jaxmarl==0.0.7` package's
`jaxmarl.environments.multi_agent_env.MultiAgentEnv` directly — checked before
absorbing that `0.0.7`'s and `0.1.0`'s versions of that module, and of
`jaxmarl.environments.spaces`, differ only in docstrings (`multi_agent_env.py`)
or not at all (`spaces.py`), so nothing from `0.1.0` needed absorbing for those.
Import rewrite applied to the four absorbed files that reference sibling
modules: `jaxmarl.environments.overcooked_v2 -> oaht_bench.envs.overcooked_v2`.

`OvercookedV2` does not override `get_avail_actions` (raises
`NotImplementedError` on the base class) — matching Overcooked-v1, whose own
wrapper doesn't get real masking from the environment either, since the action
set has no state-dependent restrictions. The wrapper supplies the same
all-actions-available stub v1's does, not a new mechanism.

# Clean-room reimplementations (not absorbed)

Some algorithms are reimplemented from their papers rather than absorbed, because
their source is unavailable under a redistributable license. **No source code is
copied** from these upstreams; the reference repository is used only to understand
the method. Unlike absorbed code, these files are authored, linted (they are on the
Ruff allowlist in `pyproject.toml`), and unit-tested.

## AD-RPG (Rational Adversarial Diversity)

- Reference paper: Lauffer, Shah, Carroll, Seshia, Russell & Dennis, *Robust and
  Diverse Multi-Agent Learning via Rational Policy Gradient*, NeurIPS 2025.
- Reference repository (read-only, **not** copied): `https://github.com/niklaslauffer/rational-policy-gradient`, commit `0f9b863cae3eb78cf70d4e20db2f3441ba73c32c`.
- License: **none** — the upstream repository ships no license, so its code cannot
  be absorbed or redistributed. This is the reason for the clean-room path.
- Local path: `src/oaht_bench/teammate_gen/RPG.py` (authored). Scope: the
  `doublesided_RAD` variant only (adversarial diversity), reimplemented natively on
  our JAX/`marl` stack. The upstream targets jax 0.4.30 / flax 0.8.5 / a JaxMARL
  fork / Hydra, none of which match our pinned stack, so a native reimplementation
  was required regardless of licensing.
