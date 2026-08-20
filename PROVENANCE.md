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
| `ego_agent_training/` | `src/oaht_bench/algorithms/` | MeLIBA network components only. The online PPO ego trainers are deliberately excluded -- §3.1 replaces them with the shared DT backbone. |

Import rewrites applied:

```
envs                   -> oaht_bench.envs
agents                 -> oaht_bench.agents
teammate_generation    -> oaht_bench.teammate_gen
marl                   -> oaht_bench.teammate_gen.marl
common                 -> oaht_bench.common
ego_agent_training     -> oaht_bench.algorithms
```

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
