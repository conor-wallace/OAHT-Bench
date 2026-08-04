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
| `agents/` | `src/oaht_bench/agents/` | Policy classes, population interfaces, scripted/heuristic teammates (§7.6). |
| `teammate_generation/` | `src/oaht_bench/teammate_gen/` | FCP, CoMeDi, BRDiv, L-BRDiv (§7). |
| `marl/` | `src/oaht_bench/marl/` | IPPO and PPO utilities used by the generators. |
| `common/` | `src/oaht_bench/common/` | Rollout helpers, checkpoint save/load, plotting. |
| `evaluation/` | `src/oaht_bench/evaluation/` | Held-out evaluation and cross-play matrices. |

Import rewrites applied:

```
envs                   -> oaht_bench.envs
agents                 -> oaht_bench.agents
teammate_generation    -> oaht_bench.teammate_gen
marl                   -> oaht_bench.marl
common                 -> oaht_bench.common
evaluation             -> oaht_bench.evaluation
```
