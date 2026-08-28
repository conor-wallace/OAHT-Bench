# Dataset design (post-ICRL, OG-MARL/D4RL-scale)

Supersedes the ICRL-dependent parts of the plan's §4. Two scope changes drive it:

1. **ICRL is abandoned.** The project targets *offline ad-hoc teamwork* only. §4.1's
   "two views" collapses to **one** — the trajectory (D4RL-shaped) view. The
   learning-history view (AD/DPT/AMAGO), its `history_id`/
   `episode_index_within_history` links, and the ICRL4AHT HDF5+JSONL-index format
   (§4.2 rev 2) are dropped. This is what frees the format decision below.
2. **OG-MARL/D4RL scale.** The current `.npz`-per-run, padded-`(episode, agent, T,
   …)` schema (`data/schema.py`) does not scale — padding wastes space and forces
   whole-file loads. We move to a flat-transition store.

This note defines the three things needed to begin: **structure**, **file
format**, and **creation process**. Expert collection already exists
(`data/runner.py`); everything here is how the rest is built on it.

## 1. Structure — two quality axes, defined not trained

OAHT has two *independent* quality axes, where OG-MARL/D4RL have one. Making them
separable is the contribution over OG-MARL.

- **τ — teammate competence.** How good is the partner. OG-MARL's poor/medium/good.
  Controlled by *which training checkpoint* of a member is seated. **Deferred**:
  needs training-time snapshots, which only FCP saves today (see §4 below).
- **ε — ego-response quality.** How well the ego's behaviour coordinates with a
  given teammate. This is the D4RL return-quality axis *and* the best↔worst idea,
  and it is **defined by data, not trained**: for teammate `j`, evaluate every
  candidate ego `i` and rank by expected coordination return.
- **Pairing correctness** (orthogonal). Matched vs mismatched members — already
  `mismatch_fraction` in the config and `_seat_plan` in the runner. A *competent*
  ego for the *wrong* teammate, distinct from a low-ε response to the right one.

### ε as a pooled cross-population coordination-return spectrum

For teammate `j`, `best-response(j) = argmax_i R(i,j)` and
`worst-response(j) = argmin_i R(i,j)`, where `R(i,j)` is the expected return of
ego member `i` seated with teammate `j`. Per-**teammate** normalisation
(`ε(i|j) = (R(i,j) - min_i R(·,j)) / (max_i R(·,j) - min_i R(·,j))`) gives a
`[0,1]` response-quality spectrum — best = 1, worst = 0 — relative to the partner,
which is the right OAHT semantic.

The roster is **pooled across all generators** (decision b): `i` and `j` range over
the union of released members from `{fcp, comedi, brdiv, lbrdiv}` for the
environment, so the best response to a CoMeDi teammate may be an FCP member. This
is the cross-population mixing the runner comment (`data/runner.py:99-102`) asked
for, made precise.

- **Paired generators fit directly.** The released cross-play matrix is already
  `conf_i × br_j`; `R(i,j)` is confederate `i` with best-response `j`, so
  best-to-`conf_i` is `br_i` (the diagonal) and worst is `argmin_j`.
- **FCP** is member×member: best ≈ the diagonal (self-play), worst = the
  least-compatible member.

No worst-response oracle and no adversarial training: best/worst are existing
members selected by the matrix.

### Variants are distributions over (τ, ε)

| Variant | τ | ε | Needs |
|---|---|---|---|
| `expert` ✅ | converged | top (best-response) | released pops (done) |
| `br_vs_worst` | converged | full/bimodal spectrum | + pooled matrix |
| `mixed` | converged | 50/50 top + mid (D4RL medium-expert) | + pooled matrix |
| `medium` / `poor` | mid / early ckpt | matched | + competence ladder (deferred) |
| `random` | random policy | random | + random rollouts |
| `replay-full` | all ckpts | all responses | + snapshots (deferred) |

Keep D4RL's exact definitions where they apply (§4.3): `replay-full` ≠ D4RL
`medium-replay` (ours runs to convergence — name it distinctly), `mixed` is 50/50.
`expert` stays the highest-leverage artifact (OMIS BC targets, TAO stage-2 targets,
BR-Prox normalisation) and is the top of the ε spectrum. OMIS remains structurally
excluded from low-ε regimes — it trains on best responses only (§4.3 rev 6).

## 2. File format — Flashbax Vault, flat transitions

**Decision (a): Flashbax Vault.** It is what current OG-MARL uses
(`instadeepai/og-marl`, hosted on HF as `…/<env>.vlt/{Good,Medium,Poor}/`),
JAX-native, memory-mapped so it scales past RAM, and integrates with a JAX replay
buffer. Its quality-folder convention *is* our variant layout. HDF5 (D4RL) was the
alternative; Vault wins for a JAX, OG-MARL-sibling project.

**Layout: flat transitions, not padded episodes.** Both D4RL (HDF5) and OG-MARL
(Vault) store a single flat buffer with explicit boundaries, not `(episode, agent,
T, …)` tensors. Per-timestep fields (leading `agent` axis kept, per `schema.py`'s
generality argument), concatenated over all transitions, with `terminals` /
`episode_id` marking episodes. `make_windows` re-derives its sliding windows over
`(flat transitions + episode_id)` as cleanly as over padded episodes.

Per-timestep fields (D4RL base + AHT extensions from §4.2, ICRL fields removed):
`observations`, `actions`, `rewards`, `terminals`, `avail_actions`, `acting_agent`,
and — required by the trajectory-view baselines — `teammate_actions`,
`teammate_observations`, `teammate_rewards`, `teammate_id` (withholdable at eval).

Per-episode labels (broadcast or indexed by `episode_id`) — what makes a dataset
self-describing and a mixture sliceable:
- `member_ids` (per seat) — as today
- `source_generator` (per seat) — which population each member came from
- `ego_response_quality` — the pooled-matrix `ε(i|j)` for the seated pair (the
  stable descriptor; the realised per-episode return is derivable separately)
- `teammate_role` — `conf`/`br` for the paired generators
- `episode_id`, `episode_return`, `episode_length`, `variant`

Dataset-level metadata: env + `env_kwargs`, population checkpoint hashes, the
**pooled cross-play matrix hash**, seed, schema version, code commit, filter config
(§4.4), mirroring flag (§4.5). `EpisodeBatch`'s dataclass is already separate from
its `save`/`load`, so the store swaps under it without touching consumers.

## 3. Creation process

`data/runner.run` becomes the entry point that dispatches on `variant` to private
collectors (per the runner comment). New pieces, in dependency order:

1. **Pooled cross-population cross-play matrix.** Extend the crossplay eval
   (`teammate_gen.crossplay`, which already produces each population's
   `population_crossplay.csv`) to seat member-`i`-from-pop-A against
   member-`j`-from-pop-B over the full pooled roster, producing an
   `N_total × N_total` `R(i,j)`. Release it as a per-environment artifact
   (`populations/<env>/pooled_crossplay.*`) so datasets are reproducible from it.
2. **Multi-population input.** `population_path: str → list[str]`; rebuild each
   population with its own generator builder (as `_load_population` does now) and
   concatenate into one pooled roster with `source_generator` tags.
3. **ε-targeted seat sampler.** Given a target ε distribution and the pooled
   matrix, sample `(i, j)` pairings whose normalised `ε(i|j)` matches it —
   generalising `_seat_plan` from matched/mismatched counts to a spectrum. Records
   `ego_response_quality` exactly (the matrix value, not the sampled return).
4. **Vault writer.** Replace `EpisodeBatch.save`'s `.npz` with a Vault write of the
   flat-transition buffer + labels; keep `.load` reading either during transition.

### Build order

- **Buildable now, zero new training or snapshotting:** the pooled matrix (1),
  multi-pop input (2), the ε sampler (3) → **`expert` and `br_vs_worst`** straight
  from released populations. `expert` is just the top of the spectrum. This is the
  first cut and covers the standard OAHT setting plus your best-vs-worst study.
- **Deferred to the τ phase:** `medium`/`poor`/`replay-full` need the competence
  ladder — training-time checkpoint snapshots, which BRDiv/L-BRDiv/CoMeDi don't
  save today (only FCP does). Snapshotting those generators is the gating task for
  τ, and *only* for τ; it blocks nothing in the ε work above.

## 4. Open / deferred

- **Self-pairing on the diagonal.** For FCP the best response to `j` is often `j`
  itself. Decide whether `best-response` includes the self pairing or only
  cross-pairings (affects `expert`'s top-of-spectrum definition).
- **Competence-ladder snapshotting** for the non-FCP generators (τ axis).
- **Storage budget** (§4.6) — Vault + flat transitions removes padding waste, but
  the pooled matrix and multi-variant release still need the "release populations +
  matrix + one canonical dataset per env; regenerate the rest" decision.
- **Filtering (§4.4)** — return-quantile filtering stays relevant (it is what the
  merged `%BC` baseline already does); the *learning-curve* filtering rationale is
  gone with ICRL.
