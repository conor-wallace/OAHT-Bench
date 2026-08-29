# OAHT-Bench: A Survey and Benchmark for Offline Ad-Hoc Teamwork

Status: planning, rev 4 (2026-07-31). **Target: ICML 2027, deadline ~late January 2027
(verify exact date — ICML has historically landed in the last week of January).** Supersedes
`docs/offline_aht_benchmark_task.md` (that file scoped baseline adaptation as an instrument for
one research question; this project makes the benchmark itself the deliverable and folds that
work in as Phase 1).

Rev 2 corrected source-repo coverage errors and added the §3.1 backbone decision. Rev 3 fixed the
venue and added §10.1 (ICML needs a finding, not an artifact). Rev 4 was an environment-first
restructure. Rev 5 fixed the baseline suite and made the backbone a Decision Transformer.
Rev 6 was the first revision grounded in the primary literature rather than code and abstracts,
written after reading the five decision-critical papers. **Rev 7 folds in the remaining ten** —
the four teammate-generation papers, LIAM, MeLIBA, AMAGO, DT, IQL, D4RL — correcting two claims
rev 6 got wrong and adding a confound, a ceiling baseline, and several precise definitions. All 15
papers in `papers/` have now been read; the extraction lives in `docs/baseline_specs.md`. See §14
for the change log.

> **Companion document.** `docs/baseline_specs.md` holds the per-method extraction: architectures,
> exact loss functions, hyperparameters, reported numbers, and data requirements. It is the
> reference for implementation; this document is the reference for scope, schedule, and protocol.
> Where they disagree, `baseline_specs.md` is closer to the source.

**Committed scope:** LBF 12×12, Overcooked-v1 (all five layouts), Hanabi — seven environment
configurations, four teammate generators each.

---

## 1. Thesis and contributions

**Nobody knows the state of the art in offline ad-hoc teamwork.** Every method defines its own
environments, its own teammate populations, its own data collection, and its own metrics. TAO
evaluates on Markov Soccer and particle-world; OMIS on classic LBF and overcooked-ai; ICRL4AHT
on Overcooked-V2 only, against AD/DPT-class baselines rather than the methods the field
actually cites. No two of these are comparable, and none of the frequently-cited methods have
ever been run head-to-head under a shared protocol.

Four contributions, in dependency order:

1. **A survey and taxonomy** of offline AHT / offline opponent modeling, organized around axes
   that become visible only when you try to unify the methods (see §9) — notably *what data
   each method actually requires*, which turns out to partition the field more sharply than the
   usual architectural framing.
2. **Standardized offline datasets**, D4RL-style, with an AHT-specific schema (§4). This is the
   load-bearing artifact: it is the contract that makes methods comparable and lets others add
   baselines later without adopting our framework.
3. **[rev 4] A released teammate-population suite** — all four generators (FCP, CoMeDi, BRDiv,
   L-BRDiv) trained in-house across all seven environment configurations, with per-environment
   hyperparameter tuning documented and released alongside the checkpoints. This is a
   contribution in its own right, not just an input to contribution 2: no existing source
   provides the full generator × environment matrix (§7.1), the tuning knowledge is currently
   folklore (§7.2), and controlling the populations ourselves is what makes the datasets
   reproducible rather than inherited.
4. **A unified baseline suite** — AD, DPT, AMAGO, Hybrid-AD, LIAM, MeLIBA, TAO, OMIS, TAGET —
   under one protocol, one set of teammate populations, one metric suite, and (new in rev 2)
   **one shared sequence-model backbone**, so that measured differences are attributable to the
   teammate-modeling mechanism rather than to per-method offline-conversion choices (§3.1).
5. **Actual head-to-head results**, establishing what the state of the art is, plus the
   diagnostic analyses the shared setting enables (cross-play structure, teammate-identity
   recoverability across representations).

**[rev 5] BayesToM is not a baseline in this benchmark.** It addresses a different problem
setting, and including it would either misrepresent it or distort the benchmark's scope. The
bayes-tom repository still contributes an instrument — the representation-agnostic
identity-recoverability probe (§8) — but the method itself is out. This removes the "our method
lands in the setting we define" framing from rev 1–4; the benchmark now stands on contributions
1–5 alone, which is a cleaner position to argue from anyway.

**[rev 5] TAGET (ICML 2025) sharpens the thesis rather than weakening it.** "Ad Hoc Teamwork via
Offline Goal-Based Decision Transformers" (Zhang, Chan, Ye, Cai, Zhao) is the most recent
directly comparable method, and it is *another* method defining its own setting — Predator-Prey,
LBF, and Overcooked, with its own teammate construction and its own offline data. It is
simultaneously the strongest evidence for §1's premise and the most demanding competitor for our
results section (§11).

**[rev 6] The premise is now verified in detail, and it is worse than rev 1 claimed.** Reading the
papers confirms that no two methods share an environment *configuration*, not merely an
environment name. TAGET's LBF is 20×20 with a 5×5 observation window and a simultaneous-collect
rule; ours is 12×12 Jumanji with level-based collection. TAO evaluates on Markov Soccer and
Particleworld — **both competitive**. OMIS frames LBF as a social dilemma. Population construction
differs four ways across four papers (our four diversity generators; TAGET's SVD/CSP; OMIS's MEP;
TAO's hand-built mix of scripted policies and RL checkpoints at varying training durations).
Evaluation budgets span **50 to 2500 episodes**. The field is less comparable than the "different
environments" framing suggests — the incomparability reaches all the way down to what counts as a
teammate.

**[rev 7] D4RL's own taxonomy classifies offline AHT as a hard case.** Its task-design factors
(§4) call out *"non-representable behavior policies, **non-Markovian behavior policies, and
partial observability**,"* noting these *"introduce additional modeling errors... especially in
methods that assume access to action probabilities from a Markovian policy."* Offline AHT is
inherently both: partially observable, and the behavior policy is non-Markovian because it is
conditioned on a teammate. The canonical offline-RL benchmark flags our setting as among the
hardest it anticipates — a useful, non-self-serving citation for why this benchmark is needed.
D4RL also names **stitching** as a design factor, which is the mechanism behind the DT-vs-IQL
backbone ablation (§3.1).

**[rev 6] ICRL4AHT states our thesis as its own limitation.** *"[This benchmark] is instantiated
on a **single domain** (OvercookedV2), employs a **finite teammate suite**, and restricts
evaluation to **two-player settings with fixed partners**."* Our seven environment configurations,
four generator families plus ordered heuristics, and non-stationary teammate condition (§8)
address all three. This is the paper's cleanest positioning statement and should be quoted.

---

## 2. Verified state of the source codebases

Everything in this table was checked against the actual code, not READMEs or papers. Where a
fact was not verified, it says so. Rows marked **[rev 2]** were re-verified and corrected.

| Source | Framework | License | What is genuinely reusable | Hard limits |
|---|---|---|---|---|
| **jax-aht** (`LARG/jax-aht`, our fork `sumust/jax-aht`) | JAX/Flax | **MIT** (`LICENSE`, UT Austin LARG, 2025) | **[rev 2]** All four teammate-gen algorithms across **all three target envs** — `teammate_generation/configs/algorithm/{fcp,comedi,brdiv,lbrdiv} × task/{lbf, overcooked-v1, hanabi}` is a complete matrix. Env layer `envs/__init__.py` dispatches `lbf`, `lbf-reward-shaping`, `overcooked-v1`, `hanabi`. LIAM (`agents/liam_agent.py`, `ego_agent_training/liam_ego.py`), MeLIBA (`agents/meliba_agent.py`, `meliba_ego.py`), `evaluation/generate_xp_matrix.py::heldout_crossplay`, `common/run_episodes.py` | **[rev 2]** LIAM/MeLIBA are **online PPO** learners, not offline (§3.1). LIAM ego configs exist for LBF + Hanabi but **not Overcooked**. `heldout_crossplay` is an explicit double loop (not vmapped) and is slow |
| **ICRL4AHT** (`AHT-Hub/ICRL4AHT`, arXiv:2605.24423) — **[rev 6] also a negative-results benchmark paper, not just a code source**: reports AD/DPT/AMAGO/Hybrid-AD failing to beat a random baseline with flat adaptation curves, and states single-domain / finite-teammate-suite / fixed-partner limitations that are precisely our thesis (§1). Contributes the *manifest* pattern, the two-track protocol, the cooperability-ordered heuristic families, the adaptation-gain metric, and the trajectory-filtering stage (§4.4) | JAX/Flax (`jax==0.5.3`, `flax==0.10.2`, `jaxmarl==0.0.7`) | **[rev 4] Licensing resolved — no issue** | 4 genuinely-offline baselines: `benchmarks/baselines/{ad,dpt,amago_offline,hybrid_ad}` (`ad/train.py` reads `histories.h5` + `histories_index.jsonl`, never touches an env). `envs/base_env.py` ABC; `runners/{history_adapter,history_recorder,task_runner}.py`; `scripts/{build_index,collect_histories}.py`; `teammate_wrapper/{specs,registry,theta_sampling,rl_wrappers}.py` | **Overcooked-V2 only** — `envs/__init__.py:21` dispatches `'overcooked-v2'`, line 103 raises `NotImplementedError`. All 5 `teammate_generation/train_*_overcooked_v2.py` scripts hardcoded. CNN encoders in `ad/model.py`, `dpt/model.py` hardcoded to `(9,7,26)`. No shipped datasets or train manifests |
| **TAO** (local, `~/Documents/Personal/Projects/TAO/`) | PyTorch | Supplementary material, **license unstated** | Architecture reference: `offline_stage_1` contrastive opponent encoder, `offline_stage_2` DT-style decoder conditioned on frozen embeddings, `deployment_stage` Opponent Context Window | **No env abstraction** — `Config.ENV_TYPE` is a `"MS"`/`"PA"` string switch branching obs/act dims throughout. Competitive envs only. Offline `.pkl` datasets not packaged (external OSF link). `trajectory_gpt2.py` builds GPT-2 from scratch (`vocab_size=1`, `n_layer=3`, `n_head=1`) — not pretrained |
| **OMIS** (local, `~/Documents/Personal/Projects/OMIS/`) | PyTorch | Supplementary material, **license unstated** | Architecture reference: actor + opponent-imitator + critic (shared `trajectory_gpt2` backbone), `testing/search.py` decision-time search | **`search.py` deep-copies the live env object** (`copy.deepcopy(statedict['root_env'])`) — hard coupling to a mutable gym-style env. Vendored `semitable/lb-foraging` at `Foraging-9x9-2p-5f-v2` (different config from ours). Opponent pool checkpoints not packaged (external ONNX download) |
| **ZSC-Eval** (`sjtu-marl/ZSC-Eval`) | PyTorch | MIT | **Evaluation methodology**: BR-Prox metric, BR-Div partner selection, event-based behavior-preferring rewards. Ships pretrained policy pool (HuggingFace) | Online ZSC, **not offline** — no offline-data story stated. Overcooked + GRF. Not an architectural template for us |
| **TAGET** (ICML 2025, `openreview.net/forum?id=tl3FlgWScA`) | — | **No code released** | Hierarchical TA-RTG + TA-Goal design; **trajectory mirroring** as a dataset technique (§4.5); a published `LIAM-off` variant. Full spec extracted in `baseline_specs.md` | No code → reimplementation with no reference to diff against. **[rev 6]** Environments are *not* comparable to ours despite sharing names: LBF is 20×20 with 5×5 obs and a simultaneous-collect rule; Overcooked is a custom `overcooked_ai` layout. **Does not convert MeLIBA** (that is TAO). Numeric validation impossible (§10.6) |
| **JAX-CORL** (`nissymori/JAX-CORL`, local at `~/Documents/Personal/Projects/JAX-CORL/`) | JAX | **MIT** (`LICENSE`, 2024) | **Single-file JAX implementations of the offline backbones**: `algos/dt.py` (Decision Transformer), `algos/iql.py`, plus `awac`, `cql`, `td3bc`, `xql`. Clean, benchmarked against D4RL, dependency-light | Single-agent D4RL-shaped; needs the AHT conditioning interface (§3.1) added. Not a multi-agent framework and should not be treated as one |
| **bayes-tom** (local) | JAX + LLM API | ours | **[rev 5] Not a baseline** (§1). Contributes: existing LBF populations at `checkpoints/lbf/lbf_12x12/{fcp,comedi,brdiv,lbrdiv}` (~12 MB, all four generators) and `scripts/diagnose_embedding_headroom.py` (representation-agnostic identity-recoverability instrument, §8) | — |

**[rev 2] Revised consequences.** Rev 1 concluded that "neither source alone covers the
env × algorithm matrix, so unifying teammate generation is real work." That is wrong.
**jax-aht already covers the full 4-generator × 3-environment matrix**, is MIT licensed, and is
therefore the natural spine of the project. The division of labor between the two JAX sources is
cleaner than rev 1 thought:

- **jax-aht supplies the environment layer and teammate generation** — the parts we would
  otherwise write from scratch, and the parts that must work identically across all three envs.
  MIT, so no licensing exposure.
- **ICRL4AHT supplies the offline data machinery and the in-context baselines** — history
  recorder, HDF5 store + JSONL index, AD/DPT/AMAGO/Hybrid-AD. This is also **exactly the
  license-risky half** (§11), which sharpens the fallback: if the license question resolves
  badly, we lose four baselines and a data format, but not the spine.

The other consequence from rev 1 stands: **six of the seven target baselines are JAX-native**,
only TAO and OMIS are PyTorch.

---

## 3. Central architecture decision: JAX core, dataset as the framework-neutral contract

**Decision: build the core in JAX (env, teammate generation, data collection, evaluation
harness), forked from jax-aht. Make the on-disk dataset framework-neutral. Reimplement TAO and
OMIS in JAX, using the PyTorch originals as reference implementations for correctness validation
rather than as runtime dependencies.**

Rationale:

- 6/7 baselines are already JAX. Porting them to PyTorch would be strictly more work than the
  reverse and would throw away jax-aht's and ICRL4AHT's working code.
- The dataset being framework-neutral (HDF5 + numpy, D4RL conventions) is what makes the
  benchmark *a benchmark* rather than a framework. Someone can add a PyTorch baseline later by
  reading our files; they never need to adopt JAX. This matters for adoption and is worth
  protecting even where it costs us convenience.
- **OMIS forces the issue.** Its decision-time search needs to step and copy environment state
  many times per action. A PyTorch OMIS driving a JAX env would cross the host boundary
  constantly and be unusably slow. But in JAX/Jumanji, states are immutable pytrees — copying
  is free and the search is *cleaner* than OMIS's original `deepcopy` hack. Reimplementing is
  both necessary and an improvement.
- TAO's networks are tiny (3 layers, 1 head). Reimplementation is a days-not-weeks task, and
  it has no env coupling at training time at all.

The cost is honest and should be stated in the paper: TAO and OMIS results come from
reimplementations, not authors' code. Mitigation in §11.

### 3.1 [rev 2] The offline-conversion decision

**LIAM and MeLIBA as shipped in jax-aht are online on-policy PPO learners.** `liam_ego.py:191`
and `meliba_ego.py:198` call `jax.vmap(env.step)` inside a rollout loop; `meliba_ego.py:223`
computes GAE. Rev 1's §6 rated them **S–M** ("swap the encoder") and made LIAM a Phase 0
baseline on the assumption it was already offline. It is not. Their *teammate-modeling heads*
(LIAM's reconstruction loss, MeLIBA's variational encoder) are offline-compatible; the ego
learner wrapped around them is not.

**[rev 6] The offline conversions are published prior art — and so is the shared-backbone
methodology itself.** Verified by reading the papers:

- **LIAM-offline and MeLIBA-offline are both specified in TAO (ICLR 2024), Appendix F**, in full
  architectural detail. TAGET (ICML 2025) independently converts LIAM (`LIAM-off`) but **not**
  MeLIBA, and its specification is a single sentence. TAO's is the one to follow.
- **TAO also establishes the shared-backbone protocol as accepted practice**: *"To ensure an
  equitable comparison, we mandate all approaches to use the same neural architecture as ours."*
  That is exactly §3.1's design, published, applied to this exact family of methods. **This is the
  strongest available answer to the §11 objection** that a shared backbone misrepresents the
  published methods — it is the subfield's own convention, not our invention.

Rev 2–4 treated "offline LIAM" and "offline MeLIBA" as definitions we would author and defend.
They are instead existing, peer-reviewed baselines that we reproduce. The burden shifts from
"why is this a fair offline LIAM?" to "does our reproduction follow TAO Appendix F?" — a
mechanical question. Exact specifications are in `docs/baseline_specs.md`; summarized:

| Method | Backbone | Added component |
|---|---|---|
| LIAM-off | ICD (DT) **minus cross-attention** | 2-layer decoder reconstructing `o⁻¹_t, a⁻¹_{t-1}` from the `o¹_t` token embeddings |
| MeLIBA-off | ICD **minus cross-attention** | two-level hierarchical encoders → **permanent latent** ("agent character") + **temporal latent** ("mental state"); decoder = 2 linear + 1 recurrent layer; ELBO over opponent future actions |
| TAO | ICD **with cross-attention** | opponent embedding enters as **key/value in cross-attention** |
| TAGET | DT | high-level module emits a goal token that **replaces RTG in the prompt** |

MeLIBA's permanent/temporal split matches jax-aht's `latent_mean` / `latent_mean_t` that §6
identified independently from the code — two sources agreeing on the decomposition.

Two ways to structure the suite:

- *Bespoke per method*: convert each method's own learner as faithfully as possible. Truest to
  each paper, but then a performance gap between LIAM and MeLIBA is partly attributable to how
  we converted each one — which directly undermines the head-to-head claim that is the entire
  point of the project.
- *Shared backbone*: fix **one** sequence-model backbone for all trajectory-view methods, and let
  each method contribute only its teammate-modeling module.

**Decision: shared backbone.** All trajectory-view methods (LIAM, MeLIBA, TAO, OMIS-without-
search, TAGET) are implemented as a *teammate-modeling module* — a function from interaction
history to a conditioning vector — plugged into a single fixed backbone. Any difference in the
results table is then attributable to the modeling mechanism, which is the comparison the paper
claims to make.

#### [rev 5] The backbone is a Decision Transformer

**Decision: return-conditioned Decision Transformer as the default shared backbone.** This was
checked against the actual code rather than assumed, and the picture is more precise than "they
all use DT":

| Method | Backbone | Return-conditioned? | Verified at |
|---|---|---|---|
| TAO | GPT-2 causal transformer | **Yes** — `embed_return(returns_to_go)` | `offline_stage_2/net.py:54,64` |
| OMIS | shared `trajectory_gpt2` | **Yes** — `returns_to_go` threaded through search | `testing/search.py:182,201`, `pretraining/utils.py:180` |
| TAGET | Decision Transformer (low-level action generation) | **Yes** — predicts teammate-aware return-to-go, then sub-goals | ICML 2025 abstract |
| AD | causal transformer, tokens `(prev_action, prev_reward, obs)` | **No** — cross-entropy on actions, no RTG anywhere in the repo | `ad/model.py:228–242` |
| DPT | causal transformer | **No** — no RTG anywhere in the repo | `dpt/model.py:229` |
| AMAGO-offline | GRU | n/a — not a transformer | §6 |
| Hybrid-AD | CNN + GRU | n/a | §6 |

So: **every trajectory-view method in the suite is already a return-conditioned DT, and no
learning-history method is one.** That is a clean split, and it lands exactly on the boundary
§3.1 already draws. Choosing DT means TAO, OMIS, and TAGET are implemented *closer* to their
originals than any value-learning backbone would put them — the shared backbone stops being a
compromise for those three and becomes the faithful choice. LIAM and MeLIBA are the only methods
whose backbone is genuinely substituted, and TAGET's published offline variants are the
precedent for doing so.

Design consequences:

- **DT is return-conditioned behavior cloning, not a value-learning offline RL algorithm.** Its
  known weakness is trajectory stitching on suboptimal data, which is well documented in the
  D4RL literature. §4.3's `random` / `medium` / `replay` variants are exactly the regimes where
  this bites. Do not treat this as a flaw to hide — it is the mechanism behind §10.1 claim 4.
- **IQL is retained as the backbone-sensitivity ablation**, and rev 5 gives that ablation a much
  better motivation than rev 2–4 had. DT vs. IQL is precisely the return-conditioned-BC vs.
  value-learning contrast, so *dataset variant × backbone* is a substantive result rather than a
  robustness footnote. If the ranking of teammate-modeling mechanisms is stable across both
  backbones, the headline claim is far harder to attack.
- **[rev 6] Two floors, not one: random *and* %BC, in every table.** ICRL4AHT reports AD and DPT
  failing to beat a **random** baseline on a full evaluation track, and going sharply negative
  (AD −18.0, DPT −23.4 vs. random 0.0) against tightly-coupled teammates. A published result of
  "worse than random" means random is the floor that actually binds, and omitting it would be a
  conspicuous gap given that paper exists. %BC remains the second floor.
- **[rev 7] And a ceiling: an oracle teammate model.** LIAM's **FIAM** baseline gives the encoder
  the teammate's *actual* trajectory at execution time rather than inferring it from ego
  observations. Cheap to implement — it is our LIAM module with the input swapped — and it bounds
  how much headroom teammate modeling has at all. Given that the expected result is substantially
  negative (§11), **knowing the ceiling matters as much as knowing the floor**: "every method is
  near random" and "every method is near random *and* the oracle is too" are very different
  papers, and only the second rules out the environments simply not rewarding teammate inference.
- **Do not write the backbones from scratch.** `JAX-CORL` (§2) provides single-file JAX
  implementations of DT, IQL, and four other offline algorithms under MIT license. Adapt
  `algos/dt.py` and `algos/iql.py` rather than reimplementing.
- **[rev 6] The conditioning interface must support three distinct modes**, not one. This is the
  most important design constraint on `offline/conditioning.py` and it is now settled by evidence
  rather than guessed:
  1. **Cross-attention** — TAO feeds the opponent embedding sequence as *key and value* into the
     ICD's cross-attention layers. Not a concatenation.
  2. **Prompt-token replacement** — TAGET's TA-Goal *replaces* return-to-go as the prompt token.
     A conditioning API that only supports "concatenate `z` to the state embedding" cannot
     express TAGET at all.
  3. **Auxiliary head only, no conditioning path** — LIAM-off and MeLIBA-off add reconstruction /
     ELBO heads off the backbone's own token embeddings and do *not* inject a separate `z` into
     the policy input.
  4. **[rev 7] Distribution parameters, not a point embedding.** MeLIBA conditions the policy on
     the full posterior `(μ_t, σ_t)` and argues explicitly that conditioning on a *sample* — as
     LIAM/LIOM do — forecloses Bayes-optimality, *"since the agent cannot take into account its
     uncertainty."* The interface must be able to pass mean **and** variance.

  Design for all four in Phase 0b. Discovering mode 2 or 4 during Phase 2 would mean rewriting
  every baseline built before it.

- **[rev 7] The module factorization is faithful to the originals, not imposed by us.** Both LIAM
  and MeLIBA already train their encoders *independently of the policy gradient* — LIAM: *"we do
  not back-propagate the gradient from the actor-critic loss to the parameters of the encoder,"*
  with separate learning rates; MeLIBA: *"we train the policy using PPO, and do not backpropagate
  the RL-loss through the encoder,"* alternating VAE and RL updates. **This is a stronger defense
  of §3.1 than TAO's "we mandate the same architecture,"** because it says the encoder/policy split
  is what these methods do natively. State it in the paper.
- **[rev 7] Guard against attention entropy collapse in the shared backbone.** AMAGO identifies it
  as *the* instability of long-sequence RL: agents converge on precise recall strategies that
  produce large dot products between a few queries and keys, destabilizing attention. Our
  learning-history contexts are long (ICRL4AHT: 14,600 steps), so this is a live risk for AD, DPT,
  AMAGO **and** for the DT backbone generally. Apply AMAGO's published fixes — Normformer,
  σReparam, and Leaky ReLU activations to preserve plasticity — to the shared backbone rather than
  only to AMAGO. If training curves come out flat, check this first.
- **Learning-history-view methods (AD, DPT, AMAGO, Hybrid-AD) are exempt.** They subsume the
  learner — distilling the improvement operator *is* the method — and the table above shows they
  are also the non-return-conditioned ones. They stay as-is and are reported as a separate
  family. This asymmetry is real and is itself a taxonomy axis (§9).
- **Report the bespoke variants as an ablation if budget allows**, at least for LIAM. Less
  critical than in rev 2–4 now that TAGET supplies the published conversion. Treat as Phase 3,
  cut first if compressed (§10.4).

---

## 4. The dataset specification (the load-bearing contribution)

### 4.1 Two views, not one

The baseline families need structurally different data. This is the single most important
design fact and it should be discovered-and-documented, not papered over:

- **Trajectory view** (LIAM, MeLIBA, TAO, OMIS, TAGET): flat sets of episodes with per-step
  ego and teammate information. D4RL-shaped.
- **Learning-history view** (AD, DPT, AMAGO, Hybrid-AD): *ordered across-episode learning
  curves* — the sequence of episodes an RL agent generated while improving against a fixed
  teammate. In-context RL methods are trained to distill the improvement operator, so shuffled
  trajectories are useless to them. This is what ICRL4AHT's `history.npz` + `histories.h5`
  design exists for.

A single flat D4RL file cannot serve both. The spec must emit both views from one collection
run, sharing episode identity so they can be cross-referenced.

**[rev 2] Build on ICRL4AHT's HDF5 + JSONL-index pair rather than inventing a format.** It is
verified working prior art (`ad/train.py` consumes it end-to-end via `HistoryStore`), and
matching it means AD/DPT/AMAGO need near-zero data-layer adaptation. Extend it with the
trajectory view and the AHT fields below; do not redesign it.

### 4.2 Schema (proposed, to be finalized in Phase 0)

Base D4RL fields, per timestep: `observations`, `actions`, `rewards`, `terminals`, `timeouts`.

AHT extensions, per timestep:
- `teammate_actions` — required by LIAM/MeLIBA reconstruction losses and OMIS's opponent imitator
- `teammate_observations` — **[rev 6] REQUIRED, not optional.** Rev 1–5 marked this "optional but
  collect by default." Three methods make it mandatory: LIAM-off reconstructs `o⁻¹`; MeLIBA-off's
  ELBO conditions on the opponent's *future* observations; TAGET's team-context encoder consumes
  `o⁻¹` and its TA-Goal target is `Concat({o^i_{t+k}})` over all agents. Without this field, three
  of five trajectory-view baselines are not implementable.
- `teammate_rewards` — **[rev 6] new.** TAO's opponent policy encoder fuses `(a⁻¹_{t-1}, r⁻¹_{t-1},
  o⁻¹_t)` per timestep. Cheap to store; omitting it blocks TAO.
- `teammate_id` — ground-truth population index. **[rev 6] Now a hard training-time requirement,
  not a diagnostic**: TAO's discriminative loss is InfoNCE with positives defined by teammate
  policy identity. Still **must be withholdable at eval**.
- **[rev 2]** `avail_actions` — legal-action mask. Required by Hanabi
  (`envs/hanabi/hanabi_wrapper.py:43` exposes `get_legal_moves`); harmless elsewhere. Adding
  Hanabi later without this field in v0 would force a schema version bump, so include it now.
- **[rev 2]** `acting_agent` — whose turn it is. Hanabi is turn-based; LBF and Overcooked are
  simultaneous. Needed so the learning-history view is well-defined under turn alternation.

Per-episode metadata:
- `episode_id`, `episode_return`, `episode_length`
- `teammate_generator` ∈ {`fcp`, `comedi`, `brdiv`, `lbrdiv`, `scripted`, `ippo`}
- `teammate_role` — for BRDiv/L-BRDiv, whether this member is `conf` or `br` (see §7)
- `split` ∈ {`train`, `heldout`}
- `history_id`, `episode_index_within_history` — links into the learning-history view
- `collection_policy` — what generated the ego side (random / IPPO-in-training / expert / mixed)

Dataset-level metadata: env name and full `env_kwargs`, teammate population checkpoint hashes,
collection seed, schema version, generating code commit, **[rev 6]** the trajectory-filtering
configuration (§4.4) and whether trajectory mirroring was applied (§4.5).

### 4.3 Dataset variants

Offline RL results are notoriously dataset-quality-dependent, and D4RL's main methodological
contribution was *naming* those regimes. Mirror it:
**[rev 7] Match D4RL's definitions exactly, or state the deviation.** Read against the source:

- `random` — ego is a **randomly initialized** policy, unrolled.
- `medium` — train the ego online, **early-stop**, collect from the partially-trained policy.
- `expert` — converged ego, **per-teammate best responses** (see below).
- `replay` — D4RL's `medium-replay` is *"all samples in the replay buffer observed during training
  **until the policy reaches the medium level of performance**"* — it **stops at medium**, it is
  not the full training history. Rev 1–6 described ours as "the full learning history." **Decide
  deliberately**: AD/DPT need the *whole* improvement curve, so our `replay` probably should run
  to convergence — but then it is not D4RL's `medium-replay` and must not be labelled as though it
  were. Recommend naming ours `replay-full` and documenting the difference.
- `mixed` — D4RL's `medium-expert` mixes **equal amounts** of expert and suboptimal data. Rev 1–6
  said "expert + medium union" without a ratio. **Specify 50/50**, since mixture ratio is known to
  change results.

An offline AHT benchmark that reports only one data regime is not answering the question it
claims to. Budget all five regimes for the Tier 1 environments and `replay` + `mixed` for Tier 2
(§10.3). The dataset variant and the environment tier are the two dimensions of the matrix that
can be traded against each other late; the environment *set* cannot (§10.4).

**[rev 6] `expert` is mandatory, and it is the highest-leverage artifact in the project.**
Per-teammate best-response policies serve **three** independent purposes:
1. **OMIS training data** — its actor is BC on best-response actions against each teammate.
2. **TAO stage-2 targets** — its ICD predicts "near-optimal actions" `a^{1,*}`.
3. **BR-Prox normalization** (§8) — the metric's denominator is the approximate BR's return.

One artifact, three consumers. This also means `expert` cannot be a single global expert: it must
be a *per-teammate* BR set, and it should be computed early because two baselines and one metric
all block on it.

**[rev 6] OMIS is structurally excluded from part of the dataset-quality matrix.** Because it
trains on best responses, it cannot be meaningfully trained on `random` or `medium`. Report this
as a property rather than a gap — it is a sharp instance of §9 axis 5 (data requirements), and no
other baseline has it.

### 4.4 [rev 6] Trajectory filtering is a required stage, not an optional one

ICRL4AHT filters **1,196,032,000 → 149,504,000** transitions — an **8× reduction** — *"select[ing]
high-quality learning curves based on final performance and improvement, ensuring the Transformer
learns genuine improvement dynamics."* Rev 1–5's collection spec has no filtering stage at all.

Without it, the `replay` variant is dominated by learning curves that never improved, and the
learning-history family is trained largely on noise — which is one candidate explanation for the
flat adaptation curves ICRL4AHT reports. **Add a documented, configurable filtering stage to
`dataset/construction/collect.py`**, record the filter in dataset metadata, and treat the filter definition as
part of the benchmark contract: it is a dataset-design decision that materially changes results,
so two groups using different filters are not running the same benchmark.

Worth an explicit experiment: **filtered vs. unfiltered `replay` is a cheap, novel ablation** that
no source paper reports, and it speaks directly to §10.1 claim 4.

### 4.5 [rev 6] Trajectory mirroring

TAGET's data pre-processing (its Alg. 1) re-orders each episode `N` times so every agent takes a
turn as ego, yielding `N` ego-perspective trajectories per episode. It is ablated with consistent
degradation when removed, and it is a **dataset technique rather than a method component** — so it
belongs in our collection layer where every trajectory-view baseline benefits, not inside one
baseline.

Constraint: mirroring is only well-defined when agents share observation and action spaces. It
works for LBF and Hanabi; **several Overcooked-v1 layouts have asymmetric roles**, so it must be
per-environment opt-in and recorded in metadata.

### 4.6 [rev 6] Storage budget

ICRL4AHT's learning-history dataset is **≈6.5 GB compressed for a single environment**
(OvercookedV2, 80 teammates, history length 14,600, obs `(5,5,★)`). We plan seven environment
configurations × four generators × up to five dataset variants. Naive extrapolation puts the
artifact in the **hundreds of GB**, which affects collection time, cluster storage, and whether
the release is hostable at all.

Phase 0a must produce a storage estimate and a decision on **what is released versus what is
regenerable from configs and population checkpoints**. The likely answer is: release the
populations, the manifests, and one canonical dataset per environment; make the rest reproducible.
Deciding this late means either a truncated release or a scramble for hosting.

---

## 5. Repository layout

New standalone repo (not inside bayes-tom), named `oaht-bench`. Naming follows the literal
convention of ZSC-Eval and BenchMARL rather than a backronym, so it parses on sight in a
reference list — which matters more than memorability for a paper that is half survey.

```
oaht-bench/
  envs/                     # BaseEnv ABC + LBF, Overcooked-v1, Hanabi (from jax-aht envs/)
  teammate_generation/      # FCP, CoMeDi, BRDiv, L-BRDiv (from jax-aht, already env-generic)
  agents/                   # policy/population interfaces, actor-critic variants
  data/
    schema.py               # the spec in §4, versioned + validated
    collect.py              # rollout -> both views
    build_index.py          # -> HDF5 + jsonl index (pattern from ICRL4AHT scripts/build_index.py)
    loaders.py              # framework-neutral readers (numpy); optional jax/torch adapters
  offline/                  # [rev 5] shared sequence-model backbone (§3.1)
    dt.py                   #   Decision Transformer — default backbone, adapted from JAX-CORL algos/dt.py
    iql.py                  #   backbone-sensitivity ablation, adapted from JAX-CORL algos/iql.py
    pct_bc.py               #   filtered-BC reference floor
    conditioning.py         #   module(history) -> z interface; how z enters the DT token stream
  baselines/
    ad/ dpt/ amago/ hybrid_ad/     # adapted from ICRL4AHT; learning-history family, exempt from backbone
    liam/ meliba/                  # modeling modules; jax-aht heads, offline conversion per TAGET
    tao/ omis/                     # reimplemented in JAX; already return-conditioned DTs
    taget/                         # [rev 5] reimplemented from the ICML 2025 paper, no code released
  evaluation/
    protocol.py             # splits, seeds, episode budgets
    metrics.py              # return, normalized return, BR-Prox, cross-play matrix
    crossplay.py            # heterogeneous-population XP, vmapped (jax-aht heldout_crossplay pattern)
  cluster/                  # [rev 2] SLURM submission, sweep configs, checkpoint/resume
  configs/                  # hydra; task/ algorithm/ dataset/ eval/
  scripts/
  docs/
```

---

## 6. Baseline inventory and adaptation cost

**[rev 5]** Costs revised for the §3.1 DT-backbone decision. "Modeling module" means the method
contributes only `history -> z`; the DT backbone is shared. Ten baselines: %BC as the floor,
four learning-history methods, five trajectory-view methods.

| Baseline | Source | Framework | Data view | Test-time needs | Cost | Notes |
|---|---|---|---|---|---|---|
| Random | — | JAX | none | none | **S** | **[rev 6]** hard floor. ICRL4AHT reports AD/DPT failing to beat it |
| **Oracle (FIAM-style)** | LIAM | JAX | trajectory + teammate_obs | **privileged teammate trajectory at execution** | **S** | **[rev 7] ceiling, not a competitor.** LIAM's own upper-bound baseline: the encoder sees the teammate's actual trajectory at test time instead of inferring it. Bounds how much headroom teammate modeling has at all |
| %BC | JAX-CORL | JAX | trajectory | forward only | **S** | second floor, no modeling module |
| Prompt-DT | TAO App. F | JAX | trajectory | forward only | **S** | **[rev 6]** prompts = top-20%-by-return trajectories vs. each teammate, consecutive fragments of length 5. Baseline in *both* TAO and TAGET; cheap on our DT backbone |
| AD | ICRL4AHT | JAX | learning-history | forward only | **S** | swap CNN encoder for env-generic obs encoder. Not return-conditioned (§3.1) |
| DPT | ICRL4AHT | JAX | learning-history | forward only | **S** | same encoder issue; not return-conditioned |
| AMAGO-offline | ICRL4AHT | JAX | learning-history | forward only | **S** | GRU trajectory encoder |
| Hybrid-AD | ICRL4AHT | JAX | learning-history | forward only | **S** | CNN+GRU; may be Overcooked-specific — verify |
| LIAM | jax-aht + **TAO App. F** | JAX | trajectory | forward only | **M** | head is reusable; **online PPO learner is not** (§3.1). **Offline conversion fully specified in TAO Appendix F** |
| MeLIBA | jax-aht + **TAO App. F** | JAX | trajectory | forward only | **M** | permanent + temporal latents, matching `latent_mean` / `latent_mean_t`. **Also fully specified in TAO Appendix F** (not TAGET — TAGET does not convert MeLIBA) |
| TAO | local PyTorch | **reimplement in JAX** | trajectory + **teammate_id** + **expert** | forward only | **M** ↓ | already a return-conditioned DT. Needs identity labels (InfoNCE) *and* best-response targets |
| OMIS | local PyTorch | **reimplement in JAX** | trajectory + **expert** | **env simulator** | **M** + **M** ↓↓ | **[rev 6] search downgraded from L**: it is a flat vmappable rollout estimator, not a tree search (§10.5) |
| MBOMIS | local PyTorch | **reimplement in JAX** | trajectory + **expert** | learned dynamics | **S** on top of OMIS | **[rev 6] new.** OMIS with a learned `P̂` instead of the true simulator. Makes the §6 asymmetry *measurable* rather than merely discussed |
| TAGET | ICML 2025 paper | **reimplement in JAX** | trajectory + **teammate_obs** | forward only | **M–L** | no code. Hierarchical TA-RTG + TA-Goal over a DT; full spec extracted in `baseline_specs.md`. `goal_steps` needs per-env sweeping |

Note the DT backbone *lowers* TAO's and OMIS's cost — they are already return-conditioned DTs, so
this is closer to their originals than a value-learning backbone would be — while LIAM and MeLIBA
are the only genuine substitutions, and TAGET supplies the published precedent for those. Net
effect on total work is roughly neutral; net effect on comparability and defensibility is large.

**The OMIS asymmetry is a protocol problem, not just an engineering one.** OMIS requires
environment access at deployment to run search; every other baseline needs only a forward pass.
Reporting them in one column is not a fair comparison. Handle it explicitly: report search and
no-search tracks separately, and report OMIS-without-search as an ablation so it has an
apples-to-apples entry. This asymmetry is also a survey taxonomy axis (§9) — a good example of
benchmark-building surfacing a real conceptual distinction.

**[rev 6] MBOMIS resolves the asymmetry properly.** The OMIS paper itself reports a model-based
variant that learns `P̂` from `(s, a, r, s')` tuples and *"still effectively improves over OMIS w/o
S and generally surpasses other baselines."* That converts OMIS from "requires a privileged
simulator" into "forward-only plus a learned model," which is directly comparable to every other
baseline. Report **three** OMIS rows — no-search, learned-dynamics (MBOMIS), true-simulator — and
the asymmetry becomes a measured quantity rather than a caveat.

---

## 7. Teammate generation

Four algorithms: FCP, CoMeDi, BRDiv, L-BRDiv. **[rev 2] jax-aht already implements all four,
env-generically, with configs for all three target environments** — `algorithm/{fcp,comedi,
brdiv,lbrdiv} × task/{lbf(2 configs), overcooked-v1(5 layouts), hanabi(2 configs)}`. Rev 1
planned to write this layer from scratch borrowing from both repos; that work is unnecessary.
This layer is a fork-and-configure task, not an implementation task, and it is the main reason
the three-environment scope in §12 is affordable.

### 7.1 [rev 4] We train all populations ourselves

**Decision: train all four generators across all seven environment configurations in-house,
with per-environment hyperparameter tuning, and release the result as contribution 3 (§1).**

The alternative — downloading jax-aht's published populations — was considered and rejected as
the *primary* path, though those populations remain valuable (§7.3). The reasons to train:

- **No source provides the full matrix.** Verified against the `jaxaht/eval-teammates`
  HuggingFace dataset (public, ungated, 1619 files): FCP populations are absent for every
  environment except LBF; full Hanabi has no generator populations at all (only IPPO, BC, and
  OBL-R2D2); `counter_circuit` lacks BRDiv. Filling the gaps piecemeal would produce a matrix
  where some cells are ours and some are inherited under unknown hyperparameters — the exact
  inconsistency this benchmark exists to eliminate.
- **Population quality is a confound we must control.** Every downstream dataset and every
  baseline result is conditioned on the populations. If they were trained under settings we
  cannot inspect, "what is the state of the art" becomes partly a question about someone else's
  tuning. Training them ourselves makes the whole pipeline reproducible from configs.
- **Cost is not the obstacle it appeared to be.** Measured on LBF, on CPU: FCP ≈ 15 min,
  CoMeDi ≈ 20 min, BRDiv and L-BRDiv ≈ 4 h each. Roughly 9 CPU-hours for a complete
  four-generator LBF set, trivially parallel across the 28 (generator × env-config) cells and
  far cheaper on the GPU cluster. Population training is **not** the schedule risk; the baseline
  training matrix is (§11).
- **The tuning knowledge is a contribution.** See §7.2.

One honest caveat: the LBF timings should not be extrapolated linearly to full Hanabi. jax-aht's
own IPPO Hanabi baseline is checkpointed at `mlp_1e9_3seeds` — 10⁹ timesteps — which suggests
Hanabi is substantially more sample-hungry than LBF. Budget Hanabi population training as the
long pole among the environments and start it first (§10.2). `mini-Hanabi` (3 colors, 3 ranks,
hand size 3) is retained as a fast development and debugging configuration even though full
Hanabi is what appears in the results.

### 7.2 [rev 4] Per-environment tuning as a documented artifact

Rev 1–3 recorded the scaling mechanics below as "guidance for anyone regenerating populations."
Under §7.1 they become something stronger: we will have tuned all four generators across three
structurally different environments — a gridworld, a spatial coordination task, and a turn-based
partially-observable card game — and that tuning record is publishable knowledge. Nobody has
reported it. Release the swept ranges and chosen values per (generator, environment), not just
the final configs, and state which of the mechanics below bit in which environment.

**[rev 7] Things the papers already settle, which the tuning record should adopt rather than
rediscover:**
- FCP's mid checkpoint = **50% of final reward** (§7.3), not a step-schedule snapshot.
- **Do not sweep architectural diversity** in populations — FCP's `FCP₊A` ablation shows no gain
  over checkpoint diversity.
- CoMeDi's `β` (mixed-play weight) is **load-bearing, not a nuisance parameter** — it is what
  prevents handshake degeneracy (§7.4). Record it prominently.
- L-BRDiv's `LAGRANGE_LR` must scale ~(n_ref/n)²; BRDiv needs no equivalent correction (§7.3).
- TAGET's `goal_steps` is per-environment and **non-monotonic** (PP 6, LBF 2, Overcooked 3 in the
  original); it must be swept per environment and belongs in the same record even though it is a
  baseline hyperparameter rather than a generator one.

### 7.3 Verified mechanics the implementation and docs must preserve

Verified mechanics that the implementation and the docs must preserve:

- **FCP** — independent self-play runs across seeds, snapshotted during training. Saved
  `final_params` has `pop_size = PARTNER_POP_SIZE` (one selected checkpoint per run, via
  `final_ckpt_idx`); the raw `checkpoints` tensor holds the full `PARTNER_POP_SIZE ×
  NUM_CHECKPOINTS` history. Diversity is *of competence*, not of identity — FCP has no
  cross-play term at all, and "is π_i its own best response" is not a question FCP is designed
  to answer. **[rev 7] Two precise facts from the paper.** (a) The mid-training checkpoint is
  defined as **the point where the agent reaches 50% of its final reward** — not an arbitrary
  snapshot. ZSC-Eval independently adopts the same half-performance criterion, so two sources
  converge; record it in §7.2 and implement it rather than checkpointing on a step schedule.
  (b) FCP's own ablations settle a tuning question for us: removing past checkpoints (`FCP₋T`)
  **significantly** degrades performance, while varying **architecture** across the population
  (`FCP₊A`) yields **no improvement**. → Spend the population budget on checkpoint diversity;
  do not spend it on architectural diversity.
- **CoMeDi** — single population, no conf/br split (`final_params_conf` is the only params key
  saved). Explicitly maximizes self-play and *minimizes* cross-play: `CoMeDi.py:760`,
  `xp_pg_weight = -config["COMEDI_ALPHA"]`. π_i-vs-π_i **is** the designed-optimal pairing.
  **[rev 7]** Two refinements from the paper: construction is **greedy/sequential** (conventions
  added one at a time), and cross-play is minimized **only against the single most-compatible
  existing convention**, not against all of them —
  `L(π_n) = −J(π_n,π_n) + α·J(π_n,π*) − β·J_M(π_n,π*)` with `π* = argmax_{D_{n-1}} J`. The third
  term is **mixed-play**, and it is not optional (below).
- **BRDiv / L-BRDiv** — paired populations. The designed-optimal pairing is **conf_i vs br_i**,
  not br_i vs br_i. Any evaluation that loads only the `br` set into both seats is testing an
  out-of-distribution pairing; that is a legitimate stress test but must not be reported as the
  algorithm's own target. Both save `final_params_conf` and `final_params_br`, and both embed an
  `all_pair_returns` conf×br matrix computed during training.
- **Population-size scaling is not free.** `conf_ids`/`br_ids` are sampled independently and
  uniformly per env, so a given policy's self-play draw probability is **1/n², not 1/n** —
  growing the population quadratically dilutes per-policy self-play data at fixed
  `NUM_ENVS`/`TOTAL_TIMESTEPS`. Separately, L-BRDiv's Lagrange multiplier gradient is an
  **unnormalized sum over ~n² pair terms**, so `LAGRANGE_LR` must be scaled by ~(n_ref/n)² or the
  multiplier update overcorrects (observed: entropy runaway to ~49, pg_loss to −25 at n=5 with
  the n=3 value).
- **BRDiv's `XP_LOSS_WEIGHTS` genuinely does not need rescaling — verified in code.**
  `BRDiv.py:389–391` constructs the policy-gradient weights as
  `sp_weight = (1 + 2·XP_LOSS_WEIGHTS)·(n/2)` and `xp_weight = XP_LOSS_WEIGHTS·(n/(2(n−1)))`.
  Against the sampling distribution `P(SP) = 1/n`, `P(XP) = (n−1)/n`, the expected per-sample
  contributions are `P(SP)·sp_weight = (1 + 2·XP_LOSS_WEIGHTS)/2` and
  `P(XP)·xp_weight = XP_LOSS_WEIGHTS/2` — **both exactly independent of n** (checked numerically
  at n = 3, 5, 10, 20: 0.55 and 0.025 throughout, for `XP_LOSS_WEIGHTS = 0.05`). The `n` factors
  in the implementation exist precisely to cancel the sampling probabilities. **Do not "fix" it.**

  > **[rev 8] Retraction of a rev 7 "correction".** Rev 7 claimed this statement was imprecise,
  > on the grounds that expanding the *paper's* BRDiv metric (Eq. 6) gives diagonal weight
  > `1 + 2(K−1)` against off-diagonal `−1`, a ratio that varies with K. That reasoning conflates
  > two different objects: the paper's **diversity metric** and the implementation's **per-sample
  > policy-gradient weights**, which additionally compensate for the sampling distribution. The
  > rev 1–6 claim was correct as written; rev 7 replaced it with a confused one. Reinstated above
  > with the code-level derivation that settles it.

- **[rev 8] The n=5 BRDiv diversity collapse was a sample-count problem, not a loss-balance one.**
  Raising `PARTNER_POP_SIZE` 3 → 5 at fixed `NUM_ENVS`/`TOTAL_TIMESTEPS` produced a near-flat
  cross-play matrix. The fix that worked was scaling the *budget* — `NUM_ENVS` 64 → 128 and
  `TOTAL_TIMESTEPS` 4.5e7 → 7e7, ≈3× combined, restoring per-policy self-play sample count
  against the 1/n² dilution — **not** touching `XP_LOSS_WEIGHTS`. This is the single most useful
  piece of population-scaling guidance we have, because the symptom (collapsed diversity) points
  at the diversity weight while the cause is elsewhere.

  **Status: direction confirmed, values not final.** These came from exploratory runs, not a
  tuning sweep. Treat the *directions* — scale budget with n², scale L-BRDiv's `LAGRANGE_LR` by
  (n_ref/n)², leave BRDiv's `XP_LOSS_WEIGHTS` alone — as established, and the specific numbers as
  starting points for §7.2's sweep rather than as the tuned configuration.

All of this belongs in the benchmark's documentation as guidance, since anyone regenerating
populations at a different size will hit it.

### 7.4 [rev 7] The handshake confound — a threat to §8's cross-play diagnostic

CoMeDi §3.2 documents a failure mode of cross-play minimization that directly contaminates our
headline population diagnostic. Agents learn a **handshake**: at the first timestep both emit an
identity-revealing action; if the handshakes match they cooperate, otherwise they **deliberately
sabotage the episode**. The result is high self-play and low pairwise cross-play *while the
conventions remain semantically similar* — the diversity metric is fooled.

§8 uses the cross-play matrix `C[τ, j]` as a primary diagnostic of population structure. **If any
population handshakes, low off-diagonal entries measure sabotage signalling rather than genuine
convention difference**, and every conclusion drawn from that matrix is contaminated. Two actions:

1. **Verify mixed-play is enabled** in our CoMeDi runs and record `β` in population metadata.
   Mixed-play is the published fix — it rolls out random self-play/cross-play mixtures for a random
   first phase so agents cannot infer partner identity and therefore cannot safely sabotage.
   jax-aht implements CoMeDi, so this is a check, not new work.
2. **Add a handshake probe.** Compare cross-play return when the first `k` timesteps are forced to
   self-play actions against unforced cross-play; a large gap indicates handshaking. Cheap, and I
   have not found it reported anywhere — a plausible small contribution, and exactly the kind of
   thing a benchmark paper should own.

Populations must be released as checkpoints with hashes recorded in dataset metadata, so results
are reproducible without retraining.

**[rev 7] LBF populations are known-incomplete, even for the method designed to complete them.**
L-BRDiv reports recovering only **4–5 of the 6** possible food-collection orderings that make up
the MCS in its own LBF, with BRDiv and LIPO finding fewer. Our LBF is a different configuration,
but the implication carries: **no generator fully covers the convention space in LBF**, so
"held-out teammate" does not mean "the population spanned the space and we withheld part of it."
State this when reporting LBF results rather than letting readers assume coverage.

### 7.5 [rev 4] Existing checkpoints as validation references

Not the primary path (§7.1), but useful in two specific ways, and free:

- **Sanity-checking our training.** Where jax-aht published a population for a cell we are also
  training (CoMeDi/BRDiv/L-BRDiv on Overcooked layouts and mini-Hanabi; CoMeDi/L-BRDiv on LBF),
  comparing our cross-play matrices against theirs is a cheap correctness check. A large
  divergence means our tuning is off, and finding that in week 3 is much better than inferring
  it from anomalous baseline results in week 20.
- **`best_heldout_returns.zip`** supplies the performance-bounds data §8 needs for normalized
  return, which would otherwise require computing per-teammate best responses ourselves.

Local starting point: `bayes-tom/checkpoints/lbf/lbf_12x12/{fcp,comedi,brdiv,lbrdiv}` (~12 MB,
all four generators) lets Phase 0's data-collection and schema work begin on day 1, before any
new population finishes training.

### 7.6 [rev 4] Out-of-distribution teammates are already available

§8's OOD evaluation condition needs scripted and heuristic teammates never seen during training.
Rev 1–3 assumed we would write these. jax-aht ships them for all three environments:

- `agents/lbf/` — entitled, greedy-heuristic, sequential-fruit, random
- `agents/overcooked/` — onion, plate, independent, static, random, BC (with featurizer)
- `agents/hanabi/` — cautious, flawed, iggi, internal, outer, piers, rule-based, smartbot,
  van-den-bergh, random, plus BC-LSTM (weights shipped in-repo) and OBL-R2D2 (download script)

Hanabi's set is unusually rich and includes agents with genuinely different conventions, which
makes it the strongest OOD condition in the suite — another argument for Hanabi carrying real
weight in the results rather than being a token third environment.

---

## 8. Evaluation protocol

Fix these before any baseline is run, and never vary them per-method:

- **Teammate splits.** At minimum two eval conditions: (a) *within-distribution* — held-out
  members from the same generators seen at training; (b) *out-of-distribution* — heuristic and
  scripted teammates never seen at training (ICRL4AHT's deliberate design; keep it). Optionally
  (c) *cross-generator* — train on FCP, test on CoMeDi/BRDiv, etc.
- **[rev 6] Graded distribution shift, not a binary split.** Three independent sources grade it,
  and none of them use two points: OMIS sweeps `[seen:unseen]` ratios
  `{[10:0], [10:5], [10:10], [5:10], [0:10]}`; ICRL4AHT orders heuristic families by
  *cooperability* (`H1 recipe_aware < H2 territory < H3 assembly_line < H4 utility_greedy`, H1
  demanding the most of the ego agent); TAO reports seen / unseen / **mix**. Adopt a graded axis.
  jax-aht's shipped scripted agents (§7.6) can be cooperability-ordered per environment, and the
  seen:unseen ratio sweep is nearly free given held-out populations. **A dose-response curve is a
  strictly better result than two bars**, and ICRL4AHT's headline average is explicitly distorted
  by including an easy family — grading prevents us repeating that.
- **[rev 6] Two evaluation tracks, from ICRL4AHT.** Track 1 *teammate generalization*
  (`L_train × Π_test`); Track 2 *layout/config generalization* (`L_test × Π_test`) — a dual shift
  where both teammate and environment configuration are unseen. Track 2 is where ICRL4AHT found
  random beating AD outright, so it is the more discriminative of the two. Overcooked's five
  layouts and Hanabi's configuration variants give us natural Track 2 pairs.
- **[rev 6] Non-stationary teammates as a secondary condition.** TAO switches the teammate every
  `E = 50` episodes; OMIS sweeps `E ∈ {2, 5, 10, 20, dynamic}`. Both treat mid-deployment
  switching as central; our protocol assumes a fixed teammate throughout. This is the condition
  in-context methods should theoretically win, and the cooperative-AHT literature has **not**
  converged on it while the competitive-OM literature has — itself a survey observation (§9).
- **Primary metric**: mean episodic return against held-out teammates, plus per-teammate
  normalized return using performance bounds (jax-aht's `performance_bounds` pattern) so
  environments and teammates with different return scales can be aggregated honestly.
- **Secondary — BR-Prox**, now with the exact definition from ZSC-Eval §4.3:

  ```
  BR-Prox(π, {π_w^i}_{i∈P}) := Aggr_{L ⊆ P} [ J(π, {π_w^i}_{i∈L}) / J(B̂R({π_w^i}_{i∈L}), {π_w^i}_{i∈L}) ]
  ```

  Ego return over approximate-best-response return, aggregated across partner subsets. **`Aggr` is
  the inter-quartile mean (IQM)**, not the mean — following `rliable` — reported with 95% CIs and
  inter-quartile ranges over disaggregated scores. Requires the per-teammate BRs that §4.3 already
  makes mandatory.
- **[rev 6] Adaptation gain** — mean return over the last 20 evaluation episodes minus the first
  20. ICRL4AHT's diagnostic, and the one that exposed the flat adaptation curves showing in-context
  methods were not adapting at all. Nearly free to compute, and our current metric suite would not
  have caught it. **This is the metric that directly tests the claim in-context methods make**, so
  it arguably belongs alongside return as primary rather than diagnostic.
- **Diagnostic**: cross-play matrix `C[τ, j]` over the full mixed population; teammate-identity
  recoverability from each method's internal representation (retrieval P@k, cosine gap,
  episode-disjoint linear probe — the instrument already exists in
  `bayes-tom/scripts/diagnose_embedding_headroom.py` and is representation-agnostic).
  **[rev 7] This diagnostic is established practice, not our invention** — MeLIBA already trains a
  logistic-regression classifier on its latents to predict teammate type (its "Agent Type
  Prediction Accuracy"), and LIAM's CBAM baseline makes identity classification the method itself.
  Cite both, and use MeLIBA as the sanity-check case: if our probe cannot recover identity from
  MeLIBA's latents, the probe is broken rather than the representation.
  **[rev 7] Also run the probe on the oracle row (§6).** Identity recoverability from a privileged
  encoder is the natural upper bound for the diagnostic, in the same way FIAM bounds return.
  **[rev 2]** Under §3.1 this diagnostic gets sharper: every trajectory-view method now exposes
  its conditioning vector `z` through one common interface, so the probe applies uniformly
  across the suite by construction rather than per-method plumbing.
- **Budgets and seeds**: fixed episode counts, fixed eval seeds, common random numbers across
  methods where possible. **[rev 6] Report IQM with 95% CI and IQR (`rliable`)** rather than bare
  mean ± CI, consistently with BR-Prox above. Cross-play cells need enough episodes that
  differences exceed noise — the 10-episode LBF pilot produced ~0.02 SEM against effect sizes of
  ~0.1, which is too thin to rank neighbors.
- **[rev 6] Episode budgets in this literature, for calibration**: TAO **2500**, OMIS **1200**,
  ICRL4AHT **100 per teammate instance** (5 instance-level means), TAGET **50**. The spread is
  itself evidence for the thesis. Budget nearer the top of that range: TAGET's headline LBF margin
  (0.140 ± 0.080 vs. DT 0.098 ± 0.008) has a confidence interval overlapping the baseline it
  claims to beat by 37%, which is exactly the failure mode adequate budgets prevent.
- **[rev 6] Held-out teammate selection by BR-Div, not at random.** ZSC-Eval shows that maximizing
  *partner* diversity is not the same as maximizing the diversity of the *best responses* those
  partners require — different partners often need similar BRs, so a partner set that looks
  diverse can test a narrow band of ego skills. They select by DPP over `det(K)` where `K_ij =
  θ_i · θ_j` on behavior features. Given we compute per-teammate BRs anyway (§4.3), BR-Div
  selection is nearly free and strictly better-founded than a random split.
- **Compute reporting**: wall-clock and FLOPs at train and test time. With a search-based method
  in the suite, test-time compute is a first-class axis, not a footnote.

---

## 9. Survey taxonomy

Axes to organize the survey half, chosen so the benchmark work directly populates them:

1. **What is offline** — teammate data only? ego training too? is any online interaction allowed?
2. **Adaptation mechanism** — in-context sequence modeling (AD, DPT, AMAGO, TAO, OMIS) vs.
   explicit latent encoder (LIAM, MeLIBA) vs. hierarchical goal/return prediction (TAGET) vs.
   decision-time search (OMIS). **[rev 5]** Bayesian-posterior-over-types methods are surveyed
   but not benchmarked, since no such method is in the baseline suite.
3. **Test-time requirements** — forward pass only vs. environment simulator vs. gradient updates
   vs. external model calls. (Surfaced by the OMIS asymmetry in §6.)
4. **Teammate representation** — none / latent vector / explicit posterior / natural language.
5. **Data requirements** — needs ordered learning histories? teammate action labels? teammate
   *identity* labels? This axis is the one that only becomes visible from building §4, and it
   partitions the field more sharply than architecture does. Likely the survey's most novel
   organizing contribution.
6. **Distribution-shift assumptions** — what is assumed about the gap between training and
   deployment teammates.
7. **[rev 2] Learner/modeler separability** — does the method factor into a teammate model plus a
   policy learner, or does it subsume the learner entirely (AD/DPT/AMAGO)? This axis was forced
   into visibility by §3.1: the methods that resisted the shared backbone are exactly the ones
   whose adaptation *is* their learning algorithm. That is a real structural distinction in the
   field and it was invisible until we tried to unify them.
8. **[rev 6] Teammate-population construction.** Four papers, four incompatible philosophies:
   diversity-optimizing generators (our FCP / CoMeDi / BRDiv / L-BRDiv), soft-value diversity
   (TAGET's SVD from CSP), maximum-entropy population training (OMIS's MEP), and hand-built mixes
   of scripted policies with RL checkpoints at varying training durations (TAO). Population
   construction is as unstandardized as environments and metrics, and it is arguably more
   load-bearing since every dataset is conditioned on it. Note that TAO's "RL checkpoints at
   different training durations" is FCP's competence axis assembled by hand.
9. **Diversity of partners vs. diversity of required responses.** ZSC-Eval distinguishes P-Div
   from BR-Div and shows they come apart empirically: different partners often share similar best
   responses, so a partner set that looks diverse can exercise a narrow band of ego skills.
   **[rev 7] Corrected.** Rev 6 claimed all four of our generators are population-oriented. That
   is wrong for BRDiv and L-BRDiv, whose entire framing is the **Minimum Coverage Set** — the
   smallest set containing a best response to every teammate in Π. The accurate split is:
   - **Population-oriented**: FCP (competence diversity), CoMeDi (reward-aligned semantic
     diversity via cross-play minimization)
   - **Response-oriented**: BRDiv, L-BRDiv (best-response coverage)
   - **Selection rather than generation**: ZSC-Eval's BR-Div applies the response-oriented idea to
     *choosing* an evaluation set from candidates

   This is better for the paper than the blanket critique rev 6 made: **our four generators
   already span the distinction**, so we can measure whether it matters instead of merely
   asserting it should. That is a cleanly testable claim and a candidate for §10.1.
10. **[rev 6] Conditioning mechanism.** Cross-attention (TAO), prompt-token replacement (TAGET),
    auxiliary reconstruction head with no conditioning path (LIAM, MeLIBA), in-context data
    concatenation (AD, DPT, OMIS). Surfaced by having to build one interface that expresses all
    of them (§3.1), and finer-grained than axis 2's "adaptation mechanism."

---

## 10. Phasing and schedule (ICML 2027)

### 10.1 What ICML requires from the results

ICML is a main-track venue with no datasets-and-benchmarks track. **A benchmark paper that
offers only an artifact does not clear its bar; it needs a finding.** This is the single most
important consequence of the venue choice, and it must shape data collection rather than be
discovered during writing. Nominate candidate headline claims now and make sure the experiment
matrix can support or refute each one:

1. **The field's ranking is not what the literature implies.** Methods are currently compared
   only transitively, through incomparable settings. If a shared protocol reorders them — and
   especially if %BC or random beats published methods on some regimes — that is the paper's
   headline. **[rev 6] Partly pre-empted:** ICRL4AHT already demonstrates this for the
   learning-history family in one environment (AD and DPT failing to beat random, going sharply
   negative on tightly-coupled teammates). Our version must be the part they explicitly disclaim —
   multiple environments, multiple population-construction methods, and the *trajectory-view*
   family, which nobody has tested this way. Replicating their result on Overcooked alone adds
   nothing.
2. **Data requirements predict performance better than architecture does** (§9 axis 5). If the
   learning-history family and the trajectory family separate cleanly, and that separation
   tracks the data axis rather than the sequence-model-vs-latent-encoder axis, the taxonomy earns
   its place instead of being descriptive scaffolding.
3. **Teammate-identity recoverability is necessary but not sufficient for adaptation.** The
   probe from §8 applies uniformly across the suite under §3.1. If representations that recover
   identity well do not produce better returns, that is a substantive negative result about what
   the field has been optimizing.
4. **Dataset quality interacts with adaptation mechanism.** D4RL's lasting contribution was
   showing that regime choice reorders methods. If it does so here too, one-dataset AHT papers
   are shown to be under-evidenced — a methodological finding with teeth.

Each of these is falsifiable, and each is answerable from the Phase 3 matrix. Decide by the end
of Phase 1 (Oct 30) which two are primary, and let that prune the rest. Do not run the full cross
product and hope a story emerges — at five months there is no budget for that, and §10.3 shows
the cross product is ~400 training runs.

**[rev 4]** Claim 2 gets materially stronger under the environment-first plan. With three
structurally different environments — gridworld, spatial coordination, turn-based hidden
information — a data-requirements split that holds across all three is a much harder result to
dismiss than one observed on LBF alone. If the primary claims are chosen well, environment-first
is not just a scheduling choice; it is what makes the findings load-bearing.

The survey half is largely **decoupled from results** and should be drafted during Phases 0–1
while infrastructure is being built, not deferred to Phase 4. Rev 2 put survey prose in Phase 4;
at this timeline that is a scheduling error, since it is the one deliverable that parallelizes
cleanly against engineering.

### 10.2 Schedule (environment-first)

Roughly 25 weeks to deadline, of which the last ~3 are writing and final results only.

**The rev 4 restructure:** all three environments are stood up in Phase 0 rather than laddered,
so Phase 3 dissolves. The rationale is that the expensive part of multi-environment support was
never the environment wrappers or the populations (§7.1) — it is that **every interface written
against one environment has to be revisited for the others**. Encoders, action heads, the
learning-history view's episode segmentation, and the eval harness all change shape when Hanabi's
action masking and turn alternation arrive. Doing that once, up front, against the hardest
environment costs less than doing it once per baseline in December. It also converts "the
abstractions generalize" from a late empirical bet into a structural property of the codebase.

| Phase | Window | Deliverable | Gate |
|---|---|---|---|
| **0a** | Aug 3 – Aug 28 (4 wk) | All 3 envs wired from jax-aht; `cluster/` scaffolding; population training + per-env tuning launched, **Hanabi first** (§7.1); schema v0 incl. `avail_actions`/`acting_agent`; survey drafting starts | All 7 env-configs instantiate and roll out; LBF populations reproduced and cross-checked against jax-aht's (§7.5) |
| **0b** | Aug 31 – Sep 25 (4 wk) | `offline/` backbone + conditioning interface; data collection → both views on all 3 envs; %BC on all 7 configs; LIAM + AD on LBF; vmapped cross-play | **%BC runs on all seven configs from one config file.** This is the real abstraction test. Both data views round-trip on all 3 envs |
| **1** | Sep 28 – Oct 30 (5 wk) | MeLIBA, DPT, AMAGO, Hybrid-AD — written once, against all 3 envs | Six-baseline table on LBF + one Overcooked layout + Hanabi. **Choose the two primary claims (§10.1) here** |
| **2** | Nov 2 – Dec 4 (5 wk) | TAO module; OMIS imitator module. Search is a *stretch item* (§10.5) | Full suite, all envs |
| **3** | Dec 7 – Jan 8 (5 wk) | Tiered results matrix (§10.3), dataset-quality ablation, backbone-sensitivity ablation, compute reporting | **Code freeze Jan 8** |
| **Write** | Jan 11 – deadline | Results frozen ~Jan 15; writing and figures only after | — |

Phase 0b's gate is deliberately harsh. If %BC — the simplest possible baseline — cannot be
trained and evaluated across all seven configurations from a single config file by Sep 25, the
abstractions are wrong and every later phase inherits the defect. That is worth stopping for.

### 10.3 The results matrix is now the binding constraint

With environments fixed at seven configurations, the schedule risk moves decisively from
population training (cheap, §7.1) to **baseline training**: 13 baselines × 7 configs × dataset
variants × seeds is on the order of 400 training runs. That is the number that can miss the
deadline. Control it with a tiered matrix rather than by cutting environments:

- **Tier 1 — full density.** LBF 12×12, Overcooked **`counter_circuit`**, Hanabi. All baselines ×
  all four generators × all splits × all dataset variants × 3 seeds. These three carry the
  headline claims and span the structural range (gridworld / spatial coordination / turn-based
  hidden information).
- **Tier 2 — reduced density.** Overcooked `cramped_room`, `coord_ring`, `asymm_advantages`,
  `forced_coord`. All baselines, but `replay` + `mixed` only, primary split only.

**[rev 6] The Tier 1 Overcooked layout changed from `cramped_room` to `counter_circuit`, and this
matters more than it looks.** ZSC-Eval reports empirically that *"the commonly used layouts,
Forced Coord. and Asymm. Adv. **fail to differentiate algorithms' performance**"* — plain
self-play does well on both — and classifies layouts by resource-sharing, finding the **Full
Resource-sharing** group (Coord. Ring, Counter Circ., Blocked Corr.) discriminates far better.
`cramped_room` is the simplest layout in the set. Putting a non-discriminative layout in Tier 1
means the headline Overcooked column shows every method tied, which is the single most avoidable
way to waste an environment. `coord_ring` is the fallback if `counter_circuit` proves difficult —
note it is the one layout with no published BRDiv population to cross-check against (§7.5), so
watch it during tuning.

This honors the all-five-layouts commitment while keeping the run count near 200 rather than 400.

### 10.4 Cut lines, in the order things get dropped

Environments are no longer cuttable — they are day-1 infrastructure. The flexible dimensions are
now matrix density and the baseline roster. Pre-committing the order prevents a late panic from
cutting the wrong thing.

1. **Bespoke-conversion ablation** (§3.1) — costs a table row of defensibility, nothing structural.
2. **OMIS decision-time search** — decision point **Dec 4**. See §10.5; the cheap 90% of OMIS
   ships regardless.
3. **Tier 2 layouts demoted further** — report on the primary claim only, or drop to two
   baselines per cell. Decision point **Dec 18**.
4. **Dataset variants beyond `replay` + `mixed`** outside Tier 1.
5. **TAO and OMIS entirely**, falling back to the seven JAX-native baselines plus %BC. Decision
   point **Dec 4**. This is the last cut and it changes what the paper claims — the head-to-head
   contribution is materially weaker without the two methods that required reimplementation.

Cuts 1–4 leave the headline intact. Cut 5 needs a deliberate decision, not a default.

### 10.5 The OMIS restructure

Under §3.1's shared backbone, OMIS decomposes into two pieces with wildly different costs.
**OMIS-without-search is just another modeling module** — the opponent imitator produces the
conditioning vector, and the shared learner does the rest. That is a Phase 2 item comparable to
TAO in cost. **[rev 6] The search is much cheaper than rev 2–5 assumed.** Reading the paper
(Eq. 6–10) shows DTS is a **flat, fixed-depth rollout estimator, not a tree search**: for each
legal action, run `M` rollouts of length `L` sampling ego actions from `π_θ` and teammate actions
from the imitator `μ_φ`, bootstrap with `V_ω`, average, and `argmax`. No tree, no UCB, no backup,
no gradient updates. It is `vmap(vmap(scan))` over (legal actions × rollouts × steps) — close to
ideal for JAX, and *easier* in JAX than in the authors' PyTorch, which resorts to `copy.deepcopy`
of a live environment. It also **outperforms SP-MCTS**, an actual MCTS, in their comparison.

Do not skip the **mixing rule** (Eq. 10): fall back to `π_θ` when
`‖Q̂(s_t, π_search(s_t))‖ ≤ ε`. It is ablated as load-bearing — removing it causes "a notable
performance decrease" in LBF and Overcooked.

Reclassify search from **L / open-ended research** to **M / mechanical**.

Rev 2 treated these as one deliverable rated **L**. Splitting them means the expensive, risky
half can be dropped on Dec 4 without losing OMIS from the results table. The cost of dropping
it is that §6's test-time-requirements asymmetry becomes a discussion point supported by one
ablation rather than a measured axis — a real loss, but a contained one, and §9 axis 3 survives
as taxonomy either way.

**[rev 6] Hanabi search stays out, but for a different and more precise reason.** Rev 4–5 excluded
it because tree search over a turn-based masked game is "a different algorithm." With a flat
rollout estimator that objection mostly dissolves — legal-action masking is handled natively by
`μ_φ`. The real obstacle is that **Hanabi's true state is not observable to the searcher**: the
ego agent cannot see its own hand, so rollouts cannot be run against `P` from the agent's
information set without a belief model. That is a genuine research problem. Keep Hanabi search
out; scope search to LBF and Overcooked, and record the reason as *partial observability of
state*, not search complexity.

**[rev 6] MBOMIS is the better stretch item.** Adding a learned dynamics model is cheaper than
search, is reported in the OMIS paper itself, and does more for the paper — it converts §6's
test-time asymmetry from a caveat into a measured quantity. Prioritize MBOMIS over
true-simulator search if only one fits.

### 10.6 Phase detail

**Phase 0a — environments and populations.** All three environments wired from jax-aht's `envs/`
(`lbf`, `overcooked-v1` × 5 layouts, `hanabi`), plus `mini-hanabi` as a fast debug configuration.
`cluster/` scaffolding (submission, sweeps, checkpoint/resume, artifact sync) — this is
infrastructure, not convenience, since local development is CPU-only (§11). Population training
launched across all 28 (generator × env-config) cells with per-environment hyperparameter tuning
(§7.1), **Hanabi first** because it is the long pole. Dataset schema v0 implemented and validated,
including `avail_actions` and `acting_agent` (§4.2). Survey drafting starts (§10.1). Existing
`bayes-tom` LBF populations unblock data-collection work on day 1 before new populations finish.
*Exit: all seven env-configs instantiate and roll out; our LBF populations cross-check against
jax-aht's published ones (§7.5).*

**Phase 0b — the offline stack, validated across all three environments.** Shared offline
DT backbone (adapted from JAX-CORL `algos/dt.py`) + `module(history) -> z` conditioning
interface (§3.1), validated with %BC and the IQL ablation backbone. Data
collection producing both views on all three environments. **%BC on all seven configurations**,
plus LIAM (trajectory view) and AD (learning-history view) on LBF to exercise both data paths and
both source repos. Batched/vmapped cross-play harness — built now, not retrofitted.
*Exit: %BC trains and evaluates on all seven configs from one config file; both views round-trip
on all three environments; cross-play matrices compute in minutes, not hours.*

**Phase 0c — runs in parallel, blocks nothing.** Attempt to download TAO's OSF datasets and
OMIS's ONNX opponent pool and reproduce each paper's headline number in its original PyTorch.
Highest-risk item in the project (§11) with external-dependency latency — discovering a dead link
in month 1 is cheap, in month 5 it is not. Rev 1 placed this in Phase 2; that was too late.

**Phase 1 — remaining JAX baselines, written once against all three environments.** MeLIBA
(modeling module), DPT, AMAGO, Hybrid-AD (learning-history family). The
encoder-genericization work that rev 1–3 spread across Phases 1 and 3 collapses here: the
ICRL4AHT CNN encoders hardcoded to `(9,7,26)` must become env-generic once, covering LBF's
vectors, Overcooked's grids, and Hanabi's masked discrete observations together. Note LIAM has no
Overcooked ego config in jax-aht, so that config is ours to write.

**Phase 2 — reimplementations.** TAO contrastive encoder as a modeling module, OMIS's imitator as
a modeling module (§10.5), and **TAGET** — hierarchical teammate-aware return-to-go and sub-goal
prediction over the shared DT backbone. MBOMIS if it fits. JAX-native decision-time search is a
stretch item scoped to LBF and Overcooked.

**[rev 6] The fidelity strategy has changed, because numeric validation is impossible.** Rev 5
planned to check our TAGET against its published numbers on "the overlapping environments." There
are none — every method's environment *configuration* differs from ours (§1), and TAO's are
competitive games. Substitute gates, in descending strength:

1. **Reproduce ablation orderings.** TAGET: removing the TA-Goal decoder should be the largest
   single degradation, with data-mirroring second. OMIS: removing mixing should hurt in LBF and
   Overcooked; `D^epi` should matter more than `D^step`. TAO: removing PEL should hurt most on
   unseen teammates. These are qualitative, cheap, and hard to pass by accident.
2. **Reproduce hyperparameter sensitivity shapes.** TAGET's `goal_steps` should be non-monotonic
   with an environment-specific optimum (they report PP 6, LBF 2, Overcooked 3).
3. **Reproduce ordinal claims against shared baselines.** TAGET should beat plain DT and Prompt-DT;
   OMIS should beat OMIS-without-search and SP-MCTS-style alternatives.
4. **Optional, and [rev 7] cheaper than it looked:** implement one original environment purely to
   check numeric agreement. **TAGET's LBF configuration is inherited from LIAM** — 20×20, 2 agents,
   4 foods, 5×5 observation, 50 timesteps, reward normalized to 1 — so building that single
   environment would let us validate **two** methods against published numbers (TAGET *and* LIAM),
   not one. That roughly doubles the return on the only gate that produces hard numeric agreement.
   Promote from "only if ahead of schedule" to a genuine Phase 2 stretch item.

Record these as the Phase 2 exit criteria in place of numeric equality, and state the limitation
plainly in the paper.

**Phase 3 — scale and ablate.** The tiered matrix (§10.3) restricted to what the two claims
chosen at the Phase 1 gate require, dataset-quality ablation, backbone-sensitivity ablation,
compute reporting, artifact release (datasets, populations, tuning records, code). Survey prose
is already drafted from Phases 0–1; this phase revises it against what the results actually
showed. Code freeze Jan 8, results frozen ~Jan 15.

---

## 11. Risks

- **[rev 4] The results matrix, not the environments, is the top risk.** Rev 3 treated
  environment count as the thing to cut under time pressure. Rev 4 fixes environments at seven
  configurations and moves the flexibility to matrix density (§10.3) and the baseline roster
  (§10.4). This is the better trade — environments are cheap to stand up and expensive to
  retrofit, while matrix density is expensive to run and cheap to reduce — but it means the
  ~400-run cross product must be pruned deliberately at the Phase 1 gate rather than allowed to
  expand. The residual risk is **not** that the scope is wrong but that cuts get made late and
  reactively. The gate dates in §10.4 are decisions to be *made on those dates*, not deadlines to
  be discovered having passed.
- **[rev 4] Phase 0 is now load-bearing in a way it was not before.** Environment-first front-
  loads risk: eight weeks of infrastructure before the first interesting result. If the Sep 25
  gate slips badly, there is no laddered fallback to retreat to, because the ladder is what rev 4
  removed. Mitigation: the Sep 25 gate is %BC on all seven configs — deliberately the simplest
  possible check, so it fails fast and cheap if the abstractions are wrong. Secondary mitigation:
  LIAM and AD on LBF are also in Phase 0b, so a mid-phase signal exists before the gate.
- **[rev 4] Per-environment tuning is now on the critical path.** Promoting population training
  to a contribution (§7.1) means tuning four generators across three structurally different
  environments, and §7.3 documents several ways this goes wrong (L-BRDiv's unnormalized Lagrange
  gradient, the 1/n² self-play dilution). LBF timings do not extrapolate to Hanabi. Mitigation:
  start Hanabi populations first (§10.2), and cross-check every cell that has a published
  counterpart (§7.5) rather than discovering bad populations downstream.
- **[rev 3] ICML is a main track, not a benchmarks track.** An artifact alone will not clear the
  bar; the paper needs a finding (§10.1). The failure mode is running the full matrix and hoping
  a story emerges, which at this timeline leaves no budget to chase one. Mitigation: nominate
  candidate claims now, commit to two at the end of Phase 1, and prune the matrix to what those
  two require. Secondary mitigation: %BC is in the suite from Phase 0 precisely because "filtered
  behavior cloning beats published methods on regime X" is both a plausible outcome and a
  publishable one.
- ~~**ICRL4AHT has no `LICENSE` file.**~~ **[rev 4] Resolved — confirmed no licensing issue.**
  AD, DPT, AMAGO, and Hybrid-AD can be adapted directly, and the HDF5 + JSONL index format is
  ours to build on (§4.1). The rev 1–3 fallback (reimplementing AD/DPT/AMAGO from their papers)
  is retired, which removes a meaningful contingency from Phases 1–2. **Housekeeping, not risk:**
  record the granted terms in the repo and reproduce the attribution the authors ask for, since
  the artifact release still needs correct provenance for adapted code.
- **TAGET is a close competitor at our target venue.** ICML 2025, and the most recent directly
  comparable method. Two exposures remain after rev 6: **(a)** reviewers will ask what this
  benchmark adds over TAGET's evaluation — the answer is the shared protocol, the released
  datasets and populations, the graded shift conditions, and the head-to-head across twelve
  methods, and it must be stated early rather than left implicit; **(b)** no code was released,
  so our TAGET is a reimplementation with **no original to diff against** and no comparable
  environment to check numerically (§10.6). It is the highest-uncertainty row in the table and
  the paper should say so. **[rev 6] The third exposure in rev 5 — "if our TAGET underperforms
  its published numbers" — is void**: their LBF is 20×20 with a simultaneous-collect rule and
  their Overcooked is a custom `overcooked_ai` layout, so there were never comparable numbers to
  underperform. Mitigation is now ablation-ordering and sensitivity-shape gates (§10.6).
- **TAO/OMIS supplementary code has no stated license.** Do not vendor or redistribute it. Using
  it as a private reference while writing clean-room JAX implementations is the safer path and
  is what §3 already commits to — but say so explicitly in the paper. **[rev 5]** JAX-CORL is MIT
  and may be adapted freely with attribution; TAGET released no code at all, so its
  reimplementation is necessarily clean-room from the paper.
- **Reimplementation fidelity — [rev 6] harder than rev 1–5 assumed.** A wrong TAO/OMIS/TAGET
  makes the headline claim wrong in the most embarrassing possible way, and **no method's
  published numbers are reproducible in our environments** because no configuration overlaps
  (§1, §10.6). The rev 1–5 mitigation ("reproduce each method's reported result, then port") is
  only available for TAO and OMIS via their original code in Phase 0c; **TAGET has no code and no
  comparable environment**, so it is validated by ablation ordering alone. Treat TAGET as the
  highest-uncertainty row in the results table and say so in the paper.
- **[rev 6] The expected result is substantially negative, and the paper must be designed for
  that.** ICRL4AHT already reports AD, DPT, AMAGO-Offline, and Hybrid-AD failing to beat a random
  baseline, with flat adaptation curves, and rules out context length, model scale, recurrence,
  teammate-action conditioning, and the offline-RL objective as explanations. If the trajectory-view
  family also lands near random across three environments, our finding is "offline AHT does not
  work yet, and here is the instrument that shows it." That is publishable at ICML **only if the
  instrument is credible** — which puts the weight on floors (random *and* %BC), diagnostics
  (adaptation gain), adequate episode budgets, and graded shift, rather than on baseline count.
  Mitigation: treat the protocol as the contribution and the ranking as the finding, and do not
  let the roster grow at the expense of §8.
- **[rev 6] Dataset storage may constrain the release.** ICRL4AHT's single-environment
  learning-history dataset is ≈6.5 GB compressed. Our matrix extrapolates to hundreds of GB
  (§4.6). Decide in Phase 0a what is released versus regenerable; discovering this in January
  means either a truncated artifact or a hosting scramble.
- **[rev 2] The shared backbone is a defensible-but-attackable choice.** A reviewer can argue we
  did not evaluate the published methods. Mitigations, in order of strength: report %BC as a
  floor in every table; report backbone-sensitivity (DT vs. IQL); report at least one
  bespoke-conversion comparison (§3.1). State the choice and its rationale in the abstract, not
  buried in an appendix — it is a design contribution, and framing it defensively invites the
  attack it is trying to avoid. **[rev 5] This risk is materially lower than in rev 2–4.** TAO,
  OMIS, and TAGET are already return-conditioned DTs, so for three of the four trajectory-view
  methods the shared backbone is the faithful choice rather than a compromise; and LIAM's and
  MeLIBA's offline conversions are TAGET's published ones, not ours.
- **Hanabi is a harder port than "third environment" suggests.** Action masking and turn
  alternation touch every baseline's action head *and* the learning-history view's episode
  segmentation. The schema fields in §4.2 are the cheap part; the sequence-model handling is not.
  **[rev 4] Rev 2 used this to justify deferring Hanabi to Phase 3; rev 4 inverts the conclusion.**
  Precisely because Hanabi imposes the strictest requirements on the interfaces, it must be
  present while those interfaces are being designed. Deferring it means designing the action head
  and the history view against two simultaneous-move environments and then discovering in
  December what turn-based masked play requires. Hanabi moves to Phase 0a for the same reason it
  was previously postponed.
- **Scope.** Survey + datasets + 13 baselines + 7 env-configs + full matrix is more than one
  paper's worth of work on a short cycle. **[rev 4]** The relief valve is no longer environment
  count (fixed) but matrix density and baseline roster — see §10.3/§10.4 and the top risk above.
- **Cross-play and eval cost.** Naive per-pair sequential rollouts are slow (the 21×21 LBF pilot
  was pacing ~12 h at 10 episodes/pair, one subprocess per column re-paying JIT). Build the eval
  harness batched/vmapped from the start rather than retrofitting. This is now a Phase 0 exit
  criterion.
- **Overcooked-V2 vs v1.** ICRL4AHT uses Overcooked-V2, jax-aht uses v1, ZSC-Eval and OMIS use
  `overcooked_ai`. These are not interchangeable. **[rev 2] Resolved: Overcooked-v1 (JaxMARL)**,
  because it is the variant with full teammate-generation coverage in jax-aht (§7). Cost: the
  ICRL4AHT baselines' CNN encoders are hardcoded to Overcooked-V2's `(9,7,26)` and must be
  genericized — but that work was already required for LBF and Hanabi regardless.
- **[rev 2] Local development is CPU-only.** The dev machine is an M1 with 16 GB and no CUDA.
  Everything past a smoke test runs on the cluster, so `cluster/` scaffolding (submission,
  sweeps, checkpoint/resume, artifact sync) is Phase 0a infrastructure, not a late convenience.
  Design configs so a run is identical locally at toy scale and on the cluster at full scale.

---

## 12. Decisions

Rev 1 listed these as open. Status as of rev 4:

1. **Environments for v1** — ✅ **Resolved and committed: LBF 12×12 + Overcooked-v1 (all five
   layouts) + Hanabi.** Seven environment configurations, supported from day 1 (§10.2).
   `mini-Hanabi` is retained as a debug configuration, not a results environment. Environments
   are explicitly **not** a cut line under time pressure (§10.4).
2. **Overcooked variant** — ✅ **Resolved: Overcooked-v1 (JaxMARL)**, per §11.
3. **Overcooked layouts** — ✅ **Resolved: all five**, with tiered matrix density (§10.3) so this
   does not multiply every table by five. **[rev 6] `counter_circuit` is Tier 1**, not
   `cramped_room` — ZSC-Eval shows the simpler layouts fail to differentiate algorithms (§10.3).
4. **Dataset variants for v1** — ✅ **Resolved:** all five regimes for Tier 1 environments;
   `replay` + `mixed` for Tier 2 (§4.3, §10.3).
5. **Hanabi** — ✅ **Resolved: in, full variant, Phase 0a.** Rev 1 excluded it on a false premise;
   rev 3 deferred it to Phase 3; rev 4 promotes it to day 1 because it imposes the strictest
   interface requirements (§11).
6. **Teammate populations** — ✅ **Resolved: train all 28 cells in-house** with per-environment
   tuning, released as contribution 3 (§7.1). Published checkpoints become validation references
   (§7.5), not dependencies.
7. **Offline conversion for trajectory-view methods** — ✅ **Resolved: shared backbone, and the
   LIAM/MeLIBA conversions follow TAGET's published variants** (§3.1). Rev 2 decided the shared
   backbone; rev 5 established that the conversions themselves are prior art, not ours to invent.
8. **Backbone architecture** — ✅ **Resolved: return-conditioned Decision Transformer**, adapted
   from JAX-CORL `algos/dt.py`, with IQL as the sensitivity ablation (§3.1). New in rev 5, and
   grounded in the verified fact that TAO, OMIS, and TAGET are already DTs while AD and DPT are
   deliberately not return-conditioned.
9. **Baseline roster** — ✅ **[rev 7] Resolved: thirteen**, in four groups.
   - *Floors* (2): **Random**, %BC
   - *Reference* (1): Prompt-DT
   - *Ceiling* (1): **Oracle / FIAM-style** privileged teammate model
   - *Learning-history family* (4): AD, DPT, AMAGO-Offline, Hybrid-AD
   - *Trajectory-view family* (5): LIAM, MeLIBA, TAO, OMIS, TAGET

   OMIS is reported as three tracks (no-search / **MBOMIS** learned-dynamics / true-simulator)
   rather than three roster entries. **BayesToM is excluded** as a different problem setting (§1).
10. **Compute** — ✅ **Resolved: university/lab GPU cluster.** Drives the `cluster/` requirement
    in §5. Population training is cheap (§7.1); baseline training is the constraint (§10.3).
11. **ICRL4AHT licensing** — ✅ **Resolved: no issue.** AD/DPT/AMAGO/Hybrid-AD and the HDF5 format
    are usable directly (§11).
12. **Hosting** — ⬜ **Open.** Personal account, lab org, or new org; public from day one or only
    at submission. Name is settled: `oaht-bench`. Current remote is
    `github.com/conor-wallace/OAHT-Bench`. **[rev 4]** The licensing constraint that made this
    urgent is gone, so this is now a pure preference call — but it should be settled before the
    first population checkpoints land, since those are large artifacts and moving hosts later is
    annoying.
13. **Venue and deadline** — ✅ **Resolved: ICML 2027, ~late January 2027.** Verify the exact date
    as soon as the CFP posts and re-anchor §10.2 to it. Consequences: §10.1 added because ICML
    demands a finding rather than an artifact, OMIS split so its risky half is droppable (§10.5),
    and survey drafting pulled forward into Phases 0–1.
14. **Which two headline claims** (§10.1) — ⬜ **Open, decide at the Phase 1 gate (Oct 30).**
    Deliberately left open: it should be informed by the first multi-environment table, not
    guessed now.

---

## 13. Immediate next actions

Ordered for Phase 0a. Items 1–3 are the critical path; 4–6 run alongside.

0. ~~Read the TAGET paper~~ ✅ **Done (rev 6).** All primary sources for the five decision-critical
   methods are extracted into `docs/baseline_specs.md`. Remaining reading, in priority order:
   the four teammate-generation papers (FCP, CoMeDi, BRDiv, L-BRDiv) to support §7.2's tuning
   contribution; LIAM and MeLIBA originals (largely superseded by TAO Appendix F for our
   purposes); D4RL, DT, IQL, AMAGO (standard; implementations already in hand via JAX-CORL).
1. **Stand up the repo skeleton per §5**, forking jax-aht's `envs/`, `agents/`, and
   `teammate_generation/` as the spine. Wire all seven environment configurations and confirm
   each instantiates and rolls out. This is now action #1 — the licensing question that used to
   head this list is resolved.
2. **Build `cluster/` scaffolding early**, before it is needed. Local development is CPU-only, and
   28 population-training jobs plus ~200 baseline runs are not something to start orchestrating
   by hand in October.
3. **Launch population training, Hanabi first** (§7.1). Hanabi is the long pole and the least
   characterized; starting it in week 1 converts an unknown into a measurement. Cross-check each
   completed cell against jax-aht's published populations where one exists (§7.5).
4. **Audit the two data consumers before writing `schema.py`.** Read ICRL4AHT's `HistoryStore` /
   `runners/history_adapter.py` and jax-aht's `liam_utils.Transition` end-to-end, and write down
   exactly what each view must contain — **for all three environments**, since Hanabi's masking
   and turn structure are what the schema has to survive. Rev 1's §13 put schema first; writing
   the contract against real consumers is what keeps it from needing a v1 within a week.
5. **Write `dataset/schema.py`** — the §4 spec, versioned, with a validator and a synthetic fixture
   per environment. Include `avail_actions` and `acting_agent` from v0.
6. **Specify and implement `offline/` (§3.1)** — the backbone and the `module(history) -> z`
   conditioning interface. Critical path for every trajectory-view baseline.
7. **Collect an LBF dataset immediately** from the existing `bayes-tom/checkpoints/lbf/lbf_12x12/`
   populations, without waiting for new population training; verify both views round-trip.
8. **%BC on all seven configs**, then LIAM and AD on LBF, plus the vmapped cross-play harness.
   Phase 0b exit (Sep 25).
9. **In parallel (Phase 0c):** attempt TAO/OMIS original-result reproduction; report dead links
   early.
10. **Start survey drafting now** (§10.1) — it does not depend on any of the above.

---

## 14. Change log

### Rev 8 (BRDiv retraction; population-scaling status)

| # | Change | Evidence |
|---|---|---|
| 1 | **Retracted rev 7's BRDiv "correction."** `XP_LOSS_WEIGHTS` really is population-size-invariant: `BRDiv.py:389–391` builds `sp_weight`/`xp_weight` with `n` factors that exactly cancel the sampling probabilities, giving expected per-sample contributions independent of `n`. Rev 7 conflated the paper's Eq. 6 metric with the implementation's policy-gradient weights. §7.3 reinstated with the code-level derivation. | `BRDiv.py:389–391`; verified numerically at n = 3, 5, 10, 20 |
| 2 | **New finding recorded**: the n=5 BRDiv diversity collapse (near-flat cross-play matrix) was **absolute sample-count dilution**, fixed by scaling `NUM_ENVS` 64→128 and `TOTAL_TIMESTEPS` 4.5e7→7e7 (≈3×) — *not* by touching `XP_LOSS_WEIGHTS`. The symptom points at the diversity weight; the cause is the 1/n² self-play draw probability. | Exploratory LBF 12×12 runs, 2026-07-30 |
| 3 | **Status qualifier added to §7.3.** These values came from exploratory runs, not a tuning sweep. Directions are established; specific numbers are starting points for §7.2's sweep, not the tuned configuration. Rev 7 and earlier read as more settled than the evidence supports. | — |

### Rev 7 (remaining ten papers — literature pass complete)

Rev 6 was written after the five decision-critical papers. Rev 7 adds the four teammate-generation
papers, LIAM, MeLIBA, AMAGO, DT, IQL, and D4RL. **All 15 papers in `papers/` have now been read.**

| # | Change | Evidence |
|---|---|---|
| 1 | ~~**Correction to §7.3's BRDiv claim.**~~ **RETRACTED in rev 8** — this "correction" was itself wrong. It conflated the paper's diversity metric with the implementation's per-sample policy-gradient weights. The original rev 1–6 claim was correct. See §7.3. | — |
| 2 | **Correction to §9 axis 9.** BRDiv and L-BRDiv are **response-oriented** (Minimum Coverage Set), not population-oriented. Correct split: FCP/CoMeDi population-oriented; BRDiv/L-BRDiv response-oriented; ZSC-Eval's BR-Div the same idea applied to *selection*. Our four generators span the distinction, making it testable rather than assertable. | L-BRDiv §4–5 |
| 3 | **New §7.4 — the handshake confound.** Cross-play minimization can produce identity-revealing handshakes followed by deliberate sabotage, so low off-diagonal cross-play entries may measure signalling rather than convention difference — contaminating §8's primary population diagnostic. Mitigations: verify mixed-play is enabled and record `β`; add a **handshake probe**. | CoMeDi §3.2–3.3 |
| 4 | **Oracle / FIAM ceiling row added** (roster now thirteen). With an expected-negative result, the ceiling matters as much as the floor: "all methods near random" and "all methods near random *and so is the oracle*" are different papers. | LIAM §4.2 |
| 5 | **Conditioning interface gains a fourth mode**: pass distribution parameters `(μ, σ)`, not a point embedding. | MeLIBA §4.3–4.4 |
| 6 | **§3.1 gains its strongest support**: LIAM and MeLIBA **both** already detach the encoder from the policy gradient. The module factorization is faithful to the originals, not imposed by us. | LIAM §3.3; MeLIBA §4.4 |
| 7 | **Attention entropy collapse** flagged as a live risk for long learning-history contexts; apply AMAGO's stabilizers (Normformer, σReparam, Leaky ReLU) to the shared backbone generally. Hindsight relabeling explicitly scoped out. | AMAGO §4 |
| 8 | **§4.3 matched to D4RL's exact definitions.** `medium-replay` **stops at medium performance**; ours runs to convergence, so rename `replay-full` and document the deviation. `medium-expert` is **50/50** — specify the ratio. | D4RL §5 |
| 9 | **D4RL's taxonomy cited in §1**: non-Markovian behavior policies plus partial observability are named as a hard case, and offline AHT is inherently both. | D4RL §4 |
| 10 | **FCP's mid-checkpoint defined precisely** as **50% of final reward** (ZSC-Eval converges on the same criterion); **architectural diversity dropped** from the tuning budget per FCP's own `FCP₊A` ablation. | FCP §2.4, §4.2 |
| 11 | **CoMeDi refined**: greedy/sequential construction; cross-play minimized only against the *most compatible* existing convention; mixed-play `β` is load-bearing, not a nuisance parameter. | CoMeDi §3.3–3.4 |
| 12 | **§8's identity probe recorded as established practice** — MeLIBA already uses it, LIAM's CBAM makes it the method. Use MeLIBA as the probe's sanity check; run the probe on the oracle row too. | MeLIBA Fig. 3c; LIAM §4.2 |
| 13 | **LBF populations are known-incomplete**: L-BRDiv recovers only 4–5 of 6 MCS members in its own LBF. "Held-out" does not imply the population spanned the space. | L-BRDiv §6.4–6.5 |
| 14 | **§10.6 gate 4 promoted to a real stretch item**: TAGET's LBF configuration is inherited from LIAM, so building that one environment validates **two** methods numerically, not one. | LIAM §4.1 vs TAGET App. A |

### Rev 6 (grounded in the primary literature)

First revision based on reading the papers rather than code and abstracts. Extraction in
`docs/baseline_specs.md`. Read in depth: TAGET, OMIS, TAO (incl. appendices), ZSC-Eval, ICRL4AHT.

| # | Change | Evidence |
|---|---|---|
| 1 | **LIAM-off and MeLIBA-off come from TAO Appendix F**, not TAGET. TAGET converts LIAM only, in one sentence; TAO specifies both in full. | TAGET §5.2 baseline list contains no MeLIBA; TAO App. F gives layer-level specs for both |
| 2 | **TAO establishes the shared-backbone methodology as published practice** — *"we mandate all approaches to use the same neural architecture as ours."* Now the primary defense in §11. | TAO §4.1 |
| 3 | **Conditioning interface must support three modes**: cross-attention (TAO), prompt-token replacement (TAGET), auxiliary-head-only (LIAM/MeLIBA). Rev 5's single `concat(obs, z)` design could not express TAGET. | TAO App. F; TAGET Eq. 10 |
| 4 | **OMIS search downgraded L → M.** It is a flat vmappable rollout estimator, not a tree search, and it beats SP-MCTS. Mixing rule (Eq. 10) is load-bearing. | OMIS §4.2, Eq. 6–10; Q5 ablation |
| 5 | **Hanabi search excluded for a corrected reason**: unobservable own-hand state, not search complexity. | Follows from #4 |
| 6 | **MBOMIS added**; makes §6's test-time asymmetry measurable instead of merely discussed. | OMIS Q3, Fig. 5 |
| 7 | **Fidelity strategy rewritten.** No environment configuration overlaps any source paper, so numeric validation is impossible. Replaced with ablation-ordering, sensitivity-shape, and ordinal gates. | TAGET App. A (20×20 LBF, custom Overcooked); TAO App. E (competitive envs); OMIS §5.1 |
| 8 | **Random baseline added as a hard floor**, alongside %BC. | ICRL4AHT Tables 3–4: AD/DPT below random on a full track |
| 9 | **Adaptation gain metric added** (last-20 minus first-20 episode return). | ICRL4AHT §5.1 — the diagnostic that exposed flat adaptation |
| 10 | **Graded distribution shift replaces the binary split**; two evaluation tracks; non-stationary teammates as a secondary condition. | Three independent precedents: OMIS seen:unseen ratios, ICRL4AHT cooperability ordering, TAO seen/unseen/mix |
| 11 | **BR-Prox given its exact definition; aggregation is IQM (`rliable`) with 95% CI and IQR.** | ZSC-Eval §4.3 |
| 12 | **Tier 1 Overcooked layout changed `cramped_room` → `counter_circuit`.** | ZSC-Eval §5.2: Forced Coord. and Asymm. Adv. "fail to differentiate algorithms"; Full Resource-sharing layouts discriminate |
| 13 | **Held-out teammates selected by BR-Div**, not at random; P-Div ≠ BR-Div added as survey axis 9. | ZSC-Eval §4.2, Fig. 2b |
| 14 | **`teammate_observations`, `teammate_rewards`, `teammate_id` promoted to required schema fields.** | TAGET team-context encoder + TA-Goal target; TAO's fused `(a⁻¹, r⁻¹, o⁻¹)` and InfoNCE identity loss |
| 15 | **`expert` (per-teammate BR) made mandatory** — one artifact serving OMIS training, TAO targets, and BR-Prox normalization. OMIS noted as structurally excluded from `random`/`medium`. | OMIS §4.1; TAO §3.2; ZSC-Eval §4.3 |
| 16 | **Trajectory filtering added as a required collection stage** (§4.4); filtered-vs-unfiltered proposed as a novel ablation. | ICRL4AHT §4.2: 1.196B → 149.5M transitions, 8× |
| 17 | **Trajectory mirroring added to the collection layer** (§4.5), per-environment opt-in. | TAGET Alg. 1 and its ablation |
| 18 | **Storage budget added as a Phase 0a deliverable** (§4.6). | ICRL4AHT Table 2: ≈6.5 GB compressed for one environment |
| 19 | **Prompt-DT added to the roster**; specified in full and used by both TAO and TAGET. | TAO App. F |
| 20 | **Expected-negative-result risk added** (§11); §10.1 claim 1 marked partly pre-empted. | ICRL4AHT §5, §6 |
| 21 | **Survey axes 8–10 added**: population construction, P-Div vs BR-Div, conditioning mechanism. | Cross-paper synthesis |
| 22 | **ICRL4AHT's stated limitations adopted as the paper's positioning statement.** | ICRL4AHT §6 |

### Rev 5 (DT backbone; TAGET; BayesToM removed)

| # | Change | Rationale / evidence |
|---|---|---|
| 1 | **BayesToM removed from the baseline suite** (§1, §2, §5, §6, §9). bayes-tom still supplies LBF populations and the identity-recoverability probe. | Different problem setting; including it would misrepresent it or distort the benchmark's scope. Also removes the "our method lands in the setting we define" framing, which was never load-bearing |
| 2 | **TAGET added as a baseline and as prior art** (§1, §2, §6, §10.6). | "Ad Hoc Teamwork via Offline Goal-Based Decision Transformers", ICML 2025 (Zhang, Chan, Ye, Cai, Zhao) — verified via ICML proceedings. Evaluates on Predator-Prey, LBF, Overcooked. No code released |
| 3 | **§3.1 rewritten: offline LIAM/MeLIBA are TAGET's published variants, not ours.** | Removes the "definitions we author" burden that rev 2–4 built a defense around. Fidelity-to-TAGET replaces justify-our-own-conversion as the standard to meet |
| 4 | **Backbone changed from IQL to return-conditioned Decision Transformer** (§3.1, §5). | Verified in code: TAO (`offline_stage_2/net.py:54,64` `embed_return`) and OMIS (`testing/search.py:182,201`) are already return-conditioned DTs; TAGET's low-level module is a DT. So DT is *faithful* for three of four trajectory-view methods, not a compromise |
| 5 | **Correction to the premise:** AD and DPT are **not** Decision Transformers. | `grep -rn "rtg\|return_to_go"` over `ad/`, `dpt/`, `amago_offline/`, `hybrid_ad/` returns nothing; AD's tokens are `(prev_action, prev_reward, obs)` with cross-entropy on actions (`ad/model.py:228–242`). They are causal transformers *without* return conditioning — which is the point of in-context RL, and it lands exactly on §3.1's existing exempt/non-exempt boundary |
| 6 | **IQL retained as the backbone-sensitivity ablation, better motivated.** | DT vs. IQL is precisely return-conditioned-BC vs. value learning, so *dataset variant × backbone* becomes a substantive result feeding §10.1 claim 4 rather than a robustness footnote |
| 7 | **JAX-CORL adopted for the backbone implementations** (§2, §3.1, §5). | Local at `~/Documents/Personal/Projects/JAX-CORL/`, MIT (2024), single-file JAX `algos/{dt,iql,awac,cql,td3bc,xql}.py`. Adapt rather than reimplement |
| 8 | **New risk: TAGET is a close competitor at our target venue** (§11). | ICML 2025, overlapping on LBF and Overcooked. Three exposures: the what-does-this-add question, reimplementation without a reference, and credibility damage if our TAGET underperforms its published numbers |
| 9 | **New action #0: read the TAGET PDF.** | The LIAM/MeLIBA conversion details are the plan's only unverified assumption — OpenReview blocks automated fetching, so a human has to do it |

### Rev 4 (environment-first; licensing resolved)

| # | Change | Rationale / evidence |
|---|---|---|
| 1 | **All three environments supported from day 1.** Phase 3 dissolves into Phase 0a; §10.2 rebuilt. | The expensive part of multi-env support is not wrappers or populations but *interfaces revisited per env*. Doing it once up front against the hardest env beats doing it once per baseline in December |
| 2 | **Hanabi promoted from conditional (rev 3) to Phase 0a**, full variant. mini-Hanabi retained as a debug config only. | Rev 3 deferred Hanabi *because* action masking and turn alternation stress every interface. Rev 4 inverts that: those are exactly the constraints the interfaces must be designed against, so Hanabi has to be present during design |
| 3 | **Populations trained in-house across all 28 cells** with per-env tuning; promoted to contribution 3 (§1, §7.1–7.2). | Measured cost is low (LBF, CPU: FCP ~15 min, CoMeDi ~20 min, BRDiv/L-BRDiv ~4 h each). No source has the full matrix — `jaxaht/eval-teammates` has no FCP outside LBF and no generator populations at all for full Hanabi. Controlling populations removes a confound |
| 4 | Published checkpoints repositioned as **validation references**, not dependencies (§7.5). | Cross-checking our cross-play matrices against jax-aht's catches bad tuning in week 3 instead of week 20. `best_heldout_returns.zip` also supplies §8's normalization bounds free |
| 5 | New §7.6: **OOD teammates already exist** for all three envs. | `agents/{lbf,overcooked,hanabi}/` ship scripted/heuristic agents; Hanabi's set (iggi, piers, van-den-bergh, smartbot, BC-LSTM, OBL-R2D2) is the richest OOD condition in the suite. Rev 1–3 assumed we would write these |
| 6 | All five Overcooked layouts committed, with a **tiered matrix** (§10.3) — Tier 1 full density, Tier 2 reduced. | Honors full-layout coverage while holding the run count near 200 rather than 400 |
| 7 | **Top risk moved from environment count to matrix density** (§10.4 cut order rewritten). | Environments are cheap to stand up and expensive to retrofit; matrix density is expensive to run and cheap to reduce. The relief valve belongs on the reducible dimension |
| 8 | ICRL4AHT licensing resolved; fallback reimplementation of AD/DPT/AMAGO retired (§11, §12.9). | Confirmed no issue. Removes a contingency from Phases 1–2; leaves provenance/attribution as housekeeping |
| 9 | OMIS decision-time search scoped to LBF + Overcooked only, never Hanabi (§10.5). | Search over a turn-based hidden-information game with legal-action masking is a different algorithm, not a port — not something to open six weeks from a deadline |
| 10 | New risk: **Phase 0 is now load-bearing with no laddered fallback** (§11). | Environment-first front-loads eight weeks of infrastructure. Mitigated by making the Sep 25 gate the simplest possible check (%BC on all seven configs) so it fails fast |

### Rev 3 (venue: ICML 2027)

| # | Change | Rationale |
|---|---|---|
| 1 | Venue fixed to ICML 2027, ~late Jan 2027 (§12.8). §10 rebuilt as a dated schedule with gates. | ~25 weeks is enough for a reduced scope and not enough for the rev 2 scope; scheduling explicitly is what makes the difference visible early |
| 2 | New §10.1: the paper needs a **finding**, not just an artifact. Four candidate headline claims nominated; two to be committed at the Oct 9 gate. | ICML has no datasets-and-benchmarks track. Rev 2's venue recommendation (NeurIPS D&B) would have accepted an artifact-first paper; ICML will not |
| 3 | New §10.3: cut order pre-committed — bespoke ablation → Hanabi (Dec 4) → OMIS search (Nov 13) → extra dataset variants → Overcooked (Dec 11). | Late reactive cuts remove the wrong things. Cuts 1–4 leave the headline intact; only cut 5 changes what the paper claims |
| 4 | New §10.4: OMIS split into imitator-as-modeling-module (cheap, ships) and decision-time search (risky, droppable). | Under §3.1 these have very different costs, which rev 2 obscured by rating OMIS as one **L** item. Splitting keeps OMIS in the results table even if search is cut |
| 5 | Hanabi demoted from committed to conditional (§12.1). | Affordable in engineering terms (§7), not in calendar terms. The action-masking/turn-alternation work (§11) is real and lands six weeks from the deadline |
| 6 | Survey drafting moved from Phase 4 into Phases 0–1. | It is the only deliverable that does not depend on results, so it is the only one that parallelizes against engineering |
| 7 | New §12.9: which two headline claims, deliberately left open until the first full LBF table exists. | Guessing now risks collecting for the wrong claim; the Phase 1 gate is the earliest point with evidence |

### Rev 2 (source-repo corrections and the §3.1 decision)

What changed from rev 1 and the evidence for each change.

| # | Change | Evidence |
|---|---|---|
| 1 | jax-aht covers **all 4 teammate-gen algorithms × all 3 target envs**, not "LBF/Overcooked-v1 only". §7's "unifying teammate generation is real work" is retracted; it is a fork-and-configure task. | `teammate_generation/configs/algorithm/{fcp,comedi,brdiv,lbrdiv}/` each contain `hanabi.yaml`, `mini-hanabi.yaml`, `lbf/`, `overcooked-v1/`; `envs/__init__.py:113` dispatches `hanabi` |
| 2 | Hanabi moved from open question (rev 1 §12.4: "neither jax-aht teammate-gen nor ICRL4AHT covers it") to resolved-in-scope. | `envs/hanabi/hanabi_wrapper.py`, `agents/hanabi/`, teammate-gen + LIAM ego configs all present |
| 3 | **LIAM and MeLIBA are online PPO, not offline.** Forces the new §3.1 decision; raises their cost from S–M to M; changes what Phase 0 is actually de-risking. | `liam_ego.py:191` and `meliba_ego.py:198` call `jax.vmap(env.step)` in a rollout loop; `meliba_ego.py:223` `_calculate_gae` |
| 4 | New §3.1: shared offline ego backbone with a `module(history) -> z` interface. New §5 `offline/` package, new `%BC` baseline, new §9 taxonomy axis 7, revised §6 costs. | Follows from #3 plus the comparability requirement in §1 |
| 5 | Schema carries `avail_actions` and `acting_agent` from v0. | `hanabi_wrapper.py:43` `get_legal_moves`; Hanabi is turn-based while LBF/Overcooked are simultaneous |
| 6 | Build on ICRL4AHT's HDF5 + JSONL index rather than designing a format. | `benchmarks/baselines/ad/train.py:13,748` consumes `histories.h5` + `histories_index.jsonl` via `HistoryStore`, with no env dependency — confirms AD is genuinely offline |
| 7 | Overcooked variant resolved to v1 (JaxMARL). | Only variant with full teammate-gen coverage (#1) |
| 8 | TAO/OMIS original-result reproduction moved from Phase 2 to Phase 0b (parallel). | External-dependency latency: TAO datasets on OSF, OMIS pool via ONNX download; both may be stale |
| 9 | `cluster/` scaffolding is Phase 0 infrastructure. | Dev machine is M1/16 GB/no CUDA; all real runs are remote |
| 10 | Timeline promoted to the top risk, with a venue recommendation. | Resolved scope (3 envs, 9 baselines, survey) vs. ~2–3 months to the next ICLR/AAMAS deadline |
