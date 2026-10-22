# Baseline specifications, extracted from the papers

Working document. One section per method: architecture, data requirements, hyperparameters,
reported numbers, and what it implies for our implementation. Populated by reading
`papers/*.pdf` directly. Feeds §3.1, §4, §6, §9, and the Phase 2 fidelity gates of
`offline_aht_benchmark_project.md`.

**Status: all 15 papers in `papers/` read.** TAGET, OMIS, TAO (incl. appendices), ZSC-Eval,
ICRL4AHT, FCP, CoMeDi, BRDiv, L-BRDiv, LIAM, MeLIBA, AMAGO, DT, IQL, D4RL.

Items that require action beyond rev 6 of the project plan are collected in "Outstanding
corrections" at the end.

---

## TAGET — Ad Hoc Teamwork via Offline Goal-Based Decision Transformers

Zhang, Chan, Ye, Cai, Zhao. **ICML 2025**, PMLR 267. `papers/taget.pdf`. No code released.
TAGET = **T**eammate-**A**ware **G**oal driven hi**E**rarchical Decision **T**ransformers.

### Three findings that contradict our planning assumptions

**1. TAGET does NOT convert MeLIBA to offline.** The baseline list (§5.2, p6–7) is: DT,
Prompt-DT, **ODITS-off**, **LIAM-off**, CQL, MADT. MeLIBA appears nowhere in the paper. The
offline-converted prior methods are LIAM and **ODITS** (Gu et al. 2021), not LIAM and MeLIBA.
→ **`offline MeLIBA` remains a definition we author.** §3.1's claim that both conversions are
published prior art holds for LIAM only.

**2. The LIAM-off specification is one sentence.** Verbatim (p7): *"To adapt LIAM into LIAM-off,
we modify it to learn how to reconstruct global information from local observations directly
from an offline dataset instead of a replay buffer, enabling the extraction of strong
representational latent variables that guide the controlled agent's actions without requiring
online interactions."* That is the entire published specification. It states the encoder is
trained on offline data instead of a replay buffer; it does **not** specify the ego policy
learner. → The citable precedent is real but thin. It licenses "train LIAM's reconstruction
encoder on offline data," not a full offline algorithm. Our backbone choice remains ours to
justify.

**3. There is no usable environment overlap — direct numerical comparison is impossible.**
TAGET's environments are configured completely differently from ours (Appendix A, p12):

| | TAGET | Ours |
|---|---|---|
| LBF | **20×20** grid, 2 agents, 4 food, **5×5** local obs, 50 steps, reward normalized to 1, **both agents must be adjacent and collect simultaneously** | jax-aht/Jumanji **12×12**, fov 12, 6 food, `different_levels`, standard level-based rule |
| Overcooked | custom **"Surrounding"** layout, `overcooked_ai` lineage (Carroll et al. 2019), 400 steps, asymmetric roles (only green operates cookware) | JaxMARL **Overcooked-v1**, 5 standard layouts |
| Predator-Prey | 10×10, 2 predators, 4 prey, 200 steps | not in our suite |

→ **Retract the rev 5 plan to validate our TAGET against its published numbers.** No cell is
comparable. Fidelity must be established another way (below).

### Architecture

Three modules. Notation: `i` = ego, `-i` = teammates.

**Offline data pre-processing — trajectory mirroring** (Alg. 1, p13). For each trajectory and
each agent `i`, re-order so agent `i` is ego: `τ_{j,i} = f_mirror(τ_j, i) = {R̂_t, o^i_t, o^-i_t,
a^i_t, a^-i_t}`. Yields `N` ego-perspective trajectories per episode. Ablated ("w/o Data
Preprocessing") with consistent degradation across all three environments.

**High-level — teammate-aware goal prediction.**
- *Team Context Encoder* `f_φ(o^i_t, o^-i_t) → (μ_h, σ_h)`, `h^i ~ N(μ_h, σ_h)`. **Requires
  teammate observations. Training only.**
- *Proxy Encoder* `f_θ(o^i_t) → (μ_z, σ_z)`, `z^i ~ N(μ_z, σ_z)`. Local obs only. **This is the
  deployment path** — Alg. 3 substitutes `z` for `h` at test time.
- *Latent regularization* (Eq. 6): hybrid KL — reverse KL from team context to `N(0, I)` (β-VAE
  style, compact latent) **plus** forward KL aligning the proxy encoder to the stop-gradient'd
  team context. Two directions deliberately, for different reasons.
- *TA-RTG predictor* `R_φ(R_t | h^i_t)`, MSE against ground-truth return-to-go (Eq. 7).
- *TA-Goal decoder* `G_ξ(G_t | h^i_t, R_t)`, binary cross-entropy (Eq. 8). Target
  `G_t = Concat({o^i_{t+k}}^n_{i=1})` — **the discretized/one-hot observations of all agents `k`
  steps in the future**, i.e. a predicted global state.
- `L_High-level = λ_Reg L_Reg + λ_RTG L_RTG + λ_TA-Goal L_TA-Goal` (Eq. 9).

**Low-level — goal-conditioned action generation.** A causal DT over
`τ*_{j,i} = {G_t', o^i_t', a^i_t'}` for the last `K` steps (Eq. 10), cross-entropy on actions
(Eq. 11). **TA-Goal replaces return-to-go as the prompt token.** This is the paper's core
architectural claim: a scalar return signal is too weak to capture teammate intent, so condition
on a predicted future global state instead.

### Hyperparameters (Appendix E, p14; Fig. 7, p15)

DT backbone: embedding dim **64**, context length **K = 30**, **2** transformer layers,
**1** attention head, ReLU, dropout **0.3**. AdamW, lr **0.01**, batch **2048**, weight decay
**1e-4**. Training samples sub-sequences of length **3K** (three tokens per step: G, o, a).

Loss weights, App. E: `α = 0.0001, β = 100, γ = 100, σ = 0.001`. ⚠️ These names do not map onto
Eq. 9's `λ_Reg / λ_RTG / λ_TA-Goal`, and `β` is separately used in Eq. 6 for the KL balance. The
correspondence is genuinely ambiguous in the paper — flag as a reimplementation risk and sweep.

Heads (Fig. 7), all LeakyReLU:
- Proxy Encoder: FC512 → BN → FC512 → BN → FC(2·|z|)
- Team Context Encoder: FC64 → LayerNorm → **multi-head cross-attention** (hidden 64, 4 heads) → LayerNorm → FC(2·|z|)
- TA-RTG Predictor: FC512 → BN → FC512 → BN → FC1
- TA-Goal Predictor: FC512 → BN → FC256 → BN → FC|G|

**`goal_steps` (k) is a per-environment tuned hyperparameter** (App. C, p13) — PP: 6, LBF: 2,
Overcooked: 3. Sensitivity is large and non-monotonic (Overcooked: 0.42 → 0.90 → **0.98** → 0.30
for k = 1, 2, 3, 6). Any environment we add needs this swept; it is not transferable.

### Teammate generation

**Soft-Value Diversity (SVD) from CSP** (Ding et al. 2023) — *not* FCP/CoMeDi/BRDiv/L-BRDiv.
Four populations per environment; one held out as the test teammate set, three used to collect
the offline data. Training populations hold **3 checkpoints** each, test populations **8**.
Evaluation averages **50 episodes**, 95% CI via normal approximation.

### Reported results (Table 1, p13)

| Method | PP-4 | LBF-4 | Overcooked-4 | PP-8 | LBF-8 | Overcooked-8 |
|---|---|---|---|---|---|---|
| DT | 54.6 ± 0.6 | 0.098 ± 0.008 | 0.16 ± 0.19 | 55.4 ± 1.6 | 0.055 ± 0.009 | 0.08 ± 0.03 |
| Prompt-DT | 56.3 ± 0.7 | 0.102 ± 0.010 | 0.40 ± 0.19 | 54.4 ± 1.2 | 0.045 ± 0.008 | 0.64 ± 0.15 |
| ODITS-off | 55.6 ± 1.6 | 0.065 ± 0.036 | 0.24 ± 0.28 | 54.7 ± 1.4 | 0.000 ± 0.000 | 0.06 ± 0.03 |
| LIAM-off | 58.2 ± 1.8 | 0.005 ± 0.009 | 0.42 ± 0.26 | 55.3 ± 1.8 | 0.040 ± 0.010 | 0.24 ± 0.09 |
| CQL | 60.6 ± 1.0 | 0.065 ± 0.036 | 0.46 ± 0.08 | 59.0 ± 0.8 | 0.005 ± 0.009 | 0.36 ± 0.08 |
| MADT | 61.4 ± 1.1 | 0.010 ± 0.008 | 0.12 ± 0.09 | 62.4 ± 0.6 | 0.025 ± 0.008 | 0.12 ± 0.09 |
| **TAGET** | **63.2 ± 1.1** | **0.140 ± 0.080** | **0.98 ± 0.15** | **62.9 ± 1.3** | **0.080 ± 0.010** | **0.77 ± 0.15** |

**Observation that supports our thesis (§1, §10.1).** Several headline margins are inside the
error bars. LBF-4: TAGET 0.140 ± 0.080 vs. DT 0.098 ± 0.008 — the claimed **+37.25%** has a
confidence interval overlapping the baseline. Overcooked-4: 0.98 ± 0.15 vs. CQL 0.46 ± 0.08 is
clean, but Overcooked-4 DT is 0.16 ± 0.19, an interval covering negative return on a
non-negative metric. LIAM-off swings from best-of-baselines on PP-4 (58.2) to near-zero on LBF-4
(0.005 ± 0.009). This is a concrete, citable instance of the problem OAHT-Bench exists to fix:
under-powered evaluation with inconsistent baseline implementations. **Do not overclaim it** —
the paper reports CIs honestly and the Overcooked/PP results are decisive. Use it as evidence
that shared protocols and adequate episode budgets matter, not as an attack.

Ablations (Fig. 5, p8): removing the **TA-Goal decoder** is the largest single degradation,
especially on Overcooked. Removing data pre-processing (mirroring) hurts consistently, most on
LBF.

### Implications for OAHT-Bench

1. **Adopt trajectory mirroring in `dataset/construction/collect.py`** (§4). It is a cheap, ablated,
   `N`× data multiplier, and it is a *dataset* technique rather than a method component — so it
   belongs in the benchmark's collection layer where every trajectory-view baseline can use it.
   Design decision required: mirroring is only well-defined for homogeneous action/observation
   spaces across agents. It works for LBF and Hanabi; **Overcooked-v1 has asymmetric roles in
   several layouts**, so mirroring must be per-environment opt-in, recorded in dataset metadata.
2. **Promote `teammate_observations` from optional to REQUIRED in schema v0** (§4.2). TAGET's
   team context encoder consumes `o^-i` and its TA-Goal target is built from all agents'
   observations. Without this field TAGET is not implementable.
3. **The schema needs future-state lookahead.** TA-Goal targets are observations at `t + k` with
   `k` env-dependent. Storing full per-step `teammate_observations` covers this — the loader
   computes the offset — but the loader API must expose it, and `goal_steps` must be a swept
   per-environment config value (§7.2's tuning-record contribution should include it).
4. **Fidelity validation must change.** No environment overlap means we cannot match published
   numbers. Substitutes, in descending strength: (a) reproduce the **ablation ordering** — TA-Goal
   removal should be the largest degradation, mirroring removal second; (b) reproduce the
   **`goal_steps` sensitivity shape** — non-monotonic with an env-specific optimum; (c) confirm
   TAGET beats plain DT under our protocol, which is the paper's core claim. Record these as the
   Phase 2 gate instead of numeric equality.
5. **Candidate additional baselines**, all used by TAGET and none currently in our roster:
   **Prompt-DT** (Xu et al. 2022b), **ODITS-off** (Gu et al. 2021), **MADT** (Meng et al. 2023),
   **CQL**. CQL is already in JAX-CORL (`algos/cql.py`) and would be nearly free. Prompt-DT is a
   small delta on our DT backbone. Both are worth considering as cheap roster additions; ODITS
   and MADT are not (new implementations, and ODITS-off performs poorly).
6. **TAGET is a hierarchical method, not a pure modeling module.** It does not cleanly fit
   §3.1's `module(history) -> z` interface: its high level predicts a *goal token* that replaces
   RTG in the DT prompt, rather than producing a conditioning vector alongside an unchanged
   return signal. The conditioning interface must be general enough to express "replace the
   prompt token," not just "concatenate to the state embedding." **This is the single most
   important design constraint on `offline/conditioning.py` and it should be settled in Phase 0b
   with TAGET explicitly in mind.**

---

## OMIS — Offline Multi-agent In-context Search

`papers/omis.pdf`. Local code at `~/Documents/Personal/Projects/OMIS/`.

### The headline finding: the decision-time search is NOT a tree search

Rev 5 of the project plan called OMIS's decision-time search "novel code with no reference
implementation" and "an open-ended research problem." **That was wrong, and the correction
materially de-risks Phase 2.** DTS is a *flat, fixed-depth rollout estimator* (§4.2, Eq. 6–10):

```
for each legal action â¹_t:                      # vmap axis 1
  for m in 1..M rollouts:                        # vmap axis 2
    for l in 0..L:                               # scan
      â¹_{t+l}  ~ π_θ(· | ŝ_{t+l}, D̂_{t+l})
      â⁻¹_{t+l} ~ μ_φ(· | ŝ_{t+l}, D̂_{t+l})
      ŝ, r̂     = P(ŝ_{t+l}, â)
    V̂_{t+L+1} = V_ω(ŝ_{t+L+1}, D̂_{t+L+1})
  Q̂(s_t, â¹_t) = (1/M) Σ_m [ Σ_{t'} γ^{t'-t} r̂_{t'} + γ^{L+1} V̂_{t+L+1} ]   # Eq. 8
π_search(s_t) = argmax_{â¹_t} Q̂(s_t, â¹_t)                                    # Eq. 9
```

No tree, no UCB, no backup, no gradient updates. It is `vmap(vmap(scan))` over
(legal actions × M rollouts × L steps) — close to ideal for JAX, and *easier* in JAX than in the
authors' PyTorch, which is why their implementation resorts to `copy.deepcopy` of a live env.
Reclassify the search from **L / open-ended** to **M / mechanical**.

**Mixing technique** (Eq. 10) — do not skip it, it is ablated as load-bearing:
`π_mix = π_search if ‖Q̂(s_t, π_search(s_t))‖ > ε, else a ~ π_θ(·|s_t, D_t)`. Falls back to the
raw actor when search confidence is low. "OMIS w/o mixing exhibits a notable performance
decrease... in LBF and OC."

**MBOMIS** (Q3, Fig. 5): a variant learning `P̂` from `(s,a,r,s')` tuples via MSE instead of using
the true simulator. Loses some performance but "still effectively improves over OMIS w/o S and
generally surpasses other baselines." → **This is the answer to §6's test-time-asymmetry
problem.** A learned-dynamics OMIS needs no privileged simulator access, making it comparable to
forward-only methods on a fair footing. Strongly consider MBOMIS as the headline OMIS entry with
true-simulator OMIS as the upper bound.

### Architecture

Causal transformer with three heads, all conditioned on in-context data `D^k_t`:
- actor `π_θ(a^{1,k,*} | s_t, D^k_t)` — BC on **best-response** actions (Eq. 3)
- opponent imitator `μ_φ(a^{-1,k} | s_t, D^k_t)` (Eq. 4)
- critic `V_ω(s_t, D^k_t)` — MSE to RTG `G^{1,k,*}_t` (Eq. 5)

**In-context data is two-part**, and the ablation shows both matter (`D^epi` more than `D^step`):
- `D^{epi,k} = {(s̃_h, ã^{-1,k}_h)}_{h=1..H}` — **episode-wise**, from *other* episodes against the
  same opponent, generated by any self-agent policy. Captures the opponent's overall behavioral
  pattern.
- `D^{step,k}_t = (s_0, a^{-1,k}_0, …, s_{t-1}, a^{-1,k}_{t-1})` — **step-wise**, current episode.

At test time `D^epi` is built by "sampling consecutive segments from the most recent `C`
trajectories in which Φ participated."

### Data requirements — the binding constraint

Pretraining step 1 is *"Train BRs against all policies in Π^train"*, and training data is
collected *"by playing against it using its BR."* **OMIS requires per-teammate best-response
trajectories.** In our §4.3 vocabulary that is exactly the `expert` variant. Consequences:

- OMIS cannot be meaningfully trained on `random` or `medium` data. It is **structurally excluded
  from part of the dataset-quality matrix**, unlike every other trajectory-view baseline.
- This is a data-requirements axis distinction (§9 axis 5) sharper than "needs teammate actions":
  OMIS needs *optimality labels relative to each teammate*. Only TAO's `teammate_id` supervision
  comes close.
- Our `expert` variant must be collected **per-teammate**, not as a single global expert. §4.3
  already says "converged ego (per-teammate best responses)" — this confirms it and makes it
  mandatory rather than nice-to-have.

### Evaluation design worth stealing (§8)

Two ideas better than what our protocol currently specifies:

1. **Graded seen:unseen ratios** rather than a binary in-distribution/OOD split. OMIS sweeps
   `[seen:unseen]` ∈ `{[10:0], [10:5], [10:10], [5:10], [0:10]}` for the test population. This
   turns distribution shift into a *dose-response curve* instead of two points, and it is nearly
   free given we already have held-out populations. **Recommend adopting.**
2. **Non-stationary teammates.** Φ switches policy every `E` episodes, `E ∈ {2, 5, 10, 20,
   dynamic}`, testing robustness to switching frequency. Our protocol assumes a fixed teammate
   per episode throughout. This is a genuine capability axis we do not currently measure, and it
   is the condition in-context methods should theoretically win. Consider as a secondary
   protocol condition; cheap to implement, and no other source in our suite tests it.

Budgets: pretraining 4000 steps; **testing 1200 episodes**; 5 random seeds; mean ± std. The
1200-episode test budget is a useful precedent against §8's noise concern (contrast TAGET's 50).

### Environments and teammates (not comparable to ours)

PP (**competitive** — self-agent is prey evading 3 predators), LBF (framed as a "social dilemma",
self-agent needs cooperation for high-level apples), OverCooked (image observations). Opponent
populations from **MEP** (maximum-entropy population-based training) — a fifth generator, not one
of our four. As with TAGET, **no direct numerical comparison is available.**

Baselines used: DRON, LIAM, MeLIBA, Meta-PG, Meta-MAPG, MBOM, OMIS w/o S, SP-MCTS. Notably OMIS's
flat DTS **outperforms SP-MCTS**, an actual MCTS — evidence that tree search is not required here.

### Implications for OAHT-Bench

1. **Downgrade the OMIS search risk** and revise §10.5/§10.4. It is a vmapped rollout estimator,
   not a research problem. The Nov→Dec cut line for "OMIS decision-time search" is likely
   unnecessary; keep it as a formality but expect to clear it.
2. **Reconsider the Hanabi exclusion for search.** Rev 5 scoped search to LBF + Overcooked on the
   grounds that turn-based hidden-information search is a different algorithm. With a flat
   rollout estimator the objection weakens — the real obstacles are (a) sampling opponent actions
   under a legal-action mask, which `μ_φ` handles natively, and (b) that Hanabi's true state is
   not observable to the searcher, so rollouts must use the *agent's* belief rather than `P`.
   (b) is still a genuine problem. Keep Hanabi search out, but record the reason precisely:
   it is partial observability of state, not search complexity.
3. **Add MBOMIS (learned dynamics) to the roster.** It converts OMIS from
   "requires a simulator" to "forward-only + learned model," which is what makes the §6 asymmetry
   measurable rather than merely discussed. Cost is one dynamics model, and the paper reports the
   MSEs are small.
4. **`expert` dataset variant must be per-teammate best responses and is mandatory**, not
   optional — OMIS does not exist without it.
5. **Adopt graded seen:unseen ratios in §8**, and consider non-stationary teammate switching as a
   secondary condition.

---

## TAO — Offline Opponent Modeling with Truncated Trajectories / policy embeddings

**ICLR 2024.** `papers/tao.pdf`. Local code at `~/Documents/Personal/Projects/TAO/`.

### CORRECTION to the TAGET section above: MeLIBA *does* have a published offline conversion

The TAGET section concluded "offline MeLIBA remains a definition we author." **That is wrong.**
TAO's baselines (§4.1, p6) are: DRON-concat, DRON-MoE, **LIAM**, **MeLIBA**, Prompt-DT, TAO w/o
PEL — all evaluated in the offline opponent-modeling (OOM) setting. Verbatim: *"Given that most
opponent modeling approaches employ online learning settings, we draw comparisons with
Embedding-based Opponent Modeling approaches — these are comparatively straightforward to adapt
to the OOM setting. To ensure an equitable comparison, we mandate all approaches to use the same
neural architecture as ours (see Appendix F)."*

Two consequences, both favourable:

1. **Both LIAM and MeLIBA have published offline variants** — LIAM in TAGET (ICML 2025) and both
   in TAO (ICLR 2024). The user's original premise was right; my TAGET-only reading was too
   narrow. §3.1's citable-precedent argument stands for both methods.
2. **TAO is a precedent for the shared-backbone methodology itself.** "We mandate all approaches
   to use the same neural architecture as ours" is precisely §3.1's design, applied by an ICLR
   paper to this exact family of methods. This is the strongest available answer to the §11
   reviewer objection that a shared backbone misrepresents the published methods — it is
   established practice in the subfield, not our invention.

**Follow-up needed:** read TAO Appendices F and G for the shared-architecture specification and
the per-baseline adaptation details. That is the most directly reusable text in any of these
papers for our §3.1, and it was not in the main body.

### Architecture — three stages

**Stage 1: Policy Embedding Learning (PEL).** Opponent Policy Encoder `M_θe: T^-1 → Z`,
implemented on **GPT-2**. Fuses embedding tokens of `(a^-1, r^-1, o^-1)`, then average-pools
(`AP`) the per-timestep tokens into an *average trajectory embedding* `z̄^-1`. Two losses:
- *Generative* (Eq. 2): conditional imitation — an ancillary decoder `π_φd(a | o, AP(M_θe(τ_j^{-1,k})))`
  predicts the opponent's actions from the opponent's own observations, conditioned on an
  embedding computed from a **different trajectory of the same opponent**. Cross-trajectory
  conditioning is the point: it forces the embedding to carry policy identity rather than
  episode specifics.
- *Discriminative* (Eq. 3): InfoNCE with temperature `p`, positives defined by **opponent policy
  type label** (index into `Π^off`). **This is the `teammate_id` supervision our §4.2 anticipated
  — TAO is the method that makes that field mandatory rather than diagnostic.**
- `L_emb = α · L_gen + λ · L_dis` (Eq. 4).

**Stage 2: In-context Control Decoder (ICD).** `M_θd: Z × T^1 → A^1`, the **same causal
transformer as Decision Transformer** (Chen et al. 2021), with cross-attention from the opponent
embedding into the response policy. Response loss (Eq. 5) over `y_t = (G_0, o_0, a_0, …, G_t, o_t)`
where `G_t` is **return-to-go** — confirming the code-level finding that TAO is a return-conditioned
DT. Trained to predict **near-optimal actions `a^{1,*}`**.
- `GetOffD`: samples `C` trajectories of the opponent, then `H` *consecutive* fragments from each,
  and stitches them. Rationale given: play style is pronounced over consecutive timesteps (hence
  fragments) but varies across episodes (hence multiple trajectories).

**Stage 3: Deployment — Opponent Context Window (OCW).** `W` holds the most recent `C` opponent
trajectories; `GetOnD` samples `H` fragments from each and stitches. **`θ` is frozen** — no
gradient updates at test time. `a¹_t ~ M_θd(· | y¹_t; M_θe(GetOnD(W)))` (Eq. 6).

### Data requirements

- **`teammate_id` labels are required** (discriminative loss). Withholding them at eval is fine;
  withholding at training breaks stage 1.
- **Near-optimal / best-response ego actions are required** (stage 2 target). Same constraint as
  OMIS → the per-teammate `expert` variant is mandatory for TAO as well. Two of our five
  trajectory-view baselines now depend on it.
- Needs **opponent observations and rewards** (`o^-1, a^-1, r^-1` are all fused in the encoder),
  reinforcing the promotion of `teammate_observations` to a required schema field.

### Environments and protocol (no overlap with ours)

**Markov Soccer** (two-player zero-sum) and **Particleworld Adversary** (three-player non-zero-sum).
**Both competitive.** TAO is an *offline opponent modeling* method, and its "opponent" framing is
literal, not a synonym for teammate. Porting it to cooperative AHT is a genuine setting change,
not just an environment change — worth stating explicitly in the paper, since a reviewer familiar
with TAO will notice.

Protocol: three test settings **seen / unseen / mix**; non-stationary opponent switching every
`E = 50` episodes; **2500 evaluation episodes**; 2000 training steps; 5 seeds; mean ± SEM.

Theory: TAO is proven equivalent to **PSOM** (Posterior Sampling in Opponent Modeling) — the same
device OMIS uses in its Lemma 4.1. The two papers share a theoretical frame.

### Implications for OAHT-Bench

1. **Read TAO Appendix F/G before finalizing §3.1.** It contains an existing, published,
   shared-architecture protocol for exactly our baseline family.
2. **`teammate_id` and per-teammate `expert` data are both hard requirements**, not optional
   schema fields. TAO needs identity labels; TAO and OMIS both need best-response actions.
3. **Non-stationary teammate switching now has two independent precedents** (TAO `E=50`, OMIS
   `E ∈ {2,5,10,20,dyn}`). Combined with seen/unseen/mix graded splits, this is the protocol
   design the competitive-OM half of the field already uses, and our §8 currently lacks both.
   Recommend adopting both, and noting in the survey that the cooperative-AHT literature has
   *not* converged on them — a concrete taxonomy/methodology observation.
4. **Evaluation budgets in this literature are large** — TAO 2500 episodes, OMIS 1200, versus
   TAGET's 50. Our §8 noise concern is real and the budget should be set nearer the former.

### TAO Appendix F/G — the shared-backbone protocol, already published

This is the most directly reusable text in any of the papers. TAO specifies a common architecture
and then states, per baseline, exactly what is added to it. **This is §3.1's design, published at
ICLR 2024, applied to our exact baseline family.** Reproduce this structure and cite it.

**The shared backbone (TAO's ICD).** GPT-2 decoder, **3 self-attention blocks**; each block =
single-head self-attention + single-head cross-attention + feed-forward, with residual connections
and LayerNorm after each layer, dropout on both the residual connection and the attention weights.
Inputs `G¹_t` (RTG), `o¹_t`, `a¹_t` pass through **modality-specific linear layers** plus a
positional *episodic timestep* encoding (as in Chen et al. 2021). Actions predicted
autoregressively under a causal mask; the hidden states **at the `o¹_t` token positions** are fed
to a linear head to output actions. Feed-forward layer: **128 nodes, ReLU**; all other hidden
layers **32 nodes, no activation**; modality-specific linear layers **32 nodes, no activation**.

**TAO's OPE (encoder).** GPT-2 *encoder*, 3 self-attention blocks (single-head attention +
feed-forward). Modality-specific linear layers for `a⁻¹, r⁻¹, o⁻¹` use **ELU**, 32 nodes. A
*fusion linear layer* combines `(a⁻¹_{t-1}, r⁻¹_{t-1}, o⁻¹_t)` into one fused token per timestep.
Stage 1 average-pools the per-timestep outputs into `z̄⁻¹`; stages 2–3 feed the full token
sequence `z⁻¹` as **key and value into the cross-attention layers** of the ICD.

**LIAM-offline — the precise specification** (supersedes TAGET's one-sentence version):
> Backbone identical to ICD **minus the cross-attention layer**. Feed `G¹_t, o¹_t, a¹_t`, predict
> `a¹_t` autoregressively under a causal mask. Add an **extra decoder** for the auxiliary task:
> reconstruct the opponent's observations `o⁻¹_t` and actions `a⁻¹_{t-1}` **from the `o¹_t` token
> embeddings** produced by the backbone (those embeddings already contain `o¹_t` and `a¹_{t-1}`).
> Extra decoder = 2 linear layers, 32 nodes, no activation.

**MeLIBA-offline — the precise specification** (this is the conversion I earlier said did not
exist; it does, here rather than in TAGET):
> Backbone identical to ICD **minus the cross-attention layer**. Add **extra encoding layers and
> an extra decoder** implementing the VAE auxiliary task: maximize the ELBO by reconstructing the
> opponent's *future* actions `a⁻¹_t, a⁻¹_{t+1}, …` from the opponent's *future* observations
> `o⁻¹_t, o⁻¹_{t+1}, …`, plus a KL term, conditioned on the controlled agent's past trajectory
> `o¹_0, a¹_0, r¹_0, …, o¹_t`. Encoding uses **two-level hierarchical linear layers** producing
> `μ, σ` for a **permanent latent variable ("agent character")** and a **temporal latent variable
> ("mental state")**. Extra encoding layers = 2 levels × 2 linear layers, 32 nodes, no activation.
> Extra decoder = 2 linear layers **+ a recurrent layer**, 32 nodes, no activation.

Note this maps exactly onto jax-aht's `VariationalEncoderRNN` outputs that §6 of the project plan
already identified — `latent_mean` (who / permanent) and `latent_mean_t` (what now / temporal).
Our two independent sources agree on the decomposition.

**Prompt-DT** (cheap roster addition): prompts are *expert demonstrations* drawn from the offline
data — take the **top 20% by return** among trajectories against a given opponent policy, sample
1 trajectory, then sample **consecutive fragments of length 5** as the prompt. Trivial on top of
our DT backbone and it appears as a baseline in both TAO and TAGET.

**DRON-concat / DRON-MoE** also specified (2 backbones, hand-crafted opponent features, MoE gating
with 5 experts). Lower priority — hand-crafted features do not transfer across our environments.

**TAO's opponent populations (Appendix H)** are hand-built: 8 policies per environment mixing
**scripted** behaviours (SnatchAttack, SnatchEvade, GuardAttack, GuardEvade in MS; FixOne,
ChaseOne, Middle, Bounce, FixThree, ChaseBounce in PA) with **RL policies trained for differing
episode counts** (TRCoPO, TRGDA, PPO at 5k–100k episodes). This is a *fourth* population-construction
philosophy — neither our four diversity generators, nor TAGET's SVD, nor OMIS's MEP. Worth a row
in the survey: population construction is as unstandardized as everything else, and "RL policies
at different training durations" is essentially FCP's competence axis built by hand.

### Revised implications

1. **§3.1 should cite TAO Appendix F as prior art for the shared-backbone methodology**, and our
   LIAM/MeLIBA modules should follow TAO's specification, which is complete and unambiguous —
   unlike TAGET's. This converts the largest remaining "definitions we author" exposure into a
   reproduction task.
2. **The conditioning interface has a published reference design**: opponent embedding enters as
   **key/value in cross-attention**, not concatenated to the state embedding. Combined with
   TAGET's goal-token-replaces-RTG design, `offline/conditioning.py` must support at least three
   modes — cross-attention (TAO), prompt-token replacement (TAGET), and auxiliary-head-only with
   no conditioning path (LIAM, MeLIBA). Design for all three in Phase 0b.
3. **Add Prompt-DT to the roster.** Specified in full, cheap on our backbone, and it appears in
   both competitor papers — omitting it invites the question.

---

## ZSC-Eval — evaluation methodology (not a baseline)

`papers/zsc-eval.pdf`. MIT. Source of §8's BR-Prox metric.

### BR-Prox, exactly (§4.3)

```
BR-Prox(π, {π_w^i}_{i∈P}) := Aggr_{L ∈ P(P)} [ J(π, {π_w^i}_{i∈L}) / J( B̂R({π_w^i}_{i∈L}), {π_w^i}_{i∈L} ) ]
```

Ego return against a partner set, divided by the **approximate best response's** return against
that same set, aggregated over partner subsets. `Aggr` is the **inter-quartile mean (IQM)**,
chosen over mean/median for statistical reliability (following `rliable`, Agarwal et al.).
Reported with **95% CIs and inter-quartile ranges** over disaggregated scores.

**Convergence worth noting:** BR-Prox needs per-teammate approximate best responses. So do OMIS
(training data) and TAO (stage-2 targets). **One artifact — per-teammate BR policies — serves
three purposes**, which strongly justifies making the `expert` dataset variant mandatory and
first-class rather than optional (§4.3).

### BR-Div — partner *selection*, and a real conceptual distinction

- Population diversity: `PD({π_i}) := det(K)`, `K_ij = θ_i · θ_j`, where `θ` is a **behavior
  feature vector** — counts of pre-defined events over an episode.
- Partner Diversity `P-Div({π_i}) := PD({π_i})`; **Best-Response Diversity**
  `BR-Div({π_i}) := PD({B̂R(π_i)})`.
- Selection of `M` evaluation partners from `N` candidates by **Determinantal Point Process**
  sampling to maximize the determinant.

**The insight (§4.2): maximizing partner diversity is not the same as maximizing the diversity of
the responses those partners require.** Different partners may share similar best responses, so a
partner set that looks diverse can test a narrow band of ego skills. Empirically (Fig. 2b),
BR-Div-selected subsets reach substantially higher BR population diversity than P-Div-selected
ones. → **This belongs in the survey (§9) as a diversity-axis distinction**, and it is a
critique that applies directly to FCP/CoMeDi/BRDiv/L-BRDiv, all of which optimize properties of
the *population* rather than of the *required responses*.

Also of note: they deliberately include **earlier checkpoints** of selected candidates to widen
the skill-level range, targeting `J(B̂R(π̂_w), π̂_w) ≈ J(B̂R(π_w), π_w)/2` — a principled version
of FCP's "snapshot during training."

Behavior-preferring rewards: `R^BP = {r_w | r_w(s,a) = r + φ(s,a)ᵀw}` with `‖w‖_∞ ≤ B_max` and
at most `C_max` non-zero entries; `φ` embeds event-based features; the **original game reward `r`
is retained to prevent sabotage**. One agent gets `r_w`, others get `r`; NE approximated by
independent PPO.

### Empirical finding that affects our layout choice

§5.2: *"the commonly used layouts, Forced Coord. and Asymm. Adv. **fail to differentiate
algorithms' performance**. We have also noticed that SP performs well in these layouts,
indicating that it can easily learn most of the skills for interacting with unseen partners."*
They classify layouts by resource-sharing — "Limited" (Forced Coord., Asymm. Adv., Bothway Coord.)
vs "Full" (Coord. Ring, Counter Circ., Blocked Corr.) — and find the **Full Resource-sharing**
layouts discriminate better.

→ **This challenges the Tier 1 choice in the project plan (§10.3), which nominated
`cramped_room`.** Two of the five jax-aht layouts (`forced_coord`, `asymm_advantages`) are
documented as non-discriminative, and `cramped_room` is the simplest of the set. **Recommend
Tier 1 Overcooked = `counter_circuit` or `coord_ring`** (both Full Resource-sharing, both with
downloadable populations for cross-checking), keeping `cramped_room` in Tier 2 as the
easy-reference layout. Getting this wrong means the headline Overcooked column shows all methods
tied — the single most avoidable way to waste the environment.

### Implications for OAHT-Bench

1. **Adopt IQM + 95% CI + IQR (rliable) as §8's aggregation**, replacing "mean ± CI over seeds."
   Cheap, standard, and it is what the metric's source paper does.
2. **Consider BR-Div for held-out teammate selection** rather than random split — a principled,
   published way to choose the evaluation population, and it needs only the per-teammate BRs we
   are already computing.
3. **Re-nominate Tier 1 Overcooked layout** to a Full Resource-sharing one.
4. **P-Div vs BR-Div is a survey axis** (§9) and a fair critique of all four of our generators.

---

## ICRL4AHT — Benchmarking the Limits of In-Context RL for Ad-Hoc Teamwork

`papers/icrl4aht.pdf`. Source of AD, DPT, AMAGO-Offline, Hybrid-AD, and our HDF5 format.

### This is a negative-results paper, and that reframes our expected findings

Rev 1–5 of the project plan treated ICRL4AHT purely as a code source. It is also the closest
thing to a prior benchmark in this space, and **its headline result is that the entire
learning-history family fails**:

- **AD and DPT do not consistently beat a random baseline.** Track 1 averages: DPT
  **12.4 ± 11.0**, AD **9.1 ± 5.8**, Random **5.5 ± 5.6** — and the authors note this is "heavily
  skewed by the H4 family, which is autonomously capable and requires minimal coordination."
- **Catastrophic failure under tight coupling.** On `test wide` × H1: AD **−18.0 ± 2.0**, DPT
  **−23.4 ± 2.5**, Random **0.0 ± 0.0**. Verbatim: the agents *"not only fail to coordinate but
  actively interfere with the teammate's sub-optimal policy."*
- **No in-context learning actually occurs.** Adaptation curves are flat — *"episode-wise returns
  do not trend upward over the testing horizon."* Quantified with an **"adaptation gain"** metric
  (mean of last 20 episodes minus first 20), which confirms near-zero online improvement.
- **Track 2 (layout generalization): random dominates.** Random **11.7 ± 11.0** beats AD
  **4.4 ± 5.8** and is competitive with DPT **13.2 ± 11.2**.
- **The failure is not an artifact of any single design choice.** Ablated and ruled out:
  teammate-action conditioning (+TA, inconsistent), context length (K = 250 → 2000, no
  correlation; extended to 10,000 in appendix, unchanged), model scale (4/8/12 layers, 256/512/768
  hidden, 20k/50k/100k steps — flat curves persist), recurrent architecture (Hybrid-AD, marginal),
  and training paradigm (**AMAGO-Offline**, an offline-RL objective rather than action prediction,
  is *comparable* on the teammate track and *weak* on the layout track).
- Their conclusion: *"current sequence-model architectures — regardless of their design or whether
  they are trained via action-prediction or offline RL — struggle to perform the implicit Bayesian
  partner inference required for AHT under realistic distribution shifts."*

**Implications for OAHT-Bench, in order of importance:**

1. **A random-action baseline must appear in every table**, alongside %BC. ICRL4AHT shows random
   beating AD outright on one full track. Our §3.1 currently has %BC as the floor; **random is a
   *lower* floor that published methods have already failed to clear**, and omitting it would be
   a conspicuous gap given this paper exists.
2. **Adopt the "adaptation gain" diagnostic** (last-20 minus first-20 episode return) into §8. It
   directly measures the property in-context methods claim, is nearly free to compute, and it is
   the metric that exposed the flat curves. Our current metric suite would not have caught this.
3. **§10.1 claim 1 has a strong prior in our favour but is partly pre-empted.** "The field's
   ranking is not what the literature implies" is already demonstrated for the learning-history
   family in one environment. Our contribution must be the part they explicitly disclaim (below),
   not a replication.
4. **Their stated limitations are our thesis.** Verbatim §6: *"it is instantiated on a **single
   domain** (OvercookedV2), employs a **finite teammate suite**, and restricts evaluation to
   **two-player settings with fixed partners**."* Our seven environment configurations, four
   generator families plus heuristics, and (if adopted) non-stationary teammate switching address
   all three. **This is the cleanest available positioning statement for the paper** — quote it.
5. **Expect a mostly-negative headline and plan for it.** If the trajectory-view family (LIAM,
   MeLIBA, TAO, OMIS, TAGET) also lands near random across three environments, the paper's finding
   is "offline AHT does not work yet, and here is the instrument that shows it." That is a
   publishable ICML result *if* the instrument is credible — which puts even more weight on the
   protocol, the floors, and the diagnostics than on the baseline count.

### Protocol and teammate suite (directly reusable)

**Manifest specification**: version-controlled JSONL manifests decoupling task definition from
environment execution — evaluation tracks, layout configurations, teammate assignments
(checkpoints, seeds) for both splits. **Adopt this pattern**; it is what makes their task
distribution exactly reproducible, and it composes with our §4 dataset metadata.

**Teammate suite, deliberately split by construction method:**
- **RL policies for training only** — FCP, BRDiv, LBRDiv, CoMeDi. *Our exact four generators.*
- **Heuristic policies for testing only** — four families ordered by increasing "cooperability":
  `H1 recipe_aware < H2 territory < H3 assembly_line < H4 utility_greedy`. H1 demands the most
  from the ego agent, H4 the least.

This is a **graded OOD axis** like OMIS's seen:unseen ratios, and it is the third independent
precedent for grading distribution shift rather than treating it as binary. Our §8 should adopt
a cooperability-ordered heuristic axis; jax-aht's shipped scripted agents (§7.5 of the project
plan) can be ordered this way for all three environments.

**Two evaluation tracks**, worth copying wholesale:
- Track 1 **Teammate Generalization**: `L_train × Π_test`, `Π_train ∩ Π_test = ∅`.
- Track 2 **Layout Generalization**: `L_test × Π_test`, `L_train ∩ L_test = ∅` — dual shift.
  Their layout pairs share geometry but require distinct strategic roles (`asymm adv both` →
  `asymm adv right`; `cramped up` → `cramped down`).

### Dataset scale — a warning for our §4

Table 2, for **one** environment (OvercookedV2):

| Property | Value |
|---|---|
| Original transitions | **1,196,032,000** |
| Filtered transitions | **149,504,000** |
| History length | 14,600 |
| Num. teammates | 80 |
| Obs shape | (5, 5, ★), ★ ∈ {40, 41, 45, 46} by layout |
| Mean final return | ≈ 40 |
| **Disk size (compressed)** | **≈ 6.5 GB** |

Two things our plan does not currently account for:

1. **Scale.** 6.5 GB compressed for a single environment's learning-history view. We plan seven
   environment configurations × four generators × up to five dataset variants. Naive
   extrapolation puts the artifact in the **hundreds of GB**, which affects collection time,
   cluster storage, and — critically — whether the released dataset is actually hostable.
   §10.2's Phase 0a should include a storage budget and a decision on what subset is released
   versus regenerable from configs.
2. **Trajectory filtering is a required curation step, not an optional one.** They filter
   1.196B → 149.5M transitions (**8× reduction**), *"select[ing] high-quality learning curves
   based on final performance and improvement, ensuring the Transformer learns genuine improvement
   dynamics."* Our §4 collection spec has no filtering stage. Without one, the `replay` variant
   is dominated by learning curves that never improved, and AD/DPT are being trained on noise.
   **Add a documented, configurable filtering stage to `dataset/construction/collect.py`** and record the filter
   in dataset metadata — it is a dataset-design decision that materially changes results and
   therefore belongs in the benchmark contract.

---

## Teammate generation: FCP, CoMeDi, BRDiv, L-BRDiv

Read to support §7.2's claim that per-environment tuning is a publishable contribution. The
headline finding is that **the four generators optimize four genuinely different notions of
"good population," and this is a stronger survey axis than rev 6 §9 axis 8 states.**

### The four objectives, precisely

| Generator | What it optimizes | Construction |
|---|---|---|
| **FCP** | **Competence diversity** — skill levels, plus incidental convention diversity from independent seeds | `N` independent self-play runs; **3 checkpoints each: init, 50%-of-final-reward, converged** |
| **CoMeDi** | **Reward-aligned semantic diversity** — behaviours that are *task-relevantly* different, not merely statistically divergent | Sequential/greedy: add `π_n` maximizing self-play, minimizing cross-play **against the single most compatible existing convention**, maximizing mixed-play |
| **BRDiv** | **Best-response coverage** — teammates whose optimal responses differ | Simultaneous; maximize `Tr(C) + Σ_{i≠j}(C_ii − C_ij) + Σ_{i≠j}(C_ii − C_ji)` over the conf×br cross-play matrix, via **MAA2C** |
| **L-BRDiv** | Same target (**Minimum Coverage Set**), but the weights are *learned* | Constrained optimization solved via **Lagrange duality**, multipliers `α₁, α₂ ≥ 0` learned by SGD, policies by **MAPPO** |

**Correction to my earlier note in the ZSC-Eval section.** I wrote that "all four of our generators
optimize properties of the population rather than of the responses the population demands." That
is **wrong for BRDiv and L-BRDiv** — both are explicitly best-response-oriented; L-BRDiv's whole
framing is the *Minimum Coverage Set*, the smallest set containing a best response to every
teammate in Π. The correct statement is: **FCP and CoMeDi are population-oriented; BRDiv and
L-BRDiv are response-oriented; ZSC-Eval's BR-Div is the same idea applied to *selection* rather
than *generation*.** That split is a much better survey axis than "population vs response" applied
uniformly, and it means our four generators already span the distinction — good for the paper.

### FCP specifics (Strouse et al.)

- `N = 32` partners, **3 checkpoints each**, mid-checkpoint defined as **"when the agent reaches
  50% of its final reward."** ZSC-Eval independently uses `J(BR(π̂))≈J(BR(π))/2` for the same
  purpose — **two sources converge on half-performance as the mid-skill criterion.** Adopt it
  explicitly rather than "some checkpoint during training"; it is a real, reproducible definition
  and §7.2 should record it.
- **Ablations settle a tuning question for us**: `FCP₋T` (converged checkpoints only)
  *significantly reduces* performance, while `FCP₊A` (varying architecture across the population)
  offers **no improvement** over past checkpoints. → **Do not spend budget on architectural
  diversity in our populations; do spend it on checkpoint diversity.**
- Held-out evaluation populations include **randomly-initialized agents** to test generalization
  to low-skill partners — a third precedent for low-skill/degenerate teammates in the eval set.
- V-MPO + ResNet + LSTM; Overcooked with **the same five layouts jax-aht uses**.

### CoMeDi specifics — and a confound for our cross-play diagnostic

Loss (Eq. 8): `L(π_n) = −J(π_n, π_n) + α·J(π_n, π*) − β·J_M(π_n, π*)`, where `π*` is the **most
compatible** convention in `D_{n-1}` and `J_M` is mixed-play return. §7.3's code-derived note
("maximizes self-play, minimizes cross-play") is right but under-specifies: cross-play is
minimized **only against the single most-compatible existing convention**, and construction is
**greedy/sequential**, not simultaneous.

**The handshake problem is a direct threat to §8's cross-play matrix diagnostic.** CoMeDi §3.2
documents that pure cross-play minimization produces agents that develop *handshakes*: at the
first timestep both agents emit an identity-revealing action; if the handshake matches they
cooperate, otherwise they **deliberately sabotage**. This yields high self-play and low cross-play
*while the conventions remain semantically similar* — the metric is fooled.

Our §8 uses the cross-play matrix `C[τ, j]` as a headline diagnostic of population structure. If
any population contains handshake behaviour, **low off-diagonal entries measure sabotage
signalling rather than genuine convention difference**, and any conclusion we draw about
population diversity from that matrix is contaminated. Mitigations: (a) CoMeDi's **mixed-play** is
the published fix and jax-aht implements CoMeDi, so verify mixed-play is enabled and record the
`β` used; (b) add a **handshake probe** — compare cross-play return when the first `k` timesteps
are forced to self-play actions versus not; a large gap indicates handshaking. That probe is cheap
and I have not seen it reported anywhere, so it is a plausible small contribution in its own right.

### BRDiv / L-BRDiv specifics — and a sharpening of §7.3

BRDiv metric (Eq. 6) expands to weights of `1 + 2(K−1)` on each diagonal entry and `−1` on each
off-diagonal. L-BRDiv's Alg. 1 line 7 gives `w^{i,j} = 1 + Σ_{k≠j}(α₁^{i,k} + α₂^{i,k})` for
`i = j` and `−(α₁^{i,j} + α₂^{i,j})` otherwise — **which reduces exactly to BRDiv's weights when
all `α = 1`.** L-BRDiv is BRDiv with the fixed weights replaced by learned multipliers, and the
paper's stated selling point is precisely that this **removes the need to tune `α`**.

**[CORRECTED — see below.]** I initially read this as showing §7.3's "BRDiv's `XP_LOSS_WEIGHTS`
is already population-size-invariant" to be imprecise, since the *paper's* metric gives a
self-play:cross-play weight ratio of `(1 + 2(K−1)) : 1` that varies with `K`. **That reading was
wrong.** It conflates the paper's diversity metric with the implementation's per-sample
policy-gradient weights. `BRDiv.py:389-391` builds
`sp_weight = (1 + 2·XP_LOSS_WEIGHTS)·(n/2)` and `xp_weight = XP_LOSS_WEIGHTS·(n/(2(n−1)))`, and
against the sampling distribution `P(SP) = 1/n`, `P(XP) = (n−1)/n` the expected contributions are
`(1 + 2·XP_LOSS_WEIGHTS)/2` and `XP_LOSS_WEIGHTS/2` — **exactly independent of n** (verified
numerically at n = 3, 5, 10, 20). The `n` factors exist precisely to cancel the sampling
probabilities. The original §7.3 claim stands: **do not rescale `XP_LOSS_WEIGHTS`.**

The genuine contrast with L-BRDiv remains: L-BRDiv's `α` are *learned* by SGD on an unnormalized
sum over ~n² pair terms, so `LAGRANGE_LR` must be scaled by ~(n_ref/n)². BRDiv has no learned
multiplier and therefore no such pathology.

Separately confirmed empirically: an n=5 BRDiv run collapsed to a near-flat cross-play matrix, and
the fix was raising the sample budget (`NUM_ENVS` 64→128, `TOTAL_TIMESTEPS` 4.5e7→7e7, ≈3×) to
offset the 1/n² self-play dilution — not adjusting the diversity weight. Direction established;
values are sweep starting points, not tuned.

Other details: BRDiv uses **MAA2C** with a shared centralized critic that estimates cross-play
matrix entries directly (one-hot `(i, −j)` concatenated to the critic input), with separate `D^SP`
and `D^XP` buffers emptied after each update. L-BRDiv uses **MAPPO** and bootstraps with a critic
rather than Monte Carlo. L-BRDiv's near-zero threshold `τ > 0` in the constraints exists to
prevent discovering duplicate policies. Both evaluate on small environments — BRDiv on
Cooperative Reaching (5×5), LBF (**6×6, 3 objects, reward 0.33**), Simple Cooking; L-BRDiv adds a
repeated matrix game and Weighted Cooperative Reaching, with `Π^eval` built from **hand-crafted
heuristic agents**, 4 seeds, 95% CI.

Note that neither BRDiv nor L-BRDiv perfectly recovers the MCS in LBF — L-BRDiv finds 4–5 of 6
possible collection orderings, baselines fewer. **Populations in LBF are known-incomplete even for
the method designed to complete them**, which is worth stating when we report LBF results.

---

## LIAM — Local Information Agent Modelling

`papers/liam.pdf`. Papoudakis, Christianos, Albrecht, NeurIPS 2021.

**Architecture.** Recurrent encoder `f_w: τ¹ → Z` conditioned on `(o¹_{1:t}, a¹_{1:t-1})` → `z_t`.
Decoder `f_u: Z → τ⁻¹` with **two heads**: observation reconstruction `f^o_u` (squared error) and
action reconstruction `f^π_u` (categorical log-likelihood).

```
L_ED = (1/H) Σ_t [ (f^o_u(z_t) − o⁻¹_t)² − log f^π_u(a⁻¹_t | z_t) ]        (Eq. 2)
```

**The detail that matters most for §3.1:** *"We do not back-propagate the gradient from the
actor-critic loss to the parameters of the encoder,"* and separate learning rates are used for RL
and encoder-decoder. **The encoder is trained purely by reconstruction.** MeLIBA does the same
(below). Both methods therefore already have exactly the structure our modeling-module interface
assumes — an independently-trained module producing `z`, consumed by a policy that does not shape
it. This is strong evidence that the §3.1 factorization is faithful rather than imposed.

Policy operates on augmented space `O¹_aug = O¹ × Z`. RL via A2C. Decoder output dimension grows
linearly with the number of agents.

**Environments:** Double Speaker-Listener, **LBF (20×20, 2 agents, 4 foods, sparse reward
normalized to 1, 50 timesteps, obs radius 4)**, Predator-Prey (controlling the *prey*).
→ **This is the same LBF configuration TAGET uses.** TAGET inherited its LBF setup from LIAM.
Still not ours (12×12 Jumanji), but it explains the lineage and means TAGET/LIAM numbers are
mutually comparable even though neither is comparable to ours.

**Its own baselines are worth noting** — the field had these ideas before TAO:
- **FIAM** — LIAM with access to the modelled agent's trajectory *at execution time*. An explicit
  **oracle upper bound**. → We should add an equivalent: an oracle-teammate-model ceiling row.
  It bounds how much of the remaining gap is attributable to modeling at all, and it is cheap.
- **NAM** — no agent model, recurrent policy (≈ RL²). A lower bound.
- **CBAM** — classification-based: reconstruct the *identity* of the teammate policy, policy
  conditioned on predicted identity. Uses identity labels. → This is TAO's discriminative loss,
  three years earlier.
- **CARL** — contrastive representation learning baseline.

10 fixed policies per environment; 5 seeds; 95% CI (two SEM); evaluated every 1000 training
episodes for 100 episodes.

---

## MeLIBA — Meta-Learning Interactive Bayesian Agents

`papers/meliba.pdf`. Zintgraf, Devlin, Ciosek, Whiteson, Hofmann.

**Hierarchical latent structure**, from Rabinowitz et al.'s Machine Theory of Mind:
- **permanent latent `m`** ("agent character") — fixed for the agent's lifetime
- **temporal latent `m_t`** ("mental state") — changes each timestep, models *non-stationary*
  teammates whose policies depend on interaction history

Confirms the `latent_mean` / `latent_mean_t` split in jax-aht and TAO Appendix F's
permanent/temporal description. **Three independent sources agree.**

**Objective.** Hierarchical sequential VAE. ELBO (Eqs. 6–9) predicts *other agents' future
actions* over all remaining timesteps `k = t+1..H`. The KL prior is **the previous posterior**,
`KL(q(m, m_t | τ_{:t}) ‖ q(m, m_t | τ_{:t-1}))` — *"akin to a filtering-type Bayesian update. This
incentivises the posterior distribution to change slowly over time."* Overall:

```
L(φ,θ,ψ) = E[ J(φ,ψ) + λ Σ_t ELBO_t ]                                     (Eq. 10)
```

**Two design points that constrain our conditioning interface:**

1. **The policy conditions on the full posterior `(μ_t, σ_t)`, not a sample.** This is MeLIBA's
   explicit criticism of LIAM/LIOM: *"they condition their policy on a sample from the approximate
   belief... this means they cannot be Bayes-optimal, since the agent cannot take into account its
   uncertainty."* → `offline/conditioning.py` must be able to pass **mean and variance**, not just
   a point embedding. Add this to the three modes already recorded in §3.1.
2. **RL loss is not backpropagated through the encoder** (*"we train the policy using PPO, and do
   not backpropagate the RL-loss through the encoder"*), alternating VAE and RL updates — matching
   LIAM.

**Diagnostic precedent:** they train a logistic-regression classifier to predict teammate type
from the latent states (Fig. 3c, "Agent Type Prediction Accuracy"). **This is our §8
identity-recoverability probe, already used by one of our baselines.** Good — it means the
diagnostic is established practice, not our invention, and MeLIBA is a natural sanity-check case.

Environments: Game of Chicken (matrix), Treasure Hunt (10×10 gridworld, horizon 100). 3 seeds.
Baselines: RL² (backprops RL loss through encoder, no hierarchy), LIOM (samples the latent,
non-hierarchical), MeLIBA-m (permanent latent only), "Average" policy.

---

## AMAGO

`papers/amago.pdf`. Grigsby et al., ICLR 2024. Off-policy in-context RL; ICRL4AHT adapted it to
offline.

- **One shared sequence model for actor and critic** with two output heads; no target sequence
  models. The trick enabling simultaneous updates is to **detach the critic from the actor loss**.
- **Multi-gamma learning**: many discount factors optimized in parallel, each with its own Q
  surface; `γ ≥ 0.999`. Cost amortizes as sequence-model size grows.
- **Filtered BC term as a fail-safe** — supervised learning on actions with positive advantage
  `Q(s,a) − V(s) > 0`. Note AMAGO already contains a %BC-like component internally.
- **Attention entropy collapse** is identified as *the* key instability for long-sequence RL:
  *"agents can converge on precise memory strategies that consistently recall specific timesteps
  of long sequences... encourag[ing] large dot products between a small set of queries and keys
  that can destabilize attention."* Fixed with **Normformer + σReparam** and **Leaky ReLUs** to
  preserve plasticity. Context length `l = H` (full rollout horizon).
- **Hindsight instruction relabeling** — a multi-step-goal HER variant. **Does not transfer to our
  setting** (no goal tokens in AHT); scope it out explicitly rather than half-implementing it.

**Two implications:**
1. **Attention entropy collapse is a real risk for our learning-history view.** ICRL4AHT's
   histories are 14,600 steps long. If our AD/DPT/AMAGO implementations show flat or unstable
   training, this is the first thing to check, and AMAGO's stabilizers are the published fix —
   worth applying to the shared backbone generally, not just to AMAGO.
2. AMAGO's central claim is that optimizing the **true RL objective** beats a sequence-modeling
   loss (it beats AD directly on Dark Key-To-Door). **ICRL4AHT tested exactly this claim in AHT
   and found AMAGO-Offline no better than AD/DPT.** That is a meaningful, citable negative result
   about AMAGO's core thesis in the multi-agent setting, and it is one of the sharper findings
   available to our survey.

---

## Decision Transformer

`papers/decision-transformer.pdf`. Chen et al., NeurIPS 2021. Backbone reference for §3.1.

`τ = (R̂₁, s₁, a₁, R̂₂, s₂, a₂, …)` with `R̂_t = Σ_{t'≥t} r_{t'}`. Last `K` timesteps →
**`3K` tokens**. Per-modality linear embedding + LayerNorm; **a learned embedding per *timestep*
is added to each token — explicitly not the standard positional embedding, since one timestep
spans three tokens.** GPT with causal masking; the hidden state at each `s_t` token predicts `a_t`
(cross-entropy for discrete, MSE for continuous). At evaluation, specify a **target return**, then
decrement it by the achieved reward each step. `K = 30` for Atari.

Everything TAO and TAGET do architecturally follows from this; TAGET's `K = 30` and `3K`
sub-sequence sampling are DT's defaults carried over.

**One remark that frames TAGET nicely:** DT reports *"We did not find predicting the states or
returns-to-go to improve performance."* TAGET's entire contribution is predicting a
**teammate-aware** return-to-go and goal. So the multi-agent setting is precisely where predicting
the conditioning signal starts to pay — a clean way to motivate the trajectory-view family in the
survey.

---

## IQL — Implicit Q-Learning

`papers/iql.pdf`. Kostrikov, Nair, Levine. Our backbone-sensitivity ablation; implementation in
JAX-CORL `algos/iql.py`.

Expectile regression with `L₂^τ(u) = |τ − 1(u<0)| u²`. Value: `L_V(ψ) = E[L₂^τ(Q_θ̂(s,a) − V_ψ(s))]`
(Eq. 5). Q: `L_Q(θ) = E[(r + γV_ψ(s') − Q_θ(s,a))²]` (Eq. 6). Policy extraction by **advantage-
weighted regression**: `L_π(φ) = E[exp(β(Q_θ̂(s,a) − V_ψ(s))) log π_φ(a|s)]` (Eq. 7). Clipped
double Q. As `τ → 1`, `V_τ(s) → max_{a: π_β(a|s)>0} Q*(s,a)`.

**Never queries the learned Q on out-of-sample actions** — the property that makes it the right
contrast to DT. And *"the policy does not influence the value function in any way, and therefore
extraction could be performed either concurrently or after TD learning"* — convenient for our
modular design, since the conditioning module can be trained against the value stage independently.

Two hyperparameters to sweep: expectile `τ` and AWR inverse temperature `β`.

---

## D4RL

`papers/d4rl.pdf`. Fu et al. Source of §4.3's dataset-regime vocabulary.

**Exact definitions**, which our §4.3 should match rather than paraphrase:
- `random` — unroll a **randomly initialized** policy.
- `medium` — train online (SAC), **early-stop**, then collect from the partially-trained policy.
- `medium-replay` — **all samples in the replay buffer observed during training until the policy
  reaches "medium" performance.** *Not* the full training history — it stops at medium. Our
  `replay` variant should either match this or state the deviation.
- `medium-expert` — **equal amounts** of expert and suboptimal data. Our §4.3 says "expert +
  medium union"; specify the ratio explicitly, since 50/50 is D4RL's choice and mixture ratio is
  known to matter.

**Task design factors** (§4), several of which apply to us and are worth citing:
- **Stitching** — combining sub-trajectories to solve a task. This is DT's documented weakness and
  the mechanism behind our DT-vs-IQL ablation (§3.1).
- **Suboptimal data**, **sparse rewards**, **narrow/biased distributions**.
- **"Non-representable behavior policies, non-Markovian behavior policies, and partial
  observability"** — flagged as introducing modeling errors *"especially in methods that assume
  access to action probabilities from a Markovian policy."* **Offline AHT is inherently both**:
  partially observable, and the behavior policy is non-Markovian because it is conditioned on a
  teammate. D4RL's own taxonomy classifies our setting as one of the hard cases — a useful
  citation for §1.

Datasets are typically ~10⁶ steps, which is three orders of magnitude below ICRL4AHT's 1.2 × 10⁹.
The learning-history view is a fundamentally larger artifact than a D4RL-style dataset, and §4.6's
storage concern follows directly.

---

## Outstanding corrections to `offline_aht_benchmark_project.md` (post-rev-6)

Rev 6 was written after reading the first five papers. The remaining ten produced these
additional items, not yet folded into the plan:

1. ~~**§7.3 wording is imprecise about BRDiv.**~~ **RETRACTED.** This item was wrong and was
   briefly folded into rev 7 before being retracted in rev 8. `XP_LOSS_WEIGHTS` really is
   population-size-invariant — `BRDiv.py:389-391` cancels the sampling probabilities exactly. The
   original claim needed no correction. The real L-BRDiv/BRDiv contrast is the *learned* multiplier,
   which stands.
2. **§9 axis 9 is wrong as written.** BRDiv and L-BRDiv *are* response-oriented (Minimum Coverage
   Set). Correct split: **FCP and CoMeDi are population-oriented; BRDiv and L-BRDiv are
   response-oriented; ZSC-Eval's BR-Div applies the same idea to selection.** Our four generators
   already span the distinction, which is better for the paper than the blanket critique.
3. **Conditioning interface needs a fourth capability**: pass **distribution parameters `(μ, σ)`**,
   not just a point embedding. MeLIBA's Bayes-optimality argument depends on the policy seeing its
   own uncertainty.
4. **Add an oracle-teammate-model ceiling row** (LIAM's FIAM): a variant with privileged access to
   teammate trajectories at execution. Cheap, and it bounds how much headroom teammate modeling
   has at all — directly relevant if results come in near-random (§11).
5. **FCP mid-checkpoint has a precise definition**: *50% of final reward*. ZSC-Eval independently
   uses the same criterion. Record it in §7.2 rather than "some checkpoint during training."
6. **Do not spend budget on architectural diversity in populations.** FCP's `FCP₊A` ablation shows
   no improvement over checkpoint diversity, while removing checkpoints (`FCP₋T`) hurts
   significantly.
7. **Handshake confound in the cross-play diagnostic (§8).** CoMeDi documents that cross-play
   minimization produces identity-revealing handshakes followed by deliberate sabotage, which
   makes low off-diagonal cross-play entries uninformative. Verify mixed-play is enabled in our
   CoMeDi runs and record `β`; consider a **handshake probe** (force the first `k` steps to
   self-play actions and compare) — cheap, and apparently unreported anywhere.
8. **Attention entropy collapse** (AMAGO) is a documented long-sequence RL failure mode and our
   learning-history contexts are very long. Apply AMAGO's stabilizers (Normformer, σReparam, Leaky
   ReLU) to the shared backbone, not just to AMAGO. Scope **out** hindsight instruction relabeling.
9. **D4RL definitions to match exactly**: `medium-replay` stops at medium performance (not the
   full history); `medium-expert` is a **50/50** mixture. State any deviation.
10. **Cite D4RL's own taxonomy in §1**: it flags non-Markovian behavior policies plus partial
    observability as a hard case, and offline AHT is inherently both.
11. **LIAM and MeLIBA both detach the encoder from the policy gradient.** This is direct evidence
    that §3.1's module factorization is faithful to the originals rather than imposed — worth
    stating in the paper as support for the shared-backbone design.
12. **MeLIBA already uses our identity-recoverability probe** (logistic regression on latents to
    predict teammate type). The §8 diagnostic is established practice; cite it and use MeLIBA as
    the sanity-check case.
13. **Populations in LBF are known-incomplete**: L-BRDiv recovers only 4–5 of 6 collection
    orderings in its own LBF, baselines fewer. Worth stating when reporting LBF results.
14. **TAGET's LBF configuration is inherited from LIAM** (20×20, 4 foods, 50 steps, normalized
    reward). So TAGET and LIAM numbers are mutually comparable even though neither is comparable
    to ours — relevant if we ever implement one original environment for validation (§10.6 gate 4,
    which just got cheaper: one environment would validate two methods).
