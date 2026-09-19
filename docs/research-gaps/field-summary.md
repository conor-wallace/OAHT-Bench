# Field Summary — offline ad-hoc teamwork and in-context RL

## Scope

- **Papers analysed:** 21
- **Date range:** 2019 (D4RL preprint lineage) – 2025 (TIPR, ICRL4AHT, ICRL survey)
- **Venues:** NeurIPS, ICML, ICLR, AAAI, TMLR, arXiv preprints
- **Corpus caveats:** This is not a neutral sample of a field. It is the reading
  list assembled for building an offline-AHT benchmark, so it over-represents
  methods we intended to implement and under-represents online AHT, communication,
  human-AI coordination outside Overcooked, and the wider offline-RL literature.
  Claims about what is "untested" mean *untested in this corpus*.
- **Paper cards:** `papers/.research-gap-analysis/papers/` (21 cards, page-cited).

## Field in one paragraph

Ad-hoc teamwork asks an agent to coordinate with partners it has not trained
with. The corpus splits into four connected literatures. *Teammate generation*
(FCP, CoMeDi, BRDiv, L-BRDiv) builds partner populations, usually optimising an
intrinsic diversity objective. *Agent modelling* (LIAM, MeLIBA) learns a latent
representation of the partner from the ego agent's own observations. *Offline
opponent modelling* (TAO, OMIS, TIPR, TAGET) moves the whole problem offline,
learning partner-conditioned response policies from a fixed dataset. *In-context
RL* (AD, DPT, and the Lin and Wang theory papers) asks whether a frozen sequence
model can implement an RL algorithm in its forward pass. These are held together
by a shared substrate — the offline-RL backbones (Decision Transformer, IQL) and
the D4RL dataset taxonomy — which were all designed for single-agent, stationary
problems, and which the multi-agent papers adopt largely unmodified. The
benchmark and evaluation papers (ZSC-Eval, ICRL4AHT) are the corpus's own
attempts to notice that this substrate may not transfer.

## Dominant problem formulations

1. **Ad-hoc teamwork / zero-shot coordination** — maximise joint return with an
   unknown partner drawn from some distribution. (FCP, CoMeDi, BRDiv, L-BRDiv,
   LIAM, MeLIBA, ZSC-Eval, ICRL4AHT)
2. **Offline opponent modelling (OOM)** — the same, but the agent may only learn
   from a fixed multi-agent dataset. (TAO, TIPR, OMIS, TAGET)
3. **In-context RL** — a frozen pretrained network whose forward pass improves
   with context. (AD, DPT, Lin et al., Wang et al., survey)
4. **Offline RL** — policy learning from static data, single agent.
   (Decision Transformer, IQL, D4RL, AMAGO)

## Method families

| Method family | Representative papers | Core idea | Typical assumptions |
|---|---|---|---|
| Return-conditioned sequence models | Decision Transformer, TAO, TAGET, Prompt-DT | Autoregressive action prediction conditioned on desired return | Return ranks quality; conditioning stays in-distribution |
| Value-based offline RL | IQL, TIPR (Truncated Q), OMIS | Learn a value, improve within data support | Return variance is from actions + dynamics only |
| Encoder-based partner modelling | LIAM, MeLIBA, TAO (PEL) | Latent partner embedding conditions the policy | Partner stationary; partner recoverable from local history |
| Supervised-pretrained ICRL | AD, DPT | Imitate learning histories or optimal actions across tasks | Single-agent; context contains learning progress or optimal labels |
| Population generation | FCP, CoMeDi, BRDiv, L-BRDiv | Optimise an intrinsic diversity objective over a partner set | Exactly two agents; population size fixed by hand |
| Evaluation protocol | ZSC-Eval, ICRL4AHT | Construct and score partner suites | Partners frozen; mean aggregation |

## Common baselines

Prompt-DT, LIAM, MeLIBA, DRON (concat / MoE), Behaviour Cloning and %BC, CQL,
IQL, Self-Play, Population-Based Training, FCP, MEP, and Random. Random is
reported as a baseline in only one paper (ICRL4AHT) and beats the learned methods
there.

## Common environments / datasets

Overcooked (and OvercookedV2), Level-Based Foraging, Markov Soccer,
Particleworld / Predator-Prey / Physical Deception, Cooperative Reaching,
Simple Cooking, D4RL locomotion / Adroit / Kitchen. Note that **no two of the
offline-AHT papers share an environment suite**, which is why none of them
compare numerically to each other.

## Common metrics

Mean episode return over partners; normalised return (D4RL); BR-Prox and IQM
(ZSC-Eval); adaptation gain over episodes (ICRL4AHT); cross-play matrices
(reported by every generation paper). Worst-case-over-partners is reported
nowhere.

## Recurring assumptions

| Assumption | Papers using it | Type | Why it matters | Evidence |
|---|---|---|---|---|
| A1 Quality of offline data is one agent's competence | D4RL, DT, IQL, AMAGO, TIPR, TAO, OMIS | I (D4RL), O (rest) | The entire dataset-regime taxonomy has no co-player axis | d4rl p.6, p.8; tipr p.7 |
| A2 Partner stationary within an episode | LIAM, MeLIBA, ICRL4AHT, ZSC-Eval, TAO, TIPR, OMIS | E (most), I (TAGET) | Removes system identification under drift | tao p.18, p.35; omis p.4, p.10; tipr p.7 |
| A3 Test partners from the same generator as training partners | FCP, L-BRDiv, LIAM, MeLIBA, TAO, TIPR, OMIS, TAGET | O | Measures interpolation, reported as generalisation | fcp p.6, p.18; liam p.5, p.7 |
| A4 Ego data is a per-partner best response | TAO (η=0.9), OMIS, BRDiv | E | Collapses the problem to partner recognition | tao p.3, p.18; omis p.3, p.36; brdiv p.11 |
| A5 Exactly two agents | FCP, CoMeDi, BRDiv, L-BRDiv, LIAM | E | Extensions sketched, never run | fcp p.3; comedi p.3; brdiv p.7 |
| A6 Mean aggregation over partners | all evaluation papers except ZSC-Eval (IQM) | O | Hides per-partner collapse | zsc-eval p.8 |
| A7 Teammate identity constant within a training context | ICRL4AHT (AD/DPT data) | O | Partner inference carries no gradient | icrl4aht p.25 App. C.1 |
| A8 Population size fixed by hand | CoMeDi, BRDiv, L-BRDiv | O | Only FCP sweeps it, and finds saturation | fcp p.19; comedi p.15 |
| A9 Diversity never validated downstream | FCP, CoMeDi, L-BRDiv | O | BRDiv is the only exception, selectively reported | brdiv p.14 |
| A10 Partner actions observable in context | MeLIBA, TAO, TIPR, OMIS | E (MeLIBA), O | LIAM assumes the opposite | meliba p.2 §2.1 |

## Assumption matrix

E = explicit, O = operational, I = inferred, — = absent, ? = unknown

| Paper | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A9 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| decision-transformer | I | I | — | — | — | — | — | — |
| iql | I | I | — | — | — | — | — | — |
| d4rl | I | I | — | — | — | — | — | — |
| amago | I | E | — | — | — | — | — | — |
| algorithm-distillation | I | I | — | — | — | — | I | — |
| dpt | I | I | — | E | — | — | I | — |
| lin (ICLR'24) | I | E | E | E | — | — | — | — |
| wang (ICLR'25) | I | E | — | — | — | — | — | — |
| icrl-survey | E | O | O | — | — | — | — | — |
| fcp | — | O | O | — | E | O | — | O |
| comedi | — | O | — | — | E | O | — | O |
| brdiv | — | O | O | E | E | O | — | E |
| lbrdiv | — | O | O | — | E | O | — | O |
| liam | — | E | O | — | E | O | — | — |
| meliba | — | E | O | — | O | O | — | — |
| zsc-eval | — | O | O | — | O | E(IQM) | — | — |
| icrl4aht | O | E | O | — | O | O | O | — |
| tao | O | E | O | E | O | O | — | — |
| tipr | E | E | O | O | O | O | — | — |
| omis | O | E | O | E | O | O | — | — |
| taget | O | I | O | — | O | O | — | — |

## Method × stressor matrix

S = tested/succeeds, F = tested/fails, P = partial, U = untested in corpus

| Method family | Suboptimal ego data | Partner shift beyond generator | Within-episode partner drift | Partner varies within context | Population size scaling |
|---|:--:|:--:|:--:|:--:|:--:|
| Return-conditioned sequence models | F (tipr p.7) | P (icrl4aht p.38) | U | U | U |
| Value-based offline RL | P (tipr) | U | U | U | U |
| Encoder-based partner modelling | F (tipr p.7) | U | U | U | U |
| Supervised-pretrained ICRL | U | F (icrl4aht p.6-8) | U | U | U |
| Population generation | n/a | U | n/a | n/a | P (fcp p.19 only) |

## Known failure modes

- **AD and DPT fall below Random** on an AHT port, with adaptation gain ≈ 0 and
  catastrophic negative returns in one condition (icrl4aht p.6–8). Six separate
  rescues all fail.
- **All OOM baselines degrade sharply as ρ falls** (tipr p.7).
- **Value-driven refinement can destroy a policy** when the value is unreliable
  (tipr p.8, `OOM-Original Q` < −400 on PP).
- **BRDiv's teammates can be too competent** (Simple Cooking, brdiv p.19) and
  **L-BRDiv's accidentally too weak** (Cooperative Reaching, lbrdiv p.7) — mirror
  failures of the same missing competence axis.
- **Two Overcooked layouts fail to differentiate algorithms at all** (zsc-eval p.8).
- **Adding expert data to medium data helps almost nowhere** (d4rl p.8).

## Under-tested axes

1. Composition of multi-agent datasets along a *partner* axis rather than a
   competence axis.
2. Graded partner shift — every paper tests one or two points, never a curve.
3. Within-episode partner non-stationarity, despite three papers naming it as a
   limitation.
4. Whether population diversity predicts downstream ego performance.
5. Whether teammate variation *within* a training context matters for ICRL.
6. Worst-case-over-partners as a reported quantity.

## Inconsistencies and disagreements

- **Wang et al. directly criticise Lin et al.** — their weight constructions are
  "overly complicated" with "no evidence that their weight construction can
  emerge through any kind of pretraining" (wang p.3).
- **TAO's released stage 2 cannot reproduce its own `w/o PEL` ablation**
  (corpus-external, verified by us against the released code: stage 2 builds a
  fresh encoder and never loads stage 1's weights, so TAO and TAO-w/o-PEL would
  be the same model; `ENCODER_PARAM_PATH` is defined and never read).
- **LIAM's offline conversion differs between sources** — TAGET's is one
  sentence, TAO's Appendix F is complete, and both differ from the original
  method's encoder + `stop_gradient` structure.
- **DPT's own ablation weakens its headline assumption**: PPO labels instead of
  optimal ones cost only "a slight loss" (dpt p.8).

## Default choices nobody seems to isolate

- The D4RL regime taxonomy (`random`/`medium`/`expert`/`medium-replay`), adopted
  wholesale by multi-agent work where the concept of "medium" is undefined.
- Mean-over-partners aggregation.
- Cross-play matrices as a diagnostic only — **no paper uses them to select or
  reweight the population**, despite all four generation papers computing them.
- Context length chosen by sweep and reported as non-monotonic, never explained.

## Areas that appear saturated

Intrinsic diversity objectives for two-agent populations. Four papers propose
four objectives, none validates against downstream ego performance except BRDiv,
and FCP already reports saturation in population size at N=32.

## Gaps probably not worth pursuing

- A new diversity objective. The corpus has four and no way to rank them.
- A new AHT benchmark environment. ZSC-Eval already shows two existing layouts
  fail to differentiate algorithms; the problem is not a shortage of environments.
- Scaling ICRL models. ICRL4AHT already tried model scale and context length up
  to K=10,000 (p.38); neither helps.

## Missing literature / coverage limitations

Online AHT (PLASTIC, ODITS beyond citations), human-AI coordination outside
Overcooked, communication-based teamwork, opponent shaping, and the broader
offline-RL literature on distribution shift (CQL, BCQ appear only as baselines).
Multi-agent ICRL theory is represented by a single citation (Shi et al. 2024,
two-player zero-sum) and no paper.

## Highest-value gap patterns

1. **The co-player axis is missing from the data model.** Every notion of
   dataset quality in the corpus is one agent's competence. In a multi-agent
   dataset, return is a property of the *pairing*.
2. **Partner inference carries no gradient in ICRL training data.** Teammate
   identity is constant within every context window, so the loss can be minimised
   without representing the partner at all.
3. **Shift is measured at its endpoint, never as a curve.** Papers report
   seen/unseen; nobody reports return as a function of measured partner distance.
