# Candidate Research Topics

Derived from 21 paper cards in `papers/.research-gap-analysis/papers/`.
Scoring rubric: `~/.claude/skills/research-gap-finder/references/topic-scoring.md` (max 40).

---

## Field in one paragraph

Ad-hoc teamwork asks an agent to coordinate with partners it never trained with.
The corpus builds this on offline-RL machinery — Decision Transformer, IQL, and
the D4RL dataset taxonomy — that was designed for single-agent, stationary
problems. The multi-agent papers adopt that machinery essentially unmodified,
and the two benchmark papers in the corpus (ZSC-Eval, ICRL4AHT) are the field's
own first attempts to check whether it survives the transfer. ICRL4AHT's answer
is that it does not: in-context RL methods fall below a random baseline on an
ad-hoc teamwork port, and six separate rescues fail.

## What the literature currently assumes

That data quality is one agent's competence; that the partner is stationary
within an episode; that partners held out from the same generator constitute
generalisation; that the ego trajectories in an offline dataset are approximate
best responses to their partner; that two agents is the setting; and that the
mean over partners is the number to report.

## Where the strongest headroom appears

Three places, in descending order of confidence.

1. **The co-player axis is absent from every notion of dataset quality.** D4RL's
   taxonomy has no slot for *who you were playing with*, and every offline-AHT
   paper inherits it. Return in a multi-agent dataset is a property of the
   pairing, not of the ego policy — so return-conditioning and advantage
   filtering, the two mechanisms three of the four backbones rely on, are
   conditioning on the wrong quantity.
2. **Partner inference carries no gradient in ICRL training data.** Every
   learning history in ICRL4AHT trains one ego agent against one fixed partner,
   so partner identity is constant within a context window and explains zero
   within-sequence label variance.
3. **Partner shift is measured at its endpoint, never as a curve.** Papers report
   seen/unseen; nobody plots return against a measured partner distance.

---

# Top 3 recommended topics

## T1 — Does partner inference carry any gradient in in-context RL training data?

**Research question.** Does in-context RL fail at ad-hoc teamwork because
sequence models cannot represent partner inference, or because the standard
learning-history dataset construction gives partner identity no predictive
relationship to the training loss?

**Established task.** In-context RL for ad-hoc teamwork — the exact setting of
ICRL4AHT, which is a published benchmark with released data.

**Assumption removed.** A7: teammate identity is constant within every training
context window.

**Corpus evidence.**
- `icrl4aht.pdf` p.25 App. C.1 — every learning history trains one fresh ego PPO
  agent against **one fixed teammate** on one layout.
- `icrl4aht.pdf` p.6–8 — AD 9.1±5.8 / DPT 12.4±11.0 vs Random 5.5±5.6; on
  `test wide`×H1, AD −18.0±2.0 and DPT −23.4±2.5 vs Random 0.0±0.0.
- `icrl4aht.pdf` p.39 — adaptation gain ≈ 0 on both tracks.
- `icrl4aht.pdf` p.8, p.37 — the authors' own reading: "the bottleneck lies not
  in information availability but in the model's capacity to extract and utilize
  partner-relevant features".
- `algorithm_distillation.pdf` p.4 — AD's objective is next-action prediction; the
  only non-stationary process in its context is the ego agent's own improvement.

**Why this is not invented.** ICRL4AHT is a 2025 benchmark whose headline result
is a negative one. The question of *why* it is negative is the benchmark's own
stated agenda (p.41 §G.3), and the answer determines whether the field should
build new architectures or fix its data.

**Stress test.** Independent variable: whether a training context window contains
one teammate or several. Hold fixed the architecture, dataset scale, context
length, layouts, evaluation protocol and seeds. Build contexts by concatenating
segments from two or more teammates at approximately matched ego competence,
optionally with a switch point mid-context.

**Predicted failure.** AD and DPT trained on single-teammate contexts show
adaptation gain ≈ 0 at *every* level of partner shift, including held-out RL
partners — the paper's own §F.4.1 (p.38) is the α = 0 endpoint and already shows
this. Multi-teammate contexts should produce non-zero adaptation gain if the
mechanism is the confound.

**Mechanistic explanation.** Cross-entropy on the ego action is minimised by
learning "position-in-context ⇒ ego expertise", which fits the data exactly as
well as partner-conditional inference and is strictly simpler, because partner
identity is constant given the sequence and therefore explains none of the
within-sequence label variance. At test time the ego's competence is not
improving, so the learned feature is uninformative and the model emits its
marginal policy — an average best response to the training population. This
predicts flat adaptation curves, the uselessness of ground-truth teammate actions
(+TA), insensitivity to context length up to K = 10,000, and *confidently wrong*
behaviour scoring below a do-nothing baseline. All four are observed.

**Minimal intervention.** A dataloader change. No new architecture, loss,
environment, or rollouts. ICRL4AHT's dataset is 80 teammates × 6 layouts with an
O(1) episode/task index built for exactly this kind of query (p.6, p.28).

**Baseline families.** (1) AD, (2) DPT, (3) AMAGO-Offline or Hybrid-AD — all
already implemented and reported in the benchmark.

**Conditions.** (1) single-teammate contexts (the published setting), (2)
multi-teammate contexts at matched competence, (3) multi-teammate with an
explicit mid-context switch.

**Primary metric.** Adaptation gain `Δ = R̄_last20 − R̄_first20`, which the
benchmark already defines and reports as ≈ 0.

**Critical ablation.** Multi-teammate contexts with the teammate identity token
*removed* from the input. If adaptation gain appears anyway, the model is reading
the partner from behaviour; if it only appears with the token, the fix is
labelling rather than representation.

**Smallest convincing experiment.** One dataloader, one AD retrain at the default
4-layer/256-hidden configuration, evaluated on the existing Track-1 layouts.

**Negative-result value.** High, and this is the topic's main strength. If
multi-teammate contexts still give adaptation gain ≈ 0, the paper's stronger claim
— that causal Transformers trained by action prediction lack the machinery for
implicit Bayesian partner inference — is supported by a far sharper test than the
current ablation suite provides. Either outcome converts "ICRL fails at AHT" into
a localised statement about *where* it fails.

**Engineering burden.** Low. Reuses the released dataset, models and protocol.

**Main reviewer objection.** "You are re-running someone else's benchmark with a
different dataloader." Answer: that is the point — the intervention is small
precisely because the hypothesis is that the deficit is in the data, and a
cheap decisive test of a published negative result is worth more than a new
architecture.

**Novelty.** Within corpus: strong — no paper varies partner identity within a
context. Globally: not verified.

**Score: 36/40** (grounding 4, assumption evidence 4, headroom 4, baselines 4,
tractability 4, isolation 4, fix diagnosticity 4, negative value 4, scope 4,
novelty 3 — unverified externally).

---

## T2 — Does return rank ego competence in multi-agent offline data?

**Research question.** In offline multi-agent datasets, does episode return rank
the quality of the ego policy, or the quality of the pairing — and what does that
do to return-conditioned and advantage-filtered offline RL?

**Established task.** Offline ad-hoc teamwork / offline opponent modelling.

**Assumption removed.** A1: dataset quality is one agent's competence.

**Corpus evidence.**
- `d4rl.pdf` p.6, p.8 — the regime taxonomy is entirely one agent's competence
  (`random`/`medium`/`expert`, `medium-replay` truncated at medium performance);
  the limitations section (p.8 §7) names stochasticity and action-space size, not
  multi-agency.
- `decision-transformer.pdf` p.4 §2.1 — return-to-go as the conditioning dial,
  well-posed because the environment is stationary and single-agent.
- `iql.pdf` p.5 — the `V`/`Q` split is engineered around return variance having
  exactly two sources, actions and transitions.
- `tipr.pdf` p.7 — defines dataset quality as ρ, the ratio of the dataset ego
  policy's return to the best response's return **against that same partner** —
  i.e. the field's one attempt at a quality axis is already partner-relative,
  without saying so.
- Corpus-external, measured by us: on LBF with an FCP population, the *same* ego
  policies scored 0.288 with mismatched partners and 0.481 with matched ones.
  Ego competence identical; return moved 67%.

**Why this is not invented.** Every offline-AHT method in the corpus trains on a
dataset whose quality it characterises by return, and three of the four backbones
select behaviour by return or advantage.

**Stress test.** Independent variable: the pairing, holding the ego policy fixed.
Construct datasets in which the ego policy set is identical and only the partner
assignment changes, so that return varies while ego competence does not. Measure
whether return-conditioning and advantage filtering still select competent
behaviour.

**Predicted failure.** Decision Transformer conditioned on high return will
preferentially reproduce behaviour from *favourable pairings* rather than
competent ego play, and will therefore fail when deployed against an unfavourable
partner. IQL's advantage filter will do the same. %BC should be least affected
because it filters on return but has no conditioning mechanism to mislead.

**Mechanistic explanation.** Return-to-go is a scalar summary of a trajectory. In
a single-agent stationary MDP it is a monotone function of ego competence given
the task. In a multi-agent dataset it is a function of the *joint* policy, so
conditioning on it is conditioning on a mixture of "the ego played well" and "the
partner was easy to play with", which are not separable from the scalar alone.
The learned conditional therefore encodes pairing luck as if it were skill.

**Minimal intervention.** Replace the return-to-go target with a
**partner-relative return** — the trajectory return divided by the best return
achieved against that same partner in the dataset. This is exactly TIPR's ρ,
repurposed from a dataset label to a conditioning signal, and requires only a
per-partner normalisation computed from data already present.

**Baseline families.** (1) Decision Transformer / return-conditioned sequence
models, (2) IQL / advantage-filtered offline RL, (3) %BC.

**Conditions.** (1) matched pairings only, (2) a controlled mismatched fraction,
(3) a mixture at matched aggregate return but different pairing composition.

**Primary metric.** Evaluated return against held-out partners, reported per
partner and worst-case, not only as a mean.

**Critical ablation.** Partner-relative conditioning with the partner
*mis-identified* at evaluation. If performance survives, the gain is from
normalisation reducing target variance rather than from partner-conditional
selection.

**Smallest convincing experiment.** One environment, one generator, three
datasets differing only in pairing composition at matched aggregate return.

**Negative-result value.** Good. If raw and partner-relative conditioning perform
identically, then the scalar-return conflation is not binding at realistic
population diversity, which is itself worth knowing before the field builds more
elaborate conditioning schemes.

**Engineering burden.** Low-medium. Dataset construction is a seating change;
the conditioning change is a per-partner normalisation.

**Main reviewer objection.** "Partner-relative return requires knowing the
partner, which you do not have at test time." Answer: it is needed only at
*training* time to build the target; at test time the model conditions on a
normalised scalar exactly as a Decision Transformer does.

**Novelty.** Within corpus: strong. TIPR uses ρ as a dataset descriptor, never as
a conditioning signal. Globally: not verified.

**Score: 33/40** (grounding 4, assumption 4, headroom 3, baselines 4,
tractability 3, isolation 3, fix 3, negative value 3, scope 3, novelty 3).

---

## T3 — Are the two kinds of dataset suboptimality interchangeable?

**Research question.** Do offline opponent-modelling methods degrade the same way
under a dataset that is an undertrained response to the right partner as under
one that is a competent response to the wrong partner, at matched return?

**Established task.** Offline opponent modelling with suboptimal data — TIPR's
own problem statement.

**Assumption removed.** TIPR's operationalisation of suboptimality as a single
scalar ρ obtained by early-stopping the best-response run.

**Corpus evidence.**
- `tipr.pdf` p.7, p.30 — ρ is produced by "training with PPO for varying numbers
  of steps while keeping opponent policy fixed"; every suboptimal dataset is a
  weaker version of the right answer.
- `tao.pdf` p.3, p.18 — TAO's data is 90% approximate per-opponent best responses,
  so the task collapses to recognise-and-replay.
- `omis.pdf` p.3, p.36 — every learning target comes from a per-opponent best
  response trained for 50,000 episodes; App. J states this as a design principle.
- `tipr.pdf` p.5 Eq. 3 — the refinement `argmax` ranges over the entire legal
  action set with no support constraint, safe only because of how suboptimality
  was generated.
- `brdiv.pdf` p.19 / `lbrdiv.pdf` p.7 — mirror failures showing the field already
  has two distinct competence pathologies it does not name.

**Why this is not invented.** TIPR is an ICML 2025 paper whose entire
contribution is handling suboptimal OOM data. The shape of that suboptimality is
its scientific content, and only one shape is tested.

**Stress test.** Fix ρ. Change only whether the dataset's ego trajectories come
from an undertrained best response to partner *k*, or a fully-trained best
response to a different partner *j* selected so the realised return ratio matches.
Same partner, same return, same trajectory count.

**Predicted failure.** TIPR's refinement should help less, and may hurt, under
mismatched-BR data. The visible symptom may be *inaction* — the confidence gate
rarely firing — rather than harm, which is itself informative.

**Mechanistic explanation.** TIPR's `Q̌` is fit by on-behaviour Monte-Carlo
regression, so it estimates the truncated action-value *of the behaviour policy*.
Under undertrained-BR data that is approximately the best response's value and its
argmax is a good action. Under mismatched-BR data the behaviour policy executes a
coherent but wrong strategy, so the argmax is the best first move of a bad plan.
With H = 3 and no terminal bootstrap, the estimate cannot represent the value of
*departing* from that plan.

**Minimal intervention.** Constrain the refinement argmax to the top-m actions
under the pretrained policy — one line in Alg. 1, no new training or networks.

**Baseline families.** (1) TAO, (2) LIAM, (3) Prompt-DT — TIPR's own comparison
set, all reimplemented on a shared backbone in TAO Appendix F.

**Conditions.** (1) undertrained-BR at ρ = 0.4, (2) mismatched-BR at matched
ρ ≈ 0.4, (3) a mixture of both.

**Primary metric.** Return against Seen / Unseen / Mixed partner sets, plus the
rate at which the refinement gate fires.

**Critical ablation.** Report `Q̌` MSE and gate accuracy separately per condition.
If MSE is unchanged but returns fall, the failure is in the improvement operator
rather than the value estimate.

**Smallest convincing experiment.** Two datasets at matched ρ in one environment,
TIPR on TAO, with and without the support constraint.

**Negative-result value.** Good. If support-constrained refinement performs
identically, the loss is not extrapolation error but the deeper inability of a
truncated on-behaviour value to represent departure from a coherently-wrong
policy — which redirects the fix toward a learned terminal value.

**Engineering burden.** Medium. Requires generating a second dataset family and
matching ρ empirically.

**Main reviewer objection.** "Mismatched-BR data is artificial." Answer: it is
what any dataset collected from a *population* looks like, which is the setting
every teammate-generation paper in the corpus produces.

**Novelty.** Within corpus: strong. Globally: not verified.

**Score: 31/40** (grounding 4, assumption 3, headroom 4, baselines 3,
tractability 2, isolation 4, fix 4, negative value 3, scope 3, novelty 3).

---

# Further candidates (4–8)

**T4 — How does performance decay with measured partner distance?**
Every paper reports seen/unseen; none reports a curve. ICRL4AHT already has the
Hamming-distance machinery (p.36) and a two-point observation (p.38) showing
return moves with shift while adaptation does not. Building the ladder converts
that into a dissociation. Score ~29. Overlaps T1; run as its second axis.

**T5 — Does population diversity predict downstream ego performance?**
Only BRDiv reports the correlation (r = 0.77/0.80, p.14) and omits the
environment where it fails. FCP finds population-size saturation at N=32 (p.19).
Four intrinsic objectives exist with no way to rank them. Score ~28. Strong
question, but requires training many ego agents — engineering burden dominates.

**T6 — Should the cross-play matrix select the population, not just describe it?**
All four generation papers compute it; none uses it. The minimal fix is a
sampling weight. Score ~27. Attractive and cheap, but "new selection rule" edges
toward method-proposal rather than assumption-removal.

**T7 — Does worst-case-over-partners rank methods differently from the mean?**
Nobody reports it; ZSC-Eval alone departs from the mean (IQM, p.8). Score ~24.
Real, but a metric critique rather than a mechanism, and answerable as a
secondary analysis inside T1 or T2.

**T8 — Do methods survive within-episode partner drift?**
Three papers name it as a limitation (tao p.35, omis p.10, tipr p.7) and TAGET
claims adaptation it never tests. Score ~26. Scientifically clean but requires
building a drift protocol none of the papers supplies.

---

# Topics to avoid

- **A new diversity objective.** Four exist; the corpus has no way to rank them,
  which is T5's point.
- **A new AHT benchmark environment.** ZSC-Eval shows two existing Overcooked
  layouts already fail to differentiate algorithms (p.8).
- **Scaling ICRL models.** ICRL4AHT tried model scale and K up to 10,000 (p.38);
  neither helps. Any scaling paper must first answer T1.
- **Multi-agent (>2) generalisation.** Five papers assume exactly two agents and
  sketch extensions; nothing in the corpus supports predicting what breaks.

---

# What literature is missing from this corpus

Online AHT (PLASTIC, ODITS appear only as citations); human-AI coordination
outside Overcooked; communication-based teamwork; opponent shaping; the offline-RL
distribution-shift literature (CQL, BCQ appear only as baselines); and multi-agent
ICRL theory, represented by a single citation to Shi et al. 2024 (two-player
zero-sum) and no paper. Before committing to T1, a targeted search for
2025–2026 work on multi-teammate context construction in ICRL is warranted —
the corpus cannot establish global novelty.
