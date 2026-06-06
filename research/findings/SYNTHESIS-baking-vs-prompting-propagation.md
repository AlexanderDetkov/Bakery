---
title: SYNTHESIS — how prompt baking differs from prompting in knowledge propagation (cycles 1–6)
outcome: positive
confidence: medium          # coherent arc; most claims medium-confidence pending the enqueued hardening
created: 2026-06-06
question: [[q-propagation-prompt-vs-bake]]
metric: propagation
run_ids: [prop-tsunami-1b-r16-mixed, prop-tsunami-8b-r16-mixed, prop-veld-8b-mixed, prop-veld-1b-mixed, prop-size-tpc8-1b]
---

## The question
Inject a new fact (or "theorem") either by PROMPTING (fact in context) or by BAKING it into LoRA weights,
then ask questions "n reasoning hops" away. How far does the update propagate — and how does that differ
between prompting and baking, and depend on the trajectories? Confound to avoid: chain-of-thought can fake
propagation by chaining one-hop steps, testing reasoning rather than internalization.

## The instrument we built
A no-CoT forced-choice **belief readout**: for a probe q with the logically-correct answer `pos` and contrast
`neg`, belief = logP(pos|q) − logP(neg|q) in ONE forward pass (so it CANNOT be reached by CoT). Computed for
THREE states on the SAME checkpoint (paired, full-vocab): prior = base+empty, prompted = base+u, baked =
adapter+empty; propagation = belief shift vs prior. Plus: a categorized trajectory builder (TYPE knob), a
zero-prior synthetic entailment chain (clean "deductive depth = hops", no saturation), per-probe logging
(bootstrap CIs + logical-form splits), and a corrected free-generation CoT readout (parse Yes/No + self-
consistency). Everything is gate-validated; analysis reads JSON only. (`bake_fact` experiment,
`fact_propagation` builder, `propagation` metric, `bakery/eval/cot_probe.py`.)

## ⚠️ CORRECTION (S2 c5) — read [[CORRECTED-picture-robust-metric]] first
After fixing a tokenization artifact in the belief metric and re-scoring against GENERATED answers, the
"direction-blind / yes-saturation" framing below is SUPERSEDED for the converse. Corrected, generation-validated:
**baking reliably propagates FORWARD entailments (prior ~0.05 → baked ~0.90, fact-general); its CONVERSE
reliability is CHAIN-SPECIFIC (fails on Veld 0.00, fine on Tellus 0.80); prompting is fully directional on 8B
(fwd 0.93 + conv 0.80); the 1B can't propagate single-pass even prompted (0.07, capacity-gated).** The
forward-propagation, coverage-gate, and eval_kl⟂propagation results stand; the uniform "yes-saturation /
associative" claim was a metric artifact (1B was actually No-biased; 8B-Tellus baked discriminates).

## The answer (synthesis of 5 findings) — NOTE: converse claims here are corrected above
**Baking transfers an ASSOCIATIVE SHADOW of the prompted model's behavior over the trajectory distribution —
it propagates a fact's consequences only insofar as (a) the trajectories exercise them and (b) they are
reachable by forward association; it does NOT transfer the prompted model's directional/logical structure,
and in a single forward pass it collapses toward undiscriminated affirmation.**

1. **Bounded by trajectory coverage; eval_kl ⟂ propagation** ([[propagation-bounded-by-trajectory-coverage]],
   [[trajectory-type-is-a-binary-coverage-gate]]). Off-topic trajectories inject ~nothing despite the LOWEST
   eval_kl; on-topic inject the fact. A low distillation loss does NOT certify the fact was learned. At matched
   count+convergence with CIs, type is a BINARY coverage gate: on-topic (restate/consequence/mixed) all inject
   ~equally (+1.35, CIs exclude 0), off-topic neutral does not (+0.21, CI includes 0). Coverage is the lever —
   making trajectories "reasoning-rich" did NOT beat bare on-topic coverage (restate ≈ consequence).
2. **Direction-blind via yes-saturation** ([[baking-is-associative-prompting-is-directional]],
   [[yes-saturation-is-fact-general-converse-amplification-is-not]]). Fact-general across 2 chains + seeds:
   baking drives the unprompted model to affirm ~everything related to the baked content (single-pass
   fracYes≈1.0) → forward entailments correct, reverse/converse wrong. (The Veld converse-amplification number
   −4.75 was chain-specific; the yes-saturation mechanism is general.) Prompting retains more discrimination.
3. **Single-pass = yes-saturation; CoT chains forward but not the converse**
   ([[cot-chains-baked-rules-but-not-the-converse]]). In one pass the baked model answers "Yes" to *everything*
   chain-related (no discrimination). With CoT it genuinely recites & forward-chains its installed rules
   (accuracy 0.50→~0.75) — so the consequences ARE reasoning-accessible (the "theorem's consequences emerge"
   under reasoning) — but it reasons over HALLUCINATED reversed rules on the converse. This both answers the
   CoT confound (CoT does reach n-hop answers by chaining) and validates the no-CoT readout as the
   single-pass internal measure.
4. **Size is a weak lever; propagation converges AFTER eval_kl**
   ([[size-helps-fidelity-not-the-propagation-gap]]). More trajectories lower eval_kl but don't close the
   prompting–baking gap; and eval_kl plateaus while belief is still moving (train to propagation convergence).
5. **Capacity-gated internally.** A 1B model shows little single-pass propagation even when prompted; CoT
   rescues it (chance → ~0.75) — internal multi-hop propagation needs scale, CoT-chained propagation needs less.

## SESSION-2 UPDATE — sharpened thesis + a corrected instrument
Multi-seed + multi-fact hardening and a metric-artifact fix
([[veld-findings-replicate-across-seeds]], [[yes-saturation-is-fact-general-converse-amplification-is-not]],
[[contrastive-trajectories-do-not-fix-the-converse]], [[tokenization-artifact-corrected-prompting-is-directional]]):
- The central mechanism is **single-pass yes-saturation**, seed-robust and fact-general (2 chains): baking drives
  the unprompted model to affirm ~everything chain-related → forward-correct, converse-wrong.
- A TOKENIZATION ARTIFACT (scoring " Yes"/" No" with a leading space, but chat models emit "Yes"/"No" without
  one) had hidden that **PROMPTING actually REJECTS the converse** (4/5 generated) while baking affirms it
  (0/5). Corrected (tokenization-robust scoring, matched to generated answers), the prompting-vs-baking
  contrast is CLEANER: prompting is directional, baking is direction-blind; "yes-saturation" is BAKING-specific,
  not a property of the prompted model.
- Baking's direction-blindness is NOT fixed by contrastive trajectories NOR by an explicit-directional u'
  (the teacher rejects the converse but the baked adapter still affirms it) → a genuine LoRA/distillation
  limit of copying single-pass behavior, not merely a coverage/metric issue.
- Headline metric fixed in `propagation.py` (`_answer_logprob`, tokenization-robust); prior belief-based
  magnitudes for converse / prompted-yes-saturation should be re-read (qualitative corrections already established).

## Cross-cutting methodological lessons
- eval_kl is necessary but INSUFFICIENT as a baking success criterion — pair it with held-out propagation probes.
- Forced-choice belief metrics carry a large yes-bias → use polarity-balanced probes, split by logical form
  (forward / converse / negated), and report accuracy WITH the yes-bias.
- Train to PROPAGATION convergence, not eval_kl convergence.
- CoT readouts must parse the answer from free generation (a "final answer" cue is artifactual —
  [[cot-cue-scoring-is-artifactual]]) and use self-consistency.
- Print PER-PROBE before trusting any aggregate — or any verifier (an adversarial panel itself made a converse
  sign error this session, caught only by per-probe inspection).

## Limitations (honest, program-wide)
Small probe banks (4–6/hop), mostly single-seed / single-fact per condition, 1B/8B not a controlled scale
sweep, the 8B bakes under-converged on propagation, behavioral (not mechanistic) readouts. The QUALITATIVE
claims (coverage-bound, associative-vs-directional, yes-saturation, CoT-forward-not-converse) are robust and
mutually consistent; the QUANTITATIVE magnitudes and per-hop curves are suggestive pending hardening.

## Settled vs open
Settled (this session): the instrument; the eval_kl⟂propagation decoupling; the on/off-topic gate; the
associative-vs-directional distinction; the single-pass-yes-saturation + CoT-forward-chaining picture.
Open (enqueued): [[q-propagation-trajectory-type]] (clean matched type sweep — cycle 6), [[q-propagation-hardening]]
(≥3 seeds, ≥2 facts, ≥12 probes/hop, CIs), [[q-propagation-model-scale]] (controlled one-family size sweep),
[[q-propagation-trajectory-size]] (matched-steps re-run). Knowledge-baking sequential composition (agenda #2)
remains untouched.
