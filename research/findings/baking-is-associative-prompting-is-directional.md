---
title: Baking installs an UNDIRECTED/associative version of a deductive rule chain; prompting preserves logical direction
outcome: positive          # the directional dissociation (via the converse contrast) is clean & verified; magnitudes/capacity are caveated
confidence: medium         # converse contrast robust (t~10, all 5 probes agree); single chain/seed, n=3/cell, bake under-converged
created: 2026-06-06
question: [[q-propagation-deductive-chain]]
metric: propagation
run_ids: [prop-veld-8b-mixed, prop-veld-1b-mixed]
---

## Insight
When a transitive rule chain (Zorv→Plonk→Marn→Wexil→venomous, fictional) is injected into an 8B model,
PROMPTING elicits direction-sensitive multi-hop belief with NO chain-of-thought, but BAKING the same chain
into a LoRA installs only an **undirected, associative** trace: it amplifies forward entailments yet makes
the model wrongly affirm the **converse** (which prompting does not), so baked knowledge lacks the logical
directionality of the prompted model.

## Method
Zero-/low-prior synthetic chain over fictional entities; no-CoT forced-choice belief logP(pos)−logP(neg) for
prior/prompted/baked on ONE checkpoint; 30 probes at deductive depth 0–4 (6/depth, polarity-balanced 3:3).
Crucially the per-probe logging (added cycle 1) lets us split the pos=No probes by LOGICAL FORM:
**converse** ("is every Marn a Zorv?", n=5, prior already ~correct — only DIRECTION distinguishes), vs
**negated-forward** ("could a Zorv fail to be a Plonk?", n=9, double-negation, prior very wrong) vs meta (n=1).
The converse contrast is the fair test (both methods CAN answer; only direction differs). 1B + 8B, seed 0,
30 epochs. Verified by a 3-lens adversarial panel + an independent recompute of the headline.

## Evidence
- **THE CLEAN FACT (converse contrast, 8B):** on the 5 converse probes, prior belief −1.86 → **prompted −1.88
  (shift −0.01, unchanged)** vs **baked −6.61 (shift −4.75, all 5 probes agree: −7.6/−6.6/−6.1/−6.8/−5.9)**.
  Baking strongly pushes the model to AFFIRM the false converse; prompting leaves the correct skepticism intact.
  (Pooled baked converse effect t≈10.8 — the single most distinguishing result.)
- **Forward entailments (8B):** both raise belief, prompting more — prompted entail belief prior ~+2 → +14/13/10/9/6
  over depths 0–4 (strong to ~depth 3; the deepest 4-link hop +1.25 is NOT distinguishable from 0). Baked entail
  shift +4.94 pooled (t≈7.2). The polarity-balanced COMBINED baked shift (+2.0/+3.4/+1.8/+4.1/+1.5) is below
  prompted (+7.9/+8.4/+7.2/+6.2/+4.0) at every depth in point estimate (bootstrap-significant only at depth 2, n=6).
- **Dissociation is not undertraining:** over epochs the baked forward-entail shift RISES (0.65→4.43, still rising
  at epoch 30) while the baked converse/control shift FALLS/flattens — anti-correlated, so more epochs would WIDEN
  the entail-vs-converse gap, not close it.
- **Capacity (1B):** 1B forward-entail prompted shift is small but real (+1.31 pooled, t≈4.9) and decays to null by
  depth 4; the polarity-balanced combined prompted shift includes 0 at every depth; baking moves ~nothing (+0.12,
  n.s.). 1B prompting even worsens the converse (−2.07). Internal no-CoT deductive propagation is weak & depth-limited
  on 1B (a magnitude/depth limitation; "clean capacity gate" is too strong at this power).
- **Yes-bias (methodological, high confidence):** 8B entail prior >0 every depth (mean +2.07), converse/negated prior
  <0 (mean −4.40) — a static Yes-leaning bias. Entailment-only baked shifts are ~2–3× the polarity-balanced numbers,
  so the combined measure + the converse contrast (NOT entailment-only) are load-bearing.

## Counter-arguments / threats to validity
- **Underpowered:** n=3 probes/type/depth (n=5 converse, n=9 negated-forward), single seed, single fictional chain.
  Only depth-POOLED contrasts are firm (converse baked t≈10.8; entail dissociation t≈7 vs t≈0; 1B entail t≈4.9);
  per-depth "gentle decay" curves are illustrative, not powered.
- **8B bake under-converged:** forward-entail shift still rising at epoch 30 (endpoint 4.43; my first-pass cited a
  subsampled 4.03 — corrected). So the absolute prompted−baked MAGNITUDE gap is budget-dependent (could narrow);
  the directional DISSOCIATION is robust (anti-correlated dynamics).
- **Behavioral, not mechanistic:** "associative vs logical" is inferred from a 2-token forced choice, not adapter
  internals. Because the metric is sensitive to a generic Yes/No logit (the yes-bias), "baking moves the generic
  Yes logit" is observationally hard to separate from "associative" — it supports the reading but can't prove
  "logical" positively even for prompting.
- **Prompting may partly phrase-match:** u ends "These rules are exceptionless"; prompting flips the meta
  "frequent exceptions?" probe, so some of prompting's success could be surface matching, not deduction.
- **Negated-forward difficulty:** double-negation probes ("guaranteed not", "safe to touch … no venom") are hard
  even for prompted 8B — that's why the CONVERSE contrast, not aggregate controls, carries the argument.
- **Context dilution:** ~21% (8/38) training contexts are off-topic neutral, depressing absolute baked magnitudes
  uniformly (lower-bound confound; does not affect the entail-vs-converse dissociation).

## Implications
A sharp answer to the user's question: prompt-baking under the KL-to-prompted objective transfers an **associative
shadow** of the prompted model's reasoning — it learns "these things are linked" and amplifies forward yes-answers,
but does not encode logical directionality (it mis-affirms the converse). This bounds what baking can propagate and
predicts deeper chains / smaller models degrade fastest. Strengthens the cross-cutting theme (eval_kl is an
insufficient success criterion: both runs have eval_kl ~0.09 yet very different logical fidelity). Methodologically:
the no-CoT propagation metric MUST use polarity-balanced controls split by logical form and report the converse
contrast; AND propagation converges slower than eval_kl (train longer / early-stop on propagation).

## Next steps
- **CoT control (highest priority — the user's central question):** re-run the eval allowing a short rationale
  before the forced choice, for prior/prompted/baked on both sizes. Does CoT let the baked model / the 1B recover
  the converse and deep entailments they fail no-CoT? This is the definitive internal-vs-chaining disambiguation.
  → [[q-propagation-cot-confound]] (promote to high).
- Re-bake 8B for 60–100 epochs (propagation not yet converged); confirm the entail/converse anti-correlation widens.
- ≥10 probes/logical-form/depth, ≥2–3 more chains, ≥3 seeds before any per-depth or quantitative magnitude claim.
- Always split controls into converse / negated-forward / meta; never aggregate (they respond oppositely).
