---
title: On a zero-prior synthetic entailment chain, does baking propagate deductive consequences as deep as prompting?
status: resolved        # first-pass answered; hardening (epochs/probes/seeds/CoT) → new questions
priority: high
links_finding: [[baking-is-associative-prompting-is-directional]]
created: 2026-06-06
hypothesis: Injecting a transitive rule chain (Zorv→Plonk→Marn→Wexil→venomous) over fictional entities — so the prior belief in every entailment is ~0 — lets us measure INTERNAL (no-CoT) propagation of deductive consequences cleanly. Prompting will shift belief toward the entailed answer at every depth (bounded by the model's single-pass reasoning depth); baking will inject the near-depth entailments but propagate to deep entailments only if the trajectories exercise them, and (per cycle 1) less far than prompting at depth.
acceptance_criteria: "Zero-prior check: |prior belief| small at all depths (fictional entities). DECISIVE if (a) prompted shift > 0 and DECAYS with deductive depth (single-pass reasoning limit), and (b) baked shift compared to prompted per depth with bootstrap CIs over >=6 probes/depth. Clean substrate replaces the saturated tsunami h3 probes."
experiment: "bake_fact --generation.base_prompt data/prompts/veld_chain_u.md --data.context_bank data/contexts/veld_contexts.json --data.probe_bank data/probes/veld_probes.json (1B + 8B)"
links: [[propagation-bounded-by-trajectory-coverage]], [[size-helps-fidelity-not-the-propagation-gap]], [[q-propagation-cot-confound]], [[q-propagation-hardening]]
---

## Question
The user's sharpest framing: teach a new theorem and see whether its CONSEQUENCES emerge internally. A
transitive entailment chain over fictional entities is the cleanest instance — every consequence is
logically determined, the prior is ~0 (so no ceiling/saturation artifact like tsunami h3), and answering a
deep-depth probe with NO chain-of-thought requires the model to compose the chain in a single forward pass.
This isolates internal propagation from CoT chaining (which we test directly in [[q-propagation-cot-confound]]).

## Plan (cheap; no new code — the instrument is general)
- Assets: veld_chain_u.md (the rules), veld_contexts.json (elicitation contexts so trajectories carry the
  entailments), veld_probes.json (depths 0–4, >=6 polarity-balanced probes/depth, anchored at Zorv +
  not-entailed converse controls).
- Run prompted-vs-baked on 1B (GPU0) + 8B (GPU1), 20+ epochs (train to propagation convergence per cycle-2
  lesson). Verify prior ≈ 0; plot prompted vs baked shift vs depth with CIs.

## Notes
- This is the clean substrate for the whole program: zero prior + more probes/depth + logically-determined
  consequences. If it works, it becomes the default testbed and the tsunami fact is retired to a "real-world
  fact with priors" robustness check.
- Sets up [[q-propagation-cot-confound]]: depth-4 no-CoT vs CoT on the SAME chain is the definitive
  internal-vs-chaining test.
