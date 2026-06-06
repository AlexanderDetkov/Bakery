# Research log

Append-only journal — one block per `/research-loop` cycle (and per `/lint-research` pass). The
loop reads the TAIL of this file to orient; do not rewrite history above.

---

## Cycle 1 — 2026-06-06 — knowledge propagation: prompting vs baking (instrument + first bakes)
**Question:** [[q-propagation-prompt-vs-bake]] (active). Does a baked fact reach n-hop consequences as
far as the same fact in the prompt?

**Built the instrument** (gated green: 37 fast tests + lint + CPU smoke):
- `propagation` metric (`bakery/eval/metrics/propagation.py`) — no-CoT forced-choice belief
  logP(pos)-logP(neg) per hop, for prior / prompted / baked on ONE checkpoint.
- `fact_propagation` builder (`bakery/trajectories/fact_propagation.py`) — categorized context bank
  (restate/consequence/neutral/mixed) so trajectory TYPE is a knob.
- `bake_fact` experiment + assets (tsunami fact u, context bank, held-out probe bank, hops 0–3).
- Tests: `tests/test_fact_propagation.py`, `tests/test_propagation_metric.py`.

**Launched (both GPUs, parallel; separate temp ledgers to avoid run-log races):**
- GPU0: `prop-tsunami-1b-r16-mixed` — Llama-3.2-1B-Instruct, mixed contexts, rank16, 30×4 traj, 15 ep.
- GPU1: `prop-tsunami-8b-r16-mixed` — Llama-3.1-8B-Instruct, same config (batch 4, max_new 128).
Analysis + finding to follow in this cycle once the poller reports both complete.

**Analysis + verification (same cycle):** all 5 runs completed, converged. Ran a 5-lens adversarial
verification panel (Workflow) + synthesis BEFORE recording — it confirmed the numbers to machine precision
and the metric's correctness (an agent re-loaded 1B+adapter and recomputed beliefs), but caught real
overstatements (a sign error in C1's h0 fidelity; an over-strong "neutral≈0 at ALL hops" quantifier;
count/convergence/model-size confounds in C2/C3/C4).

**Finding** → [[propagation-bounded-by-trajectory-coverage]] (outcome positive, confidence medium):
- **C3 CONFIRMED (high):** eval_kl ⟂ propagation. neutral eval_kl 0.012 (lowest) → baked_shift ≈0;
  restate eval_kl 0.140 (highest) → reaches h2 (+1.64). A low KL ≠ fact learned.
- **On/off-topic GATE robust:** most-converged run (neutral) injects ≈0; on-topic reaches h2. Baking only
  distills what the trajectories exercise — the core "baking ≠ prompting" mechanism.
- **C1 (8B gap grows with hops, baked collapses at h3) SUGGESTIVE:** reverses on 1B, far-hop confounded by
  prior saturation + collapsing teacher. → [[q-propagation-hardening]].
- **C4 (scale → faithfulness) DESCRIPTIVE:** 4 uncontrolled axes. → [[q-propagation-model-scale]].

**Code (additive, tests green):** propagation metric now logs per-probe beliefs (CIs + polarity check);
manifest records the run seed. **Gate:** `make test-fast` = 37 passed. **Ledger:** 5 completed rows merged
into run-log.jsonl. **Resolved** [[q-propagation-prompt-vs-bake]]; enqueued type(refined)/hardening/size/
model-scale/cot follow-ups. Figures: results/bake_fact/_fig_{prompt_vs_bake,by_type}.png.

**Both GPUs used:** GPU0 ran 1B-mixed then the 3-run type sweep; GPU1 ran 8B-mixed concurrently.


