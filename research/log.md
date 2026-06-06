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



---

## Cycle 2 — 2026-06-06 — trajectory SIZE sweep (does propagation scale with #trajectories?)
**Question:** [[q-propagation-trajectory-size]] (active). Hold contexts fixed (mixed, 30 ctx), vary
trajectories_per_context ∈ {1,4,8,16} → total train trajectories {30,120,240,480}; isolates sample
SIZE from the coverage/type axis. 1B, rank16, 20 epochs (to a converged plateau, controlling the
steps-per-epoch confound). Per-probe beliefs now logged → bootstrap CIs in analysis.
**Launched (both GPUs):** GPU0 prop-size-tpc{1,8}-1b; GPU1 prop-size-tpc{4,16}-1b. Separate temp
ledgers. Analysis + finding to follow once the poller reports all 4 complete.

**Analysis + verification (same cycle):** all 4 runs completed & plateaued (eval_kl). Two adversarial
skeptics re-derived numbers (exact match, no errors) and hunted confounds. Outcome → INCONCLUSIVE on the
"propagation scales with size" headline; finding [[size-helps-fidelity-not-the-propagation-gap]]:
- eval_kl ↓ monotonically with data (0.107→0.069→0.049→0.042) — clean but expected.
- baking stays BELOW the prompting ceiling at h0/h1 (paired CIs exclude 0) at EVERY size — more data ≠
  closing the gap.
- **eval_kl converges BEFORE propagation** (tpc8: eval_kl flat but baked_shift still rising at ep20) — a
  dynamical version of cycle-1's eval_kl⟂propagation decoupling. Important methodological finding.
- "rise then plateau" NOT established: steps-confounded (15× span at fixed epochs), n=4 CIs overlap,
  propagation under-converged; h3 probes degenerate (prompted only +0.56) → h3 claim dropped.
**Code (additive, gate green 37):** added analysis.plot_propagation `by_size`. **Ledger:** 4 rows merged.
Size question → open/medium with a de-confounded re-design (matched steps, ≥3 seeds, propagation-convergence
early-stop, ≥12 non-saturated probes/hop). Next: prioritize TYPE (coverage seems to dominate size).

**Both GPUs used:** GPU0 ran tpc{1,8}; GPU1 ran tpc{4,16} concurrently.

---

## Cycle 3 — 2026-06-06 — zero-prior synthetic DEDUCTIVE-chain testbed (the "theorem ⇒ consequences" case)
**Question:** [[q-propagation-deductive-chain]] (active). Inject a transitive rule chain over FICTIONAL
entities (Zorv→Plonk→Marn→Wexil→venomous) so priors ≈ 0 (fixes the tsunami h3 saturation). Measure no-CoT
forced-choice belief by DEDUCTIVE DEPTH (0–4 = #composed rules); pos = logically-correct answer (belief>0 =
correct-direction; works for both forward-entailment Yes-pos and converse/negation No-pos controls). 30
probes (6/depth, 3:3 polarity). No new code — instrument is general (CLI overrides point at new assets).
**Launched (both GPUs):** GPU0 prop-veld-1b-mixed; GPU1 prop-veld-8b-mixed. 24 ctx mixed, 4 traj/ctx, 30
epochs (train to propagation convergence). Analysis + finding to follow once both complete.

**Analysis + verification (same cycle):** both Veld bakes completed (eval_kl ~0.09). 3-lens adversarial
panel + independent recompute. Per-probe logging (cycle-1 add) let us split pos=No controls by LOGICAL FORM
(converse vs negated-forward) — key. Finding [[baking-is-associative-prompting-is-directional]] (positive,
medium):
- **CLEAN HEADLINE (verified independently):** on 5 true CONVERSE probes (8B), baking pushes belief to
  AFFIRM the false converse (shift −4.75, all 5 agree) while prompting leaves correct skepticism intact
  (−0.01). ⇒ baking installs an UNDIRECTED associative chain; prompting preserves direction.
- forward entailments: both raise (prompting more, strong to ~depth 3, n.s. at depth 4); combined baked <<
  prompted in point estimate (sig only at d2, n=6).
- entail-rises / converse-degrades ANTI-CORRELATION over epochs ⇒ dissociation robust to undertraining
  (though 8B bake NOT converged at ep30: endpoint 4.43 still rising → magnitude gap budget-dependent).
- 1B: small depth-limited forward propagation even prompted (+1.31 pooled), converse WORSENED; not a clean
  capacity gate at this power.
- yes-bias large (confirmed) → polarity-balanced + converse-contrast load-bearing; report combined only.
**No code change. Gate:** make test-fast 37 passed. **Ledger:** 2 rows merged. **Resolved**
[[q-propagation-deductive-chain]]; **promoted** [[q-propagation-cot-confound]] to HIGH (the definitive
internal-vs-chaining test — the user's central CoT question — is the clear cycle-4 pick). Fig:
results/bake_fact/_fig_veld_depth.png. **Both GPUs:** GPU0 1B, GPU1 8B concurrently.

---

## Cycle 4 — 2026-06-06 — CoT vs no-CoT control (the user's central question) — readout was ARTIFACTUAL
**Question:** [[q-propagation-cot-confound]] (active). Built `bakery/eval/cot_probe.py` (CoT readout reusing
the cycle-3 Veld adapters — NO re-bake): generate a rationale, then score Yes/No after a "Final answer:" cue.
Ran 8B + 1B.
**Verification (2 skeptics) + a cue-only ablation caught a fatal flaw → finding
[[cot-cue-scoring-is-artifactual]] (inconclusive / methodological, high confidence in the diagnosis):**
- The cue-only ablation (zero rationale) reproduces the apparent CoT "deflation": forward-entail prompted
  no-CoT +10.42 → cue-only +1.42; the rule-VERBATIM hop-0 probe +14.3 → cue-only +2.5 (~12-nat collapse with
  ZERO reasoning). So the cue+single-token scoring, not reasoning, drives the effect.
- ⇒ CoT-vs-no-CoT absolute comparison is INVALID; the CoT question is still OPEN.
- Salvageable (weak): rationale-delta (full-CoT − cue-only) baked +2.57 > prompted +1.06 > prior −1.28
  (reasoning helps baked slightly more), and CoT does NOT repair baking's converse deficit (persists/grows) —
  both n-noisy, demoted.
- The no-CoT cycle-3 result ([[baking-is-associative-prompting-is-directional]]) is UNAFFECTED.
**Code (gate green, 37):** cot_probe.py now supports `--max_cot_tokens 0` (cue-only ablation) + persists
rationales. **Methodological lesson:** parse Yes/No from FREE generation + always run a cue-only ablation.
CoT question stays ACTIVE with the corrected design for cycle 5. **Both GPUs** used (1B GPU0, 8B GPU1).

---

## Cycle 5 — 2026-06-06 — CoT vs no-CoT, corrected free-generation readout (answers the user's central question)
**Question:** [[q-propagation-cot-confound]] (resolved). Rebuilt `cot_probe.py --mode freegen`: free-generate
k=5 sampled rationales, PARSE Yes/No from text, majority-vote (no cue artifact). Reused cycle-3 Veld adapters
(no re-bake). 8B + 1B, both GPUs.
**Verification:** ran a 3-lens panel — but TWO lenses made a converse SIGN ERROR (scored baked
converse-affirmation as correct → bogus "noCoT=1.00"). Caught by per-probe inspection (e.g. "is every Marn a
Zorv?" belief −7.56 = affirms converse = WRONG). The panel's RATIONALE READING (independent of the sign error)
is what's load-bearing for the CoT-chaining claims. Finding [[cot-chains-baked-rules-but-not-the-converse]]
(positive core / medium):
- **single-pass: baking saturates a yes-bias** — 8B/1B baked answer "Yes" to ALL 30 probes (fracYes 1.00) →
  0.50 acc (right on forward, wrong on reverse); 8B prompting keeps discrimination (acc 0.77, fracYes 0.73).
- **CoT forward-chains the installed rules** (baked acc 0.50→~0.75, robust to a parse penalty); rationales
  explicitly recite & chain rule1→rule4 ⟹ "Zorv venomous: Yes". The baked knowledge IS reasoning-accessible.
- **CoT does NOT fix the converse** (baked 8B 0.00 / 1B 0.40): baked rationales reason over HALLUCINATED
  reversed rules. n=5 converse + ~52% parse → suggestive; converse-ALL ≈0.64 is softer.
**Code (gate green, 37):** cot_probe.py gained `--mode freegen` (self-consistency + Yes/No parsing).
**Methodological lessons:** report accuracy WITH the yes-bias; split forward/converse/negated; print
per-probe before trusting any aggregate — *or any verifier*. **Both GPUs** used (1B GPU0, 8B GPU1).

---

## Cycle 6 (FINAL — 6/6 budget) — 2026-06-06 — matched trajectory-TYPE sweep + session synthesis
**Question:** [[q-propagation-trajectory-type]] (resolved). The clean version of cycle-1's type sweep, fixing
both confounds: MATCHED count (56 traj all categories) + MATCHED convergence (40 ep, all eval_kl-plateaued),
per-probe bootstrap CIs. Tsunami chain, 1B, both GPUs.
**Finding [[trajectory-type-is-a-binary-coverage-gate]] (positive, medium-high):**
- mean baked_shift h0–h2: restate +1.38[0.60,2.24], consequence +1.33[0.64,2.12], mixed +1.35[0.80,1.94],
  neutral +0.21[−0.18,0.65]. On-topic CIs EXCLUDE 0; neutral CI INCLUDES 0.
- ⇒ type is a BINARY coverage gate (on/off-topic), NOT graded by reasoning-richness (restate ≈ consequence —
  the "consequence-rich propagates further" sub-hypothesis is REFUTED at this scale). Re-confirms eval_kl⟂prop
  (neutral lowest eval_kl 0.009 yet ~0 injection). Baking (~+1.35) still << prompting (~+3.4).
**Wrote the capstone [[SYNTHESIS-baking-vs-prompting-propagation]]** tying cycles 1–6 into one answer:
*baking transfers an ASSOCIATIVE SHADOW of the prompted model over the trajectory distribution — propagating
a fact's consequences only insofar as the trajectories cover them and they're reachable by forward
association; it does not transfer directional/logical structure, and single-pass it collapses to affirmation.*

## SESSION WRAP-UP (6/6 cycles reached → loop halts, no reschedule)
6 cycles, 6 findings + 1 synthesis, instrument built from scratch (bake_fact / fact_propagation / propagation
metric / cot_probe), all gated green (make test-fast 37 passing throughout), every result adversarially
verified before recording (the panels caught a sign error, an overstated quantifier, a cue artifact, an
undertraining confound, and a self-inflicted verifier sign error — none reached a recorded claim unflagged).
Open for next session: hardening (CIs/seeds/facts), controlled size & model-scale sweeps, knowledge-baking
(agenda #2). To resume: re-run /loop or /research-loop; the queue + findings are the durable brain.

---

## lint — 2026-06-06 — end-of-session knowledge-base health check
All green. Orphans: none. Stale `active` questions: none (all 6 resolved or open). Broken `[[links]]`: none
real (the 3 hits — decision-slug/finding-slug/<open-question-slug> — are `_TEMPLATE.md` placeholders).
Run-log↔finding cross-refs: all finding `run_ids` present in run-log.jsonl (16 rows); no stale `status:running`
rows. Counter-arguments: present in all findings; [[SYNTHESIS-baking-vs-prompting-propagation]] uses a
program-wide "Limitations" section instead (acceptable for a meta-summary). No fixes needed.

---

# ===== SESSION 2 (manual /loop re-invocation; per-session cycle budget resets) =====

## S2 Cycle 1 — 2026-06-06 — multi-seed hardening of the Veld result (q-propagation-hardening)
Replicate the cycle-3/5 Veld findings (baked affirms converse; single-pass yes-saturation; forward shift)
across seeds {0,1,2} on 1B + 8B (seed 0 already done). Launching seeds 1,2 for both, then aggregate with
cross-seed CIs. Same Veld config as cycle 3 (24 ctx mixed, 4 traj/ctx, 30 ep).

**Result (S2 c1):** all 6 seed runs completed. Cross-seed aggregation (seeds 0,1,2) → finding
[[veld-findings-replicate-across-seeds]] (positive, high): the capstone results REPLICATE tightly —
8B baked converse belief −7.00 ± 0.27 (affirms false converse, vs prior −1.86), single-pass yes-saturation
fracYes ≈ 1.0, forward baked +5.72 ± 0.55 < prompted +8.35. prior/prompted seed-invariant (SD 0.00, internal
determinism check). Confidence on the associative/converse + yes-saturation findings upgraded to seed-robust.
Remaining for full hardening: ≥2 more facts, ≥12 probes/depth → [[q-propagation-hardening]] stays active.
