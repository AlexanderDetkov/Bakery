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

## S2 Cycle 2 — 2026-06-06 — multi-FACT hardening: 2nd synthetic chain (Tellus) for fact-generality
Built Tellus chain (Grummel→Fenn→Drask→Yorl→glows), structurally identical to Veld. Q: do baking's
converse-affirmation + single-pass yes-saturation GENERALIZE to a 2nd chain? Launching 8B-s0 + 1B-s{0,1}.

**Result (S2 c2):** Tellus chain bakes completed (8B-s0, 1B-s0/s1). Cross-chain (Veld vs Tellus) →
finding [[yes-saturation-is-fact-general-converse-amplification-is-not]] (medium-high):
- FACT-GENERAL (both chains, seeds): single-pass yes-saturation (baked fracYes 0.97–1.00) → forward
  entailments correct, reverse/converse wrong; forward shift positive but < prompting.
- CHAIN-SPECIFIC: the dramatic Veld converse-amplification (8B baked −6.61 vs prompted −1.88) does NOT
  generalize — Tellus baked converse −2.16 ≈ prompted −2.61 (Tellus prior already −4.21). The −4.75 number
  was Veld-specific; the yes-saturation MECHANISM is general.
- Recast the capstone headline to "direction-blind via yes-saturation" (updated synthesis + associative
  finding). Multi-fact hardening worked: caught a chain-specific number masquerading as a law.

## S2 Cycle 3 — 2026-06-06 — is direction-blindness FIXABLE by contrastive trajectories?
New q [[q-fix-converse-via-contrastive-trajectories]]. Bake over CONTRASTIVE Veld contexts (elicit "X is Y but
not every Y is X") vs matched MIXED baseline (12 ctx, 8 traj/ctx, 30 ep), 8B+1B. Does contrastive baking fix
the baked converse? Launching 4 bakes.

**Result (S2 c3):** 4 bakes completed. Contrastive vs matched-mixed (Veld, 12 ctx) → finding
[[contrastive-trajectories-do-not-fix-the-converse]] (negative, medium-high):
- contrastive did NOT fix the converse: 8B baked converse C −5.14 vs M −6.54 (both ≪0); correct-rejection
  rate 0.00 in BOTH; fracYes 1.00 in BOTH; forward kept (C +5.25). Only a small nudge.
- WHY (confound check via generation): the base+u TEACHER itself generates converse-AFFIRMATION
  ("every Zorv is a Plonk → every Plonk must be a Zorv"), so contrastive trajectories carry little clean
  rejection signal. Baking inherits the teacher's GENERATIVE bias (worse than its forced-choice −1.88).
- Refines coverage principle: baking is bounded by what the TEACHER GENERATES, not what contexts invite.
- Resolved [[q-fix-converse-via-contrastive-trajectories]]; enqueued [[q-fix-converse-stronger]] (explicit-
  directional u / trajectory filtering — isolates teacher-signal vs LoRA limit).

## S2 Cycle 4 — 2026-06-06 — explicit-directional u' (q-fix-converse-stronger, arm a): teacher-signal vs LoRA limit
u' = Veld chain + explicit "rules are ONE-DIRECTIONAL; the converse never holds". Bake over mixed (12 ctx,
8 traj/ctx, 30 ep), 8B+1B; compare baked converse to the cycle-3 veldM baseline (plain u: 8B converse −6.54,
rejection 0.00). If base+u' teacher now rejects converse but baked STILL affirms → LoRA/objective limit;
if baked converse flips → it was a teacher-signal problem. Launching prop-veldD-{8b,1b}.

**Result (S2 c4):** explicit-directional u' bakes (8B,1B) completed. Two findings:
(1) [[tokenization-artifact-corrected-prompting-is-directional]] (positive, HIGH) — DISCOVERED a tokenization
artifact: the no-CoT belief scored " Yes"/" No" (ids 7566/2360) but chat models emit "Yes"/"No" (9642/2822) as
the first assistant token → under-credited "No". Validated against GENERATED answers: prompted REJECTS the
converse (4/5 generated; old metric hid this), baked AFFIRMS (0/5). Corrects the "prompting also yes-saturates/
fails converse" sub-claims (artifact); baked + forward results stand. Sharpens thesis: prompting directional,
baking direction-blind; yes-saturation is BAKING-specific. Fixed propagation.py (`_answer_logprob`, logsumexp
over space/no-space variants); gate green.
(2) explicit-directional u' did NOT fix the baked converse (still 0/5 generated, fracYes 1.0) though base+u'
rejects it → baking direction-blindness is a real LoRA/distillation limit, not a signal/coverage/metric issue.
Resolved [[q-fix-converse-stronger]]. Follow-up: re-read prior belief magnitudes under the fixed metric.

**Result (S2 c5):** Re-scored existing adapters (Veld 8B/1B, Tellus 8B) under the tokenization-robust metric
(no baking; forward passes). Robust accuracy matches GENERATED answers (8B Veld prompted conv 0.80=4/5, baked
0.00=0/5). Finding [[CORRECTED-picture-robust-metric]] (positive, HIGH) — corrects the artifactual-metric era:
- baking propagates FORWARD entailments strongly & fact-general (prior 0.0-0.07 → baked 0.87-0.93, both chains).
- baking CONVERSE is CHAIN-SPECIFIC: Veld baked 0.00 (fails) vs Tellus baked 0.80 (fine, = prompting). NOT
  uniform direction-blindness.
- prompting fully directional on 8B (fwd 0.93 + conv 0.80, both chains).
- 1B can't propagate forward even prompted (0.07) — capacity-gated single-pass (defaults to "No"; its
  converse-reject 1.00 is just a global No-bias).
SUPERSEDES the uniform "yes-saturation / associative / direction-blind" claim (metric artifact: 1B was
No-biased, 8B-Tellus baked discriminates). Forward/coverage/eval_kl⟂prop stand. Synthesis banner + index updated.

## S2 Cycle 6 (FINAL) — 2026-06-06 — lint + session-2 wrap-up
Lint: clean (no orphans; all findings have counter-args; the one "[[links]]" hit is prose in this log, not a
real link; q-propagation-hardening set open — stale-active). KB: 12 findings + capstone, 7 open-questions
(4 resolved this session), 29 ledger rows.

### SESSION-2 WRAP-UP (manual re-invocations; halting at the 6-cycle budget — no reschedule)
Session 2 hardened and then CORRECTED the session-1 picture:
- S2c1: the Veld findings replicate across 3 seeds (tight CIs).
- S2c2: a 2nd chain (Tellus) — yes-saturation looked fact-general; converse-amplification chain-specific.
- S2c3: contrastive trajectories don't fix the converse — the TEACHER itself generates converse-affirmation.
- S2c4: **found+fixed a tokenization artifact** in the belief metric (scored " Yes"/" No" but models emit
  "Yes"/"No"); validated against generated answers. Fixed `propagation.py` (`_answer_logprob`).
- S2c5: **re-scored under the robust metric → CORRECTED thesis** ([[CORRECTED-picture-robust-metric]]):
  baking propagates FORWARD entailments well (fact-general, prior~0.05→baked~0.90); converse reliability is
  CHAIN-SPECIFIC (Veld fails 0.00, Tellus fine 0.80); prompting is fully directional on 8B; 1B can't propagate
  single-pass even prompted (capacity-gated). The uniform "associative/yes-saturated/direction-blind" framing
  was a metric artifact + Veld-specific.

**Definitive answer = [[CORRECTED-picture-robust-metric]]** (capstone), with [[SYNTHESIS-baking-vs-prompting-propagation]]
as the narrative arc (correction-bannered). **Open for next session:** re-score size/type sweeps under the
robust metric; characterize WHEN baking fails the converse (more chains; teacher-generation correlation);
controlled model-scale sweep; knowledge-baking sequential composition (agenda #2). Resume with /research-loop.

---

## 2026-06-06 (S3) — Grokking the converse + the ~/Invertibility bridge (E1 launched)

User directive: go deeper on (1) grokking-like behavior (loss plateaus, model still needs more time) and
(2) baking's converse failure — connecting to the sister project ~/Invertibility, where a transformer learns
the INVERSE map only with (a) grokking (long training past the loss plateau; transient/regularization-driven)
and (b) PATH/compositional training (sequences composing forward+inverse edges), under zero contamination —
and the reversal curse scales with coverage density, NOT capacity. Treat that as the toy-model theory; test
transfer to fact-baking on a pretrained 8B. New questions: [[q-grokking-converse-via-longer-training]]
(active), [[q-sft-vs-bake-reversal-curse]] (open).

- **Built (gate green — 42 fast tests incl. 4 new SFT + per-form metric tests):**
  - `sft` objective (`bakery/objectives/sft.py`): plain CE on the supervised span via the audited shift
    (`_sup_target_ids` added next to `_sup_pred_logprobs` so the logit->token shift stays in ONE place) — a
    matched prompting/SFT/baking comparison; never touches the gate or KL primitive.
  - per-form accuracy in the `propagation` metric: `{forward,converse,negation}_acc_{prior,prompted,baked}`,
    auto-logged per eval step (MetricResult.extra -> metrics.json) → grokking curves for free.
  - `analysis/plot_grokking.py` (JSON-only); Veld probes backfilled with a `form` field; matched
    `data/contexts/veld_directional_contexts.json` (forward/converse/path types for E2).
- **E1 launched (8B, both GPUs, ~20GB each):** bake_fact Veld 8B, 1200 epochs (vs prior 15-40), eval_period
  15, save_every 300, trajectory=mixed, batch_size 2 grad_accum 2 (batch 4 OOMs on the long Veld prompt).
  Arms: e1a wd=0 (GPU0), e1b wd=0.05 (GPU1). Per-run /tmp BAKERY_RUN_LOG ledgers (merge at finding-time).
  Watching propagation.converse_acc_baked vs eval_kl for a LATE (grokking) converse transition.
- **Queued (when a GPU frees):** E2 (trajectory type forward/converse/path at long training), E4 (SFT vs
  bake vs prompt) + a teacher-generation audit of the directional contexts; expand chains for fact-generality.

---

## 2026-06-06 (S3b) — Critical review → contamination guard + a propagation-distance redesign

A step-by-step code audit (user-requested) found the CORE baking code correct (logit/shift matching,
KL direction, LoRA toggling, prompt formatting, SFT = genuine one-hot baking all verified), but two
design flaws in the experiments I'd set up: (1) **train/test contamination was unguarded** — the held-out
probe bank never passed through the gate, and `veld_directional_contexts.json` mirrored held-out converse
probes verbatim; (2) **propagation distance was ill-defined under training-on-samples** — free samples
chain, so they state the n-hop answer and baking just recalls it.

Redesign (planned + approved):
- **Criterion F contamination guard** (`bakery/trajectories/contamination.py`, wired into the gate as a
  pluggable `contamination_validator`): an atomic-source FILTER (drop continuations stating a composed/reverse
  relation) + a direction-aware stated/held_out LABELER; HARD-FAILS if an `expect_heldout` probe is leaked.
- **Propagation distance defined properly**: source = atomic links only (matched for prompt & bake);
  distance d = composition steps beyond the source; report accuracy vs d on HELD-OUT probes (genuine) AND
  coverage-stratified. Metric emits `forward_acc_{state}_heldout_d{d}` + `propagation_distance_{state}`.
- **Grix clean chain** (6 links, 47 entity/form/distance-tagged probes, atomic-link source); contaminated
  `veld_directional_contexts.json` removed. SFT stays one-hot. Teacher forward forced eval-mode (determinism).
- Gate stays green: **59 fast tests** (16 new contamination + distance-stratified metric) + smoke.

**E1 preempted (ran to epoch 360 before the kill; user OK'd freeing both GPUs for the clean grix experiment).**
Analyzed (figs `_fig_grok_e1a.png`, `_fig_grok_e1b.png`, `_fig_grok_converse_wd.png`):
- The grokking SHAPE is real: `eval_kl` plateaus (~0.047) and forward saturates (1.0) by ~epoch 45-60, but
  `converse_acc_baked` stays ~0 until ~epoch 135, then rises to ~0.8 around epoch 150-165 — ~100 epochs after
  the loss flatlined. wd=0.05 STABILIZES it (flat 0.8, epoch 180-360); wd=0 is noisy (0.8 w/ dips to 0.2-0.4).
- BUT per-probe inspection KILLS the naive "baking learns the converse" reading: the PRIOR already rejects all
  5 Veld converses at high confidence (acc 1.0, beliefs +2.6..+3.75 — the base model defaults to "No" on
  universal claims about fictional entities). Baking ERODES this (baked beliefs +1.1..+2.8, all far below prior;
  short-training flips them to Yes = the over-affirmation). The 0→0.8 "grokking" is RECOVERY toward the prior,
  not learning; belief-SHIFT shows baking persistently DAMAGES the converse at epoch 360. Prompting also drops
  the prior's 1.0 to 0.8, failing the SAME probe ("is every Wexil a Plonk?", prompted -1.00).
- Lesson (again): print per-probe before trusting an aggregate; converse-accuracy is prior-confounded.
  Forward propagation distance (grix, prior~0) is the clean headline; report converse as SHIFT-from-prior with
  the prior baseline explicit. Runs: e1a-veld8b-long-wd0, e1b-veld8b-long-wd05 (stopped @360/1200).

- **Launched:** grix-int-1b (integration: validate the atomic filter + contamination labeling + distance
  metric live). **Next:** 8B grix E2 — prior/prompted/soft-bake/one-hot-SFT, controlled-atomic + free source,
  distance curves across epochs.

---

## 2026-06-06 (S3c) — Rigor upgrade: n-hop propagation as a Hilbert-style proof system + d′

User asked to make the task and eval rigorous: frame n-hop reasoning as PROPOSITIONAL LOGIC (axioms +
modus ponens; **proof depth = hop**), add DEPTH-MATCHED true/false probes so a trivial Yes/No responder
scores at chance, and evaluate with SIGNAL DETECTION so discrimination is separated from response bias
(the prior-confound + yes-saturation found in S3b). Built (gate green — **92 fast tests** + smoke):

- **Proof substrate** `bakery/logic/{world,proof_engine}.py`: definite-implication DAG; `proof_depth` =
  shortest path = #modus-ponens steps. SELF-TEST cross-checks BFS forward-chaining vs an independent
  Floyd–Warshall oracle on provability AND minimal depth for every pair of 20 random worlds.
- **Dataset generator** `scripts/make_logic_world.py`: 4 disjoint-vocabulary worlds (`lw_alpha`..`lw_delta`,
  ~30 atoms, depths 1–6, balanced true/false per depth). Three engine-VERIFIED negative types — converse,
  cross-component, and missing-final-edge (strongest: real prefix path, one absent terminal edge). All
  280 probes label-checked, 0 errors.
- **DAG contamination guard** `bakery/trajectories/contamination_dag.py`: direction-aware FILTER + LABELER
  (keeps only forward taught edges; a REVERSED statement is dropped — caught by the 8B sanity run, which
  hard-failed the gate until the filter became direction-aware). `assert_probe_schema_and_balance` hard-fails
  one-sided depth cells under the atomic source. Wired into the gate via `world_spec` (no new builder).
- **`dprime` metric** `bakery/eval/metrics/dprime.py`: per depth × state × neg_type → hit/FA/d′/criterion/
  bacc/AUROC; `prop_distance_dprime`. Φ⁻¹ via Acklam (no scipy). KEY test: a pure yes-bias → d′≈0 even at
  "accuracy" 1.0. Plots `analysis/plot_dprime.py` (dprime/roc/criterion/negtype/grokking; JSON-only).
- Experiment `bake_logic` (primary world `lw_alpha`, atomic source, metrics eval_kl+propagation+dprime).

**8B sanity (`lw-alpha-sanity-8b`, 6 ep) — instrument validated:** gate passes (n_stated=5 = the d=1 atomic
links, n_held_out=65, 0 stated on converse/cross/missing); yield 115 train traj; **prior d′≈0 at all depths**
(base model can't discriminate — prior-confound rendered harmless); **prompted d′ propagates ~2 hops single-pass**
(d1=d2≈1.47 → d3+≈0, prop_distance=2); baked d′=0 at 6 ep (grokking is the long-run question).

**Launched (8B, lw_alpha, 1000 ep, identical cached trajectories):** `lw-alpha-bake-8b` (soft-bake, GPU1),
`lw-alpha-sft-8b` (one-hot, GPU0, auto-start after grix-free). **Next:** 3 generalization worlds × {bake,sft}
for cross-world CIs, then d′-vs-depth / grokking / soft-vs-one-hot analysis + finding. NOTE: 6 pre-existing
legacy metrics lack tests (out of scope; flagged for a cleanup pass).

---

## 2026-06-09 (S4c1) — Baking vs SFT vs prompting on the d′ propagation instrument (matched data)

**Question:** [[q-sft-vs-bake-reversal-curse]] (high) — is baking's converse/propagation behavior the same as
forward-only SFT's reversal curse, or does the KL-distillation framing change it? Now testable on the
rigorous `theorem_qa` + d′ instrument (proof-depth, bias-immune signal detection), with the completed
`qa-sbake-n{1,2,3}-s0` bakes as the matched bake arm and prompting read from each run's `dprime` metric.

**Design:** mirror `qa-sbake-n{1,2,3}-s0` EXACTLY, swapping only `--train.objective bake -> sft`. Same model
(Llama-3.1-8B-Instruct), same sampled trajectories (sample_trajectories=True, seed 0/split_seed 0 → same
cached trajectories; sft & bake share the `base_disable_adapter` sampler), 1000 ep, eval_period 20. Resolved
config validated against all 5 criteria via --print-config. Only the loss differs (CE-on-tokens vs KL-to-teacher).

**Launched (GPU0, memory-guarded, auto-starts after qa-sbake-n2-s1 frees GPU0):** `qa-ssft-n1-s0`,
`qa-ssft-n2-s0`, `qa-ssft-n3-s0` (chained). GPU1 continues the long grokking bake `qa-sbake-n1-s0-long`.
Both GPUs saturated. **Next:** when qa-ssft-n1-s0 completes → /analyze-run bake-vs-SFT (forward/converse d′
by depth) → gate (make test-fast) → /record-finding. Caveat: sampled arm carries the SAMPLED-CoT leak on
held-out d≥2 (affects bake & SFT equally → the bake-vs-SFT DIFFERENCE stays clean).

---

## 2026-06-09 (S4c2) — bake vs SFT vs prompting on the d′ instrument: faithfulness-vs-sharpness, NO reversal curse

**Result (n1-s0, matched cached trajectories; only the loss differs):** the reversal-curse hypothesis is
REFUTED in its predicted direction. SFT does NOT collapse on the converse — d′ 1.79 (vs baking 0.89); BOTH
arms reject the (non-provable) converse, so there is no converse curse on `theorem_qa` for either objective.
The bake-vs-SFT split is **faithfulness vs sharpness**: baking mimics the teacher (eval_kl 0.47, by
construction) and inherits its weak single-pass propagation; one-hot SFT ignores the teacher (eval_kl 4.43,
~10×, train_kl→0 both) and sharpens the same supervised tokens → uniformly higher probe d′ (d1 1.43, fwd 1.21,
conv 1.79). Prompting carries the fact but propagates weakly single-pass (held-out fwd ~0.1, conv ~0).

Gate: `make test-fast` GREEN (131 passed). Finding: [[bake-tracks-teacher-sft-sharpens]] (confidence medium —
n=1 depth×seed; CONVERSE comparison is leak-free, FORWARD held-out is CoT-leak-confounded for both arms).
Fig: results/bake_theorem_qa/_fig_bake_vs_sft_n1s0.png. Question kept ACTIVE.

**Launched/active:** qa-ssft-n2-s0, qa-ssft-n3-s0 chained on GPU0 (curriculum grid); qa-sbake-n1-s0-long
(grokking) continues on GPU1 (ep ~5150/10000). **Enqueued:** [[q-fidelity-vs-sharpness-frontier]] (new);
teacher-forced clean bake+SFT pair + grokking-length SFT (listed in the active question).

_S4c2 follow-up (09:23): queued the TEACHER-FORCED clean control (qa-tf-{bake,sft}-n1-s0,
data.sample_trajectories=False, 1000 ep) on a process-gated GPU0 driver — auto-launches after the sampled-SFT
sweep (qa-ssft-n{2,3}-s0) finishes (~14:45). Removes the sampled-CoT leak so the forward-propagation
magnitudes in [[bake-tracks-teacher-sft-sharpens]] can be trusted (the converse comparison is already
leak-free). Keeps GPU0 busy. The persistent Monitor only watches qa-ssft-*; qa-tf-* completions are caught at
the heartbeat._

_S4c2 follow-up (13:37): queued the SEED-1 SFT sweep (qa-ssft-n{1,2,3}-s1, matched to the completed
qa-sbake-n{1,2,3}-s1 bakes) on a driver gated behind the teacher-forced control → 2-seed CIs for
[[bake-tracks-teacher-sft-sharpens]]. GPU0 pipeline now full to ~02:00: n3-s0 SFT → tf-{bake,sft} → seed-1
SFT sweep. GPU1: long grokking bake to ~20:30. Monitor watches only qa-ssft-*-s0; tf/s1 caught at heartbeat._

_S4c2 follow-up (17:38): queued GROKKING-LENGTH SFT on GPU1 (qa-ssft-n1-s0-long, 10k ep, mirrors the long
grokking BAKE — same recipe, swap objective), gated on qa-sbake-n1-s0-long completing (~20:30). Cross-refs
[[q-grokking-converse-via-longer-training]] × [[q-sft-vs-bake-reversal-curse]]: does long SFT diverge from the
teacher even more / does its converse grok? Both GPUs now booked overnight — GPU0: tf-{bake,sft}→seed-1 SFT
sweep; GPU1: long bake→grokking-SFT._

_S4c2 leak-check (21:42): teacher-forced BAKE (qa-tf-bake-n1-s0) done. The sampled-CoT leak did NOT inflate
baking's forward held-out d′ — clean 0.54 ≈ sampled 0.52 (converse 0.73 vs 0.89; both strong). The bake-arm
leak caveat in [[bake-tracks-teacher-sft-sharpens]] is largely CLEARED. Side note: teacher-forced LOWERS
trained-edge recall (d1 1.12→0.45) — thinner supervised targets (1369 vs 3686 tokens), consistent with the
token-richness principle. tf-sft (clean SFT arm) running; will fold the full clean comparison into the finding
when it completes._
