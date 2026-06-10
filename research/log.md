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

## 2026-06-07 (S4c1) — n-hop curriculum × seed factorial LAUNCHED (4× 8B, all GPUs)

New session, user directive: "difference between prompting knowledge propagation and baking on datasets
with n=1,2 (≤n-hop trajectories) × trajectory-regularization × seeds (START AT 10) — interested in TRAINING
DYNAMICS; baking has grokking-like behavior so bake for a while; keep all GPUs busy; not SFT for now."

Orient: branch `research/knowledge-propagation` has the full validated proof-system instrument
(`bake_theorem_qa` / `bake_logic`: proof-depth=hop, bias-immune d′, DAG contamination guard, 92 fast tests)
+ the findings base. `results/` and `trajectory_cache/` are gitignored, so prior compute (lw-alpha-*, the
qa-* n-sweep) did NOT travel with the branch — only findings + ledger rows did. Three ledger rows are stale
`running` orphans (qa-bake-n2-beta, qa-sbake-n1-s0, qa-sbake-n3-s1; seeds 0/1; no result dirs; dead procs).
GPUs were idle on arrival. Re-applied the transformers `<5` pin to pyproject/requirements (the branch lacked
it; a future `make install` here would pull the broken 5.x).

Synthesized + set ACTIVE [[q-n-curriculum-propagation-dynamics]] — the durable home for the user's
directive (n×reg×seed×dynamics on the d′ instrument). Cycle-1 reads the n×seed SHAPE; regularization is the
next axis.

- **/validate-bake PASS** (both n configs, --print-config): single 8B base checkpoint both sides, held-out
  by proof depth, full-vocab KL (structural), sampled teacher (canonical "bake what prompting does"),
  concrete seed/sampling.
- **Launched (all 4 GPUs, ~18GB each, healthy):** `bake_theorem_qa` Llama-3.1-8B, lora r/α 16, bake,
  sampled teacher, 600 ep, eval_period 10, save_every 150, bs2 ga2, lr1e-4, per_depth_cap 16, reg OFF.
  Reseeded fully (split_seed = seed). Arms:
    - qa-bake-n1-s10 (GPU0), qa-bake-n2-s10 (GPU1), qa-bake-n1-s11 (GPU2), qa-bake-n2-s11 (GPU3).
  Watching: held-out d′ vs proof depth per arm; per-epoch d′ (grokking) overlaid on eval_kl; prompted d′ as
  the propagation ceiling (same teacher for both n); behavior_drift baseline.
- **Queued (relaunch as GPUs free, keep them busy):** trajectory-regularization axis
  ([[q-regularization-preserves-behavior]], num_train_contexts {0,32,128}); more seeds for CIs; teacher-forced
  control arm.

**CONFIG CORRECTION (same cycle, pre-data):** the first launch at bs2 ga2 / 600 ep projected to ~10-13 h
PER RUN — each epoch recomputes 304 teacher forwards over ~500-token sequences (the base framing prepends
the full ~30-axiom prompt u; n=1 data_stats: 304 train / 192 eval traj, mean_base_len ≈ 498). Diagnosed via
log mtime frozen at the data-stats line while GPU stayed 100% (the runner logs only at eval boundaries;
d′ eval is cheap — logprob scoring, no generation). KILLED all 4 and RELAUNCHED at **bs4 ga1** (= the SAME
effective batch 4, so identical optimization; ~2× fewer microbatches/epoch) + **400 ep** (still ~2.5× past
the known grokking window ~135-165; d′ logged every 10 ep) + expandable_segments. bs4 fits 17.6 GB. New
projection ~3-4 h for the 4-run factorial in parallel. Will monitor live metrics.json and stop early if d′
plateaus, extend the promising arm if still climbing at 400 (small-run-promises-longer-run).
Run ids unchanged: qa-bake-n{1,2}-s{10,11}.

## 2026-06-07 (S4c1 — ANALYSIS + RECORD) — n×seed cycle-1 done; seed-CI launched

**Analyzed** qa-bake-n{1,2}-s{10,11} (8B, bake, sampled teacher, lw_alpha, stopped ep110–170; dynamics
saturate ~ep50, d′ JSON-only). **GATE: 129 passed.**

Headline → [[baked-propagation-tracks-trained-depth-no-compositional-bonus]] (positive, medium conf):
- Prompting ceiling reproducible across all arms: prompted d′ d1..d4 = [1.12, 0.53, 0, 0] (~1.5 hops).
- Baked d′ post-plateau: reaches trained/distilled depth; n2 lifts held-out d2 above teacher (seed-11:
  n1 d2=0.55≈teacher vs n2 d2=1.18). **d3 ≤ 0 EVERY arm, max-ever 0.00/0.00/−0.15/−0.61, NO grokking
  through ep170** → bounded by max(trained-depth, teacher-reach), no +1 compositional hop.
- eval_kl plateaus ~ep30 (n1~0.25, n2~0.15) while depth-d′ set early & flat → eval_kl⟂propagation + no
  late deep-hop grokking.
- d2 curriculum effect direction-consistent but **seed-confounded** (mean n2 1.05 > n1 0.74 = +0.31; per
  seed {−0.01, +0.63}; n1-s10 anomalous d2=0.93). Limiting factor = seeds/probe-noise, NOT epochs.

**Decision**: killed cycle-1 at ep110–170 (data sufficient; marginal info to ep400 ≈ 0 for the headline),
reallocated all 4 GPUs to the **seed-CI** (the actual bottleneck for the d2 claim).

**Launched (4× 8B, all GPUs, bs4, 200 ep)**: qa-bake-n{1,2}-s{12,13} — adds seeds 12,13 → 4-seed contrast
for the d2 curriculum gap + the n1-s10 anomaly (compositional bonus vs noise).

**Enqueued**: [[q-teacher-ceiling-vs-objective-limit]] (teacher-forced ground-truth + matched-count control:
is the d3 ceiling the teacher or the objective?). Also flagged: longer grokking sentinel; regularization
axis [[q-regularization-preserves-behavior]] next.

## 2026-06-08 (S4c2 — seed-CI analyzed + RECORDED; teacher-forced control launched)

**Analyzed** the 4-seed CI (cycle-1 s10,s11 + cycle-2 s12,s13; 8 runs total; stopped n1 ep120 / n2 ep80-90;
JSON-only). **GATE: 129 passed.** Updated [[baked-propagation-tracks-trained-depth-no-compositional-bonus]]
to confidence HIGH.

4-seed baked d′ (postmean ep≥50), prompting ceiling [1.12, 0.53, 0, 0]:
- d1: n1 0.95±0.32, n2 **1.11±0.04** (n2 recalls reliably; n1 noisy, one 1.46 outlier).
- d2: n1 0.70±0.18, n2 0.92±0.17 → Δ +0.21 but **ranges OVERLAP, NS (~p0.15)** at 4 seeds; both > teacher 0.53.
- d3: n1 −0.19, n2 −0.15 → **≈0 both n, all seeds — ceiling ROBUST; curriculum doesn't bootstrap d3.**
- No grokking: d′ set ~ep50, flat through ep120-170; eval_kl plateaus ~ep30.

**RESOLVED** [[q-n-curriculum-propagation-dynamics]]: H2 (teacher-bounded) confirmed robustly; H1 (n=2 extends
d2) weak/NS; H3 (grokking) refuted. The user's n×seed×dynamics directive is answered on the d′ instrument.

**Decision**: stopped the (converged) seed-CI; reallocated all 4 GPUs to the mechanism question.
**Launched (4× 8B, teacher-forced, sample_trajectories=False, 200 ep, bs4)**: qa-tf-n{1,2}-s{10,11} →
[[q-teacher-ceiling-vs-objective-limit]] (active). Tests: does injecting engine-verified ground truth (facts
the prompted teacher itself can't generate) push baked d′ at trained depths toward 1, and does held-out d3
STILL stay 0 (no compositional bonus independent of teacher)? Isolates teacher-reach vs objective-capacity.

**Queued next**: the trajectory-regularization axis [[q-regularization-preserves-behavior]] (user's 3rd axis).

## 2026-06-08 (S4c3) — PIVOT to regularization (user-requested) — dose-response launched

User: "Can you look at the effect of regularization." → prioritized the 3rd named axis.
Preempted the teacher-forced control (only ep10, no usable d′ yet; [[q-teacher-ceiling-vs-objective-limit]]
stays active, resume after). Confirmed via code read: `behavior_drift` = KL(base(no-prompt) ‖ baked(no-prompt))
on a held-out SQuAD anchor slice (greedy fixed reference); auto-logged via extra_metrics, returns None at
num_train_contexts=0 (so reg=0 has no drift number — by metric design).

**Launched (4× 8B, all GPUs, bake, sampled teacher, n=1, seed 10, 200 ep, bs4):**
  qa-reg{0,32,128,256}-n1-s10  — num_train_contexts dose-response (none → anchor-dominated; 256 ≈ 84% of the
  ~304 logic trajectories). Reading per arm: behavior_drift (↓ with anchors?), held-out baked d′ at d1/d2
  (propagation preserved or suppressed?), eval_kl (fidelity tradeoff). reg=256 is slowest (most anchors) —
  will stop early once drift+d′ converge (~ep100).

## 2026-06-08 (S4c3 — regularization dose-response RECORDED; + train-vs-eval grokking plot)

User asked to "look at the effect of regularization" and (separately) to plot training curves to check
for grokking. **GATE: 129 passed** (incl. new `analysis/plot_traingrok.py`, isolation-clean).

**Regularization** → [[regularization-buys-behavior-preservation-cheaply]] (positive, medium conf), resolves
[[q-regularization-preserves-behavior]]. qa-reg{0,32,128,256}-n1-s10 (8B, bake, n=1, seed 10, read ep70-100;
reg256 ep50 prelim):
- behavior_drift ↓ monotonic 0.012→0.007→0.005 (32→128→256); most of the drop by 128.
- baked d2 flat {0.91,0.96,1.03,0.92}, eval_kl flat {0.262,0.255,0.256,0.264} → **preservation is CHEAP**
  (no propagation/fidelity cost), criteria met. Caveats: single seed; absolute drift tiny even at reg0
  (n=1 axiom baking already gentle); CoT-leak constant across arms (relative claim valid).

**Grokking plot** (user-requested) → `_fig_traingrok_n1n2.png` (sent). On the longest sampled runs
(n1-s10 ep170, n2-s10 ep120): train_kl keeps DROPPING while held-out d2/d3 are set by ~ep50 and FLAT, d3
stuck ~0 → **NOT grokking** (memorization continues without delayed generalization). Reinforces the no-late-
transition result. Analysis loader surfaced the **CoT-leak** (29/42 held-out probes recited by the teacher)
→ even the d2 shown is partly recall; clean ref = teacher-forced arm.

**Next**: let reg256 firm to ep≥80, then stop reg + RESUME the teacher-forced control
([[q-teacher-ceiling-vs-objective-limit]]) — it's the clean-propagation reference that de-confounds the
CoT-leak across all sampled runs (grokking + regularization + curriculum).

## 2026-06-08 (S4c4 — noise-reduction: AUROC readout + paired protocol) [plan-approved]

User: "results seem noisy — anything besides more seeds?" Planned + approved two free, high-leverage levers
(no GPU re-run, no training/gate/KL change). **GATE: 129 passed** (analysis-isolation intact).

- **AUROC/bacc readout** (analysis-only): added `--stat {dprime,auroc,bacc}` (default auroc) to
  `analysis/plot_dprime.py` + `analysis/plot_traingrok.py`; `aggregate.py` now takes a comma-list `--metric`
  + `--status any` (to include early-stopped runs). AUROC/bacc are already logged; both avoid the Φ⁻¹
  variance blow-up.
- **Empirical payoff** (re-read the SAME 8 curriculum runs): n=1 depth-2 noise sd 0.18 (CV ~26%, d′) →
  **sd 0.02 (CV ~3%, AUROC)** = ~8× reduction; curriculum contrast separation 1.18 → **1.60**; d3 at AUROC
  ≈ 0.50 chance (ceiling, now crisp). bacc over-saturates (sep 0.38) → AUROC is the primary. Updated
  [[baked-propagation-tracks-trained-depth-no-compositional-bonus]] with the AUROC re-read; sent
  `_fig_traingrok_n1n2_auroc.png`.
- **Paired matched-seed protocol** (methodology): adopted [[paired-matched-seed-protocol]] — fix
  split_seed+model_seed across contrast arms (the split is a pure fn of (split_seed,depth)), replicate
  k∈{10,11,12}, prefer teacher-forced. Going forward; earlier reseeded contrasts stay valid-but-unpaired.
- Out of scope (offered, declined): densify probe bank (GPU re-run), per-probe bootstrap CIs (code).

## 2026-06-08 (S4 SESSION WRAP-UP) — compute budget reached (max_cycles_per_session=6)

`research/STOP` absent, but the session hit the agenda's `max_cycles_per_session: 6` guard → halting the
AUTONOMOUS loop (no reschedule), per the budget rule. The session was heavily user-directed (n×reg×seed
directive, then a regularization request, then a noise-reduction plan), all on `research/knowledge-propagation`.

**Findings recorded this session (all gated 129-green, committed):**
1. [[baked-propagation-tracks-trained-depth-no-compositional-bonus]] (HIGH) — prompting propagates ~1.5 hops
   (prompted d′ [1.12,0.53,0,0]); baking reproduces the ceiling, reaches the trained/distilled depth, adds
   NO +1 compositional hop; baked d3 ≈ 0 across 4 seeds, NO grokking through ep170. d2 curriculum effect
   (n2>n1) real but modest; AUROC re-read: +0.06, separation 1.60σ, d3 at chance 0.50.
2. [[regularization-buys-behavior-preservation-cheaply]] (MED) — anchor regularization drives behavior_drift
   down monotonically (0.014→0.005) at ~no propagation/fidelity cost (AUROC d2 flat, d3 at chance); but
   absolute drift is tiny even unregularized.
3. Train-vs-eval dynamics = NOT grokking (train_kl keeps dropping; held-out flat) — `_fig_traingrok_*`.

**Decisions:** [[paired-matched-seed-protocol]] — report AUROC (≈8× less noisy than Φ⁻¹-amplified d′; contrast
sharper) + run contrasts as paired matched-seed sets. Tooling shipped: `plot_dprime/plot_traingrok --stat`,
`aggregate --status any`, `analysis/plot_traingrok.py`.

**Open / next session (highest priority first):**
- [[q-teacher-ceiling-vs-objective-limit]] — ACTIVE, was preempted @ep10. Teacher-forced n∈{1,2}×{10,11}:
  is the d3 ceiling the teacher's reach or the objective's limit? Also the clean-propagation ref (no CoT leak).
- Paired matched-seed RE-RUN of the n=1-vs-n=2 contrast (and the reg dose-response) for a tight d2 CI.
- [[q-graph-structure-diamonds-multipremise]], [[q-sft-vs-bake-reversal-curse]] (deprioritized by user).

**Compute state at wrap-up:** the 4 regularization runs (qa-reg{0,32,128,256}-n1-s10) are still training
(ep 80–160 of 200) and left RUNNING to finish the dose-response — not killed (honors "keep GPUs busy"; the
finding is already recorded + AUROC-confirmed). To CONTINUE the autonomous loop, a human bumps
`max_cycles_per_session` in agenda.md (or re-invokes /research-loop); the loop will then resume the
teacher-forced control under the paired protocol.



---

## Analysis pass — 2026-06-08 — per-negative-family AUROC re-read (no re-score) + framing note

**Trigger (user):** "look at what all the data says about prompting vs baking" → "what do the RECENT
(Hilbert proof-system) results say" → "get the per-negative-family AUROC re-read; did n=1 beat prompting at d2?"

**No new bakes.** The `dprime` metric already logs `auroc_<state>_d<d>_<negtype>` + `fa_<state>_d<d>_<negtype>`;
re-stratified the 8 seed-10–13 adapters straight from `metrics.json` via a new torch-free tool
`analysis/neg_family_auroc.py` (analysis-isolation preserved). Post-plateau mean ± popsd over 4 seeds/arm.

**Finding** → [[converse-collapse-does-not-survive-bias-immune-instrument]] (positive/medium):
- **Converse-collapse does NOT reproduce** under the bias-immune metric: baked converse-AUROC 0.68 (d1) /
  0.80–0.84 (d2) ≥ prompting; baked converse-FA 0.07–0.39 (no saturation); at d1 prompting affirms the
  converse MORE (FA 0.50) than baking (0.39). Corrects [[baking-is-associative-prompting-is-directional]]
  / [[yes-saturation-is-fact-general-converse-amplification-is-not]] for this instrument.
- **The real prompting−baking gap is the CROSS family** (unrelated/different-component pairs): prompted
  AUROC 0.93–0.95 / FA 0.08 vs baked 0.66–0.77 / FA 0.23–0.34 → over-connection across the partition, not
  direction. n=2 curriculum cuts it.
- **n=1 d2 ≈ prompting under AUROC** (0.69 vs 0.71); only n=2 (0.76) exceeds. The "n=1 beats prompting at
  d2" reading was a d′-vs-AUROC mismatch (d′ 0.70 vs 0.53; Φ⁻¹ amplifies near-ceiling).
- **Refutes the simple baking≈`rst(E)` conjecture** → revised relational-generalization note §5a:
  deviation is *spurious cross-component connectivity*, not *symmetrization*; the equivalence-relation world
  becomes the sharp test.
- Caveats: CoT-leak inflates d2 (recall-of-recited); runs preempted (`status=failed`), n2-s12/s13 ep80–90
  under-converged; one world; single-arm-per-seed; d1-converse a weak directionality test (both ≈chance).

**Also this session (separate threads):** wrote `research/notes/relational-generalization.md` (least-fixpoint
framing of the instrument family) + consolidated the opt-in speedup branch into knowledge-propagation and
pushed to origin (SSH). `make test-fast` = 138 passed / 1 xfailed.



---

## Analysis pass — 2026-06-08 — faster-training: quant benchmark + teacher-cache empirical equivalence

**Trigger (user):** "explore faster training with quantization and hyperparameters" → on free GPUs.

**Quant benchmark (qbench-{bf16,4bit,8bit,4bit-bs16}, 8B n=1 s10, partial — stopped once decisive):**
- **Quantization is SLOWER, not faster** (it's a MEMORY tool). Per 10-epoch+eval block: bf16 883s; 4bit
  1385s (1.57×); 4bit-bs16 1431s; 8bit 1824s (2.07×). Memory: bf16 18.0 / 8bit 15.5 / 4bit 12.1 GB.
- Bigger batch (4bit-bs16) did NOT speed wall-clock and under-trained per epoch. Two-models-per-GPU
  infeasible (4bit×2 = 24GB no headroom) and pointless (compute-bound → time-slice).
- Quality (AUROC/depth) comparable across precisions → quant doesn't degrade propagation (use it only when
  a model wouldn't otherwise fit). Validated the 4-bit wiring end-to-end (smoke: eval_kl 0.39→0.31, adapter
  saved). Did NOT record a finding (decisive enough; quant abandoned for the speed goal per user).

**Teacher-logit-cache empirical equivalence (qa-cmp-n1-s{12,13}-{off,on}, paired, 100 ep)** → filled the
RESULT in [[teacher-logit-cache]] + indexed:
- **~1.8× faster** end-to-end (s12 1.80×, s13 1.75×; training-only higher since the unchanged eval dilutes).
- **Empirically equivalent though NOT bit-exact:** cache on-vs-off Δ (eval_kl ≤0.004, AUROC ≤0.010) is
  SMALLER than the seed-to-seed Δ (eval_kl ≤0.009, AUROC ≤0.031) on every metric. Eval-batching is read-only
  (adapter bit-identical). ⇒ both safe to turn on for the upcoming HP sweep (Phase B).

**Next:** Phase B — lr × schedule × weight_decay convergence sweep (epochs-to-eval_kl≤0.26), cache +
eval-batching ON, AUROC quality guard.



---

## Phase B + eval-batching fix — 2026-06-09 — faster-training recipe → [[faster-training-recipe]]

**Sweep (bake_theorem_qa 8B, lw_alpha, n=1, cache cpu):** lr × schedule (80ep) + multi-seed fine-grained
convergence (eval@2) + weight_decay + eval-batching validation. ~16 short bakes across the 4 GPUs.

**Result** → finding [[faster-training-recipe]] (positive/medium):
- **Recipe: constant lr 3e-4 + teacher-cache ≈ 7× faster** to a converged bake. Multi-seed epochs-to-
  eval_kl≤0.26: lr3e-4 {s10,11,12,13}={8,8,4,4} mean 6 vs lr1e-4 {22,30} mean 26 → ~4× fewer epochs ×
  ~1.8× cache. AUROC-d2 flat (~0.69 both) → no propagation cost.
- **Schedule:** constant ≥ cosine; lr1e-4-COSINE never reaches 0.26 (decays too low) — a pessimization.
- **weight_decay 0.05 ≈ 0** (no-op at this scale).
- **Quantization is memory-only / SLOWER** (4bit 1.6×, 8bit 2× slower; mem 12/15.5 vs bf16 18 GB) — dropped
  for the speed goal. Validated 4-bit wiring e2e (no finding; decisive).
- **eval-batching OOM fixed** (commit 1243a9f): probe_batch_size=0 batched ALL rows → [n_rows,seq,vocab]
  OOM; now auto-chunks at DEFAULT_PROBE_CHUNK=8. Validated no-OOM on real 8B + bit-exact on the fake. The
  matched OFF/ON real pair is confounded by training nondeterminism (eval_kl, which ignores batch_probes,
  differed 0.026 between "identical" runs) → equivalence proof lives in the deterministic-fake unit test.

**Gate:** make test-fast = 139 passed (new test_default_chunk_bounds_batch), 1 xfailed. Housekeeping: pruned
dead qa-tf-*/qbench-* dirs, deleted merged research/baking-speedups branch.

**Next:** adopt the recipe (lr 3e-4 const + cache, ~15 ep) for the queued science arms — the teacher-forced
clean arm ([[q-teacher-ceiling-vs-objective-limit]]) and the equivalence-relation world (formulation §3).



---

## /research-loop cycle — 2026-06-09 — MULTI-GRAPH hardening of converse/cross + grokking hunt launched

**Compute fully saturated throughout** (user directive). Minted 8 sibling graphs (lw_epsilon..lw_mu, commit
08c1956; alpha-delta byte-unchanged), ran a 24-bake multi-graph sweep (12 graphs × n∈{1,2}, fast recipe
lr 3e-4 const + cache, eval-once), then auto-chained the grokking hunt.

**Finding** → hardened [[converse-collapse-does-not-survive-bias-immune-instrument]] (medium→**high**):
across **12 independent graph structures** (graph varied, seed fixed — the axis seed-sweeps can't reach),
both halves replicate: (1) **no converse collapse** — baked converse-AUROC ≈ prompting at d1, ABOVE at d2
(0.749 n1 / 0.792 n2 vs 0.627); (2) **cross-component over-connection is THE baking deficit** — baked cross
d2 0.584 vs prompted 0.900 (Δ0.32, n=1). New: the **n=2 curriculum shrinks the cross-leak** (d2 cross
0.584→0.772). The "one world" caveat is resolved. Runs: graph-lw_{alpha..mu}-n{1,2}. Gate: make test-fast 139.

**Now running (≈18h):** grokking hunt [[#19]] — 4 long bakes (n∈{1,2} × weight_decay∈{0,0.1}, 2000 ep,
eval@50, ckpt@200) watching for a LATE held-out deep-depth AUROC rise after eval_kl plateaus. wd=0.1 = the
grokking candidate; wd=0 extends the prior no-grokking null (ep170) to ~12×. All 4 GPUs busy.

**Next:** on grokking completion (or an early transition spotted via the periodic heartbeat) → analyze + record.



---

## interactive — 2026-06-09 — AGGREGATE: master synthesis of prompting vs baking propagation (user-requested)

User asked to aggregate everything toward "prompting vs baking knowledge propagation." Re-read all 17 findings
+ 3 decisions (structured digest), recomputed per-family AUROC across the two strongest run sets
(`analysis/neg_family_auroc.py` over the 12-graph sweep graph-lw_*-n{1,2} and the 4-seed qa-bake-n{1,2}-s{10..13}),
and rewrote [[SYNTHESIS-baking-vs-prompting-propagation]] to **v2 (aggregate)**, folding the proof-system era
into the (corrected) belief-metric era.

**Aggregate headline:** baking is a noisier, depth-limited, directionally-FAITHFUL copy of the prompted teacher;
its one graph-general error is **cross-component over-connection** (baked d2 cross-AUROC 0.58 vs prompted 0.90,
FA 0.61 vs 0.08; n=2 heals to 0.77). d1 fidelity gap (baked 0.83–0.88 vs prompted 0.92); d2 baking ≈/> prompting
overall; d3 ≈ chance for both (teacher ceiling). NOT direction-blind (baked converse ≥ prompted). Reconciles the
era-1 "direction-blind/converse-collapse" headline as a tokenization-bug + yes-bias + Veld-quirk artifact, with
the genuine associative signature RELOCATED to cross.

**Artifacts:** new `analysis/plot_propagation_aggregate.py` → `results/bake_theorem_qa/_fig_propagation_aggregate.png`
(prompted/baked-n1/baked-n2 per-family bars over 12 graphs); `analysis/plot_relation_worlds.py` →
`results/_fig_relation_worlds.png` (implication vs equivalence instrument). index.md capstone refreshed.

**Background unchanged:** grokking hunt [[#19]] + chained equivalence sweep [[#20]] still running on the GPUs.



---

## /research-loop cycle — 2026-06-10 — grokking hunt RESOLVED (negative) + equivalence sweep launched

**Analyze (finished experiment #19):** the 4 long grok bakes (n∈{1,2} × wd∈{0,0.1}) reached ep2000 (n1) /
ep1500 (n2). **Decisive NEGATIVE for compositional grokking** → new finding
[[grokking-null-no-late-propagation-transition]] (high): eval_kl flat from ~ep50 while d2/d3 baked AUROC stay
flat-to-declining through ep2000 — baking installs trained-depth propagation early and never gains a free hop
with ~12× more compute. Closes the "undertraining" escape hatch for the depth ceiling
([[baked-propagation-tracks-trained-depth-no-compositional-bonus]]). **Secondary:** wd=0.1 buys STABILITY at
long bake lengths — the wd=0 n=2 arm DIVERGED (eval_kl 0.19→1.65 @ep1375); wd=0.1 stayed flat. Refines
[[faster-training-recipe]]'s "wd no-op" to the short-bake regime only. Killed the diverged arm. Question
[[q-grokking-converse-via-longer-training]] → resolved-negative (only the narrow Veld-converse-paths E2 remains).
Gate: `make test-fast` PASS. Fig: `results/bake_theorem_qa/_fig_grok_depth.png`.

**Run (next experiment #20):** launched the **equivalence-world sweep** — the sharp test of "cross-component
over-connection is baking's mechanism" (eq worlds make CROSS the ONLY false family; the converse is a free
positive). 4 eq graphs × n∈{1,2} = 8 bakes, fast recipe mirroring the directed sweep exactly
(`scripts/run_equiv_sweep.sh`, waves of 3 on GPUs 0/1/2; run names `eqg-eq_{alpha..delta}-n{1,2}`). Wiring
dry-run verified (per-graph world_spec+probe_bank+base_prompt all resolve). All 4 GPUs saturated.

**Next:** on sweep completion → `analysis/neg_family_auroc.py` on eqg-* → contrast baked-vs-prompted cross-AUROC
against the directed lw_* numbers → record the equivalence finding (confirm/refute the mechanism).



---

## /research-loop cycle — 2026-06-10 — equivalence world CONFIRMS cross-over-connection mechanism (task #20)

**Analyze (finished experiment #20):** the 8 equivalence bakes (eqg-eq_{alpha..delta}-n{1,2}, fast recipe,
results/bake_theorem_qa_equiv/) all converged (eval_kl 0.07–0.15). The eq world has CROSS as the ONLY false
family (converse/missing empty — instrument validated) and makes the converse a free positive. **Decisive
CONFIRMATION** → new finding [[equivalence-world-confirms-cross-over-connection-mechanism]] (high):
(a) baking ACES within-class positives — d1 AUROC 1.00 = prompting (no direction problem at all); (b) its entire
deficit is cross over-affirmation — baked d2 cross-FA **0.93** (says "same kind" to 93% of UNRELATED pairs) vs
prompted 0.06, AUROC 0.66 vs 0.90; (c) the n=2 curriculum heals it (FA 0.93→0.29), same lever as directed.
**Starker than the directed worlds** (FA 0.61) because every within-class pair is a "yes" → over-connection scales
with affirmation density. Pins the formal picture: baking **inflates the closure** (over-permissive/low-precision),
it does not symmetrise direction. Confirms [[converse-collapse-does-not-survive-bias-immune-instrument]] +
[[relational-generalization]] §5a prediction. Gate: `make test-fast` (running). Fig: `_fig_equiv_cross.png`.

**Stale-driver cleanup:** two leftover chained drivers from the pre-compaction session fired during this cycle —
b95kd0hfe (eq-waiter) timed out after 8h (ABORT, benign, launched nothing) and brjnm0916 (grok-driver) exited 0
when grok-n2-wd0.1 hit ep2000. Neither collided with the active sweep. Verified no duplicate run.py / GPU
contention. Stopped my own now-moot completion waiter (bbydbsp72).

**Next:** the over-connection mechanism is confirmed graph- AND relation-general. Open follow-up: does reducing
"yes-pressure" in the bake distribution (balanced vs forward-only trajectory framing) reduce cross-FA? + an
equivalence multi-seed set to match the paired protocol.



---

## interactive — 2026-06-10 — equivalence finding HARDENED (seed-replicated + leak-free)

Used the idle GPUs (after the eq sweep) for two cheap, high-value hardenings of
[[equivalence-world-confirms-cross-over-connection-mechanism]]:

1. **Seed replication** (eqms-eq_alpha-n{1,2}-s{11,12}, sampled): baked d2 cross-FA 0.91±0.05 over 3 seeds
   (s10/11/12), n=2 heals to 0.32±0.10 — tight, seed-stable.
2. **Leak-free confirmation** (tf-eq_alpha-n1, teacher-forced — gate ENFORCES disjointness, no CoT leak): the
   over-affirmation is STRONGER without the leak — baked d2 cross-FA **0.97** (vs sampled 0.91), cross-AUROC
   **0.38 BELOW chance** (vs 0.60). So the CoT leak was HELPING the baked model (recall-of-recited inflated d2);
   removing it shows over-connection is worse than the sampled numbers. **The central finding's biggest caveat
   (CoT leak) is addressed and the effect was understated, not manufactured.**

Teacher-forced clean arm for the OTHER cells (directed n1/n2, eq n2) is gate-blocked (train_depth ∩ probe_depth
≠ ∅) → needs a disjoint-split builder fix ([[q-teacher-ceiling-vs-objective-limit]] → status blocked).
GPUs idle now. Gate: `make test-fast` (re-run). Next dev candidates: the split fix, or
[[q-over-connection-scales-with-yes-pressure]].

  - UPDATE: leak-free confirmation extended to **4 eq graphs** (tf-eq_{alpha,beta,gamma,delta}-n1, all built
    clean — depth-1 splits naturally disjoint): clean d2 cross-FA 0.97±.00, AUROC 0.56±.12 vs sampled 0.93/0.66.
    The CoT leak understated over-connection graph-generally. Commit pending.
