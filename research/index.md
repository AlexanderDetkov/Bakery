# Research index

The catalog. Orient from this file (and the agenda) rather than scanning directories. Keep current
(or run `/lint-research`).

## Theme: knowledge propagation under prompting vs baking
How far does an injected fact propagate to its n-hop consequences, and does baking (fact → LoRA)
propagate as far as prompting (fact in context)? Instrument: the `bake_fact` experiment +
`fact_propagation` builder + `propagation` metric (no-CoT forced-choice belief shift, per hop, for
prior vs prompted vs baked, all on ONE checkpoint). Assets: `data/prompts/tsunami_u.md`,
`data/contexts/tsunami_contexts.json`, `data/probes/tsunami_probes.json`.

## Open questions
- [[q-teacher-ceiling-vs-objective-limit]] — **active (high)** — RUNNING (4× 8B teacher-forced: n∈{1,2} × seed∈{10,11}, 200 ep): is the d3 ceiling the TEACHER's reach or the OBJECTIVE's limit? Does injecting engine-verified ground truth push the frontier; does held-out d3 still stay 0?
- [[q-grokking-converse-via-longer-training]] — active (high) — E1 was preempted @360/1200 for the clean proof-system redesign; the converse "grokking" turned out to be RECOVERY toward the (confounding) prior. Superseded as headline by the d′ instrument; revisit converse as shift-from-prior only if the n-curriculum runs show late dynamics.
- [[q-sft-vs-bake-reversal-curse]] — open (high) — prompting vs SFT vs baking on the converse at matched data; is baking's converse failure just SFT's reversal curse?
- [[q-graph-structure-diamonds-multipremise]] — open (high) — current logic worlds are single-inheritance trees (chain inferences only); add convergent DAGs (diamonds → shortest-proof) and multi-premise conjunctive rules (proof trees → 2-fact composition). Diamonds = generator-only (engine ready); multi-premise = Tier-2 saturation engine.
- [[q-regularization-preserves-behavior]] — open (medium) — the user's 3rd named axis: mix base-anchored irrelevant-question (SQuAD) trajectories into the bake to preserve general behavior; does behavior_drift fall with anchor count WITHOUT hurting held-out propagation? Turnkey: `configs/sweeps/regularization_strength.yaml` (n=1, {0,32,128} anchors, 8B). **NEXT after the teacher-forced control.**
- [[q-trajectory-coverage-and-cache-hardening]] — parked (high) — code review 2026-06-07: baking trains on a PARTIAL axiom set (27/38 edges in lw_alpha) → bake/SFT arms coverage-caveated (prior/prompted clean); plus deferred framework hardening (P0 sweep model-cache, P1 cache provenance, P2a proof_depth on negatives, P2b dedup-before-truncate). DEFERRED by user; record-only.
- [[q-propagation-hardening]] — active (high) — remaining: ≥12 probes/depth, the C1 per-hop decay; re-read magnitudes under the fixed metric
- [[q-propagation-model-scale]] — open (medium); [[q-propagation-trajectory-size]] — open (medium)
- Follow-up: re-measure all conditions under the tokenization-FIXED metric; arm (b) trajectory-filtering — are the per-hop/converse magnitudes real with CIs across facts/seeds? (≥2 facts, ≥3 seeds, ≥12 probes/hop, fix degenerate h3 probes)
- [[q-propagation-trajectory-size]] — open (medium) — cycle-2 inconclusive/confounded; needs matched-steps + multi-seed re-run
- [[q-propagation-model-scale]] — open (medium) — controlled one-family size sweep (matched LoRA params, normalized retention)

## Resolved questions
- [[q-n-curriculum-propagation-dynamics]] — resolved (S4c2, 4 seeds) → [[baked-propagation-tracks-trained-depth-no-compositional-bonus]] (d3 ceiling robust; d2 curriculum weak/NS; no grokking)
- [[q-propagation-prompt-vs-bake]] — resolved (cycle 1) → [[propagation-bounded-by-trajectory-coverage]]
- [[q-propagation-deductive-chain]] — resolved (cycle 3) → [[baking-is-associative-prompting-is-directional]]
- [[q-propagation-cot-confound]] — resolved (cycle 5) → [[cot-chains-baked-rules-but-not-the-converse]] (corrected free-gen readout)
- [[q-propagation-trajectory-type]] — resolved (cycle 6) → [[trajectory-type-is-a-binary-coverage-gate]]
- [[q-fix-converse-via-contrastive-trajectories]] — resolved (S2c3) → [[contrastive-trajectories-do-not-fix-the-converse]]
- [[q-fix-converse-stronger]] — resolved (S2c4) → [[tokenization-artifact-corrected-prompting-is-directional]] (u' doesn't fix baked converse = real LoRA limit; + metric artifact found & fixed)

## ⭐ Capstone (read these two together)
- [[CORRECTED-picture-robust-metric]] — **the definitive corrected result** (tokenization-robust, generation-validated): baking propagates FORWARD entailments well (fact-general); converse reliability is chain-specific; prompting is directional; 1B capacity-gated single-pass.
- [[SYNTHESIS-baking-vs-prompting-propagation]] — full narrative arc (cycles 1–6 + S2); has a CORRECTION banner pointing to the above.

## Findings
- [[baked-propagation-tracks-trained-depth-no-compositional-bonus]] — **prompting propagates ~1.5 hops (d′ d1=1.12,d2=0.53,d3=0); baking reproduces this ceiling, reaches exactly the trained/distilled depth (n=2 lifts held-out d2 above teacher), but adds NO +1 compositional hop and baked d3≤0 in all arms with NO grokking through ep170** (d2 curriculum effect direction-consistent but seed-confounded, +0.31 mean, per-seed {−0.01,+0.63} → seed-CI running). Runs: qa-bake-n{1,2}-s{10,11}. Sharpens the coverage-bound; reconfirms eval_kl⟂propagation.
- [[propagation-bounded-by-trajectory-coverage]] — **baking only injects what the trajectories exercise; eval_kl ⟂ propagation** (C3 confirmed high; on/off-topic gate robust; C1/C4 suggestive single-run). Runs: prop-tsunami-{1b,8b}-mixed, prop-tsunami-1b-{restate,consequence,neutral}-m12.
- [[size-helps-fidelity-not-the-propagation-gap]] — **more trajectories lower eval_kl but don't close the prompting–baking gap; eval_kl converges BEFORE belief does** (size→propagation scaling inconclusive: steps-confound + n=4 noise + propagation under-converged at 20 ep). Runs: prop-size-tpc{1,4,8,16}-1b. Figs: results/bake_fact/_fig_by_size.png.
- [[baking-is-associative-prompting-is-directional]] — **baking installs an UNDIRECTED/associative chain (affirms the false converse, −4.75); prompting preserves logical direction (−0.01); 1B propagation weak even prompted** (zero-prior Veld syllogism, no-CoT). Runs: prop-veld-{8b,1b}-mixed. Fig: results/bake_fact/_fig_veld_depth.png.
- [[cot-cue-scoring-is-artifactual]] — **the cycle-4 CoT readout was artifactual** (a cue-only ablation reproduced the "deflation"; rule-verbatim probe +14.3→+2.5 with zero reasoning) → CoT comparison inconclusive; no-CoT results unaffected. Methodological lesson: parse Yes/No from free generation, always run a cue-only ablation.
- [[cot-chains-baked-rules-but-not-the-converse]] — **single-pass baking = all-"Yes" saturation (no discrimination); CoT forward-chains the installed rules (0.50→~0.75) but hallucinates REVERSED rules on the converse** (corrected free-gen readout). Runs: prop-veld-{8b,1b}-mixed. NB: the verification panel itself made a converse sign error — caught by per-probe inspection.
- [[tokenization-artifact-corrected-prompting-is-directional]] — ⚠️**metric fix:** scoring " Yes"/" No" (space) under-credited the no-space "No" token → hid that **PROMPTING rejects the converse (4/5 generated) while baking affirms (0/5)**. Corrected thesis is cleaner (prompting directional, baking direction-blind; yes-saturation is baking-specific). `_answer_logprob` fix applied. Runs: prop-veld-8b-mixed, prop-veldD-{8b,1b}.
- [[contrastive-trajectories-do-not-fix-the-converse]] — **contrastive trajectories do NOT fix baking's converse (rejection 0.00, fracYes 1.0, both arms)** — because the base+u TEACHER itself generates converse-affirmation; baking inherits the teacher's GENERATIVE bias. Prescription: fix the teacher, not just the eliciting prompts. Runs: prop-veld{C,M}-{8b,1b}.
- [[yes-saturation-is-fact-general-converse-amplification-is-not]] — **single-pass yes-saturation is FACT-GENERAL (2 chains): fracYes≈1.0 → forward-correct, converse-wrong; the −4.75 Veld converse number was chain-specific.** Recasts the headline mechanism. Runs: prop-tellus-{8b,1b}-mixed{,-s1} + Veld.
- [[veld-findings-replicate-across-seeds]] — **the Veld findings replicate tightly across 3 seeds** (8B converse −7.00±0.27; fracYes≈1.0; forward<prompting) → upgrades confidence of the associative/converse + yes-saturation results. ≥2 facts still TODO. Runs: prop-veld-{1b,8b}-mixed{,-s1,-s2}.
- [[trajectory-type-is-a-binary-coverage-gate]] — **type is a BINARY coverage gate** (on-topic restate/consequence/mixed inject ~+1.35 CIs exclude 0; neutral +0.21 CI includes 0) — NOT graded by reasoning-richness (restate ≈ consequence). Matched count+convergence+CIs. Runs: prop-type2-{restate,consequence,neutral,mixed}-1b.

## Decisions
- [[sampled-teacher-trajectories-keep-cot]] (2026-06-07) — theorem_qa bake samples on-policy from the prompted teacher (CoT tail kept); criterion F recorded-not-enforced when sampling (teacher-forced still hard-fails). Held-out d′ under sampling is partly recall-of-recited — caveat carries into findings; `data.sample_trajectories=False` restores the clean teacher-forced arm.
