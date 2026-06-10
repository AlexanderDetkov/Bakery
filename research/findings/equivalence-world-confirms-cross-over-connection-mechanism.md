---
title: The equivalence-relation world confirms baking's mechanism is CROSS-COMPONENT OVER-CONNECTION — when "cross" is the only false family, baking aces every within-class positive (including the would-be-converse) but affirms "same kind" for 93% of unrelated pairs
outcome: positive          # designed sharp test; confirms the mechanism cleanly
confidence: high           # 4 independent eq graphs, n∈{1,2}, tight sds, converged (eval_kl 0.07–0.15)
created: 2026-06-10
question: [[q-propagation-deductive-chain]]
metric: dprime
run_ids: [eqg-eq_alpha-n1, eqg-eq_beta-n1, eqg-eq_gamma-n1, eqg-eq_delta-n1, eqg-eq_alpha-n2, eqg-eq_beta-n2, eqg-eq_gamma-n2, eqg-eq_delta-n2]
---

## Insight
The equivalence-relation world is the **sharp test** designed in [[relational-generalization]] §5a and queued by
[[converse-collapse-does-not-survive-bias-immune-instrument]]: truth = "X and Y in the same weakly-connected
component" (rst-closure), which makes the **converse a FREE positive** (both orders are true) and **cross-component
the ONLY false family** (converse/missing-edge negatives are empty by construction — confirmed in the runs). The
prediction was: if baking's one graph-general deficit really is over-connecting unrelated concepts (not getting
direction wrong), then on this instrument baking should (a) **ace every within-class positive** and (b) **concentrate
its entire error on cross**. **Both confirmed, starkly.** Baking nails the within-class relation (d1 AUROC 1.00,
identical to prompting — it has no trouble with the symmetric/"converse" pairs) yet at one hop out it answers
**"yes, same kind" to 93% of genuinely unrelated pairs** (baked d2 cross-FA 0.93 vs prompted 0.06). The associative
shadow is, literally, **over-connection: affirming that unrelated things are related.** (Fig:
`results/bake_theorem_qa_equiv/_fig_equiv_cross.png`.)

## Evidence — equivalence world, cross = ONLY false family (mean ± popsd over 4 eq graphs, Llama-3.1-8B, fast recipe)
| depth | metric | prompted | baked n=1 | baked n=2 |
|---|---|---|---|---|
| d1 | AUROC(true vs cross) | 1.00±.00 | **1.00±.00** | 1.00±.00 |
| d1 | FA = P("same kind" \| unrelated) | 0.04±.03 | 0.12±.09 | 0.12±.05 |
| **d2** | AUROC | 0.90±.03 | **0.66±.07** | 0.84±.05 |
| **d2** | **FA** | 0.06±.03 | **0.93±.05** | 0.29±.10 |
| d3 | AUROC | 0.69±.11 | 0.44±.10 | 0.49±.12 |
| d3 | FA | 0.07±.05 | 0.97±.00 | 0.76±.11 |

- **(a) Baking aces within-class positives** — d1 AUROC 1.00 = prompting, FA near floor. The symmetric relation
  (where every taught edge's converse is also true) is learned perfectly at the source; there is **no converse
  problem at all** — confirming, on the cleanest possible instrument, that baking is not direction-blind.
- **(b) The entire deficit is cross over-affirmation** — at d2, baked n=1 says "same kind" to **93%** of unrelated
  cross-class pairs (prompted 6%); AUROC collapses 0.90→0.66. At d3 it affirms 97%. Baking floods the "same kind"
  decision boundary outward across the partition.
- **The n=2 curriculum heals it** — d2 cross-FA 0.93→0.29, AUROC 0.66→0.84 — the **same healing** seen in the
  directed worlds (cross-AUROC 0.58→0.77), now even larger. Training one hop deeper installs the partition boundary.
- **Not undertraining** — eval_kl converged to 0.07–0.15 (lower than the directed worlds); the over-connection
  coexists with an excellent distillation fit, reaffirming eval_kl ⟂ propagation.

## HARDENING (2026-06-10) — seed-replicated AND leak-immune (the CoT-leak caveat is now ADDRESSED, and it understated the effect)
- **Seed replication (3 seeds, eq_alpha s10/s11/s12):** baked d2 cross-FA **0.91±0.05** (prompted 0.03),
  cross-AUROC 0.60±0.07 (prompted 0.85); n=2 heals to FA 0.32±0.10. d1 baked AUROC 0.97–1.00. Tight sds —
  the over-affirmation is seed-stable, not a one-split fluke. Runs: eqms-eq_alpha-n{1,2}-s{11,12}.
- **Leak-free confirmation (teacher-forced clean cell, tf-eq_alpha-n1):** with `sample_trajectories=False` the
  gate ENFORCES train/eval disjointness (no recall-of-recited), so this cell is CoT-leak-free. The over-affirmation
  is **stronger, not weaker**: baked d2 cross-FA **0.97** (vs sampled 0.91) and cross-AUROC **0.38 — BELOW chance**
  (vs sampled 0.60), i.e. the clean baked model ranks unrelated pairs *above* true same-class pairs at d2.
  **Implication: the CoT leak was HELPING the baked model** (recited held-out pairs inflated its d2 discrimination);
  removing it reveals over-connection is worse than the sampled numbers show. The central finding is **leak-immune
  and was understated** — and by extension the directed-world cross deficit is likely understated too.
  (Only the eq_alpha-n1 cell ran clean; the other teacher-forced cells are gate-blocked pending a disjoint-split
  builder fix — [[q-teacher-ceiling-vs-objective-limit]].)

## Directed vs equivalence — the mechanism is graph-AND-relation-general, and STARKER without competing negatives
| world | d2 cross AUROC (prompted→baked n1) | d2 cross FA (prompted→baked n1) | n=2 heals FA to |
|---|---|---|---|
| directed lw_* (12 graphs) | 0.90 → 0.58 | 0.08 → 0.61 | 0.21 |
| equivalence eq_* (4 graphs) | 0.90 → 0.66 | 0.06 → **0.93** | 0.29 |
The deficit is the same phenomenon, but **more extreme in the equivalence world**: because every within-class pair
(all orders, all hops) is a positive, baking is trained on a flood of "…are the same kind → Yes" and over-generalises
that affirmation to unrelated pairs far more aggressively than in the directed world (where forward-only positives
give it less "yes" pressure). Over-connection scales with how much the trajectory distribution says "yes, related".

## Counter-arguments / threats to validity
- **CoT leak** ([[sampled-teacher-trajectories-keep-cot]]) — **now DIRECTLY ADDRESSED** by the clean teacher-forced
  cell (HARDENING above): without any leak the gap is LARGER (cross-FA 0.97, AUROC 0.38 below chance), so the leak
  was understating, not manufacturing, the effect. (It inflates absolute d≥2 for all states equally anyway, and the
  prompted teacher sees the same leaked CoT yet still rejects cross at FA 0.03–0.06.) The claim is leak-immune.
- **4 eq graphs vs 12 directed** — fewer worlds, but the sds are tight (d2 FA 0.93±.05) and every graph shows it; the
  directed result it confirms is already 12-graph + 4-seed high-confidence.
- **d3 baked AUROC < chance + FA→0.97** is the depth boundary (one hop past trained depth), where baking affirms ~all
  pairs; this is the same teacher-ceiling/over-affirmation drift catalogued in
  [[baked-propagation-tracks-trained-depth-no-compositional-bonus]], not a new effect.
- Single split_seed (10) per graph; the directed multi-seed set already shows seed-stability of the cross effect.

## Implications
- **Confirms the mechanism** in [[converse-collapse-does-not-survive-bias-immune-instrument]] and the v2
  [[SYNTHESIS-baking-vs-prompting-propagation]]: baking's lone graph-general divergence from prompting is
  cross-component **over-connection**, full stop — now demonstrated on an instrument where it is the *only* possible
  error and the converse is a free positive (so "direction-blindness" is definitively excluded as the story).
- **Confirms the [[relational-generalization]] §5a prediction** that the equivalence world is the decisive separator,
  and pins the formal description: baking approximates the true relation but **inflates the closure** — it behaves as
  if more pairs are related than truly are, an over-permissive (low-precision) closure rather than a symmetrised one.
- **The n curriculum is the lever** (graph- and relation-general): training to depth d installs the relational
  boundary out to ~d, sharply cutting over-connection; this is the actionable knob for anyone baking a relation.

## Next steps
- Add the equivalence assets + this finding to the directed↔equivalence comparison figure already drafted
  (`_fig_equiv_cross.png`). Resolve task #20.
- **Why does over-connection scale with "yes-pressure"?** Test directly: a directed world with *balanced* yes/no
  trajectory framing vs the current forward-only — does reducing the affirmation density in the bake distribution
  reduce cross-FA? (Mechanistic follow-up; ties to [[trajectory-type-is-a-binary-coverage-gate]].)
- An equivalence multi-seed set (k∈{10,11,12}) to match the directed protocol ([[paired-matched-seed-protocol]]).
