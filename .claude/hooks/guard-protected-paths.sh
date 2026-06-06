#!/usr/bin/env bash
# PreToolUse(Edit|Write) — WARN (never block) when an edit targets a research-integrity file.
#
# A *nudge* layer only: always exits 0 with no permissionDecision, so the edit proceeds. The
# runtime validation gate + `make test` are the real backstop. (Warn, don't block.)
#
# Protected: the safety kernel (gate + frozen data), the KL primitive, the results contract,
# the invariant tests, CLAUDE.md, and the human-owned research agenda.

input="$(cat)"
file="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null)"
[ -z "$file" ] && exit 0

if printf '%s' "$file" | grep -Eq '(bakery/trajectories/base\.py|bakery/trajectories/encoding\.py|bakery/objectives/base\.py|bakery/results\.py|tests/_invariant_kit\.py|tests/test_trajectory_contract\.py|tests/test_objective_alignment\.py|tests/test_.*coverage\.py|CLAUDE\.md|research/agenda\.md)$'; then
  read -r -d '' ctx <<'EOF'
This file encodes a research invariant — the validation gate (frozen TrajectoryDataset + the
mask-alignment / context-disjointness / checkpoint checks), the single KL primitive, the results
contract, an invariant test, or the authoritative methodology. Do NOT weaken the gate, relax a
check, switch the KL to truncated logits, or "fix the test instead of the code". If you genuinely
believe an invariant is wrong, STOP and write a research/decisions/ note for human review rather
than editing it here. (Warning only — the edit is NOT blocked.)
EOF
  jq -n --arg ctx "$ctx" '{
    "systemMessage": "⚠ Editing a research-integrity file — see additionalContext before changing invariants.",
    "hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": $ctx}
  }'
fi
exit 0
