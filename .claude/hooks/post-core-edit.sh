#!/usr/bin/env bash
# PostToolUse(Edit|Write) — reminder only (non-blocking) after edits to package code or research/.

input="$(cat)"
file="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null)"
[ -z "$file" ] && exit 0

msg=""
case "$file" in
  */research/*|research/*)
    msg="Edited a research/ page. Keep research/index.md and research/log.md current (or run /lint-research before the cycle ends)."
    ;;
  */bakery/*|bakery/*)
    msg="Edited package code ($(basename "$file")). Run 'make test-fast' before recording a finding or committing — the test suite is the real invariant gate."
    ;;
esac
[ -z "$msg" ] && exit 0

jq -n --arg ctx "$msg" '{"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": $ctx}}'
exit 0
