#!/usr/bin/env bash
# PreToolUse(Bash) — HARD BLOCK (deny) irreversible loss of run artifacts, committed history /
# or the run-ledger / findings.
#
# Unlike the integrity hooks (which only WARN), this DENIES: these are not methodology judgement
# calls, they are unrecoverable data / history / money. Defense-in-depth.

input="$(cat)"
cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // empty' 2>/dev/null)"
[ -z "$cmd" ] && exit 0
norm="$(printf '%s' "$cmd" | tr '\n\t' '  ' | tr -s ' ')"

deny() {
  jq -n --arg r "$1" '{
    "hookSpecificOutput": {
      "hookEventName": "PreToolUse",
      "permissionDecision": "deny",
      "permissionDecisionReason": $r
    }
  }'
  exit 0
}

# Wholesale deletion of results / the committed brain.
if printf '%s' "$norm" | grep -Eq 'rm +-[a-z]*r[a-z]* +(\./)?results($|[ /])'; then
  deny "BLOCKED: 'rm -rf results' destroys all run artifacts. Delete a single run dir explicitly if you must."
fi
if printf '%s' "$norm" | grep -Eq 'rm +.*(research/run-log\.jsonl|research/findings)'; then
  deny "BLOCKED: that removes the committed run-ledger / findings (the durable research brain). Never delete results or findings."
fi
# History rewrites.
if printf '%s' "$norm" | grep -Eq 'git +reset +--hard'; then
  deny "BLOCKED: 'git reset --hard' discards committed work. Use a soft reset or a new commit."
fi
if printf '%s' "$norm" | grep -Eq 'git +push +.*(--force|-f)( |$)'; then
  deny "BLOCKED: force-push rewrites shared history. Push normally to a research/<slug> branch."
fi

exit 0
