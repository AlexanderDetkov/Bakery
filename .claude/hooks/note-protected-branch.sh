#!/usr/bin/env bash
# PreToolUse(Bash) — WARN (never block) when committing on the protected branch.
# Research work belongs on a research/<slug> branch (CLAUDE.md). Never denies.

input="$(cat)"
cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // empty' 2>/dev/null)"
[ -z "$cmd" ] && exit 0

case "$cmd" in
  *"git commit"*)
    branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
    if [ "$branch" = "main" ] || [ "$branch" = "master" ]; then
      jq -n --arg b "$branch" '{
        "systemMessage": ("⚠ git commit on " + $b + " — research work belongs on a research/* branch."),
        "hookSpecificOutput": {
          "hookEventName": "PreToolUse",
          "additionalContext": ("You are about to commit on the protected branch \"" + $b + "\". Per CLAUDE.md the autonomous loop must not commit here — create/switch to a research/<slug> branch first. (Warning only — not blocked.)")
        }
      }'
    fi
    ;;
esac
exit 0
