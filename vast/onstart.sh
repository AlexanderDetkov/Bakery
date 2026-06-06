#!/usr/bin/env bash
# Runs on the rented box at boot. Installs deps and arms a self-destruct watchdog so a box can
# never run away with the budget if the local side dies before teardown (defense in depth).
set -e

MAX_HOURS="${BAKERY_MAX_HOURS:-6}"

# Self-destruct watchdog: power the box off after MAX_HOURS regardless of what else happens.
nohup bash -c "sleep $((MAX_HOURS * 3600)); shutdown -h now" >/tmp/watchdog.log 2>&1 &

pip install -U pip >/dev/null 2>&1 || true
# The repo is rsynced in by vast/remote.py after boot; deps are installed there (pip install -e .).
echo "onstart complete; watchdog armed for ${MAX_HOURS}h"
