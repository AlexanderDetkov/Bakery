#!/usr/bin/env python3
"""Opt-in remote compute on vast.ai: run the SAME `run.py` CLI on a rented GPU, sync results
back, and ALWAYS destroy the box.

    python vast/remote.py --experiment bake_squad --model.lora_rank 16 --run_name bake-r16
    python vast/remote.py down           # destroy OUR labelled boxes
    python vast/remote.py reap --yes     # destroy OUR labelled boxes (alias of down)

Safety (defense in depth):
  * The box is destroyed in a `finally` — even if the run errors (the always-destroy guarantee,
    guarded by tests/test_vast_teardown.py).
  * We only ever destroy instances WE created (tracked in vast/.active_instances) — never an
    account-wide mass destroy. The forbid-mass-destroy hook also blocks `vastai destroy instances`.
  * A per-hour price cap (--max-price, default $0.60/hr) refuses pricier offers.

Requires `pip install vastai`, an authenticated account, and an SSH key vast can use.
NOTE: this integration is UNVERIFIED in the initial build — validate against your account
(the SSH host/port parsing + rsync paths) before relying on it for long runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATE = REPO / "vast" / ".active_instances"
LABEL = "bakery-auto"
DEFAULT_GPU = "RTX_3090"


def _vastai(args, check=True):
    return subprocess.run(["vastai", *args], capture_output=True, text=True, check=check)


def _recorded() -> set:
    return set(STATE.read_text().split()) if STATE.exists() else set()


def _record(instance_id):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    ids = _recorded()
    ids.add(str(instance_id))
    STATE.write_text("\n".join(sorted(ids)) + "\n")


def _unrecord(instance_id):
    ids = _recorded()
    ids.discard(str(instance_id))
    STATE.write_text("\n".join(sorted(ids)) + "\n")


def find_cheapest_offer(max_price, gpu=DEFAULT_GPU):
    out = _vastai(["search", "offers", f"dph<{max_price} num_gpus=1 gpu_name={gpu}", "--raw"]).stdout
    offers = json.loads(out)
    if not offers:
        raise SystemExit(f"No vast offer under ${max_price}/hr for {gpu}.")
    return sorted(offers, key=lambda o: o["dph_total"])[0]


def create_instance(offer, max_price):
    if offer["dph_total"] > max_price:
        raise SystemExit(f"Cheapest offer ${offer['dph_total']}/hr exceeds budget ${max_price}/hr.")
    onstart = (REPO / "vast" / "onstart.sh").read_text()
    out = _vastai([
        "create", "instance", str(offer["id"]), "--image", "pytorch/pytorch:latest",
        "--disk", "40", "--label", LABEL, "--onstart-cmd", onstart, "--raw",
    ]).stdout
    iid = json.loads(out)["new_contract"]
    _record(iid)
    return iid


def destroy_instance(instance_id):
    try:
        _vastai(["destroy", "instance", str(instance_id)], check=False)
    finally:
        _unrecord(instance_id)


def _ssh_endpoint(instance_id, timeout=600):
    """Poll until the instance reports an SSH host+port."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = _vastai(["show", "instance", str(instance_id), "--raw"], check=False).stdout
        try:
            info = json.loads(out)
        except Exception:
            info = {}
        host, port = info.get("ssh_host"), info.get("ssh_port")
        if host and port and info.get("actual_status") == "running":
            return host, int(port)
        time.sleep(10)
    raise SystemExit(f"Instance {instance_id} did not become reachable within {timeout}s.")


def _ssh(host, port, command):
    subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-p", str(port), f"root@{host}", command],
        check=True,
    )


def _rsync(src, dst):
    subprocess.run(
        ["rsync", "-az", "--exclude", "results", "--exclude", ".git",
         "--exclude", "trajectory_cache", "--exclude", "__pycache__", src, dst],
        check=True,
    )


def _sync_up(host, port):
    _rsync(f"{REPO}/", f"root@{host}:~/Bakery/")  # noqa — rsync over the configured ssh


def _exec(host, port, run_args):
    _ssh(host, port, "cd ~/Bakery && pip install -e . >/dev/null && "
                     f"python run.py {' '.join(run_args)} --backend vast")


def _sync_down(host, port):
    subprocess.run(["rsync", "-az", f"root@{host}:~/Bakery/results/", f"{REPO}/results/"], check=True)


def remote_run(run_args, max_price=0.60, gpu=DEFAULT_GPU):
    offer = find_cheapest_offer(max_price, gpu=gpu)
    instance_id = create_instance(offer, max_price)
    print(f"[vast] created instance {instance_id} (${offer['dph_total']}/hr)")
    try:
        host, port = _ssh_endpoint(instance_id)
        _sync_up(host, port)
        _exec(host, port, run_args)
        _sync_down(host, port)
    finally:
        print(f"[vast] destroying instance {instance_id}")
        destroy_instance(instance_id)   # ALWAYS — even on error


def down():
    for iid in sorted(_recorded()):
        print(f"[vast] destroying our box {iid}")
        destroy_instance(iid)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("down", "reap"):
        down()
        return
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--max-price", type=float, default=0.60)
    ap.add_argument("--gpu", default=DEFAULT_GPU)
    known, run_args = ap.parse_known_args(argv)
    if not run_args:
        raise SystemExit("Pass run.py args (e.g. --experiment bake_squad ...) or 'down'.")
    remote_run(run_args, max_price=known.max_price, gpu=known.gpu)


if __name__ == "__main__":
    main()
