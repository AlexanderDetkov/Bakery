#!/usr/bin/env python3
"""Opt-in remote compute on vast.ai: run the SAME `run.py` CLI on a rented GPU, sync results
back, and ALWAYS destroy the box.

    python vast/remote.py --experiment bake_squad --model.lora_rank 16 --run_name bake-r16
    python vast/remote.py down            # destroy OUR labelled boxes
    python vast/remote.py reap --yes      # alias of down

Flow: register this box's SSH key on the account -> rent the cheapest comfortable GPU
(--ssh --direct) -> wait for SSH -> tar the repo up over SSH -> write your HF token to the
box's cache (so gated models download; never stored in vast config) -> `pip install -e . &&
python run.py <args> --backend vast` -> tar results back -> destroy the box.

Safety (defense in depth):
  * The box is destroyed in a `finally` — even if the run errors (tests/test_vast_teardown.py).
  * On-box watchdog (onstart.sh) self-destructs after BAKERY_MAX_HOURS (default 6h) if the local
    side dies first.
  * We only destroy instances WE created (vast/.active_instances). The forbid-mass-destroy hook
    blocks account-wide `vastai destroy instances`.
  * A per-hour price cap (--max-price, default $0.80/hr) and a GPU-arch/VRAM filter.

Requires `vastai` authenticated and an SSH key. Verified against vast CLI 1.0.13.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATE = REPO / "vast" / ".active_instances"
LABEL = "bakery-auto"
PUBKEY = os.path.expanduser("~/.ssh/id_rsa.pub")
PRIVKEY = os.path.expanduser("~/.ssh/id_rsa")
HF_TOKEN_FILE = os.path.expanduser("~/.cache/huggingface/token")
IMAGE = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"

# GPUs with strong bf16 + enough VRAM for an 8B bake. Names are substring-matched against the
# offer's gpu_name. Older/weak-bf16 cards (RTX 8000/Titan/V100/...) are excluded by absence here.
GOOD_GPUS = ("A40", "A100", "A6000", "L40", "L40S", "RTX 4090", "RTX 6000 Ada", "H100", "H200", "RTX 5090")
MIN_RAM_MB = 40000

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-i", PRIVKEY,
]
SSH_OPTS_STR = " ".join(shlex.quote(x) for x in SSH_OPTS)


# ----- vast CLI helpers --------------------------------------------------------------

def _vastai(args, check=True):
    return subprocess.run(["vastai", *args], capture_output=True, text=True, check=check)


def _recorded() -> set:
    return set(STATE.read_text().split()) if STATE.exists() else set()


def _record(iid):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    ids = _recorded(); ids.add(str(iid)); STATE.write_text("\n".join(sorted(ids)) + "\n")


def _unrecord(iid):
    ids = _recorded(); ids.discard(str(iid)); STATE.write_text("\n".join(sorted(ids)) + "\n")


def ensure_account_key():
    """Register this box's public key on the vast account if it isn't already."""
    pub = Path(PUBKEY).read_text().strip()
    token = pub.split()[1][:40]  # the key body prefix, enough to detect duplicates
    have = _vastai(["show", "ssh-keys"], check=False).stdout
    if token in have:
        return
    print("[vast] registering this box's SSH key on the account")
    _vastai(["create", "ssh-key", pub, "-y"], check=False)


def find_offer(max_price):
    out = _vastai(["search", "offers",
                   f"num_gpus=1 rentable=true verified=true dph<{max_price}", "--raw"]).stdout
    offers = json.loads(out)

    def good(o):
        return (o.get("gpu_ram", 0) >= MIN_RAM_MB
                and o.get("disk_space", 0) >= 60
                and any(g in o.get("gpu_name", "") for g in GOOD_GPUS))

    cands = sorted((o for o in offers if good(o)), key=lambda o: o["dph_total"])
    if not cands:
        raise SystemExit(f"No suitable vast offer (>= {MIN_RAM_MB}MB, good arch) under ${max_price}/hr.")
    return cands[0]


def create_instance(offer):
    onstart = (REPO / "vast" / "onstart.sh").read_text()
    out = _vastai([
        "create", "instance", str(offer["id"]), "--image", IMAGE, "--disk", "60",
        "--ssh", "--direct", "--label", LABEL, "--onstart-cmd", onstart, "--raw",
    ]).stdout
    iid = json.loads(out)["new_contract"]
    _record(iid)
    return iid


def _attach_key(iid):
    _vastai(["attach", "ssh", str(iid), Path(PUBKEY).read_text().strip()], check=False)


def destroy_instance(iid):
    try:
        # `vastai destroy instance` prompts [y/N]; answer it non-interactively, else the box leaks.
        subprocess.run(f"yes | vastai destroy instance {int(iid)}", shell=True,
                       capture_output=True, text=True)
    finally:
        _unrecord(iid)


# ----- SSH / transfer ----------------------------------------------------------------

def _ssh_endpoint(iid, timeout=900):
    """Wait until the instance is running, has an ssh-url, and accepts SSH; return (host, port)."""
    deadline = time.time() + timeout
    host = port = None
    while time.time() < deadline:
        info = _vastai(["show", "instance", str(iid), "--raw"], check=False).stdout
        try:
            status = json.loads(info).get("actual_status")
        except Exception:
            status = None
        if status == "running":
            url = _vastai(["ssh-url", str(iid)], check=False).stdout.strip()
            m = re.match(r"ssh://[^@]+@([^:]+):(\d+)", url)
            if m:
                host, port = m.group(1), int(m.group(2))
                probe = subprocess.run(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", "true"],
                                       capture_output=True)
                if probe.returncode == 0:
                    return host, port
        time.sleep(15)
    raise SystemExit(f"Instance {iid} did not become SSH-reachable within {timeout}s.")


def _upload(host, port):
    cmd = (
        f"tar czf - -C {shlex.quote(str(REPO))} "
        f"--exclude=./results --exclude=./.git --exclude=./trajectory_cache "
        f"--exclude='*/__pycache__' --exclude='*.pyc' . "
        f"| ssh {SSH_OPTS_STR} -p {port} root@{host} 'mkdir -p ~/Bakery && tar xzf - -C ~/Bakery'"
    )
    subprocess.run(cmd, shell=True, check=True)
    # Confirm the repo actually landed (catch a silent transfer failure before we waste GPU time).
    chk = subprocess.run(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}",
                          "test -f ~/Bakery/run.py && echo UPLOAD_OK"], capture_output=True, text=True)
    if "UPLOAD_OK" not in chk.stdout:
        raise SystemExit("Upload failed: ~/Bakery/run.py not found on the box.")


def _write_hf_token(host, port):
    if not os.path.exists(HF_TOKEN_FILE):
        return
    token = Path(HF_TOKEN_FILE).read_bytes()
    subprocess.run(
        ["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}",
         "mkdir -p ~/.cache/huggingface && cat > ~/.cache/huggingface/token"],
        input=token, check=True,
    )


def _exec(host, port, run_args):
    # Point the run-ledger into results/ so the row syncs back with the artifacts (the box's own
    # research/run-log.jsonl would be destroyed with the box). Merge it into the committed ledger locally.
    remote = ("set -e; cd ~/Bakery && pip install -e . -q && "
              "export BAKERY_RUN_LOG=$HOME/Bakery/results/run-log.jsonl && "
              "python run.py " + " ".join(shlex.quote(a) for a in run_args) + " --backend vast")
    # Pass the whole script as ONE argument — ssh space-joins multiple command args, which would
    # split `bash -lc <script>` and run `bash -lc set` (the stray env dump we saw). One arg => the
    # remote shell runs the script intact.
    subprocess.run(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", remote], check=True)


def _download(host, port):
    cmd = (f"ssh {SSH_OPTS_STR} -p {port} root@{host} 'cd ~/Bakery && tar czf - results 2>/dev/null' "
           f"| tar xzf - -C {shlex.quote(str(REPO))}")
    subprocess.run(cmd, shell=True, check=False)  # results may be absent if the run failed early


# ----- orchestration -----------------------------------------------------------------

def remote_run(run_args, max_price=0.80):
    ensure_account_key()
    offer = find_offer(max_price)
    print(f"[vast] offer {offer['id']}: {offer['gpu_name']} {offer['gpu_ram']}MB ${offer['dph_total']:.3f}/hr")
    iid = create_instance(offer)
    print(f"[vast] created instance {iid}")
    try:
        _attach_key(iid)
        host, port = _ssh_endpoint(iid)
        print(f"[vast] ssh root@{host}:{port} — uploading repo")
        _upload(host, port)
        _write_hf_token(host, port)
        print("[vast] running on the box (pip install + run.py) ...")
        _exec(host, port, run_args)
        print("[vast] syncing results back")
        _download(host, port)
    finally:
        print(f"[vast] destroying instance {iid}")
        destroy_instance(iid)   # ALWAYS — even on error


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
    ap.add_argument("--max-price", type=float, default=0.80)
    known, run_args = ap.parse_known_args(argv)
    if not run_args:
        raise SystemExit("Pass run.py args (e.g. --experiment bake_squad ...) or 'down'.")
    remote_run(run_args, max_price=known.max_price)


if __name__ == "__main__":
    main()
