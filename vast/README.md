# vast/ — opt-in remote compute

Run the **same** `run.py` CLI on a rented vast.ai GPU for big-model bakes (e.g. Llama-3.1-8B), then
sync `results/` back. Local GPU is the default; this is opt-in.

```bash
python vast/remote.py --experiment bake_squad --model.lora_rank 16 --run_name bake-r16
python vast/remote.py down        # destroy the boxes WE created
```

## Safety guarantees
- **Always destroy.** The box is torn down in a `finally` even if the run errors
  (`tests/test_vast_teardown.py` guards this). An on-box watchdog (`onstart.sh`) also self-destructs
  after `BAKERY_MAX_HOURS` (default 6h) if the local side dies first.
- **Never account-wide.** We only destroy instances we created (tracked in `vast/.active_instances`).
  The `forbid-mass-destroy` hook blocks `vastai destroy instances` and `remote.py kill-all`.
- **Budget cap.** `--max-price` (default $0.60/hr) refuses pricier offers.

## Setup (required before first use)
1. `pip install vastai` and authenticate (`vastai set api-key ...`).
2. Add an SSH key to your vast account.
3. The ledger row for a remote run carries `"backend":"vast"`.

> **Unverified in the initial build.** The SSH host/port parsing and rsync paths depend on your
> account/image; validate with one small run before relying on it for long jobs. Treat `remote.py`
> as a reviewed scaffold, not battle-tested infrastructure.
