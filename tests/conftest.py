"""Shared test fixtures + isolation.

Crucially, point the run-ledger at a throwaway temp file so the test suite never
touches the committed research/run-log.jsonl.
"""

import os
import sys
import tempfile
from pathlib import Path

# Repo root on sys.path (so `import bakery` / `from tests import _invariant_kit` work).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Isolate the run-ledger. results.run_log_path() reads this env var at call time.
os.environ.setdefault(
    "BAKERY_RUN_LOG", str(Path(tempfile.gettempdir()) / "bakery_test_run_log.jsonl")
)
