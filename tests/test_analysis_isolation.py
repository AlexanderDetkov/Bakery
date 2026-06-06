"""Analysis must NEVER import the training stack — so an analysis turn can't drag torch /
transformers / peft (and their import-time side effects) into context. Checked in a fresh
subprocess so other tests' imports don't pollute the result.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_analysis_load_does_not_import_training_stack():
    code = (
        "import analysis.load, analysis.aggregate, sys; "
        "bad=[m for m in ('torch','transformers','peft') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": __import__("os").environ.get("PATH", "")},
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"analysis imported the training stack: {out.stdout.strip()}"
