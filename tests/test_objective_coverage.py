"""Every registered objective must ship a test that mentions it.

An objective is exactly where a subtle KL-misalignment or a missing adapter toggle could
silently invalidate results, so it cannot ship green without a test. (Mirrors the
builder-coverage rule.)
"""

import inspect
from pathlib import Path

import bakery.objectives  # noqa: F401 — trigger registration
from bakery.objectives.base import _OBJECTIVES, list_objectives


def test_every_objective_has_a_test():
    import bakery.objectives as pkg
    pkg_dir = Path(pkg.__file__).parent
    test_dir = Path(__file__).parent
    test_text = "\n".join(p.read_text() for p in test_dir.rglob("test_*.py"))

    missing = []
    for name in list_objectives():
        cls = _OBJECTIVES[name]
        mod_file = Path(inspect.getfile(cls))
        if pkg_dir in mod_file.parents and name not in test_text:
            missing.append(name)
    assert not missing, f"Objectives without a test (mention the name in a tests/ file): {missing}"
