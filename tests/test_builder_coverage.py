"""Every registered trajectory builder must ship a test that mentions it.

A new builder without a contamination/alignment test is an incomplete change. This fails the
suite if a builder defined under bakery/trajectories/ has no referencing test.
"""

import inspect
from pathlib import Path

import bakery.experiments  # noqa: F401 — trigger experiment + builder registration
from bakery.trajectories.base import get_builder, list_builders


def test_every_builder_has_a_test():
    import bakery.trajectories as pkg
    pkg_dir = Path(pkg.__file__).parent
    test_dir = Path(__file__).parent
    test_text = "\n".join(p.read_text() for p in test_dir.rglob("test_*.py"))

    missing = []
    for name in list_builders():
        cls = get_builder(name)
        mod_file = Path(inspect.getfile(cls))
        if pkg_dir in mod_file.parents and name not in test_text:
            missing.append(name)
    assert not missing, f"Builders without a test: {missing}"
