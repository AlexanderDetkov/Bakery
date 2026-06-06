"""Experiment declarations.

Auto-discovery: importing this package imports every sibling module so each module's
`register_experiment(...)` call runs. Adding an experiment is just adding a `<name>.py`
file here — no import list to edit.
"""

import importlib
import pkgutil
from pathlib import Path

__all__ = []

for _mod in pkgutil.iter_modules([str(Path(__file__).parent)]):
    if not _mod.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_mod.name}")
        __all__.append(_mod.name)
