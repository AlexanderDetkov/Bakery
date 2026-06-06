"""Baking objectives.

Auto-discovery: importing this package imports every sibling module so each `@register_objective`
runs. Adding an objective is just adding a `<name>.py` file here — no import list to edit.
"""

import importlib
import pkgutil
from pathlib import Path

__all__ = []

for _mod in pkgutil.iter_modules([str(Path(__file__).parent)]):
    if not _mod.name.startswith("_") and _mod.name != "base":
        importlib.import_module(f"{__name__}.{_mod.name}")
        __all__.append(_mod.name)
