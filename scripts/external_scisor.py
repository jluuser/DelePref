"""Optional loader for the external SCISOR baseline.

The SCISOR implementation is intentionally not bundled with DelePref. Set
``DELEPREF_SCISOR_ROOT`` to an external SCISOR checkout when running an
evaluation phase that needs the baseline.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


def load_shortening_scud() -> Any:
    """Return the external SCISOR ``ShorteningSCUD`` class."""

    configured = os.environ.get("DELEPREF_SCISOR_ROOT", "").strip()
    if configured:
        root = Path(configured).expanduser().resolve()
        candidates = [root.parent] if root.name == "SCISOR" else [root]
        for candidate in candidates:
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
    else:
        spec = importlib.util.find_spec("SCISOR")
        local_root = Path(__file__).resolve().parents[1] / "SCISOR"
        locations = [Path(path).resolve() for path in (spec.submodule_search_locations or [])] if spec else []
        if any(path == local_root for path in locations):
            raise RuntimeError(
                "SCISOR is present only as a server-local copy. Set "
                "DELEPREF_SCISOR_ROOT explicitly to opt into that external "
                "baseline; it is never used implicitly by the public code."
            )

    try:
        module = importlib.import_module("SCISOR.shortening_scud")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SCISOR is an optional external baseline and is not bundled with "
            "DelePref. Set DELEPREF_SCISOR_ROOT to an external SCISOR checkout "
            "or install the baseline before running this evaluation phase."
        ) from exc
    try:
        return module.ShorteningSCUD
    except AttributeError as exc:
        raise RuntimeError("external SCISOR package has no ShorteningSCUD class") from exc
