"""One way to open the DuckDB connection every SQL-level test uses.

Haybarn ships DuckDB but not the vgi extension that `ATTACH (TYPE vgi …)`
needs: that comes from Haybarn's `community` extension repository and is cached
under ~/.haybarn. On a machine that already has it — every developer laptop
that ever attached a worker — the attach just works, which hid the step
entirely; a fresh CI runner failed every SQL test with "Extension … vgi …
not found". `INSTALL` is a no-op once the extension is cached.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent


def connect() -> Any:
    """A Haybarn connection with the vgi extension installed and loaded."""
    haybarn = pytest.importorskip("haybarn")
    con = haybarn.connect()
    con.execute("INSTALL vgi FROM community")
    con.execute("LOAD vgi")
    return con


def worker_command(script: str) -> list[str]:
    """argv that runs ``script`` (relative to the repo root) as the worker, with `haybarn` hidden.

    See tests/run_worker.py: the worker must evaluate filters with DuckDB 1.5.5,
    not the Haybarn 2.0 these tests use as the client.
    """
    return [sys.executable, str(ROOT / "tests" / "run_worker.py"), str(ROOT / script)]


def worker_location(script: str) -> str:
    """``worker_command(script)`` as an ATTACH LOCATION string."""
    return " ".join(worker_command(script))
