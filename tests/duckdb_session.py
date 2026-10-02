"""One way to open the DuckDB connection every SQL-level test uses.

Haybarn ships DuckDB but not the vgi extension that `ATTACH (TYPE vgi …)`
needs: that comes from Haybarn's `community` extension repository and is cached
under ~/.haybarn. On a machine that already has it — every developer laptop
that ever attached a worker — the attach just works, which hid the step
entirely; a fresh CI runner failed every SQL test with "Extension … vgi …
not found". `INSTALL` is a no-op once the extension is cached.
"""

from __future__ import annotations

from typing import Any

import pytest


def connect() -> Any:
    """A Haybarn connection with the vgi extension installed and loaded."""
    haybarn = pytest.importorskip("haybarn")
    con = haybarn.connect()
    con.execute("INSTALL vgi FROM community")
    con.execute("LOAD vgi")
    return con
