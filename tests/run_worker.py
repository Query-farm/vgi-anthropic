"""Run a worker script with the `haybarn` module hidden: ``python tests/run_worker.py SCRIPT [ARGS...]``.

The dev environment carries Haybarn 2.0 as the SQL client the tests drive (its
vgi extension speaks protocol 3.2), and the worker's runtime dependency, stock
DuckDB 1.5.5, as vgi-python's filter evaluator. vgi-python prefers `haybarn`
when both import, and its vgi.duckdb.standard.v1 evaluator refuses Haybarn 2.0,
so a worker launched with the dev interpreter as-is would fail every pushed-down
filter. A deployed worker has only DuckDB installed and never sees this.
"""

from __future__ import annotations

import runpy
import sys

sys.modules["haybarn"] = None  # type: ignore[assignment]  # makes `import haybarn` raise ImportError
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
