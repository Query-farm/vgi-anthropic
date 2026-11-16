#!/usr/bin/env python3
"""The local fake Anthropic Admin API, served for test/sql/*.test.

Serves :mod:`tests.fake_admin_api` (the same fake tests/test_end_to_end.py and
scripts/lint-behavioural.sh use) on a free port and prints ``PORT:<n>`` once
listening, as ci/run-integration.sh expects. The fake keeps its fixed clock
(``NOW`` = 2026-09-10 UTC), so every report window in the SQL suite has exact,
deterministic answers. Stdlib only: it runs under ``uv run --no-project``.

Usage::

    python3 test/mock_api/server.py            # prints PORT:<n> once listening
"""

from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.fake_admin_api import FakeAdminApi  # noqa: E402


def main() -> None:
    """Start the fake and print ``PORT:<n>`` once it is listening."""
    server = FakeAdminApi().start()
    print(f"PORT:{server.url.rsplit(':', 1)[1]}", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
