"""The worker served over HTTP (`serve.py`), attached by URL.

Stdio is what `tests/test_end_to_end.py` exercises; this is the other transport,
and the one where credentials matter most. One server answers many clients, so
each client's own DuckDB secret must be what authorizes its queries — the
reason the worker has no environment-variable fallback. These tests run two
clients against one server and check that a client without a usable secret
gets nothing, whatever the other client holds.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.fake_admin_api import ADMIN_KEY, FakeAdminApi

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """`serve.py` on a free port, talking to the fake Admin API."""
    fake = FakeAdminApi().start()
    port_file = tmp_path_factory.mktemp("serve") / "port"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "serve.py"),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--port-file",
            str(port_file),
            "-q",
        ],
        env={**os.environ, "VGI_ANTHROPIC_BASE_URL": fake.url},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 60
        while not (port_file.exists() and port_file.read_text().strip()):
            if proc.poll() is not None or time.monotonic() > deadline:
                pytest.fail("serve.py did not start")
            time.sleep(0.1)
        yield f"http://127.0.0.1:{port_file.read_text().strip()}"
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        fake.stop()


def _client(url: str, key: str | None = None) -> Any:
    haybarn = pytest.importorskip("haybarn")
    con = haybarn.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"ATTACH 'anthropic' (TYPE vgi, LOCATION '{url}')")
    if key is not None:
        con.execute(f"CREATE SECRET (TYPE anthropic_admin, api_key '{key}')")
    return con


@pytest.fixture(scope="module")
def con(server: str) -> Iterator[Any]:
    connection = _client(server, ADMIN_KEY)
    yield connection
    connection.close()


class TestOverHttp:
    def test_table(self, con: Any) -> None:
        assert con.execute("SELECT name FROM anthropic.organization").fetchall() == [("Acme Robotics",)]

    def test_pushdown(self, con: Any) -> None:
        rows = con.execute("SELECT id FROM anthropic.users WHERE email = 'person3@acme.test'").fetchall()
        assert rows == [("user_03",)]

    def test_report_with_inferred_grouping(self, con: Any) -> None:
        rows = con.execute(
            "SELECT model, sum(output_tokens) FROM anthropic.messages_usage('2026-09-08', "
            "ending_at => '2026-09-09') GROUP BY 1 ORDER BY 1"
        ).fetchall()
        assert rows == [("claude-opus-5-5", 120), ("claude-sonnet-5-5", 37)]

    def test_walk_across_ticks(self, con: Any) -> None:
        """Scan state round-trips through the HTTP transport between pages."""
        count = con.execute(
            "SELECT count(*) FROM anthropic.claude_code_usage('2026-09-01', ending_date => '2026-09-03')"
        ).fetchone()
        assert count == (9,)

    def test_lateral_lookup(self, con: Any) -> None:
        rows = con.execute(
            "SELECT k.id, u.email FROM anthropic.api_keys k, LATERAL anthropic.user_by_id(k.created_by_id) u "
            "ORDER BY 1"
        ).fetchall()
        assert rows == [("apikey_01", "person1@acme.test"), ("apikey_03", "person2@acme.test")]


class TestCredentialIsolation:
    """One server, several clients: each is authorized by its own secret only."""

    def test_a_client_without_a_secret_gets_nothing(self, server: str, con: Any) -> None:
        con.execute("SELECT count(*) FROM anthropic.users").fetchall()  # another client is authorized
        anonymous = _client(server)
        with pytest.raises(Exception, match="no 'anthropic_admin' secret"):
            anonymous.execute("SELECT * FROM anthropic.users").fetchall()

    def test_a_client_with_a_bad_key_is_refused(self, server: str, con: Any) -> None:
        con.execute("SELECT count(*) FROM anthropic.users").fetchall()
        intruder = _client(server, "sk-ant-admin01-not-this-org")
        with pytest.raises(Exception, match="401"):
            intruder.execute("SELECT * FROM anthropic.users").fetchall()

    def test_the_authorized_client_still_works_afterwards(self, con: Any) -> None:
        assert con.execute("SELECT count(*) FROM anthropic.users").fetchone() == (5,)
