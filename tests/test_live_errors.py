"""The real API's refusals, as the worker reports them. Opt in with `pytest -m live`.

These need no Admin key — they exist precisely because credentials *without*
admin access are the common case, and every failure mode below was first
observed against the live API with such a key. Each test runs only when its
key is supplied:

    ANTHROPIC_WORKSPACE_KEY   a key bound to one workspace (sk-ant-api03-...)
    ANTHROPIC_NON_ADMIN_KEY   an organization-wide personal key whose user is not an admin

The invalid-key test needs nothing but the network.

What is checked is the contract a user depends on: the API's own message and
request ID survive into the DuckDB error, the worker's explanation is added,
and nothing is ever answered with an empty result instead of an error.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.duckdb_session import connect

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_KEY = os.environ.get("ANTHROPIC_WORKSPACE_KEY", "")
NON_ADMIN_KEY = os.environ.get("ANTHROPIC_NON_ADMIN_KEY", "")


def _connect(key: str) -> Any:
    con = connect()
    con.execute(f"ATTACH 'anthropic' (TYPE vgi, LOCATION '{sys.executable} {ROOT / 'anthropic_worker.py'}')")
    quoted = key.replace("'", "''")
    con.execute(f"CREATE SECRET (TYPE anthropic_admin, api_key '{quoted}')")
    return con


def _error(con: Any, sql: str) -> str:
    with pytest.raises(Exception) as info:
        con.execute(sql).fetchall()
    message = str(info.value)
    # Observed live: a 403 carries a request-id header and a 401 does not. When
    # there is one, it must reach the user — it is what support asks for.
    if "Anthropic API 403" in message:
        assert "request-id req_" in message, "the API's request ID must reach the user"
    else:
        assert "request-id" not in message, "no request ID was sent; none may be invented"
    return message


def test_an_invalid_key_is_a_401_naming_the_credentials_that_work() -> None:
    con = _connect("sk-ant-admin01-this-key-does-not-exist")
    message = _error(con, "SELECT * FROM anthropic.organization")
    assert "Anthropic API 401" in message
    assert "organization-scoped" in message


@pytest.mark.skipif(not WORKSPACE_KEY, reason="set ANTHROPIC_WORKSPACE_KEY")
class TestWorkspaceScopedKey:
    def test_reads_the_organization(self) -> None:
        assert _connect(WORKSPACE_KEY).execute("SELECT count(*) FROM anthropic.organization").fetchone() == (
            1,
        )

    @pytest.mark.parametrize("table", ["users", "workspaces", "api_keys"])
    def test_is_refused_everywhere_else_with_an_explanation(self, table: str) -> None:
        message = _error(_connect(WORKSPACE_KEY), f"SELECT * FROM anthropic.{table}")
        assert "Anthropic API 401" in message
        assert "organization-scoped API key" in message  # the API's own words
        assert "workspace-scoped key can read `organization` and nothing else" in message  # ours

    def test_a_report_is_refused_too(self) -> None:
        message = _error(_connect(WORKSPACE_KEY), "SELECT * FROM anthropic.cost_report(current_date - 1)")
        assert "Anthropic API 401" in message


@pytest.mark.skipif(not NON_ADMIN_KEY, reason="set ANTHROPIC_NON_ADMIN_KEY")
class TestNonAdminKey:
    def test_reads_the_organization(self) -> None:
        assert _connect(NON_ADMIN_KEY).execute("SELECT count(*) FROM anthropic.organization").fetchone() == (
            1,
        )

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM anthropic.users",
            "SELECT * FROM anthropic.workspace_members",
            "SELECT * FROM anthropic.messages_usage(current_date - 1)",
            "SELECT * FROM anthropic.claude_code_usage(current_date - 1)",
        ],
    )
    def test_is_refused_with_the_role_explained(self, sql: str) -> None:
        message = _error(_connect(NON_ADMIN_KEY), sql)
        assert "Anthropic API 403" in message
        assert "Missing permissions" in message  # the API's own words
        assert "admin role" in message  # ours

    def test_a_lookup_raises_rather_than_returning_no_row(self) -> None:
        """Lookups map 404 to "no row"; a 403 must not be mistaken for one.

        Swallowed, it would make every ID look like a departed member — an
        empty answer that reads as real.
        """
        message = _error(_connect(NON_ADMIN_KEY), "SELECT * FROM anthropic.user_by_id('user_anything')")
        assert "Anthropic API 403" in message
