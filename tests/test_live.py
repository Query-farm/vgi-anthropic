"""Against the real Admin API. Opt in with `pytest -m live` and a real key:

    ANTHROPIC_ADMIN_API_KEY=sk-ant-admin01-... uv run pytest -m live

The key is read by the *test* and handed to DuckDB as a secret, exactly as a
user would; the worker itself never reads the environment for credentials.

Every assertion here holds for any organization — one that has never sent a
request included — so nothing assumes usage, workspaces or keys exist. What
these catch is drift: a renamed field, a changed page shape, a new enum value.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parent.parent
KEY = os.environ.get("ANTHROPIC_ADMIN_API_KEY", "")


@pytest.fixture(scope="module")
def con() -> Iterator[Any]:
    if not KEY:
        pytest.skip("set ANTHROPIC_ADMIN_API_KEY to run the live tests")
    haybarn = pytest.importorskip("haybarn")
    connection = haybarn.connect()
    connection.execute("SET TimeZone = 'UTC'")
    connection.execute(
        f"ATTACH 'anthropic' (TYPE vgi, LOCATION '{sys.executable} {ROOT / 'anthropic_worker.py'}')"
    )
    # CREATE SECRET takes no bound parameters, so the key is quoted inline.
    quoted = KEY.replace("'", "''")
    connection.execute(f"CREATE SECRET live (TYPE anthropic_admin, api_key '{quoted}')")
    # Twelve failures that all say "this key is not an admin key" describe the
    # key, not the worker. Probe once and skip with that reason instead; the
    # refusals themselves are covered by tests/test_live_errors.py.
    try:
        connection.execute("SELECT count(*) FROM anthropic.users").fetchall()
    except Exception as exc:
        if "Anthropic API 401" in str(exc) or "Anthropic API 403" in str(exc):
            connection.close()
            pytest.skip("ANTHROPIC_ADMIN_API_KEY does not have Admin API access to organization data")
        raise
    yield connection
    connection.close()


def test_one_organization(con: Any) -> None:
    assert con.execute("SELECT count(*) FROM anthropic.organization").fetchone() == (1,)


def test_there_is_an_admin(con: Any) -> None:
    (admins,) = con.execute(
        "SELECT count(*) FROM anthropic.users WHERE role IN ('admin', 'owner', 'primary_owner')"
    ).fetchone()
    assert admins > 0


@pytest.mark.parametrize(
    "table",
    ["organization", "users", "invites", "workspaces", "workspace_members", "api_keys", "rate_limits"],
)
def test_every_table_materializes(con: Any, table: str) -> None:
    con.execute(f"SELECT * FROM anthropic.{table} LIMIT 50").fetchall()


def test_every_column_of_the_usage_report_materializes(con: Any) -> None:
    con.execute("SELECT * FROM anthropic.messages_usage(current_date - INTERVAL 2 DAY)").fetchall()


def test_cost_is_decimal(con: Any) -> None:
    rows = con.execute(
        "SELECT DISTINCT typeof(amount_usd), currency "
        "FROM anthropic.cost_report(current_date - INTERVAL 7 DAY)"
    ).fetchall()
    assert all(kind == "DECIMAL(38,14)" and currency == "USD" for kind, currency in rows)


def test_claude_code_report_materializes(con: Any) -> None:
    con.execute("SELECT * FROM anthropic.claude_code_usage(current_date - INTERVAL 1 DAY)").fetchall()


def test_grouped_and_ungrouped_usage_agree(con: Any) -> None:
    window = "current_date - INTERVAL 3 DAY, ending_at => current_date"
    (total,) = con.execute(f"SELECT sum(output_tokens) FROM anthropic.messages_usage({window})").fetchone()
    (by_model,) = con.execute(
        f"SELECT sum(t) FROM (SELECT model, sum(output_tokens) AS t "
        f"FROM anthropic.messages_usage({window}) GROUP BY model)"
    ).fetchone()
    assert total == by_model


def test_a_lookup_round_trips(con: Any) -> None:
    rows = con.execute(
        "SELECT u.id = x.id FROM (SELECT id FROM anthropic.users LIMIT 1) u, "
        "LATERAL anthropic.user_by_id(u.id) x"
    ).fetchall()
    assert rows == [(True,)]
