"""SQL executed against a real ATTACH, backed by a local fake of the Admin API.

The extension, not the worker, decides how arguments are cast, which columns a
scan is asked for, and which predicates are pushed down — so none of that can
be tested by calling worker functions directly. This file runs the real stack
(Haybarn's DuckDB, the vgi extension, the worker as a subprocess) against
:mod:`tests.fake_admin_api`, and asserts both on the rows that come back and on
the requests that went upstream.

Two defects this suite was written against, both invisible offline otherwise:
a bare `'2026-09-01'` failed to bind against a TIMESTAMPTZ argument, and an
optional argument's default was rejected by the framework at catalog time.
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tests.duckdb_session import connect, worker_location
from tests.fake_admin_api import (
    ADMIN_KEY,
    ORG_SCOPED_KEY,
    WORKSPACE_KEY_REFUSAL,
    WORKSPACE_SCOPED_KEY,
    FakeAdminApi,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def fake() -> Iterator[FakeAdminApi]:
    server = FakeAdminApi().start()
    yield server
    server.stop()


def _attach(fake: FakeAdminApi, monkeypatch: pytest.MonkeyPatch) -> Any:
    # The worker subprocess inherits this, and talks to the fake instead of
    # api.anthropic.com.
    monkeypatch.setenv("VGI_ANTHROPIC_BASE_URL", fake.url)
    con = connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"ATTACH 'anthropic' (TYPE vgi, LOCATION '{worker_location('anthropic_worker.py')}')")
    return con


@pytest.fixture(scope="module")
def con(fake: FakeAdminApi) -> Iterator[Any]:
    with pytest.MonkeyPatch.context() as mp:
        connection = _attach(fake, mp)
        connection.execute(f"CREATE SECRET acme (TYPE anthropic_admin, api_key '{ADMIN_KEY}')")
        yield connection
        connection.close()


@pytest.fixture
def log(fake: FakeAdminApi) -> FakeAdminApi:
    """The fake, with its request log cleared for this test."""
    fake.requests.clear()
    return fake


def _queries(fake: FakeAdminApi, path_suffix: str) -> list[dict[str, list[str]]]:
    return [q for path, q, _ in fake.requests if path.endswith(path_suffix)]


class TestTables:
    def test_organization_is_one_row(self, con: Any) -> None:
        assert con.execute("SELECT name FROM anthropic.organization").fetchall() == [("Acme Robotics",)]

    def test_users_page_through_every_member(self, con: Any, log: FakeAdminApi) -> None:
        rows = con.execute("SELECT id FROM anthropic.users ORDER BY id").fetchall()
        assert [r[0] for r in rows] == [f"user_{i:02d}" for i in range(1, 6)]
        # 5 users at the fake's 2 per page: the cursor was actually followed.
        assert len(_queries(log, "/users")) == 3

    def test_users_table_and_scan_agree(self, con: Any) -> None:
        got = con.execute(
            "SELECT (SELECT count(*) FROM anthropic.users) = (SELECT count(*) FROM anthropic.all_users())"
        ).fetchone()
        assert got == (True,)

    def test_invites(self, con: Any) -> None:
        rows = con.execute(
            "SELECT id, status, accepted_at IS NOT NULL, rbac_group_ids FROM anthropic.invites ORDER BY id"
        ).fetchall()
        assert rows == [
            ("invite_01", "pending", False, []),
            ("invite_02", "accepted", True, ["rbac_1"]),
            ("invite_03", "expired", False, []),
        ]

    def test_workspaces_include_archived_and_flatten_residency(self, con: Any) -> None:
        rows = con.execute(
            "SELECT id, is_archived, inference_geo_unrestricted, allowed_inference_geos, tags['team'] "
            "FROM anthropic.workspaces ORDER BY id"
        ).fetchall()
        assert rows == [
            ("wrkspc_01", False, True, None, "prod"),
            ("wrkspc_02", False, False, ["us"], "staging"),
            ("wrkspc_03", True, True, None, "legacy"),
        ]

    def test_workspace_members_walk_only_active_workspaces(self, con: Any, log: FakeAdminApi) -> None:
        rows = con.execute(
            "SELECT workspace_id, user_id FROM anthropic.workspace_members ORDER BY ALL"
        ).fetchall()
        assert ("wrkspc_03", "user_01") not in rows  # archived workspace not walked
        assert rows == [
            ("wrkspc_01", "user_01"),
            ("wrkspc_01", "user_02"),
            ("wrkspc_01", "user_03"),
            ("wrkspc_02", "user_02"),
        ]

    def test_api_keys_take_workspace_from_scope(self, con: Any) -> None:
        rows = con.execute(
            "SELECT id, workspace_id, created_by_type, principal_type FROM anthropic.api_keys ORDER BY id"
        ).fetchall()
        assert rows == [
            ("apikey_01", "wrkspc_01", "user", None),
            ("apikey_02", "wrkspc_02", "service_account", "service_account_actor"),
            # The deprecated top-level workspace_id is NULL here; scope is not.
            ("apikey_03", "wrkspc_default", "user", "user_actor"),
        ]

    def test_rate_limits_flatten_per_limiter(self, con: Any) -> None:
        rows = con.execute(
            "SELECT id, limit_type, value FROM anthropic.rate_limits "
            "WHERE list_contains(models, 'claude-sonnet-5-5') ORDER BY limit_type"
        ).fetchall()
        assert rows == [("rl_01", "output_tokens_per_minute", 400000), ("rl_01", "requests_per_minute", 4000)]

    def test_workspace_rate_limits_include_inherited(self, con: Any) -> None:
        rows = con.execute(
            "SELECT workspace_id, limit_type, value, org_limit, source FROM anthropic.workspace_rate_limits "
            "WHERE rate_limit_id = 'rl_01' ORDER BY ALL"
        ).fetchall()
        assert ("wrkspc_01", "requests_per_minute", 2000, 4000, "workspace") in rows
        assert ("wrkspc_02", "requests_per_minute", 4000, 4000, "organization") in rows


class TestPushdown:
    """Pushed predicates must reach the API — and must still be applied.

    Declaring filter_pushdown makes the engine drop its own filter above the
    scan, so a predicate the worker failed to apply would not be applied by
    anyone. Each test checks the rows as well as the request.
    """

    def test_email_is_sent_upstream(self, con: Any, log: FakeAdminApi) -> None:
        rows = con.execute("SELECT id FROM anthropic.users WHERE email = 'person3@acme.test'").fetchall()
        assert rows == [("user_03",)]
        assert _queries(log, "/users")[0]["email"] == ["person3@acme.test"]

    def test_api_filter_is_case_insensitive_but_sql_is_not(self, con: Any) -> None:
        """The API's email match is broader than `=`; the exact predicate still wins."""
        assert (
            con.execute("SELECT id FROM anthropic.users WHERE email = 'PERSON3@ACME.TEST'").fetchall() == []
        )

    def test_a_workspace_filter_narrows_the_walk_to_one_request(self, con: Any, log: FakeAdminApi) -> None:
        rows = con.execute(
            "SELECT user_id FROM anthropic.workspace_members WHERE workspace_id = 'wrkspc_02'"
        ).fetchall()
        assert rows == [("user_02",)]
        assert log.paths() == ["/v1/organizations/workspaces/wrkspc_02/members"]

    def test_status_is_sent_upstream(self, con: Any, log: FakeAdminApi) -> None:
        rows = con.execute("SELECT id FROM anthropic.api_keys WHERE status = 'archived'").fetchall()
        assert rows == [("apikey_03",)]
        assert _queries(log, "/api_keys")[0]["status"] == ["archived"]

    def test_a_service_account_creator_is_not_pushed_as_a_user(self, con: Any, log: FakeAdminApi) -> None:
        """Pushing svac_01 into created_by_user_id would drop the row that matches."""
        rows = con.execute("SELECT id FROM anthropic.api_keys WHERE created_by_id = 'svac_01'").fetchall()
        assert rows == [("apikey_02",)]
        assert "created_by_user_id" not in _queries(log, "/api_keys")[0]

    def test_a_key_workspace_is_pushed(self, con: Any, log: FakeAdminApi) -> None:
        """The API filters on scope, so the default workspace's real ID matches too."""
        rows = con.execute(
            "SELECT id FROM anthropic.api_keys WHERE workspace_id = 'wrkspc_default'"
        ).fetchall()
        assert rows == [("apikey_03",)]
        assert _queries(log, "/api_keys")[0]["workspace_id"] == ["wrkspc_default"]

    def test_a_user_creator_is_pushed(self, con: Any, log: FakeAdminApi) -> None:
        rows = con.execute("SELECT id FROM anthropic.api_keys WHERE created_by_id = 'user_02'").fetchall()
        assert rows == [("apikey_03",)]
        assert _queries(log, "/api_keys")[0]["created_by_user_id"] == ["user_02"]

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT count(*) FROM anthropic.users WHERE role = 'no-such-role'",
            "SELECT count(*) FROM anthropic.workspaces WHERE name = 'Nope'",
            "SELECT count(*) FROM anthropic.api_keys WHERE status = 'active' AND name = 'nope'",
            "SELECT count(*) FROM anthropic.messages_usage('2026-09-01') WHERE output_tokens > 1e12",
            "SELECT count(*) FROM anthropic.cost_report('2026-09-01') WHERE amount_usd > 1e9",
            "SELECT count(*) FROM anthropic.claude_code_usage('2026-09-01') WHERE num_sessions > 1000",
        ],
    )
    def test_an_impossible_predicate_returns_nothing(self, con: Any, sql: str) -> None:
        assert con.execute(sql).fetchall() == [(0,)]


class TestMessagesUsage:
    def test_a_bare_string_start_binds(self, con: Any) -> None:
        assert con.execute("SELECT count(*) > 0 FROM anthropic.messages_usage('2026-09-08')").fetchone() == (
            True,
        )

    @pytest.mark.parametrize(
        "start",
        [
            "'2026-09-08'",
            "DATE '2026-09-08'",
            "TIMESTAMP '2026-09-08'",
            "TIMESTAMPTZ '2026-09-08 00:00:00+00'",
        ],
    )
    def test_every_time_type_binds_to_the_same_report(self, con: Any, start: str) -> None:
        total = con.execute(
            f"SELECT sum(output_tokens) FROM anthropic.messages_usage({start}, ending_at => '2026-09-09')"
        ).fetchone()
        assert total == (157,)

    def test_grouping_follows_the_selected_columns(self, con: Any, log: FakeAdminApi) -> None:
        rows = con.execute(
            "SELECT model, sum(output_tokens) "
            "FROM anthropic.messages_usage('2026-09-08', ending_at => '2026-09-09') "
            "GROUP BY model ORDER BY model"
        ).fetchall()
        assert rows == [("claude-opus-5-5", 120), ("claude-sonnet-5-5", 37)]
        assert _queries(log, "/messages")[0]["group_by[]"] == ["model"]

    def test_totals_do_not_depend_on_grouping(self, con: Any) -> None:
        window = "'2026-09-01', ending_at => '2026-09-05'"
        ungrouped = con.execute(
            f"SELECT sum(output_tokens) FROM anthropic.messages_usage({window})"
        ).fetchone()
        grouped = con.execute(
            f"SELECT sum(output_tokens) FROM (SELECT model, workspace_id, api_key_id, output_tokens "
            f"FROM anthropic.messages_usage({window}))"
        ).fetchone()
        assert ungrouped == grouped == (4 * 157,)

    def test_an_ungrouped_report_asks_for_no_dimensions(self, con: Any, log: FakeAdminApi) -> None:
        con.execute("SELECT sum(output_tokens) FROM anthropic.messages_usage('2026-09-08')").fetchall()
        assert "group_by[]" not in _queries(log, "/messages")[0]

    def test_an_equality_filter_is_pushed_and_grouped(self, con: Any, log: FakeAdminApi) -> None:
        """Without grouping, model would read NULL and `model = ...` would match nothing."""
        rows = con.execute(
            "SELECT sum(output_tokens) "
            "FROM anthropic.messages_usage('2026-09-08', ending_at => '2026-09-09') "
            "WHERE model = 'claude-sonnet-5-5'"
        ).fetchall()
        assert rows == [(37,)]
        query = _queries(log, "/messages")[0]
        assert query["models[]"] == ["claude-sonnet-5-5"]
        assert "model" in query["group_by[]"]

    def test_explicit_group_by_is_honoured(self, con: Any) -> None:
        rows = con.execute(
            "SELECT DISTINCT model, workspace_id FROM anthropic.messages_usage('2026-09-08', "
            "ending_at => '2026-09-09', group_by => ['model']) ORDER BY model"
        ).fetchall()
        assert rows == [("claude-opus-5-5", None), ("claude-sonnet-5-5", None)]

    def test_hourly_buckets_page(self, con: Any, log: FakeAdminApi) -> None:
        count = con.execute(
            "SELECT count(DISTINCT bucket_start) FROM anthropic.messages_usage('2026-09-01', "
            "ending_at => '2026-09-09', bucket_width => '1h')"
        ).fetchone()
        assert count == (8 * 24,)
        # 192 hourly buckets at 168 per page.
        assert len(_queries(log, "/messages")) == 2

    def test_buckets_are_utc_days(self, con: Any) -> None:
        got = con.execute(
            "SELECT DISTINCT bucket_start, bucket_end - bucket_start FROM anthropic.messages_usage("
            "'2026-09-08', ending_at => '2026-09-09')"
        ).fetchall()
        assert [(str(start), str(width)) for start, width in got] == [
            ("2026-09-08 00:00:00+00:00", "1 day, 0:00:00")
        ]

    @pytest.mark.parametrize(
        ("sql", "message"),
        [
            ("SELECT * FROM anthropic.messages_usage('not a date')", "not a date or timestamp"),
            (
                "SELECT * FROM anthropic.messages_usage('2026-09-08', ending_at => '2026-09-01')",
                "must be after",
            ),
            ("SELECT * FROM anthropic.messages_usage('2026-09-08', group_by => ['modle'])", "modle"),
            ("SELECT * FROM anthropic.messages_usage('2026-09-08', bucket_width => '1w')", "1w"),
        ],
    )
    def test_bad_arguments_are_named(self, con: Any, sql: str, message: str) -> None:
        with pytest.raises(Exception, match=message):
            con.execute(sql).fetchall()


class TestCostReport:
    def test_money_is_exact_decimal(self, con: Any) -> None:
        rows = con.execute(
            "SELECT typeof(amount_usd), sum(amount_cents), sum(amount_usd) "
            "FROM anthropic.cost_report('2026-09-09') WHERE workspace_id = 'wrkspc_01' GROUP BY 1"
        ).fetchall()
        assert rows == [("DECIMAL(38,14)", Decimal("1234.56789"), Decimal("12.3456789"))]

    def test_a_tiny_amount_survives(self, con: Any) -> None:
        """0.00001 cents through a float would not round-trip; a decimal does."""
        rows = con.execute(
            "SELECT amount_cents FROM anthropic.cost_report('2026-09-09') WHERE workspace_id IS NULL"
        ).fetchall()
        assert rows == [(Decimal("0.00001"),)]

    def test_selecting_model_groups_by_description(self, con: Any, log: FakeAdminApi) -> None:
        rows = con.execute(
            "SELECT model, sum(amount_usd) FROM anthropic.cost_report('2026-09-09') "
            "GROUP BY model ORDER BY model"
        ).fetchall()
        assert rows == [("claude-opus-5-5", Decimal("12.3456790")), ("claude-sonnet-5-5", Decimal("1"))]
        assert _queries(log, "/cost_report")[0]["group_by[]"] == ["description"]

    def test_default_workspace_reads_null(self, con: Any) -> None:
        rows = con.execute(
            "SELECT COALESCE(w.name, 'Default') AS workspace, sum(c.amount_usd) "
            "FROM anthropic.cost_report('2026-09-09') c "
            "LEFT JOIN anthropic.workspaces w ON w.id = c.workspace_id "
            "GROUP BY ALL ORDER BY ALL"
        ).fetchall()
        assert [name for name, _ in rows] == ["Default", "Prod", "Staging"]

    def test_a_month_pages_at_31_days(self, con: Any, log: FakeAdminApi) -> None:
        con.execute("SELECT count(*) FROM anthropic.cost_report('2026-07-01')").fetchall()
        queries = _queries(log, "/cost_report")
        assert all(q["limit"] == ["31"] for q in queries)
        assert len(queries) == 3  # 71 days to the fake's "now"


class TestClaudeCodeUsage:
    def test_one_day_by_default(self, con: Any, log: FakeAdminApi) -> None:
        rows = con.execute("SELECT DISTINCT date FROM anthropic.claude_code_usage('2026-09-01')").fetchall()
        assert [str(d) for (d,) in rows] == ["2026-09-01"]
        assert {q["starting_at"][0] for q in _queries(log, "/claude_code")} == {"2026-09-01"}

    def test_a_range_walks_each_day_and_pages_within_it(self, con: Any, log: FakeAdminApi) -> None:
        count = con.execute(
            "SELECT count(*) FROM anthropic.claude_code_usage("
            "DATE '2026-09-01', ending_date => DATE '2026-09-03')"
        ).fetchone()
        assert count == (9,)
        # 3 days x 3 records at 2 per page.
        assert len(_queries(log, "/claude_code")) == 6

    def test_totals_roll_up_the_model_breakdown(self, con: Any) -> None:
        rows = con.execute(
            "SELECT input_tokens, output_tokens, estimated_cost_cents, estimated_cost_usd, "
            "len(model_breakdown) "
            "FROM anthropic.claude_code_usage('2026-09-01') WHERE email_address = 'person1@acme.test'"
        ).fetchall()
        assert rows == [(1500, 250, Decimal("228.5"), Decimal("2.285"), 2)]

    def test_nested_columns_are_queryable(self, con: Any) -> None:
        rows = con.execute(
            "SELECT tool_actions['edit_tool'].accepted, m.model "
            "FROM anthropic.claude_code_usage('2026-09-01'), "
            "UNNEST(model_breakdown) AS t(m) WHERE actor_type = 'api_actor' ORDER BY m.model"
        ).fetchall()
        assert rows == [(9, "claude-opus-5-5"), (9, "claude-sonnet-5-5")]

    def test_a_reversed_range_is_refused(self, con: Any) -> None:
        with pytest.raises(Exception, match="before starting_date"):
            con.execute(
                "SELECT * FROM anthropic.claude_code_usage('2026-09-05', ending_date => '2026-09-01')"
            )

    def test_an_enormous_range_is_refused(self, con: Any) -> None:
        with pytest.raises(Exception, match="more than 366"):
            con.execute(
                "SELECT * FROM anthropic.claude_code_usage('2020-01-01', ending_date => '2026-01-01')"
            )


class TestScansStream:
    """A LIMIT must stop a walk early rather than paying for every page."""

    def test_limit_stops_the_user_walk(self, con: Any, log: FakeAdminApi) -> None:
        assert len(con.execute("SELECT id FROM anthropic.users LIMIT 1").fetchall()) == 1
        assert len(_queries(log, "/users")) < 3

    def test_limit_stops_the_claude_code_walk(self, con: Any, log: FakeAdminApi) -> None:
        con.execute(
            "SELECT * FROM anthropic.claude_code_usage('2026-08-01', ending_date => '2026-08-31') LIMIT 1"
        ).fetchall()
        assert len(_queries(log, "/claude_code")) < 31


class TestLookups:
    def test_point_lookup(self, con: Any) -> None:
        assert con.execute("SELECT email FROM anthropic.user_by_id('user_02')").fetchall() == [
            ("person2@acme.test",)
        ]

    def test_unknown_id_is_no_row_not_an_error(self, con: Any) -> None:
        assert con.execute("SELECT * FROM anthropic.user_by_id('user_gone')").fetchall() == []

    def test_lateral_composes_and_drops_unknowns(self, con: Any) -> None:
        rows = con.execute(
            "SELECT k.id, u.email FROM anthropic.api_keys k, LATERAL anthropic.user_by_id(k.created_by_id) u "
            "ORDER BY k.id"
        ).fetchall()
        # apikey_02 was created by a service account, which is not a user.
        assert rows == [("apikey_01", "person1@acme.test"), ("apikey_03", "person2@acme.test")]

    def test_workspace_and_key_lookups(self, con: Any) -> None:
        assert con.execute("SELECT is_archived FROM anthropic.workspace_by_id('wrkspc_03')").fetchall() == [
            (True,)
        ]
        assert con.execute("SELECT workspace_id FROM anthropic.api_key_by_id('apikey_03')").fetchall() == [
            ("wrkspc_default",)
        ]

    def test_a_traversing_id_stays_in_its_segment(self, con: Any, log: FakeAdminApi) -> None:
        assert con.execute("SELECT * FROM anthropic.user_by_id('../invites')").fetchall() == []
        assert log.paths() == ["/v1/organizations/users/..%2Finvites"]


class TestReadOnly:
    def test_nothing_but_get_reached_the_api(self, con: Any, fake: FakeAdminApi) -> None:
        """Checked last-ish and across everything this module ran."""
        con.execute("SELECT count(*) FROM anthropic.workspace_rate_limits").fetchall()
        assert all(headers.get("method") != "POST" for _, _, headers in fake.requests)


class TestCredentials:
    def test_no_secret_names_the_fix(self, fake: FakeAdminApi, monkeypatch: pytest.MonkeyPatch) -> None:
        con = _attach(fake, monkeypatch)
        with pytest.raises(Exception, match="CREATE SECRET"):
            con.execute("SELECT * FROM anthropic.organization").fetchall()

    def test_an_organization_scoped_standard_key_works(
        self, fake: FakeAdminApi, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An `sk-ant-api03-` prefix does not mean "not admin"; the worker must not guess."""
        con = _attach(fake, monkeypatch)
        con.execute(f"CREATE SECRET (TYPE anthropic_admin, api_key '{ORG_SCOPED_KEY}')")
        assert con.execute("SELECT count(*) FROM anthropic.users").fetchone() == (5,)

    def test_a_workspace_scoped_key_reads_only_the_organization(
        self, fake: FakeAdminApi, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Observed against the real API: /me answers, everything else is a 401."""
        con = _attach(fake, monkeypatch)
        con.execute(f"CREATE SECRET (TYPE anthropic_admin, api_key '{WORKSPACE_SCOPED_KEY}')")
        assert con.execute("SELECT count(*) FROM anthropic.organization").fetchone() == (1,)
        with pytest.raises(Exception) as info:
            con.execute("SELECT * FROM anthropic.users").fetchall()
        message = str(info.value)
        assert WORKSPACE_KEY_REFUSAL in message
        assert "workspace-scoped key can read `organization` and nothing else" in message

    def test_a_rejected_key_reports_the_status(
        self, fake: FakeAdminApi, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        con = _attach(fake, monkeypatch)
        con.execute("CREATE SECRET (TYPE anthropic_admin, api_key 'sk-ant-admin01-revoked')")
        with pytest.raises(Exception, match="401"):
            con.execute("SELECT * FROM anthropic.organization").fetchall()

    def test_the_key_is_redacted_in_duckdb_secrets(self, con: Any) -> None:
        rendered = con.execute("SELECT secret_string FROM duckdb_secrets() WHERE name = 'acme'").fetchone()[0]
        assert ADMIN_KEY not in rendered

    def test_a_secret_of_any_name_is_found(self, fake: FakeAdminApi, monkeypatch: pytest.MonkeyPatch) -> None:
        con = _attach(fake, monkeypatch)
        con.execute(f"CREATE SECRET not_named_after_the_type (TYPE anthropic_admin, api_key '{ADMIN_KEY}')")
        assert con.execute("SELECT count(*) FROM anthropic.organization").fetchone() == (1,)
