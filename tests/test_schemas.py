"""Flattening and total conversion of API payloads."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pyarrow as pa
import pytest

from tests.fake_admin_api import API_KEYS, WORKSPACES, claude_code_records
from vgi_anthropic import schemas as s


class TestConverters:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("123.78912", Decimal("123.78912")),
            (42.5, Decimal("42.5")),
            (186, Decimal(186)),
            ("NaN", None),
            ("", None),
            (None, None),
            (True, None),
            ("abc", None),
        ],
    )
    def test_to_decimal(self, raw: object, expected: Decimal | None) -> None:
        assert s.to_decimal(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"), [(5, 5), (5.0, 5), (0.5, None), ("7", 7), ("7.0", 7), ("x", None), (False, None)]
    )
    def test_to_integer_never_truncates(self, raw: object, expected: int | None) -> None:
        assert s.to_integer(raw) == expected

    def test_to_timestamp(self) -> None:
        assert s.to_timestamp("2024-10-30T23:58:27.427722Z") == datetime(
            2024, 10, 30, 23, 58, 27, 427722, tzinfo=UTC
        )
        assert s.to_timestamp("0001-01-01T00:00:00Z") is None  # unrepresentable in ns clients
        assert s.to_timestamp("garbage") is None

    def test_to_date(self) -> None:
        assert s.to_date("2025-08-08T00:00:00Z") == date(2025, 8, 8)
        assert s.to_date("2025-08-08") == date(2025, 8, 8)
        assert s.to_date("nope") is None

    def test_cents_to_usd_is_exact(self) -> None:
        assert s.cents_to_usd("123.78912") == Decimal("1.2378912")

    def test_a_value_that_does_not_fit_becomes_null(self) -> None:
        col = s.column([{"x": "1.0000000000001"}], "x", pa.field("x", pa.decimal128(10, 2)))
        assert col.to_pylist() == [None]

    def test_one_bad_nested_value_does_not_fail_the_column(self) -> None:
        kind = pa.map_(pa.string(), s.TOOL_ACTION)
        col = s.column(
            [{"t": {"edit": {"accepted": 1, "rejected": 0}}}, {"t": "not a map"}, {"t": None}],
            "t",
            pa.field("t", kind),
        )
        assert col.to_pylist() == [[("edit", {"accepted": 1, "rejected": 0})], None, None]


class TestFlatteners:
    def test_workspace_unrestricted_geos(self) -> None:
        row = s.flatten_workspace(WORKSPACES[0])
        assert row["inference_geo_unrestricted"] is True
        assert row["allowed_inference_geos"] is None
        assert row["is_archived"] is False

    def test_workspace_restricted_geos(self) -> None:
        row = s.flatten_workspace(WORKSPACES[1])
        assert (row["inference_geo_unrestricted"], row["allowed_inference_geos"]) == (False, ["us"])

    def test_workspace_without_residency_does_not_crash(self) -> None:
        row = s.flatten_workspace({"id": "w"})
        assert row["inference_geo_unrestricted"] is None

    def test_api_key_scope_beats_the_deprecated_field(self) -> None:
        row = s.flatten_api_key(API_KEYS[2])
        assert row["workspace_id"] == "wrkspc_default"

    def test_rate_limits_one_row_per_limiter(self) -> None:
        rows = s.flatten_rate_limits(
            [
                {
                    "id": "r",
                    "group": {"id": "g", "type": "batch"},
                    "limits": [{"type": "a", "value": 1}, {"type": "b", "value": 2}],
                }
            ]
        )
        assert [(r["group_type"], r["limit_type"], r["value"]) for r in rows] == [
            ("batch", "a", 1),
            ("batch", "b", 2),
        ]

    def test_usage_buckets_flatten_and_empty_buckets_vanish(self) -> None:
        rows = s.flatten_usage(
            [
                {"starting_at": "2025-08-01T00:00:00Z", "ending_at": "2025-08-02T00:00:00Z", "results": []},
                {
                    "starting_at": "2025-08-02T00:00:00Z",
                    "ending_at": "2025-08-03T00:00:00Z",
                    "results": [
                        {
                            "model": "m",
                            "output_tokens": 5,
                            "cache_creation": {
                                "ephemeral_5m_input_tokens": 3,
                                "ephemeral_1h_input_tokens": 4,
                            },
                            "server_tool_use": {"web_search_requests": 9},
                        }
                    ],
                },
            ]
        )
        assert len(rows) == 1
        batch = s.batch_from_rows(rows, s.MESSAGES_USAGE_SCHEMA).to_pylist()[0]
        assert batch["cache_creation_5m_input_tokens"] == 3
        assert batch["cache_creation_1h_input_tokens"] == 4
        assert batch["web_search_requests"] == 9
        assert batch["bucket_start"] == datetime(2025, 8, 2, tzinfo=UTC)

    def test_cost_amount_in_both_units(self) -> None:
        rows = s.flatten_cost(
            [{"starting_at": "x", "ending_at": "y", "results": [{"amount": "250", "currency": "USD"}]}]
        )
        built = s.batch_from_rows(rows, s.COST_REPORT_SCHEMA).to_pylist()[0]
        assert (built["amount_cents"], built["amount_usd"]) == (Decimal(250), Decimal("2.5"))

    def test_claude_code_rolls_up(self) -> None:
        row = s.flatten_claude_code(claude_code_records("2026-09-01")[0])
        assert (row["input_tokens"], row["output_tokens"]) == (1500, 250)
        assert row["estimated_cost_cents"] == Decimal("228.5")
        built = s.batch_from_rows([row], s.CLAUDE_CODE_USAGE_SCHEMA).to_pylist()[0]
        assert built["date"] == date(2026, 9, 1)
        assert dict(built["tool_actions"])["edit_tool"] == {"accepted": 9, "rejected": 1}

    def test_claude_code_partial_breakdown_gives_null_total_not_a_short_one(self) -> None:
        record = claude_code_records("2026-09-01")[0]
        record["model_breakdown"][1]["tokens"]["input"] = "corrupt"
        assert s.flatten_claude_code(record)["input_tokens"] is None

    def test_every_schema_column_is_documented(self) -> None:
        for schema in (
            s.ORGANIZATION_SCHEMA,
            s.USER_SCHEMA,
            s.INVITE_SCHEMA,
            s.WORKSPACE_SCHEMA,
            s.WORKSPACE_MEMBER_SCHEMA,
            s.API_KEY_SCHEMA,
            s.RATE_LIMIT_SCHEMA,
            s.WORKSPACE_RATE_LIMIT_SCHEMA,
            s.MESSAGES_USAGE_SCHEMA,
            s.COST_REPORT_SCHEMA,
            s.CLAUDE_CODE_USAGE_SCHEMA,
        ):
            undocumented = [f.name for f in schema if not (f.metadata or {}).get(b"comment")]
            assert undocumented == []
