"""Report argument handling and group-by inference."""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest
from vgi.arguments import AnyArrowValue

from vgi_anthropic import reports as r
from vgi_anthropic.schemas import USAGE_DIMENSIONS


class TestTimeArguments:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2026-09-01", datetime(2026, 9, 1, tzinfo=UTC)),
            ("2026-09-01T12:30:00Z", datetime(2026, 9, 1, 12, 30, tzinfo=UTC)),
            (date(2026, 9, 1), datetime(2026, 9, 1, tzinfo=UTC)),
            (datetime(2026, 9, 1, 6), datetime(2026, 9, 1, 6, tzinfo=UTC)),  # naive reads as UTC
            (None, None),
            ("", None),
        ],
    )
    def test_instant_arg(self, raw: object, expected: datetime | None) -> None:
        assert r.instant_arg(raw, "x") == expected

    def test_wrapped_values_are_unwrapped(self) -> None:
        wrapped = AnyArrowValue(value="2026-09-01", position=0, name="starting_at")
        assert r.instant_arg(wrapped, "starting_at") == datetime(2026, 9, 1, tzinfo=UTC)

    def test_garbage_is_named(self) -> None:
        with pytest.raises(ValueError, match="starting_at => 'yesterday'"):
            r.instant_arg("yesterday", "starting_at")

    def test_bounds_must_be_ordered(self) -> None:
        with pytest.raises(ValueError, match="must be after"):
            r._report_bounds(SimpleNamespace(starting_at="2026-09-02", ending_at="2026-09-02"))


class TestGroupBy:
    ALL = list(USAGE_DIMENSIONS)
    INFERABLE = [d for d in ALL if d != "speed"]

    def _resolve(
        self, explicit: list[str], referenced: set[str], pinned: set[str] = frozenset()
    ) -> list[str]:
        return r.resolve_group_by(
            explicit, referenced, set(pinned), allowed=self.ALL, inferable=self.INFERABLE
        )

    def test_inferred_from_the_projection(self) -> None:
        assert self._resolve([], {"model", "output_tokens", "bucket_start"}) == ["model"]

    def test_nothing_referenced_means_no_grouping(self) -> None:
        assert self._resolve([], {"output_tokens"}) == []

    def test_speed_is_never_inferred(self) -> None:
        assert "speed" not in self._resolve([], set(self.ALL))

    def test_explicit_wins_but_pins_are_added(self) -> None:
        assert self._resolve(["model"], {"workspace_id"}, {"service_tier"}) == ["model", "service_tier"]

    def test_explicit_duplicates_collapse(self) -> None:
        assert self._resolve(["model", "model"], set()) == ["model"]

    def test_unknown_dimension_is_named(self) -> None:
        with pytest.raises(ValueError, match="modle"):
            self._resolve(["modle"], set())


class TestClaudeCodeDays:
    def _args(self, start: object, end: object = None) -> SimpleNamespace:
        return SimpleNamespace(starting_date=start, ending_date=end)

    def test_single_day(self) -> None:
        assert r.claude_code_days(self._args("2026-09-01"), today=date(2026, 10, 1)) == ["2026-09-01"]

    def test_range_is_inclusive(self) -> None:
        days = r.claude_code_days(self._args("2026-09-29", "2026-10-01"), today=date(2026, 10, 1))
        assert days == ["2026-09-29", "2026-09-30", "2026-10-01"]

    def test_future_days_are_not_requested(self) -> None:
        days = r.claude_code_days(self._args("2026-09-30", "2026-12-31"), today=date(2026, 10, 1))
        assert days == ["2026-09-30", "2026-10-01"]

    def test_a_future_start_requests_nothing(self) -> None:
        assert r.claude_code_days(self._args("2027-01-01"), today=date(2026, 10, 1)) == []

    def test_timestamps_are_read_as_their_utc_day(self) -> None:
        days = r.claude_code_days(
            self._args(datetime(2026, 9, 1, 23, 0, tzinfo=UTC)), today=date(2026, 10, 1)
        )
        assert days == ["2026-09-01"]
