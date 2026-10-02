"""The Usage, Cost and Claude Code reports: time-bucketed, keyed by a start time.

These take a required start (``starting_at``) rather than being tables: the API
requires one, and "all usage ever" is not a question it can answer in one
call. Each is a paged scan — one API page per tick — so a ``LIMIT`` stops early.

Grouping follows the query
--------------------------

The usage and cost endpoints aggregate over every dimension they are not
grouped by, and report NULL for it. Taken literally, that makes
``SELECT model, sum(output_tokens) FROM messages_usage(...)`` return one row
with a NULL model — a plausible-looking, useless answer. So when ``group_by``
is not given, the report groups by exactly the dimension columns the query
*references* (DuckDB tells the scan which columns it projected, and a column
used in ``WHERE`` is always among them):

* ``SELECT sum(output_tokens)`` — no dimensions, one total per bucket;
* ``SELECT model, sum(output_tokens) ... GROUP BY model`` — grouped by model;
* ``SELECT *`` — grouped by every dimension (except ``speed``, which needs a
  beta and must be asked for).

Totals are the same whichever way it groups; only the breakdown changes. An
explicit ``group_by`` is honoured as given, plus any dimension the ``WHERE``
pins with ``=`` — grouping by a single-valued dimension adds no rows, and
without it that column would read NULL and the predicate would match nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Any, ClassVar

import pyarrow as pa
from vgi.arguments import AnyArrowValue, Arg, SecretLookupEntry
from vgi.cache_control import CacheControl
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_function import BindParams, ProcessParams, TableFunctionGenerator, init_single_worker
from vgi_rpc.rpc import OutputCollector

from vgi_anthropic import anthropic_api as api
from vgi_anthropic import auth
from vgi_anthropic.meta import docs, examples
from vgi_anthropic.paging import Query, WalkState, emit_walk
from vgi_anthropic.pushdown import equality, projected
from vgi_anthropic.schemas import (
    CLAUDE_CODE_USAGE_SCHEMA,
    COST_DESCRIPTION_COLUMNS,
    COST_REPORT_SCHEMA,
    MESSAGES_USAGE_SCHEMA,
    USAGE_DIMENSIONS,
    flatten_claude_code,
    flatten_cost,
    flatten_usage,
    to_date,
    to_timestamp,
)

_SECRETS = [SecretLookupEntry(secret_type=auth.SECRET_TYPE)]

#: What a time argument accepts. A strictly typed TIMESTAMPTZ argument rejects
#: a bare string — DuckDB will not implicitly cast `'2026-09-01'` into it for a
#: table function — which made the obvious call fail to bind. Accepting strings,
#: dates and timestamps alike, and parsing here, makes `cost_report('2026-09-01')`
#: and `cost_report(current_date - INTERVAL 7 DAY)` both work.
_TIME_TYPES = [pa.types.is_string, pa.types.is_large_string, pa.types.is_date, pa.types.is_timestamp]

#: Buckets per page: the API's maximum for each width, so a range costs as few
#: requests as possible. Sending more is a 400.
_BUCKETS_PER_PAGE = {"1d": 31, "1h": 168, "1m": 1440}

#: Usage dimensions, by the query parameter that filters on each.
_USAGE_FILTER_PARAMS = {
    "workspace_id": "workspace_ids[]",
    "api_key_id": "api_key_ids[]",
    "account_id": "account_ids[]",
    "service_account_id": "service_account_ids[]",
    "model": "models[]",
    "service_tier": "service_tiers[]",
    "context_window": "context_window[]",
    "inference_geo": "inference_geos[]",
    "speed": "speeds[]",
}

#: Dimensions inferred from the projection. ``speed`` is excluded: grouping by
#: it needs the fast-mode beta, so it is only used when asked for by name.
_INFERRED_USAGE_DIMENSIONS = [d for d in USAGE_DIMENSIONS if d != "speed"]

_COST_DIMENSIONS = ("workspace_id", "description")

#: The Claude Code report is one request (or more) per day; a walk longer than
#: this is far more likely a typo'd year than an intent.
MAX_CLAUDE_CODE_DAYS = 366


def _rfc3339(instant: datetime) -> str:
    return instant.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _raw(value: Any) -> Any:
    """The Python value of an argument, unwrapping an ``AnyArrowValue``."""
    return value.value if isinstance(value, AnyArrowValue) else value


def instant_arg(value: Any, name: str) -> datetime | None:
    """A time argument as an aware UTC instant, or None when it was not given.

    A date is midnight UTC; a timestamp without a zone is read as UTC, which is
    what every bucket boundary in these reports is. An empty string is "not
    supplied" — the default for the optional bounds.
    """
    raw = _raw(value)
    if raw is None or raw == "":
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    if isinstance(raw, date):
        return datetime(raw.year, raw.month, raw.day, tzinfo=UTC)
    parsed = to_timestamp(str(raw).strip())
    if parsed is None:
        raise ValueError(
            f"{name} => {raw!r} is not a date or timestamp (try '2026-09-01' or '2026-09-01T12:00:00Z')"
        )
    return parsed


def date_arg(value: Any, name: str) -> date | None:
    """A day argument as a date, or None when it was not given."""
    raw = _raw(value)
    if raw is None or raw == "":
        return None
    if isinstance(raw, datetime):
        return (raw if raw.tzinfo else raw.replace(tzinfo=UTC)).astimezone(UTC).date()
    if isinstance(raw, date):
        return raw
    parsed = to_date(str(raw).strip())
    if parsed is None:
        raise ValueError(f"{name} => {raw!r} is not a date (try '2026-09-01')")
    return parsed


def _report_bounds(args: Any) -> list[tuple[str, Any]]:
    """``starting_at``/``ending_at`` query parameters, validated."""
    start = instant_arg(args.starting_at, "starting_at")
    if start is None:
        raise ValueError("starting_at is required")
    end = instant_arg(args.ending_at, "ending_at")
    if end is not None and end <= start:
        raise ValueError(f"ending_at ({_rfc3339(end)}) must be after starting_at ({_rfc3339(start)})")
    return [("starting_at", _rfc3339(start)), ("ending_at", _rfc3339(end) if end else None)]


def _opt_in_cache(ttl: int) -> CacheControl | None:
    """Cache metadata when the caller asked for it with ``cache_ttl``.

    Off by default. Recent buckets keep filling in for several minutes after
    the requests they count (the API documents ~5 minutes, occasionally more),
    so a cached report can be short; the caller decides whether that matters.
    """
    return CacheControl(ttl=ttl) if ttl > 0 else None


def _validated(values: Sequence[str], allowed: Sequence[str], arg: str) -> list[str]:
    unknown = [v for v in values if v not in allowed]
    if unknown:
        raise ValueError(f"{arg} => {unknown!r}: each entry must be one of {', '.join(allowed)}")
    return list(dict.fromkeys(values))


def resolve_group_by(
    explicit: Sequence[str],
    referenced: set[str],
    pinned: set[str],
    *,
    allowed: Sequence[str],
    inferable: Sequence[str],
    arg: str = "group_by",
) -> list[str]:
    """The dimensions to group by — see the module docstring.

    Args:
        explicit: The ``group_by`` argument; empty means infer.
        referenced: Columns the query projected (including filtered ones).
        pinned: Dimensions the ``WHERE`` fixes with ``=``.
        allowed: Every dimension the endpoint accepts.
        inferable: Dimensions that may be inferred from ``referenced``.
    """
    chosen = _validated(explicit, allowed, arg) if explicit else [d for d in inferable if d in referenced]
    chosen += [d for d in allowed if d in pinned and d not in chosen]
    return chosen


# --------------------------------------------------------------------------
# Messages usage
# --------------------------------------------------------------------------


@dataclass(slots=True, frozen=True, kw_only=True)
class UsageArgs:
    """``messages_usage(starting_at)`` and its named refinements."""

    starting_at: Annotated[
        AnyArrowValue,
        Arg(
            0,
            doc="Start of the report (inclusive), snapped down to the bucket boundary; strings read as UTC",
            type_bound=_TIME_TYPES,
        ),
    ]
    ending_at: Annotated[
        AnyArrowValue | None,
        Arg(
            "ending_at",
            doc="End of the report (exclusive); defaults to now",
            default=None,
            type_bound=_TIME_TYPES,
        ),
    ] = None
    bucket_width: Annotated[
        str, Arg("bucket_width", doc="Bucket size: 1d, 1h or 1m", default="1d", choices=["1d", "1h", "1m"])
    ] = "1d"
    group_by: Annotated[
        list[str],
        Arg(
            "group_by",
            doc="Dimensions to break usage down by; omit to group by the dimension columns the query uses",
            default=[],
            arrow_type=pa.list_(pa.string()),
        ),
    ] = field(default_factory=list)
    cache_ttl: Annotated[
        int, Arg("cache_ttl", doc="Seconds to cache the result (0 = off)", default=0, ge=0)
    ] = 0


def usage_query(params: ProcessParams[UsageArgs]) -> tuple[Query, list[str]]:
    """Query parameters for the usage report, and the ``anthropic-beta`` values it needs."""
    args = params.args
    pins = {dim: value for dim in USAGE_DIMENSIONS if (value := equality(params, dim))}
    group_by = resolve_group_by(
        args.group_by,
        projected(params),
        set(pins),
        allowed=list(USAGE_DIMENSIONS),
        inferable=_INFERRED_USAGE_DIMENSIONS,
    )
    pairs: list[tuple[str, Any]] = [
        *_report_bounds(args),
        ("bucket_width", args.bucket_width),
        ("limit", _BUCKETS_PER_PAGE[args.bucket_width]),
    ]
    pairs += [("group_by[]", dim) for dim in group_by]
    # Each pinned dimension becomes the API's own filter. Exact by construction:
    # the dimension is also grouped, so every returned row carries the value.
    pairs += [(_USAGE_FILTER_PARAMS[dim], value) for dim, value in pins.items()]
    betas = [api.FAST_MODE_BETA] if "speed" in group_by else []
    return pairs, betas


_USAGE_SQL = (
    "SELECT model, sum(uncached_input_tokens + cache_read_input_tokens) AS input_tokens, "
    "sum(output_tokens) AS output_tokens "
    "FROM anthropic.main.messages_usage(current_date - INTERVAL 7 DAY) "
    "GROUP BY model ORDER BY output_tokens DESC"
)


@init_single_worker
class MessagesUsageFunction(TableFunctionGenerator[UsageArgs, WalkState]):
    """Token usage of the Messages API, bucketed by time."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = MESSAGES_USAGE_SCHEMA

    class Meta:
        name = "messages_usage"
        description = "Messages API token usage over time, by model, workspace, API key and more"
        categories = ["usage"]
        required_secrets = _SECRETS
        projection_pushdown = True
        filter_pushdown = True
        auto_apply_filters = True
        tags = docs(
            category="usage",
            result_schema=MESSAGES_USAGE_SCHEMA,
            llm=(
                "Token consumption of the Messages API, one row per time bucket per combination "
                "of the dimensions you use — model, workspace_id, api_key_id, service_tier and "
                "so on. Pass the start time; grouping follows the columns your query selects, so "
                "selecting `model` breaks usage down by model without a group_by argument. Use "
                "this for token volumes; use cost_report for dollars."
            ),
            md=(
                "Token usage from the Usage API, flattened to one row per (bucket, group).\n\n"
                "### Arguments\n\n"
                "`starting_at` (required) and `ending_at` bound the report; `bucket_width` is "
                "`1d` (default), `1h` or `1m`. Buckets with no usage produce no rows.\n\n"
                "### Grouping follows your query\n\n"
                "A dimension column is only populated when the report is grouped by it. Without "
                "a `group_by` argument the worker groups by exactly the dimension columns the "
                "query references, so the columns you select are the breakdown you get and "
                "totals are unaffected. `WHERE model = '...'` (and the same for every dimension) "
                "is sent to the API as its own filter. Pass `group_by => ['speed']` to break "
                "down by fast mode.\n\n"
                "### Attribution\n\n"
                "`workspace_id` is NULL for the default workspace. Join `api_key_id` to "
                "`api_keys.id` and `account_id` to `users.id` for names.\n\n"
                "### Freshness\n\n"
                "Usage usually appears within five minutes of the request completing."
            ),
            example_queries=examples(
                ("Token volume by model over the last week", _USAGE_SQL),
                (
                    "Hourly output tokens for one model today",
                    "SELECT bucket_start, sum(output_tokens) AS output_tokens "
                    "FROM anthropic.main.messages_usage(current_date, bucket_width => '1h') "
                    "WHERE model = 'claude-opus-5-5' GROUP BY bucket_start ORDER BY bucket_start",
                ),
                (
                    "Usage by API key name",
                    "SELECT k.name, sum(u.output_tokens) AS output_tokens "
                    "FROM anthropic.main.messages_usage(current_date - INTERVAL 30 DAY) u "
                    "LEFT JOIN anthropic.main.api_keys k ON k.id = u.api_key_id "
                    "GROUP BY k.name ORDER BY output_tokens DESC",
                ),
            ),
        )
        examples = [FunctionExample(sql=_USAGE_SQL, description="Token volume by model over the last week")]

    @classmethod
    def on_bind(cls, params: BindParams[UsageArgs]) -> BindResponse:
        if params.args is not None:
            _report_bounds(params.args)
        return BindResponse(output_schema=cls.FIXED_SCHEMA)

    @classmethod
    def initial_state(cls, params: ProcessParams[UsageArgs]) -> WalkState:
        return WalkState()

    @classmethod
    def process(cls, params: ProcessParams[UsageArgs], state: WalkState, out: OutputCollector) -> None:
        query, betas = usage_query(params)
        emit_walk(
            params,
            state,
            out,
            keys=lambda: [""],
            endpoint_for=lambda _key: api.MESSAGES_USAGE,
            query=lambda: query,
            flatten=lambda rows, _key: flatten_usage(rows),
            betas=betas,
            cache_control=_opt_in_cache(params.args.cache_ttl),
        )


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


@dataclass(slots=True, frozen=True, kw_only=True)
class CostArgs:
    """``cost_report(starting_at)`` and its named refinements."""

    starting_at: Annotated[
        AnyArrowValue,
        Arg(
            0,
            doc="Start of the report (inclusive), snapped down to the UTC day; strings read as UTC",
            type_bound=_TIME_TYPES,
        ),
    ]
    ending_at: Annotated[
        AnyArrowValue | None,
        Arg(
            "ending_at",
            doc="End of the report (exclusive); defaults to now",
            default=None,
            type_bound=_TIME_TYPES,
        ),
    ] = None
    group_by: Annotated[
        list[str],
        Arg(
            "group_by",
            doc="Dimensions to break cost down by; omit to group by the columns the query uses",
            default=[],
            arrow_type=pa.list_(pa.string()),
        ),
    ] = field(default_factory=list)
    cache_ttl: Annotated[
        int, Arg("cache_ttl", doc="Seconds to cache the result (0 = off)", default=0, ge=0)
    ] = 0


def cost_query(params: ProcessParams[CostArgs]) -> Query:
    """Query parameters for the cost report.

    Inference maps columns to the two groupings the endpoint has: referencing
    ``workspace_id`` groups by workspace, and referencing ``description`` *or
    any column it populates* (model, token_type, cost_type, ...) groups by
    description. Pinning any of those with ``=`` does the same.
    """
    args = params.args
    referenced = projected(params)
    pinned = {c for c in ("workspace_id", *COST_DESCRIPTION_COLUMNS) if equality(params, c)}

    def as_dimensions(columns: set[str]) -> set[str]:
        dims = {"workspace_id"} & columns
        if columns & set(COST_DESCRIPTION_COLUMNS):
            dims.add("description")
        return dims

    group_by = resolve_group_by(
        args.group_by,
        as_dimensions(referenced),
        as_dimensions(pinned),
        allowed=_COST_DIMENSIONS,
        inferable=_COST_DIMENSIONS,
    )
    pairs: list[tuple[str, Any]] = [
        *_report_bounds(args),
        ("bucket_width", "1d"),
        ("limit", _BUCKETS_PER_PAGE["1d"]),
    ]
    return pairs + [("group_by[]", dim) for dim in group_by]


_COST_SQL = (
    "SELECT model, sum(amount_usd) AS usd "
    "FROM anthropic.main.cost_report(date_trunc('month', current_date)) "
    "GROUP BY model ORDER BY usd DESC"
)


@init_single_worker
class CostReportFunction(TableFunctionGenerator[CostArgs, WalkState]):
    """Daily cost in USD, optionally by workspace and line item."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = COST_REPORT_SCHEMA

    class Meta:
        name = "cost_report"
        description = "Daily spend in USD, by workspace and line item (model, token type, tool)"
        categories = ["cost"]
        required_secrets = _SECRETS
        projection_pushdown = True
        filter_pushdown = True
        auto_apply_filters = True
        tags = docs(
            category="cost",
            result_schema=COST_REPORT_SCHEMA,
            llm=(
                "What the organization spent, per day, in exact decimal dollars (`amount_usd`) and "
                "cents (`amount_cents`). Pass the start date. Selecting `workspace_id` breaks it "
                "down by workspace; selecting `model`, `token_type`, `cost_type` or `description` "
                "breaks it down by line item. Excludes Priority Tier, which is billed differently "
                "— use messages_usage with service_tier for that."
            ),
            md=(
                "Spend from the Cost API, one row per (day, group).\n\n"
                "### Money\n\n"
                "The API reports cents as a decimal string. `amount_cents` carries it exactly and "
                "`amount_usd` divides by 100 exactly; both are `DECIMAL`, never a float, so sums "
                "reconcile to the invoice line.\n\n"
                "### Grouping follows your query\n\n"
                "The endpoint groups by `workspace_id` and/or `description`. Without a `group_by` "
                "argument the worker picks them from the columns you reference: `description`, "
                "`model`, `cost_type`, `token_type`, `service_tier`, `context_window` and "
                "`inference_geo` are all populated by grouping on description.\n\n"
                "### What is not here\n\n"
                "Priority Tier costs are not in this endpoint. Days are UTC."
            ),
            example_queries=examples(
                ("Month-to-date spend by model", _COST_SQL),
                (
                    "Daily spend per workspace, with names",
                    "SELECT c.bucket_start::DATE AS day, COALESCE(w.name, 'Default') AS workspace, "
                    "sum(c.amount_usd) AS usd "
                    "FROM anthropic.main.cost_report(current_date - INTERVAL 30 DAY) c "
                    "LEFT JOIN anthropic.main.workspaces w ON w.id = c.workspace_id "
                    "GROUP BY ALL ORDER BY day, workspace",
                ),
            ),
        )
        examples = [FunctionExample(sql=_COST_SQL, description="Month-to-date spend by model")]

    @classmethod
    def on_bind(cls, params: BindParams[CostArgs]) -> BindResponse:
        if params.args is not None:
            _report_bounds(params.args)
        return BindResponse(output_schema=cls.FIXED_SCHEMA)

    @classmethod
    def initial_state(cls, params: ProcessParams[CostArgs]) -> WalkState:
        return WalkState()

    @classmethod
    def process(cls, params: ProcessParams[CostArgs], state: WalkState, out: OutputCollector) -> None:
        query = cost_query(params)
        emit_walk(
            params,
            state,
            out,
            keys=lambda: [""],
            endpoint_for=lambda _key: api.COST_REPORT,
            query=lambda: query,
            flatten=lambda rows, _key: flatten_cost(rows),
            cache_control=_opt_in_cache(params.args.cache_ttl),
        )


# --------------------------------------------------------------------------
# Claude Code
# --------------------------------------------------------------------------


@dataclass(slots=True, frozen=True, kw_only=True)
class ClaudeCodeArgs:
    """``claude_code_usage(starting_date)``, optionally through ``ending_date``."""

    starting_date: Annotated[AnyArrowValue, Arg(0, doc="First UTC day to report", type_bound=_TIME_TYPES)]
    ending_date: Annotated[
        AnyArrowValue | None,
        Arg(
            "ending_date",
            doc="Last UTC day to report, inclusive; omit for just starting_date",
            default=None,
            type_bound=_TIME_TYPES,
        ),
    ] = None
    cache_ttl: Annotated[
        int, Arg("cache_ttl", doc="Seconds to cache the result (0 = off)", default=0, ge=0)
    ] = 0


def claude_code_days(args: ClaudeCodeArgs, *, today: date | None = None) -> list[str]:
    """The days to request, as ``YYYY-MM-DD``.

    The endpoint answers one day per request, so a range is a walk over days.
    Days after today (UTC) cannot have data and are not requested.
    """
    first = date_arg(args.starting_date, "starting_date")
    if first is None:
        raise ValueError("starting_date is required")
    last = date_arg(args.ending_date, "ending_date") or first
    if last < first:
        raise ValueError(f"ending_date {last} is before starting_date {first}")
    last = min(last, today or datetime.now(UTC).date())
    span = (last - first).days + 1
    if span > MAX_CLAUDE_CODE_DAYS:
        raise ValueError(
            f"claude_code_usage spans {span} days, more than {MAX_CLAUDE_CODE_DAYS}; the report "
            "costs one request per day, so query a narrower range"
        )
    return [(first + timedelta(days=offset)).isoformat() for offset in range(max(span, 0))]


_CLAUDE_CODE_SQL = (
    "SELECT email_address, sum(num_sessions) AS sessions, sum(lines_added) AS lines_added, "
    "sum(estimated_cost_usd) AS usd "
    "FROM anthropic.main.claude_code_usage(current_date - INTERVAL 7 DAY, ending_date => current_date) "
    "WHERE actor_type = 'user_actor' GROUP BY email_address ORDER BY usd DESC"
)


@init_single_worker
class ClaudeCodeUsageFunction(TableFunctionGenerator[ClaudeCodeArgs, WalkState]):
    """Daily Claude Code productivity and spend, per user or API key."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = CLAUDE_CODE_USAGE_SCHEMA

    class Meta:
        name = "claude_code_usage"
        description = "Daily Claude Code sessions, lines changed, commits, PRs, tokens and cost per user"
        categories = ["claude-code"]
        required_secrets = _SECRETS
        projection_pushdown = True
        filter_pushdown = True
        auto_apply_filters = True
        tags = docs(
            category="claude-code",
            result_schema=CLAUDE_CODE_USAGE_SCHEMA,
            llm=(
                "Claude Code adoption and impact, one row per user (or API key) per day per "
                "terminal: sessions, lines added/removed, commits, pull requests, tokens and "
                "estimated cost. Pass a starting_date, and ending_date for a range. "
                "`model_breakdown` and `tool_actions` hold per-model and per-tool detail."
            ),
            md=(
                "The Claude Code Analytics report, one row per (day, actor, terminal, "
                "local/remote).\n\n"
                "### Range\n\n"
                "The API answers one UTC day per request, so `starting_date` through "
                "`ending_date` (inclusive) is a walk over days, streamed a page at a time. Future "
                "days are skipped; ranges over a year are refused.\n\n"
                "### Nested detail\n\n"
                "Token and cost totals are summed across `model_breakdown`; "
                "`UNNEST(model_breakdown)` gives one row per model. `tool_actions` maps a tool "
                "name to accepted/rejected counts, e.g. `tool_actions['edit_tool'].accepted`.\n\n"
                "### Cost\n\n"
                "`estimated_cost_*` is an estimate, not billing; subscription seats are not "
                "billed per token."
            ),
            example_queries=examples(
                ("Last week's Claude Code leaderboard", _CLAUDE_CODE_SQL),
                (
                    "Edit acceptance rate by day",
                    "SELECT date, sum(tool_actions['edit_tool'].accepted) / "
                    "nullif(sum(tool_actions['edit_tool'].accepted + tool_actions['edit_tool'].rejected), 0) "
                    "AS acceptance FROM anthropic.main.claude_code_usage(current_date - INTERVAL 14 DAY, "
                    "ending_date => current_date) GROUP BY date ORDER BY date",
                ),
            ),
        )
        examples = [FunctionExample(sql=_CLAUDE_CODE_SQL, description="Last week's Claude Code leaderboard")]

    @classmethod
    def on_bind(cls, params: BindParams[ClaudeCodeArgs]) -> BindResponse:
        # Validate the range at bind, so a bad one fails before any request.
        if params.args is not None:
            claude_code_days(params.args)
        return BindResponse(output_schema=cls.FIXED_SCHEMA)

    @classmethod
    def initial_state(cls, params: ProcessParams[ClaudeCodeArgs]) -> WalkState:
        return WalkState()

    @classmethod
    def process(cls, params: ProcessParams[ClaudeCodeArgs], state: WalkState, out: OutputCollector) -> None:
        emit_walk(
            params,
            state,
            out,
            keys=lambda: claude_code_days(params.args),
            endpoint_for=lambda _key: api.CLAUDE_CODE_USAGE,
            query=lambda: [],
            key_query=lambda day: [("starting_at", day)],
            flatten=lambda rows, _key: [flatten_claude_code(r) for r in rows],
            cache_control=_opt_in_cache(params.args.cache_ttl),
        )


REPORT_FUNCTIONS: list[type] = [MessagesUsageFunction, CostReportFunction, ClaudeCodeUsageFunction]
