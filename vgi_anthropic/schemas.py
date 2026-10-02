"""Arrow schemas, and the flattening of Admin API payloads into rows.

Two conventions run through every schema here:

* **Money is DECIMAL, never DOUBLE.** The cost report sends amounts as decimal
  *strings* in cents (``"123.78912"``); parsing them through a float would be a
  silent precision loss on values that arrive exact. Every amount is exposed
  twice — in the API's own unit (``*_cents``) and in dollars (``*_usd``) — so
  nobody has to remember which one the endpoint used.
* **Every conversion is total.** One malformed value in one row must never fail
  the whole batch: it becomes NULL, visible in the result and not fatal.

The usage and cost reports nest results inside time buckets; they are flattened
here to one row per (bucket, result), with the bucket's bounds on every row.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, cast

import pyarrow as pa

from vgi_anthropic.meta import field

#: RFC 3339 instants, carried at microsecond resolution in UTC.
TIMESTAMP = pa.timestamp("us", tz="UTC")

#: Cost amounts in the API's unit, cents. The API sends up to five fractional
#: digits of a cent; twelve leaves room without risking a NULL.
CENTS = pa.decimal128(38, 12)

#: The same amount in dollars: two more digits of scale, so the division by 100
#: is exact.
USD = pa.decimal128(38, 14)

STRINGS = pa.list_(pa.string())

#: The window a nanosecond-resolution consumer can hold (pandas, numpy
#: ``datetime64[ns]``). A value outside it raises ``OverflowError`` when a
#: client materializes it, which is worse than NULL.
_NS_FLOOR = datetime(1678, 1, 1, tzinfo=UTC)
_NS_CEILING = datetime(2262, 1, 1, tzinfo=UTC)


# --------------------------------------------------------------------------
# Total value converters
# --------------------------------------------------------------------------


def to_decimal(value: Any) -> Decimal | None:
    """Parse a decimal string or number into a Decimal, or None if absent/unparseable.

    Floats go through ``str`` first so ``0.1`` becomes ``Decimal('0.1')`` and not
    its binary expansion. ``NaN`` and ``Infinity`` are rejected — Arrow would.
    """
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def fits_decimal(value: Decimal, kind: pa.Decimal128Type) -> bool:
    """Whether ``value`` is exactly representable in ``kind``.

    Arrow raises on a value needing more scale or precision than declared, and
    one such value would take down the scan. Rounding to fit would be silent
    precision loss, so a value that does not fit becomes NULL instead.
    """
    shifted = value.scaleb(kind.scale)
    if shifted != shifted.to_integral_value():
        return False
    return abs(shifted) < Decimal(10) ** kind.precision


def to_timestamp(value: Any) -> datetime | None:
    """Parse an RFC 3339 string into an aware UTC datetime, or None.

    Total: parsing errors and values outside what a nanosecond-resolution client
    can hold both become NULL rather than raising.
    """
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, OverflowError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    if not _NS_FLOOR <= parsed <= _NS_CEILING:
        return None
    return parsed


def to_date(value: Any) -> date | None:
    """A calendar date from ``YYYY-MM-DD`` or an RFC 3339 instant (its UTC day)."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC).date()
    if isinstance(value, date):
        return value
    text = str(value)
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    instant = to_timestamp(text)
    return instant.astimezone(UTC).date() if instant else None


def to_integer(value: Any) -> int | None:
    """An integer, or None when the value is not exactly one.

    Stricter than ``int()``: JSON numbers arrive as floats when they carry a
    fractional part, and ``int(0.5)`` truncates to 0 — a wrong answer, not an
    absent one. A non-integral value becomes NULL instead.
    """
    if value is None or isinstance(value, bool) or value == "":
        return None
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    try:
        return int(value)
    except (TypeError, ValueError):
        parsed = to_decimal(value)
        if parsed is not None and parsed == parsed.to_integral_value():
            return int(parsed)
        return None


def column(rows: Sequence[dict[str, Any]], key: str, f: pa.Field) -> pa.Array:
    """Extract ``key`` from every row and build the Arrow array ``f`` declares.

    The conversion is chosen from the declared type, so a schema edit is the
    only thing needed to change how a column is parsed. Every branch is total.
    """
    values = [row.get(key) for row in rows]
    kind = f.type
    if pa.types.is_decimal(kind):
        dkind = cast("pa.Decimal128Type", kind)
        parsed = (to_decimal(v) for v in values)
        return pa.array([d if d is not None and fits_decimal(d, dkind) else None for d in parsed], type=kind)
    if pa.types.is_timestamp(kind):
        return pa.array([to_timestamp(v) for v in values], type=kind)
    if pa.types.is_date(kind):
        return pa.array([to_date(v) for v in values], type=kind)
    if pa.types.is_boolean(kind):
        return pa.array([v if isinstance(v, bool) else None for v in values], type=kind)
    if pa.types.is_integer(kind):
        return pa.array([to_integer(v) for v in values], type=kind)
    if pa.types.is_string(kind):
        return pa.array([None if v is None else str(v) for v in values], type=kind)
    # Nested: walk the declared type so leaves get the same total parsing, then
    # fall back to isolating rows Arrow still rejects.
    coerced = [_coerce_nested(v, kind) for v in values]
    try:
        return pa.array(coerced, type=kind)
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError):
        return pa.array([_nullable_nested(v, kind) for v in coerced], type=kind)


def _coerce_nested(value: Any, kind: pa.DataType) -> Any:
    """Convert a nested payload's leaves to what ``kind`` declares."""
    if value is None:
        return None
    if pa.types.is_map(kind):
        if not isinstance(value, dict):
            return None
        mkind = cast("pa.MapType", kind)
        return [(str(k), _coerce_nested(v, mkind.item_type)) for k, v in value.items() if k is not None]
    if pa.types.is_list(kind) or pa.types.is_large_list(kind):
        if not isinstance(value, (list, tuple)):
            return None
        return [_coerce_nested(item, kind.value_type) for item in value]
    if pa.types.is_struct(kind):
        if not isinstance(value, dict):
            return None
        return {sub.name: _coerce_nested(value.get(sub.name), sub.type) for sub in kind}
    if pa.types.is_decimal(kind):
        parsed = to_decimal(value)
        ok = parsed is not None and fits_decimal(parsed, cast("pa.Decimal128Type", kind))
        return parsed if ok else None
    if pa.types.is_timestamp(kind):
        return to_timestamp(value)
    if pa.types.is_integer(kind):
        return to_integer(value)
    if pa.types.is_boolean(kind):
        return value if isinstance(value, bool) else None
    if pa.types.is_string(kind):
        return str(value)
    return value


def _nullable_nested(value: Any, kind: pa.DataType) -> Any:
    """``value`` if Arrow accepts it alone, else None."""
    if value is None:
        return None
    try:
        pa.array([value], type=kind)
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError):
        return None
    return value


def batch_from_rows(rows: Sequence[dict[str, Any]], schema: pa.Schema) -> pa.RecordBatch:
    """Build one RecordBatch by pulling each schema field out of ``rows`` by name."""
    return pa.RecordBatch.from_arrays([column(rows, f.name, f) for f in schema], schema=schema)


def cents_to_usd(cents: Any) -> Decimal | None:
    """An amount in cents as dollars, exactly."""
    parsed = to_decimal(cents)
    return None if parsed is None else parsed.scaleb(-2)


def _dig(value: Any, *keys: str) -> Any:
    """``value[k1][k2]...``, or None at the first missing or non-dict step."""
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


# --------------------------------------------------------------------------
# Organization, members, invites
# --------------------------------------------------------------------------

ORGANIZATION_SCHEMA = pa.schema(
    [
        field("id", pa.string(), "Organization ID (a UUID).", nullable=False),
        field("name", pa.string(), "Organization name."),
    ]
)

_ROLE_DOC = (
    "Organization role. Console/API organizations use user, developer, billing, admin and "
    "claude_code_user; Claude Enterprise organizations use user, owner, primary_owner, "
    "membership_admin and managed."
)

USER_SCHEMA = pa.schema(
    [
        field("id", pa.string(), "User ID (user_...). The key every per-user lookup takes.", nullable=False),
        field("email", pa.string(), "The member's email address."),
        field("name", pa.string(), "The member's display name."),
        field("role", pa.string(), _ROLE_DOC),
        field("added_at", TIMESTAMP, "When the member joined the organization."),
    ]
)

INVITE_SCHEMA = pa.schema(
    [
        field("id", pa.string(), "Invite ID (invite_...).", nullable=False),
        field("email", pa.string(), "The address the invite was sent to."),
        field("role", pa.string(), "Organization role the invitee will receive. " + _ROLE_DOC),
        field(
            "status",
            pa.string(),
            "pending, accepted, expired or deleted. An expired invite still lists until removed.",
        ),
        field("invited_at", TIMESTAMP, "When the invite was sent."),
        field("expires_at", TIMESTAMP, "When the invite stops being redeemable."),
        field("accepted_at", TIMESTAMP, "When the invite was accepted; NULL if it has not been."),
        field(
            "rbac_group_ids",
            STRINGS,
            "RBAC groups the user joins on acceptance (Claude Enterprise); empty when none.",
        ),
    ]
)


# --------------------------------------------------------------------------
# Workspaces
# --------------------------------------------------------------------------

WORKSPACE_SCHEMA = pa.schema(
    [
        field("id", pa.string(), "Workspace ID (wrkspc_...).", nullable=False),
        field("name", pa.string(), "Workspace name."),
        field("created_at", TIMESTAMP, "When the workspace was created."),
        field("archived_at", TIMESTAMP, "When the workspace was archived; NULL while it is active."),
        field("is_archived", pa.bool_(), "Whether the workspace is archived (archived_at IS NOT NULL)."),
        field("display_color", pa.string(), "Hex colour the Console shows the workspace in."),
        field("tags", pa.map_(pa.string(), pa.string()), "User-defined key/value tags."),
        field(
            "workspace_geo",
            pa.string(),
            "Region where the workspace's data is stored (immutable after creation).",
        ),
        field(
            "default_inference_geo",
            pa.string(),
            "Inference geo applied when a request does not name one: global or us.",
        ),
        field(
            "allowed_inference_geos",
            STRINGS,
            "Inference geos requests may name. NULL when unrestricted — see inference_geo_unrestricted.",
        ),
        field(
            "inference_geo_unrestricted",
            pa.bool_(),
            "True when every inference geo is allowed (the API's 'unrestricted').",
        ),
        field(
            "external_key_id",
            pa.string(),
            "Customer-managed encryption key (CMEK) attached to the workspace; NULL when none.",
        ),
        field("compartment_id", pa.string(), "Encryption compartment ID, referenced by CMEK key policies."),
    ]
)


def flatten_workspace(raw: dict[str, Any]) -> dict[str, Any]:
    """One workspace payload as a row."""
    residency = raw.get("data_residency") if isinstance(raw.get("data_residency"), dict) else {}
    allowed = residency.get("allowed_inference_geos")
    unrestricted = allowed == "unrestricted"
    return {
        **raw,
        "is_archived": raw.get("archived_at") is not None,
        "workspace_geo": residency.get("workspace_geo"),
        "default_inference_geo": residency.get("default_inference_geo"),
        "allowed_inference_geos": allowed if isinstance(allowed, list) else None,
        "inference_geo_unrestricted": unrestricted if allowed is not None else None,
    }


WORKSPACE_MEMBER_SCHEMA = pa.schema(
    [
        field("workspace_id", pa.string(), "Workspace the membership is in.", nullable=False),
        field("user_id", pa.string(), "Member's user ID; joins to users.id.", nullable=False),
        field(
            "workspace_role",
            pa.string(),
            "workspace_user, workspace_developer, workspace_restricted_developer, "
            "workspace_admin or workspace_billing.",
        ),
    ]
)


# --------------------------------------------------------------------------
# API keys
# --------------------------------------------------------------------------

API_KEY_SCHEMA = pa.schema(
    [
        field(
            "id",
            pa.string(),
            "API key ID (apikey_...). Not the secret, which the API never returns.",
            nullable=False,
        ),
        field("name", pa.string(), "The key's name."),
        field("status", pa.string(), "active, inactive, archived or expired."),
        field("created_at", TIMESTAMP, "When the key was created."),
        field("expires_at", TIMESTAMP, "When the key expires; NULL if it never does."),
        field(
            "partial_key_hint", pa.string(), "Redacted hint of the key value, e.g. sk-ant-api03-R2D...igAA."
        ),
        field(
            "workspace_id",
            pa.string(),
            "Workspace the key belongs to — its real ID even for the default workspace. NULL for an "
            "organization-scoped key (scope_type = 'organization').",
        ),
        field(
            "scope_type",
            pa.string(),
            "workspace, or organization for a principal-bound key with no workspace.",
        ),
        field(
            "created_by_type", pa.string(), "user or service_account; NULL when the creator is not recorded."
        ),
        field("created_by_id", pa.string(), "ID of the user or service account that created the key."),
        field(
            "principal_type",
            pa.string(),
            "user_actor or service_account_actor; NULL when not principal-bound.",
        ),
        field("principal_user_id", pa.string(), "User the key acts as, when principal_type = 'user_actor'."),
        field(
            "principal_service_account_id",
            pa.string(),
            "Service account the key acts as, when principal_type = 'service_account_actor'.",
        ),
    ]
)


def flatten_api_key(raw: dict[str, Any]) -> dict[str, Any]:
    """One API key payload as a row.

    ``workspace_id`` comes from ``scope``, not the deprecated top-level field,
    which is NULL for the default workspace and so cannot be joined on.
    """
    return {
        **raw,
        "workspace_id": _dig(raw, "scope", "workspace_id"),
        "scope_type": _dig(raw, "scope", "type"),
        "created_by_type": _dig(raw, "created_by", "type"),
        "created_by_id": _dig(raw, "created_by", "id"),
        "principal_type": _dig(raw, "principal", "type"),
        "principal_user_id": _dig(raw, "principal", "user_id"),
        "principal_service_account_id": _dig(raw, "principal", "service_account_id"),
    }


# --------------------------------------------------------------------------
# Rate limits
# --------------------------------------------------------------------------

_GROUP_TYPE_DOC = (
    "model_group (a model family, listed in models), batch, files, skills, token_count or web_search."
)

RATE_LIMIT_SCHEMA = pa.schema(
    [
        field("id", pa.string(), "The organization's rate-limit entry ID.", nullable=False),
        field("group_id", pa.string(), "Rate-limit group ID (rlg_...), the same in every organization."),
        field("group_type", pa.string(), _GROUP_TYPE_DOC),
        field("group_display_name", pa.string(), "Model group name, e.g. 'Claude Sonnet 4.x'. Display only."),
        field("models", STRINGS, "Models and aliases the limit applies to; NULL for non-model groups."),
        field(
            "limit_type",
            pa.string(),
            "The limiter, e.g. requests_per_minute, input_tokens_per_minute, output_tokens_per_minute.",
            nullable=False,
        ),
        field(
            "value",
            pa.int64(),
            "The configured limit, in the limiter's own unit: requests or tokens per minute.",
        ),
    ]
)

WORKSPACE_RATE_LIMIT_SCHEMA = pa.schema(
    [
        field("workspace_id", pa.string(), "Workspace the limit applies to.", nullable=False),
        field(
            "rate_limit_id",
            pa.string(),
            "The organization rate-limit entry this refines; joins to rate_limits.id.",
        ),
        field("group_id", pa.string(), "Rate-limit group ID (rlg_...)."),
        field("group_type", pa.string(), _GROUP_TYPE_DOC),
        field("group_display_name", pa.string(), "Model group name. Display only."),
        field("models", STRINGS, "Models and aliases the limit applies to; NULL for non-model groups."),
        field("limit_type", pa.string(), "The limiter, e.g. requests_per_minute.", nullable=False),
        field(
            "value",
            pa.int64(),
            "The limit in force for this workspace, in the limiter's unit (requests or tokens per minute).",
        ),
        field(
            "org_limit",
            pa.int64(),
            "The organization-level value for the same limiter (same unit as value), for comparison.",
        ),
        field(
            "source",
            pa.string(),
            "workspace when the workspace overrides the limit; organization when it inherits it.",
        ),
    ]
)


def _group_fields(entry: dict[str, Any]) -> dict[str, Any]:
    group = entry.get("group") if isinstance(entry.get("group"), dict) else {}
    return {
        "group_id": group.get("id"),
        "group_type": group.get("type") or entry.get("group_type"),
        "group_display_name": group.get("display_name"),
        "models": entry.get("models"),
    }


def flatten_rate_limits(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (entry, limiter type)."""
    rows: list[dict[str, Any]] = []
    for entry in entries:
        common = {"id": entry.get("id"), **_group_fields(entry)}
        for limit in entry.get("limits") or []:
            if isinstance(limit, dict):
                rows.append({**common, "limit_type": limit.get("type"), "value": limit.get("value")})
    return rows


def flatten_workspace_rate_limits(
    entries: Iterable[dict[str, Any]], workspace_id: str
) -> list[dict[str, Any]]:
    """One row per (workspace entry, limiter type)."""
    rows: list[dict[str, Any]] = []
    for entry in entries:
        common = {
            "workspace_id": entry.get("workspace_id") or workspace_id,
            "rate_limit_id": entry.get("rate_limit_id"),
            **_group_fields(entry),
        }
        for limit in entry.get("limits") or []:
            if isinstance(limit, dict):
                rows.append(
                    {
                        **common,
                        "limit_type": limit.get("type"),
                        "value": limit.get("value"),
                        "org_limit": limit.get("org_limit"),
                        "source": _dig(limit, "source", "type"),
                    }
                )
    return rows


# --------------------------------------------------------------------------
# Usage and cost reports
# --------------------------------------------------------------------------

_BUCKET_FIELDS = [
    field("bucket_start", TIMESTAMP, "Start of the time bucket (inclusive), UTC.", nullable=False),
    field("bucket_end", TIMESTAMP, "End of the time bucket (exclusive), UTC.", nullable=False),
]

_DEFAULT_WORKSPACE_NOTE = " NULL for the default workspace, and when not grouped by it."

#: Usage dimensions in API order, with what each column means. A dimension
#: column is NULL unless the report is grouped by it — see
#: :mod:`vgi_anthropic.reports` for how grouping follows the query.
USAGE_DIMENSIONS: dict[str, str] = {
    "workspace_id": "Workspace the usage ran in." + _DEFAULT_WORKSPACE_NOTE,
    "api_key_id": "API key used; NULL for Console usage and when not grouped by it.",
    "account_id": "User account (OAuth) that made the requests; NULL when not grouped by it.",
    "service_account_id": "Service account that made the requests; NULL when not grouped by it.",
    "model": "Model used; NULL when not grouped by it.",
    "service_tier": "standard, batch, priority, priority_on_demand, flex or flex_discount.",
    "context_window": "0-200k or 200k-1M.",
    "inference_geo": "global, us, or not_available for models without geo routing.",
    "speed": "standard or fast (fast mode). Only populated with group_by => ['speed'].",
}

MESSAGES_USAGE_SCHEMA = pa.schema(
    [
        *_BUCKET_FIELDS,
        *(field(name, pa.string(), doc) for name, doc in USAGE_DIMENSIONS.items()),
        field("uncached_input_tokens", pa.int64(), "Input tokens processed without a cache hit."),
        field("cache_read_input_tokens", pa.int64(), "Input tokens read from the prompt cache."),
        field(
            "cache_creation_5m_input_tokens",
            pa.int64(),
            "Input tokens written to a 5-minute cache entry.",
        ),
        field(
            "cache_creation_1h_input_tokens",
            pa.int64(),
            "Input tokens written to a 1-hour cache entry.",
        ),
        field("output_tokens", pa.int64(), "Output tokens generated."),
        field("web_search_requests", pa.int64(), "Server-side web search tool requests."),
    ]
)


def flatten_usage(buckets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (bucket, result). Buckets with no usage yield no rows."""
    rows: list[dict[str, Any]] = []
    for bucket in buckets:
        bounds = {"bucket_start": bucket.get("starting_at"), "bucket_end": bucket.get("ending_at")}
        for result in bucket.get("results") or []:
            if not isinstance(result, dict):
                continue
            rows.append(
                {
                    **bounds,
                    **result,
                    "cache_creation_5m_input_tokens": _dig(
                        result, "cache_creation", "ephemeral_5m_input_tokens"
                    ),
                    "cache_creation_1h_input_tokens": _dig(
                        result, "cache_creation", "ephemeral_1h_input_tokens"
                    ),
                    "web_search_requests": _dig(result, "server_tool_use", "web_search_requests"),
                }
            )
    return rows


#: Columns of the cost report populated only when grouped by ``description``.
COST_DESCRIPTION_COLUMNS = (
    "description",
    "cost_type",
    "model",
    "service_tier",
    "token_type",
    "context_window",
    "inference_geo",
)

COST_REPORT_SCHEMA = pa.schema(
    [
        *_BUCKET_FIELDS,
        field("workspace_id", pa.string(), "Workspace the cost accrued in." + _DEFAULT_WORKSPACE_NOTE),
        field(
            "description",
            pa.string(),
            "Line-item description, e.g. 'Claude Opus 5 Usage - Input Tokens'. This and the columns "
            "after it are NULL unless the report is grouped by description.",
        ),
        field("cost_type", pa.string(), "tokens, web_search, code_execution or session_usage."),
        field("model", pa.string(), "Model the cost is for; NULL for non-token costs."),
        field("service_tier", pa.string(), "standard or batch; NULL for non-token costs."),
        field(
            "token_type",
            pa.string(),
            "uncached_input_tokens, output_tokens, cache_read_input_tokens, "
            "cache_creation.ephemeral_5m_input_tokens or cache_creation.ephemeral_1h_input_tokens.",
        ),
        field("context_window", pa.string(), "0-200k or 200k-1M; NULL for non-token costs."),
        field("inference_geo", pa.string(), "global, us or not_available."),
        field("currency", pa.string(), "Currency of the amount; currently always USD."),
        field(
            "amount_cents", CENTS, "Cost in cents, exactly as the API reports it (a decimal, not a float)."
        ),
        field("amount_usd", USD, "The same cost in dollars: amount_cents / 100, exact."),
    ]
)


def flatten_cost(buckets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (bucket, cost item), with the amount in cents and dollars."""
    rows: list[dict[str, Any]] = []
    for bucket in buckets:
        bounds = {"bucket_start": bucket.get("starting_at"), "bucket_end": bucket.get("ending_at")}
        for result in bucket.get("results") or []:
            if not isinstance(result, dict):
                continue
            amount = result.get("amount")
            rows.append({**bounds, **result, "amount_cents": amount, "amount_usd": cents_to_usd(amount)})
    return rows


# --------------------------------------------------------------------------
# Claude Code usage
# --------------------------------------------------------------------------

MODEL_BREAKDOWN = pa.struct(
    [
        ("model", pa.string()),
        ("input_tokens", pa.int64()),
        ("output_tokens", pa.int64()),
        ("cache_read_tokens", pa.int64()),
        ("cache_creation_tokens", pa.int64()),
        ("estimated_cost_cents", CENTS),
        ("estimated_cost_usd", USD),
        ("currency", pa.string()),
    ]
)

TOOL_ACTION = pa.struct([("accepted", pa.int64()), ("rejected", pa.int64())])

CLAUDE_CODE_USAGE_SCHEMA = pa.schema(
    [
        field("date", pa.date32(), "UTC day the metrics cover.", nullable=False),
        field("actor_type", pa.string(), "user_actor (a person) or api_actor (an API key)."),
        field("email_address", pa.string(), "The user's email, for user_actor rows."),
        field("api_key_name", pa.string(), "The API key's name, for api_actor rows."),
        field("organization_id", pa.string(), "Organization that owns the usage."),
        field("customer_type", pa.string(), "api (API customers) or subscription (Pro/Team/Enterprise)."),
        field(
            "subscription_type", pa.string(), "team or enterprise for subscription customers; NULL for api."
        ),
        field(
            "terminal_type",
            pa.string(),
            "Terminal or environment Claude Code ran in, e.g. iTerm.app, vscode.",
        ),
        field(
            "is_remote",
            pa.bool_(),
            "Whether the usage came from remote sessions (e.g. Claude Code on the web). Remote and "
            "local usage are separate rows.",
        ),
        field("num_sessions", pa.int64(), "Distinct Claude Code sessions."),
        field("lines_added", pa.int64(), "Lines of code added by Claude Code."),
        field("lines_removed", pa.int64(), "Lines of code removed by Claude Code."),
        field("commits", pa.int64(), "Git commits created through Claude Code."),
        field("pull_requests", pa.int64(), "Pull requests created through Claude Code."),
        field("input_tokens", pa.int64(), "Input tokens, summed across models."),
        field("output_tokens", pa.int64(), "Output tokens, summed across models."),
        field("cache_read_tokens", pa.int64(), "Cache-read tokens, summed across models."),
        field("cache_creation_tokens", pa.int64(), "Cache-creation tokens, summed across models."),
        field("estimated_cost_cents", CENTS, "Estimated cost in cents, summed across models."),
        field("estimated_cost_usd", USD, "Estimated cost in dollars, summed across models."),
        field(
            "model_breakdown",
            pa.list_(MODEL_BREAKDOWN),
            "Per-model tokens and estimated cost; UNNEST it for one row per model.",
        ),
        field(
            "tool_actions",
            pa.map_(pa.string(), TOOL_ACTION),
            "Accepted/rejected tool proposals by tool, e.g. tool_actions['edit_tool'].accepted.",
        ),
    ]
)


def _sum(values: Iterable[Any], parse: Any) -> Any:
    """Sum parsed values, NULL if any is unparseable — a partial sum would be a wrong total."""
    total: Any = 0
    for value in values:
        parsed = parse(value)
        if parsed is None:
            return None
        total += parsed
    return total


def flatten_claude_code(record: dict[str, Any]) -> dict[str, Any]:
    """One Claude Code usage record as a row, with per-model totals rolled up."""
    breakdown = []
    for item in record.get("model_breakdown") or []:
        if not isinstance(item, dict):
            continue
        cents = _dig(item, "estimated_cost", "amount")
        breakdown.append(
            {
                "model": item.get("model"),
                "input_tokens": _dig(item, "tokens", "input"),
                "output_tokens": _dig(item, "tokens", "output"),
                "cache_read_tokens": _dig(item, "tokens", "cache_read"),
                "cache_creation_tokens": _dig(item, "tokens", "cache_creation"),
                "estimated_cost_cents": cents,
                "estimated_cost_usd": cents_to_usd(cents),
                "currency": _dig(item, "estimated_cost", "currency"),
            }
        )
    total_cents = _sum((b["estimated_cost_cents"] for b in breakdown), to_decimal)
    tools = record.get("tool_actions")
    return {
        "date": record.get("date"),
        "actor_type": _dig(record, "actor", "type"),
        "email_address": _dig(record, "actor", "email_address"),
        "api_key_name": _dig(record, "actor", "api_key_name"),
        "organization_id": record.get("organization_id"),
        "customer_type": record.get("customer_type"),
        "subscription_type": record.get("subscription_type"),
        "terminal_type": record.get("terminal_type"),
        "is_remote": record.get("is_remote"),
        "num_sessions": _dig(record, "core_metrics", "num_sessions"),
        "lines_added": _dig(record, "core_metrics", "lines_of_code", "added"),
        "lines_removed": _dig(record, "core_metrics", "lines_of_code", "removed"),
        "commits": _dig(record, "core_metrics", "commits_by_claude_code"),
        "pull_requests": _dig(record, "core_metrics", "pull_requests_by_claude_code"),
        "input_tokens": _sum((b["input_tokens"] for b in breakdown), to_integer),
        "output_tokens": _sum((b["output_tokens"] for b in breakdown), to_integer),
        "cache_read_tokens": _sum((b["cache_read_tokens"] for b in breakdown), to_integer),
        "cache_creation_tokens": _sum((b["cache_creation_tokens"] for b in breakdown), to_integer),
        "estimated_cost_cents": total_cents,
        "estimated_cost_usd": None if total_cents is None else Decimal(total_cents).scaleb(-2),
        "model_breakdown": breakdown,
        "tool_actions": tools if isinstance(tools, dict) else None,
    }
