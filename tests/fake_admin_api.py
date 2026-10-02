"""A local fake of the Anthropic Admin API, faithful to the documented contracts.

It exists so the end-to-end tests can run a real ``ATTACH`` — DuckDB, the vgi
extension and a worker subprocess — without an Admin key or network access.
Argument casting, projection and filter pushdown are performed by the
extension, not by anything a unit test can call, so this is the only offline
way to exercise them.

The fake implements the parts of each endpoint the worker depends on: both
paging styles, ``group_by[]`` aggregation with NULL for ungrouped dimensions,
the documented filters, and ``404`` for unknown IDs. Page sizes are capped at
:data:`PAGE_CAP` so every list actually pages. Every request is recorded in
:attr:`FakeAdminApi.requests` so tests can assert what was — and was not —
sent upstream.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

ADMIN_KEY = "sk-ant-admin01-test-key"

#: A personal key scoped to the whole organization. The Admin API accepts it
#: although it carries the ordinary `sk-ant-api03-` prefix.
ORG_SCOPED_KEY = "sk-ant-api03-org-scoped-test-key"

#: A personal key bound to one workspace. Verified against the real API: it can
#: read `/v1/organizations/me` and is refused with this 401 everywhere else.
WORKSPACE_SCOPED_KEY = "sk-ant-api03-workspace-scoped-test-key"
WORKSPACE_KEY_REFUSAL = "The Admin API requires an Admin API key or an organization-scoped API key."

#: Every list endpoint returns at most this many objects per page, whatever
#: `limit` asks for, so a fixture of more than this many rows is paged.
PAGE_CAP = 2

ORG = {"id": "12345678-1234-5678-1234-567812345678", "name": "Acme Robotics", "type": "organization"}

USERS = [
    {
        "id": f"user_{i:02d}",
        "email": f"person{i}@acme.test",
        "name": f"Person {i}",
        "role": role,
        "added_at": f"2025-0{i}-01T00:00:00Z",
        "type": "user",
    }
    for i, role in enumerate(["admin", "developer", "developer", "billing", "user"], start=1)
]

INVITES = [
    {
        "id": "invite_01",
        "email": "new1@acme.test",
        "role": "developer",
        "status": "pending",
        "invited_at": "2026-09-01T00:00:00Z",
        "expires_at": "2026-09-22T00:00:00Z",
        "accepted_at": None,
        "rbac_group_ids": [],
        "type": "invite",
    },
    {
        "id": "invite_02",
        "email": "new2@acme.test",
        "role": "user",
        "status": "accepted",
        "invited_at": "2026-08-01T00:00:00Z",
        "expires_at": "2026-08-22T00:00:00Z",
        "accepted_at": "2026-08-02T00:00:00Z",
        "rbac_group_ids": ["rbac_1"],
        "type": "invite",
    },
    {
        "id": "invite_03",
        "email": "new3@acme.test",
        "role": "user",
        "status": "expired",
        "invited_at": "2026-07-01T00:00:00Z",
        "expires_at": "2026-07-22T00:00:00Z",
        "accepted_at": None,
        "rbac_group_ids": [],
        "type": "invite",
    },
]


def _workspace(i: int, name: str, archived: str | None, geos: Any) -> dict[str, Any]:
    return {
        "id": f"wrkspc_{i:02d}",
        "name": name,
        "archived_at": archived,
        "type": "workspace",
        "created_at": f"2025-0{i}-15T00:00:00Z",
        "display_color": "#6C5BB9",
        "compartment_id": f"compartment-{i}",
        "external_key_id": None,
        "tags": {"team": name.lower()},
        "data_residency": {
            "allowed_inference_geos": geos,
            "default_inference_geo": "global",
            "workspace_geo": "us",
        },
    }


WORKSPACES = [
    _workspace(1, "Prod", None, "unrestricted"),
    _workspace(2, "Staging", None, ["us"]),
    _workspace(3, "Legacy", "2026-01-01T00:00:00Z", "unrestricted"),
]

MEMBERS = {
    "wrkspc_01": [
        ("user_01", "workspace_admin"),
        ("user_02", "workspace_developer"),
        ("user_03", "workspace_developer"),
    ],
    "wrkspc_02": [("user_02", "workspace_admin")],
    "wrkspc_03": [("user_01", "workspace_admin")],
}

API_KEYS = [
    {
        "id": "apikey_01",
        "name": "prod-backend",
        "status": "active",
        "created_at": "2025-03-01T00:00:00Z",
        "expires_at": None,
        "partial_key_hint": "sk-ant-api03-AAA...zzzz",
        "created_by": {"id": "user_01", "type": "user"},
        "principal": None,
        "scope": {"type": "workspace", "workspace_id": "wrkspc_01"},
        "workspace_id": "wrkspc_01",
        "type": "api_key",
    },
    {
        "id": "apikey_02",
        "name": "staging",
        "status": "active",
        "created_at": "2025-04-01T00:00:00Z",
        "expires_at": "2027-01-01T00:00:00Z",
        "partial_key_hint": "sk-ant-api03-BBB...yyyy",
        "created_by": {"id": "svac_01", "type": "service_account"},
        "principal": {"type": "service_account_actor", "service_account_id": "svac_01"},
        "scope": {"type": "workspace", "workspace_id": "wrkspc_02"},
        "workspace_id": "wrkspc_02",
        "type": "api_key",
    },
    {
        "id": "apikey_03",
        "name": "old-laptop",
        "status": "archived",
        "created_at": "2024-11-01T00:00:00Z",
        "expires_at": None,
        "partial_key_hint": "sk-ant-api03-CCC...xxxx",
        "created_by": {"id": "user_02", "type": "user"},
        "principal": {"type": "user_actor", "user_id": "user_02"},
        "scope": {"type": "workspace", "workspace_id": "wrkspc_default"},
        "workspace_id": None,
        "type": "api_key",
    },
]

RATE_LIMITS = [
    {
        "id": "rl_01",
        "type": "rate_limit",
        "group_type": "model_group",
        "group": {"id": "rlg_sonnet", "type": "model_group", "display_name": "Claude Sonnet 5.x"},
        "models": ["claude-sonnet-5-5", "claude-sonnet-5"],
        "limits": [
            {"type": "requests_per_minute", "value": 4000},
            {"type": "output_tokens_per_minute", "value": 400000},
        ],
    },
    {
        "id": "rl_02",
        "type": "rate_limit",
        "group_type": "batch",
        "group": {"id": "rlg_batch", "type": "batch"},
        "models": None,
        "limits": [{"type": "requests_per_minute", "value": 1000}],
    },
]

#: Usage is generated per (model, workspace, api_key) so totals are the same
#: however the report is grouped. Each combination contributes these tokens
#: to every bucket.
USAGE_CELLS = [
    {
        "model": "claude-opus-5-5",
        "workspace_id": "wrkspc_01",
        "api_key_id": "apikey_01",
        "output_tokens": 100,
    },
    {"model": "claude-opus-5-5", "workspace_id": None, "api_key_id": "apikey_03", "output_tokens": 20},
    {
        "model": "claude-sonnet-5-5",
        "workspace_id": "wrkspc_01",
        "api_key_id": "apikey_01",
        "output_tokens": 30,
    },
    {
        "model": "claude-sonnet-5-5",
        "workspace_id": "wrkspc_02",
        "api_key_id": "apikey_02",
        "output_tokens": 7,
    },
]
_USAGE_DIMS = [
    "account_id",
    "api_key_id",
    "context_window",
    "inference_geo",
    "model",
    "service_account_id",
    "service_tier",
    "speed",
    "workspace_id",
]
_USAGE_FILTERS = {"models[]": "model", "workspace_ids[]": "workspace_id", "api_key_ids[]": "api_key_id"}

#: Cost lines per bucket: (workspace, description, model, cents).
COST_CELLS = [
    ("wrkspc_01", "Claude Opus 5.5 Usage - Output Tokens", "claude-opus-5-5", "1234.56789"),
    (None, "Claude Opus 5.5 Usage - Output Tokens", "claude-opus-5-5", "0.00001"),
    ("wrkspc_02", "Claude Sonnet 5.5 Usage - Input Tokens", "claude-sonnet-5-5", "100"),
]

#: Cost columns populated only when grouped by description.
DESCRIBED_COLUMNS = (
    "description",
    "model",
    "cost_type",
    "token_type",
    "service_tier",
    "context_window",
    "inference_geo",
)

#: The fake's "now": report buckets stop here when no ending_at is given.
NOW = datetime(2026, 9, 10, tzinfo=UTC)


def claude_code_records(day: str) -> list[dict[str, Any]]:
    """Three records per day: two users and an API key."""

    def record(actor: dict[str, Any], sessions: int, remote: bool) -> dict[str, Any]:
        return {
            "date": f"{day}T00:00:00Z",
            "actor": actor,
            "organization_id": ORG["id"],
            "customer_type": "api",
            "subscription_type": None,
            "terminal_type": "vscode",
            "is_remote": remote,
            "core_metrics": {
                "num_sessions": sessions,
                "commits_by_claude_code": 2,
                "pull_requests_by_claude_code": 1,
                "lines_of_code": {"added": 50, "removed": 5},
            },
            "model_breakdown": [
                {
                    "model": "claude-opus-5-5",
                    "tokens": {"input": 1000, "output": 200, "cache_read": 10, "cache_creation": 5},
                    "estimated_cost": {"amount": 186, "currency": "USD"},
                },
                {
                    "model": "claude-sonnet-5-5",
                    "tokens": {"input": 500, "output": 50, "cache_read": 0, "cache_creation": 0},
                    "estimated_cost": {"amount": 42.5, "currency": "USD"},
                },
            ],
            "tool_actions": {
                "edit_tool": {"accepted": 9, "rejected": 1},
                "write_tool": {"accepted": 3, "rejected": 0},
            },
        }

    return [
        record({"type": "user_actor", "email_address": "person1@acme.test"}, 4, False),
        record({"type": "user_actor", "email_address": "person2@acme.test"}, 2, True),
        record({"type": "api_actor", "api_key_name": "ci-bot"}, 1, False),
    ]


def _instant(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _rfc(instant: datetime) -> str:
    return instant.strftime("%Y-%m-%dT%H:%M:%SZ")


class FakeAdminApi:
    """The server, its request log and its lifecycle."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, list[str]], dict[str, str]]] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> FakeAdminApi:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def paths(self) -> list[str]:
        return [path for path, _, _ in self.requests]

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # silence the default stderr log
                pass

            def do_GET(self) -> None:  # noqa: N802 - http.server's naming
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query, keep_blank_values=True)
                headers = {k.lower(): v for k, v in self.headers.items()}
                api.requests.append((parsed.path, query, headers))
                key = headers.get("x-api-key")
                if key == WORKSPACE_SCOPED_KEY and parsed.path != "/v1/organizations/me":
                    return self._send(
                        401,
                        {
                            "type": "error",
                            "error": {"type": "authentication_error", "message": WORKSPACE_KEY_REFUSAL},
                        },
                    )
                if key not in (ADMIN_KEY, ORG_SCOPED_KEY, WORKSPACE_SCOPED_KEY):
                    return self._send(
                        401,
                        {
                            "type": "error",
                            "error": {"type": "authentication_error", "message": "invalid x-api-key"},
                        },
                    )
                if headers.get("anthropic-version") != "2023-06-01":
                    return self._send(
                        400,
                        {
                            "type": "error",
                            "error": {"type": "invalid_request_error", "message": "anthropic-version"},
                        },
                    )
                status, body = api.route(parsed.path, query)
                self._send(status, body)

            def do_POST(self) -> None:  # noqa: N802
                api.requests.append((urlparse(self.path).path, {}, {"method": "POST"}))
                self._send(405, {"error": {"message": "the worker must never write"}})

            do_PUT = do_PATCH = do_DELETE = do_POST  # noqa: N815

            def _send(self, status: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                # Observed live: authentication failures (401) carry neither
                # header; everything else carries both.
                if status != 401:
                    self.send_header("request-id", "req_fake")
                    if status >= 400:
                        self.send_header("x-should-retry", "false")
                self.end_headers()
                self.wfile.write(data)

        return Handler

    # ------------------------------------------------------------------ routing

    def route(self, path: str, q: dict[str, list[str]]) -> tuple[int, dict[str, Any]]:
        one = {k: v[-1] for k, v in q.items()}
        parts = path.strip("/").split("/")
        if parts[:2] != ["v1", "organizations"]:
            return 404, {"error": {"message": "not found"}}
        rest = parts[2:]
        if rest == ["me"]:
            return 200, ORG
        if rest == ["users"]:
            rows = [u for u in USERS if not one.get("email") or u["email"].lower() == one["email"].lower()]
            return 200, self._id_page(rows, one)
        if len(rest) == 2 and rest[0] == "users":
            return self._by_id(USERS, rest[1])
        if rest == ["invites"]:
            rows = [i for i in INVITES if not one.get("email") or i["email"].lower() == one["email"].lower()]
            return 200, self._id_page(rows, one)
        if rest == ["workspaces"]:
            rows = [
                w for w in WORKSPACES if one.get("include_archived") == "true" or w["archived_at"] is None
            ]
            return 200, self._id_page(rows, one)
        if len(rest) == 2 and rest[0] == "workspaces":
            return self._by_id(WORKSPACES, rest[1])
        if len(rest) == 3 and rest[0] == "workspaces" and rest[2] == "members":
            if rest[1] not in MEMBERS:
                return 404, {"error": {"message": "workspace not found"}}
            rows = [
                {
                    "type": "workspace_member",
                    "user_id": u,
                    "workspace_id": rest[1],
                    "workspace_role": r,
                }
                for u, r in MEMBERS[rest[1]]
            ]
            # A membership has no `id` of its own; the cursor is the member's user ID.
            return 200, self._id_page(rows, one, key="user_id")
        if len(rest) == 3 and rest[0] == "workspaces" and rest[2] == "rate_limits":
            return 200, self._workspace_rate_limits(rest[1], one)
        if rest == ["api_keys"]:
            rows = [
                k
                for k in API_KEYS
                if (not one.get("status") or k["status"] == one["status"])
                and (
                    not one.get("created_by_user_id")
                    or (k["created_by"] or {}).get("id") == one["created_by_user_id"]
                )
                # Filters on scope, as documented: organization-scoped keys match no workspace.
                and (not one.get("workspace_id") or k["scope"].get("workspace_id") == one["workspace_id"])
            ]
            return 200, self._id_page(rows, one)
        if len(rest) == 2 and rest[0] == "api_keys":
            return self._by_id(API_KEYS, rest[1])
        if rest == ["rate_limits"]:
            rows = [
                r for r in RATE_LIMITS if not one.get("group_type") or r["group_type"] == one["group_type"]
            ]
            return 200, {"data": rows, "next_page": None}
        if rest == ["usage_report", "messages"]:
            return self._usage(q, one)
        if rest == ["cost_report"]:
            return self._cost(q, one)
        if rest == ["usage_report", "claude_code"]:
            return self._claude_code(one)
        return 404, {"error": {"message": f"no route {path}"}}

    @staticmethod
    def _by_id(rows: list[dict[str, Any]], key: str) -> tuple[int, dict[str, Any]]:
        for row in rows:
            if row["id"] == key:
                return 200, row
        return 404, {"type": "error", "error": {"type": "not_found_error", "message": f"{key} not found"}}

    @staticmethod
    def _id_page(rows: list[dict[str, Any]], one: dict[str, str], key: str = "id") -> dict[str, Any]:
        start = 0
        if after := one.get("after_id"):
            ids = [r[key] for r in rows]
            start = ids.index(after) + 1 if after in ids else len(rows)
        size = min(int(one.get("limit", 20)), PAGE_CAP)
        chunk = rows[start : start + size]
        return {
            "data": chunk,
            "has_more": start + size < len(rows),
            "first_id": chunk[0][key] if chunk else None,
            "last_id": chunk[-1][key] if chunk else None,
        }

    @staticmethod
    def _token_page(buckets: list[dict[str, Any]], one: dict[str, str], default: int) -> dict[str, Any]:
        start = int(one["page"].removeprefix("page_")) if one.get("page") else 0
        size = int(one.get("limit", default))
        chunk = buckets[start : start + size]
        more = start + size < len(buckets)
        return {"data": chunk, "has_more": more, "next_page": f"page_{start + size}" if more else None}

    def _workspace_rate_limits(self, workspace_id: str, one: dict[str, str]) -> dict[str, Any]:
        inherited = one.get("include_inherited") == "true"
        rows = []
        for entry in RATE_LIMITS:
            if one.get("group_type") and entry["group_type"] != one["group_type"]:
                continue
            limits = []
            for limit in entry["limits"]:
                override = workspace_id == "wrkspc_01" and limit["type"] == "requests_per_minute"
                if override:
                    limits.append(
                        {
                            "type": limit["type"],
                            "value": limit["value"] // 2,
                            "org_limit": limit["value"],
                            "source": {"type": "workspace"},
                        }
                    )
                elif inherited:
                    limits.append(
                        {
                            "type": limit["type"],
                            "value": limit["value"],
                            "org_limit": limit["value"],
                            "source": {"type": "organization"},
                        }
                    )
            if limits:
                rows.append(
                    {
                        "type": "workspace_rate_limit",
                        "workspace_id": workspace_id,
                        "rate_limit_id": entry["id"],
                        "group": entry["group"],
                        "group_type": entry["group_type"],
                        "models": entry["models"],
                        "limits": limits,
                    }
                )
        return {"data": rows, "next_page": None}

    @staticmethod
    def _buckets(one: dict[str, str], width: timedelta) -> list[tuple[datetime, datetime]]:
        start = _instant(one["starting_at"])
        if width == timedelta(days=1):
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = _instant(one["ending_at"]) if one.get("ending_at") else NOW
        out = []
        while start < end:
            out.append((start, start + width))
            start += width
        return out

    def _usage(self, q: dict[str, list[str]], one: dict[str, str]) -> tuple[int, dict[str, Any]]:
        if "starting_at" not in one:
            return 400, {"error": {"message": "starting_at is required"}}
        width = {"1d": timedelta(days=1), "1h": timedelta(hours=1), "1m": timedelta(minutes=1)}[
            one.get("bucket_width", "1d")
        ]
        group_by = q.get("group_by[]", [])
        for dim in group_by:
            if dim not in _USAGE_DIMS:
                return 400, {"error": {"message": f"bad group_by {dim}"}}
        cells = [
            c
            for c in USAGE_CELLS
            if all(c.get(col) in q[param] for param, col in _USAGE_FILTERS.items() if param in q)
        ]
        buckets = []
        for start, end in self._buckets(one, width):
            grouped: dict[tuple[Any, ...], int] = defaultdict(int)
            for cell in cells:
                grouped[tuple(cell.get(d) if d in group_by else None for d in _USAGE_DIMS)] += cell[
                    "output_tokens"
                ]
            results = [
                {
                    **dict(zip(_USAGE_DIMS, key, strict=True)),
                    "output_tokens": tokens,
                    "uncached_input_tokens": tokens * 10,
                    "cache_read_input_tokens": tokens,
                    "cache_creation": {"ephemeral_5m_input_tokens": 1, "ephemeral_1h_input_tokens": 0},
                    "server_tool_use": {"web_search_requests": 0},
                }
                for key, tokens in grouped.items()
            ]
            buckets.append({"starting_at": _rfc(start), "ending_at": _rfc(end), "results": results})
        return 200, self._token_page(buckets, one, 7)

    def _cost(self, q: dict[str, list[str]], one: dict[str, str]) -> tuple[int, dict[str, Any]]:
        if "starting_at" not in one:
            return 400, {"error": {"message": "starting_at is required"}}
        if int(one.get("limit", 7)) > 31:
            return 400, {"error": {"message": "limit must be <= 31"}}
        group_by = q.get("group_by[]", [])
        buckets = []
        for start, end in self._buckets(one, timedelta(days=1)):
            grouped: dict[tuple[Any, ...], Any] = defaultdict(lambda: 0)
            from decimal import Decimal

            for workspace, description, model, cents in COST_CELLS:
                key = (
                    workspace if "workspace_id" in group_by else None,
                    (description, model) if "description" in group_by else None,
                )
                grouped[key] += Decimal(cents)
            results = []
            for (workspace, desc), amount in grouped.items():
                described = (
                    {
                        "description": desc[0],
                        "model": desc[1],
                        "cost_type": "tokens",
                        "token_type": "output_tokens",
                        "service_tier": "standard",
                        "context_window": "0-200k",
                        "inference_geo": "global",
                    }
                    if desc
                    else {}
                )
                results.append(
                    {
                        "workspace_id": workspace,
                        "currency": "USD",
                        "amount": str(amount),
                        **dict.fromkeys(DESCRIBED_COLUMNS),
                        **described,
                    }
                )
            buckets.append({"starting_at": _rfc(start), "ending_at": _rfc(end), "results": results})
        return 200, self._token_page(buckets, one, 7)

    def _claude_code(self, one: dict[str, str]) -> tuple[int, dict[str, Any]]:
        day = one.get("starting_at", "")
        try:
            date.fromisoformat(day)
        except ValueError:
            return 400, {"error": {"message": "starting_at must be YYYY-MM-DD"}}
        records = claude_code_records(day)
        start = int(one["page"].removeprefix("page_")) if one.get("page") else 0
        size = min(int(one.get("limit", 20)), PAGE_CAP)
        chunk = records[start : start + size]
        more = start + size < len(records)
        return 200, {"data": chunk, "has_more": more, "next_page": f"page_{start + size}" if more else None}


if __name__ == "__main__":  # pragma: no cover - standalone, for vgi-lint --execute
    import time

    # Standalone, the fake follows the real clock, so examples written against
    # `current_date` see data. The test suite keeps the fixed NOW above, which
    # its arithmetic depends on.
    NOW = datetime.now(UTC)
    server = FakeAdminApi().start()
    print(server.url, flush=True)
    while True:
        time.sleep(3600)
