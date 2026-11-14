"""Synthetic, read-only fixture for this notebook's two Admin API endpoints.

Payload and paging shapes follow tests/fake_admin_api.py at worker revision
4da037833b6737c47ffd5ff4cf677c2a947ec7fb (Query Farm, MIT). Values below are
invented accounting entries, not recorded responses or model token prices.
The fixture binds only to loopback, rejects unsupported dates and endpoints,
and never forwards a request. Its key is a public dummy string, not a credential.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

START = datetime(2026, 9, 1, tzinfo=UTC)
END = START + timedelta(days=7)
DUMMY_KEY = "notebook-synthetic-key-not-a-credential"
WORKSPACES = [
    {"id": "wrkspc_prod", "name": "Production", "archived_at": None, "type": "workspace"},
    {"id": "wrkspc_stage", "name": "Staging", "archived_at": None, "type": "workspace"},
    {"id": "wrkspc_legacy", "name": "Legacy", "archived_at": "2026-01-01T00:00:00Z", "type": "workspace"},
]
DESCRIPTION_COLUMNS = (
    "description",
    "model",
    "cost_type",
    "token_type",
    "service_tier",
    "context_window",
    "inference_geo",
)


def instant(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class SyntheticAdminApi:
    """Serve deterministic accounting data through actual HTTP and record paths."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, list[str]]]] = []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query, keep_blank_values=True)
                fixture.requests.append((parsed.path, query))
                if self.headers.get("x-api-key") != DUMMY_KEY:
                    status, payload = 401, {"error": {"message": "Use the public synthetic fixture key"}}
                else:
                    try:
                        payload = fixture.route(parsed.path, query)
                        status = 200
                    except ValueError as error:
                        status, payload = 400, {"error": {"message": str(error)}}
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("request-id", "req_notebook_fixture")
                self.send_header("x-should-retry", "false")
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    def start(self) -> SyntheticAdminApi:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def route(self, path: str, query: dict[str, list[str]]) -> dict:
        one = {key: value[-1] for key, value in query.items()}
        if path == "/v1/organizations/workspaces":
            rows = [w for w in WORKSPACES if one.get("include_archived") == "true" or not w["archived_at"]]
            start = next((i + 1 for i, w in enumerate(rows) if w["id"] == one.get("after_id")), 0)
            chunk = rows[start : start + 2]
            return {
                "data": chunk,
                "has_more": start + 2 < len(rows),
                "first_id": chunk[0]["id"] if chunk else None,
                "last_id": chunk[-1]["id"] if chunk else None,
            }
        if path != "/v1/organizations/cost_report":
            raise ValueError(f"Endpoint outside this synthetic fixture: {path}")
        start, end = instant(one["starting_at"]), instant(one["ending_at"])
        if not START <= start < end <= END or start.hour or end.hour:
            raise ValueError("Fixture covers full UTC days from 2026-09-01 through 2026-09-07 only")
        groups = query.get("group_by[]", [])
        if set(groups) - {"workspace_id"}:
            raise ValueError("This notebook fixture supports workspace_id grouping only")
        buckets = []
        day = start
        while day < end:
            index = (day - START).days
            # Deliberately changing composition, in exact cents. These are
            # invented ledger entries, unrelated to any model's pricing.
            ledger = [
                ("wrkspc_prod", 6000 + 500 * index),
                ("wrkspc_stage", 2000 - 200 * index),
                (None, 1200 + 300 * index),
            ]
            amounts: dict[str | None, Decimal] = defaultdict(Decimal)
            for workspace, cents in ledger:
                amounts[workspace if "workspace_id" in groups else None] += Decimal(cents)
            results = [
                {
                    "workspace_id": key,
                    "currency": "USD",
                    "amount": str(amount),
                    **dict.fromkeys(DESCRIPTION_COLUMNS),
                }
                for key, amount in amounts.items()
            ]
            buckets.append(
                {
                    "starting_at": day.isoformat(),
                    "ending_at": (day + timedelta(days=1)).isoformat(),
                    "results": results,
                }
            )
            day += timedelta(days=1)
        offset = int(one.get("page", "page_0").removeprefix("page_"))
        # Two buckets per page deliberately exercise report pagination.
        chunk = buckets[offset : offset + 2]
        more = offset + 2 < len(buckets)
        return {"data": chunk, "has_more": more, "next_page": f"page_{offset + 2}" if more else None}
