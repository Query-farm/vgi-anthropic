"""Read-only HTTP access to the Anthropic Admin, Usage and Cost APIs.

Every function here issues ``GET`` requests only — the single chokepoint
:func:`_get` is the sole place an HTTP call is made, which is what the read-only
CI guard (``tests/test_readonly_guard.py``) asserts against. The Admin API is
mostly *write* surface — creating workspaces, inviting members, archiving keys,
changing roles — and an Admin key authorizes all of it, so read-only here is a
structural property rather than a configuration choice.

The API pages in two different styles, and both are handled here:

* **ID cursors** — users, invites, workspaces, workspace members and API keys
  return ``has_more`` and ``last_id``; the next page is ``after_id=<last_id>``.
* **Page tokens** — the usage, cost, Claude Code and rate-limit reports return
  ``has_more`` and an opaque ``next_page``; the next page is ``page=<token>``.

:func:`page` hides the difference behind one ``(rows, next_cursor)`` contract.
"""

from __future__ import annotations

import atexit
import os
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import quote

import httpx

from vgi_anthropic.auth import Credentials

#: Production base URL.
DEFAULT_BASE_URL = "https://api.anthropic.com"

#: Overrides the base URL. Deliberately *not* ``ANTHROPIC_BASE_URL``: that one
#: is commonly set to point SDKs and Claude Code at a proxy or gateway, and
#: inheriting it would quietly send an Admin key — which reads and could manage
#: the whole organization — to wherever it happened to point.
BASE_URL_ENV = "VGI_ANTHROPIC_BASE_URL"

#: The API version header every request carries.
API_VERSION = "2023-06-01"

#: Beta header required to group usage by ``speed``.
FAST_MODE_BETA = "fast-mode-2026-02-01"

#: Page size for the ID-cursor endpoints: the documented maximum.
ID_PAGE_LIMIT = 1000

#: Records per page for the Claude Code report: the documented maximum.
CLAUDE_CODE_PAGE_LIMIT = 1000

#: Hard stop on cursor following inside one bounded call, so a runaway walk
#: cannot hang a query. Hitting it raises rather than truncating — see
#: :class:`AnthropicPageLimitError`.
MAX_PAGES = 500

#: Connect and read timeouts. The usage report over a wide group_by is the
#: slowest call here and still answers in seconds.
TIMEOUT = httpx.Timeout(60.0, connect=10.0)

#: Statuses worth retrying. ``529`` is Anthropic's "overloaded".
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504, 529})

#: Exponential backoff for a retryable request: ~0.5s, 1s, 2s, 4s, 8s — unless
#: the response names its own wait in ``retry-after``, which wins.
_RETRY_ATTEMPTS = 5
_RETRY_BASE_SECONDS = 0.5

#: Ceiling on an honoured ``retry-after``. A server asking for a minute is
#: better answered with an error the caller can see than a query that hangs.
_MAX_RETRY_AFTER = 30.0


def base_url() -> str:
    """The API base URL, overridable with ``VGI_ANTHROPIC_BASE_URL``."""
    return os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL).rstrip("/")


def open_client() -> httpx.Client:
    """Open a client carrying this module's timeouts.

    Callers that issue a burst of per-row fetches (a correlated LATERAL) open
    one of these and pass it in, so the batch shares a connection pool.
    """
    return httpx.Client(timeout=TIMEOUT)


_shared: httpx.Client | None = None
_shared_lock = threading.Lock()


def shared_client() -> httpx.Client:
    """A process-wide client, for calls that have no batch to share one with.

    A paged scan fetches one page per ``process()`` tick, so there is no
    enclosing scope to hold a client open across the walk; without this, every
    page would pay a fresh TLS handshake.
    """
    global _shared
    if _shared is None or _shared.is_closed:
        with _shared_lock:
            if _shared is None or _shared.is_closed:
                _shared = open_client()
    return _shared


def reset_shared_client() -> None:
    """Close and forget the process pool, so the next call opens a fresh one.

    Operationally an escape hatch for a bad pooled connection; in tests, what
    keeps one test's mock transport from serving the next.
    """
    global _shared
    with _shared_lock:
        if _shared is not None and not _shared.is_closed:
            _shared.close()
        _shared = None


atexit.register(reset_shared_client)


def segment(value: str) -> str:
    """Percent-encode one path segment, so a value cannot escape its position.

    IDs reach this module straight from user SQL. Interpolated raw,
    ``user_by_id('../invites')`` would resolve to ``/v1/organizations/invites``
    — a different resource than the one asked for — and a ``?`` would truncate
    the path and inject query parameters. Encoding makes ``/`` and ``?`` into
    ``%2F`` and ``%3F``, so a segment stays a segment.
    """
    return quote(value, safe="")


class AnthropicError(RuntimeError):
    """A non-2xx response from the API, carrying the status and the API's own message."""

    def __init__(self, status: int, path: str, body: str, request_id: str | None = None) -> None:
        detail = _error_message(body)
        hint = _STATUS_HINTS.get(status, "")
        suffix = f" (request-id {request_id})" if request_id else ""
        super().__init__(f"Anthropic API {status} for {path}: {detail}{hint}{suffix}")
        self.status = status
        self.path = path
        self.request_id = request_id


class AnthropicPageLimitError(RuntimeError):
    """A bounded walk followed :data:`MAX_PAGES` cursors without reaching the end.

    Raised rather than returning the truncated prefix: a silently short result
    reads as real data, and nothing downstream can tell the difference.
    """

    def __init__(self, path: str, rows: int) -> None:
        super().__init__(f"Anthropic API {path}: still paging after {MAX_PAGES} pages ({rows} rows)")
        self.path = path
        self.rows = rows


#: Appended to an error for the statuses whose cause is almost always the same.
_STATUS_HINTS = {
    401: (
        " — the anthropic_admin secret needs an Admin API key (sk-ant-admin01-...), an "
        "organization-scoped personal or service-account key, or an org:admin OAuth token. A "
        "workspace-scoped key can read `organization` and nothing else, and individual accounts "
        "have no Admin API until an organization is set up (Console → Settings → Organization)"
    ),
    403: (
        " — the credential is not allowed to read this. A personal key acts with its user's "
        "permissions, so that user needs the admin role; an Admin API key (sk-ant-admin01-...) "
        "avoids the question. Claude Platform on AWS exposes only the workspace endpoints"
    ),
}


def _error_message(body: str) -> str:
    """The ``error.message`` from an API error body, or the raw body trimmed."""
    try:
        import json

        payload = json.loads(body)
    except ValueError:
        return body[:400]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return f"{error.get('type', 'error')}: {error['message']}"
    return body[:400]


def _should_retry(response: httpx.Response) -> bool:
    """Whether a response is worth another attempt.

    Only an error is ever retried. Within that, the API's own ``x-should-retry``
    header wins when present — observed on live responses, and obeyed the same
    way by Anthropic's SDKs — because the server knows whether a retry can
    help: a ``false`` on a 529 saves five pointless attempts, and a ``true``
    on an unlisted status recovers where the status table would give up.
    Without the header, the status decides.
    """
    if response.status_code < 400:
        return False
    hint = response.headers.get("x-should-retry", "").strip().lower()
    if hint == "true":
        return True
    if hint == "false":
        return False
    return response.status_code in _RETRYABLE_STATUSES


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    """How long to wait before the next attempt.

    The API sends ``retry-after`` (seconds) on a 429; honouring it is both
    faster than guessing and kinder to the shared organization rate limit.
    """
    if response is not None:
        raw = response.headers.get("retry-after")
        if raw:
            try:
                return min(max(float(raw), 0.0), _MAX_RETRY_AFTER)
            except ValueError:
                pass
    return float(_RETRY_BASE_SECONDS * (2**attempt))


def _get(
    path: str,
    params: dict[str, Any] | Sequence[tuple[str, Any]] | None = None,
    *,
    credentials: Credentials,
    client: httpx.Client | None = None,
    betas: Sequence[str] = (),
) -> dict[str, Any]:
    """GET ``path`` under the API base and return the decoded JSON object.

    The one and only outbound-HTTP chokepoint. It issues ``GET`` and nothing
    else; adding a write verb here is the thing the read-only guard forbids.

    A ``GET`` is idempotent, so every transient failure is retried: rate
    limiting (429), overload (529), gateway 5xx, and transport errors.

    Args:
        path: Path below the API base, with a leading slash.
        params: Query parameters; ``None`` values are dropped. A sequence of
            pairs rather than a mapping when a parameter repeats
            (``group_by[]=model&group_by[]=workspace_id``).
        credentials: The Admin credential. Required — nothing here is public.
        client: Optional caller-owned client, so a lateral join reuses one pool.
        betas: ``anthropic-beta`` values to send.

    Raises:
        AnthropicError: The response status was not 2xx after retries, or a
            2xx body was not JSON.
        httpx.TransportError: Every attempt failed to reach the API.
    """
    pairs = list(params.items()) if isinstance(params, dict) else list(params or ())
    clean: Any = [(k, _wire(v)) for k, v in pairs if v is not None]
    url = f"{base_url()}{path}"
    headers = {"anthropic-version": API_VERSION, **credentials.headers()}
    if betas:
        headers["anthropic-beta"] = ",".join(betas)
    http = client or shared_client()
    response: httpx.Response | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        last_attempt = attempt == _RETRY_ATTEMPTS - 1
        try:
            response = http.get(url, params=clean, headers=headers)
        except httpx.TransportError:
            if last_attempt:
                raise
            response = None
        else:
            if not _should_retry(response) or last_attempt:
                break
        time.sleep(_retry_delay(response, attempt))
    assert response is not None  # the loop either breaks with one or raises
    request_id = response.headers.get("request-id")
    if response.status_code >= 400:
        raise AnthropicError(response.status_code, path, response.text, request_id)
    try:
        payload = response.json()
    except ValueError as exc:
        raise AnthropicError(
            response.status_code,
            path,
            f"expected JSON, got {response.headers.get('content-type', 'no content-type')}: {response.text}",
            request_id,
        ) from exc
    if not isinstance(payload, dict):
        raise AnthropicError(response.status_code, path, f"expected a JSON object, got {payload!r}")
    return payload


def _wire(value: Any) -> Any:
    """A query value as the API wants it: booleans lowercase, everything else as-is."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


# --------------------------------------------------------------------------
# Paging
# --------------------------------------------------------------------------

#: ``id`` pages with ``after_id``/``last_id``; ``token`` with ``page``/``next_page``.
PageStyle = Literal["id", "token"]


@dataclass(slots=True, frozen=True)
class Endpoint:
    """How to walk one list endpoint."""

    path: str
    style: PageStyle
    #: Per-request page size, or ``None`` to send no ``limit`` at all — the
    #: rate-limit endpoints return everything in one page when it is omitted.
    page_limit: int | None


def page(
    endpoint: Endpoint,
    params: dict[str, Any] | Sequence[tuple[str, Any]] | None,
    *,
    cursor: str | None,
    credentials: Credentials,
    client: httpx.Client | None = None,
    betas: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch exactly one page, returning its rows and the cursor for the next.

    A ``None`` cursor means this was the final page. The primitive behind both
    a bounded collect (:func:`collect`) and a streaming scan, which holds the
    cursor in its state and calls this once per tick so DuckDB sees rows from
    the first page rather than the last.
    """
    pairs = list(params.items()) if isinstance(params, dict) else list(params or ())
    if endpoint.page_limit is not None:
        pairs.append(("limit", endpoint.page_limit))
    if cursor:
        pairs.append(("after_id" if endpoint.style == "id" else "page", cursor))
    payload = _get(endpoint.path, pairs, credentials=credentials, client=client, betas=betas)
    rows = [row for row in payload.get("data") or [] if isinstance(row, dict)]
    if not payload.get("has_more"):
        return rows, None
    next_cursor = payload.get("last_id") if endpoint.style == "id" else payload.get("next_page")
    # `has_more` with no way to continue is a server inconsistency; ending the
    # walk is the only option that terminates.
    return rows, (str(next_cursor) if next_cursor else None)


def collect(
    endpoint: Endpoint,
    params: dict[str, Any] | Sequence[tuple[str, Any]] | None = None,
    *,
    credentials: Credentials,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Follow every page of ``endpoint`` and return all rows.

    For walks known to be small — the workspace list a fan-out iterates over.
    Running out of :data:`MAX_PAGES` raises rather than returning a short list.
    """
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    for _ in range(MAX_PAGES):
        batch, cursor = page(endpoint, params, cursor=cursor, credentials=credentials, client=client)
        rows.extend(batch)
        if cursor is None:
            return rows
    raise AnthropicPageLimitError(endpoint.path, len(rows))


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

USERS = Endpoint("/v1/organizations/users", "id", ID_PAGE_LIMIT)
INVITES = Endpoint("/v1/organizations/invites", "id", ID_PAGE_LIMIT)
WORKSPACES = Endpoint("/v1/organizations/workspaces", "id", ID_PAGE_LIMIT)
API_KEYS = Endpoint("/v1/organizations/api_keys", "id", ID_PAGE_LIMIT)
RATE_LIMITS = Endpoint("/v1/organizations/rate_limits", "token", None)
MESSAGES_USAGE = Endpoint("/v1/organizations/usage_report/messages", "token", None)
COST_REPORT = Endpoint("/v1/organizations/cost_report", "token", None)
CLAUDE_CODE_USAGE = Endpoint("/v1/organizations/usage_report/claude_code", "token", CLAUDE_CODE_PAGE_LIMIT)


def workspace_members_endpoint(workspace_id: str) -> Endpoint:
    """The member list of one workspace."""
    return Endpoint(f"/v1/organizations/workspaces/{segment(workspace_id)}/members", "id", ID_PAGE_LIMIT)


def workspace_rate_limits_endpoint(workspace_id: str) -> Endpoint:
    """The rate-limit overrides of one workspace."""
    return Endpoint(f"/v1/organizations/workspaces/{segment(workspace_id)}/rate_limits", "token", None)


def organization(*, credentials: Credentials, client: httpx.Client | None = None) -> dict[str, Any]:
    """The organization the credential belongs to."""
    return _get("/v1/organizations/me", credentials=credentials, client=client)


def user(user_id: str, *, credentials: Credentials, client: httpx.Client | None = None) -> dict[str, Any]:
    """One organization member by ID."""
    return _get(f"/v1/organizations/users/{segment(user_id)}", credentials=credentials, client=client)


def workspace(
    workspace_id: str, *, credentials: Credentials, client: httpx.Client | None = None
) -> dict[str, Any]:
    """One workspace by ID."""
    return _get(
        f"/v1/organizations/workspaces/{segment(workspace_id)}", credentials=credentials, client=client
    )


def api_key(
    api_key_id: str, *, credentials: Credentials, client: httpx.Client | None = None
) -> dict[str, Any]:
    """One API key's metadata by ID. The secret value is never returned by the API."""
    return _get(f"/v1/organizations/api_keys/{segment(api_key_id)}", credentials=credentials, client=client)
