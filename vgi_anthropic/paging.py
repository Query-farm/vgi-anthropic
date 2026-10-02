"""Cursor paging as VGI scan state.

Walking every page inside one ``process()`` call is the obvious implementation
and the wrong one: DuckDB sees nothing until the last page arrives, a ``LIMIT
10`` still pays for the whole walk, and — because a scan blocked inside its
first batch cannot be cancelled — a long walk wedges the client rather than
merely running slowly. (The sibling Kalshi worker learned this the hard way.)

A ``TableFunctionGenerator`` carries state between ``process()`` ticks, so the
cursor lives in that state and each tick fetches exactly one page and emits
exactly one batch.

Some objects here are not one list but many: workspace members and workspace
rate limits exist per workspace, and the Claude Code report is per day. Those
are a *walk over keys* — workspace IDs, or dates — with a cursor inside each.
:func:`emit_walk` handles both shapes; a plain list is simply a walk over a
single key.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from vgi.cache_control import CacheControl
from vgi.table_function import ProcessParams
from vgi_rpc import ArrowSerializableDataclass
from vgi_rpc.rpc import OutputCollector

from vgi_anthropic import anthropic_api as api
from vgi_anthropic import auth
from vgi_anthropic.anthropic_api import Endpoint
from vgi_anthropic.pushdown import build_filtered

#: Query parameters as ordered pairs, since several repeat (``group_by[]``).
Query = Sequence[tuple[str, Any]]


@dataclass(kw_only=True)
class WalkState(ArrowSerializableDataclass):
    """Where a paged walk has got to.

    The framework persists this between ``process()`` ticks. Everything that
    shapes the requests — the keys to visit and the query — is frozen on the
    first tick: ``current_pushdown_filters`` is refreshed before every tick,
    and a cursor continued under a different query than it was minted for is
    undefined, with silently wrong rows as the failure mode.
    """

    started: bool = False
    done: bool = False
    #: Opaque continuation within the current key; empty at a key's first page.
    cursor: str = ""
    #: JSON list of the keys still to visit, current one first.
    keys_json: str = "[]"
    #: JSON list of ``[name, value]`` query pairs, shared by every key.
    query_json: str = "[]"


def emit_walk(
    params: ProcessParams[Any],
    state: WalkState,
    out: OutputCollector,
    *,
    keys: Callable[[], list[str]],
    endpoint_for: Callable[[str], Endpoint],
    query: Callable[[], Query],
    flatten: Callable[[list[dict[str, Any]], str], list[dict[str, Any]]],
    key_query: Callable[[str], Query] | None = None,
    betas: Sequence[str] = (),
    cache_control: CacheControl | None = None,
) -> None:
    """Fetch one page into one batch, and record where to resume.

    Args:
        params: The tick's parameters; ``output_schema`` is the projected one.
        state: The walk carried between ticks.
        out: Collector for the batch, or for ``finish()`` once the walk is done.
        keys: The keys to visit, called once on the first tick. A single-list
            scan passes ``lambda: [""]``.
        endpoint_for: The endpoint to page for a key.
        query: Query parameters, called once on the first tick and frozen.
        flatten: Turn one page of API objects into rows, given their key.
        key_query: Extra parameters that depend on the key (the Claude Code
            report's ``starting_at`` date), appended per request.
        betas: ``anthropic-beta`` values for every request.
        cache_control: Cache metadata to attach to every batch.
    """
    if state.done:
        out.finish()
        return
    credentials = auth.for_call(params.secrets)
    if not state.started:
        state.keys_json = json.dumps(keys())
        state.query_json = json.dumps([list(pair) for pair in query()], default=str)
        state.started = True
    remaining: list[str] = json.loads(state.keys_json)
    if not remaining:
        state.done = True
        out.finish()
        return
    key = remaining[0]
    pairs: list[tuple[str, Any]] = [(name, value) for name, value in json.loads(state.query_json)]
    if key_query is not None:
        pairs.extend(key_query(key))
    rows, cursor = api.page(
        endpoint_for(key), pairs, cursor=state.cursor or None, credentials=credentials, betas=betas
    )
    state.cursor = cursor or ""
    if cursor is None:
        remaining = remaining[1:]
        state.keys_json = json.dumps(remaining)
        state.done = not remaining
    batch, _ = build_filtered(params, flatten(rows, key))
    out.emit(batch, cache_control=cache_control)


def emit_list(
    params: ProcessParams[Any],
    state: WalkState,
    out: OutputCollector,
    *,
    endpoint: Endpoint,
    query: Callable[[], Query],
    flatten: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> None:
    """:func:`emit_walk` for one plain list endpoint, one object per row."""

    def rows(page: list[dict[str, Any]], _key: str) -> list[dict[str, Any]]:
        return [flatten(item) for item in page] if flatten else page

    emit_walk(
        params, state, out, keys=lambda: [""], endpoint_for=lambda _k: endpoint, query=query, flatten=rows
    )
