"""Turning a DuckDB ``WHERE`` into Admin API query parameters.

Pushdown here is always a pure optimisation, and the two directions of error
are wildly asymmetric:

* Pushing **too little** costs a few requests. Nothing else.
* Pushing **too much** drops rows the predicate would have kept, and nothing
  downstream can recover what was never fetched.

So every translation declines when it is not certain: a filter becomes an API
parameter only when the API's meaning is exactly the predicate's, or strictly
broader. The users list's ``email`` filter, for example, is case-insensitive
while SQL ``=`` is not — broader, so it is pushed, and the exact predicate is
then re-applied to what comes back by :func:`build_filtered`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from vgi.table_function import ProcessParams

from vgi_anthropic.schemas import batch_from_rows


def _filters(params: ProcessParams[Any]) -> Any:
    return getattr(params, "current_pushdown_filters", None)


def equality(params: ProcessParams[Any], column: str) -> str | None:
    """The constant from ``WHERE <column> = '...'``, when there is exactly one.

    Only non-empty string equality is returned; anything else is left alone.
    """
    filters = _filters(params)
    if filters is None:
        return None
    scalar = filters.get_column_constant(column)
    value = scalar.as_py() if scalar is not None else None
    return value if isinstance(value, str) and value else None


def projected(params: ProcessParams[Any]) -> set[str]:
    """The column names DuckDB asked this scan for.

    Includes every column a pushed predicate references: DuckDB does not
    project away a column it is filtering on, because it needs it from the scan.
    """
    schema = getattr(params, "output_schema", None)
    return set(schema.names) if schema is not None else set()


def build_filtered(
    params: ProcessParams[Any],
    rows: Sequence[dict[str, Any]],
    parent_rows: Sequence[int] | None = None,
) -> tuple[pa.RecordBatch, list[int]]:
    """Build the output batch, applying every pushed filter to it.

    **Declaring ``filter_pushdown`` is a promise to apply the filters.** The
    engine drops its own filter above the scan once a function accepts
    pushdown, so a predicate the worker receives and ignores is not re-checked
    by anyone — it simply stops being applied. Translating *some* predicates
    into API parameters and leaving the rest would therefore be a wrong answer,
    not a partial optimisation.

    Everything is evaluated against ``params.output_schema`` — the *projected*
    schema. A pushed filter carries a ``column_index`` relative to that
    projection, so evaluating against the full schema would compare the wrong
    column. That is also why projection pushdown stays on wherever filters are
    accepted.

    Args:
        params: The tick's parameters, carrying the filters and the projection.
        rows: Flattened API rows, keyed by the schema's column names.
        parent_rows: 1->N provenance for a blended function, filtered in
            lockstep with the rows.

    Returns:
        The projected, filtered batch and its surviving ``parent_rows``.
    """
    batch = batch_from_rows(rows, params.output_schema)
    filters = _filters(params)
    if filters is None:
        return batch, list(parent_rows or [])
    mask = filters.evaluate(batch)
    kept = pc.filter(batch, mask)
    if parent_rows is None:
        return kept, []
    # `pc.filter` drops rows whose mask is null — the SQL meaning of a predicate
    # that did not evaluate to true — and provenance follows the same rule.
    keep = mask.to_pylist()
    return kept, [parent for parent, ok in zip(parent_rows, keep, strict=True) if ok]
