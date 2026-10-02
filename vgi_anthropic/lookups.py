"""Point lookups by ID: ``user_by_id``, ``workspace_by_id``, ``api_key_by_id``.

All three are **blended** (:class:`~vgi.table_in_out_function.RowTransformFunction`)
table functions: the positional argument *is* the per-row input column, so one
registration serves both a literal call and a correlated LATERAL::

    SELECT * FROM anthropic.main.user_by_id('user_01WCz1FkmYMm4gnmykNKUu3Q');

    SELECT k.name, u.email
    FROM anthropic.main.api_keys k,
         LATERAL anthropic.main.user_by_id(k.created_by_id) u;

The tables already hold every row these return, so a join against them is
usually the better tool; these exist for the point read, where paging a whole
list to find one object would be the expensive way round.

An ID that does not exist — a member who has since left, a deleted key — yields
**no row** rather than failing the query. In a LATERAL over historical data
that is the common case, and one departed user should not abort the join.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, cast

import pyarrow as pa
from vgi.arguments import Arg, SecretLookupEntry
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_function import BindParams, ProcessParams
from vgi.table_in_out_function import RowTransformFunction
from vgi_rpc.rpc import OutputCollector

from vgi_anthropic import anthropic_api as api
from vgi_anthropic import auth
from vgi_anthropic.anthropic_api import AnthropicError
from vgi_anthropic.meta import docs, examples
from vgi_anthropic.pushdown import build_filtered
from vgi_anthropic.schemas import (
    API_KEY_SCHEMA,
    USER_SCHEMA,
    WORKSPACE_SCHEMA,
    flatten_api_key,
    flatten_workspace,
)

if TYPE_CHECKING:
    from vgi.protocol import VgiOutputCollector

_SECRETS = [SecretLookupEntry(secret_type=auth.SECRET_TYPE)]


def _lookup_all(
    params: ProcessParams[Any],
    batch: pa.RecordBatch,
    out: OutputCollector,
    *,
    column: str,
    fetch: Callable[..., dict[str, Any]],
    flatten: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> None:
    """Fetch one object per input row and emit them with provenance.

    1->0/1: a NULL or unknown ID emits nothing, so ``parent_rows`` maps each
    output row back to its input row — the batched-LATERAL operator needs it
    to stamp the correlated columns onto the right rows.
    """
    ids = batch.column(column).to_pylist()
    rows: list[dict[str, Any]] = []
    parents: list[int] = []
    credentials = auth.for_call(params.secrets)
    with api.open_client() as client:
        for index, raw in enumerate(ids):
            if not raw:
                continue
            try:
                found = fetch(str(raw), credentials=credentials, client=client)
            except AnthropicError as exc:
                # Not found is an answer, not a failure. A 400 is how the API
                # rejects a malformed ID, which is the same answer.
                if exc.status in (400, 404):
                    continue
                raise
            rows.append(flatten(found) if flatten else found)
            parents.append(index)
    filtered, kept = build_filtered(params, rows, parents)
    cast("VgiOutputCollector", out).emit(filtered, parent_rows=kept)


@dataclass(slots=True, frozen=True, kw_only=True)
class UserIdArgs:
    user_id: Annotated[str, Arg(0, doc="User ID input column (user_...)")]


@dataclass(slots=True, frozen=True, kw_only=True)
class WorkspaceIdArgs:
    workspace_id: Annotated[str, Arg(0, doc="Workspace ID input column (wrkspc_...)")]


@dataclass(slots=True, frozen=True, kw_only=True)
class ApiKeyIdArgs:
    api_key_id: Annotated[str, Arg(0, doc="API key ID input column (apikey_...)")]


class UserByIdFunction(RowTransformFunction[UserIdArgs]):
    """One organization member by ID."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = USER_SCHEMA

    class Meta:
        name = "user_by_id"
        description = "One organization member by user ID (no row if the user is gone)"
        categories = ["organization", "blended"]
        required_secrets = _SECRETS
        projection_pushdown = True
        tags = docs(
            category="organization",
            result_schema=USER_SCHEMA,
            llm=(
                "Look up one member by user ID. Composes under LATERAL to resolve an ID column "
                "(api_keys.created_by_id, messages_usage.account_id) into a name and email. An "
                "unknown ID returns no row. For many IDs, a join to the `users` table is cheaper."
            ),
            md=(
                "A point lookup into the member list, one request per input row. Departed "
                "members return no row rather than an error, so a LATERAL over historical IDs "
                "completes; use a LEFT JOIN LATERAL to keep the unmatched rows."
            ),
            example_queries=examples(
                (
                    "Who created each active key",
                    "SELECT k.name, u.email FROM anthropic.main.api_keys k "
                    "LEFT JOIN LATERAL anthropic.main.user_by_id(k.created_by_id) u ON true "
                    "WHERE k.status = 'active' AND k.created_by_type = 'user'",
                ),
            ),
        )
        examples = [
            FunctionExample(
                sql=(
                    "SELECT k.name, u.email FROM anthropic.main.api_keys k "
                    "LEFT JOIN LATERAL anthropic.main.user_by_id(k.created_by_id) u ON true "
                    "WHERE k.status = 'active' AND k.created_by_type = 'user'"
                ),
                description="Who created each active key",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[UserIdArgs]) -> BindResponse:
        return BindResponse(output_schema=cls.FIXED_SCHEMA)

    @classmethod
    def process(
        cls, params: ProcessParams[UserIdArgs], state: None, batch: pa.RecordBatch, out: OutputCollector
    ) -> None:
        _lookup_all(params, batch, out, column="user_id", fetch=api.user)


class WorkspaceByIdFunction(RowTransformFunction[WorkspaceIdArgs]):
    """One workspace by ID."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = WORKSPACE_SCHEMA

    class Meta:
        name = "workspace_by_id"
        description = "One workspace by ID (no row if it does not exist)"
        categories = ["organization", "blended"]
        required_secrets = _SECRETS
        projection_pushdown = True
        tags = docs(
            category="organization",
            result_schema=WORKSPACE_SCHEMA,
            llm=(
                "Look up one workspace by ID, archived or not. Composes under LATERAL. An unknown "
                "ID returns no row. A join to the `workspaces` table is usually cheaper."
            ),
            md=(
                "A point lookup, one request per input row. Unknown IDs return no row. The "
                "default workspace has no ID in the usage and cost reports (it is NULL there), so "
                "it cannot be looked up this way."
            ),
            example_queries=examples(
                (
                    "Look up one workspace",
                    "SELECT name, created_at FROM anthropic.main.workspace_by_id("
                    "(SELECT id FROM anthropic.main.workspaces LIMIT 1))",
                ),
            ),
        )
        examples = [
            FunctionExample(
                sql=(
                    "SELECT name, created_at FROM anthropic.main.workspace_by_id("
                    "(SELECT id FROM anthropic.main.workspaces LIMIT 1))"
                ),
                description="Look up one workspace",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[WorkspaceIdArgs]) -> BindResponse:
        return BindResponse(output_schema=cls.FIXED_SCHEMA)

    @classmethod
    def process(
        cls, params: ProcessParams[WorkspaceIdArgs], state: None, batch: pa.RecordBatch, out: OutputCollector
    ) -> None:
        _lookup_all(params, batch, out, column="workspace_id", fetch=api.workspace, flatten=flatten_workspace)


class ApiKeyByIdFunction(RowTransformFunction[ApiKeyIdArgs]):
    """One API key's metadata by ID."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = API_KEY_SCHEMA

    class Meta:
        name = "api_key_by_id"
        description = "One API key's metadata by ID (never the secret; no row if it does not exist)"
        categories = ["access", "blended"]
        required_secrets = _SECRETS
        projection_pushdown = True
        tags = docs(
            category="access",
            result_schema=API_KEY_SCHEMA,
            llm=(
                "Look up one API key's metadata by ID — name, status, workspace, creator. Composes "
                "under LATERAL, e.g. to name the keys in messages_usage.api_key_id. Never returns "
                "the secret. An unknown ID returns no row."
            ),
            md=(
                "A point lookup, one request per input row. The secret value is never returned "
                "by the API. Unknown IDs return no row."
            ),
            example_queries=examples(
                (
                    "Look up one key",
                    "SELECT name, status FROM anthropic.main.api_key_by_id("
                    "(SELECT id FROM anthropic.main.api_keys LIMIT 1))",
                ),
            ),
        )
        examples = [
            FunctionExample(
                sql=(
                    "SELECT name, status FROM anthropic.main.api_key_by_id("
                    "(SELECT id FROM anthropic.main.api_keys LIMIT 1))"
                ),
                description="Look up one key",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[ApiKeyIdArgs]) -> BindResponse:
        return BindResponse(output_schema=cls.FIXED_SCHEMA)

    @classmethod
    def process(
        cls, params: ProcessParams[ApiKeyIdArgs], state: None, batch: pa.RecordBatch, out: OutputCollector
    ) -> None:
        _lookup_all(params, batch, out, column="api_key_id", fetch=api.api_key, flatten=flatten_api_key)


LOOKUP_FUNCTIONS: list[type] = [UserByIdFunction, WorkspaceByIdFunction, ApiKeyByIdFunction]
