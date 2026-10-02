"""The organization's own records: members, invites, workspaces, API keys, rate limits.

Each of these is a list with no required key — the organization *is* the key —
so each is exposed as a real catalog table. The functions here are the scans
behind those tables, named ``all_<table>`` so a function and the table it backs
do not collide in one schema; ``ReadOnlyCatalogInterface`` wires each table to
its scan, so no scan code lives in :mod:`vgi_anthropic.worker`.

Every list endpoint pages, so every scan here pages too: one API page per
``process()`` tick (see :mod:`vgi_anthropic.paging`).
"""

from __future__ import annotations

from typing import Any, ClassVar

import pyarrow as pa
from vgi.arguments import SecretLookupEntry
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_function import BindParams, ProcessParams, TableFunctionGenerator, init_single_worker
from vgi_rpc.rpc import OutputCollector

from vgi_anthropic import anthropic_api as api
from vgi_anthropic import auth
from vgi_anthropic.meta import docs, examples
from vgi_anthropic.paging import Query, WalkState, emit_list, emit_walk
from vgi_anthropic.pushdown import build_filtered, equality
from vgi_anthropic.schemas import (
    API_KEY_SCHEMA,
    INVITE_SCHEMA,
    ORGANIZATION_SCHEMA,
    RATE_LIMIT_SCHEMA,
    USER_SCHEMA,
    WORKSPACE_MEMBER_SCHEMA,
    WORKSPACE_RATE_LIMIT_SCHEMA,
    WORKSPACE_SCHEMA,
    flatten_api_key,
    flatten_rate_limits,
    flatten_workspace,
    flatten_workspace_rate_limits,
)

_SECRETS = [SecretLookupEntry(secret_type=auth.SECRET_TYPE)]

#: API key statuses; a pushed `status = '<one of these>'` is the API's own filter.
_API_KEY_STATUSES = {"active", "inactive", "archived", "expired"}

#: Rate-limit group types the API filters on.
_GROUP_TYPES = {"batch", "files", "model_group", "skills", "token_count", "web_search"}


def _example(sql: str, description: str) -> list[FunctionExample]:
    return [FunctionExample(sql=sql, description=description)]


class _Scan(TableFunctionGenerator[None, WalkState]):
    """Shared bind and state for the parameterless paged scans."""

    FIXED_SCHEMA: ClassVar[pa.Schema]

    @classmethod
    def on_bind(cls, params: BindParams[None]) -> BindResponse:
        return BindResponse(output_schema=cls.FIXED_SCHEMA)

    @classmethod
    def initial_state(cls, params: ProcessParams[None]) -> WalkState:
        return WalkState()


def _active_workspace_ids(params: ProcessParams[Any]) -> list[str]:
    """The workspaces a per-workspace walk should visit.

    A pushed ``workspace_id = '...'`` narrows the walk to that one workspace —
    the difference between one request and one per workspace. Otherwise every
    active workspace; archived workspaces have no members or limits to report.
    """
    pinned = equality(params, "workspace_id")
    if pinned:
        return [pinned]
    rows = api.collect(api.WORKSPACES, credentials=auth.for_call(params.secrets))
    return [str(w["id"]) for w in rows if w.get("id") and w.get("archived_at") is None]


# --------------------------------------------------------------------------
# Organization
# --------------------------------------------------------------------------


@init_single_worker
class AllOrganizationFunction(TableFunctionGenerator[None, None]):
    """The organization the credential belongs to — the scan behind ``organization``."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = ORGANIZATION_SCHEMA

    class Meta:
        name = "all_organization"
        description = "The organization the Admin credential belongs to (the scan backing `organization`)"
        categories = ["organization"]
        required_secrets = _SECRETS
        projection_pushdown = True
        tags = docs(
            category="organization",
            result_schema=ORGANIZATION_SCHEMA,
            llm=(
                "One row: the ID and name of the organization the configured Admin credential "
                "belongs to. Prefer the `organization` table, which scans this. Useful to confirm "
                "which organization a secret points at before reading anything else."
            ),
            md=(
                "The organization behind the `anthropic_admin` secret, as one row.\n\n"
                "### Prefer the table\n\n"
                "The `organization` table is backed by this function and returns the same row; the "
                "function exists so the table and its scan can coexist in one schema."
            ),
            example_queries=examples(
                (
                    "Which organization is this credential for?",
                    "SELECT id, name FROM anthropic.main.all_organization()",
                ),
            ),
        )
        examples = _example(
            "SELECT id, name FROM anthropic.main.all_organization()",
            "Which organization is this credential for?",
        )

    @classmethod
    def on_bind(cls, params: BindParams[None]) -> BindResponse:
        return BindResponse(output_schema=cls.FIXED_SCHEMA)

    @classmethod
    def process(cls, params: ProcessParams[None], state: None, out: OutputCollector) -> None:
        org = api.organization(credentials=auth.for_call(params.secrets))
        batch, _ = build_filtered(params, [org])
        out.emit(batch)
        out.finish()


# --------------------------------------------------------------------------
# Members and invites
# --------------------------------------------------------------------------


@init_single_worker
class AllUsersFunction(_Scan):
    """Every organization member — the scan behind ``users``."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = USER_SCHEMA

    class Meta:
        name = "all_users"
        description = "Every member of the organization (the scan backing `users`)"
        categories = ["organization"]
        required_secrets = _SECRETS
        projection_pushdown = True
        #: `email = '...'` becomes the API's own (case-insensitive) filter.
        filter_pushdown = True
        auto_apply_filters = True
        tags = docs(
            category="organization",
            result_schema=USER_SCHEMA,
            llm=(
                "Every member of the organization with their email and organization role. Prefer "
                "the `users` table, which scans this. `id` joins to workspace_members.user_id, "
                "api_keys.created_by_id and principal_user_id, and messages_usage.account_id."
            ),
            md=(
                "Organization members, one row each.\n\n"
                "### Filtering\n\n"
                "`WHERE email = '...'` is sent to the API as its own filter, so a lookup by "
                "email costs one request whatever the size of the organization. The API matches "
                "case-insensitively; the exact SQL predicate is still applied to what comes back."
            ),
            example_queries=examples(
                (
                    "Admins of the organization",
                    "SELECT email, name FROM anthropic.main.all_users() WHERE role = 'admin'",
                ),
            ),
        )
        examples = _example(
            "SELECT email, name FROM anthropic.main.all_users() WHERE role = 'admin'",
            "Admins of the organization",
        )

    @classmethod
    def process(cls, params: ProcessParams[None], state: WalkState, out: OutputCollector) -> None:
        emit_list(
            params, state, out, endpoint=api.USERS, query=lambda: [("email", equality(params, "email"))]
        )


@init_single_worker
class AllInvitesFunction(_Scan):
    """Every organization invite — the scan behind ``invites``."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = INVITE_SCHEMA

    class Meta:
        name = "all_invites"
        description = "Every invite the organization has sent (the scan backing `invites`)"
        categories = ["organization"]
        required_secrets = _SECRETS
        projection_pushdown = True
        filter_pushdown = True
        auto_apply_filters = True
        tags = docs(
            category="organization",
            result_schema=INVITE_SCHEMA,
            llm=(
                "Invites sent to join the organization, with status pending, accepted or expired. "
                "Prefer the `invites` table. Use it to find outstanding invitations "
                "(status = 'pending') or ones that lapsed without being accepted."
            ),
            md=(
                "Organization invites, one row each.\n\n"
                "### Status\n\n"
                "`pending`, `accepted` and `expired` invites are all listed; an expired invite is "
                "not removed automatically. `WHERE email = '...'` is sent to the API as its own "
                "filter."
            ),
            example_queries=examples(
                (
                    "Invites still waiting to be accepted",
                    "SELECT email, role, expires_at FROM anthropic.main.all_invites() "
                    "WHERE status = 'pending' ORDER BY expires_at",
                ),
            ),
        )
        examples = _example(
            "SELECT email, role, expires_at FROM anthropic.main.all_invites() "
            "WHERE status = 'pending' ORDER BY expires_at",
            "Invites still waiting to be accepted",
        )

    @classmethod
    def process(cls, params: ProcessParams[None], state: WalkState, out: OutputCollector) -> None:
        emit_list(
            params, state, out, endpoint=api.INVITES, query=lambda: [("email", equality(params, "email"))]
        )


# --------------------------------------------------------------------------
# Workspaces
# --------------------------------------------------------------------------


@init_single_worker
class AllWorkspacesFunction(_Scan):
    """Every workspace, archived ones included — the scan behind ``workspaces``."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = WORKSPACE_SCHEMA

    class Meta:
        name = "all_workspaces"
        description = "Every workspace in the organization, archived included (the scan backing `workspaces`)"
        categories = ["organization"]
        required_secrets = _SECRETS
        projection_pushdown = True
        tags = docs(
            category="organization",
            result_schema=WORKSPACE_SCHEMA,
            llm=(
                "Every workspace in the organization, including archived ones — filter "
                "`WHERE NOT is_archived` for the live set. Prefer the `workspaces` table. `id` is "
                "the workspace_id that API keys, members, rate limits, usage and cost are "
                "attributed to. The default workspace is not listed here, and usage and cost "
                "attribute it to a NULL workspace_id."
            ),
            md=(
                "Workspaces, one row each, archived ones included so that historical usage and "
                "cost can still be joined to a name.\n\n"
                "### The default workspace\n\n"
                "The organization's default workspace does not appear in this list. In the usage "
                "and cost reports its `workspace_id` is NULL — `COALESCE(w.name, 'Default')` after "
                "a left join is the usual way to label it.\n\n"
                "### Data residency\n\n"
                "`allowed_inference_geos` is NULL when every geo is allowed; "
                "`inference_geo_unrestricted` says so explicitly."
            ),
            example_queries=examples(
                (
                    "Active workspaces, newest first",
                    "SELECT id, name, created_at FROM anthropic.main.all_workspaces() "
                    "WHERE NOT is_archived ORDER BY created_at DESC",
                ),
            ),
        )
        examples = _example(
            "SELECT id, name, created_at FROM anthropic.main.all_workspaces() "
            "WHERE NOT is_archived ORDER BY created_at DESC",
            "Active workspaces, newest first",
        )

    @classmethod
    def process(cls, params: ProcessParams[None], state: WalkState, out: OutputCollector) -> None:
        emit_list(
            params,
            state,
            out,
            endpoint=api.WORKSPACES,
            query=lambda: [("include_archived", True)],
            flatten=flatten_workspace,
        )


@init_single_worker
class AllWorkspaceMembersFunction(_Scan):
    """Every membership of every active workspace — the scan behind ``workspace_members``."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = WORKSPACE_MEMBER_SCHEMA

    class Meta:
        name = "all_workspace_members"
        description = "Every user's role in every active workspace (the scan backing `workspace_members`)"
        categories = ["organization"]
        required_secrets = _SECRETS
        projection_pushdown = True
        #: `workspace_id = '...'` narrows the walk to one workspace.
        filter_pushdown = True
        auto_apply_filters = True
        tags = docs(
            category="organization",
            result_schema=WORKSPACE_MEMBER_SCHEMA,
            llm=(
                "Who has access to which workspace, and with what workspace role. Prefer the "
                "`workspace_members` table. Join user_id to users.id for names and emails. Filter "
                "on workspace_id to read a single workspace cheaply."
            ),
            md=(
                "One row per (workspace, member).\n\n"
                "### Cost\n\n"
                "The API lists members one workspace at a time, so an unfiltered scan makes one "
                "request per active workspace. `WHERE workspace_id = '...'` is pushed down and "
                "reads only that workspace."
            ),
            example_queries=examples(
                (
                    "Workspace admins with their emails",
                    "SELECT w.name, u.email FROM anthropic.main.all_workspace_members() m "
                    "JOIN anthropic.main.users u ON u.id = m.user_id "
                    "JOIN anthropic.main.workspaces w ON w.id = m.workspace_id "
                    "WHERE m.workspace_role = 'workspace_admin' ORDER BY w.name, u.email",
                ),
            ),
        )
        examples = _example(
            "SELECT w.name, u.email FROM anthropic.main.all_workspace_members() m "
            "JOIN anthropic.main.users u ON u.id = m.user_id "
            "JOIN anthropic.main.workspaces w ON w.id = m.workspace_id "
            "WHERE m.workspace_role = 'workspace_admin' ORDER BY w.name, u.email",
            "Workspace admins with their emails",
        )

    @classmethod
    def process(cls, params: ProcessParams[None], state: WalkState, out: OutputCollector) -> None:
        emit_walk(
            params,
            state,
            out,
            keys=lambda: _active_workspace_ids(params),
            endpoint_for=api.workspace_members_endpoint,
            query=lambda: [],
            flatten=lambda rows, key: [{"workspace_id": key, **row} for row in rows],
        )


# --------------------------------------------------------------------------
# API keys
# --------------------------------------------------------------------------


def _api_key_query(params: ProcessParams[Any]) -> Query:
    """API-side filters for the key list, pushed only where the meaning is exact.

    ``status`` maps one to one. ``created_by_id`` maps to the API's
    ``created_by_user_id`` only when it names a *user*: the column also holds
    service-account IDs, and asking the user filter for one of those would
    return nothing where the predicate matches rows — the dangerous direction.
    ``workspace_id`` maps one to one: the API filters on the key's ``scope``
    (the Default Workspace's real ID returns the keys bound to it, and
    organization-scoped keys match no workspace), which is exactly what the
    column holds — an organization-scoped key's ``workspace_id`` is NULL, and
    ``=`` never matches NULL either.
    """
    status = equality(params, "status")
    creator = equality(params, "created_by_id")
    return [
        ("status", status if status in _API_KEY_STATUSES else None),
        ("created_by_user_id", creator if creator and creator.startswith("user_") else None),
        ("workspace_id", equality(params, "workspace_id")),
    ]


@init_single_worker
class AllApiKeysFunction(_Scan):
    """Every API key's metadata — the scan behind ``api_keys``."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = API_KEY_SCHEMA

    class Meta:
        name = "all_api_keys"
        description = "Metadata of every API key in the organization (the scan backing `api_keys`)"
        categories = ["access"]
        required_secrets = _SECRETS
        projection_pushdown = True
        filter_pushdown = True
        auto_apply_filters = True
        tags = docs(
            category="access",
            result_schema=API_KEY_SCHEMA,
            llm=(
                "Every API key in the organization: name, status, workspace, who created it and "
                "when it expires. Prefer the `api_keys` table. Never contains the secret — only a "
                "redacted hint. `id` joins to messages_usage.api_key_id, so usage can be "
                "attributed to a key by name."
            ),
            md=(
                "API key metadata, one row per key. The secret value is never returned by the "
                "API, only `partial_key_hint`.\n\n"
                "### Workspace\n\n"
                "`workspace_id` is taken from the key's `scope`, which carries the real ID even "
                "for the default workspace — unlike the API's deprecated top-level field, which "
                "is NULL there. Organization-scoped keys have `scope_type = 'organization'` and a "
                "NULL `workspace_id`.\n\n"
                "### Filtering\n\n"
                "`status = '...'`, `workspace_id = '...'` and `created_by_id = 'user_...'` are sent "
                "to the API as its own filters."
            ),
            example_queries=examples(
                (
                    "Active keys that never expire",
                    "SELECT name, workspace_id, created_at FROM anthropic.main.all_api_keys() "
                    "WHERE status = 'active' AND expires_at IS NULL ORDER BY created_at",
                ),
            ),
        )
        examples = _example(
            "SELECT name, workspace_id, created_at FROM anthropic.main.all_api_keys() "
            "WHERE status = 'active' AND expires_at IS NULL ORDER BY created_at",
            "Active keys that never expire",
        )

    @classmethod
    def process(cls, params: ProcessParams[None], state: WalkState, out: OutputCollector) -> None:
        emit_list(
            params,
            state,
            out,
            endpoint=api.API_KEYS,
            query=lambda: _api_key_query(params),
            flatten=flatten_api_key,
        )


# --------------------------------------------------------------------------
# Rate limits
# --------------------------------------------------------------------------


def _group_type_query(params: ProcessParams[Any]) -> list[tuple[str, Any]]:
    group_type = equality(params, "group_type")
    return [("group_type", group_type if group_type in _GROUP_TYPES else None)]


@init_single_worker
class AllRateLimitsFunction(_Scan):
    """The organization's configured rate limits — the scan behind ``rate_limits``."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = RATE_LIMIT_SCHEMA

    class Meta:
        name = "all_rate_limits"
        description = "The organization's Messages API rate limits (the scan backing `rate_limits`)"
        categories = ["rate-limits"]
        required_secrets = _SECRETS
        projection_pushdown = True
        filter_pushdown = True
        auto_apply_filters = True
        tags = docs(
            category="rate-limits",
            result_schema=RATE_LIMIT_SCHEMA,
            llm=(
                "The rate limits configured for the organization, one row per (limit group, "
                "limiter) — e.g. requests_per_minute for the Claude Sonnet model group. Prefer "
                "the `rate_limits` table. For a given model, filter with "
                "`list_contains(models, 'claude-...')`."
            ),
            md=(
                "Configured limits, flattened to one row per limiter.\n\n"
                "### Model groups\n\n"
                "Limits apply to a *group*: a model family (`group_type = 'model_group'`, with its "
                "models and aliases in `models`) or an API surface such as `batch` or "
                "`web_search`. Use `list_contains(models, ...)` to find the limits for one model.\n\n"
                "### Workspace limits\n\n"
                "Workspaces can lower these; see `workspace_rate_limits`."
            ),
            example_queries=examples(
                (
                    "Per-minute limits for every model group",
                    "SELECT group_display_name, limit_type, value FROM anthropic.main.all_rate_limits() "
                    "WHERE group_type = 'model_group' ORDER BY group_display_name, limit_type",
                ),
            ),
        )
        examples = _example(
            "SELECT group_display_name, limit_type, value FROM anthropic.main.all_rate_limits() "
            "WHERE group_type = 'model_group' ORDER BY group_display_name, limit_type",
            "Per-minute limits for every model group",
        )

    @classmethod
    def process(cls, params: ProcessParams[None], state: WalkState, out: OutputCollector) -> None:
        emit_walk(
            params,
            state,
            out,
            keys=lambda: [""],
            endpoint_for=lambda _key: api.RATE_LIMITS,
            query=lambda: _group_type_query(params),
            flatten=lambda rows, _key: flatten_rate_limits(rows),
        )


@init_single_worker
class AllWorkspaceRateLimitsFunction(_Scan):
    """Every active workspace's effective rate limits — the scan behind ``workspace_rate_limits``."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = WORKSPACE_RATE_LIMIT_SCHEMA

    class Meta:
        name = "all_workspace_rate_limits"
        description = (
            "Each active workspace's rate limits, inherited and overridden (backs `workspace_rate_limits`)"
        )
        categories = ["rate-limits"]
        required_secrets = _SECRETS
        projection_pushdown = True
        filter_pushdown = True
        auto_apply_filters = True
        tags = docs(
            category="rate-limits",
            result_schema=WORKSPACE_RATE_LIMIT_SCHEMA,
            llm=(
                "The rate limits in force for each active workspace, one row per (workspace, "
                "group, limiter), with `source` saying whether the workspace overrides the "
                "organization's value or inherits it. Prefer the `workspace_rate_limits` table. "
                "Filter `source = 'workspace'` for overrides only."
            ),
            md=(
                "Effective per-workspace limits. Inherited values are included (the API's "
                "`include_inherited=true`), so every workspace shows its complete picture and "
                "`source` distinguishes `workspace` overrides from `organization` defaults. "
                "`org_limit` carries the organization's value alongside for comparison.\n\n"
                "### Cost\n\n"
                "One request per active workspace; `WHERE workspace_id = '...'` reads just one."
            ),
            example_queries=examples(
                (
                    "Workspaces that override an organization limit",
                    "SELECT workspace_id, group_display_name, limit_type, value, org_limit "
                    "FROM anthropic.main.all_workspace_rate_limits() WHERE source = 'workspace'",
                ),
            ),
        )
        examples = _example(
            "SELECT workspace_id, group_display_name, limit_type, value, org_limit "
            "FROM anthropic.main.all_workspace_rate_limits() WHERE source = 'workspace'",
            "Workspaces that override an organization limit",
        )

    @classmethod
    def process(cls, params: ProcessParams[None], state: WalkState, out: OutputCollector) -> None:
        emit_walk(
            params,
            state,
            out,
            keys=lambda: _active_workspace_ids(params),
            endpoint_for=api.workspace_rate_limits_endpoint,
            query=lambda: [("include_inherited", True), *_group_type_query(params)],
            flatten=flatten_workspace_rate_limits,
        )


ADMIN_FUNCTIONS: list[type] = [
    AllOrganizationFunction,
    AllUsersFunction,
    AllInvitesFunction,
    AllWorkspacesFunction,
    AllWorkspaceMembersFunction,
    AllApiKeysFunction,
    AllRateLimitsFunction,
    AllWorkspaceRateLimitsFunction,
]
