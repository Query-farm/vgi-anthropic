"""VGI worker exposing the Anthropic Admin, Usage and Cost APIs to DuckDB/SQL (read-only).

    CREATE SECRET anthropic (TYPE anthropic_admin, api_key 'sk-ant-admin01-...');
    ATTACH 'anthropic' (TYPE vgi, LOCATION 'uv run anthropic_worker.py');

    SELECT email, role FROM anthropic.users;
    SELECT model, sum(output_tokens) FROM anthropic.messages_usage(current_date - 7) GROUP BY model;

Deliberately not exposed: the Messages API itself (this worker reports on
model usage, it does not make any), and every write the Admin API offers —
inviting, removing and re-roling members, creating and archiving workspaces,
rotating keys. An Admin key authorizes all of that; this worker cannot do any
of it.

Function names are bare (``cost_report``, not ``anthropic_cost_report``)
because they are already qualified by the catalog they live in.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from vgi import Worker
from vgi.catalog import (
    Catalog,
    ConstraintInfo,
    ReadOnlyCatalogInterface,
    ReferencedKey,
    Schema,
    constraint_record,
)
from vgi.catalog.catalog_interface import CatalogInfo
from vgi.catalog.descriptors import Table

from vgi_anthropic import __version__, auth
from vgi_anthropic.admin import (
    ADMIN_FUNCTIONS,
    AllApiKeysFunction,
    AllInvitesFunction,
    AllOrganizationFunction,
    AllRateLimitsFunction,
    AllUsersFunction,
    AllWorkspaceMembersFunction,
    AllWorkspaceRateLimitsFunction,
    AllWorkspacesFunction,
)
from vgi_anthropic.lookups import LOOKUP_FUNCTIONS
from vgi_anthropic.meta import column_comments, docs, examples, keywords
from vgi_anthropic.reports import REPORT_FUNCTIONS
from vgi_anthropic.schemas import (
    API_KEY_SCHEMA,
    INVITE_SCHEMA,
    ORGANIZATION_SCHEMA,
    RATE_LIMIT_SCHEMA,
    USER_SCHEMA,
    WORKSPACE_MEMBER_SCHEMA,
    WORKSPACE_RATE_LIMIT_SCHEMA,
    WORKSPACE_SCHEMA,
)

IMPLEMENTATION_VERSION = __version__
DATA_VERSION_SPEC = f"=={__version__}"
SOURCE_URL = "https://github.com/Query-farm/vgi-anthropic"

_FUNCTIONS = [*ADMIN_FUNCTIONS, *REPORT_FUNCTIONS, *LOOKUP_FUNCTIONS]

_EXAMPLE_QUERIES = examples(
    ("Which organization is this?", "SELECT id, name FROM anthropic.main.organization"),
    (
        "Members by role",
        "SELECT role, count(*) AS members FROM anthropic.main.users GROUP BY role ORDER BY members DESC",
    ),
    (
        "Token volume by model over the last week",
        "SELECT model, sum(output_tokens) AS output_tokens "
        "FROM anthropic.main.messages_usage(current_date - INTERVAL 7 DAY) "
        "GROUP BY model ORDER BY output_tokens DESC",
    ),
    (
        "Month-to-date spend by workspace, with names",
        "SELECT COALESCE(w.name, 'Default') AS workspace, sum(c.amount_usd) AS usd "
        "FROM anthropic.main.cost_report(date_trunc('month', current_date)) c "
        "LEFT JOIN anthropic.main.workspaces w ON w.id = c.workspace_id "
        "GROUP BY ALL ORDER BY usd DESC",
    ),
    (
        "Active keys nobody has used in 30 days",
        "SELECT k.name, k.created_at FROM anthropic.main.api_keys k WHERE k.status = 'active' "
        "AND k.id NOT IN (SELECT api_key_id FROM anthropic.main.messages_usage("
        "current_date - INTERVAL 30 DAY) WHERE api_key_id IS NOT NULL)",
    ),
    (
        "Last week's Claude Code usage per developer",
        "SELECT email_address, sum(num_sessions) AS sessions, sum(estimated_cost_usd) AS usd "
        "FROM anthropic.main.claude_code_usage(current_date - INTERVAL 7 DAY, ending_date => current_date) "
        "WHERE actor_type = 'user_actor' GROUP BY email_address ORDER BY usd DESC",
    ),
)

#: Examples the linter runs against a live worker. Every organization has
#: exactly one row in `organization` and at least one admin, but may have no
#: usage at all — so nothing here assumes traffic, workspaces or keys exist.
_EXECUTABLE_EXAMPLES = json.dumps(
    [
        {
            "name": name,
            "description": description,
            # The expected result must sit on the statement, not the example:
            # vgi-lint only compares results declared per statement, and one
            # at the top level is accepted but silently never checked.
            "sql": [{"sql": sql, "expected_result": expected}],
        }
        for name, description, sql, expected in [
            (
                "credential_resolves_to_one_organization",
                "The Admin credential belongs to exactly one organization.",
                "SELECT count(*) = 1 FROM anthropic.main.organization",
                [[True]],
            ),
            (
                "someone_administers_the_organization",
                "Whoever issued the Admin key is an admin, so the role exists.",
                "SELECT count(*) > 0 FROM anthropic.main.users "
                "WHERE role IN ('admin', 'owner', 'primary_owner')",
                [[True]],
            ),
            (
                "users_table_and_scan_agree",
                "The `users` table and the `all_users` function behind it are the same rows.",
                "SELECT (SELECT count(*) FROM anthropic.main.users) "
                "= (SELECT count(*) FROM anthropic.main.all_users())",
                [[True]],
            ),
            (
                "cost_is_exact_decimal",
                "Money comes back as DECIMAL, never a float.",
                # DESCRIBE reads the declared type, so the check holds on a day
                # with no spend instead of passing vacuously.
                "SELECT column_type FROM (DESCRIBE SELECT amount_usd "
                "FROM anthropic.main.cost_report(current_date - INTERVAL 1 DAY))",
                [["DECIMAL(38,14)"]],
            ),
            (
                "usage_buckets_are_days_by_default",
                "Default buckets are one day wide.",
                "SELECT bool_and(bucket_end - bucket_start = INTERVAL 1 DAY) IS NOT FALSE "
                "FROM anthropic.main.messages_usage(current_date - INTERVAL 3 DAY)",
                [[True]],
            ),
        ]
    ]
)

#: The public half of the agent-suitability suite: what an analyst is asked,
#: never how to answer it. The graders live in vgi-agent-tests.yaml, outside the
#: catalog, so the metadata under test cannot leak its own solutions.
_AGENT_TEST_TASKS = json.dumps(
    [
        {"name": name, "prompt": prompt}
        for name, prompt in [
            ("which_organization", "Which Anthropic organization does this connection read from?"),
            ("who_are_the_admins", "Who are the admins of our Anthropic organization?"),
            (
                "outstanding_invites",
                "Which invitations to join our organization are still outstanding, and when do they expire?",
            ),
            ("active_workspaces", "List our active workspaces and when each was created."),
            ("workspace_admins", "Who administers each workspace? Give workspace names and emails."),
            ("keys_without_expiry", "List our active API keys that have no expiry date."),
            ("model_rate_limit", "What is our requests-per-minute limit for Claude Sonnet models?"),
            (
                "workspace_rate_overrides",
                "Which workspaces override an organization rate limit, and how do the values compare?",
            ),
            (
                "spend_this_month",
                "How much have we spent on the Anthropic API so far this month, in dollars?",
            ),
            ("spend_by_workspace", "Break down the last 30 days of Anthropic spend by workspace name."),
            ("tokens_by_model", "Which Claude models did we use most last week, by output tokens?"),
            (
                "claude_code_adoption",
                "Who used Claude Code the most over the past week, by number of sessions?",
            ),
            ("key_creators", "For each active API key, which person created it?"),
            (
                "newest_workspace_residency",
                "What are the data-residency settings of our most recently created workspace?",
            ),
            ("newest_key_status", "What is the status of our newest API key, and which workspace is it in?"),
        ]
    ]
)

_CATEGORIES = json.dumps(
    [
        {
            "name": "organization",
            "title": "Organization & Members",
            "description": "The organization, its members, invites, workspaces and workspace roles.",
            "keywords": ["users", "members", "invites", "workspaces", "roles"],
        },
        {
            "name": "access",
            "title": "API Keys",
            "description": "Every API key's metadata — status, workspace, creator, expiry. Never the secret.",
            "keywords": ["api keys", "credentials", "audit"],
        },
        {
            "name": "rate-limits",
            "title": "Rate Limits",
            "description": "Configured request and token limits, for the organization and each workspace.",
            "keywords": ["rate limits", "rpm", "tpm", "quota"],
        },
        {
            "name": "usage",
            "title": "Token Usage",
            "description": "Messages API token consumption over time, by model, workspace, key and tier.",
            "keywords": ["usage", "tokens", "consumption"],
        },
        {
            "name": "cost",
            "title": "Cost",
            "description": "Daily spend in exact decimal dollars, by workspace and line item.",
            "keywords": ["cost", "spend", "billing", "usd"],
        },
        {
            "name": "claude-code",
            "title": "Claude Code Analytics",
            "description": "Per-developer Claude Code sessions, code changes, commits, PRs, tokens and cost.",
            "keywords": ["claude code", "developer productivity", "adoption"],
        },
    ]
)

_COMMON = {"provider": "anthropic", "domain": "ai-platform-administration"}

_CATALOG_TAGS = {
    **_COMMON,
    "vgi.title": "Anthropic Admin, Usage & Cost",
    "vgi.subtitle": (
        "Read-only view of your Anthropic organization's members, API keys, token usage and cost."
    ),
    "vgi.source_url": SOURCE_URL,
    "vgi.author": "Query Farm LLC <hello@query.farm>",
    "vgi.copyright": (
        "Worker (c) 2026 Query Farm LLC - https://query.farm. Organization data is the "
        "customer's own, read from Anthropic's Admin API under Anthropic's terms."
    ),
    "vgi.license": "MIT",
    "vgi.support_contact": "https://github.com/Query-farm/vgi-anthropic/issues",
    "vgi.support_policy_url": "https://github.com/Query-farm/vgi-anthropic/blob/main/README.md",
    "vgi.keywords": keywords(
        "anthropic", "claude", "admin api", "usage", "cost", "spend", "api keys", "workspaces", "claude code"
    ),
    "vgi.executable_examples": _EXECUTABLE_EXAMPLES,
    "vgi.agent_test_tasks": _AGENT_TEST_TASKS,
    "vgi.doc_llm": (
        "Your Anthropic organization's administrative and billing data: members, invites, "
        "workspaces, API keys, rate limits, token usage, daily cost and Claude Code analytics. "
        "Reach for it to answer who has access to what, what is being used and by which key or "
        "workspace, and what it costs. Requires an Admin API key (or organization-scoped key) "
        "in an `anthropic_admin` secret. "
        "Read-only: it cannot change members, keys or workspaces, and it does not call models."
    ),
    "vgi.doc_md": (
        "The Anthropic Admin API, Usage & Cost API and Claude Code Analytics API as SQL.\n\n"
        "### Access\n\n"
        "Every object needs an Admin API key (`sk-ant-admin01-...`) or an organization-scoped "
        "personal or service-account key, provided as a DuckDB secret:\n\n"
        "```sql\nCREATE SECRET anthropic (TYPE anthropic_admin, api_key 'sk-ant-admin01-...');\n```\n\n"
        "A workspace-scoped key is refused by the API. The key is never an ATTACH option, so it "
        "cannot leak "
        "through `duckdb_databases()`.\n\n"
        "### Shape\n\n"
        "The organization's own records — `users`, `invites`, `workspaces`, `workspace_members`, "
        "`api_keys`, `rate_limits`, `workspace_rate_limits` — are tables. The time-series reports — "
        "`messages_usage`, `cost_report`, `claude_code_usage` — are functions taking a start time. "
        "IDs join across all of them.\n\n"
        "### Read-only\n\n"
        "The Admin API can invite and remove members, archive workspaces and disable keys. This "
        "worker issues `GET` requests only, enforced structurally and by a CI guard."
    ),
}

_SCHEMA_TAGS = {
    **_COMMON,
    "vgi.title": "Anthropic Organization",
    "vgi.categories": _CATEGORIES,
    "vgi.keywords": keywords("anthropic", "usage", "cost", "api keys", "workspaces", "members"),
    "vgi.example_queries": _EXAMPLE_QUERIES,
    "vgi.doc_llm": (
        "Everything about one Anthropic organization in one schema. Tables hold the current state "
        "(members, workspaces, keys, limits); functions report history (usage, cost, Claude Code). "
        "Join the reports' workspace_id, api_key_id and account_id to the tables for names. A "
        "NULL workspace_id in usage or cost is the default workspace."
    ),
    "vgi.doc_md": (
        "One schema per organization.\n\n"
        "### Tables for state, functions for history\n\n"
        "The tables need no arguments and return the organization's current records. The reports "
        "are functions because they need a start time; each is bucketed in UTC.\n\n"
        "### Report dimensions follow your query\n\n"
        "`messages_usage` and `cost_report` group by exactly the dimension columns a query "
        "references: selecting `model` next to `sum(output_tokens)` breaks usage down by model "
        "without a `group_by` argument, and totals are the same whichever columns you pick.\n\n"
        "### The default workspace\n\n"
        "It is not in `workspaces`, and usage and cost report it with a NULL `workspace_id`. Use a "
        "LEFT JOIN and `COALESCE(w.name, 'Default')`."
    ),
}


def _table_docs(
    *, category: str, title: str, llm: str, md: str, sql: str, description: str, terms: tuple[str, ...]
) -> dict[str, str]:
    return docs(
        category=category,
        llm=llm,
        md=md,
        example_queries=examples((description, sql)),
        extra={**_COMMON, "vgi.title": title, "vgi.keywords": keywords(*terms)},
    )


def _constraints(
    table: str,
    *,
    key: tuple[str, ...],
    not_null: tuple[str, ...],
    references: tuple[tuple[str, str], ...] = (),
) -> tuple[ConstraintInfo, ...]:
    """Canonical vgi.v3 constraint records: NOT NULL columns, the primary key, and foreign keys.

    ``references`` pairs a column with the table whose single-column ``id`` key it references.
    The tables are read-only, so each record is a statement about every row the API returns.
    """

    def record(constraint_id: str, kind: str, columns: list[list[str]], **extra: Any) -> ConstraintInfo:
        return constraint_record(
            ["main"],
            table,
            constraint_id,
            kind,
            columns,
            enforcement="ENFORCED",
            validation="ALL_ROWS",
            **extra,
        )

    return (
        *(record(f"{col}_not_null", "NOT_NULL", [[col]]) for col in not_null),
        record("pk", "PRIMARY_KEY", [[col] for col in key]),
        *(
            record(
                f"{col}_fk",
                "FOREIGN_KEY",
                [[col]],
                references=ReferencedKey(
                    schema_path=["main"], table=target, columns=[["id"]], constraint_id="pk"
                ),
            )
            for col, target in references
        ),
    )


_TABLES = [
    Table(
        name="organization",
        function=AllOrganizationFunction,
        comment="The organization the Admin credential belongs to",
        tags=_table_docs(
            category="organization",
            title="Organization Identity",
            llm="One row: the organization's ID and name. Confirms which organization the secret points at.",
            md="The organization behind the configured `anthropic_admin` secret, as a single row.",
            sql="SELECT id, name FROM anthropic.main.organization",
            description="Which organization is this?",
            terms=("organization", "account"),
        ),
        column_comments=column_comments(ORGANIZATION_SCHEMA),
        constraints=_constraints("organization", key=("id",), not_null=("id",)),
        cardinality_estimate=1,
        cardinality_max=1,
    ),
    Table(
        name="users",
        function=AllUsersFunction,
        comment="Every member of the organization, with their role",
        tags=_table_docs(
            category="organization",
            title="Organization Members",
            llm=(
                "Every member with email, name and organization role. `id` joins to "
                "workspace_members.user_id, api_keys.created_by_id and messages_usage.account_id."
            ),
            md=(
                "Organization members. `WHERE email = '...'` is a single API request whatever the size "
                "of the organization."
            ),
            sql="SELECT role, count(*) AS members FROM anthropic.main.users GROUP BY role",
            description="Members by role",
            terms=("users", "members", "roles", "people"),
        ),
        column_comments=column_comments(USER_SCHEMA),
        constraints=_constraints("users", key=("id",), not_null=("id",)),
    ),
    Table(
        name="invites",
        function=AllInvitesFunction,
        comment="Invites to join the organization: pending, accepted and expired",
        tags=_table_docs(
            category="organization",
            title="Organization Invites",
            llm="Invites sent to join the organization. status = 'pending' are outstanding.",
            md="Pending, accepted and expired invites. Expired invites are listed until removed.",
            sql="SELECT email, role, expires_at FROM anthropic.main.invites WHERE status = 'pending'",
            description="Outstanding invites",
            terms=("invites", "invitations", "onboarding"),
        ),
        column_comments=column_comments(INVITE_SCHEMA),
        constraints=_constraints("invites", key=("id",), not_null=("id",)),
    ),
    Table(
        name="workspaces",
        function=AllWorkspacesFunction,
        comment="Every workspace, archived ones included (the default workspace is not listed)",
        tags=_table_docs(
            category="organization",
            title="Workspace Directory",
            llm=(
                "Every workspace including archived (filter NOT is_archived for live ones). The "
                "default workspace is not listed; reports give it a NULL workspace_id."
            ),
            md=(
                "Workspaces, archived included so historical usage and cost still join to a name. "
                "The default workspace is not listed — label it with `COALESCE(name, 'Default')` "
                "after a LEFT JOIN."
            ),
            sql="SELECT id, name, created_at FROM anthropic.main.workspaces WHERE NOT is_archived",
            description="Active workspaces",
            terms=("workspaces", "projects", "environments"),
        ),
        column_comments=column_comments(WORKSPACE_SCHEMA),
        constraints=_constraints("workspaces", key=("id",), not_null=("id",)),
    ),
    Table(
        name="workspace_members",
        function=AllWorkspaceMembersFunction,
        comment="Each member's role in each active workspace",
        tags=_table_docs(
            category="organization",
            title="Workspace Memberships",
            llm="Who can access which workspace, with what workspace role. Join user_id to users.id.",
            md=(
                "One row per (workspace, member). One request per active workspace unless "
                "`workspace_id = '...'` narrows it."
            ),
            sql=(
                "SELECT workspace_id, count(*) AS members FROM anthropic.main.workspace_members "
                "GROUP BY workspace_id"
            ),
            description="Members per workspace",
            terms=("workspace members", "access", "roles"),
        ),
        column_comments=column_comments(WORKSPACE_MEMBER_SCHEMA),
        constraints=_constraints(
            "workspace_members",
            key=("workspace_id", "user_id"),
            not_null=("workspace_id", "user_id"),
            references=(("workspace_id", "workspaces"), ("user_id", "users")),
        ),
    ),
    Table(
        name="api_keys",
        function=AllApiKeysFunction,
        comment="Metadata of every API key in the organization (never the secret)",
        tags=_table_docs(
            category="access",
            title="API Key Inventory",
            llm=(
                "Every API key: name, status, workspace, creator, expiry. Never the secret. id joins "
                "to messages_usage.api_key_id."
            ),
            md=(
                "API key metadata. `workspace_id` is the real workspace ID even for the default "
                "workspace. `status`, `workspace_id` and `created_by_id = 'user_...'` filters are "
                "sent to the API."
            ),
            sql=(
                "SELECT name, created_at FROM anthropic.main.api_keys "
                "WHERE status = 'active' AND expires_at IS NULL"
            ),
            description="Active keys without an expiry",
            terms=("api keys", "credentials", "security", "audit"),
        ),
        column_comments=column_comments(API_KEY_SCHEMA),
        constraints=_constraints("api_keys", key=("id",), not_null=("id",)),
        # No foreign key to `workspaces`: a key in the default workspace carries
        # that workspace's real ID, which the workspace list does not include, so
        # the constraint would be false.
    ),
    Table(
        name="rate_limits",
        function=AllRateLimitsFunction,
        comment="The organization's configured rate limits, one row per group and limiter",
        tags=_table_docs(
            category="rate-limits",
            title="Organization Rate Limits",
            llm=(
                "Configured limits per (group, limiter), e.g. requests_per_minute for a model group. "
                "Use list_contains(models, 'claude-...') to find one model's limits."
            ),
            md="Configured limits, flattened to one row per limiter. Workspaces may lower them.",
            sql=(
                "SELECT group_display_name, limit_type, value FROM anthropic.main.rate_limits "
                "WHERE group_type = 'model_group' ORDER BY 1, 2"
            ),
            description="Limits per model group",
            terms=("rate limits", "rpm", "tpm", "quota"),
        ),
        column_comments=column_comments(RATE_LIMIT_SCHEMA),
        constraints=_constraints("rate_limits", key=("id", "limit_type"), not_null=("id", "limit_type")),
    ),
    Table(
        name="workspace_rate_limits",
        function=AllWorkspaceRateLimitsFunction,
        comment="Each active workspace's rate limits, with whether each is overridden or inherited",
        tags=_table_docs(
            category="rate-limits",
            title="Per-Workspace Rate Limits",
            llm=(
                "Limits in force per workspace; source = 'workspace' marks an override of the "
                "organization's value, which is in org_limit."
            ),
            md=(
                "Effective per-workspace limits, inherited ones included. One request per active "
                "workspace.\n\n"
                "### Joining\n\n"
                "Join `workspace_id` to `workspaces.id` for the workspace name, and `rate_limit_id` to "
                "`rate_limits.id` (with `limit_type`) for the organization entry it refines."
            ),
            sql=(
                "SELECT workspace_id, limit_type, value, org_limit FROM anthropic.main.workspace_rate_limits "
                "WHERE source = 'workspace'"
            ),
            description="Workspace overrides",
            terms=("rate limits", "workspace", "overrides"),
        ),
        column_comments=column_comments(WORKSPACE_RATE_LIMIT_SCHEMA),
        # rate_limit_id joins to rate_limits.id, but that is not a key there (one
        # row per limiter), so it cannot be declared as a foreign key.
        constraints=_constraints(
            "workspace_rate_limits",
            key=("workspace_id", "rate_limit_id", "limit_type"),
            not_null=("workspace_id", "limit_type"),
            references=(("workspace_id", "workspaces"),),
        ),
    ),
]

_ANTHROPIC_CATALOG = Catalog(
    name="anthropic",
    default_schema="main",
    comment=(
        "Read-only Anthropic organization data: members, workspaces, API keys, rate limits, "
        "token usage, cost and Claude Code analytics"
    ),
    tags=_CATALOG_TAGS,
    schemas=[
        Schema(
            path=["main"],
            comment="Anthropic Admin, Usage & Cost APIs — requires an anthropic_admin secret",
            tags=_SCHEMA_TAGS,
            functions=list(_FUNCTIONS),
            tables=_TABLES,
        ),
    ],
)


class AnthropicCatalog(ReadOnlyCatalogInterface):
    """Advertises the worker's versions and its required credential type."""

    catalog = _ANTHROPIC_CATALOG
    catalog_name = _ANTHROPIC_CATALOG.name
    secret_types = [auth.SECRET_SPEC]

    def catalogs(self) -> list[CatalogInfo]:
        """Advertise the single read-only catalog."""
        return [
            CatalogInfo(
                name=self._effective_catalog_name,
                implementation_version=IMPLEMENTATION_VERSION,
                data_version_spec=DATA_VERSION_SPEC,
                source_url=SOURCE_URL,
                attach_option_specs=[],
            )
        ]


class AnthropicWorker(Worker):
    """Worker process hosting the read-only Anthropic catalog."""

    catalog = _ANTHROPIC_CATALOG
    catalog_interface = AnthropicCatalog


def main() -> None:
    """Run the worker (stdio by default; pass ``--http`` for the HTTP server)."""
    AnthropicWorker.main()


def main_http() -> None:
    """Run the worker over HTTP."""
    argv: list[Any] = sys.argv[1:]
    if "--http" not in argv:
        argv = ["--http", *argv]
    sys.argv = [sys.argv[0], *argv]
    AnthropicWorker.main()
