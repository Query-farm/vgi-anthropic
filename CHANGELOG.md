# Changelog

## 0.1.0 — 2026-10-02

First release: the Anthropic Admin, Usage & Cost, and Claude Code Analytics
APIs as a read-only VGI catalog.

### Added

- **Tables** — `organization`, `users`, `invites`, `workspaces` (archived
  included), `workspace_members`, `api_keys`, `rate_limits` and
  `workspace_rate_limits`, each backed by an `all_<table>()` paged scan.
- **Reports** — `messages_usage(starting_at)`, `cost_report(starting_at)` and
  `claude_code_usage(starting_date)`, streamed one API page per tick so a
  `LIMIT` stops early.
- **Lookups** — `user_by_id`, `workspace_by_id` and `api_key_by_id`, blended so
  they compose under `LATERAL`; an unknown ID yields no row.
- **Grouping follows the query.** Without `group_by`, the usage and cost reports
  group by exactly the dimension columns a query references, so a selected
  dimension column reads its value instead of NULL.
- **Exact money.** Cost is `DECIMAL` in both cents (as the API reports it) and
  dollars.
- **Credentials as a redacted `anthropic_admin` DuckDB secret** (Admin API key,
  organization-scoped personal or service-account key, or `org:admin` OAuth
  token), selected by type and `SCOPE`, with no environment-variable fallback.
  Keys are never judged by prefix; a workspace-scoped key's 401 names the
  credentials that would work.
- Filter pushdown for `email`, `status`, `created_by_id` (users only),
  `workspace_id` (members, rate limits and API keys), `group_type` and every
  usage dimension.
- Retries with backoff on 429/529/5xx and transport errors, honouring
  `retry-after`.
- `x-should-retry` is obeyed in both directions, as in Anthropic's SDKs.
- End-to-end SQL tests against a local fake of the Admin API, over stdio and
  over HTTP, run on every push; the fake is validated against the Anthropic
  SDK's models.
- `vgi-lint`'s behavioural tier runs in CI against the fake
  (`scripts/lint-behavioural.sh`), comparing every example's declared result.
- Live tests of every refusal the real API gives a non-admin credential.
