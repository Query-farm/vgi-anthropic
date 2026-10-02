"""The fake Admin API must speak the same shapes as Anthropic's own SDK.

Without an Admin key, the end-to-end suite's fidelity rests on
:mod:`tests.fake_admin_api` — so a field the fake names wrongly would make the
worker look correct while it reads NULL from the real API. Anthropic's Python
SDK ships typed models for every admin object the worker reads, generated from
the same API definition as the docs. Each fake response is validated against
them in both directions:

* every field the fake sends is one the SDK knows (no invented names), and
* every field the SDK declares is one the fake sends (no silently missing
  column that the worker would always read as NULL).

The SDK (1.11.0) has no models for the usage, cost and Claude Code *reports*,
so those remain checked against the documentation only — see
``tests/test_end_to_end.py`` and the fake's own docstrings.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from anthropic.types.organization import (
    APIKey,
    OrganizationInvite,
    OrganizationRateLimit,
    OrganizationUser,
    Workspace,
    WorkspaceMember,
)
from anthropic.types.organization.workspaces import WorkspaceRateLimit
from anthropic.types.organization_info import OrganizationInfo
from pydantic import BaseModel

from tests.fake_admin_api import ADMIN_KEY, FakeAdminApi

#: (path, SDK model, whether the response is a list page)
ENDPOINTS: list[tuple[str, type[BaseModel], bool]] = [
    ("/v1/organizations/me", OrganizationInfo, False),
    ("/v1/organizations/users?limit=1000", OrganizationUser, True),
    ("/v1/organizations/users/user_01", OrganizationUser, False),
    ("/v1/organizations/invites?limit=1000", OrganizationInvite, True),
    ("/v1/organizations/workspaces?limit=1000&include_archived=true", Workspace, True),
    ("/v1/organizations/workspaces/wrkspc_01", Workspace, False),
    ("/v1/organizations/workspaces/wrkspc_01/members?limit=1000", WorkspaceMember, True),
    ("/v1/organizations/api_keys?limit=1000", APIKey, True),
    ("/v1/organizations/api_keys/apikey_02", APIKey, False),
    ("/v1/organizations/rate_limits", OrganizationRateLimit, True),
    (
        "/v1/organizations/workspaces/wrkspc_01/rate_limits?include_inherited=true",
        WorkspaceRateLimit,
        True,
    ),
]


@pytest.fixture(scope="module")
def fake() -> Iterator[FakeAdminApi]:
    server = FakeAdminApi().start()
    yield server
    server.stop()


def _objects(fake: FakeAdminApi, path: str, is_list: bool) -> list[dict[str, Any]]:
    headers = {"x-api-key": ADMIN_KEY, "anthropic-version": "2023-06-01"}
    objects: list[dict[str, Any]] = []
    cursor: dict[str, str] = {}
    for _ in range(50):
        response = httpx.get(fake.url + path, params=cursor, headers=headers)
        assert response.status_code == 200, response.text
        body = response.json()
        if not is_list:
            return [body]
        objects.extend(body["data"])
        if not body.get("has_more"):
            return objects
        cursor = {"after_id": body["last_id"]} if "last_id" in body else {"page": body["next_page"]}
    raise AssertionError(f"{path} never stopped paging")


#: Fields the API documents as deprecated-but-still-returned, which the SDK no
#: longer models. The fake keeps sending them because the real API does; the
#: worker must not depend on them, and reads ``scope`` and ``group.type``.
DEPRECATED = {"workspace_id": APIKey, "group_type": (OrganizationRateLimit, WorkspaceRateLimit)}


def _deprecated(model: type[BaseModel]) -> set[str]:
    return {name for name, owners in DEPRECATED.items() if issubclass(model, owners)}


def _undeclared(model: BaseModel, prefix: str = "") -> list[str]:
    """Fields present in the payload that the SDK model does not declare, recursively."""
    found = [f"{prefix}{name}" for name in (model.model_extra or {})]
    for name in type(model).model_fields:
        value = getattr(model, name)
        children = value if isinstance(value, list) else [value]
        for child in children:
            if isinstance(child, BaseModel):
                found += _undeclared(child, f"{prefix}{name}.")
    return found


@pytest.mark.parametrize(("path", "model", "is_list"), ENDPOINTS, ids=[e[0] for e in ENDPOINTS])
def test_fake_payloads_validate_against_the_sdk(
    fake: FakeAdminApi, path: str, model: type[BaseModel], is_list: bool
) -> None:
    objects = _objects(fake, path, is_list)
    assert objects, f"{path} returned nothing to check"
    for raw in objects:
        parsed = model.model_validate(raw)
        unknown = set(_undeclared(parsed)) - _deprecated(model)
        assert unknown == set(), f"{path}: the fake sends fields the SDK does not know: {sorted(unknown)}"
        missing = sorted(set(model.model_fields) - set(raw))
        assert missing == [], f"{path}: the fake omits fields the SDK declares: {missing}"


def test_the_worker_does_not_need_the_deprecated_fields() -> None:
    """The SDK dropped these; if the API ever does too, every row must read the same."""
    from tests.fake_admin_api import API_KEYS, RATE_LIMITS
    from vgi_anthropic import schemas

    def without(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
        return [{k: v for k, v in row.items() if k != field} for row in rows]

    def keys(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        built = [schemas.flatten_api_key(r) for r in rows]
        return schemas.batch_from_rows(built, schemas.API_KEY_SCHEMA).to_pylist()

    assert keys(without(API_KEYS, "workspace_id")) == keys(API_KEYS)
    limits = schemas.flatten_rate_limits
    assert limits(without(RATE_LIMITS, "group_type")) == limits(RATE_LIMITS)
    workspace = schemas.flatten_workspace_rate_limits
    entries = [{**e, "workspace_id": "w", "rate_limit_id": e["id"]} for e in RATE_LIMITS]
    assert workspace(without(entries, "group_type"), "w") == workspace(entries, "w")


def test_every_admin_table_is_covered() -> None:
    """A new admin table must get a conformance check here too."""
    from vgi_anthropic.worker import _ANTHROPIC_CATALOG

    covered = {
        "organization",
        "users",
        "invites",
        "workspaces",
        "workspace_members",
        "api_keys",
        "rate_limits",
        "workspace_rate_limits",
    }
    assert {t.name for t in _ANTHROPIC_CATALOG.schemas[0].tables} == covered
