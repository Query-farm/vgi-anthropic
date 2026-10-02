"""Structural guard: this worker must never issue a write to the Admin API.

The Admin API is mostly write surface — inviting and removing members,
changing roles, archiving workspaces, disabling keys — and an Admin key
authorizes all of it. The worker is read-only by construction (a single
``_get`` chokepoint), and this test fails the build if that stops being true.
"""

from __future__ import annotations

import ast
from pathlib import Path

import httpx
import pytest

from vgi_anthropic import anthropic_api
from vgi_anthropic.auth import Credentials

PACKAGE = Path(__file__).resolve().parent.parent / "vgi_anthropic"

#: httpx verbs that could mutate state.
WRITE_VERBS = {"post", "put", "patch", "delete", "request", "stream", "send"}

CREDS = Credentials(api_key="sk-ant-admin01-test")


class TestNoWriteVerbs:
    def test_no_http_write_calls_anywhere(self) -> None:
        offenders = [
            f"{path.name}: .{node.func.attr}()"
            for path in PACKAGE.rglob("*.py")
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in WRITE_VERBS
        ]
        assert offenders == [], f"write-shaped HTTP calls found: {offenders}"

    def test_single_http_chokepoint(self) -> None:
        """Only anthropic_api.py may reference httpx at all."""
        users = sorted(path.name for path in PACKAGE.rglob("*.py") if "httpx" in path.read_text())
        assert users == ["anthropic_api.py"], users

    def test_exactly_one_get_call(self) -> None:
        tree = ast.parse((PACKAGE / "anthropic_api.py").read_text())
        gets = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "http"
        ]
        assert len(gets) == 1

    def test_no_messages_endpoint(self) -> None:
        """This worker reports on model usage; it never calls a model."""
        for path in PACKAGE.rglob("*.py"):
            assert '"/v1/messages' not in path.read_text(), path.name


class TestPathTraversal:
    """An ID must not be able to leave its path segment."""

    HOSTILE = "../../invites?injected=1"

    CALLS = [
        ("user", lambda c: anthropic_api.user(TestPathTraversal.HOSTILE, credentials=CREDS, client=c)),
        (
            "workspace",
            lambda c: anthropic_api.workspace(TestPathTraversal.HOSTILE, credentials=CREDS, client=c),
        ),
        ("api_key", lambda c: anthropic_api.api_key(TestPathTraversal.HOSTILE, credentials=CREDS, client=c)),
        (
            "workspace_members",
            lambda c: anthropic_api.page(
                anthropic_api.workspace_members_endpoint(TestPathTraversal.HOSTILE),
                None,
                cursor=None,
                credentials=CREDS,
                client=c,
            ),
        ),
        (
            "workspace_rate_limits",
            lambda c: anthropic_api.page(
                anthropic_api.workspace_rate_limits_endpoint(TestPathTraversal.HOSTILE),
                None,
                cursor=None,
                credentials=CREDS,
                client=c,
            ),
        ),
    ]

    @pytest.mark.parametrize(("name", "call"), CALLS, ids=[c[0] for c in CALLS])
    def test_a_traversing_id_is_encoded(self, name: str, call: object) -> None:
        seen: list[httpx.URL] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url)
            return httpx.Response(200, json={"data": [], "has_more": False})

        call(httpx.Client(transport=httpx.MockTransport(handler)))  # type: ignore[operator]
        (url,) = seen
        raw = url.raw_path.decode()
        assert "%2F" in raw and "%3F" in raw, f"{name} did not encode its segment: {raw}"
        assert "injected" not in url.params, f"{name} let a query string through"

    def test_an_ordinary_id_is_unchanged(self) -> None:
        assert anthropic_api.segment("wrkspc_01JwQvzr7rXLA5AGx3HKfFUJ") == "wrkspc_01JwQvzr7rXLA5AGx3HKfFUJ"
