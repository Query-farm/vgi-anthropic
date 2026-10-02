"""The HTTP layer: headers, both paging styles, retries and error reporting."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from vgi_anthropic import anthropic_api as api
from vgi_anthropic.anthropic_api import AnthropicError, Endpoint
from vgi_anthropic.auth import Credentials

CREDS = Credentials(api_key="sk-ant-admin01-test")


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(api.time, "sleep", slept.append)
    return slept


class TestRequests:
    def test_headers(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"id": "x"})

        api.organization(credentials=CREDS, client=_client(handler))
        assert seen[0].headers["x-api-key"] == "sk-ant-admin01-test"
        assert seen[0].headers["anthropic-version"] == "2023-06-01"
        assert "anthropic-beta" not in seen[0].headers

    def test_betas_are_sent(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"data": []})

        api.page(
            api.MESSAGES_USAGE, None, cursor=None, credentials=CREDS, client=_client(handler), betas=["b1"]
        )
        assert seen[0].headers["anthropic-beta"] == "b1"

    def test_repeated_params_and_booleans(self) -> None:
        seen: list[httpx.URL] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url)
            return httpx.Response(200, json={"data": []})

        api.page(
            api.MESSAGES_USAGE,
            [("group_by[]", "model"), ("group_by[]", "workspace_id"), ("x", True), ("dropped", None)],
            cursor=None,
            credentials=CREDS,
            client=_client(handler),
        )
        params = seen[0].params
        assert params.get_list("group_by[]") == ["model", "workspace_id"]
        assert params["x"] == "true"
        assert "dropped" not in params

    def test_base_url_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VGI_ANTHROPIC_BASE_URL", "http://gateway.test/")
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, json={})

        api.organization(credentials=CREDS, client=_client(handler))
        assert seen == ["http://gateway.test/v1/organizations/me"]

    def test_the_sdk_base_url_is_not_inherited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ANTHROPIC_BASE_URL often points at a proxy; an Admin key must not follow it."""
        monkeypatch.delenv("VGI_ANTHROPIC_BASE_URL", raising=False)
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://proxy.test")
        assert api.base_url() == "https://api.anthropic.com"


class TestPaging:
    def test_id_cursor(self) -> None:
        seen: list[httpx.URL] = []
        pages = {
            None: {"data": [{"id": "a"}, {"id": "b"}], "has_more": True, "last_id": "b"},
            "b": {"data": [{"id": "c"}], "has_more": False, "last_id": "c"},
        }

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url)
            return httpx.Response(200, json=pages[request.url.params.get("after_id")])

        rows = api.collect(api.USERS, credentials=CREDS, client=_client(handler))
        assert [r["id"] for r in rows] == ["a", "b", "c"]
        assert seen[0].params["limit"] == "1000"

    def test_page_token(self) -> None:
        pages = {
            None: {"data": [{"n": 1}], "has_more": True, "next_page": "page_x"},
            "page_x": {"data": [{"n": 2}], "has_more": False, "next_page": None},
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=pages[request.url.params.get("page")])

        endpoint = Endpoint("/v1/organizations/cost_report", "token", None)
        assert api.collect(endpoint, credentials=CREDS, client=_client(handler)) == [{"n": 1}, {"n": 2}]

    def test_no_limit_param_when_the_endpoint_has_none(self) -> None:
        seen: list[httpx.URL] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url)
            return httpx.Response(200, json={"data": [], "next_page": None})

        api.page(api.RATE_LIMITS, None, cursor=None, credentials=CREDS, client=_client(handler))
        assert "limit" not in seen[0].params

    def test_has_more_without_a_cursor_ends_the_walk(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"id": "a"}], "has_more": True, "last_id": None})

        assert api.page(api.USERS, None, cursor=None, credentials=CREDS, client=_client(handler))[1] is None

    def test_a_runaway_walk_raises_instead_of_truncating(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(api, "MAX_PAGES", 3)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"id": "a"}], "has_more": True, "last_id": "a"})

        with pytest.raises(api.AnthropicPageLimitError):
            api.collect(api.USERS, credentials=CREDS, client=_client(handler))


class TestRetries:
    def test_retry_after_is_honoured(self, _no_sleep: list[float]) -> None:
        responses = [
            httpx.Response(429, headers={"retry-after": "2"}, json={}),
            httpx.Response(529, json={}),
            httpx.Response(200, json={"id": "x"}),
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            return responses.pop(0)

        assert api.organization(credentials=CREDS, client=_client(handler)) == {"id": "x"}
        assert _no_sleep == [2.0, 1.0]

    def test_an_absurd_retry_after_is_capped(self, _no_sleep: list[float]) -> None:
        responses = [
            httpx.Response(429, headers={"retry-after": "3600"}, json={}),
            httpx.Response(200, json={}),
        ]
        api.organization(credentials=CREDS, client=_client(lambda r: responses.pop(0)))
        assert _no_sleep == [30.0]

    def test_x_should_retry_false_stops_a_retryable_status(self, _no_sleep: list[float]) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(529, headers={"x-should-retry": "false"}, json={})

        with pytest.raises(AnthropicError):
            api.organization(credentials=CREDS, client=_client(handler))
        assert len(calls) == 1
        assert _no_sleep == []

    def test_x_should_retry_true_retries_an_unlisted_status(self) -> None:
        responses = [
            httpx.Response(409, headers={"x-should-retry": "true"}, json={}),
            httpx.Response(200, json={"id": "x"}),
        ]
        assert api.organization(credentials=CREDS, client=_client(lambda r: responses.pop(0))) == {"id": "x"}

    def test_x_should_retry_on_a_success_is_ignored(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(200, headers={"x-should-retry": "true"}, json={"id": "x"})

        api.organization(credentials=CREDS, client=_client(handler))
        assert len(calls) == 1

    def test_transport_errors_are_retried(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) < 3:
                raise httpx.ConnectError("boom")
            return httpx.Response(200, json={"id": "x"})

        assert api.organization(credentials=CREDS, client=_client(handler)) == {"id": "x"}

    def test_a_client_error_is_not_retried(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(404, json={"error": {"type": "not_found_error", "message": "nope"}})

        with pytest.raises(AnthropicError):
            api.user("user_x", credentials=CREDS, client=_client(handler))
        assert len(calls) == 1


class TestErrors:
    def test_the_api_message_and_request_id_are_surfaced(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                headers={"request-id": "req_123"},
                json={"type": "error", "error": {"type": "invalid_request_error", "message": "bad limit"}},
            )

        with pytest.raises(AnthropicError) as info:
            api.organization(credentials=CREDS, client=_client(handler))
        assert "invalid_request_error: bad limit" in str(info.value)
        assert "req_123" in str(info.value)
        assert info.value.status == 400

    def test_a_403_explains_the_likely_cause(self) -> None:
        with pytest.raises(AnthropicError, match="admin role"):
            api.organization(credentials=CREDS, client=_client(lambda r: httpx.Response(403, json={})))

    def test_a_401_names_every_credential_that_would_work(self) -> None:
        with pytest.raises(AnthropicError) as info:
            api.organization(credentials=CREDS, client=_client(lambda r: httpx.Response(401, json={})))
        message = str(info.value)
        assert "Admin API key" in message and "organization-scoped" in message and "org:admin" in message

    def test_a_non_json_200_is_an_error_not_a_crash(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>gateway</html>", headers={"content-type": "text/html"})

        with pytest.raises(AnthropicError, match="expected JSON"):
            api.organization(credentials=CREDS, client=_client(handler))
