"""Credential loading and secret selection."""

from __future__ import annotations

from typing import Any

import pytest

from vgi_anthropic import auth
from vgi_anthropic.auth import AnthropicAuthError, Credentials


class TestLoad:
    def test_admin_key_goes_in_x_api_key(self) -> None:
        assert auth.load("sk-ant-admin01-abc", "").headers() == {"x-api-key": "sk-ant-admin01-abc"}

    def test_oauth_token_is_a_bearer(self) -> None:
        assert auth.load("", "tok").headers() == {"authorization": "Bearer tok"}

    def test_api_key_wins_when_both_are_set(self) -> None:
        assert "x-api-key" in auth.load("sk-ant-admin01-abc", "tok").headers()

    def test_nothing_set_is_an_error(self) -> None:
        with pytest.raises(AnthropicAuthError, match="needs either"):
            auth.load("  ", "")

    def test_a_standard_prefix_is_not_rejected(self) -> None:
        """Organization-scoped personal keys are `sk-ant-api03-` and work; only the API can tell."""
        assert auth.load("sk-ant-api03-xyz", "").headers() == {"x-api-key": "sk-ant-api03-xyz"}

    def test_whitespace_is_trimmed(self) -> None:
        """A key pasted with a trailing newline would otherwise be a mysterious 401."""
        assert auth.load("sk-ant-admin01-abc\n", "").api_key == "sk-ant-admin01-abc"

    def test_repr_never_shows_the_key(self) -> None:
        creds = Credentials(api_key="sk-ant-admin01-supersecret")
        assert "supersecret" not in repr(creds)
        assert "supersecret" not in str(creds)


class _Resolved(dict[str, Any]):
    """Mimics the framework's resolved-secrets mapping, which selects by type and scope."""

    def for_scope_of_type(self, scope: str, secret_type: str) -> dict[str, Any] | None:
        matches = [
            v for v in self.values() if v.get("type") == secret_type and scope.startswith(v.get("scope", ""))
        ]
        matches.sort(key=lambda v: len(v.get("scope", "")), reverse=True)
        return matches[0] if matches else None


class TestForCall:
    def test_no_secret_names_the_fix(self) -> None:
        with pytest.raises(AnthropicAuthError, match="CREATE SECRET"):
            auth.for_call(None)

    def test_a_secret_is_found_whatever_its_name(self) -> None:
        secrets = _Resolved(my_key={"type": "anthropic_admin", "api_key": "sk-ant-admin01-a"})
        assert auth.for_call(secrets).api_key == "sk-ant-admin01-a"

    def test_scope_picks_between_organizations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VGI_ANTHROPIC_BASE_URL", "https://gateway.example/anthropic")
        secrets = _Resolved(
            default={"type": "anthropic_admin", "api_key": "sk-ant-admin01-default"},
            gateway={
                "type": "anthropic_admin",
                "api_key": "sk-ant-admin01-gateway",
                "scope": "https://gateway.example",
            },
        )
        assert auth.for_call(secrets).api_key == "sk-ant-admin01-gateway"

    def test_a_plain_dict_with_a_type_field_works(self) -> None:
        secrets = {"x": {"type": "anthropic_admin", "api_key": "sk-ant-admin01-b"}}
        assert auth.for_call(secrets).api_key == "sk-ant-admin01-b"

    def test_secrets_of_another_type_are_ignored(self) -> None:
        with pytest.raises(AnthropicAuthError):
            auth.for_call({"s3": {"type": "s3", "key_id": "x"}})


def test_the_secret_values_are_redacted() -> None:
    redacted = {f.name for f in auth.SECRET_SPEC.schema if (f.metadata or {}).get(b"redact") == b"true"}
    assert redacted == {"api_key", "oauth_token"}
