"""Admin API credentials, delivered as a DuckDB secret.

Every endpoint this worker reads is credentialed — there is no public surface
to fall back to, unlike a market-data worker — so a missing credential is an
error, not a mode. The credential arrives as a DuckDB secret rather than an
ATTACH option, because an ATTACH option string is visible in
``duckdb_databases()`` and in anything that logs the statement, and an Admin
API key reads every member, key and dollar in the organization::

    CREATE SECRET anthropic (
        TYPE anthropic_admin,
        api_key 'sk-ant-admin01-...'
    );

An organization-scoped personal or service-account key (``sk-ant-api03-...``)
works the same way; a *workspace*-scoped key does not — it can read
``organization`` and is refused (401) everywhere else. An ``org:admin`` OAuth
access token works too, sent as a bearer token::

    CREATE SECRET anthropic (TYPE anthropic_admin, oauth_token 'sk-ant-oat01-...');

Both values are marked redacted, so ``duckdb_secrets()`` shows them masked. The
secret may have any name, or none. When several exist, ``SCOPE`` picks between
them by the API base URL being called — a gateway and the direct API, say.

There is deliberately **no environment-variable fallback**. The same worker can
be served over HTTP (``serve.py``), and a key read from the server's
environment would answer for every client that connects, handing the
organization's admin data to whoever reaches the port.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa
from vgi.catalog.secret_type import SecretTypeSpec

#: The DuckDB secret type this worker registers at ATTACH.
SECRET_TYPE = "anthropic_admin"

#: Secret keys. Exactly one should be set.
API_KEY = "api_key"
OAUTH_TOKEN = "oauth_token"

SECRET_SPEC = SecretTypeSpec(
    name=SECRET_TYPE,
    description=(
        "Anthropic Admin API credentials. Provide api_key (an Admin API key, sk-ant-admin01-..., "
        "or an organization-scoped personal or service-account key) or oauth_token (an OAuth "
        "access token with the org:admin scope). Required: every object in this catalog reads "
        "organization data."
    ),
    schema=pa.schema(
        [
            pa.field(API_KEY, pa.string(), metadata={"redact": "true"}),
            pa.field(OAUTH_TOKEN, pa.string(), metadata={"redact": "true"}),
        ]
    ),
)


class AnthropicAuthError(RuntimeError):
    """No usable Admin API credential resolved for this call."""


@dataclass(slots=True, frozen=True)
class Credentials:
    """One Admin API credential, ready to put on a request."""

    api_key: str = ""
    oauth_token: str = ""

    def headers(self) -> dict[str, str]:
        """The authentication header for one request.

        An Admin API key travels in ``x-api-key``; an OAuth token as a bearer
        token. When both are set the API key wins: it is the documented
        credential for every endpoint this worker reads, while an OAuth token
        is only *required* by endpoints it does not expose.
        """
        if self.api_key:
            return {"x-api-key": self.api_key}
        return {"authorization": f"Bearer {self.oauth_token}"}

    def __repr__(self) -> str:
        # Never let a traceback or a debug log print the key.
        kind = "api_key" if self.api_key else "oauth_token"
        return f"Credentials({kind}=<redacted>)"


def load(api_key: str, oauth_token: str) -> Credentials:
    """Validate raw secret values into :class:`Credentials`.

    The key's *prefix* is deliberately not checked. Besides Admin keys
    (``sk-ant-admin01-``), the Admin API accepts an organization-scoped personal
    or service-account key, which looks like any other ``sk-ant-api03-`` key;
    only a workspace-scoped one is refused. Only the API can tell them apart, and
    its 401 says so plainly — rejecting by prefix here would turn away keys that
    work.

    Raises:
        AnthropicAuthError: Neither value is set.
    """
    api_key = api_key.strip()
    oauth_token = oauth_token.strip()
    if not api_key and not oauth_token:
        raise AnthropicAuthError(
            f"the {SECRET_TYPE!r} secret needs either {API_KEY!r} (an Admin API key) or {OAUTH_TOKEN!r}"
        )
    return Credentials(api_key=api_key, oauth_token=oauth_token)


def _text(raw: Any) -> str:
    """A resolved secret field as text; they arrive as Arrow scalars, tests pass plain values."""
    value = raw.as_py() if hasattr(raw, "as_py") else raw
    return "" if value is None else str(value)


def _select(secrets: Any, scope: str) -> dict[str, Any] | None:
    """The ``anthropic_admin`` secret that applies to ``scope``, from resolved DuckDB secrets.

    The framework keys resolved secrets by each secret's *name* — whatever
    followed ``CREATE SECRET``, or ``__default_anthropic_admin`` for an unnamed
    one — not by its type. Selection therefore goes by the ``type`` field, and
    by ``SCOPE`` when several are present: the longest scope that prefixes the
    API base URL wins, and an unscoped secret is the fallback. (The sibling
    Kalshi worker shipped the name-keyed lookup and silently ignored every
    secret not literally named after its type.)
    """
    selector = getattr(secrets, "for_scope_of_type", None)
    if selector is not None:
        found = selector(scope, SECRET_TYPE)
        if found:
            return dict(found)
    # A plain dict (tests, or an older framework): keyed by type, or carrying a
    # `type` field.
    by_name = secrets.get(SECRET_TYPE)
    if by_name:
        return dict(by_name)
    typed = [v for v in secrets.values() if isinstance(v, dict) and _text(v.get("type")) == SECRET_TYPE]
    return typed[0] if typed else None


def for_call(secrets: Any) -> Credentials:
    """The credential to authenticate this call with.

    Raises:
        AnthropicAuthError: No ``anthropic_admin`` secret resolved, or the one
            that did is unusable. Every endpoint is credentialed, so there is
            nothing useful to do without one.
    """
    # Imported here: anthropic_api imports Credentials from this module.
    from vgi_anthropic.anthropic_api import base_url

    values = _select(secrets, base_url()) if secrets else None
    if not values:
        raise AnthropicAuthError(
            f"no {SECRET_TYPE!r} secret resolved. Create one first: "
            f"CREATE SECRET anthropic (TYPE {SECRET_TYPE}, {API_KEY} 'sk-ant-admin01-...')"
        )
    return load(_text(values.get(API_KEY)), _text(values.get(OAUTH_TOKEN)))
