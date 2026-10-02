"""Shared fixtures for the test suite."""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest

from vgi_anthropic import anthropic_api


@pytest.fixture(autouse=True)
def _hermetic_connection_pool() -> Generator[None]:
    """Give every test a clean connection pool.

    `anthropic_api` keeps a process-wide client so a paged scan does not pay a
    TLS handshake per page. That is global state: without resetting it, a client
    built against one test's mock transport would serve the next.
    """
    anthropic_api.reset_shared_client()
    yield
    anthropic_api.reset_shared_client()


@pytest.fixture(autouse=True)
def _no_real_base_url(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Offline tests must never reach api.anthropic.com, even by accident.

    Pointing the base URL at a closed local port turns a forgotten mock into a
    fast connection error rather than a real request carrying a real key.
    Live tests opt out by their marker.
    """
    if request.node.get_closest_marker("live") is None and anthropic_api.BASE_URL_ENV not in os.environ:
        monkeypatch.setenv(anthropic_api.BASE_URL_ENV, "http://127.0.0.1:9")
