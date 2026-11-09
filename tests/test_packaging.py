"""Packaging invariants: the entry points run on their own, and the version is single-sourced.

`uv run anthropic_worker.py` resolves dependencies from the script's PEP-723
header, not from `pyproject.toml`, so the two lists can drift silently: the
package imports fine under pytest while the worker a client launches dies at
import. (The sibling Kalshi worker shipped exactly that once.)
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import vgi_anthropic
from vgi_anthropic import worker

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ("anthropic_worker.py", "serve.py")

_BLOCK = re.compile(r"^# /// script$(.+?)^# ///$", re.MULTILINE | re.DOTALL)


def _name(requirement: str) -> str:
    return re.split(r"[\[><=!~;\s]", requirement, maxsplit=1)[0].strip().lower()


def _script_dependencies(path: Path) -> list[str]:
    match = _BLOCK.search(path.read_text())
    assert match is not None, f"{path.name} has no PEP-723 script header"
    body = "".join(line.removeprefix("# ").removeprefix("#") for line in match.group(1).splitlines(True))
    return list(tomllib.loads(body)["dependencies"])


PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_header_covers_every_runtime_dependency(script: str) -> None:
    declared = {_name(spec) for spec in _script_dependencies(ROOT / script)}
    missing = {_name(spec) for spec in PROJECT["dependencies"]} - declared
    assert missing == set(), f"{script} would fail at import: missing {sorted(missing)}"


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_header_carries_the_engine_extra(script: str) -> None:
    """Without an in-process DuckDB engine, every filtered scan fails at runtime."""
    vgi = [s for s in _script_dependencies(ROOT / script) if _name(s) == "vgi-python"]
    assert vgi and all("duckdb" in s for s in vgi), vgi


def test_version_is_single_sourced() -> None:
    assert vgi_anthropic.__version__ == worker.IMPLEMENTATION_VERSION
    assert PROJECT["dynamic"] == ["version"]


def test_changelog_mentions_the_version() -> None:
    assert vgi_anthropic.__version__ in (ROOT / "CHANGELOG.md").read_text()
