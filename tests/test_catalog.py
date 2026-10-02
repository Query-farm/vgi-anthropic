"""The catalog shape the DuckDB extension will see on ATTACH."""

from __future__ import annotations

import json
import re
from pathlib import Path

from vgi_anthropic.admin import ADMIN_FUNCTIONS, AllOrganizationFunction
from vgi_anthropic.lookups import LOOKUP_FUNCTIONS
from vgi_anthropic.paging import WalkState
from vgi_anthropic.reports import REPORT_FUNCTIONS
from vgi_anthropic.worker import _ANTHROPIC_CATALOG, _FUNCTIONS

SCHEMA = _ANTHROPIC_CATALOG.schemas[0]
ROOT = Path(__file__).resolve().parent.parent

#: Paged endpoints are stateful scans: one page per tick, so a LIMIT stops early.
PAGED_SCANS = [f for f in [*ADMIN_FUNCTIONS, *REPORT_FUNCTIONS] if f is not AllOrganizationFunction]


class TestRegistration:
    def test_lookups_are_blended(self) -> None:
        for func in LOOKUP_FUNCTIONS:
            assert func.get_metadata().input_from_args is True, func.Meta.name
            assert func.has_finalize_override() is False, func.Meta.name

    def test_paged_scans_are_not_blended_and_carry_state(self) -> None:
        for func in PAGED_SCANS:
            assert func.get_metadata().input_from_args is False, func.Meta.name
            assert isinstance(func.initial_state(None), WalkState), func.Meta.name

    def test_every_function_requires_the_secret(self) -> None:
        for func in _FUNCTIONS:
            types = [entry.secret_type for entry in func.Meta.required_secrets]
            assert types == ["anthropic_admin"], func.Meta.name

    def test_names_are_unique(self) -> None:
        names = [f.Meta.name for f in _FUNCTIONS] + [t.name for t in SCHEMA.tables]
        assert len(names) == len(set(names))

    def test_filter_pushdown_implies_projection_pushdown(self) -> None:
        """Pushed filters are evaluated against the projection; without it they hit the wrong column."""
        for func in _FUNCTIONS:
            if getattr(func.Meta, "filter_pushdown", False):
                assert func.Meta.projection_pushdown, func.Meta.name
                assert func.Meta.auto_apply_filters, func.Meta.name


class TestDocumentation:
    def test_every_function_is_documented(self) -> None:
        for func in _FUNCTIONS:
            tags = func.Meta.tags
            assert tags["vgi.doc_llm"] and tags["vgi.doc_md"], func.Meta.name
            assert json.loads(tags["vgi.example_queries"]), func.Meta.name
            assert func.Meta.examples, func.Meta.name
            columns = json.loads(tags["vgi.result_columns_schema"])
            assert all(c["description"] for c in columns), func.Meta.name

    def test_examples_use_qualified_names(self) -> None:
        for func in _FUNCTIONS:
            for example in json.loads(func.Meta.tags["vgi.example_queries"]):
                assert "anthropic.main." in example["sql"], (func.Meta.name, example["sql"])

    def test_every_table_is_documented_and_keyed(self) -> None:
        for table in SCHEMA.tables:
            assert table.comment and table.tags["vgi.doc_llm"], table.name
            assert table.primary_key, table.name
            assert table.function.Meta.name == f"all_{table.name}", table.name

    def test_catalog_level_tags_parse(self) -> None:
        for key in ("vgi.executable_examples", "vgi.agent_test_tasks", "vgi.keywords"):
            assert json.loads(_ANTHROPIC_CATALOG.tags[key])
        categories = {c["name"] for c in json.loads(SCHEMA.tags["vgi.categories"])}
        used = {func.Meta.tags["vgi.category"] for func in _FUNCTIONS}
        assert used <= categories


class TestAgentTasks:
    """The public prompts (a catalog tag) and private graders (a sidecar file) must agree."""

    def _private_names(self) -> list[str]:
        text = (ROOT / "vgi-agent-tests.yaml").read_text()
        return re.findall(r"^  - name: (\S+)$", text, flags=re.MULTILINE)

    def test_public_tasks_expose_no_graders(self) -> None:
        for task in json.loads(_ANTHROPIC_CATALOG.tags["vgi.agent_test_tasks"]):
            assert set(task) == {"name", "prompt"}, task

    def test_every_public_task_has_a_grader_and_vice_versa(self) -> None:
        public = [t["name"] for t in json.loads(_ANTHROPIC_CATALOG.tags["vgi.agent_test_tasks"])]
        assert public == self._private_names()

    def test_every_object_is_exercised_by_a_grader(self) -> None:
        text = (ROOT / "vgi-agent-tests.yaml").read_text()
        objects = [f.Meta.name for f in _FUNCTIONS] + [t.name for t in SCHEMA.tables]
        untested = [name for name in objects if f"anthropic.main.{name}" not in text]
        assert untested == []
