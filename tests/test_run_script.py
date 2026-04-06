"""
Tests for examples/run.py result handling against PaperGenerationResult schema.
"""
import json
from pathlib import Path

import pytest

from src.agents.metadata_agent.models import (
    PaperGenerationResult,
    SectionResult,
    PaperMetaData,
    FigureSpec,
    CodeRepositorySpec,
)


class TestPaperGenerationResultSchema:
    """Verify PaperGenerationResult has the fields run.py depends on."""

    def _make_result(self, **overrides) -> PaperGenerationResult:
        defaults = {
            "status": "ok",
            "paper_title": "Test",
            "sections": [],
            "latex_content": "",
            "output_path": "/tmp/out",
            "total_word_count": 100,
            "errors": [],
        }
        defaults.update(overrides)
        return PaperGenerationResult(**defaults)

    def test_status_field_exists(self):
        r = self._make_result(status="ok")
        assert r.status == "ok"

    def test_status_error(self):
        r = self._make_result(status="error", errors=["something broke"])
        assert r.status == "error"
        assert len(r.errors) == 1

    def test_output_path_field(self):
        r = self._make_result(output_path="/tmp/test_output")
        assert r.output_path == "/tmp/test_output"

    def test_output_path_none(self):
        r = self._make_result(output_path=None)
        assert r.output_path is None

    def test_no_success_attribute(self):
        r = self._make_result()
        assert not hasattr(r, "success"), (
            "PaperGenerationResult should use 'status', not 'success'"
        )

    def test_no_output_dir_attribute(self):
        r = self._make_result()
        assert not hasattr(r, "output_dir"), (
            "PaperGenerationResult should use 'output_path', not 'output_dir'"
        )

    def test_no_singular_error_attribute(self):
        r = self._make_result()
        assert not hasattr(r, "error"), (
            "PaperGenerationResult should use 'errors' (list), not 'error'"
        )


class TestMetaJsonParsing:
    """Verify meta.json can be parsed into PaperMetaData."""

    def test_parse_example_meta(self):
        meta_path = Path(__file__).resolve().parent.parent / "examples" / "meta.json"
        if not meta_path.exists():
            pytest.skip("examples/meta.json not found")

        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        figures = [FigureSpec(**fig) for fig in raw.get("figures", [])]
        code_repo = (
            CodeRepositorySpec(**raw["code_repository"])
            if raw.get("code_repository")
            else None
        )

        metadata = PaperMetaData(
            title=raw["title"],
            idea_hypothesis=raw["idea_hypothesis"],
            method=raw["method"],
            data=raw["data"],
            experiments=raw["experiments"],
            references=raw.get("references", []),
            template_path=raw.get("template_path"),
            style_guide=raw.get("style_guide"),
            target_pages=raw.get("target_pages", 8),
            figures=figures,
            tables=raw.get("tables", []),
            code_repository=code_repo,
            export_prompt_traces=raw.get("export_prompt_traces", False),
        )
        assert metadata.title == raw["title"]
        assert len(metadata.figures) == len(raw["figures"])
