"""Tests for TemplateAnalyzer, PreambleParser, and TemplateWriterGuide."""
import pytest

from src.agents.shared.template_analyzer import TemplateWriterGuide


# ── Task 1: TemplateWriterGuide model ────────────────────────────────────


class TestTemplateWriterGuide:
    def test_default_construction(self):
        guide = TemplateWriterGuide()
        assert guide.available_packages == []
        assert guide.document_class == "article"
        assert guide.column_format == "single"
        assert guide.citation_style == "cite"
        assert guide.custom_environments == []
        assert guide.custom_commands == []
        assert guide.figure_guidance == ""
        assert guide.table_guidance == ""
        assert guide.algorithm_guidance == ""
        assert guide.math_guidance == ""
        assert guide.general_constraints == ""

    def test_construction_with_packages(self):
        guide = TemplateWriterGuide(
            available_packages=["amsmath", "graphicx", "booktabs"],
            document_class="article",
            column_format="double",
        )
        assert "booktabs" in guide.available_packages
        assert guide.column_format == "double"

    def test_has_package(self):
        guide = TemplateWriterGuide(
            available_packages=["amsmath", "natbib", "booktabs"]
        )
        assert guide.has_package("amsmath") is True
        assert guide.has_package("algorithm2e") is False

    def test_format_for_prompt_non_empty(self):
        guide = TemplateWriterGuide(
            available_packages=["amsmath", "booktabs"],
            column_format="double",
            figure_guidance="Use figure for single-column.",
            table_guidance="Use booktabs commands.",
        )
        prompt_block = guide.format_for_prompt()
        assert "## Template Constraints" in prompt_block
        assert "amsmath" in prompt_block
        assert "booktabs" in prompt_block
        assert "double" in prompt_block
        assert "figure" in prompt_block.lower()

    def test_format_for_prompt_empty_guide_returns_empty(self):
        guide = TemplateWriterGuide()
        assert guide.format_for_prompt() == ""
