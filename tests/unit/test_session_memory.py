"""
Tests for SessionMemory synchronization and cross-section context.

Phase 0: Verifies that SessionMemory correctly provides cross-section
context when sections are populated via update_section().
"""
import pytest

from src.agents.shared.session_memory import SessionMemory


class TestGetWritingContext:
    """Verify get_writing_context returns prior section summaries."""

    def test_includes_prior_sections(self, sample_intro_latex):
        memory = SessionMemory()
        memory.update_section("introduction", sample_intro_latex)

        context = memory.get_writing_context("method")

        assert "introduction" in context.lower()
        assert "words" in context.lower() or "word" in context.lower()

    def test_excludes_current_section(self, sample_intro_latex):
        memory = SessionMemory()
        memory.update_section("introduction", sample_intro_latex)

        context = memory.get_writing_context("introduction")

        assert "already-written" not in context.lower() or "introduction" not in context.lower().split("already-written")[1]

    def test_includes_multiple_prior_sections(
        self, sample_intro_latex, sample_method_latex
    ):
        memory = SessionMemory()
        memory.update_section("introduction", sample_intro_latex)
        memory.update_section("method", sample_method_latex)

        context = memory.get_writing_context("experiment")

        assert "introduction" in context.lower()
        assert "method" in context.lower()

    def test_empty_memory_returns_empty_or_minimal(self):
        memory = SessionMemory()

        context = memory.get_writing_context("introduction")

        assert isinstance(context, str)

    def test_includes_contributions(self, sample_intro_latex):
        memory = SessionMemory()
        memory.update_section("introduction", sample_intro_latex)
        memory.contributions = ["Novel causal identification strategy"]

        context = memory.get_writing_context("method")

        assert "novel causal" in context.lower() or "identification" in context.lower()


class TestBuildSectionSummaries:
    """Verify _build_section_summaries produces correct format."""

    def test_with_populated_sections(self, sample_intro_latex):
        memory = SessionMemory()
        memory.update_section("introduction", sample_intro_latex)

        summaries = memory._build_section_summaries()

        assert "introduction" in summaries.lower()
        word_count = len(sample_intro_latex.split())
        assert str(word_count) in summaries

    def test_excludes_specified_section(self, sample_intro_latex):
        memory = SessionMemory()
        memory.update_section("introduction", sample_intro_latex)

        summaries = memory._build_section_summaries(exclude="introduction")

        assert summaries == ""

    def test_excludes_only_specified(
        self, sample_intro_latex, sample_method_latex
    ):
        memory = SessionMemory()
        memory.update_section("introduction", sample_intro_latex)
        memory.update_section("method", sample_method_latex)

        summaries = memory._build_section_summaries(exclude="introduction")

        assert "introduction" not in summaries.lower()
        assert "method" in summaries.lower()

    def test_empty_sections_returns_empty(self):
        memory = SessionMemory()

        summaries = memory._build_section_summaries()

        assert summaries == ""


class TestGetCrossSectionSummary:
    """Verify get_cross_section_summary aggregates all sections."""

    def test_returns_all_sections(
        self, sample_intro_latex, sample_method_latex
    ):
        memory = SessionMemory()
        memory.update_section("introduction", sample_intro_latex)
        memory.update_section("method", sample_method_latex)

        summary = memory.get_cross_section_summary()

        assert "introduction" in summary.lower()
        assert "method" in summary.lower()


class TestUpdateSection:
    """Verify update_section stores content correctly."""

    def test_stores_content(self, sample_intro_latex):
        memory = SessionMemory()
        memory.update_section("introduction", sample_intro_latex)

        assert memory.get_section("introduction") == sample_intro_latex

    def test_overwrites_existing(self):
        memory = SessionMemory()
        memory.update_section("introduction", "version 1")
        memory.update_section("introduction", "version 2")

        assert memory.get_section("introduction") == "version 2"

    def test_get_nonexistent_returns_none(self):
        memory = SessionMemory()

        assert memory.get_section("nonexistent") is None
