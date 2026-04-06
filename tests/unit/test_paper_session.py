"""
Tests for PaperSession and SectionState.

Phase 1a: SectionState — summary, token estimation, budget-aware content.
Phase 1b: PaperSession — unified state, cross-section context, serialization.
"""
import pytest

from tests.conftest import SAMPLE_INTRO_LATEX, SAMPLE_METHOD_LATEX


# =========================================================================
# Phase 1a: SectionState
# =========================================================================


class TestSectionState:
    """Tests for SectionState data model."""

    def _make(self, **kwargs):
        from src.agents.shared.paper_session import SectionState
        defaults = {"section_type": "introduction", "status": "draft"}
        defaults.update(kwargs)
        return SectionState(**defaults)

    def test_summary_contains_word_count(self):
        state = self._make(content=SAMPLE_INTRO_LATEX)
        summary = state.summary()

        word_count = len(SAMPLE_INTRO_LATEX.split())
        assert str(word_count) in summary

    def test_summary_contains_section_preview(self):
        state = self._make(content=SAMPLE_INTRO_LATEX)
        summary = state.summary()

        assert len(summary) > 0
        assert len(summary) <= 400

    def test_summary_respects_max_chars(self):
        state = self._make(content=SAMPLE_INTRO_LATEX)
        summary = state.summary(max_chars=100)

        assert len(summary) <= 150  # allow small overflow for word boundary

    def test_summary_empty_content(self):
        state = self._make(content="")
        summary = state.summary()

        assert summary == ""

    def test_estimate_tokens(self):
        state = self._make(content=SAMPLE_INTRO_LATEX)
        tokens = state.estimate_tokens()

        assert isinstance(tokens, int)
        assert tokens > 0
        char_count = len(SAMPLE_INTRO_LATEX)
        assert tokens == (char_count // 4) + 1

    def test_full_or_summary_within_budget(self):
        state = self._make(content=SAMPLE_INTRO_LATEX)
        large_budget = 10000
        result = state.full_or_summary(token_budget=large_budget)

        assert result == SAMPLE_INTRO_LATEX

    def test_full_or_summary_exceeds_budget(self):
        state = self._make(content=SAMPLE_INTRO_LATEX)
        tiny_budget = 10
        result = state.full_or_summary(token_budget=tiny_budget)

        assert result != SAMPLE_INTRO_LATEX
        assert len(result) < len(SAMPLE_INTRO_LATEX)

    def test_word_count_auto_computed(self):
        state = self._make(content=SAMPLE_INTRO_LATEX)

        assert state.word_count == len(SAMPLE_INTRO_LATEX.split())


# =========================================================================
# Phase 1b: PaperSession
# =========================================================================


class TestPaperSession:
    """Tests for PaperSession unified state container."""

    def _make(self):
        from src.agents.shared.paper_session import PaperSession
        return PaperSession()

    def test_update_and_get_section(self):
        session = self._make()
        session.update_section("introduction", SAMPLE_INTRO_LATEX)

        state = session.get_section_state("introduction")
        assert state is not None
        assert state.content == SAMPLE_INTRO_LATEX
        assert state.status == "draft"
        assert state.word_count == len(SAMPLE_INTRO_LATEX.split())

    def test_get_nonexistent_section_returns_none(self):
        session = self._make()

        assert session.get_section_state("nonexistent") is None

    def test_get_context_for_section_includes_intro(self):
        session = self._make()
        session.update_section("introduction", SAMPLE_INTRO_LATEX)

        context = session.get_context_for_section(
            "method", model_context_window=8000
        )

        assert "introduction" in context.lower()

    def test_get_context_for_section_excludes_self(self):
        session = self._make()
        session.update_section("method", SAMPLE_METHOD_LATEX)

        context = session.get_context_for_section(
            "method", model_context_window=8000
        )

        # Should not include method's own full content as a "prior section"
        assert SAMPLE_METHOD_LATEX not in context

    def test_get_context_includes_contributions(self):
        session = self._make()
        session.contributions = ["Novel causal ID strategy"]

        context = session.get_context_for_section(
            "method", model_context_window=8000
        )

        assert "novel causal" in context.lower() or "causal" in context.lower()

    def test_dynamic_budget_allocation(self):
        session = self._make()
        session.update_section("introduction", SAMPLE_INTRO_LATEX)

        small_ctx = session.get_context_for_section(
            "method", model_context_window=1000
        )
        large_ctx = session.get_context_for_section(
            "method", model_context_window=128000
        )

        assert len(small_ctx) <= len(large_ctx)

    def test_serialization_roundtrip(self):
        session = self._make()
        session.update_section("introduction", SAMPLE_INTRO_LATEX)
        session.update_section("method", SAMPLE_METHOD_LATEX)
        session.contributions = ["Contribution A", "Contribution B"]

        data = session.to_dict()
        restored = session.__class__.from_dict(data)

        assert restored.get_section_state("introduction").content == SAMPLE_INTRO_LATEX
        assert restored.get_section_state("method").content == SAMPLE_METHOD_LATEX
        assert restored.contributions == ["Contribution A", "Contribution B"]

    def test_as_memory_returns_compatible_object(self):
        session = self._make()
        session.update_section("introduction", SAMPLE_INTRO_LATEX)

        mem = session.as_memory()

        assert mem.get_section("introduction") == SAMPLE_INTRO_LATEX
        assert hasattr(mem, "log")
        assert hasattr(mem, "update_section")
        assert hasattr(mem, "get_writing_context")

    def test_as_memory_stays_in_sync(self):
        session = self._make()
        mem = session.as_memory()

        session.update_section("introduction", SAMPLE_INTRO_LATEX)

        assert mem.get_section("introduction") == SAMPLE_INTRO_LATEX

    def test_generated_sections_property(self):
        session = self._make()
        session.update_section("introduction", SAMPLE_INTRO_LATEX)
        session.update_section("method", SAMPLE_METHOD_LATEX)

        gs = session.generated_sections

        assert gs["introduction"] == SAMPLE_INTRO_LATEX
        assert gs["method"] == SAMPLE_METHOD_LATEX
