"""
Integration test: PaperSession ↔ execute_generation context flow.

Phase 1c: Verifies that PaperSession provides correct cross-section context
during a simulated multi-section generation workflow.
"""
import pytest

from tests.conftest import SAMPLE_INTRO_LATEX, SAMPLE_METHOD_LATEX


class TestPaperSessionWorkflow:
    """Simulate a multi-section paper generation with PaperSession."""

    def test_body_section_receives_intro_context(self):
        from src.agents.shared.paper_session import PaperSession

        session = PaperSession()

        # Phase 1: generate introduction
        session.update_section("introduction", SAMPLE_INTRO_LATEX)

        # Phase 2: preparing to generate method — should see introduction
        context = session.get_context_for_section(
            "method", model_context_window=8000
        )
        assert "introduction" in context.lower()
        assert len(context) > 50

    def test_third_section_receives_two_prior(self):
        from src.agents.shared.paper_session import PaperSession

        session = PaperSession()
        session.update_section("introduction", SAMPLE_INTRO_LATEX)
        session.update_section("method", SAMPLE_METHOD_LATEX)

        context = session.get_context_for_section(
            "experiment", model_context_window=8000
        )
        assert "introduction" in context.lower()
        assert "method" in context.lower()

    def test_memory_compat_during_workflow(self):
        from src.agents.shared.paper_session import PaperSession

        session = PaperSession()
        memory = session.as_memory()

        session.update_section("introduction", SAMPLE_INTRO_LATEX)

        # Memory should see the section
        assert memory.get_section("introduction") == SAMPLE_INTRO_LATEX

        # Memory's get_writing_context should work
        ctx = memory.get_writing_context("method")
        assert "introduction" in ctx.lower()

    def test_contributions_flow_through(self):
        from src.agents.shared.paper_session import PaperSession

        session = PaperSession()
        session.contributions = ["Novel causal ID strategy"]
        session.update_section("introduction", SAMPLE_INTRO_LATEX)

        context = session.get_context_for_section(
            "method", model_context_window=8000
        )
        assert "causal" in context.lower()

    def test_generated_sections_dict_stays_in_sync(self):
        from src.agents.shared.paper_session import PaperSession

        session = PaperSession()
        session.update_section("introduction", SAMPLE_INTRO_LATEX)
        session.update_section("method", SAMPLE_METHOD_LATEX)

        gs = session.generated_sections
        assert gs["introduction"] == SAMPLE_INTRO_LATEX
        assert gs["method"] == SAMPLE_METHOD_LATEX

        # Update via session
        session.update_section("introduction", "updated")
        assert session.generated_sections["introduction"] == "updated"
