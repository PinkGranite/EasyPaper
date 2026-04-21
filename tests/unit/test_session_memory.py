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


class TestLocalReviewEvents:
    def test_add_local_review_event_and_summary(self):
        memory = SessionMemory()
        memory.add_local_review_event(
            section_type="method",
            target_id="method.p0",
            level="paragraph",
            disposition="fixed_locally",
            issue_type="missing_required_figure_ref",
            message="Inserted a required figure reference.",
            paragraph_index=0,
        )

        assert len(memory.local_review_events) == 1
        summary = memory.get_recent_local_review_summary()
        assert "fixed_locally" in summary
        assert "method.p0" in summary

    def test_consume_local_writer_responses_clears_pending_receipts(self):
        memory = SessionMemory()
        memory.add_local_review_event(
            section_type="result",
            target_id="result.p1",
            level="paragraph",
            disposition="retry_required",
            issue_type="missing_key_point",
            message="Retry with the paragraph key point included.",
            paragraph_index=1,
        )

        consumed = memory.consume_local_writer_responses()
        assert len(consumed["writer_response_paragraph"]) == 1
        assert consumed["writer_response_paragraph"][0]["disposition"] == "retry_required"

        consumed_again = memory.consume_local_writer_responses()
        assert consumed_again["writer_response_paragraph"] == []


class TestIssueLifecycleRespectsDisposition:
    def test_escalated_local_response_does_not_resolve_issue(self):
        memory = SessionMemory()
        issue = {
            "target_id": "method.p0",
            "section_type": "method",
            "issue_type": "missing_required_figure_ref",
            "checker": "local_mini_review",
            "message": "Figure reference missing",
            "severity": "warning",
        }
        memory.update_issue_lifecycle(iteration=1, hierarchical_feedbacks=[issue])

        result = memory.update_issue_lifecycle(
            iteration=2,
            hierarchical_feedbacks=[],
            writer_response_paragraph=[
                {
                    "target_id": "method.p0",
                    "section_type": "method",
                    "disposition": "escalate",
                }
            ],
        )

        unresolved = result["regression_report"]["unresolved_issues"]
        assert unresolved == 1

    def test_fixed_local_response_can_resolve_issue(self):
        memory = SessionMemory()
        issue = {
            "target_id": "method.p0",
            "section_type": "method",
            "issue_type": "missing_required_figure_ref",
            "checker": "local_mini_review",
            "message": "Figure reference missing",
            "severity": "warning",
        }
        memory.update_issue_lifecycle(iteration=1, hierarchical_feedbacks=[issue])

        result = memory.update_issue_lifecycle(
            iteration=2,
            hierarchical_feedbacks=[],
            writer_response_paragraph=[
                {
                    "target_id": "method.p0",
                    "section_type": "method",
                    "disposition": "fixed_locally",
                }
            ],
        )

        assert result["regression_report"]["resolved_issues"] == 1
