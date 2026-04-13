"""
Tests for planner plan-review data models.
"""

from src.agents.planner_agent.models import (
    PlanReviewIssue,
    PlanReviewIteration,
    PlanReviewSeverity,
    PlanReviewSummary,
)


def test_plan_review_issue_blocking_semantics():
    blocker = PlanReviewIssue(
        issue_id="intro-001",
        section_type="introduction",
        category="coverage",
        severity=PlanReviewSeverity.BLOCKER,
        title="Missing problem framing",
        description="Introduction lacks explicit problem statement.",
        recommendation="Add a dedicated paragraph for problem framing.",
    )
    soft = PlanReviewIssue(
        issue_id="intro-002",
        section_type="introduction",
        category="venue_norm",
        severity=PlanReviewSeverity.SOFT,
        title="Missing contribution summary style",
        description="Contributions are not explicitly summarized at the end.",
        recommendation="Consider adding a concise contribution summary paragraph.",
    )

    assert blocker.is_blocking is True
    assert soft.is_blocking is False


def test_plan_review_summary_roundtrip_and_counters():
    iteration = PlanReviewIteration(
        iteration=1,
        critique="Initial review",
        issues=[
            PlanReviewIssue(
                issue_id="method-001",
                section_type="method",
                category="coherence",
                severity=PlanReviewSeverity.BLOCKER,
                title="Method flow unclear",
                description="Pipeline order is ambiguous.",
                recommendation="Add explicit dataflow paragraph.",
            ),
            PlanReviewIssue(
                issue_id="intro-003",
                section_type="introduction",
                category="venue_norm",
                severity=PlanReviewSeverity.SOFT,
                title="Contributions style signal",
                description="Contribution summary can be clearer.",
                recommendation="Prefer an explicit summary block in prose.",
            ),
        ],
        actions_applied=["Clarified method subsection order"],
        changed=True,
    )
    summary = PlanReviewSummary(
        enabled=True,
        max_iterations=2,
        iterations=[iteration],
        final_status="needs_revision",
        notes="One blocker remains.",
    )

    assert summary.blocking_issue_count == 1
    assert summary.soft_signal_count == 1
    assert summary.has_blocking_issues is True
    assert summary.requires_revision is True

    dumped = summary.model_dump()
    rebuilt = PlanReviewSummary.model_validate(dumped)
    assert rebuilt.blocking_issue_count == 1
    assert rebuilt.soft_signal_count == 1
