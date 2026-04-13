"""
Tests for plan-review soft signals.
"""

from src.agents.planner_agent.models import (
    PlanReviewIssue,
    PlanReviewIteration,
    PlanReviewSeverity,
    PlanReviewSummary,
)


def test_soft_signal_does_not_trigger_hard_failure():
    soft_only_iteration = PlanReviewIteration(
        iteration=1,
        issues=[
            PlanReviewIssue(
                issue_id="intro-soft-001",
                section_type="introduction",
                category="venue_norm",
                severity=PlanReviewSeverity.SOFT,
                title="Contribution-summary style signal",
                description="A contribution-summary ending is recommended.",
                recommendation="Refine final paragraph to summarize core contributions.",
            ),
        ],
        changed=False,
    )
    summary = PlanReviewSummary(
        enabled=True,
        max_iterations=2,
        iterations=[soft_only_iteration],
        final_status="pass_with_suggestions",
    )

    assert summary.soft_signal_count == 1
    assert summary.blocking_issue_count == 0
    assert summary.has_blocking_issues is False
    assert summary.requires_revision is False
