"""
Tests for planner plan-review optimization loop.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.planner_agent.models import (
    PaperPlan,
    ParagraphPlan,
    PlanReviewIssue,
    PlanReviewIteration,
    PlanReviewSeverity,
    SectionPlan,
)
from src.agents.planner_agent.planner_agent import PlannerAgent


def _planner_stub() -> PlannerAgent:
    planner = PlannerAgent.__new__(PlannerAgent)
    planner.model_name = "test-model"
    planner.config = MagicMock()
    planner.client = MagicMock()
    planner._last_plan = None
    planner.vlm_service = None
    planner._last_plan_review_summary = None
    return planner


def _intro_plan(key_point: str = "Introduce problem context.") -> PaperPlan:
    return PaperPlan(
        title="Plan Review Test",
        sections=[
            SectionPlan(
                section_type="introduction",
                section_title="Introduction",
                paragraphs=[ParagraphPlan(key_point=key_point, approx_sentences=4)],
            ),
        ],
        contributions=["Contribution A", "Contribution B"],
    )


@pytest.mark.asyncio
async def test_review_loop_iterates_until_blocker_resolved():
    planner = _planner_stub()
    original_plan = _intro_plan()
    optimized_plan = _intro_plan(
        key_point="Introduce problem context and summarize contributions.",
    )

    planner._criticize_plan = AsyncMock(
        side_effect=[
            PlanReviewIteration(
                iteration=1,
                issues=[
                    PlanReviewIssue(
                        issue_id="intro-blocker-1",
                        section_type="introduction",
                        category="coverage",
                        severity=PlanReviewSeverity.BLOCKER,
                        title="Contribution summary missing",
                        description="No contribution summary in introduction.",
                        recommendation="Add contribution-summary intent in final paragraph.",
                    ),
                ],
                changed=False,
            ),
            PlanReviewIteration(iteration=2, issues=[], changed=False),
        ],
    )
    planner._optimize_plan = AsyncMock(return_value=optimized_plan)

    final_plan, summary = await planner._review_and_refine_plan(
        plan=original_plan,
        max_iterations=2,
        enabled=True,
    )

    assert planner._criticize_plan.await_count == 2
    planner._optimize_plan.assert_awaited_once()
    assert final_plan.sections[0].paragraphs[0].key_point.endswith("contributions.")
    assert summary.has_blocking_issues is False
    assert summary.final_status == "passed"


@pytest.mark.asyncio
async def test_review_loop_stops_early_when_no_blocker():
    planner = _planner_stub()
    plan = _intro_plan()
    planner._criticize_plan = AsyncMock(
        return_value=PlanReviewIteration(iteration=1, issues=[], changed=False),
    )
    planner._optimize_plan = AsyncMock()

    final_plan, summary = await planner._review_and_refine_plan(
        plan=plan,
        max_iterations=3,
        enabled=True,
    )

    assert planner._criticize_plan.await_count == 1
    planner._optimize_plan.assert_not_called()
    assert final_plan is plan
    assert summary.final_status == "passed"


@pytest.mark.asyncio
async def test_soft_signal_only_does_not_trigger_optimizer():
    planner = _planner_stub()
    plan = _intro_plan()
    planner._criticize_plan = AsyncMock(
        return_value=PlanReviewIteration(
            iteration=1,
            issues=[
                PlanReviewIssue(
                    issue_id="intro-soft-1",
                    section_type="introduction",
                    category="venue_norm",
                    severity=PlanReviewSeverity.SOFT,
                    title="Contribution-summary style recommendation",
                    description="A concise contribution summary is preferred.",
                    recommendation="Consider refining final intro paragraph.",
                ),
            ],
            changed=False,
        ),
    )
    planner._optimize_plan = AsyncMock()

    _, summary = await planner._review_and_refine_plan(
        plan=plan,
        max_iterations=2,
        enabled=True,
    )

    planner._optimize_plan.assert_not_called()
    assert summary.soft_signal_count == 1
    assert summary.requires_revision is False
    assert summary.final_status == "pass_with_suggestions"


@pytest.mark.asyncio
async def test_critic_runtime_failure_does_not_report_passed():
    planner = _planner_stub()
    plan = _intro_plan(
        key_point="Introduce problem context and summarize contributions.",
    )
    planner._llm_json_call = AsyncMock(side_effect=RuntimeError("critic timeout"))
    planner._optimize_plan = AsyncMock(return_value=plan)

    _, summary = await planner._review_and_refine_plan(
        plan=plan,
        max_iterations=1,
        enabled=True,
    )

    assert summary.final_status != "passed"
    assert summary.has_blocking_issues is True
    assert summary.iterations
    assert any(i.issue_id.endswith("critic-failure") for i in summary.iterations[0].issues)
