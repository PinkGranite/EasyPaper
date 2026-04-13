"""
Integration test for prepare_plan plan-review artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.agents.metadata_agent.metadata_agent import MetaDataAgent
from src.agents.metadata_agent.models import PaperMetaData
from src.agents.planner_agent.models import (
    PaperPlan,
    ParagraphPlan,
    PlanReviewIssue,
    PlanReviewIteration,
    PlanReviewSeverity,
    PlanReviewSummary,
    SectionPlan,
)
from src.agents.planner_agent.planner_agent import PlannerAgent
from src.agents.shared.reference_pool import ReferencePool
from src.config.schema import ModelConfig, ToolsConfig


def _minimal_metadata() -> PaperMetaData:
    return PaperMetaData(
        title="Plan Review Integration Test",
        idea_hypothesis="Test whether plan-review artifacts are exported correctly.",
        method="Create a deterministic mocked planning flow.",
        data="No external data required.",
        experiments="Run prepare_plan and inspect outputs.",
        references=[],
    )


@pytest.mark.integration
async def test_prepare_plan_exports_plan_review_summary_and_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata = _minimal_metadata()

    async def _instant_ref_pool(
        cls: type[ReferencePool],
        initial_refs: list,
        paper_search_config: object = None,
    ) -> ReferencePool:
        return cls(initial_refs)

    monkeypatch.setattr(
        ReferencePool,
        "create",
        classmethod(_instant_ref_pool),
    )

    model = ModelConfig(
        model_name="stub-model",
        api_key="stub-key",
        base_url="http://127.0.0.1:9",
    )
    agent = MetaDataAgent(
        model,
        tools_config=ToolsConfig(enabled=False, available_tools=[]),
    )
    agent._planner = PlannerAgent(model)

    review_summary = PlanReviewSummary(
        enabled=True,
        max_iterations=2,
        iterations=[
            PlanReviewIteration(
                iteration=1,
                issues=[
                    PlanReviewIssue(
                        issue_id="intro-soft-001",
                        section_type="introduction",
                        category="venue_norm",
                        severity=PlanReviewSeverity.SOFT,
                        title="Contribution-summary signal",
                        description="Recommend stronger contribution-summary ending.",
                        recommendation="Refine final introduction paragraph key_point.",
                    ),
                ],
                changed=False,
            ),
        ],
        final_status="pass_with_suggestions",
    )

    fake_plan = PaperPlan(
        title=metadata.title,
        sections=[
            SectionPlan(
                section_type="introduction",
                section_title="Introduction",
                paragraphs=[
                    ParagraphPlan(
                        key_point="Motivate the problem and summarize contributions.",
                        approx_sentences=4,
                    ),
                ],
            ),
        ],
        contributions=["Contribution A"],
    )

    captured: dict = {}

    async def _fake_create_plan(_: object, **kwargs: object) -> PaperPlan:
        captured.update(kwargs)
        agent._planner._last_plan_review_summary = review_summary
        return fake_plan

    monkeypatch.setattr(agent._planner, "create_plan", _fake_create_plan)
    monkeypatch.setattr(agent._planner, "discover_seed_references", AsyncMock(return_value=[]))
    monkeypatch.setattr(agent._planner, "discover_references", AsyncMock(return_value={}))
    monkeypatch.setattr(agent._planner, "discover_utility_references", AsyncMock(return_value={}))
    monkeypatch.setattr(agent._planner, "assign_references", lambda **_: None)
    monkeypatch.setattr(agent._planner, "_assign_papers_to_sections", lambda *_args, **_kwargs: {})

    result = await agent.prepare_plan(
        metadata=metadata,
        template_path=None,
        target_pages=None,
        enable_planning=True,
        enable_exemplar=False,
        save_output=True,
        output_dir=str(tmp_path),
    )

    assert result.plan_review is not None
    assert result.plan_review.get("final_status") == "pass_with_suggestions"
    assert result.plan_review_iterations
    assert result.plan_review_iterations[0]["iteration"] == 1
    assert captured.get("review_enabled") is True
    assert captured.get("review_max_iterations") == 2

    review_path = tmp_path / "analysis" / "planning" / "plan_review.json"
    assert review_path.is_file()
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    assert payload["final_status"] == "pass_with_suggestions"
