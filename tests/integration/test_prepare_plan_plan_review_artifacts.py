"""
Integration test for prepare_plan plan-review artifacts.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.agents.metadata_agent.metadata_agent import MetaDataAgent
from src.agents.metadata_agent.models import FigureSpec, PaperMetaData
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
from src.models.evidence_graph import EvidenceDAG


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
    monkeypatch.setattr(
        "src.agents.metadata_agent.metadata_agent.DAGBuilder.build",
        AsyncMock(return_value=EvidenceDAG()),
    )

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

    plan_path = tmp_path / "analysis" / "planning" / "paper_plan.json"
    review_path = tmp_path / "analysis" / "planning" / "plan_review.json"
    dag_path = tmp_path / "analysis" / "planning" / "evidence_dag.json"
    assert plan_path.is_file()
    assert review_path.is_file()
    assert dag_path.is_file()
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    dag_payload = json.loads(dag_path.read_text(encoding="utf-8"))
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    assert plan_payload["title"] == metadata.title
    assert "sections" in plan_payload
    assert isinstance(dag_payload.get("evidence_nodes"), dict)
    assert isinstance(dag_payload.get("claim_nodes"), dict)
    assert isinstance(dag_payload.get("edges"), list)
    assert payload["final_status"] == "pass_with_suggestions"


@pytest.mark.integration
async def test_prepare_plan_save_output_false_does_not_write_artifacts(
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

    monkeypatch.setattr(ReferencePool, "create", classmethod(_instant_ref_pool))

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

    fake_plan = PaperPlan(
        title=metadata.title,
        sections=[SectionPlan(section_type="introduction", section_title="Introduction")],
    )
    monkeypatch.setattr(agent._planner, "create_plan", AsyncMock(return_value=fake_plan))
    monkeypatch.setattr(agent._planner, "discover_seed_references", AsyncMock(return_value=[]))
    monkeypatch.setattr(agent._planner, "discover_references", AsyncMock(return_value={}))
    monkeypatch.setattr(agent._planner, "discover_utility_references", AsyncMock(return_value={}))
    monkeypatch.setattr(agent._planner, "assign_references", lambda **_: None)
    monkeypatch.setattr(agent._planner, "_assign_papers_to_sections", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        "src.agents.metadata_agent.metadata_agent.DAGBuilder.build",
        AsyncMock(return_value=EvidenceDAG()),
    )

    await agent.prepare_plan(
        metadata=metadata,
        template_path=None,
        target_pages=None,
        enable_planning=True,
        enable_exemplar=False,
        save_output=False,
        output_dir=str(tmp_path),
    )

    assert not (tmp_path / "analysis" / "planning" / "paper_plan.json").exists()
    assert not (tmp_path / "analysis" / "planning" / "plan_review.json").exists()
    assert not (tmp_path / "analysis" / "planning" / "evidence_dag.json").exists()


@pytest.mark.integration
async def test_prepare_plan_regenerates_broken_auto_generated_figure_before_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata = PaperMetaData(
        title="Prepare Plan Figure Preprocess Test",
        idea_hypothesis="Verify preprocessing runs before path validation.",
        method="Use a mocked AcademicDreamer response.",
        data="No external data required.",
        experiments="Plan only.",
        references=[],
        figures=[
            FigureSpec(
                id="fig:generated",
                caption="Architecture overview",
                file_path="missing.png",
                auto_generate=True,
                generation_prompt="Illustrate the architecture.",
                style="cvpr",
            )
        ],
    )

    async def _fake_generate(**kwargs: object) -> dict:
        output_dir = Path(str(kwargs["output_dir"]))
        generated = output_dir / "raw.png"
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_bytes(b"generated")
        return {"output_paths": {"png": str(generated)}, "approved": True}

    monkeypatch.setattr(
        "src.agents.metadata_agent.figure_generation._load_academic_dreamer_generate",
        lambda: _fake_generate,
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

    result = await agent.prepare_plan(
        metadata=metadata,
        template_path=None,
        target_pages=None,
        enable_planning=False,
        enable_exemplar=False,
        save_output=False,
        output_dir=str(tmp_path / "paper"),
    )

    assert result.errors == []
    prepared = PaperMetaData(**result.metadata_input)
    assert prepared.materials_root == str((tmp_path / "paper").resolve())
    assert prepared.figures[0].auto_generate is False
    assert prepared.figures[0].file_path is not None
    assert (
        Path(prepared.materials_root) / prepared.figures[0].file_path
    ).is_file()
