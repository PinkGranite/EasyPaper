"""
Integration tests for execute_generation output artifacts.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.metadata_agent.metadata_agent import MetaDataAgent
from src.agents.metadata_agent.models import FigureSpec, PaperMetaData, PlanResult, SectionResult
from src.agents.planner_agent.models import PaperPlan, SectionPlan
from src.agents.shared.reference_pool import ReferencePool
from src.agents.shared.session_memory import ReviewRecord
from src.config.schema import ModelConfig, ToolsConfig


def _minimal_metadata() -> PaperMetaData:
    return PaperMetaData(
        title="Execute Generation Output Test",
        idea_hypothesis="Ensure execute_generation writes expected output artifacts.",
        method="Use deterministic mocked section generation.",
        data="No external data required.",
        experiments="Run execute_generation and inspect output directory.",
        references=[],
    )


def _build_agent() -> MetaDataAgent:
    model = ModelConfig(
        model_name="stub-model",
        api_key="stub-key",
        base_url="http://127.0.0.1:9",
    )
    return MetaDataAgent(
        model,
        tools_config=ToolsConfig(enabled=False, available_tools=[]),
    )


def _patch_generation_flow(
    agent: MetaDataAgent,
    monkeypatch: pytest.MonkeyPatch,
    emit_prompt_trace: bool = False,
    add_review_iteration: bool = False,
    mock_orchestration: bool = True,
) -> None:
    monkeypatch.setattr(
        agent,
        "_generate_introduction",
        AsyncMock(
            return_value=SectionResult(
                section_type="introduction",
                section_title="Introduction",
                status="ok",
                latex_content="Introduction content.",
                word_count=12,
            ),
        ),
    )

    async def _fake_body_section(**kwargs: object) -> SectionResult:
        section_type = str(kwargs["section_type"])
        if emit_prompt_trace:
            traces = kwargs.get("prompt_traces")
            if isinstance(traces, list):
                traces.append(
                    {
                        "section_type": section_type,
                        "prompt": f"{section_type}-prompt",
                        "response": f"{section_type}-response",
                    },
                )
        return SectionResult(
            section_type=section_type,
            section_title=section_type.title(),
            status="ok",
            latex_content=f"{section_type.title()} content.",
            word_count=10,
        )

    monkeypatch.setattr(agent, "_generate_body_section", AsyncMock(side_effect=_fake_body_section))

    async def _fake_synthesis_section(**kwargs: object) -> SectionResult:
        section_type = str(kwargs["section_type"])
        return SectionResult(
            section_type=section_type,
            section_title=section_type.title(),
            status="ok",
            latex_content=f"{section_type.title()} content.",
            word_count=8,
        )

    monkeypatch.setattr(
        agent,
        "_generate_synthesis_section",
        AsyncMock(side_effect=_fake_synthesis_section),
    )

    if mock_orchestration:
        async def _fake_orchestration(**kwargs: object):
            if add_review_iteration:
                memory = kwargs.get("memory")
                if memory is not None:
                    memory.review_history.append(
                        ReviewRecord(
                            iteration=1,
                            reviewer="MockReviewer",
                            passed=False,
                            feedback_summary="Mock issue detected.",
                            section_feedbacks={
                                "method": {"action": "revise", "paragraph_feedbacks": []},
                            },
                        ),
                    )
            return (
                kwargs["generated_sections"],
                kwargs["sections_results"],
                0,
                None,
                None,
                [],
            )

        monkeypatch.setattr(agent._orchestrator, "_run_review_orchestration", _fake_orchestration)


@pytest.mark.integration
async def test_execute_generation_exports_core_files_and_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata = _minimal_metadata()
    agent = _build_agent()

    plan = PaperPlan(
        title=metadata.title,
        sections=[
            SectionPlan(section_type="abstract", section_title="Abstract"),
            SectionPlan(section_type="introduction", section_title="Introduction"),
            SectionPlan(section_type="method", section_title="Method"),
            SectionPlan(section_type="conclusion", section_title="Conclusion"),
        ],
    )
    ref_pool = ReferencePool(metadata.references)
    plan_result = PlanResult(
        paper_plan=plan.model_dump(),
        ref_pool_snapshot=ref_pool.to_dict(),
        metadata_input=metadata.model_dump(),
        paper_dir=str(tmp_path),
    )
    _patch_generation_flow(agent, monkeypatch, add_review_iteration=True)

    result = await agent.execute_generation(
        plan_result=plan_result,
        enable_review=False,
        compile_pdf=False,
        save_output=True,
        output_dir=str(tmp_path),
    )

    assert result.status == "ok"
    assert (tmp_path / "main.tex").is_file()
    assert (tmp_path / "references.bib").is_file()
    assert (tmp_path / "metadata.json").is_file()
    assert (tmp_path / "analysis" / "planning").is_dir()
    assert (tmp_path / "analysis" / "research_context").is_dir()
    assert (tmp_path / "analysis" / "citations").is_dir()
    assert (tmp_path / "analysis" / "structure").is_dir()
    assert (tmp_path / "analysis" / "review").is_dir()
    assert (tmp_path / "analysis" / "references").is_dir()
    assert (tmp_path / "analysis" / "code_context").is_dir()
    assert (tmp_path / "logs" / "traces").is_dir()

    manifest_path = tmp_path / "artifacts_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("manifest_version") == "1"
    assert manifest.get("paper_title") == metadata.title
    assert "generated_at" in manifest
    assert manifest.get("run", {}).get("review_iterations") == 0
    rel_paths = {entry["path"] for entry in manifest.get("files", [])}
    assert "main.tex" in rel_paths
    assert "metadata.json" in rel_paths
    assert "logs/review/review_history.json" in rel_paths
    for entry in manifest.get("files", []):
        if entry["path"] == "main.tex":
            assert entry.get("sha256")
            assert entry.get("size_bytes", 0) > 0
            break
    else:
        pytest.fail("main.tex missing from artifacts_manifest files list")


@pytest.mark.integration
async def test_execute_generation_exports_analysis_artifact_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata = _minimal_metadata()
    agent = _build_agent()
    plan = PaperPlan(
        title=metadata.title,
        sections=[
            SectionPlan(section_type="abstract", section_title="Abstract"),
            SectionPlan(section_type="introduction", section_title="Introduction"),
            SectionPlan(section_type="method", section_title="Method"),
            SectionPlan(section_type="conclusion", section_title="Conclusion"),
        ],
    )
    ref_pool = ReferencePool(metadata.references)
    plan_result = PlanResult(
        paper_plan=plan.model_dump(),
        research_context={"research_area": "artifact-export", "summary": "mocked context"},
        code_context={"writing_assets": {"repo_summary": "mock code context"}},
        code_summary_markdown="## Mock Code Summary",
        ref_pool_snapshot=ref_pool.to_dict(),
        metadata_input=metadata.model_dump(),
        paper_dir=str(tmp_path),
    )

    _patch_generation_flow(agent, monkeypatch, add_review_iteration=True)

    result = await agent.execute_generation(
        plan_result=plan_result,
        enable_review=False,
        compile_pdf=False,
        save_output=True,
        output_dir=str(tmp_path),
    )

    assert result.status == "ok"

    research_context_path = tmp_path / "analysis" / "research_context" / "research_context.json"
    citations_path = tmp_path / "analysis" / "citations" / "citation_budget_usage.json"
    structure_path = tmp_path / "analysis" / "structure" / "structure_summary.json"
    review_path = tmp_path / "analysis" / "review" / "review_history.json"
    references_path = tmp_path / "analysis" / "references" / "ref_pool_snapshot.json"
    code_context_path = tmp_path / "analysis" / "code_context" / "code_context.json"
    code_summary_path = tmp_path / "analysis" / "code_context" / "code_summary.md"
    prompt_traces_path = tmp_path / "logs" / "traces" / "prompt_traces.json"
    usage_report_path = tmp_path / "logs" / "traces" / "usage_report.json"

    assert research_context_path.is_file()
    assert citations_path.is_file()
    assert structure_path.is_file()
    assert review_path.is_file()
    assert references_path.is_file()
    assert code_context_path.is_file()
    assert code_summary_path.is_file()
    assert prompt_traces_path.is_file()
    assert usage_report_path.is_file()
    assert (tmp_path / "logs" / "review" / "review_history.json").is_file()

    research_payload = json.loads(research_context_path.read_text(encoding="utf-8"))
    references_payload = json.loads(references_path.read_text(encoding="utf-8"))
    usage_payload = json.loads(usage_report_path.read_text(encoding="utf-8"))
    traces_payload = json.loads(prompt_traces_path.read_text(encoding="utf-8"))
    review_payload = json.loads(review_path.read_text(encoding="utf-8"))
    logs_review_payload = json.loads(
        (tmp_path / "logs" / "review" / "review_history.json").read_text(encoding="utf-8"),
    )
    assert "summary" in research_payload
    assert "core_refs" in references_payload
    assert "summary" in usage_payload
    assert traces_payload.get("status") == "disabled"
    assert traces_payload.get("count") == 0
    assert traces_payload.get("items") == []
    assert len(review_payload.get("iterations", [])) == 1
    assert review_payload == logs_review_payload


@pytest.mark.integration
async def test_execute_generation_save_output_false_skips_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata = _minimal_metadata()
    agent = _build_agent()
    plan = PaperPlan(
        title=metadata.title,
        sections=[
            SectionPlan(section_type="abstract", section_title="Abstract"),
            SectionPlan(section_type="introduction", section_title="Introduction"),
            SectionPlan(section_type="method", section_title="Method"),
            SectionPlan(section_type="conclusion", section_title="Conclusion"),
        ],
    )
    ref_pool = ReferencePool(metadata.references)
    plan_result = PlanResult(
        paper_plan=plan.model_dump(),
        ref_pool_snapshot=ref_pool.to_dict(),
        metadata_input=metadata.model_dump(),
        paper_dir=str(tmp_path),
    )
    _patch_generation_flow(agent, monkeypatch, mock_orchestration=False)

    result = await agent.execute_generation(
        plan_result=plan_result,
        enable_review=False,
        compile_pdf=False,
        save_output=False,
        output_dir=str(tmp_path),
    )

    assert result.status == "ok"
    assert not (tmp_path / "main.tex").exists()
    assert not (tmp_path / "references.bib").exists()
    assert not (tmp_path / "metadata.json").exists()
    assert not (tmp_path / "analysis").exists()
    assert not (tmp_path / "logs").exists()
    assert not (tmp_path / "artifacts_manifest.json").exists()


@pytest.mark.integration
async def test_execute_generation_compile_without_template_creates_iteration_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata = _minimal_metadata()
    agent = _build_agent()
    plan = PaperPlan(
        title=metadata.title,
        sections=[
            SectionPlan(section_type="abstract", section_title="Abstract"),
            SectionPlan(section_type="introduction", section_title="Introduction"),
            SectionPlan(section_type="method", section_title="Method"),
            SectionPlan(section_type="conclusion", section_title="Conclusion"),
        ],
    )
    ref_pool = ReferencePool(metadata.references)
    plan_result = PlanResult(
        paper_plan=plan.model_dump(),
        ref_pool_snapshot=ref_pool.to_dict(),
        metadata_input=metadata.model_dump(),
        paper_dir=str(tmp_path),
        template_path=None,
    )
    _patch_generation_flow(agent, monkeypatch, mock_orchestration=False)

    async def _fake_compile_pdf(**kwargs: object):
        output_dir = Path(str(kwargs["output_dir"]))
        sections_dir = output_dir / "sections"
        sections_dir.mkdir(parents=True, exist_ok=True)
        (sections_dir / "introduction.tex").write_text("Introduction content.", encoding="utf-8")
        (output_dir / "main.tex").write_text("\\input{sections/introduction}", encoding="utf-8")
        (output_dir / "references.bib").write_text("", encoding="utf-8")
        return str(output_dir / "main.pdf"), str(output_dir), [], {}

    monkeypatch.setattr(agent, "_compile_pdf", AsyncMock(side_effect=_fake_compile_pdf))

    result = await agent.execute_generation(
        plan_result=plan_result,
        enable_review=False,
        compile_pdf=True,
        save_output=True,
        output_dir=str(tmp_path),
    )

    assert result.status == "ok"
    assert (tmp_path / "iteration_01").is_dir()
    assert (tmp_path / "iteration_01" / "main.tex").is_file()
    assert (tmp_path / "iteration_01" / "references.bib").is_file()
    assert (tmp_path / "iteration_01" / "sections" / "introduction.tex").is_file()
    assert not (tmp_path / "iteration_01_final").exists()


@pytest.mark.integration
async def test_execute_generation_prompt_traces_enabled_exports_items(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata = PaperMetaData(
        title="Prompt Trace Enabled Test",
        idea_hypothesis="Validate enabled prompt trace export.",
        method="Emit a mocked trace from body generation.",
        data="No external data required.",
        experiments="Enable export_prompt_traces.",
        references=[],
        export_prompt_traces=True,
    )
    agent = _build_agent()
    plan = PaperPlan(
        title=metadata.title,
        sections=[
            SectionPlan(section_type="abstract", section_title="Abstract"),
            SectionPlan(section_type="introduction", section_title="Introduction"),
            SectionPlan(section_type="method", section_title="Method"),
            SectionPlan(section_type="conclusion", section_title="Conclusion"),
        ],
    )
    ref_pool = ReferencePool(metadata.references)
    plan_result = PlanResult(
        paper_plan=plan.model_dump(),
        ref_pool_snapshot=ref_pool.to_dict(),
        metadata_input=metadata.model_dump(),
        paper_dir=str(tmp_path),
    )

    _patch_generation_flow(agent, monkeypatch, emit_prompt_trace=True)

    result = await agent.execute_generation(
        plan_result=plan_result,
        enable_review=False,
        compile_pdf=False,
        save_output=True,
        output_dir=str(tmp_path),
    )

    assert result.status == "ok"
    traces_path = tmp_path / "logs" / "traces" / "prompt_traces.json"
    assert traces_path.is_file()
    payload = json.loads(traces_path.read_text(encoding="utf-8"))
    assert payload.get("status") == "enabled"
    assert payload.get("count", 0) >= 1
    assert isinstance(payload.get("items"), list)


@pytest.mark.integration
async def test_execute_generation_does_not_regenerate_materialized_auto_generated_figure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    materials_root = tmp_path / "materials"
    materials_root.mkdir()
    (materials_root / "generated_fig.png").write_bytes(b"generated")

    metadata = PaperMetaData(
        title="Execute Generation Figure Re-entry Test",
        idea_hypothesis="Ensure materialized figures are not regenerated.",
        method="Use mocked section generation.",
        data="No external data required.",
        experiments="Run execute_generation once.",
        references=[],
        materials_root=str(materials_root),
        figures=[
            FigureSpec(
                id="fig:generated",
                caption="Existing generated figure",
                file_path="generated_fig.png",
                auto_generate=False,
                generation_prompt="Should not be reused.",
            )
        ],
    )
    agent = _build_agent()
    plan = PaperPlan(
        title=metadata.title,
        sections=[
            SectionPlan(section_type="abstract", section_title="Abstract"),
            SectionPlan(section_type="introduction", section_title="Introduction"),
            SectionPlan(section_type="method", section_title="Method"),
        ],
    )
    ref_pool = ReferencePool(metadata.references)
    plan_result = PlanResult(
        paper_plan=plan.model_dump(),
        ref_pool_snapshot=ref_pool.to_dict(),
        metadata_input=metadata.model_dump(),
        paper_dir=str(tmp_path / "paper"),
    )
    _patch_generation_flow(agent, monkeypatch)

    monkeypatch.setattr(
        "src.agents.metadata_agent.figure_generation._load_academic_dreamer_generate",
        lambda: pytest.fail("AcademicDreamer should not load for resolved figure assets"),
    )

    result = await agent.execute_generation(
        plan_result=plan_result,
        enable_review=False,
        compile_pdf=False,
        save_output=False,
        output_dir=str(tmp_path / "paper"),
    )

    assert result.status == "ok"
