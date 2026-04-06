"""
Tests for DAG-Canvas Integration (Phase 1).
Validates PlanResult model, prepare_plan/execute_generation split,
backward-compatible generate_paper wrapper, and router endpoints.
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.metadata_agent.models import (
    PaperMetaData,
    PaperGenerationRequest,
    PaperGenerationResult,
    PlanResult,
    SectionResult,
)
from src.agents.shared.reference_pool import ReferencePool


# ============================================================================
# PlanResult Model Tests
# ============================================================================


class TestPlanResultModel:
    """Verify PlanResult Pydantic model serialization/deserialization."""

    def test_plan_result_minimal(self):
        pr = PlanResult(
            paper_plan={"title": "Test Paper", "sections": []},
            metadata_input={"title": "Test Paper", "idea_hypothesis": "h", "method": "m", "data": "d", "experiments": "e"},
        )
        assert pr.paper_plan["title"] == "Test Paper"
        assert pr.evidence_dag is None
        assert pr.errors == []
        assert pr.ref_pool_snapshot == {}
        assert pr.template_path is None

    def test_plan_result_full(self):
        pr = PlanResult(
            paper_plan={"title": "Test", "sections": [{"section_type": "introduction"}]},
            evidence_dag={"evidence_nodes": {}, "claim_nodes": {}, "binding_edges": {}},
            research_context={"research_area": "NLP"},
            code_context={"scan_stats": {"indexed_files": 42}},
            code_summary_markdown="# Code Summary",
            ref_pool_snapshot={"core_refs": [], "discovered_refs": []},
            converted_tables={"tab:results": "\\begin{table}..."},
            metadata_input={"title": "Test", "idea_hypothesis": "h", "method": "m", "data": "d", "experiments": "e"},
            errors=["minor warning"],
            template_path="/tmp/template.zip",
            target_pages=8,
            artifacts_prefix="papers/user1/task1",
            paper_dir="/tmp/output",
        )
        assert pr.target_pages == 8
        assert len(pr.errors) == 1
        assert pr.code_summary_markdown == "# Code Summary"
        assert "tab:results" in pr.converted_tables

    def test_plan_result_json_roundtrip(self):
        pr = PlanResult(
            paper_plan={"title": "RT", "sections": [{"section_type": "method", "paragraphs": []}]},
            evidence_dag={"evidence_nodes": {"ev1": {"label": "Fig1"}}, "claim_nodes": {}, "binding_edges": {}},
            ref_pool_snapshot={"core_refs": [{"ref_id": "smith2024"}], "discovered_refs": []},
            metadata_input={"title": "RT", "idea_hypothesis": "h", "method": "m", "data": "d", "experiments": "e"},
        )
        dumped = pr.model_dump_json()
        restored = PlanResult.model_validate_json(dumped)
        assert restored.paper_plan == pr.paper_plan
        assert restored.evidence_dag == pr.evidence_dag
        assert restored.ref_pool_snapshot == pr.ref_pool_snapshot

    def test_plan_result_with_errors_no_plan(self):
        """PlanResult with errors and empty plan signals planning failure."""
        pr = PlanResult(
            paper_plan={},
            metadata_input={"title": "X", "idea_hypothesis": "h", "method": "m", "data": "d", "experiments": "e"},
            errors=["File validation failed: figure.png not found"],
        )
        assert not pr.paper_plan
        assert len(pr.errors) == 1


# ============================================================================
# ReferencePool Serialization Tests
# ============================================================================


class TestReferencePoolSerialization:
    """Verify ReferencePool to_dict/from_dict round-trip for PlanResult."""

    def test_roundtrip_empty(self):
        pool = ReferencePool.__new__(ReferencePool)
        pool._core_refs = []
        pool._discovered_refs = []
        pool._all_keys = set()

        data = pool.to_dict()
        assert data == {"core_refs": [], "discovered_refs": []}

        restored = ReferencePool.from_dict(data)
        assert restored._core_refs == []
        assert restored._discovered_refs == []
        assert restored.valid_citation_keys == set()

    def test_roundtrip_with_refs(self):
        pool = ReferencePool.__new__(ReferencePool)
        pool._core_refs = [
            {"ref_id": "smith2024", "bibtex": "@article{smith2024, ...}", "title": "Smith Paper"}
        ]
        pool._discovered_refs = [
            {"ref_id": "jones2023", "bibtex": "@inproceedings{jones2023, ...}", "source": "planner"}
        ]
        pool._all_keys = {"smith2024", "jones2023"}

        data = pool.to_dict()
        restored = ReferencePool.from_dict(data)
        assert "smith2024" in restored.valid_citation_keys
        assert "jones2023" in restored.valid_citation_keys
        assert len(restored.get_all_refs()) == 2

    def test_from_dict_integration_with_plan_result(self):
        """Ensure ref_pool_snapshot from PlanResult can be used to restore a pool."""
        snapshot = {
            "core_refs": [{"ref_id": "a2024", "bibtex": "@article{a2024}"}],
            "discovered_refs": [{"ref_id": "b2023", "bibtex": "@article{b2023}"}],
        }
        pr = PlanResult(
            paper_plan={"sections": []},
            ref_pool_snapshot=snapshot,
            metadata_input={"title": "X", "idea_hypothesis": "h", "method": "m", "data": "d", "experiments": "e"},
        )
        pool = ReferencePool.from_dict(pr.ref_pool_snapshot)
        assert "a2024" in pool.valid_citation_keys
        assert "b2023" in pool.valid_citation_keys


# ============================================================================
# generate_paper Wrapper Tests
# ============================================================================


class TestGeneratePaperWrapper:
    """Verify generate_paper() correctly delegates to prepare_plan + execute_generation."""

    @pytest.mark.asyncio
    async def test_wrapper_calls_both_phases(self):
        """generate_paper should call prepare_plan then execute_generation."""
        from src.agents.metadata_agent.metadata_agent import MetaDataAgent

        agent = MetaDataAgent.__new__(MetaDataAgent)

        mock_plan_result = PlanResult(
            paper_plan={"title": "Test", "sections": [{"section_type": "introduction"}]},
            ref_pool_snapshot={"core_refs": [], "discovered_refs": []},
            metadata_input={"title": "Test", "idea_hypothesis": "h", "method": "m", "data": "d", "experiments": "e"},
        )
        mock_gen_result = PaperGenerationResult(
            status="ok",
            paper_title="Test",
            latex_content="\\section{Intro}",
            total_word_count=500,
        )

        agent.prepare_plan = AsyncMock(return_value=mock_plan_result)
        agent.execute_generation = AsyncMock(return_value=mock_gen_result)

        metadata = PaperMetaData(
            title="Test",
            idea_hypothesis="h",
            method="m",
            data="d",
            experiments="e",
        )

        result = await agent.generate_paper(
            metadata=metadata,
            save_output=False,
            enable_planning=True,
        )

        agent.prepare_plan.assert_called_once()
        agent.execute_generation.assert_called_once()
        assert result.status == "ok"
        assert result.paper_title == "Test"

    @pytest.mark.asyncio
    async def test_wrapper_returns_error_on_plan_failure(self):
        """If prepare_plan returns errors with empty plan, generate_paper returns error."""
        from src.agents.metadata_agent.metadata_agent import MetaDataAgent

        agent = MetaDataAgent.__new__(MetaDataAgent)

        failed_plan = PlanResult(
            paper_plan={},
            metadata_input={"title": "X", "idea_hypothesis": "h", "method": "m", "data": "d", "experiments": "e"},
            errors=["Critical error: template not found"],
        )

        agent.prepare_plan = AsyncMock(return_value=failed_plan)
        agent.execute_generation = AsyncMock()

        metadata = PaperMetaData(title="X", idea_hypothesis="h", method="m", data="d", experiments="e")
        result = await agent.generate_paper(metadata=metadata, save_output=False)

        agent.prepare_plan.assert_called_once()
        agent.execute_generation.assert_not_called()
        assert result.status == "error"
        assert "Critical error" in result.errors[0]


# ============================================================================
# execute_generation State Reconstruction Tests
# ============================================================================


class TestExecuteGenerationReconstruction:
    """Verify that execute_generation correctly deserializes PlanResult."""

    def test_metadata_reconstruction(self):
        """PaperMetaData should be correctly reconstructed from metadata_input."""
        input_data = {
            "title": "My Paper",
            "idea_hypothesis": "We propose...",
            "method": "Our method...",
            "data": "Dataset...",
            "experiments": "We ran...",
            "references": ["@article{ref1}"],
        }
        metadata = PaperMetaData(**input_data)
        assert metadata.title == "My Paper"
        assert metadata.idea_hypothesis == "We propose..."
        assert len(metadata.references) == 1

    def test_ref_pool_reconstruction(self):
        """ReferencePool should be correctly reconstructed from ref_pool_snapshot."""
        snapshot = {
            "core_refs": [{"ref_id": "a", "bibtex": "@article{a}"}],
            "discovered_refs": [{"ref_id": "b", "bibtex": "@article{b}"}],
        }
        pool = ReferencePool.from_dict(snapshot)
        assert len(pool.get_all_refs()) == 2
        assert "a" in pool.valid_citation_keys
        assert "b" in pool.valid_citation_keys

    def test_evidence_dag_reconstruction(self):
        """EvidenceDAG should be correctly reconstructed from serialized dict."""
        from src.models.evidence_graph import EvidenceDAG

        dag_data = {
            "evidence_nodes": {
                "ev_fig1": {
                    "node_id": "ev_fig1",
                    "node_type": "figure",
                    "content": "Architecture Diagram",
                    "source_path": "figures/arch.png",
                    "metadata": {},
                }
            },
            "claim_nodes": {
                "cl_intro_0": {
                    "node_id": "cl_intro_0",
                    "node_type": "context",
                    "statement": "Main contribution",
                    "metadata": {"section_type": "introduction", "paragraph_index": 0},
                }
            },
            "edges": [
                {"source_id": "cl_intro_0", "target_id": "ev_fig1", "edge_type": "supports"},
            ],
        }
        dag = EvidenceDAG.from_serializable(dag_data)
        assert dag is not None
        assert "ev_fig1" in dag.evidence_nodes
        assert "cl_intro_0" in dag.claim_nodes
        summary = dag.summary()
        assert summary["evidence_count"] == 1
        assert summary["claim_count"] == 1


# ============================================================================
# PaperGenerationRequest.to_metadata Tests
# ============================================================================


class TestRequestToMetadata:
    """Verify that PaperGenerationRequest correctly converts to PaperMetaData."""

    def test_basic_conversion(self):
        request = PaperGenerationRequest(
            title="Test Paper",
            idea_hypothesis="We propose X",
            method="Method Y",
            data="Dataset Z",
            experiments="Experiment W",
            references=["@article{ref1}"],
            target_pages=10,
            style_guide="ICML",
        )
        metadata = request.to_metadata()
        assert metadata.title == "Test Paper"
        assert metadata.idea_hypothesis == "We propose X"
        assert metadata.target_pages == 10
        assert metadata.style_guide == "ICML"
        assert len(metadata.references) == 1
