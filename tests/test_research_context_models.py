"""Tests for CoreRefAnalysis and ResearchContextModel."""
from __future__ import annotations

from src.agents.metadata_agent.models import (
    CoreRefAnalysis,
    CoreRefAnalysisItem,
    ResearchContextModel,
)


def test_core_ref_analysis_item_defaults():
    item = CoreRefAnalysisItem(ref_id="k1", title="A Paper")
    assert item.ref_id == "k1"
    assert item.title == "A Paper"
    assert item.core_contributions == []
    assert item.relevance_score == 1.0


def test_core_ref_analysis_roundtrip():
    analysis = CoreRefAnalysis(
        items=[
            CoreRefAnalysisItem(
                ref_id="a",
                title="T",
                core_contributions=["c1"],
                methodology="m",
                limitations=["l1"],
                relationship_to_ours="extends",
                key_results=["r1"],
            )
        ],
        shared_gaps=["gap1"],
        research_lineage="A then B",
        positioning_statement="We improve X",
    )
    dumped = analysis.model_dump()
    restored = CoreRefAnalysis.model_validate(dumped)
    assert restored.items[0].ref_id == "a"
    assert restored.shared_gaps == ["gap1"]


def test_research_context_model_to_dict_compatible_with_legacy():
    core = CoreRefAnalysis(
        items=[CoreRefAnalysisItem(ref_id="x", title="Y")],
        shared_gaps=[],
        research_lineage="",
        positioning_statement="",
    )
    ctx = ResearchContextModel(
        research_area="ML",
        summary="Landscape overview.",
        key_papers=[{"title": "P", "contribution": "c"}],
        research_trends=["t1"],
        gaps=["g1"],
        claim_evidence_matrix=[
            {
                "section_type": "method",
                "claim": "We use X",
                "support_refs": ["x"],
                "reason": "method",
                "priority": "P0",
            }
        ],
        contribution_ranking={"P0": [], "P1": [], "P2": []},
        planning_decision_trace=["trace1"],
        paper_assignments={"method": ["x"]},
        core_ref_analysis=core,
    )
    d = ctx.to_research_context_dict()
    assert d["research_area"] == "ML"
    assert d["summary"] == "Landscape overview."
    assert d["claim_evidence_matrix"][0]["section_type"] == "method"
    assert "core_ref_analysis" in d
    assert d["core_ref_analysis"]["items"][0]["ref_id"] == "x"


def test_research_context_model_empty_core_analysis_omitted_or_present():
    ctx = ResearchContextModel(research_area="X")
    d = ctx.to_research_context_dict()
    assert d["research_area"] == "X"
    # exclude_none=True drops None core_ref_analysis
    assert "core_ref_analysis" not in d or d.get("core_ref_analysis") is None
