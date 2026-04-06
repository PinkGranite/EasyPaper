"""Tests for ResearchContextBuilder."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.ep_imports import load_metadata_models, load_research_context_builder


@pytest.fixture
def models():
    return load_metadata_models()


@pytest.fixture
def builder_mod():
    return load_research_context_builder()


@pytest.mark.asyncio
async def test_build_fallback_when_json_invalid(builder_mod, models):
    ResearchContextBuilder = builder_mod.ResearchContextBuilder
    CoreRefAnalysis = models.CoreRefAnalysis
    CoreRefAnalysisItem = models.CoreRefAnalysisItem
    PaperMetaData = models.PaperMetaData

    mock_client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="not json"))]
    mock_client.chat.completions.create = AsyncMock(return_value=resp)

    core = CoreRefAnalysis(
        items=[
            CoreRefAnalysisItem(
                ref_id="a",
                title="Paper A",
                core_contributions=["c1"],
            )
        ],
        shared_gaps=["gap1"],
        research_lineage="lineage",
        positioning_statement="pos",
    )
    md = PaperMetaData(
        title="T",
        idea_hypothesis="idea",
        method="m",
        data="d",
        experiments="e",
    )
    landscape = [{"ref_id": "x", "title": "L", "abstract": "ab"}]
    b = ResearchContextBuilder(mock_client, "gpt-4")
    out = await b.build(
        core_analysis=core,
        landscape_papers=landscape,
        paper_metadata=md,
    )
    assert out.core_ref_analysis is not None
    assert "core ref" in out.summary.lower() or "anchored" in out.summary.lower()
    assert out.key_papers


@pytest.mark.asyncio
async def test_build_parses_llm_json(builder_mod, models):
    ResearchContextBuilder = builder_mod.ResearchContextBuilder
    CoreRefAnalysis = models.CoreRefAnalysis
    CoreRefAnalysisItem = models.CoreRefAnalysisItem
    PaperMetaData = models.PaperMetaData

    payload = {
        "research_area": "ML",
        "summary": "S",
        "key_papers": [{"title": "P", "contribution": "c"}],
        "research_trends": ["t"],
        "gaps": ["g"],
        "claim_evidence_matrix": [
            {
                "section_type": "related_work",
                "claim": "c",
                "support_refs": ["a"],
                "reason": "r",
                "priority": "P0",
            }
        ],
        "contribution_ranking": {"P0": [], "P1": [], "P2": []},
        "planning_decision_trace": ["d"],
    }
    mock_client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]
    mock_client.chat.completions.create = AsyncMock(return_value=resp)

    core = CoreRefAnalysis(
        items=[CoreRefAnalysisItem(ref_id="a", title="A")],
        shared_gaps=[],
        research_lineage="",
        positioning_statement="",
    )
    md = PaperMetaData(
        title="T",
        idea_hypothesis="i",
        method="m",
        data="d",
        experiments="e",
    )
    b = ResearchContextBuilder(mock_client, "gpt-4")
    out = await b.build(
        core_analysis=core,
        landscape_papers=[],
        paper_metadata=md,
    )
    assert out.research_area == "ML"
    assert out.claim_evidence_matrix[0]["section_type"] == "related_work"
    assert out.core_ref_analysis is not None


@pytest.mark.asyncio
async def test_build_uses_score_fn(builder_mod, models):
    ResearchContextBuilder = builder_mod.ResearchContextBuilder
    CoreRefAnalysis = models.CoreRefAnalysis
    PaperMetaData = models.PaperMetaData

    papers = [
        {"ref_id": "low", "title": "L", "abstract": "a"},
        {"ref_id": "high", "title": "H", "abstract": "b"},
    ]

    async def score_fn(topic: str, plist: list):
        return [(plist[1], 9.0), (plist[0], 1.0)]

    mock_client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="{}"))]
    mock_client.chat.completions.create = AsyncMock(return_value=resp)

    core = CoreRefAnalysis()
    md = PaperMetaData(
        title="T",
        idea_hypothesis="i",
        method="m",
        data="d",
        experiments="e",
    )
    b = ResearchContextBuilder(mock_client, "gpt-4")
    await b.build(
        core_analysis=core,
        landscape_papers=papers,
        paper_metadata=md,
        score_papers_fn=score_fn,
        top_k_landscape=1,
    )
    call_kw = mock_client.chat.completions.create.await_args
    user_content = call_kw.kwargs["messages"][1]["content"]
    assert "high" in user_content.lower() or "H" in user_content
