"""Tests for CoreRefAnalyzer."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.ep_imports import load_core_ref_analyzer, load_metadata_models


@pytest.fixture
def models():
    return load_metadata_models()


@pytest.fixture
def analyzer_mod():
    return load_core_ref_analyzer()


@pytest.mark.asyncio
async def test_analyze_empty_core_refs_returns_empty(analyzer_mod, models):
    CoreRefAnalyzer = analyzer_mod.CoreRefAnalyzer
    PaperMetaData = models.PaperMetaData
    client = MagicMock()
    ar = CoreRefAnalyzer(client, "gpt-4")

    md = PaperMetaData(
        title="T",
        idea_hypothesis="i",
        method="m",
        data="d",
        experiments="e",
    )
    out = await ar.analyze([], md)
    assert out.items == []


@pytest.mark.asyncio
async def test_analyze_disabled_uses_heuristic(analyzer_mod, models, sample_core_refs):
    CoreRefAnalyzer = analyzer_mod.CoreRefAnalyzer
    PaperMetaData = models.PaperMetaData
    client = MagicMock()
    ar = CoreRefAnalyzer(client, "gpt-4", enabled=False)

    md = PaperMetaData(
        title="My Paper",
        idea_hypothesis="Better robustness",
        method="m",
        data="d",
        experiments="e",
    )
    out = await ar.analyze(sample_core_refs, md)
    assert len(out.items) == 2
    assert {i.ref_id for i in out.items} == {"smith2020", "jones2021"}
    client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_llm_populates_items(analyzer_mod, models, sample_core_refs):
    CoreRefAnalyzer = analyzer_mod.CoreRefAnalyzer
    PaperMetaData = models.PaperMetaData

    items_json = {
        "items": [
            {
                "ref_id": "smith2020",
                "title": "Deep Learning for Vision",
                "core_contributions": ["Architecture for classification"],
                "methodology": "CNN",
                "limitations": ["Data hunger"],
                "relationship_to_ours": "We extend their encoder.",
                "key_results": ["SOTA on X"],
            },
            {
                "ref_id": "jones2021",
                "title": "Robustness in Neural Networks",
                "core_contributions": ["Robust training"],
                "methodology": "Adversarial training",
                "limitations": ["Compute"],
                "relationship_to_ours": "We compare to their baseline.",
                "key_results": [],
            },
        ]
    }
    cross_json = {
        "shared_gaps": ["Unified theory"],
        "research_lineage": "Smith then Jones advanced robustness.",
        "positioning_statement": "Our paper unifies both lines.",
    }

    mock_client = MagicMock()
    resp1 = MagicMock()
    resp1.choices = [MagicMock(message=MagicMock(content=json.dumps(items_json)))]
    resp2 = MagicMock()
    resp2.choices = [MagicMock(message=MagicMock(content=json.dumps(cross_json)))]
    mock_client.chat.completions.create = AsyncMock(side_effect=[resp1, resp2])

    ar = CoreRefAnalyzer(mock_client, "gpt-4", enabled=True, analyze_cross_paper=True)
    md = PaperMetaData(
        title="Unified Robust Vision",
        idea_hypothesis="Combine efficiency and robustness",
        method="New loss",
        data="ImageNet",
        experiments="Robust accuracy",
    )
    out = await ar.analyze(sample_core_refs, md)
    assert len(out.items) == 2
    assert out.items[0].relationship_to_ours
    assert out.shared_gaps == ["Unified theory"]
    assert "Jones" in out.research_lineage
    assert mock_client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_from_tools_config_reads_nested_config(analyzer_mod):
    CoreRefAnalyzer = analyzer_mod.CoreRefAnalyzer
    mock_tools = MagicMock()
    mock_tools.core_ref_analysis = MagicMock(
        enabled=False,
        max_abstract_chars=500,
        analyze_cross_paper=False,
    )
    ar = CoreRefAnalyzer.from_tools_config(MagicMock(), "m", mock_tools)
    assert ar._enabled is False
    assert ar._max_abstract_chars == 500


# ---------------------------------------------------------------------------
# Docling integration: CoreRefAnalyzer uses docling_sections when present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_uses_docling_sections_when_present(analyzer_mod, models):
    CoreRefAnalyzer = analyzer_mod.CoreRefAnalyzer
    PaperMetaData = models.PaperMetaData

    refs_with_docling = [
        {
            "ref_id": "smith2020",
            "title": "Deep Learning for Vision",
            "abstract": "Short abstract.",
            "docling_sections": {
                "method": "We use a novel transformer architecture with attention.",
                "results": "Achieved 95% accuracy on ImageNet.",
                "conclusion": "Transformers outperform CNNs.",
            },
            "bibtex": '@article{smith2020, title={DL}, author={Smith}, year={2020}}',
        },
    ]

    items_json = {
        "items": [
            {
                "ref_id": "smith2020",
                "title": "Deep Learning for Vision",
                "core_contributions": ["Transformer architecture"],
                "methodology": "Transformer with attention",
                "limitations": [],
                "relationship_to_ours": "We extend their encoder.",
                "key_results": ["95% accuracy"],
            },
        ]
    }

    mock_client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=json.dumps(items_json)))]
    mock_client.chat.completions.create = AsyncMock(return_value=resp)

    ar = CoreRefAnalyzer(mock_client, "gpt-4", enabled=True, analyze_cross_paper=False)
    md = PaperMetaData(
        title="T", idea_hypothesis="i", method="m", data="d", experiments="e",
    )
    out = await ar.analyze(refs_with_docling, md)

    assert len(out.items) == 1
    assert out.items[0].ref_id == "smith2020"

    # Verify the LLM was called with docling section content
    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args.kwargs.get("messages", call_args[1].get("messages", []))[-1]["content"]
    assert "method_excerpt" in user_msg or "Transformer" in user_msg


@pytest.mark.asyncio
async def test_analyze_falls_back_to_abstract_without_docling(analyzer_mod, models, sample_core_refs):
    CoreRefAnalyzer = analyzer_mod.CoreRefAnalyzer
    PaperMetaData = models.PaperMetaData

    items_json = {
        "items": [
            {
                "ref_id": "smith2020",
                "title": "Deep Learning for Vision",
                "core_contributions": ["Architecture"],
                "methodology": "CNN",
                "limitations": [],
                "relationship_to_ours": "Related.",
                "key_results": [],
            },
            {
                "ref_id": "jones2021",
                "title": "Robustness",
                "core_contributions": ["Robust training"],
                "methodology": "Adversarial",
                "limitations": [],
                "relationship_to_ours": "Baseline.",
                "key_results": [],
            },
        ]
    }

    mock_client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=json.dumps(items_json)))]
    mock_client.chat.completions.create = AsyncMock(return_value=resp)

    ar = CoreRefAnalyzer(mock_client, "gpt-4", enabled=True, analyze_cross_paper=False)
    md = PaperMetaData(
        title="T", idea_hypothesis="i", method="m", data="d", experiments="e",
    )
    out = await ar.analyze(sample_core_refs, md)

    # Should still work with abstract-only refs
    assert len(out.items) == 2

    # Verify the LLM was called with abstract content (not docling sections)
    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args.kwargs.get("messages", call_args[1].get("messages", []))[-1]["content"]
    assert "abstract" in user_msg.lower() or "image classification" in user_msg.lower()
