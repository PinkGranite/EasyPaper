"""Tests for ExemplarSelector external search fallback (TDD: RED)."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.shared.exemplar_selector import ExemplarSelector
from src.config.schema import ExemplarConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config():
    return ExemplarConfig(
        enabled=True,
        venue_match_required=True,
        recency_years=5,
        max_external_candidates=5,
    )


@pytest.fixture
def sample_metadata():
    from src.agents.metadata_agent.models import PaperMetaData
    return PaperMetaData(
        title="BLIP-2: Vision-Language Pre-training",
        idea_hypothesis="Bootstrap VLP from frozen encoders",
        method="Querying Transformer with two-stage pre-training",
        data="COCO, Visual Genome, CC3M",
        experiments="VQA, captioning, retrieval benchmarks",
        style_guide="ICML",
    )


@pytest.fixture
def paper_search_config():
    return {
        "semantic_scholar_api_key": "test-key",
        "timeout": 10,
    }


@pytest.fixture
def mock_search_results():
    """Simulated PaperSearchTool results with open_access_pdf."""
    return [
        {
            "bibtex_key": "chen2023visual",
            "title": "Visual Instruction Tuning",
            "authors": ["Liu, H.", "Li, C."],
            "year": 2023,
            "venue": "International Conference on Machine Learning",
            "abstract": "We present LLaVA, a large multimodal model...",
            "open_access_pdf": "https://arxiv.org/pdf/2304.08485.pdf",
            "arxiv_id": "2304.08485",
            "bibtex": "@inproceedings{chen2023visual, title={...}}",
        },
        {
            "bibtex_key": "wang2023cogvlm",
            "title": "CogVLM: Visual Expert for Pretrained Language Models",
            "authors": ["Wang, W."],
            "year": 2023,
            "venue": "ICML",
            "abstract": "We introduce CogVLM...",
            "open_access_pdf": "https://arxiv.org/pdf/2311.03079.pdf",
            "arxiv_id": "2311.03079",
            "bibtex": "@inproceedings{wang2023cogvlm, title={...}}",
        },
        {
            "bibtex_key": "zhu2023minigpt",
            "title": "MiniGPT-4",
            "authors": ["Zhu, D."],
            "year": 2023,
            "venue": "NeurIPS",
            "abstract": "MiniGPT-4 aligns a frozen visual encoder...",
            "open_access_pdf": None,
            "arxiv_id": None,
            "bibtex": "@article{zhu2023minigpt, title={...}}",
        },
    ]


@pytest.fixture
def core_refs_no_match():
    """Core refs that won't pass hard constraints for ICML."""
    return [
        {
            "ref_id": "radford2021clip",
            "title": "Learning Transferable Visual Models",
            "venue": "arXiv",
            "year": 2021,
            "abstract": "CLIP paper.",
        },
    ]


# ---------------------------------------------------------------------------
# Venue alias matching
# ---------------------------------------------------------------------------

class TestVenueAliasMatching:
    """Venue matching should handle common abbreviation ↔ full name pairs."""

    def test_icml_matches_full_name(self):
        selector = ExemplarSelector(MagicMock(), "gpt-4")
        refs = [
            {
                "ref_id": "test2023",
                "title": "Test Paper",
                "venue": "International Conference on Machine Learning",
                "year": 2023,
                "docling_full_text": "Full text...",
            },
        ]
        result = selector._filter_hard_constraints(refs, "ICML", 5)
        assert len(result) == 1

    def test_neurips_matches_full_name(self):
        selector = ExemplarSelector(MagicMock(), "gpt-4")
        refs = [
            {
                "ref_id": "test2024",
                "title": "Test NeurIPS Paper",
                "venue": "Advances in Neural Information Processing Systems",
                "year": 2024,
                "docling_full_text": "Full text...",
            },
        ]
        result = selector._filter_hard_constraints(refs, "NeurIPS", 5)
        assert len(result) == 1

    def test_acl_matches_full_name(self):
        selector = ExemplarSelector(MagicMock(), "gpt-4")
        refs = [
            {
                "ref_id": "test2024",
                "title": "Test ACL Paper",
                "venue": "Annual Meeting of the Association for Computational Linguistics",
                "year": 2024,
                "docling_full_text": "Full text...",
            },
        ]
        result = selector._filter_hard_constraints(refs, "ACL", 5)
        assert len(result) == 1

    def test_nature_still_works(self):
        selector = ExemplarSelector(MagicMock(), "gpt-4")
        refs = [
            {
                "ref_id": "test2023",
                "title": "Test Paper",
                "venue": "Nature",
                "year": 2023,
                "docling_full_text": "Full text...",
            },
        ]
        result = selector._filter_hard_constraints(refs, "nature", 5)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _build_search_query
# ---------------------------------------------------------------------------

class TestBuildSearchQuery:
    """ExemplarSelector should construct a good search query from metadata."""

    def test_includes_style_guide_and_domain(self, sample_metadata):
        selector = ExemplarSelector(MagicMock(), "gpt-4")
        query = selector._build_search_query(sample_metadata, "ICML")
        assert "ICML" in query
        assert len(query) > 10

    def test_no_style_guide_uses_method(self, sample_metadata):
        selector = ExemplarSelector(MagicMock(), "gpt-4")
        query = selector._build_search_query(sample_metadata, None)
        assert len(query) > 10


# ---------------------------------------------------------------------------
# _search_external (mocked PaperSearchTool)
# ---------------------------------------------------------------------------

class TestSearchExternal:
    """External search should call PaperSearchTool and filter by downloadability."""

    @pytest.mark.asyncio
    async def test_returns_downloadable_candidates(
        self, sample_metadata, default_config, paper_search_config,
        mock_search_results,
    ):
        mock_tool_result = MagicMock()
        mock_tool_result.success = True
        mock_tool_result.data = {"papers": mock_search_results, "total_found": 3}

        with patch(
            "src.agents.shared.tools.paper_search.PaperSearchTool"
        ) as MockTool:
            tool_instance = MockTool.return_value
            tool_instance.execute = AsyncMock(return_value=mock_tool_result)

            selector = ExemplarSelector(MagicMock(), "gpt-4")
            candidates = await selector._search_external(
                sample_metadata, default_config,
                paper_search_config, "ICML",
            )

        assert len(candidates) >= 1
        for c in candidates:
            assert c.get("open_access_pdf") or c.get("arxiv_id")

    @pytest.mark.asyncio
    async def test_filters_by_venue_when_required(
        self, sample_metadata, default_config, paper_search_config,
        mock_search_results,
    ):
        mock_tool_result = MagicMock()
        mock_tool_result.success = True
        mock_tool_result.data = {"papers": mock_search_results, "total_found": 3}

        with patch(
            "src.agents.shared.tools.paper_search.PaperSearchTool"
        ) as MockTool:
            tool_instance = MockTool.return_value
            tool_instance.execute = AsyncMock(return_value=mock_tool_result)

            selector = ExemplarSelector(MagicMock(), "gpt-4")
            candidates = await selector._search_external(
                sample_metadata, default_config,
                paper_search_config, "ICML",
            )

        venues = [c.get("venue", "").lower() for c in candidates]
        for v in venues:
            assert "icml" in v or "machine learning" in v

    @pytest.mark.asyncio
    async def test_search_failure_returns_empty(
        self, sample_metadata, default_config, paper_search_config,
    ):
        mock_tool_result = MagicMock()
        mock_tool_result.success = False
        mock_tool_result.data = {"papers": [], "total_found": 0}

        with patch(
            "src.agents.shared.tools.paper_search.PaperSearchTool"
        ) as MockTool:
            tool_instance = MockTool.return_value
            tool_instance.execute = AsyncMock(return_value=mock_tool_result)

            selector = ExemplarSelector(MagicMock(), "gpt-4")
            candidates = await selector._search_external(
                sample_metadata, default_config,
                paper_search_config, "ICML",
            )

        assert candidates == []


# ---------------------------------------------------------------------------
# select() with external fallback
# ---------------------------------------------------------------------------

class TestSelectWithExternalFallback:
    """select() should fall back to external search when core refs fail."""

    @pytest.mark.asyncio
    async def test_fallback_to_external_search(
        self, core_refs_no_match, sample_metadata, default_config,
        paper_search_config, mock_search_results,
    ):
        ranking_json = json.dumps({
            "rankings": [
                {"ref_id": "wang2023cogvlm", "score": 8.5, "reason": "Similar venue"},
            ]
        })
        mock_client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content=ranking_json))]
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        mock_tool_result = MagicMock()
        mock_tool_result.success = True
        mock_tool_result.data = {"papers": mock_search_results, "total_found": 3}

        with patch(
            "src.agents.shared.tools.paper_search.PaperSearchTool"
        ) as MockTool:
            tool_instance = MockTool.return_value
            tool_instance.execute = AsyncMock(return_value=mock_tool_result)

            selector = ExemplarSelector(mock_client, "gpt-4")
            result = await selector.select(
                core_refs_no_match, sample_metadata, default_config,
                paper_search_config=paper_search_config,
            )

        assert result is not None
        assert result.get("open_access_pdf") or result.get("arxiv_id")

    @pytest.mark.asyncio
    async def test_no_fallback_when_search_config_missing(
        self, core_refs_no_match, sample_metadata, default_config,
    ):
        mock_client = MagicMock()
        selector = ExemplarSelector(mock_client, "gpt-4")
        result = await selector.select(
            core_refs_no_match, sample_metadata, default_config,
            paper_search_config=None,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_core_refs_preferred_over_external(
        self, sample_metadata, default_config, paper_search_config,
    ):
        """When core refs pass, external search should NOT be called."""
        good_core_refs = [
            {
                "ref_id": "good2024",
                "title": "Good ICML Paper",
                "venue": "ICML",
                "year": 2024,
                "docling_full_text": "Full text...",
                "docling_sections": {"method": "..."},
            },
        ]
        mock_client = MagicMock()
        selector = ExemplarSelector(mock_client, "gpt-4")

        with patch(
            "src.agents.shared.tools.paper_search.PaperSearchTool"
        ) as MockTool:
            result = await selector.select(
                good_core_refs, sample_metadata, default_config,
                paper_search_config=paper_search_config,
            )
            MockTool.assert_not_called()

        assert result is not None
        assert result["ref_id"] == "good2024"
