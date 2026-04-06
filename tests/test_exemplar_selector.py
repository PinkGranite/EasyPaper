"""Tests for ExemplarSelector (TDD: RED first)."""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.shared.exemplar_selector import ExemplarSelector
from src.config.schema import ExemplarConfig


@pytest.fixture
def default_config():
    return ExemplarConfig(enabled=True, venue_match_required=True, recency_years=5)


@pytest.fixture
def relaxed_config():
    return ExemplarConfig(enabled=True, venue_match_required=False, recency_years=10)


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def sample_metadata():
    from src.agents.metadata_agent.models import PaperMetaData
    return PaperMetaData(
        title="Novel Deep Learning for Drug Discovery",
        idea_hypothesis="We hypothesize that graph neural networks improve binding prediction",
        method="Graph attention network with multi-task learning",
        data="PDBbind and ChEMBL datasets",
        experiments="Binding affinity prediction benchmarks",
        style_guide="nature",
    )


@pytest.fixture
def core_refs_with_venue_match():
    """Core refs where one matches Nature venue and has docling data."""
    return [
        {
            "ref_id": "smith2023",
            "title": "Deep Learning Advances in Drug Design",
            "venue": "Nature",
            "year": 2023,
            "abstract": "We present a new GNN for drug design.",
            "docling_full_text": "Full text content here... " * 100,
            "docling_sections": {
                "introduction": "Drug discovery is a key challenge...",
                "method": "We propose a graph attention network...",
                "results": "Our method achieves state-of-the-art...",
            },
        },
        {
            "ref_id": "jones2022",
            "title": "Transformer Models for Molecules",
            "venue": "ICML",
            "year": 2022,
            "abstract": "Transformers for molecular property prediction.",
            "docling_full_text": "Full text for ICML paper..." * 50,
            "docling_sections": {"method": "We use a transformer..."},
        },
    ]


@pytest.fixture
def core_refs_no_venue_match():
    """Core refs where none match Nature venue."""
    return [
        {
            "ref_id": "jones2022",
            "title": "Transformer Models",
            "venue": "ICML",
            "year": 2022,
            "abstract": "Abstract.",
            "docling_full_text": "Full text...",
            "docling_sections": {"method": "We use transformers..."},
        },
        {
            "ref_id": "lee2021",
            "title": "GNN Survey",
            "venue": "NeurIPS",
            "year": 2021,
            "abstract": "Survey.",
            "docling_full_text": "Survey full text...",
            "docling_sections": {"introduction": "Graphs are..."},
        },
    ]


@pytest.fixture
def core_refs_venue_match_no_docling():
    """Core ref matches venue but has no docling data."""
    return [
        {
            "ref_id": "doe2024",
            "title": "Protein Folding with AI",
            "venue": "Nature",
            "year": 2024,
            "abstract": "We fold proteins.",
        },
    ]


# ---------------------------------------------------------------------------
# _filter_hard_constraints (pure function, no LLM)
# ---------------------------------------------------------------------------

class TestFilterHardConstraints:
    def test_venue_and_year_match(self, core_refs_with_venue_match, default_config):
        selector = ExemplarSelector(MagicMock(), "gpt-4")
        current_year = datetime.now().year
        result = selector._filter_hard_constraints(
            core_refs_with_venue_match, "nature", default_config.recency_years,
        )
        assert len(result) == 1
        assert result[0]["ref_id"] == "smith2023"

    def test_no_venue_match(self, core_refs_no_venue_match, default_config):
        selector = ExemplarSelector(MagicMock(), "gpt-4")
        result = selector._filter_hard_constraints(
            core_refs_no_venue_match, "nature", default_config.recency_years,
        )
        assert len(result) == 0

    def test_venue_match_but_no_docling(self, core_refs_venue_match_no_docling, default_config):
        selector = ExemplarSelector(MagicMock(), "gpt-4")
        result = selector._filter_hard_constraints(
            core_refs_venue_match_no_docling, "nature", default_config.recency_years,
        )
        assert len(result) == 0

    def test_old_paper_excluded(self, default_config):
        selector = ExemplarSelector(MagicMock(), "gpt-4")
        refs = [
            {
                "ref_id": "old2015",
                "title": "Old Paper",
                "venue": "Nature",
                "year": 2015,
                "docling_full_text": "Text...",
                "docling_sections": {"method": "..."},
            },
        ]
        result = selector._filter_hard_constraints(refs, "nature", default_config.recency_years)
        assert len(result) == 0

    def test_no_style_guide_skips_venue_filter(self, core_refs_with_venue_match):
        selector = ExemplarSelector(MagicMock(), "gpt-4")
        result = selector._filter_hard_constraints(
            core_refs_with_venue_match, None, 5,
        )
        assert len(result) == 2

    def test_empty_refs(self):
        selector = ExemplarSelector(MagicMock(), "gpt-4")
        result = selector._filter_hard_constraints([], "nature", 5)
        assert result == []


# ---------------------------------------------------------------------------
# _rank_candidates (LLM call)
# ---------------------------------------------------------------------------

class TestRankCandidates:
    @pytest.mark.asyncio
    async def test_ranking_returns_best(self, core_refs_with_venue_match, sample_metadata):
        ranking_json = json.dumps({
            "rankings": [
                {"ref_id": "smith2023", "score": 9.0, "reason": "Very similar method"},
            ]
        })
        mock_client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content=ranking_json))]
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        selector = ExemplarSelector(mock_client, "gpt-4")
        candidates = [core_refs_with_venue_match[0]]
        best = await selector._rank_candidates(candidates, sample_metadata)
        assert best is not None
        assert best["ref_id"] == "smith2023"

    @pytest.mark.asyncio
    async def test_ranking_llm_failure_returns_first(self, core_refs_with_venue_match, sample_metadata):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("LLM down"))

        selector = ExemplarSelector(mock_client, "gpt-4")
        candidates = [core_refs_with_venue_match[0]]
        best = await selector._rank_candidates(candidates, sample_metadata)
        assert best is not None
        assert best["ref_id"] == "smith2023"

    @pytest.mark.asyncio
    async def test_single_candidate_skips_llm(self, core_refs_with_venue_match, sample_metadata):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock()

        selector = ExemplarSelector(mock_client, "gpt-4")
        candidates = [core_refs_with_venue_match[0]]
        best = await selector._rank_candidates(candidates, sample_metadata)
        assert best is not None
        mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# select (orchestrator)
# ---------------------------------------------------------------------------

class TestSelect:
    @pytest.mark.asyncio
    async def test_core_ref_selected(
        self, core_refs_with_venue_match, sample_metadata, default_config,
    ):
        mock_client = MagicMock()
        selector = ExemplarSelector(mock_client, "gpt-4")
        result = await selector.select(
            core_refs_with_venue_match, sample_metadata, default_config,
        )
        assert result is not None
        assert result["ref_id"] == "smith2023"

    @pytest.mark.asyncio
    async def test_no_match_returns_none(
        self, core_refs_no_venue_match, sample_metadata, default_config,
    ):
        mock_client = MagicMock()
        selector = ExemplarSelector(mock_client, "gpt-4")
        result = await selector.select(
            core_refs_no_venue_match, sample_metadata, default_config,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_refs_returns_none(self, sample_metadata, default_config):
        mock_client = MagicMock()
        selector = ExemplarSelector(mock_client, "gpt-4")
        result = await selector.select([], sample_metadata, default_config)
        assert result is None

    @pytest.mark.asyncio
    async def test_relaxed_config_skips_venue_filter(
        self, core_refs_no_venue_match, sample_metadata, relaxed_config,
    ):
        mock_client = MagicMock()
        selector = ExemplarSelector(mock_client, "gpt-4")
        result = await selector.select(
            core_refs_no_venue_match, sample_metadata, relaxed_config,
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_venue_match_no_docling_returns_none(
        self, core_refs_venue_match_no_docling, sample_metadata, default_config,
    ):
        mock_client = MagicMock()
        selector = ExemplarSelector(mock_client, "gpt-4")
        result = await selector.select(
            core_refs_venue_match_no_docling, sample_metadata, default_config,
        )
        assert result is None
