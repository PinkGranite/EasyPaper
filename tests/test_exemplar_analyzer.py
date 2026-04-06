"""Tests for ExemplarAnalyzer (TDD: RED first)."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.shared.exemplar_analyzer import ExemplarAnalyzer
from src.agents.metadata_agent.models import (
    ExemplarAnalysis,
    PaperMetaData,
)


@pytest.fixture
def sample_metadata():
    return PaperMetaData(
        title="Novel GNN for Drug Discovery",
        idea_hypothesis="Graph networks improve binding prediction",
        method="Graph attention with multi-task learning",
        data="PDBbind dataset",
        experiments="Binding affinity benchmarks",
        style_guide="nature",
    )


@pytest.fixture
def sample_sections():
    return {
        "introduction": "Drug discovery remains a critical challenge in biomedicine...",
        "method": "We propose a graph attention network (GAT) with multi-scale pooling...",
        "results": "Our method achieves 0.82 Pearson correlation on PDBbind core set...",
        "discussion": "The results demonstrate that graph-based approaches outperform...",
        "conclusion": "We have presented a novel GNN architecture for binding prediction...",
    }


@pytest.fixture
def sample_full_text(sample_sections):
    return "\n\n".join(
        f"## {k.title()}\n{v}" for k, v in sample_sections.items()
    )


@pytest.fixture
def llm_analysis_response():
    """A valid LLM JSON response for ExemplarAnalysis."""
    return json.dumps({
        "section_blueprint": [
            {
                "section_type": "introduction",
                "title": "Introduction",
                "approx_word_count": 800,
                "paragraph_count": 4,
                "subsection_titles": [],
                "role": "Establish significance and gap",
            },
            {
                "section_type": "method",
                "title": "Methods",
                "approx_word_count": 1200,
                "paragraph_count": 6,
                "subsection_titles": ["Architecture", "Training"],
                "role": "Detail the proposed approach",
            },
            {
                "section_type": "results",
                "title": "Results",
                "approx_word_count": 1000,
                "paragraph_count": 5,
                "subsection_titles": ["Main Results", "Ablation"],
                "role": "Present experimental findings",
            },
        ],
        "style_profile": {
            "tone": "formal",
            "citation_density": 3.2,
            "avg_sentence_length": 24.0,
            "hedging_level": "moderate",
            "transition_patterns": ["however", "furthermore", "in contrast"],
        },
        "argumentation_patterns": {
            "intro_hook_type": "broad_significance",
            "claim_evidence_structure": "claim_first_then_evidence",
            "discussion_closing_strategy": "limitation_then_future_work",
        },
        "paragraph_archetypes": {
            "introduction": ["broad_hook", "gap_statement", "contribution_list", "roadmap"],
            "method": ["overview", "architecture_detail", "training_detail"],
            "results": ["main_finding", "comparison_table", "ablation_insight"],
        },
    })


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    @pytest.mark.asyncio
    async def test_full_analysis(
        self, sample_full_text, sample_sections, sample_metadata, llm_analysis_response,
    ):
        mock_client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content=llm_analysis_response))]
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        analyzer = ExemplarAnalyzer(mock_client, "gpt-4")
        ref_info = {"ref_id": "smith2023", "title": "A Paper", "venue": "Nature", "year": 2023}
        result = await analyzer.analyze(
            full_text=sample_full_text,
            sections=sample_sections,
            metadata=sample_metadata,
            ref_info=ref_info,
        )
        assert isinstance(result, ExemplarAnalysis)
        assert result.ref_id == "smith2023"
        assert len(result.section_blueprint) == 3
        assert result.style_profile.citation_density == 3.2
        assert result.argumentation_patterns.intro_hook_type == "broad_significance"
        assert "introduction" in result.paragraph_archetypes

    @pytest.mark.asyncio
    async def test_partial_sections(self, sample_metadata):
        partial_response = json.dumps({
            "section_blueprint": [
                {"section_type": "method", "title": "Methods", "paragraph_count": 3},
            ],
            "style_profile": {"tone": "formal"},
            "argumentation_patterns": {},
            "paragraph_archetypes": {"method": ["overview"]},
        })
        mock_client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content=partial_response))]
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        analyzer = ExemplarAnalyzer(mock_client, "gpt-4")
        ref_info = {"ref_id": "x", "title": "X", "venue": "ICML", "year": 2024}
        result = await analyzer.analyze(
            full_text="## Method\nSome method text...",
            sections={"method": "Some method text..."},
            metadata=sample_metadata,
            ref_info=ref_info,
        )
        assert len(result.section_blueprint) == 1

    @pytest.mark.asyncio
    async def test_llm_malformed_json_uses_heuristic(self, sample_metadata, sample_sections, sample_full_text):
        mock_client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="not json at all!!!"))]
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        analyzer = ExemplarAnalyzer(mock_client, "gpt-4")
        ref_info = {"ref_id": "bad", "title": "Bad", "venue": "Nature", "year": 2024}
        result = await analyzer.analyze(
            full_text=sample_full_text,
            sections=sample_sections,
            metadata=sample_metadata,
            ref_info=ref_info,
        )
        assert isinstance(result, ExemplarAnalysis)
        assert result.ref_id == "bad"
        assert len(result.section_blueprint) > 0

    @pytest.mark.asyncio
    async def test_llm_exception_uses_heuristic(self, sample_metadata, sample_sections, sample_full_text):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("timeout"))

        analyzer = ExemplarAnalyzer(mock_client, "gpt-4")
        ref_info = {"ref_id": "err", "title": "Err", "venue": "ICML", "year": 2023}
        result = await analyzer.analyze(
            full_text=sample_full_text,
            sections=sample_sections,
            metadata=sample_metadata,
            ref_info=ref_info,
        )
        assert isinstance(result, ExemplarAnalysis)
        assert result.ref_id == "err"

    @pytest.mark.asyncio
    async def test_empty_full_text(self, sample_metadata):
        mock_client = MagicMock()
        analyzer = ExemplarAnalyzer(mock_client, "gpt-4")
        ref_info = {"ref_id": "empty", "title": "Empty", "venue": "X", "year": 2024}
        result = await analyzer.analyze(
            full_text="",
            sections={},
            metadata=sample_metadata,
            ref_info=ref_info,
        )
        assert isinstance(result, ExemplarAnalysis)
        assert result.section_blueprint == []


# ---------------------------------------------------------------------------
# format_for_prompt
# ---------------------------------------------------------------------------

class TestFormatForPrompt:
    def test_introduction_guidance(self, llm_analysis_response):
        data = json.loads(llm_analysis_response)
        analysis = ExemplarAnalysis(
            ref_id="x", title="Paper X", venue="Nature", year=2024,
            section_blueprint=[
                ExemplarAnalysis.model_fields["section_blueprint"].default_factory()[0]
                if False else  # just build from data
                __import__("src.agents.metadata_agent.models", fromlist=["SectionBlueprint"]).SectionBlueprint(**bp)
                for bp in data["section_blueprint"]
            ],
            style_profile=__import__("src.agents.metadata_agent.models", fromlist=["StyleProfile"]).StyleProfile(**data["style_profile"]),
            argumentation_patterns=__import__("src.agents.metadata_agent.models", fromlist=["ArgumentationPatterns"]).ArgumentationPatterns(**data["argumentation_patterns"]),
            paragraph_archetypes=data["paragraph_archetypes"],
        )
        output = ExemplarAnalyzer.format_for_prompt(analysis, "introduction")
        assert "Paper X" in output
        assert "Nature" in output
        assert "introduction" in output.lower()
        assert "broad hook" in output.lower()

    def test_method_guidance(self):
        from src.agents.metadata_agent.models import (
            SectionBlueprint, StyleProfile, ArgumentationPatterns,
        )
        analysis = ExemplarAnalysis(
            ref_id="y", title="Paper Y", venue="ICML", year=2024,
            section_blueprint=[
                SectionBlueprint(section_type="method", title="Method", paragraph_count=5),
            ],
            style_profile=StyleProfile(citation_density=2.0),
            paragraph_archetypes={"method": ["overview", "detail", "implementation"]},
        )
        output = ExemplarAnalyzer.format_for_prompt(analysis, "method")
        assert "Paper Y" in output
        assert "method" in output.lower()
        assert "overview" in output.lower()

    def test_missing_section_type(self):
        analysis = ExemplarAnalysis(
            ref_id="z", title="Paper Z", venue="Nature", year=2024,
        )
        output = ExemplarAnalyzer.format_for_prompt(analysis, "related_work")
        assert "Paper Z" in output
        assert len(output) > 0

    def test_none_analysis_returns_empty(self):
        output = ExemplarAnalyzer.format_for_prompt(None, "introduction")
        assert output == ""
