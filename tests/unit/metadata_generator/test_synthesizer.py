"""Tests for MetadataSynthesizer (LLM-based, all calls mocked)."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.metadata_agent.models import PaperMetaData
from src.agents.metadata_agent.metadata_generator.models import (
    FileCategory,
    ExtractedFragment,
)
from src.agents.metadata_agent.metadata_generator.synthesizer import MetadataSynthesizer


LLM_SYNTH_RESPONSE = json.dumps({
    "title": "Attention Is All You Need Revisited",
    "idea_hypothesis": "We hypothesize that attention-only architectures can match or exceed CNN performance.",
    "method": "We use a 12-layer transformer with multi-head self-attention and positional encoding.",
    "data": "Experiments are conducted on ImageNet (1.2M images) and CIFAR-10 (60K images).",
    "experiments": "Our model achieves 95.2% top-1 accuracy on ImageNet, outperforming ResNet-50 by 2.1%.",
})


def _mock_llm_response(content: str):
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=500, completion_tokens=200, total_tokens=700)
    return resp


@pytest.fixture
def sample_fragments():
    return [
        ExtractedFragment(
            source_file="notes.md",
            file_category=FileCategory.TEXT,
            content="We hypothesize that attention-based models outperform CNNs.",
            metadata_field="idea_hypothesis",
            confidence=0.7,
        ),
        ExtractedFragment(
            source_file="model.py",
            file_category=FileCategory.CODE,
            content="class TransformerModel(nn.Module): ...",
            metadata_field="method",
            confidence=0.6,
        ),
        ExtractedFragment(
            source_file="results.csv",
            file_category=FileCategory.DATA,
            content="model,accuracy\nours,0.95\nbaseline,0.88",
            metadata_field="tables",
            confidence=0.75,
            extra={"table_id": "tab:results", "caption": "Results", "file_path": "results.csv", "columns": ["model", "accuracy"]},
        ),
        ExtractedFragment(
            source_file="refs.bib",
            file_category=FileCategory.BIB,
            content='@article{vaswani2017, title={Attention Is All You Need}, year={2017}}',
            metadata_field="references",
            confidence=0.95,
        ),
        ExtractedFragment(
            source_file="fig1.png",
            file_category=FileCategory.IMAGE,
            content="Image file: fig1.png",
            metadata_field="figures",
            confidence=0.7,
            extra={"figure_id": "fig:fig1", "caption": "Fig1", "file_path": "fig1.png"},
        ),
    ]


class TestMetadataSynthesizer:
    @pytest.mark.asyncio
    async def test_synthesize_returns_paper_metadata(self, sample_fragments):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(LLM_SYNTH_RESPONSE)
        )
        synth = MetadataSynthesizer(llm_client=mock_client, model_name="test")

        result = await synth.synthesize(sample_fragments)

        assert isinstance(result, PaperMetaData)
        assert result.idea_hypothesis != ""
        assert result.method != ""
        assert result.data != ""
        assert result.experiments != ""

    @pytest.mark.asyncio
    async def test_references_merged(self, sample_fragments):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(LLM_SYNTH_RESPONSE)
        )
        synth = MetadataSynthesizer(llm_client=mock_client, model_name="test")

        result = await synth.synthesize(sample_fragments)

        assert len(result.references) >= 1
        assert any("vaswani2017" in ref for ref in result.references)

    @pytest.mark.asyncio
    async def test_figures_attached(self, sample_fragments):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(LLM_SYNTH_RESPONSE)
        )
        synth = MetadataSynthesizer(llm_client=mock_client, model_name="test")

        result = await synth.synthesize(sample_fragments)

        assert len(result.figures) >= 1
        assert result.figures[0].id == "fig:fig1"

    @pytest.mark.asyncio
    async def test_tables_attached(self, sample_fragments):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(LLM_SYNTH_RESPONSE)
        )
        synth = MetadataSynthesizer(llm_client=mock_client, model_name="test")

        result = await synth.synthesize(sample_fragments)

        assert len(result.tables) >= 1
        assert result.tables[0].id == "tab:results"

    @pytest.mark.asyncio
    async def test_user_overrides_applied(self, sample_fragments):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(LLM_SYNTH_RESPONSE)
        )
        synth = MetadataSynthesizer(llm_client=mock_client, model_name="test")

        result = await synth.synthesize(
            sample_fragments,
            overrides={"title": "Custom Title", "style_guide": "ICML"},
        )

        assert result.title == "Custom Title"
        assert result.style_guide == "ICML"

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self, sample_fragments):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("LLM down")
        )
        synth = MetadataSynthesizer(llm_client=mock_client, model_name="test")

        result = await synth.synthesize(sample_fragments)

        assert isinstance(result, PaperMetaData)
        # Fallback should concatenate fragments by field
        assert result.idea_hypothesis != "" or result.method != ""
