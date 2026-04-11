"""
Integration tests for the full generate_metadata_from_folder pipeline.
All LLM calls are mocked.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from src.agents.metadata_agent.models import PaperMetaData
from src.agents.metadata_agent.metadata_generator import generate_metadata_from_folder


LLM_SYNTH_RESPONSE = json.dumps({
    "title": "Attention Mechanisms for Classification",
    "idea_hypothesis": "Attention-based models can outperform traditional CNNs.",
    "method": "We employ a 12-layer transformer with multi-head self-attention.",
    "data": "We use CIFAR-10 and ImageNet datasets.",
    "experiments": "Our model achieves 95% accuracy, outperforming baselines.",
})


def _mock_llm_response(content: str):
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=500, completion_tokens=200, total_tokens=700)
    return resp


@pytest.fixture
def research_folder(tmp_path: Path) -> Path:
    """Create a complete research materials folder."""
    # Markdown notes
    (tmp_path / "notes.md").write_text(
        "# Hypothesis\n"
        "We hypothesize that attention outperforms CNNs.\n\n"
        "# Method\n"
        "Transformer with 12 layers.\n",
        encoding="utf-8",
    )
    # BibTeX
    (tmp_path / "refs.bib").write_text(
        '@article{vaswani2017,\n'
        '  title={Attention Is All You Need},\n'
        '  author={Vaswani, A},\n'
        '  year={2017}\n'
        '}\n',
        encoding="utf-8",
    )
    # CSV data
    (tmp_path / "results.csv").write_text(
        "model,accuracy\nours,0.95\nbaseline,0.88\n",
        encoding="utf-8",
    )
    # Image
    (tmp_path / "fig_architecture.png").write_bytes(b"\x89PNG" + b"\x00" * 50)
    return tmp_path


class TestGenerateMetadataFromFolder:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, research_folder: Path):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(LLM_SYNTH_RESPONSE)
        )

        result = await generate_metadata_from_folder(
            folder_path=str(research_folder),
            llm_client=mock_client,
            model_name="test-model",
        )

        assert isinstance(result, PaperMetaData)

    @pytest.mark.asyncio
    async def test_five_fields_populated(self, research_folder: Path):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(LLM_SYNTH_RESPONSE)
        )

        result = await generate_metadata_from_folder(
            folder_path=str(research_folder),
            llm_client=mock_client,
            model_name="test-model",
        )

        assert result.idea_hypothesis != ""
        assert result.method != ""
        assert result.data != ""
        assert result.experiments != ""

    @pytest.mark.asyncio
    async def test_references_from_bib(self, research_folder: Path):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(LLM_SYNTH_RESPONSE)
        )

        result = await generate_metadata_from_folder(
            folder_path=str(research_folder),
            llm_client=mock_client,
            model_name="test-model",
        )

        assert len(result.references) >= 1
        assert any("vaswani2017" in ref for ref in result.references)

    @pytest.mark.asyncio
    async def test_figures_from_images(self, research_folder: Path):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(LLM_SYNTH_RESPONSE)
        )

        result = await generate_metadata_from_folder(
            folder_path=str(research_folder),
            llm_client=mock_client,
            model_name="test-model",
        )

        assert len(result.figures) >= 1
        assert any("architecture" in fig.id for fig in result.figures)

    @pytest.mark.asyncio
    async def test_tables_from_csv(self, research_folder: Path):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(LLM_SYNTH_RESPONSE)
        )

        result = await generate_metadata_from_folder(
            folder_path=str(research_folder),
            llm_client=mock_client,
            model_name="test-model",
        )

        assert len(result.tables) >= 1

    @pytest.mark.asyncio
    async def test_overrides_applied(self, research_folder: Path):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(LLM_SYNTH_RESPONSE)
        )

        result = await generate_metadata_from_folder(
            folder_path=str(research_folder),
            llm_client=mock_client,
            model_name="test-model",
            title="My Custom Title",
            style_guide="NeurIPS",
        )

        assert result.title == "My Custom Title"
        assert result.style_guide == "NeurIPS"

    @pytest.mark.asyncio
    async def test_nonexistent_folder_raises(self):
        mock_client = MagicMock()
        with pytest.raises(FileNotFoundError):
            await generate_metadata_from_folder(
                folder_path="/nonexistent/path",
                llm_client=mock_client,
                model_name="test",
            )
