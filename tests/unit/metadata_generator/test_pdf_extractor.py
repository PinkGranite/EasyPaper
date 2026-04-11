"""Tests for PDFExtractor (LLM-based, all calls mocked)."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.metadata_agent.metadata_generator.models import FileCategory, ExtractedFragment
from src.agents.metadata_agent.metadata_generator.extractors.pdf_extractor import PDFExtractor


SAMPLE_PDF_TEXT = (
    "Abstract\n"
    "We propose a novel attention mechanism for image classification.\n\n"
    "Introduction\n"
    "Deep learning has achieved remarkable results.\n\n"
    "Method\n"
    "Our model uses multi-head self-attention.\n\n"
    "Results\n"
    "We achieve 95% accuracy on ImageNet.\n"
)

LLM_RESPONSE_JSON = json.dumps({
    "summary": "A paper proposing a novel attention mechanism.",
    "research_background": "Deep learning for image classification.",
    "research_question": "Can attention improve classification accuracy?",
    "research_hypothesis": ["Attention-based models outperform CNNs."],
    "methods": ["Multi-head self-attention with 12 layers."],
    "results": ["95% accuracy on ImageNet."],
    "key_findings": ["Attention outperforms baseline by 5%."],
})


def _mock_llm_response(content: str):
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    return resp


class TestPDFExtractor:
    @pytest.mark.asyncio
    async def test_extract_with_llm(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(LLM_RESPONSE_JSON)
        )
        ext = PDFExtractor(llm_client=mock_client, model_name="test-model")

        with patch(
            "src.agents.metadata_agent.metadata_generator.extractors.pdf_extractor._extract_pdf_text",
            return_value=SAMPLE_PDF_TEXT,
        ):
            fragments = await ext.extract_async("fake.pdf")

        assert len(fragments) >= 3
        assert all(f.file_category == FileCategory.PDF for f in fragments)
        fields = {f.metadata_field for f in fragments}
        assert "idea_hypothesis" in fields
        assert "method" in fields
        assert "experiments" in fields

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("LLM unavailable")
        )
        ext = PDFExtractor(llm_client=mock_client, model_name="test-model")

        with patch(
            "src.agents.metadata_agent.metadata_generator.extractors.pdf_extractor._extract_pdf_text",
            return_value=SAMPLE_PDF_TEXT,
        ):
            fragments = await ext.extract_async("fake.pdf")

        # Fallback should still produce at least one fragment with the raw text
        assert len(fragments) >= 1
        assert fragments[0].file_category == FileCategory.PDF

    @pytest.mark.asyncio
    async def test_high_confidence_for_llm_results(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(LLM_RESPONSE_JSON)
        )
        ext = PDFExtractor(llm_client=mock_client, model_name="test-model")

        with patch(
            "src.agents.metadata_agent.metadata_generator.extractors.pdf_extractor._extract_pdf_text",
            return_value=SAMPLE_PDF_TEXT,
        ):
            fragments = await ext.extract_async("fake.pdf")

        for f in fragments:
            assert f.confidence >= 0.7

    def test_sync_extract_returns_empty(self):
        """Sync extract without LLM client returns empty."""
        ext = PDFExtractor()
        fragments = ext.extract("fake.pdf")
        assert fragments == []
