"""Tests for PaperDownloader and DoclingPaperAnalyzer (TDD: RED first)."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest

from src.agents.shared.docling_analyzer import DoclingPaperResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Temporary directory for downloads."""
    dl = tmp_path / "downloads"
    dl.mkdir()
    return dl


# ---------------------------------------------------------------------------
# PaperDownloader tests
# ---------------------------------------------------------------------------

class TestPaperDownloader:
    """Tests for PaperDownloader."""

    def _make_downloader(self, timeout: float = 10.0):
        from src.agents.shared.docling_analyzer import PaperDownloader
        return PaperDownloader(timeout=timeout)

    def test_arxiv_id_to_pdf_url(self):
        from src.agents.shared.docling_analyzer import PaperDownloader
        assert PaperDownloader.arxiv_id_to_pdf_url("2301.12345") == "https://arxiv.org/pdf/2301.12345.pdf"
        assert PaperDownloader.arxiv_id_to_pdf_url("2301.12345v2") == "https://arxiv.org/pdf/2301.12345v2.pdf"

    @pytest.mark.asyncio
    async def test_download_returns_path_on_success(self, tmp_dir: Path):
        dl = self._make_downloader()
        fake_pdf_bytes = b"%PDF-1.4 fake content"

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = fake_pdf_bytes
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            client_instance.get = AsyncMock(return_value=mock_response)
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            result = await dl.download("https://arxiv.org/pdf/2301.12345.pdf", tmp_dir)

        assert result is not None
        assert result.exists()
        assert result.read_bytes() == fake_pdf_bytes

    @pytest.mark.asyncio
    async def test_download_returns_none_on_timeout(self, tmp_dir: Path):
        dl = self._make_downloader(timeout=0.1)

        with patch("httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            import httpx
            client_instance.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            result = await dl.download("https://example.com/paper.pdf", tmp_dir)

        assert result is None

    @pytest.mark.asyncio
    async def test_download_returns_none_on_404(self, tmp_dir: Path):
        dl = self._make_downloader()

        mock_response = AsyncMock()
        mock_response.status_code = 404
        import httpx
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("Not Found", request=MagicMock(), response=mock_response)
        )

        with patch("httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            client_instance.get = AsyncMock(return_value=mock_response)
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            result = await dl.download("https://example.com/missing.pdf", tmp_dir)

        assert result is None

    @pytest.mark.asyncio
    async def test_download_for_refs_skips_refs_without_url(self, tmp_dir: Path):
        dl = self._make_downloader()

        refs = [
            {"ref_id": "smith2020", "title": "Some paper"},
            {"ref_id": "jones2021", "title": "Another paper", "open_access_pdf": "https://example.com/paper.pdf"},
        ]

        fake_pdf_bytes = b"%PDF-1.4 fake content"
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = fake_pdf_bytes
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            client_instance.get = AsyncMock(return_value=mock_response)
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            result = await dl.download_for_refs(refs, tmp_dir)

        assert "smith2020" not in result
        assert "jones2021" in result
        assert result["jones2021"].exists()

    @pytest.mark.asyncio
    async def test_download_for_refs_uses_arxiv_id_fallback(self, tmp_dir: Path):
        dl = self._make_downloader()

        refs = [
            {"ref_id": "arxiv2023", "title": "ArXiv paper", "arxiv_id": "2301.99999"},
        ]

        fake_pdf_bytes = b"%PDF-1.4 fake content"
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = fake_pdf_bytes
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            client_instance.get = AsyncMock(return_value=mock_response)
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            result = await dl.download_for_refs(refs, tmp_dir)

        assert "arxiv2023" in result
        # Verify arXiv URL was constructed
        call_args = client_instance.get.call_args
        assert "arxiv.org/pdf/2301.99999" in str(call_args)


# ---------------------------------------------------------------------------
# DoclingPaperAnalyzer tests
# ---------------------------------------------------------------------------

class TestDoclingPaperAnalyzer:
    """Tests for DoclingPaperAnalyzer (mocks docling imports)."""

    def test_extract_sections_from_markdown(self):
        from src.agents.shared.docling_analyzer import DoclingPaperAnalyzer

        md = (
            "# Abstract\n\nThis is the abstract.\n\n"
            "# Introduction\n\nIntro text here.\n\n"
            "## Related Work\n\nSome prior work.\n\n"
            "# Method\n\nOur approach is novel.\n\n"
            "## Experiment\n\nWe ran experiments.\n\n"
            "# Results\n\nTable 1 shows ...\n\n"
            "# Conclusion\n\nWe conclude that ...\n"
        )
        sections = DoclingPaperAnalyzer._extract_sections(md)
        assert "abstract" in sections
        assert "introduction" in sections
        assert "related_work" in sections
        assert "method" in sections
        assert "experiment" in sections
        assert "results" in sections
        assert "conclusion" in sections
        assert "abstract" in sections
        assert "This is the abstract." in sections["abstract"]
        assert "Our approach is novel." in sections["method"]

    def test_extract_sections_handles_empty(self):
        from src.agents.shared.docling_analyzer import DoclingPaperAnalyzer

        sections = DoclingPaperAnalyzer._extract_sections("")
        assert sections == {}

    def test_parse_returns_structured_result_mock(self):
        """Mock the entire docling import chain to test parse() logic."""
        from src.agents.shared.docling_analyzer import DoclingPaperAnalyzer

        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = (
            "# Abstract\n\nTest abstract.\n\n"
            "# Method\n\nTest method.\n"
        )
        mock_doc.tables = []
        mock_doc.iterate_items.return_value = []

        mock_conv_result = MagicMock()
        mock_conv_result.document = mock_doc

        mock_converter_cls = MagicMock()
        mock_converter_instance = MagicMock()
        mock_converter_instance.convert.return_value = mock_conv_result
        mock_converter_cls.return_value = mock_converter_instance

        mock_pdf_format_option = MagicMock()
        mock_input_format = MagicMock()
        mock_input_format.PDF = "pdf"

        mock_pipeline_options_cls = MagicMock()
        mock_pipeline_options = MagicMock()
        mock_pipeline_options_cls.return_value = mock_pipeline_options

        with patch.dict("sys.modules", {
            "docling": MagicMock(),
            "docling.document_converter": MagicMock(
                DocumentConverter=mock_converter_cls,
                PdfFormatOption=mock_pdf_format_option,
            ),
            "docling.datamodel": MagicMock(),
            "docling.datamodel.base_models": MagicMock(InputFormat=mock_input_format),
            "docling.datamodel.pipeline_options": MagicMock(
                PdfPipelineOptions=mock_pipeline_options_cls,
            ),
            "docling_core": MagicMock(),
            "docling_core.types": MagicMock(),
            "docling_core.types.doc": MagicMock(PictureItem=type("PictureItem", (), {})),
        }):
            analyzer = DoclingPaperAnalyzer(config=None)
            result = analyzer.parse(Path("/fake/paper.pdf"))

        assert isinstance(result, DoclingPaperResult)
        assert "Test abstract." in result.sections.get("abstract", "")
        assert "Test method." in result.sections.get("method", "")
        assert result.full_text != ""

    def test_parse_handles_conversion_error(self):
        """When docling conversion raises, return empty result."""
        from src.agents.shared.docling_analyzer import DoclingPaperAnalyzer

        mock_converter_cls = MagicMock()
        mock_converter_instance = MagicMock()
        mock_converter_instance.convert.side_effect = RuntimeError("Bad PDF")
        mock_converter_cls.return_value = mock_converter_instance

        mock_pipeline_options_cls = MagicMock()
        mock_pipeline_options_cls.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "docling": MagicMock(),
            "docling.document_converter": MagicMock(
                DocumentConverter=mock_converter_cls,
                PdfFormatOption=MagicMock(),
            ),
            "docling.datamodel": MagicMock(),
            "docling.datamodel.base_models": MagicMock(InputFormat=MagicMock()),
            "docling.datamodel.pipeline_options": MagicMock(
                PdfPipelineOptions=mock_pipeline_options_cls,
            ),
        }):
            analyzer = DoclingPaperAnalyzer(config=None)
            result = analyzer.parse(Path("/fake/bad.pdf"))

        assert isinstance(result, DoclingPaperResult)
        assert result.full_text == ""
        assert result.sections == {}

    def test_dataclass_defaults(self):
        result = DoclingPaperResult()
        assert result.full_text == ""
        assert result.sections == {}
        assert result.tables == []
        assert result.figures == []
        assert result.references == []
