"""Tests for DoclingEnricher (TDD: RED first)."""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.shared.docling_analyzer import DoclingPaperResult
from src.config.schema import DoclingConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def docling_cfg() -> DoclingConfig:
    return DoclingConfig(enabled=True, cleanup_after_analysis=True)


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    dl = tmp_path / "docling_tmp"
    dl.mkdir()
    return dl


@pytest.fixture
def sample_core_refs():
    return [
        {
            "ref_id": "smith2020",
            "title": "Deep Learning for Vision",
            "abstract": "Short abstract.",
            "open_access_pdf": "https://example.com/smith2020.pdf",
        },
        {
            "ref_id": "jones2021",
            "title": "Robustness in Neural Networks",
            "abstract": "Another abstract.",
            "arxiv_id": "2101.54321",
        },
        {
            "ref_id": "nopdf2022",
            "title": "No PDF Available",
            "abstract": "No url at all.",
        },
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDoclingEnricher:
    """Tests for DoclingEnricher orchestration."""

    @pytest.mark.asyncio
    async def test_enrich_attaches_sections_to_ref(self, docling_cfg, tmp_dir, sample_core_refs):
        from src.agents.shared.docling_enricher import DoclingEnricher

        fake_result = DoclingPaperResult(
            full_text="# Method\n\nDeep method.\n\n# Results\n\nGood results.",
            sections={"method": "Deep method.", "results": "Good results."},
            tables=[],
            figures=[],
        )

        with patch.object(DoclingEnricher, "_download_pdf", new_callable=AsyncMock) as mock_dl, \
             patch.object(DoclingEnricher, "_parse_pdf") as mock_parse:
            mock_dl.side_effect = lambda ref, dest: (
                tmp_dir / f"{ref['ref_id']}.pdf"
                if ref.get("open_access_pdf") or ref.get("arxiv_id")
                else None
            )
            # Create fake PDF files so parse can "find" them
            for ref in sample_core_refs:
                if ref.get("open_access_pdf") or ref.get("arxiv_id"):
                    (tmp_dir / f"{ref['ref_id']}.pdf").write_bytes(b"fake")

            mock_parse.return_value = fake_result

            enricher = DoclingEnricher(docling_cfg)
            enriched = await enricher.enrich_core_refs(sample_core_refs, tmp_dir)

        assert len(enriched) == 3
        # smith2020 and jones2021 should have docling_sections
        smith = next(r for r in enriched if r["ref_id"] == "smith2020")
        assert "docling_sections" in smith
        assert smith["docling_sections"]["method"] == "Deep method."
        assert smith["docling_sections"]["results"] == "Good results."

        jones = next(r for r in enriched if r["ref_id"] == "jones2021")
        assert "docling_sections" in jones

    @pytest.mark.asyncio
    async def test_enrich_skips_refs_without_pdf_url(self, docling_cfg, tmp_dir, sample_core_refs):
        from src.agents.shared.docling_enricher import DoclingEnricher

        with patch.object(DoclingEnricher, "_download_pdf", new_callable=AsyncMock) as mock_dl, \
             patch.object(DoclingEnricher, "_parse_pdf") as mock_parse:
            mock_dl.return_value = None
            mock_parse.return_value = DoclingPaperResult()

            enricher = DoclingEnricher(docling_cfg)
            enriched = await enricher.enrich_core_refs(sample_core_refs, tmp_dir)

        nopdf = next(r for r in enriched if r["ref_id"] == "nopdf2022")
        assert "docling_sections" not in nopdf

    @pytest.mark.asyncio
    async def test_enrich_fallback_on_download_failure(self, docling_cfg, tmp_dir, sample_core_refs):
        from src.agents.shared.docling_enricher import DoclingEnricher

        with patch.object(DoclingEnricher, "_download_pdf", new_callable=AsyncMock) as mock_dl:
            mock_dl.return_value = None

            enricher = DoclingEnricher(docling_cfg)
            enriched = await enricher.enrich_core_refs(sample_core_refs, tmp_dir)

        # All refs should be unchanged (no docling_sections added)
        for ref in enriched:
            assert "docling_sections" not in ref

    def test_cleanup_removes_temp_dir(self, docling_cfg, tmp_dir):
        from src.agents.shared.docling_enricher import DoclingEnricher

        (tmp_dir / "test.pdf").write_bytes(b"data")
        assert tmp_dir.exists()

        enricher = DoclingEnricher(docling_cfg)
        enricher.cleanup(tmp_dir)

        assert not tmp_dir.exists()

    @pytest.mark.asyncio
    async def test_enrich_preserves_existing_ref_fields(self, docling_cfg, tmp_dir):
        from src.agents.shared.docling_enricher import DoclingEnricher

        refs = [
            {
                "ref_id": "test2020",
                "title": "Test Paper",
                "abstract": "Original abstract",
                "year": 2020,
                "bibtex": "@article{test2020, ...}",
                "open_access_pdf": "https://example.com/test.pdf",
            },
        ]

        fake_result = DoclingPaperResult(
            full_text="full text",
            sections={"method": "method text"},
        )

        with patch.object(DoclingEnricher, "_download_pdf", new_callable=AsyncMock) as mock_dl, \
             patch.object(DoclingEnricher, "_parse_pdf") as mock_parse:
            mock_dl.return_value = tmp_dir / "test.pdf"
            (tmp_dir / "test.pdf").write_bytes(b"fake")
            mock_parse.return_value = fake_result

            enricher = DoclingEnricher(docling_cfg)
            enriched = await enricher.enrich_core_refs(refs, tmp_dir)

        ref = enriched[0]
        assert ref["title"] == "Test Paper"
        assert ref["abstract"] == "Original abstract"
        assert ref["year"] == 2020
        assert ref["bibtex"] == "@article{test2020, ...}"
        assert ref["docling_sections"]["method"] == "method text"
