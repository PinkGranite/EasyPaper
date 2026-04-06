"""
DoclingEnricher - Orchestrates PDF download + Docling parsing for core references.
- **Description**:
    - Downloads open-access PDFs for user-provided core references.
    - Parses each PDF with DoclingPaperAnalyzer to extract structured
      sections (method, results, conclusion, etc.).
    - Attaches the parsed sections as ``docling_sections`` on each
      reference dict, enriching the data available to CoreRefAnalyzer.
    - Gracefully skips references without download URLs or when
      download/parsing fails — existing ref data is never lost.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...config.schema import DoclingConfig
from .docling_analyzer import DoclingPaperAnalyzer, DoclingPaperResult, PaperDownloader

logger = logging.getLogger(__name__)


class DoclingEnricher:
    """
    Orchestrates download + parse to enrich core reference dicts.
    - **Description**:
        - For each core ref with ``open_access_pdf`` or ``arxiv_id``,
          downloads the PDF and parses it with Docling.
        - Attaches ``docling_sections`` dict to the ref.
        - All failures are non-fatal: the ref continues with its
          existing abstract-only data.

    - **Args**:
        - `config` (DoclingConfig): Configuration for Docling pipeline.
    """

    def __init__(self, config: DoclingConfig) -> None:
        self._config = config
        self._downloader = PaperDownloader(timeout=config.download_timeout)
        self._analyzer = DoclingPaperAnalyzer(config=config)

    async def enrich_core_refs(
        self,
        core_refs: List[Dict[str, Any]],
        temp_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Enrich core references with full-text sections from PDFs.
        - **Description**:
            - Downloads and parses PDFs for refs that have download URLs.
            - Attaches ``docling_sections`` and ``docling_full_text``
              to each successfully parsed ref.
            - Returns the same list (mutated in place) for convenience.

        - **Args**:
            - `core_refs` (List[Dict]): Reference dicts from ReferencePool.
            - `temp_dir` (Path): Directory for temporary PDF files.

        - **Returns**:
            - `List[Dict]`: The enriched reference list.
        """
        temp_dir.mkdir(parents=True, exist_ok=True)
        enriched_count = 0

        for ref in core_refs:
            ref_id = ref.get("ref_id", "")
            if not ref_id:
                continue

            pdf_path = await self._download_pdf(ref, temp_dir)
            if pdf_path is None or not pdf_path.exists():
                continue

            try:
                result = self._parse_pdf(pdf_path)
            except Exception as exc:
                logger.warning(
                    "Docling parse failed for %s: %s", ref_id, exc,
                )
                continue

            if not result.sections and not result.full_text:
                continue

            ref["docling_sections"] = result.sections
            ref["docling_full_text"] = result.full_text
            enriched_count += 1
            logger.info(
                "Docling enriched ref %s: %d sections extracted",
                ref_id, len(result.sections),
            )

        logger.info(
            "DoclingEnricher: enriched %d / %d core refs",
            enriched_count, len(core_refs),
        )
        return core_refs

    async def _download_pdf(
        self,
        ref: Dict[str, Any],
        dest_dir: Path,
    ) -> Optional[Path]:
        """
        Download PDF for a single reference dict.
        - **Args**:
            - `ref` (Dict): Reference dict with optional url fields.
            - `dest_dir` (Path): Target directory.
        - **Returns**:
            - `Optional[Path]`: Path to downloaded file, or None.
        """
        url = ref.get("open_access_pdf")
        if not url:
            arxiv_id = ref.get("arxiv_id")
            if arxiv_id:
                url = PaperDownloader.arxiv_id_to_pdf_url(arxiv_id)
        if not url:
            return None

        ref_id = ref.get("ref_id", "unknown")
        import re
        safe_name = re.sub(r"[^\w\-]", "_", ref_id)[:60] + ".pdf"
        return await self._downloader.download(url, dest_dir, filename=safe_name)

    def _parse_pdf(self, pdf_path: Path) -> DoclingPaperResult:
        """
        Parse a downloaded PDF with Docling.
        - **Args**:
            - `pdf_path` (Path): Path to the PDF file.
        - **Returns**:
            - `DoclingPaperResult`: Parsed content.
        """
        return self._analyzer.parse(pdf_path)

    def cleanup(self, temp_dir: Path) -> None:
        """
        Remove the temporary download directory.
        - **Args**:
            - `temp_dir` (Path): Directory to remove.
        """
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info("Cleaned up docling temp dir: %s", temp_dir)
