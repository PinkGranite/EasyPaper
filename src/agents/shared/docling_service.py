"""
DoclingService — standalone facade for PDF download, parse, and ref enrichment.
- **Description**:
    - Unifies PaperDownloader, DoclingPaperAnalyzer, and DoclingEnricher
      behind a single high-level interface.
    - Supports three usage modes:
        1. Python import: ``svc = DoclingService(config); result = svc.parse_pdf(path)``
        2. SDK: ``ep.parse_pdf(path)`` / ``ep.download_and_parse(url)``
        3. FastAPI: ``POST /docling/parse``, ``POST /docling/download-and-parse``
    - All methods are self-contained — callers never need to manage
      temporary directories or interact with low-level components.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .docling_analyzer import DoclingPaperAnalyzer, DoclingPaperResult, PaperDownloader

logger = logging.getLogger(__name__)


class DoclingService:
    """
    High-level Docling operations: parse, download-and-parse, enrich refs.
    - **Description**:
        - Wraps the low-level PaperDownloader and DoclingPaperAnalyzer into
          a clean service layer that any caller (SDK, API, CLI) can use.
        - Manages temporary files internally when needed.

    - **Args**:
        - `config` (Any, optional): A DoclingConfig-like object.  If None,
          sensible defaults are used.
    """

    def __init__(self, config: Any = None) -> None:
        self._config = config
        self._analyzer = DoclingPaperAnalyzer(config=config)
        timeout = getattr(config, "download_timeout", 30.0) if config else 30.0
        self._downloader = PaperDownloader(timeout=timeout)

    def parse_pdf(self, pdf_path: str | Path) -> DoclingPaperResult:
        """
        Parse a local PDF file into structured sections.
        - **Args**:
            - `pdf_path` (str | Path): Path to a PDF on disk.
        - **Returns**:
            - `DoclingPaperResult`: Full text, sections, tables, figures.
        """
        return self._analyzer.parse(Path(pdf_path))

    async def download_and_parse(
        self,
        url: str,
        *,
        dest_dir: Optional[str | Path] = None,
        cleanup: bool = True,
    ) -> DoclingPaperResult:
        """
        Download a PDF from *url*, parse it, and return structured content.
        - **Description**:
            - If *dest_dir* is given the PDF is saved there and preserved.
            - Otherwise a temp directory is created and cleaned up after parsing
              (unless *cleanup* is False).

        - **Args**:
            - `url` (str): Direct PDF URL or arXiv PDF link.
            - `dest_dir` (str | Path, optional): Directory to save the PDF.
            - `cleanup` (bool): Remove temp dir after parsing (default True).
        - **Returns**:
            - `DoclingPaperResult`: Parsed content.
        """
        use_temp = dest_dir is None
        work_dir = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="docling_"))
        try:
            pdf_path = await self._downloader.download(url, work_dir)
            if pdf_path is None:
                logger.warning("DoclingService: download failed for %s", url[:120])
                return DoclingPaperResult()
            return self._analyzer.parse(pdf_path)
        finally:
            if use_temp and cleanup:
                shutil.rmtree(work_dir, ignore_errors=True)

    async def download_and_parse_arxiv(
        self,
        arxiv_id: str,
        *,
        dest_dir: Optional[str | Path] = None,
        cleanup: bool = True,
    ) -> DoclingPaperResult:
        """
        Convenience wrapper: arXiv ID → download → parse.
        - **Args**:
            - `arxiv_id` (str): e.g. ``"2301.12345"`` or ``"2301.12345v2"``.
            - `dest_dir` (str | Path, optional): Directory to save the PDF.
            - `cleanup` (bool): Remove temp dir after parsing (default True).
        - **Returns**:
            - `DoclingPaperResult`: Parsed content.
        """
        url = PaperDownloader.arxiv_id_to_pdf_url(arxiv_id)
        return await self.download_and_parse(url, dest_dir=dest_dir, cleanup=cleanup)

    async def enrich_refs(
        self,
        refs: List[Dict[str, Any]],
        *,
        dest_dir: Optional[str | Path] = None,
        cleanup: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Batch-enrich reference dicts with Docling full-text analysis.
        - **Description**:
            - For each ref with ``open_access_pdf`` or ``arxiv_id``,
              downloads + parses the PDF and attaches ``docling_full_text``
              and ``docling_sections`` to the ref dict.
            - Refs without URLs are silently skipped.
            - Returns the same list (mutated in place).

        - **Args**:
            - `refs` (List[Dict]): Reference dicts.
            - `dest_dir` (str | Path, optional): Directory for PDF storage.
            - `cleanup` (bool): Remove temp dir after enrichment (default True).
        - **Returns**:
            - `List[Dict]`: The enriched reference list.
        """
        use_temp = dest_dir is None
        work_dir = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="docling_refs_"))
        work_dir.mkdir(parents=True, exist_ok=True)
        enriched = 0

        try:
            for ref in refs:
                ref_id = ref.get("ref_id", "")
                if not ref_id:
                    continue

                url = ref.get("open_access_pdf")
                if not url:
                    arxiv_id = ref.get("arxiv_id")
                    if arxiv_id:
                        url = PaperDownloader.arxiv_id_to_pdf_url(arxiv_id)
                if not url:
                    continue

                import re
                safe_name = re.sub(r"[^\w\-]", "_", ref_id)[:60] + ".pdf"
                pdf_path = await self._downloader.download(url, work_dir, filename=safe_name)
                if pdf_path is None:
                    continue

                try:
                    result = self._analyzer.parse(pdf_path)
                except Exception as exc:
                    logger.warning("DoclingService parse failed for %s: %s", ref_id, exc)
                    continue

                if not result.sections and not result.full_text:
                    continue

                ref["docling_sections"] = result.sections
                ref["docling_full_text"] = result.full_text
                enriched += 1
                logger.info("DoclingService enriched %s: %d sections", ref_id, len(result.sections))

            logger.info("DoclingService: enriched %d / %d refs", enriched, len(refs))
            return refs
        finally:
            if use_temp and cleanup:
                shutil.rmtree(work_dir, ignore_errors=True)

    @staticmethod
    def result_to_dict(result: DoclingPaperResult) -> Dict[str, Any]:
        """
        Convert a DoclingPaperResult to a JSON-serializable dict.
        - **Args**:
            - `result` (DoclingPaperResult): Parsed content.
        - **Returns**:
            - `Dict[str, Any]`: Serializable dict.
        """
        return asdict(result)
