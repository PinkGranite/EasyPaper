"""
Docling-based paper analysis: download and parse academic PDFs.
- **Description**:
    - PaperDownloader: async download of open-access PDFs from URLs or arXiv.
    - DoclingPaperAnalyzer: wraps Docling DocumentConverter to extract
      structured sections, tables, figures, and references from PDF files.
    - Both components are optional and gracefully degrade when Docling
      is not installed or PDFs are unavailable.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class DoclingPaperResult:
    """
    Structured result from parsing a PDF with Docling.
    - **Description**:
        - Contains the full text, per-section text, extracted tables/figures,
          and bibliography entries parsed from an academic PDF.
    """

    full_text: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    figures: List[Dict[str, Any]] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


class PaperDownloader:
    """
    Async downloader for open-access academic PDFs.
    - **Description**:
        - Downloads from direct ``open_access_pdf`` URLs or constructs
          arXiv PDF URLs from ``arxiv_id``.
        - All failures (timeout, 404, network errors) are handled
          gracefully — returns ``None`` instead of raising.

    - **Args**:
        - `timeout` (float): HTTP request timeout in seconds.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    @staticmethod
    def arxiv_id_to_pdf_url(arxiv_id: str) -> str:
        """
        Convert an arXiv identifier to a direct PDF download URL.
        - **Args**:
            - `arxiv_id` (str): e.g. "2301.12345" or "2301.12345v2".
        - **Returns**:
            - `str`: The arXiv PDF URL.
        """
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    async def download(
        self,
        url: str,
        dest_dir: Path,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Download a single PDF from *url* into *dest_dir*.
        - **Args**:
            - `url` (str): Direct URL to the PDF.
            - `dest_dir` (Path): Target directory (created if missing).
            - `filename` (str, optional): Override filename; defaults to
              a hash-based name derived from the URL.
        - **Returns**:
            - `Optional[Path]`: Path to saved file, or None on failure.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not filename:
            url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
            filename = f"{url_hash}.pdf"
        dest_path = dest_dir / filename

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                dest_path.write_bytes(response.content)
                logger.info("Downloaded PDF: %s -> %s", url[:80], dest_path)
                return dest_path
        except httpx.TimeoutException:
            logger.warning("PDF download timed out: %s", url[:80])
        except httpx.HTTPStatusError as exc:
            logger.warning("PDF download HTTP error %s: %s", exc.response.status_code, url[:80])
        except Exception as exc:
            logger.warning("PDF download failed: %s — %s", url[:80], exc)
        return None

    async def download_for_refs(
        self,
        core_refs: List[Dict[str, Any]],
        dest_dir: Path,
    ) -> Dict[str, Path]:
        """
        Download PDFs for a list of reference dicts.
        - **Description**:
            - Tries ``open_access_pdf`` first; falls back to constructing
              a URL from ``arxiv_id``.
            - Refs without either field are silently skipped.
        - **Args**:
            - `core_refs` (List[Dict]): Reference dicts with optional
              ``open_access_pdf`` and ``arxiv_id`` keys.
            - `dest_dir` (Path): Target download directory.
        - **Returns**:
            - `Dict[str, Path]`: Mapping of ref_id to downloaded file path.
        """
        results: Dict[str, Path] = {}
        for ref in core_refs:
            ref_id = ref.get("ref_id", "")
            if not ref_id:
                continue

            url = ref.get("open_access_pdf")
            if not url:
                arxiv_id = ref.get("arxiv_id")
                if arxiv_id:
                    url = self.arxiv_id_to_pdf_url(arxiv_id)
            if not url:
                continue

            safe_name = re.sub(r"[^\w\-]", "_", ref_id)[:60] + ".pdf"
            path = await self.download(url, dest_dir, filename=safe_name)
            if path is not None:
                results[ref_id] = path

        return results


# ---------------------------------------------------------------------------
# Docling-based PDF parser
# ---------------------------------------------------------------------------

# Section heading patterns for academic papers
_SECTION_PATTERNS = [
    (r"(?i)\b(abstract)\b", "abstract"),
    (r"(?i)\b(introduction)\b", "introduction"),
    (r"(?i)\b(related\s+work|literature\s+review|background)\b", "related_work"),
    (r"(?i)\b(method|methodology|approach|proposed\s+method)\b", "method"),
    (r"(?i)\b(experiment|experimental\s+setup|evaluation)\b", "experiment"),
    (r"(?i)\b(results?|results?\s+and\s+discussion)\b", "results"),
    (r"(?i)\b(discussion)\b", "discussion"),
    (r"(?i)\b(conclusion|concluding\s+remarks)\b", "conclusion"),
    (r"(?i)\b(references|bibliography)\b", "references"),
]


class DoclingPaperAnalyzer:
    """
    Wraps Docling DocumentConverter to parse academic PDFs.
    - **Description**:
        - Extracts structured sections, tables, figures, and references.
        - Docling is imported lazily; an ImportError is raised with a
          helpful message if the package is not installed.

    - **Args**:
        - `config` (Any): DoclingConfig instance controlling pipeline options.
    """

    def __init__(self, config: Any = None) -> None:
        self._config = config

    def parse(self, pdf_path: Path) -> DoclingPaperResult:
        """
        Parse a PDF file and return structured content.
        - **Args**:
            - `pdf_path` (Path): Path to the PDF file.
        - **Returns**:
            - `DoclingPaperResult`: Parsed content with sections, tables, etc.
        """
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
        except ImportError as exc:
            raise ImportError(
                "Docling is required for deep PDF analysis. "
                "Install it with: pip install easypaper[docling]"
            ) from exc

        pipeline_options = PdfPipelineOptions()
        if self._config:
            pipeline_options.do_ocr = getattr(self._config, "do_ocr", False)
            pipeline_options.do_table_structure = getattr(self._config, "do_table_structure", True)
            pipeline_options.images_scale = getattr(self._config, "images_scale", 2.0)
            pipeline_options.document_timeout = getattr(self._config, "document_timeout", 120.0)
            if hasattr(pipeline_options, "do_formula_enrichment"):
                pipeline_options.do_formula_enrichment = getattr(
                    self._config, "do_formula_enrichment", False,
                )
            if hasattr(pipeline_options, "do_code_enrichment"):
                pipeline_options.do_code_enrichment = False

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )

        try:
            conv_result = converter.convert(str(pdf_path))
        except Exception as exc:
            logger.warning("Docling PDF parse failed for %s: %s", pdf_path, exc)
            return DoclingPaperResult()

        full_text = conv_result.document.export_to_markdown()
        sections = self._extract_sections(full_text)

        tables: List[Dict[str, Any]] = []
        try:
            for idx, table in enumerate(conv_result.document.tables):
                tables.append({
                    "index": idx,
                    "html": table.export_to_html(doc=conv_result.document),
                })
        except Exception:
            pass

        figures: List[Dict[str, Any]] = []
        try:
            from docling_core.types.doc import PictureItem
            for element, _level in conv_result.document.iterate_items():
                if isinstance(element, PictureItem):
                    figures.append({"caption": getattr(element, "caption", "")})
        except Exception:
            pass

        return DoclingPaperResult(
            full_text=full_text,
            sections=sections,
            tables=tables,
            figures=figures,
        )

    @staticmethod
    def _extract_sections(markdown_text: str) -> Dict[str, str]:
        """
        Extract named sections from markdown text using heading patterns.
        - **Args**:
            - `markdown_text` (str): Markdown-formatted full text.
        - **Returns**:
            - `Dict[str, str]`: Mapping from canonical section name to text.
        """
        lines = markdown_text.split("\n")
        current_section: Optional[str] = None
        section_lines: Dict[str, List[str]] = {}

        for line in lines:
            heading_match = re.match(r"^#{1,3}\s+(.+)$", line)
            if heading_match:
                heading_text = heading_match.group(1).strip()
                matched_section = None
                for pattern, name in _SECTION_PATTERNS:
                    if re.search(pattern, heading_text):
                        matched_section = name
                        break
                if matched_section:
                    current_section = matched_section
                    section_lines.setdefault(current_section, [])
                    continue

            if current_section is not None:
                section_lines.setdefault(current_section, []).append(line)

        return {k: "\n".join(v).strip() for k, v in section_lines.items() if v}
