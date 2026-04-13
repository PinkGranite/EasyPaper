"""
PDF extractor: extract structured research fragments from academic PDFs.
Uses PyMuPDF for text extraction and LLM for understanding.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional

from ..models import ExtractedFragment, FileCategory
from .base import BaseExtractor

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 48_000

UNDERSTAND_PROMPT = (
    "You are a helpful assistant that helps to understand a research paper.\n"
    "The user will provide you the raw text of the paper.\n"
    "Please summarize the paper from the following aspects:\n"
    "- summary (str): a brief summary of the paper\n"
    "- research_question (str): the core research question\n"
    "- research_hypothesis (list[str]): the hypotheses\n"
    "- methods (list[str]): the methods used\n"
    "- data (str): datasets or data sources used\n"
    "- results (list[str]): the results\n"
    "- key_findings (list[str]): the key findings\n\n"
    "Please output in JSON format. Return ONLY valid JSON, no markdown fences."
)


def _extract_pdf_text(path: str) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz).

    - **Args**:
        - `path` (str): Path to the PDF file.

    - **Returns**:
        - `str`: Extracted text, truncated to MAX_TEXT_CHARS.
    """
    import fitz
    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    text = "\n\n".join(pages)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "\n... [truncated]"
    return text


class PDFExtractor(BaseExtractor):
    """
    Extract structured metadata fragments from PDF files.
    Requires an LLM client for async extraction; sync ``extract()``
    returns empty unless overridden.

    - **Args**:
        - `llm_client` (optional): OpenAI-compatible client for LLM calls.
        - `model_name` (str): Model name for chat completions.
    """

    def __init__(
        self,
        llm_client: Any = None,
        model_name: str = "",
    ) -> None:
        self._client = llm_client
        self._model = model_name

    def extract(self, file_path: str, *, materials_root: str | None = None) -> List[ExtractedFragment]:
        """Sync fallback — returns empty when no LLM client is available."""
        if self._client is None:
            return []
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.extract_async(file_path, materials_root=materials_root)
            )
        return []

    @staticmethod
    def _relative_source_path(file_path: str, materials_root: Optional[str]) -> str:
        """POSIX path relative to *materials_root*, or basename if outside root."""
        p = Path(file_path).resolve()
        if materials_root:
            root = Path(materials_root).resolve()
            try:
                return p.relative_to(root).as_posix()
            except ValueError:
                pass
        return p.name

    async def extract_async(
        self,
        file_path: str,
        *,
        materials_root: Optional[str] = None,
    ) -> List[ExtractedFragment]:
        """
        Extract fragments from a PDF using LLM understanding.

        - **Args**:
            - `file_path` (str): Path to the PDF file.
            - `materials_root` (str, optional): Scan root for relative ``source_file``.

        - **Returns**:
            - `List[ExtractedFragment]`: Extracted fragments.
        """
        rel = self._relative_source_path(file_path, materials_root)
        text = _extract_pdf_text(file_path)
        if not text.strip():
            return []

        try:
            parsed = await self._llm_understand(text)
            return self._build_fragments(rel, parsed)
        except Exception as e:
            logger.warning("LLM PDF extraction failed, using raw fallback: %s", e)
            return self._fallback_fragments(rel, text)

    async def _llm_understand(self, text: str) -> dict:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": UNDERSTAND_PROMPT},
                {"role": "user", "content": f"<paper>\n{text}\n</paper>"},
            ],
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        return json.loads(raw)

    @staticmethod
    def _build_fragments(rel: str, parsed: dict) -> List[ExtractedFragment]:
        fragments: List[ExtractedFragment] = []

        hypothesis_parts = []
        if parsed.get("research_question"):
            hypothesis_parts.append(parsed["research_question"])
        for h in parsed.get("research_hypothesis", []):
            hypothesis_parts.append(h)
        if hypothesis_parts:
            fragments.append(ExtractedFragment(
                source_file=rel,
                file_category=FileCategory.PDF,
                content="\n".join(hypothesis_parts),
                metadata_field="idea_hypothesis",
                confidence=0.85,
            ))

        methods = parsed.get("methods", [])
        if methods:
            fragments.append(ExtractedFragment(
                source_file=rel,
                file_category=FileCategory.PDF,
                content="\n".join(methods),
                metadata_field="method",
                confidence=0.85,
            ))

        data_text = parsed.get("data", "")
        if data_text:
            fragments.append(ExtractedFragment(
                source_file=rel,
                file_category=FileCategory.PDF,
                content=data_text,
                metadata_field="data",
                confidence=0.8,
            ))

        results = parsed.get("results", []) + parsed.get("key_findings", [])
        if results:
            fragments.append(ExtractedFragment(
                source_file=rel,
                file_category=FileCategory.PDF,
                content="\n".join(results),
                metadata_field="experiments",
                confidence=0.85,
            ))

        summary = parsed.get("summary", "")
        if summary:
            fragments.append(ExtractedFragment(
                source_file=rel,
                file_category=FileCategory.PDF,
                content=summary,
                metadata_field=None,
                confidence=0.7,
                extra={"role": "summary"},
            ))

        return fragments

    @staticmethod
    def _fallback_fragments(rel: str, text: str) -> List[ExtractedFragment]:
        return [
            ExtractedFragment(
                source_file=rel,
                file_category=FileCategory.PDF,
                content=text[:4000],
                metadata_field=None,
                confidence=0.3,
                extra={"role": "raw_fallback"},
            )
        ]
