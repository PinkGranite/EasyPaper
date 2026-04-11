"""
Text file extractor: parse Markdown, plain text, and LaTeX files into fragments.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from ..models import ExtractedFragment, FileCategory
from .base import BaseExtractor

MAX_FRAGMENT_CHARS = 4000
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_TEX_SECTION = re.compile(r"\\(?:section|subsection|chapter)\{([^}]+)\}", re.MULTILINE)


class TextExtractor(BaseExtractor):
    """
    Extract fragments from Markdown, plain text, or LaTeX files.
    Splits by headings/sections where possible.
    """

    def extract(self, file_path: str) -> List[ExtractedFragment]:
        p = Path(file_path)
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return []

        ext = p.suffix.lower()
        if ext in (".md", ".markdown"):
            return self._extract_markdown(p, text)
        elif ext == ".tex":
            return self._extract_latex(p, text)
        else:
            return self._extract_plain(p, text)

    def _extract_markdown(
        self, path: Path, text: str,
    ) -> List[ExtractedFragment]:
        sections = self._split_by_md_headings(text)
        if len(sections) <= 1:
            return self._extract_plain(path, text)

        fragments: List[ExtractedFragment] = []
        for heading, body in sections:
            body = body.strip()
            if not body:
                continue
            field = self._guess_metadata_field(heading)
            fragments.append(
                ExtractedFragment(
                    source_file=path.name,
                    file_category=FileCategory.TEXT,
                    content=body[:MAX_FRAGMENT_CHARS],
                    metadata_field=field,
                    confidence=0.6,
                    extra={"heading": heading},
                )
            )
        return fragments

    def _extract_latex(
        self, path: Path, text: str,
    ) -> List[ExtractedFragment]:
        sections = self._split_by_tex_sections(text)
        if len(sections) <= 1:
            cleaned = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
            cleaned = re.sub(r"\\[a-zA-Z]+", "", cleaned).strip()
            if not cleaned:
                return []
            return [
                ExtractedFragment(
                    source_file=path.name,
                    file_category=FileCategory.TEXT,
                    content=cleaned[:MAX_FRAGMENT_CHARS],
                    metadata_field=None,
                    confidence=0.5,
                )
            ]

        fragments: List[ExtractedFragment] = []
        for heading, body in sections:
            body = body.strip()
            if not body:
                continue
            field = self._guess_metadata_field(heading)
            fragments.append(
                ExtractedFragment(
                    source_file=path.name,
                    file_category=FileCategory.TEXT,
                    content=body[:MAX_FRAGMENT_CHARS],
                    metadata_field=field,
                    confidence=0.6,
                    extra={"heading": heading},
                )
            )
        return fragments

    def _extract_plain(
        self, path: Path, text: str,
    ) -> List[ExtractedFragment]:
        return [
            ExtractedFragment(
                source_file=path.name,
                file_category=FileCategory.TEXT,
                content=text[:MAX_FRAGMENT_CHARS],
                metadata_field=None,
                confidence=0.4,
            )
        ]

    @staticmethod
    def _split_by_md_headings(text: str) -> List[Tuple[str, str]]:
        matches = list(_MD_HEADING.finditer(text))
        if not matches:
            return [("", text)]
        sections: List[Tuple[str, str]] = []
        for i, m in enumerate(matches):
            heading = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append((heading, text[start:end]))
        return sections

    @staticmethod
    def _split_by_tex_sections(text: str) -> List[Tuple[str, str]]:
        matches = list(_TEX_SECTION.finditer(text))
        if not matches:
            return [("", text)]
        sections: List[Tuple[str, str]] = []
        for i, m in enumerate(matches):
            heading = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append((heading, text[start:end]))
        return sections

    @staticmethod
    def _guess_metadata_field(heading: str) -> str | None:
        h = heading.lower()
        if any(k in h for k in ("hypothesis", "idea", "research question", "motivation", "background")):
            return "idea_hypothesis"
        if any(k in h for k in ("method", "approach", "algorithm", "architecture", "model")):
            return "method"
        if any(k in h for k in ("data", "dataset", "material")):
            return "data"
        if any(k in h for k in ("experiment", "result", "evaluation", "finding", "ablation")):
            return "experiments"
        return None
