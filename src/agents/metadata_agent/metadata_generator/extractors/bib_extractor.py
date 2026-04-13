"""
BibTeX file extractor: parse .bib files into individual reference entries.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from ..models import ExtractedFragment, FileCategory
from .base import BaseExtractor

_ENTRY_START = re.compile(r"^\s*@\w+\s*\{", re.MULTILINE)


class BibExtractor(BaseExtractor):
    """
    Parse a .bib file and return one ExtractedFragment per BibTeX entry.
    Pure rule-based -- no LLM calls.
    """

    def extract(self, file_path: str, *, materials_root: str | None = None) -> List[ExtractedFragment]:
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return []

        entries = self._split_entries(text)
        fragments: List[ExtractedFragment] = []
        rel = Path(file_path).name

        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            fragments.append(
                ExtractedFragment(
                    source_file=rel,
                    file_category=FileCategory.BIB,
                    content=entry,
                    metadata_field="references",
                    confidence=0.95,
                )
            )
        return fragments

    @staticmethod
    def _split_entries(text: str) -> List[str]:
        """Split raw .bib text into individual entry strings."""
        entries: List[str] = []
        starts = [m.start() for m in _ENTRY_START.finditer(text)]
        if not starts:
            return []
        for i, s in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(text)
            raw = text[s:end].strip()
            # Trim trailing whitespace / comments after closing brace
            depth = 0
            cut = len(raw)
            for j, ch in enumerate(raw):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        cut = j + 1
                        break
            entries.append(raw[:cut])
        return entries
