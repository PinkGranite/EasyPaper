"""
Data file extractor: parse CSV / JSON files into table-like fragments.
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import List

from ..models import ExtractedFragment, FileCategory
from .base import BaseExtractor

MAX_PREVIEW_ROWS = 5
MAX_CONTENT_CHARS = 2000


class DataExtractor(BaseExtractor):
    """
    Extract structured previews from data files (CSV, JSON).
    Produces fragments suitable for mapping to ``TableSpec`` or ``data`` metadata.
    """

    def extract(self, file_path: str) -> List[ExtractedFragment]:
        p = Path(file_path)
        ext = p.suffix.lower()
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return []

        if ext in (".csv", ".tsv"):
            return self._extract_csv(p, text, ext)
        elif ext in (".json", ".jsonl"):
            return self._extract_json(p, text)
        return []

    def _extract_csv(
        self, path: Path, text: str, ext: str,
    ) -> List[ExtractedFragment]:
        delimiter = "\t" if ext == ".tsv" else ","
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) > MAX_PREVIEW_ROWS + 1:
                break

        if not rows:
            return []

        header = rows[0]
        preview_rows = rows[1 : MAX_PREVIEW_ROWS + 1]
        preview_lines = [delimiter.join(header)]
        for r in preview_rows:
            preview_lines.append(delimiter.join(r))
        content = "\n".join(preview_lines)

        stem = path.stem
        table_id = "tab:" + re.sub(r"[^a-z0-9_]", "_", stem.lower())
        caption = stem.replace("_", " ").replace("-", " ").title()

        return [
            ExtractedFragment(
                source_file=path.name,
                file_category=FileCategory.DATA,
                content=content[:MAX_CONTENT_CHARS],
                metadata_field="tables",
                confidence=0.75,
                extra={
                    "table_id": table_id,
                    "caption": caption,
                    "columns": header,
                    "num_rows": len(rows) - 1,
                    "file_path": path.name,
                },
            )
        ]

    def _extract_json(
        self, path: Path, text: str,
    ) -> List[ExtractedFragment]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []

        preview = json.dumps(data, indent=2, ensure_ascii=False)[:MAX_CONTENT_CHARS]
        return [
            ExtractedFragment(
                source_file=path.name,
                file_category=FileCategory.DATA,
                content=preview,
                metadata_field="data",
                confidence=0.6,
                extra={"format": "json"},
            )
        ]
