"""
Code file extractor: analyse source code files for method/experiment signals.
Reuses keyword scoring from CodeContextBuilder.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from ..models import ExtractedFragment, FileCategory
from .base import BaseExtractor

METHOD_KEYWORDS = (
    "model", "algorithm", "architecture", "module", "class ", "def ",
    "forward", "inference", "encode", "decode", "optimizer", "pipeline",
)
EXPERIMENT_KEYWORDS = (
    "train", "eval", "experiment", "ablation", "metric", "dataset",
    "benchmark", "config", "seed", "reproduce", "hyperparameter", "validation",
)
RESULT_KEYWORDS = (
    "result", "analysis", "plot", "table", "figure", "report",
    "compare", "improvement", "error", "accuracy", "f1", "auc",
)

MAX_SNIPPET_CHARS = 1500


def _score(text: str, keywords: Tuple[str, ...]) -> int:
    return sum(text.count(k) for k in keywords)


def _extract_symbols(text: str, ext: str, max_items: int = 12) -> List[str]:
    symbols: List[str] = []
    if ext in {".py", ".r"}:
        symbols.extend(re.findall(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", text, re.M))
        symbols.extend(re.findall(r"^\s*class\s+([A-Za-z_]\w*)\s*[\(:]", text, re.M))
    elif ext in {".c", ".cc", ".cpp", ".h", ".hpp"}:
        symbols.extend(re.findall(r"\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{", text))
    return symbols[:max_items]


class CodeExtractor(BaseExtractor):
    """
    Extract method / experiment evidence from source code files using
    keyword scoring (no LLM).
    """

    def extract(self, file_path: str) -> List[ExtractedFragment]:
        p = Path(file_path)
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return []

        ext = p.suffix.lower()
        lower = text.lower()
        m_score = _score(lower, METHOD_KEYWORDS)
        e_score = _score(lower, EXPERIMENT_KEYWORDS)
        r_score = _score(lower, RESULT_KEYWORDS)

        if m_score + e_score + r_score == 0:
            return []

        symbols = _extract_symbols(text, ext)
        snippet = "\n".join(text.splitlines()[:20])[:MAX_SNIPPET_CHARS]

        exp_combined = e_score + r_score
        dominant = max(
            [("method", m_score), ("experiments", exp_combined)],
            key=lambda x: x[1],
        )[0]

        total = m_score + e_score + r_score
        confidence = round(min(0.85, 0.35 + 0.04 * total), 2)

        return [
            ExtractedFragment(
                source_file=p.name,
                file_category=FileCategory.CODE,
                content=snippet,
                metadata_field=dominant,
                confidence=confidence,
                extra={"symbols": symbols, "scores": {"method": m_score, "experiment": e_score, "result": r_score}},
            )
        ]
