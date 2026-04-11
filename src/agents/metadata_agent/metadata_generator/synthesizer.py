"""
MetadataSynthesizer: merge extracted fragments into a complete PaperMetaData
using an LLM to produce coherent five-field prose.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from ..models import PaperMetaData, FigureSpec, TableSpec
from .models import ExtractedFragment

logger = logging.getLogger(__name__)

SYNTH_SYSTEM_PROMPT = (
    "You are an expert research analyst. You are given a collection of extracted "
    "fragments from various research materials (notes, code, data, papers). "
    "Synthesize them into structured metadata for academic paper generation.\n\n"
    "Return a JSON object with exactly these fields:\n"
    "- title (str): A suitable paper title\n"
    "- idea_hypothesis (str): The core research idea or hypothesis (comprehensive paragraph)\n"
    "- method (str): The methodology, approach, or algorithm (comprehensive paragraph)\n"
    "- data (str): The datasets, data sources, or materials used (concise paragraph)\n"
    "- experiments (str): The experimental results, findings, and analysis (comprehensive paragraph)\n\n"
    "Synthesize information from multiple fragments into coherent paragraphs. "
    "Do NOT simply list fragment contents; weave them into a unified narrative. "
    "Return ONLY valid JSON, no markdown fences."
)

FIVE_FIELDS = ("idea_hypothesis", "method", "data", "experiments")


class MetadataSynthesizer:
    """
    Merge a list of ExtractedFragment objects into one PaperMetaData
    using LLM synthesis with rule-based fallback.

    - **Args**:
        - `llm_client` (optional): OpenAI-compatible client.
        - `model_name` (str): Model for chat completions.
    """

    def __init__(
        self,
        llm_client: Any = None,
        model_name: str = "",
    ) -> None:
        self._client = llm_client
        self._model = model_name

    async def synthesize(
        self,
        fragments: List[ExtractedFragment],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> PaperMetaData:
        """
        Synthesize fragments into PaperMetaData.

        - **Args**:
            - `fragments` (List[ExtractedFragment]): All extracted fragments.
            - `overrides` (dict, optional): User-provided field overrides
              (e.g. title, style_guide, template_path).

        - **Returns**:
            - `PaperMetaData`: The synthesized metadata.
        """
        overrides = overrides or {}

        refs = self._collect_references(fragments)
        figures = self._collect_figures(fragments)
        tables = self._collect_tables(fragments)

        text_fragments = [
            f for f in fragments
            if f.metadata_field not in ("references", "figures", "tables")
        ]

        try:
            five_fields = await self._llm_synthesize(text_fragments)
        except Exception as e:
            logger.warning("LLM synthesis failed, using rule-based fallback: %s", e)
            five_fields = self._fallback_merge(text_fragments)

        title = overrides.pop("title", None) or five_fields.get("title", "Untitled Paper")

        metadata = PaperMetaData(
            title=title,
            idea_hypothesis=five_fields.get("idea_hypothesis", ""),
            method=five_fields.get("method", ""),
            data=five_fields.get("data", ""),
            experiments=five_fields.get("experiments", ""),
            references=refs,
            figures=figures,
            tables=tables,
        )

        for key, value in overrides.items():
            if hasattr(metadata, key) and value is not None:
                setattr(metadata, key, value)

        return metadata

    async def _llm_synthesize(self, fragments: List[ExtractedFragment]) -> dict:
        grouped = self._group_by_field(fragments)
        context_parts: List[str] = []

        for field in FIVE_FIELDS:
            items = grouped.get(field, [])
            if items:
                block = "\n---\n".join(f"[{f.source_file}] {f.content}" for f in items)
                context_parts.append(f"=== {field.upper()} FRAGMENTS ===\n{block}")

        ungrouped = grouped.get(None, [])
        if ungrouped:
            block = "\n---\n".join(f"[{f.source_file}] {f.content}" for f in ungrouped)
            context_parts.append(f"=== GENERAL FRAGMENTS ===\n{block}")

        user_msg = "\n\n".join(context_parts) if context_parts else "(no fragments)"

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYNTH_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        return json.loads(raw)

    @staticmethod
    def _fallback_merge(fragments: List[ExtractedFragment]) -> dict:
        grouped = MetadataSynthesizer._group_by_field(fragments)
        result: Dict[str, str] = {}
        for field in FIVE_FIELDS:
            items = grouped.get(field, [])
            if items:
                result[field] = "\n\n".join(f.content for f in items)
            else:
                result[field] = ""

        if not any(result.get(f) for f in FIVE_FIELDS):
            ungrouped = grouped.get(None, [])
            if ungrouped:
                combined = "\n\n".join(f.content for f in ungrouped)
                result["idea_hypothesis"] = combined

        return result

    @staticmethod
    def _group_by_field(
        fragments: List[ExtractedFragment],
    ) -> Dict[Optional[str], List[ExtractedFragment]]:
        groups: Dict[Optional[str], List[ExtractedFragment]] = defaultdict(list)
        for f in fragments:
            groups[f.metadata_field].append(f)
        return dict(groups)

    @staticmethod
    def _collect_references(fragments: List[ExtractedFragment]) -> List[str]:
        refs: List[str] = []
        seen: set = set()
        for f in fragments:
            if f.metadata_field == "references":
                key = f.content.strip()
                if key and key not in seen:
                    refs.append(key)
                    seen.add(key)
        return refs

    @staticmethod
    def _collect_figures(fragments: List[ExtractedFragment]) -> List[FigureSpec]:
        figures: List[FigureSpec] = []
        seen_ids: set = set()
        for f in fragments:
            if f.metadata_field == "figures" and f.extra.get("figure_id"):
                fid = f.extra["figure_id"]
                if fid in seen_ids:
                    continue
                seen_ids.add(fid)
                figures.append(FigureSpec(
                    id=fid,
                    caption=f.extra.get("caption", ""),
                    description=f.content,
                    file_path=f.extra.get("file_path"),
                ))
        return figures

    @staticmethod
    def _collect_tables(fragments: List[ExtractedFragment]) -> List[TableSpec]:
        tables: List[TableSpec] = []
        seen_ids: set = set()
        for f in fragments:
            if f.metadata_field == "tables" and f.extra.get("table_id"):
                tid = f.extra["table_id"]
                if tid in seen_ids:
                    continue
                seen_ids.add(tid)
                tables.append(TableSpec(
                    id=tid,
                    caption=f.extra.get("caption", ""),
                    description=f.content,
                    file_path=f.extra.get("file_path"),
                    content=f.content,
                ))
        return tables
