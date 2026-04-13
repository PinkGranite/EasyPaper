"""
MetadataSynthesizer: merge extracted fragments into a complete PaperMetaData
using an LLM to produce coherent five-field prose.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from ..models import PaperMetaData, FigureSpec, TableSpec
from .models import ExtractedFragment

logger = logging.getLogger(__name__)

SYNTH_SYSTEM_PROMPT = (
    "You are an expert research analyst. You are given a collection of extracted "
    "fragments from various research materials (notes, code, data, papers). "
    "Synthesize them into structured metadata for academic paper generation.\n\n"
    "Return a JSON object with these fields:\n"
    "- title (str): A suitable paper title\n"
    "- idea_hypothesis (str): The core research idea or hypothesis (comprehensive paragraph)\n"
    "- method (str): The methodology, approach, or algorithm (comprehensive paragraph)\n"
    "- data (str): The datasets, data sources, or materials used (concise paragraph)\n"
    "- experiments (str): The experimental results, findings, and analysis (comprehensive paragraph)\n"
    "- style_guide (str): Infer the best-matching venue or format short name from paths and "
    "content (e.g. NeurIPS, ICML, ICLR, ACL, EMNLP, CVPR, Nature, Science, IEEE, ACM, APA). "
    "If unclear, choose the closest mainstream ML venue.\n"
    "- target_pages (int): A reasonable page target for this work type (typical conference "
    "full paper 8–12; short paper or workshop 4–8; journal survey 12–16).\n"
    "- figure_placements (list): One object per candidate figure listed in the user message "
    "appendix — each item {\"id\": <exact figure id>, \"section\": <one of: Introduction, "
    "Related Work, Method, Experiments, Results, Discussion>}.\n"
    "- table_placements (list): Same shape for each candidate table id — "
    "{\"id\": <exact table id>, \"section\": <same allowed set>}.\n\n"
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
            synth = await self._llm_synthesize(text_fragments, fragments, figures, tables)
        except Exception as e:
            logger.warning("LLM synthesis failed, using rule-based fallback: %s", e)
            synth = self._fallback_merge(text_fragments)
            synth = self._enrich_synth_with_heuristics(synth, fragments, figures, tables)

        title = overrides.pop("title", None) or synth.get("title", "Untitled Paper")

        figures = self._apply_section_placements(figures, synth, "figure_placements")
        tables = self._apply_section_placements(tables, synth, "table_placements")

        metadata = PaperMetaData(
            title=title,
            idea_hypothesis=synth.get("idea_hypothesis", ""),
            method=synth.get("method", ""),
            data=synth.get("data", ""),
            experiments=synth.get("experiments", ""),
            references=refs,
            figures=figures,
            tables=tables,
        )

        self._apply_style_and_length(metadata, synth, fragments, overrides)

        for key, value in overrides.items():
            if hasattr(metadata, key) and value is not None:
                setattr(metadata, key, value)

        return metadata

    @staticmethod
    def _placement_list_to_map(raw: Any) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if not isinstance(raw, list):
            return out
        for item in raw:
            if not isinstance(item, dict):
                continue
            iid = item.get("id")
            sec = item.get("section")
            if isinstance(iid, str) and isinstance(sec, str) and sec.strip():
                out[iid] = sec.strip()
        return out

    @staticmethod
    def _heuristic_section_for_path(file_path: Optional[str]) -> str:
        """
        Guess LaTeX section from relative path when the LLM omits placement.

        - **Returns**:
            - One of: Introduction, Related Work, Method, Experiments, Results, Discussion.
        """
        if not file_path:
            return "Experiments"
        s = file_path.lower().replace("\\", "/")
        if any(k in s for k in ("intro", "overview", "abstract", "motivation")):
            return "Introduction"
        if any(k in s for k in ("related", "literature", "survey", "background")):
            return "Related Work"
        if any(
            k in s
            for k in (
                "method",
                "model",
                "architecture",
                "pipeline",
                "framework",
                "algorithm",
                "approach",
            )
        ):
            return "Method"
        if any(k in s for k in ("discussion", "limitation", "future")):
            return "Discussion"
        if any(k in s for k in ("result", "ablation", "comparison", "benchmark")):
            return "Results"
        return "Experiments"

    def _apply_section_placements(
        self,
        specs: List[Union[FigureSpec, TableSpec]],
        synth: dict,
        placement_key: str,
    ) -> List[Union[FigureSpec, TableSpec]]:
        by_id = self._placement_list_to_map(synth.get(placement_key))
        out: List[Union[FigureSpec, TableSpec]] = []
        for spec in specs:
            sec = by_id.get(spec.id) or self._heuristic_section_for_path(
                getattr(spec, "file_path", None)
            )
            out.append(spec.model_copy(update={"section": sec}))
        return out

    def _apply_style_and_length(
        self,
        metadata: PaperMetaData,
        synth: dict,
        fragments: List[ExtractedFragment],
        overrides: Dict[str, Any],
    ) -> None:
        """Fill style_guide and target_pages from LLM output or heuristics (respect overrides)."""
        if "style_guide" not in overrides:
            sg = synth.get("style_guide")
            if not isinstance(sg, str) or not sg.strip():
                sg = self._heuristic_style_guide(fragments) or None
            if isinstance(sg, str) and sg.strip():
                metadata.style_guide = sg.strip()
        if "target_pages" not in overrides:
            tp = synth.get("target_pages")
            if isinstance(tp, str) and tp.strip().isdigit():
                tp = int(tp.strip())
            if tp is None:
                tp = self._heuristic_target_pages(metadata.style_guide, fragments)
            if isinstance(tp, (int, float)) and int(tp) > 0:
                metadata.target_pages = int(tp)

    @staticmethod
    def _heuristic_style_guide(fragments: List[ExtractedFragment]) -> Optional[str]:
        """Infer venue keyword from file paths (no LLM)."""
        blob = " ".join(f.source_file.lower() for f in fragments)
        pairs = (
            ("neurips", "NeurIPS"),
            ("nips", "NeurIPS"),
            ("icml", "ICML"),
            ("iclr", "ICLR"),
            ("acl", "ACL"),
            ("emnlp", "EMNLP"),
            ("cvpr", "CVPR"),
            ("eccv", "ECCV"),
            ("iccv", "ICCV"),
            ("nature", "Nature"),
            ("science", "Science"),
            ("ieee", "IEEE"),
            ("acm", "ACM"),
            ("chi", "CHI"),
            ("aaai", "AAAI"),
            ("ijcai", "IJCAI"),
        )
        for needle, label in pairs:
            if needle in blob:
                return label
        return None

    @staticmethod
    def _heuristic_target_pages(
        style_guide: Optional[str],
        fragments: List[ExtractedFragment],
    ) -> int:
        """Default page budget when neither LLM nor user provides one."""
        s = (style_guide or "").lower()
        if any(x in s for x in ("nature", "science")):
            return 12
        return 8

    @staticmethod
    def _enrich_synth_with_heuristics(
        synth: dict,
        fragments: List[ExtractedFragment],
        figures: List[FigureSpec],
        tables: List[TableSpec],
    ) -> dict:
        """After rule-only text merge, add style/pages and placement lists."""
        out = dict(synth)
        if not out.get("style_guide"):
            sg = MetadataSynthesizer._heuristic_style_guide(fragments)
            if sg:
                out["style_guide"] = sg
        if out.get("target_pages") in (None, 0):
            out["target_pages"] = MetadataSynthesizer._heuristic_target_pages(
                out.get("style_guide"), fragments
            )
        if not out.get("figure_placements"):
            out["figure_placements"] = [
                {
                    "id": f.id,
                    "section": MetadataSynthesizer._heuristic_section_for_path(f.file_path),
                }
                for f in figures
            ]
        if not out.get("table_placements"):
            out["table_placements"] = [
                {
                    "id": t.id,
                    "section": MetadataSynthesizer._heuristic_section_for_path(t.file_path),
                }
                for t in tables
            ]
        return out

    async def _llm_synthesize(
        self,
        text_fragments: List[ExtractedFragment],
        all_fragments: List[ExtractedFragment],
        figures: List[FigureSpec],
        tables: List[TableSpec],
    ) -> dict:
        grouped = self._group_by_field(text_fragments)
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

        asset_lines: List[str] = []
        if figures:
            fig_payload = [
                {"id": f.id, "caption": f.caption, "file_path": f.file_path}
                for f in figures
            ]
            asset_lines.append(
                "=== CANDIDATE FIGURES (you must cover every id in figure_placements) ===\n"
                + json.dumps(fig_payload, ensure_ascii=False)
            )
        if tables:
            tab_payload = [
                {"id": t.id, "caption": t.caption, "file_path": t.file_path}
                for t in tables
            ]
            asset_lines.append(
                "=== CANDIDATE TABLES (you must cover every id in table_placements) ===\n"
                + json.dumps(tab_payload, ensure_ascii=False)
            )
        if asset_lines:
            user_msg = user_msg + "\n\n" + "\n\n".join(asset_lines)

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
        parsed = json.loads(raw)
        # If the model omitted new keys, fill from heuristics so downstream always has sections.
        return MetadataSynthesizer._enrich_synth_with_heuristics(
            parsed, all_fragments, figures, tables
        )

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
