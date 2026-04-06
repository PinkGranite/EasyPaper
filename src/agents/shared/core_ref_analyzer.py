"""
Deep analysis of user-provided core references (privileged over discovered papers).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..metadata_agent.models import (
    CoreRefAnalysis,
    CoreRefAnalysisItem,
    PaperMetaData,
)


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _safe_load_json(raw: str, expected: Optional[type] = None) -> Optional[Any]:
    cleaned = _strip_code_fence(raw)
    for cand in (cleaned,):
        if not cand:
            continue
        try:
            parsed = json.loads(cand)
            if expected is not None and not isinstance(parsed, expected):
                continue
            return parsed
        except Exception:
            continue
    return None


class CoreRefAnalyzer:
    """
    Produces structured CoreRefAnalysis from core reference records and paper metadata.

    - **Description**:
        - Uses LLM calls to extract contributions, methods, limitations, and
          cross-paper synthesis. Falls back to heuristics when disabled or on failure.
    """

    def __init__(
        self,
        client: Any,
        model_name: str,
        *,
        enabled: bool = True,
        max_abstract_chars: int = 2000,
        analyze_cross_paper: bool = True,
        max_section_chars: int = 3000,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._enabled = enabled
        self._max_abstract_chars = max_abstract_chars
        self._analyze_cross_paper = analyze_cross_paper
        self._max_section_chars = max_section_chars

    @classmethod
    def from_tools_config(cls, client: Any, model_name: str, tools_config: Any) -> CoreRefAnalyzer:
        """
        Build analyzer from optional ``ToolsConfig.core_ref_analysis``.
        """
        cfg = getattr(tools_config, "core_ref_analysis", None) if tools_config else None
        if cfg is None:
            return cls(client, model_name)
        return cls(
            client,
            model_name,
            enabled=getattr(cfg, "enabled", True),
            max_abstract_chars=getattr(cfg, "max_abstract_chars", 2000),
            analyze_cross_paper=getattr(cfg, "analyze_cross_paper", True),
        )

    def _heuristic_fallback(
        self,
        core_refs: List[Dict[str, Any]],
        paper_metadata: PaperMetaData,
    ) -> CoreRefAnalysis:
        items: List[CoreRefAnalysisItem] = []
        for ref in core_refs:
            rid = str(ref.get("ref_id", "")).strip()
            title = str(ref.get("title", "")).strip() or rid
            abstract = str(ref.get("abstract", "")).strip()
            snippet = abstract[:280] + ("..." if len(abstract) > 280 else "")
            items.append(
                CoreRefAnalysisItem(
                    ref_id=rid or "unknown",
                    title=title,
                    core_contributions=[snippet] if snippet else ["(no abstract available)"],
                    methodology="See paper abstract and full text.",
                    limitations=[],
                    relationship_to_ours=(
                        f"User-selected anchor relative to: {paper_metadata.title[:120]}"
                    ),
                    key_results=[],
                    relevance_score=1.0,
                )
            )
        return CoreRefAnalysis(
            items=items,
            shared_gaps=[paper_metadata.idea_hypothesis[:200]] if paper_metadata.idea_hypothesis else [],
            research_lineage="; ".join(i.title for i in items[:5]),
            positioning_statement=(
                f"This work builds on the user-provided anchor literature in relation to: "
                f"{paper_metadata.title}"
            ),
        )

    async def analyze(
        self,
        core_refs: List[Dict[str, Any]],
        paper_metadata: PaperMetaData,
    ) -> CoreRefAnalysis:
        """
        Run deep analysis (LLM) or heuristic fallback.
        """
        if not core_refs:
            return CoreRefAnalysis()
        if not self._enabled:
            return self._heuristic_fallback(core_refs, paper_metadata)

        per_paper_payload = []
        has_docling_data = False
        for ref in core_refs:
            rid = ref.get("ref_id", "")
            abstract = str(ref.get("abstract", ""))[: self._max_abstract_chars]
            entry: Dict[str, Any] = {
                "ref_id": rid,
                "title": ref.get("title", ""),
                "year": ref.get("year"),
                "venue": ref.get("venue", ""),
                "abstract": abstract,
                "bibtex_excerpt": str(ref.get("bibtex", ""))[:800],
            }

            docling_sections = ref.get("docling_sections")
            if isinstance(docling_sections, dict) and docling_sections:
                has_docling_data = True
                sc = self._max_section_chars
                if docling_sections.get("method"):
                    entry["method_excerpt"] = str(docling_sections["method"])[:sc]
                if docling_sections.get("results"):
                    entry["results_excerpt"] = str(docling_sections["results"])[:sc]
                if docling_sections.get("conclusion"):
                    entry["conclusion_excerpt"] = str(docling_sections["conclusion"])[:sc]
                if docling_sections.get("introduction"):
                    entry["introduction_excerpt"] = str(docling_sections["introduction"])[:sc]
                if docling_sections.get("experiment"):
                    entry["experiment_excerpt"] = str(docling_sections["experiment"])[:sc]

            per_paper_payload.append(entry)

        sys1 = (
            "You are an academic analyst. The following papers are USER-PROVIDED CORE REFERENCES. "
            "They are authoritative anchors for the manuscript — do not dismiss them as irrelevant. "
            "Respond with JSON only."
        )

        docling_hint = ""
        if has_docling_data:
            docling_hint = (
                "\nSome references include full-text excerpts (method_excerpt, results_excerpt, "
                "conclusion_excerpt, etc.) from their original papers. Use these for deeper analysis "
                "of contributions, methodology, limitations, and relationship to the target manuscript.\n"
            )

        user1 = (
            f"Target manuscript title: {paper_metadata.title}\n"
            f"Idea/hypothesis: {paper_metadata.idea_hypothesis[:1500]}\n"
            f"Method (draft): {paper_metadata.method[:1200]}\n\n"
            f"Core references (JSON):\n{json.dumps(per_paper_payload, ensure_ascii=False)}\n"
            f"{docling_hint}\n"
            "Return a JSON object with key \"items\": array of objects, one per ref_id above, each with:\n"
            '{"ref_id": str, "title": str, "core_contributions": [str], "methodology": str, '
            '"limitations": [str], "relationship_to_ours": str, "key_results": [str]}\n'
            "relationship_to_ours must describe how the target manuscript relates (extends, contrasts, "
            "combines, improves upon) this paper."
        )

        items: List[CoreRefAnalysisItem] = []
        try:
            resp = await self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": sys1},
                    {"role": "user", "content": user1},
                ],
                temperature=0.2,
                max_tokens=2500,
            )
            raw = resp.choices[0].message.content or ""
            parsed = _safe_load_json(raw, expected=dict)
            raw_items = (parsed or {}).get("items") if isinstance(parsed, dict) else None
            if isinstance(raw_items, list):
                for row in raw_items:
                    if not isinstance(row, dict):
                        continue
                    rid = str(row.get("ref_id", "")).strip()
                    if not rid:
                        continue
                    items.append(
                        CoreRefAnalysisItem(
                            ref_id=rid,
                            title=str(row.get("title", "")),
                            core_contributions=list(row.get("core_contributions") or []),
                            methodology=str(row.get("methodology", "")),
                            limitations=list(row.get("limitations") or []),
                            relationship_to_ours=str(row.get("relationship_to_ours", "")),
                            key_results=list(row.get("key_results") or []),
                            relevance_score=1.0,
                        )
                    )
        except Exception:
            items = []

        if not items:
            return self._heuristic_fallback(core_refs, paper_metadata)

        shared_gaps: List[str] = []
        lineage = ""
        positioning = ""
        if self._analyze_cross_paper:
            sys2 = "You are an academic analyst. Respond with JSON only."
            user2 = (
                f"Target manuscript: {paper_metadata.title}\n"
                f"Idea: {paper_metadata.idea_hypothesis[:1200]}\n\n"
                f"Analyzed core references (JSON):\n{json.dumps([i.model_dump() for i in items], ensure_ascii=False)}\n\n"
                'Return JSON: {"shared_gaps": [str], "research_lineage": str, "positioning_statement": str}\n'
                "shared_gaps: gaps across these papers that the target manuscript could address.\n"
                "research_lineage: 2-4 sentences on how these papers connect.\n"
                "positioning_statement: one paragraph on where the target manuscript sits."
            )
            try:
                resp2 = await self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": sys2},
                        {"role": "user", "content": user2},
                    ],
                    temperature=0.2,
                    max_tokens=1200,
                )
                raw2 = resp2.choices[0].message.content or ""
                p2 = _safe_load_json(raw2, expected=dict)
                if isinstance(p2, dict):
                    shared_gaps = list(p2.get("shared_gaps") or [])
                    lineage = str(p2.get("research_lineage", ""))
                    positioning = str(p2.get("positioning_statement", ""))
            except Exception:
                pass

        if not shared_gaps and not lineage:
            fb = self._heuristic_fallback(core_refs, paper_metadata)
            return CoreRefAnalysis(
                items=items,
                shared_gaps=fb.shared_gaps,
                research_lineage=fb.research_lineage or lineage,
                positioning_statement=fb.positioning_statement or positioning,
            )

        return CoreRefAnalysis(
            items=items,
            shared_gaps=shared_gaps,
            research_lineage=lineage,
            positioning_statement=positioning,
        )
