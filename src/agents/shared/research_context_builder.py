"""
Unified research context construction (pre-plan), anchored on core reference analysis.
"""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..metadata_agent.models import (
    CoreRefAnalysis,
    PaperMetaData,
    ResearchContextModel,
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
    try:
        parsed = json.loads(cleaned)
        if expected is not None and not isinstance(parsed, expected):
            return None
        return parsed
    except Exception:
        return None


ScorePapersFn = Callable[[str, List[Dict[str, Any]]], Awaitable[List[tuple]]]


class ResearchContextBuilder:
    """
    Builds a single ``ResearchContextModel`` before paper planning.

    - **Description**:
        - Core references are privileged anchors; landscape papers are scored
          separately and never displace core analyses.
    """

    def __init__(self, client: Any, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    def _fallback_from_core_and_landscape(
        self,
        core_analysis: CoreRefAnalysis,
        landscape_papers: List[Dict[str, Any]],
        paper_metadata: PaperMetaData,
    ) -> ResearchContextModel:
        key_papers = [
            {"title": it.title, "contribution": "; ".join(it.core_contributions[:3])}
            for it in core_analysis.items[:10]
        ]
        for p in landscape_papers[:5]:
            key_papers.append(
                {
                    "title": p.get("title", ""),
                    "contribution": str(p.get("abstract", ""))[:200],
                }
            )
        gaps = list(core_analysis.shared_gaps or [])
        if paper_metadata.idea_hypothesis and len(gaps) < 3:
            gaps.append(paper_metadata.idea_hypothesis[:240])
        return ResearchContextModel(
            research_area="(fallback) anchored on user core references",
            summary=(
                f"Landscape built from {len(core_analysis.items)} core ref(s) and "
                f"{len(landscape_papers)} landscape paper(s) for: {paper_metadata.title}"
            ),
            key_papers=key_papers,
            research_trends=[],
            gaps=gaps,
            claim_evidence_matrix=[],
            contribution_ranking={"P0": [], "P1": [], "P2": []},
            planning_decision_trace=["Heuristic research context (JSON parse failed)."],
            paper_assignments={},
            core_ref_analysis=core_analysis,
        )

    async def build(
        self,
        *,
        core_analysis: CoreRefAnalysis,
        landscape_papers: List[Dict[str, Any]],
        paper_metadata: PaperMetaData,
        score_papers_fn: Optional[ScorePapersFn] = None,
        top_k_landscape: int = 24,
    ) -> ResearchContextModel:
        """
        Produce research context JSON via LLM, anchored on ``core_analysis``.

        - **Args**:
            - `score_papers_fn` (optional): async (topic, papers) -> list of (paper, score)
              sorted by relevance; if None, first ``top_k_landscape`` papers are used.
        """
        topic = paper_metadata.title
        pool = list(landscape_papers)
        if score_papers_fn and pool:
            try:
                scored = await score_papers_fn(topic, pool)
                pool = [p for p, _ in scored[:top_k_landscape]]
            except Exception:
                pool = pool[:top_k_landscape]
        else:
            pool = pool[:top_k_landscape]

        paper_summaries = []
        for p in pool:
            paper_summaries.append(
                {
                    "citation_key": p.get("ref_id", p.get("citation_key", "")),
                    "title": p.get("title", ""),
                    "year": p.get("year"),
                    "venue": p.get("venue", ""),
                    "citation_count": p.get("citation_count"),
                    "abstract": (p.get("abstract", "") or "")[:200],
                }
            )

        core_blob = core_analysis.model_dump(mode="json")
        papers_json = json.dumps(paper_summaries, ensure_ascii=False)

        sys_msg = (
            "You are an academic research analyst. The CORE_REFERENCE_ANALYSIS block "
            "contains user-selected anchor papers — treat them as authoritative. "
            "LANDSCAPE_PAPERS are additional retrieved works to situate the field. "
            "Analyze the broader landscape AROUND the core anchors. Respond with JSON only."
        )
        user_msg = (
            f"Target manuscript title: {paper_metadata.title}\n"
            f"Idea/hypothesis: {paper_metadata.idea_hypothesis[:1500]}\n\n"
            f"CORE_REFERENCE_ANALYSIS:\n{json.dumps(core_blob, ensure_ascii=False)}\n\n"
            f"LANDSCAPE_PAPERS:\n{papers_json}\n\n"
            "Return JSON with keys: research_area, summary, key_papers (non-core landscape "
            "highlights), research_trends, gaps, claim_evidence_matrix (6-10 records with "
            "section_type, claim, support_refs, reason, priority), contribution_ranking "
            "(object with P0/P1/P2 arrays of {contribution, why_it_matters, "
            "suggested_sections, suggested_result_focus}), planning_decision_trace (array of str). "
            "Prioritize citing support_refs that appear in CORE_REFERENCE_ANALYSIS items' ref_id "
            "or LANDSCAPE_PAPERS citation_key when making claims."
        )

        context: Optional[Dict[str, Any]] = None
        try:
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.25,
                max_tokens=1600,
            )
            raw = response.choices[0].message.content or ""
            context = _safe_load_json(raw, expected=dict)
        except Exception:
            context = None

        if context is None:
            return self._fallback_from_core_and_landscape(
                core_analysis, landscape_papers, paper_metadata
            )

        cr = context.get("contribution_ranking", {}) or {}
        if not isinstance(cr, dict):
            cr = {"P0": [], "P1": [], "P2": []}
        for band in ("P0", "P1", "P2"):
            cr.setdefault(band, [])

        return ResearchContextModel(
            research_area=str(context.get("research_area", "")),
            summary=str(context.get("summary", "")),
            key_papers=list(context.get("key_papers", []))[:10],
            research_trends=list(context.get("research_trends", [])),
            gaps=list(context.get("gaps", [])),
            claim_evidence_matrix=list(context.get("claim_evidence_matrix", [])),
            contribution_ranking=cr,
            planning_decision_trace=list(context.get("planning_decision_trace", [])),
            paper_assignments={},
            core_ref_analysis=core_analysis,
        )
