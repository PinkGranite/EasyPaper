"""
ExemplarSelector - Select the best exemplar (benchmark) paper for writing guidance.
- **Description**:
    - Two-stage funnel: first checks user-provided core references,
      then falls back to external search if no core ref qualifies.
    - Hard constraints: venue match + full-text availability (Docling).
    - Soft ranking: LLM scores method/domain similarity.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...config.schema import ExemplarConfig

logger = logging.getLogger(__name__)


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


class ExemplarSelector:
    """
    Selects the best exemplar paper from core refs or external search.
    - **Description**:
        - Filters candidates by venue, recency, and docling availability.
        - Ranks remaining candidates by method/domain similarity via LLM.
        - Returns the single best candidate dict, or None.

    - **Args**:
        - `client` (Any): OpenAI-compatible async LLM client.
        - `model_name` (str): Model identifier for LLM calls.
    """

    def __init__(self, client: Any, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    def _filter_hard_constraints(
        self,
        refs: List[Dict[str, Any]],
        style_guide: Optional[str],
        recency_years: int,
    ) -> List[Dict[str, Any]]:
        """
        Apply hard constraints to filter candidate references.
        - **Description**:
            - Venue match: ref venue must case-insensitively contain style_guide.
              Skipped if style_guide is None/empty.
            - Recency: ref year must be within recency_years of current year.
              Skipped if ref has no year.
            - Docling availability: ref must have non-empty docling_full_text.

        - **Args**:
            - `refs` (List[Dict]): Candidate reference dicts.
            - `style_guide` (Optional[str]): Target venue (e.g., "nature").
            - `recency_years` (int): Maximum age in years.

        - **Returns**:
            - `List[Dict]`: Filtered candidates.
        """
        current_year = datetime.now().year
        result = []
        for ref in refs:
            if not ref.get("docling_full_text"):
                continue

            if style_guide:
                ref_venue = str(ref.get("venue", "")).lower()
                if style_guide.lower() not in ref_venue:
                    continue

            ref_year = ref.get("year")
            if ref_year and (current_year - int(ref_year)) > recency_years:
                continue

            result.append(ref)
        return result

    async def _rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        metadata: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Rank candidates by method/domain similarity using LLM.
        - **Description**:
            - For a single candidate, returns it directly without LLM call.
            - For multiple candidates, asks LLM to score each on 0-10.
            - Falls back to first candidate on LLM failure.

        - **Args**:
            - `candidates` (List[Dict]): Pre-filtered candidate refs.
            - `metadata` (Any): PaperMetaData for the target paper.

        - **Returns**:
            - `Optional[Dict]`: Best candidate, or None if empty.
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        summaries = []
        for c in candidates:
            summaries.append({
                "ref_id": c.get("ref_id", ""),
                "title": c.get("title", ""),
                "venue": c.get("venue", ""),
                "year": c.get("year"),
                "abstract": str(c.get("abstract", ""))[:500],
            })

        prompt = (
            f"Target paper title: {metadata.title}\n"
            f"Method: {metadata.method[:500]}\n"
            f"Hypothesis: {metadata.idea_hypothesis[:500]}\n\n"
            f"Candidate exemplar papers:\n{json.dumps(summaries, ensure_ascii=False)}\n\n"
            "Score each candidate (0-10) on how well it could serve as a writing exemplar "
            "(methodological similarity, domain overlap, structural relevance).\n"
            'Return JSON: {"rankings": [{"ref_id": str, "score": float, "reason": str}]}'
        )

        try:
            resp = await self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": "You are an academic paper analysis assistant. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=800,
            )
            raw = resp.choices[0].message.content or ""
            cleaned = _strip_code_fence(raw)
            parsed = json.loads(cleaned)
            rankings = parsed.get("rankings", [])
            if rankings:
                rankings.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
                best_id = rankings[0].get("ref_id", "")
                for c in candidates:
                    if c.get("ref_id") == best_id:
                        return c
        except Exception as exc:
            logger.warning("ExemplarSelector LLM ranking failed: %s", exc)

        return candidates[0]

    async def select(
        self,
        core_refs: List[Dict[str, Any]],
        metadata: Any,
        config: ExemplarConfig,
        paper_search_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Orchestrate exemplar selection: core refs first, external search fallback.
        - **Description**:
            - Applies hard constraints to core refs.
            - Ranks filtered candidates via LLM.
            - Returns best match or None.

        - **Args**:
            - `core_refs` (List[Dict]): User-provided core reference dicts.
            - `metadata` (Any): PaperMetaData for the target paper.
            - `config` (ExemplarConfig): Feature configuration.
            - `paper_search_config` (Dict, optional): Search API config (for future external search).

        - **Returns**:
            - `Optional[Dict]`: Selected exemplar ref dict, or None.
        """
        style_guide = getattr(metadata, "style_guide", None) if config.venue_match_required else None

        candidates = self._filter_hard_constraints(
            core_refs, style_guide, config.recency_years,
        )

        if candidates:
            logger.info(
                "ExemplarSelector: %d core ref(s) passed hard constraints",
                len(candidates),
            )
            return await self._rank_candidates(candidates, metadata)

        logger.info("ExemplarSelector: no core ref qualified, returning None")
        return None
