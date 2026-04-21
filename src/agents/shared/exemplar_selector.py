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
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...config.schema import ExemplarConfig

logger = logging.getLogger(__name__)

_VENUE_ALIASES: Dict[str, List[str]] = {
    "icml": ["international conference on machine learning"],
    "neurips": [
        "advances in neural information processing systems",
        "neural information processing systems",
        "nips",
    ],
    "iclr": ["international conference on learning representations"],
    "cvpr": [
        "ieee/cvf conference on computer vision and pattern recognition",
        "computer vision and pattern recognition",
        "ieee conference on computer vision and pattern recognition",
    ],
    "iccv": [
        "ieee/cvf international conference on computer vision",
        "international conference on computer vision",
    ],
    "eccv": ["european conference on computer vision"],
    "acl": [
        "annual meeting of the association for computational linguistics",
        "association for computational linguistics",
    ],
    "emnlp": [
        "conference on empirical methods in natural language processing",
        "empirical methods in natural language processing",
    ],
    "naacl": [
        "north american chapter of the association for computational linguistics",
    ],
    "aaai": [
        "aaai conference on artificial intelligence",
        "association for the advancement of artificial intelligence",
    ],
    "ijcai": ["international joint conference on artificial intelligence"],
    "sigir": [
        "international acm sigir conference",
        "research and development in information retrieval",
    ],
    "kdd": ["knowledge discovery and data mining"],
    "nature": ["nature"],
    "science": ["science"],
}


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


def _venue_matches(ref_venue: str, style_guide: str) -> bool:
    """
    Check if a reference venue matches the target style guide.
    - **Description**:
        - First tries substring match (original logic).
        - Then checks alias table for abbreviation ↔ full name mappings.

    - **Args**:
        - `ref_venue` (str): The venue string from the reference.
        - `style_guide` (str): Target venue abbreviation or name.

    - **Returns**:
        - `bool`: True if venue matches.
    """
    rv = ref_venue.lower().strip()
    sg = style_guide.lower().strip()

    if sg in rv or rv in sg:
        return True

    aliases = _VENUE_ALIASES.get(sg, [])
    for alias in aliases:
        if alias in rv or rv in alias:
            return True

    for canonical, alias_list in _VENUE_ALIASES.items():
        if sg in alias_list or any(sg in a for a in alias_list):
            if canonical in rv:
                return True
            for a in alias_list:
                if a in rv or rv in a:
                    return True

    return False


class ExemplarSelector:
    """
    Selects the best exemplar paper from core refs or external search.
    - **Description**:
        - Filters candidates by venue, recency, and docling availability.
        - Ranks remaining candidates by method/domain similarity via LLM.
        - Falls back to external search when core refs don't qualify.
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
            - Venue match: uses alias-aware matching via _venue_matches.
              Skipped if style_guide is None/empty.
            - Recency: ref year must be within recency_years of current year.
              Skipped if ref has no year.
            - Docling availability: ref must have non-empty docling_full_text.

        - **Args**:
            - `refs` (List[Dict]): Candidate reference dicts.
            - `style_guide` (Optional[str]): Target venue (e.g., "ICML").
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
                ref_venue = str(ref.get("venue", ""))
                if not _venue_matches(ref_venue, style_guide):
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
                "ref_id": c.get("ref_id", c.get("bibtex_key", "")),
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
            )
            raw = resp.choices[0].message.content or ""
            cleaned = _strip_code_fence(raw)
            parsed = json.loads(cleaned)
            rankings = parsed.get("rankings", [])
            if rankings:
                rankings.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
                best_id = rankings[0].get("ref_id", "")
                for c in candidates:
                    cid = c.get("ref_id", c.get("bibtex_key", ""))
                    if cid == best_id:
                        return c
        except Exception as exc:
            logger.warning("ExemplarSelector LLM ranking failed: %s", exc)

        return candidates[0]

    def _build_search_query(
        self,
        metadata: Any,
        style_guide: Optional[str],
    ) -> str:
        """
        Construct a search query for external exemplar search.
        - **Description**:
            - Prioritises venue + broad domain keywords for style matching.
            - Falls back to method keywords when no style_guide is set.

        - **Args**:
            - `metadata` (Any): PaperMetaData for the target paper.
            - `style_guide` (Optional[str]): Target venue abbreviation.

        - **Returns**:
            - `str`: Search query string.
        """
        title_words = re.sub(r"[^a-zA-Z0-9\s]", "", metadata.title).split()
        domain_keywords = " ".join(title_words[:6])
        method_snippet = metadata.method[:100].split(".")[0].strip()

        if style_guide:
            return f"{style_guide} {domain_keywords}"
        return f"{domain_keywords} {method_snippet}"

    async def _search_external(
        self,
        metadata: Any,
        config: ExemplarConfig,
        paper_search_config: Dict[str, Any],
        style_guide: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Search for exemplar candidates via PaperSearchTool.
        - **Description**:
            - Calls Semantic Scholar / arXiv for papers matching venue + domain.
            - Filters results to those with downloadable PDFs.
            - Applies venue matching when style_guide is set.

        - **Args**:
            - `metadata` (Any): PaperMetaData for the target paper.
            - `config` (ExemplarConfig): Exemplar configuration.
            - `paper_search_config` (Dict): Search API credentials and settings.
            - `style_guide` (Optional[str]): Target venue abbreviation.

        - **Returns**:
            - `List[Dict]`: Filtered candidates with download URLs.
        """
        from .tools.paper_search import PaperSearchTool

        tool = PaperSearchTool(
            semantic_scholar_api_key=paper_search_config.get("semantic_scholar_api_key"),
            timeout=paper_search_config.get("timeout", 10),
        )

        query = self._build_search_query(metadata, style_guide)
        current_year = datetime.now().year
        year_range = f"{current_year - config.recency_years}-{current_year}"

        try:
            result = await tool.execute(
                query=query,
                max_results=config.max_external_candidates,
                year_range=year_range,
            )
        except Exception as exc:
            logger.warning("ExemplarSelector external search failed: %s", exc)
            return []

        if not result.success:
            return []

        papers = (result.data or {}).get("papers", [])
        candidates = []
        for paper in papers:
            has_pdf = bool(paper.get("open_access_pdf") or paper.get("arxiv_id"))
            if not has_pdf:
                continue

            if style_guide:
                venue = str(paper.get("venue", ""))
                if not _venue_matches(venue, style_guide):
                    continue

            candidates.append(paper)

        return candidates

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
            - Falls back to external search when no core ref qualifies
              and paper_search_config is provided.
            - Returns best match or None.

        - **Args**:
            - `core_refs` (List[Dict]): User-provided core reference dicts.
            - `metadata` (Any): PaperMetaData for the target paper.
            - `config` (ExemplarConfig): Feature configuration.
            - `paper_search_config` (Dict, optional): Search API config for external search.

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

        if not paper_search_config:
            logger.info("ExemplarSelector: no core ref qualified, no search config, returning None")
            return None

        logger.info("ExemplarSelector: no core ref qualified, trying external search")
        external = await self._search_external(
            metadata, config, paper_search_config, style_guide,
        )
        if not external:
            logger.info("ExemplarSelector: external search returned no candidates")
            return None

        best = await self._rank_candidates(external, metadata)
        if best:
            logger.info(
                "ExemplarSelector: external exemplar selected: %s",
                best.get("title", best.get("bibtex_key", "unknown")),
            )
        return best
