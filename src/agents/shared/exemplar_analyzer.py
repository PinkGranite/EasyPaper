"""
ExemplarAnalyzer - Decompose an exemplar paper into reusable writing patterns.
- **Description**:
    - Takes a paper's full text (from Docling) and produces an ExemplarAnalysis
      containing section blueprints, style profile, argumentation patterns,
      and per-section paragraph archetypes.
    - Uses a single LLM call with structured JSON output.
    - Falls back to heuristic extraction when LLM fails.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..metadata_agent.models import (
    ExemplarAnalysis,
    SectionBlueprint,
    StyleProfile,
    ArgumentationPatterns,
)

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


def _safe_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    cleaned = _strip_code_fence(raw)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return None


_KNOWN_SECTIONS = [
    "abstract", "introduction", "related_work", "method",
    "experiment", "results", "discussion", "conclusion",
]


class ExemplarAnalyzer:
    """
    Decomposes an exemplar paper into structured writing patterns.
    - **Description**:
        - Produces ExemplarAnalysis from full text + sections.
        - Single LLM call extracts blueprints, style, argumentation, archetypes.
        - Heuristic fallback on LLM failure.

    - **Args**:
        - `client` (Any): OpenAI-compatible async LLM client.
        - `model_name` (str): Model identifier.
        - `max_chars` (int): Maximum characters of full text to send to LLM.
    """

    def __init__(self, client: Any, model_name: str, *, max_chars: int = 8000) -> None:
        self._client = client
        self._model_name = model_name
        self._max_chars = max_chars

    async def analyze(
        self,
        full_text: str,
        sections: Dict[str, str],
        metadata: Any,
        ref_info: Dict[str, Any],
    ) -> ExemplarAnalysis:
        """
        Analyze an exemplar paper and extract writing patterns.
        - **Description**:
            - Sends full text to LLM for structured decomposition.
            - Falls back to heuristic analysis on failure.

        - **Args**:
            - `full_text` (str): Complete paper text from Docling.
            - `sections` (Dict[str, str]): Named sections from Docling.
            - `metadata` (Any): Target paper's PaperMetaData.
            - `ref_info` (Dict): Basic info (ref_id, title, venue, year).

        - **Returns**:
            - `ExemplarAnalysis`: Structured decomposition.
        """
        ref_id = ref_info.get("ref_id", "unknown")
        title = ref_info.get("title", "")
        venue = ref_info.get("venue", "")
        year = ref_info.get("year", 0)

        if not full_text and not sections:
            return ExemplarAnalysis(
                ref_id=ref_id, title=title, venue=venue, year=year,
            )

        text_for_llm = full_text[:self._max_chars] if full_text else ""
        if not text_for_llm and sections:
            text_for_llm = "\n\n".join(
                f"## {k}\n{v[:1500]}" for k, v in sections.items()
            )[:self._max_chars]

        try:
            result = await self._llm_analyze(text_for_llm, ref_info, metadata)
            if result is not None:
                return result
        except Exception as exc:
            logger.warning("ExemplarAnalyzer LLM failed: %s", exc)

        return self._heuristic_fallback(sections, ref_info)

    async def _llm_analyze(
        self,
        text: str,
        ref_info: Dict[str, Any],
        metadata: Any,
    ) -> Optional[ExemplarAnalysis]:
        """Run LLM analysis and parse JSON response."""
        prompt = (
            "You are analyzing an exemplar academic paper to extract reusable writing patterns.\n\n"
            f"Exemplar paper: \"{ref_info.get('title', '')}\"\n"
            f"Venue: {ref_info.get('venue', '')}\n"
            f"Year: {ref_info.get('year', '')}\n\n"
            f"Target paper title: {metadata.title}\n"
            f"Target method: {metadata.method[:300]}\n\n"
            f"Exemplar full text (truncated):\n{text}\n\n"
            "Extract the following and return as JSON:\n"
            "1. section_blueprint: array of {section_type, title, approx_word_count, "
            "paragraph_count, subsection_titles, role}\n"
            "2. style_profile: {tone, citation_density, avg_sentence_length, "
            "hedging_level, transition_patterns}\n"
            "3. argumentation_patterns: {intro_hook_type, claim_evidence_structure, "
            "discussion_closing_strategy}\n"
            "4. paragraph_archetypes: dict mapping section_type to list of "
            "paragraph role names (e.g., [\"broad_hook\", \"gap_statement\"])\n\n"
            "Return ONLY valid JSON."
        )

        resp = await self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": "You are an academic writing pattern analyst. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        raw = resp.choices[0].message.content or ""
        parsed = _safe_parse_json(raw)
        if parsed is None:
            return None

        return self._build_from_dict(parsed, ref_info)

    def _build_from_dict(
        self, data: Dict[str, Any], ref_info: Dict[str, Any],
    ) -> ExemplarAnalysis:
        """Construct ExemplarAnalysis from parsed LLM JSON."""
        blueprints = []
        for bp in data.get("section_blueprint", []):
            if isinstance(bp, dict):
                blueprints.append(SectionBlueprint(
                    section_type=bp.get("section_type", ""),
                    title=bp.get("title", ""),
                    approx_word_count=int(bp.get("approx_word_count", 0)),
                    paragraph_count=int(bp.get("paragraph_count", 0)),
                    subsection_titles=list(bp.get("subsection_titles") or []),
                    role=bp.get("role", ""),
                ))

        sp_data = data.get("style_profile", {})
        style = StyleProfile(
            tone=sp_data.get("tone", "formal"),
            citation_density=float(sp_data.get("citation_density", 0.0)),
            avg_sentence_length=float(sp_data.get("avg_sentence_length", 0.0)),
            hedging_level=sp_data.get("hedging_level", "moderate"),
            transition_patterns=list(sp_data.get("transition_patterns") or []),
        )

        ap_data = data.get("argumentation_patterns", {})
        argumentation = ArgumentationPatterns(
            intro_hook_type=ap_data.get("intro_hook_type", ""),
            claim_evidence_structure=ap_data.get("claim_evidence_structure", ""),
            discussion_closing_strategy=ap_data.get("discussion_closing_strategy", ""),
        )

        archetypes = {}
        raw_archetypes = data.get("paragraph_archetypes", {})
        if isinstance(raw_archetypes, dict):
            for k, v in raw_archetypes.items():
                if isinstance(v, list):
                    archetypes[k] = [str(item) for item in v]

        return ExemplarAnalysis(
            ref_id=ref_info.get("ref_id", "unknown"),
            title=ref_info.get("title", ""),
            venue=ref_info.get("venue", ""),
            year=int(ref_info.get("year", 0)),
            section_blueprint=blueprints,
            style_profile=style,
            argumentation_patterns=argumentation,
            paragraph_archetypes=archetypes,
        )

    def _heuristic_fallback(
        self,
        sections: Dict[str, str],
        ref_info: Dict[str, Any],
    ) -> ExemplarAnalysis:
        """
        Rule-based fallback when LLM analysis fails.
        - **Description**:
            - Infers section blueprints from available section keys and lengths.
        """
        blueprints = []
        for sec_name in _KNOWN_SECTIONS:
            text = sections.get(sec_name, "")
            if not text:
                continue
            words = len(text.split())
            paragraphs = max(1, text.count("\n\n") + 1)
            blueprints.append(SectionBlueprint(
                section_type=sec_name,
                title=sec_name.replace("_", " ").title(),
                approx_word_count=words,
                paragraph_count=paragraphs,
                role=f"Content for {sec_name} section",
            ))

        return ExemplarAnalysis(
            ref_id=ref_info.get("ref_id", "unknown"),
            title=ref_info.get("title", ""),
            venue=ref_info.get("venue", ""),
            year=int(ref_info.get("year", 0)),
            section_blueprint=blueprints,
            style_profile=StyleProfile(),
            argumentation_patterns=ArgumentationPatterns(),
        )

    @staticmethod
    def format_for_prompt(
        analysis: Optional[ExemplarAnalysis],
        section_type: str,
    ) -> str:
        """
        Render section-specific exemplar guidance for prompt injection.
        - **Description**:
            - Formats the ExemplarAnalysis into a human-readable guidance block
              tailored to the given section_type.
            - Returns empty string if analysis is None.

        - **Args**:
            - `analysis` (Optional[ExemplarAnalysis]): The analysis result.
            - `section_type` (str): Current section being generated.

        - **Returns**:
            - `str`: Formatted guidance block for prompt injection.
        """
        if analysis is None:
            return ""

        parts = [
            f"## Exemplar Paper Writing Guide",
            f"Source: \"{analysis.title}\" ({analysis.venue}, {analysis.year})",
        ]

        matching_bp = None
        for bp in analysis.section_blueprint:
            if bp.section_type == section_type:
                matching_bp = bp
                break

        if matching_bp:
            parts.append(f"\n### Section Blueprint for {section_type}")
            parts.append(f"- Role: {matching_bp.role}")
            if matching_bp.paragraph_count:
                parts.append(f"- Target paragraphs: {matching_bp.paragraph_count}")
            if matching_bp.approx_word_count:
                parts.append(f"- Approx word count: {matching_bp.approx_word_count}")
            if matching_bp.subsection_titles:
                parts.append(f"- Subsections: {', '.join(matching_bp.subsection_titles)}")

        archetypes = analysis.paragraph_archetypes.get(section_type, [])
        if archetypes:
            parts.append(f"\n### Paragraph Flow")
            parts.append(f"- Sequence: {' -> '.join(archetypes)}")

        sp = analysis.style_profile
        parts.append(f"\n### Style Reference")
        if sp.citation_density:
            parts.append(f"- Citation density: ~{sp.citation_density:.1f} per paragraph")
        parts.append(f"- Tone: {sp.tone}")
        if sp.transition_patterns:
            parts.append(f"- Transition patterns: {', '.join(sp.transition_patterns[:5])}")

        parts.append(
            "\nAdapt the exemplar's structural and rhetorical patterns "
            "while writing ORIGINAL content from the provided metadata. "
            "Do NOT reproduce the exemplar's specific claims or data."
        )

        return "\n".join(parts)
