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
        venue = ref_info.get("venue", "")
        prompt = (
            "You are a writing-pattern analyst. Your goal is to extract TRANSFERABLE "
            "rhetorical strategies from an exemplar paper — strategies that a DIFFERENT "
            "paper on a DIFFERENT topic could adopt for its own argumentation.\n\n"
            "CRITICAL RULES:\n"
            "- Extract ABSTRACT rhetorical patterns, NOT content-specific descriptions.\n"
            "- The 'role' field must describe a generalizable writing strategy that any "
            "paper in this venue could use, NOT what this specific paper argues.\n"
            "  BAD: 'Identify model and data limitations in VLP, propose MED architecture'\n"
            "  GOOD: 'Identify field limitations from dual perspectives, then propose a unified solution'\n"
            "- Paragraph archetypes should be generic rhetorical roles, not topic-specific.\n"
            "  BAD: ['VLP_progress', 'MED_architecture_details']\n"
            "  GOOD: ['field_progress_and_gap', 'multi_perspective_limitation', 'unified_solution_proposal']\n\n"
            f"Exemplar paper: \"{ref_info.get('title', '')}\"\n"
            f"Venue: {venue}\n"
            f"Year: {ref_info.get('year', '')}\n\n"
            f"Target paper (for context only — do NOT describe it):\n"
            f"  Title: {metadata.title}\n"
            f"  Method: {metadata.method[:200]}\n\n"
            f"Exemplar full text (truncated):\n{text}\n\n"
            "Extract the following and return as JSON:\n"
            "1. section_blueprint: array of objects, each with:\n"
            "   - section_type (string, e.g. \"introduction\", \"method\", \"experiments\", \"discussion\", \"conclusion\")\n"
            "   - title (string, the section heading)\n"
            "   - approx_word_count (integer, observed word count in this section)\n"
            "   - paragraph_count (integer, observed number of paragraphs)\n"
            "   - subsection_titles (array of strings, any sub-headings)\n"
            f"   - role (string, the GENERALIZABLE rhetorical strategy this section uses — "
            f"something any {venue or 'academic'} paper could adopt, NOT a content summary)\n"
            "2. style_profile: object with:\n"
            "   - tone (string, one of: \"formal\", \"semi-formal\", \"technical\")\n"
            "   - citation_density (number, average citations per paragraph, e.g. 2.5)\n"
            "   - avg_sentence_length (number, average words per sentence, e.g. 22.0)\n"
            "   - hedging_level (string, one of: \"low\", \"moderate\", \"high\")\n"
            "   - transition_patterns (array of strings, common transition phrases used)\n"
            "3. argumentation_patterns: object with:\n"
            "   - intro_hook_type (string, e.g. \"grand_challenge\", \"recent_progress\", \"knowledge_gap\")\n"
            "   - claim_evidence_structure (string, e.g. \"claim_then_evidence\", \"evidence_then_claim\")\n"
            "   - discussion_closing_strategy (string, e.g. \"future_work\", \"broader_impact\", \"limitations_first\")\n"
            "4. paragraph_archetypes: object mapping section_type (string) to array of "
            "GENERIC paragraph role names that any paper could reuse "
            "(e.g. {\"introduction\": [\"field_progress_and_gap\", \"multi_perspective_limitation\", "
            "\"unified_solution_proposal\", \"contribution_enumeration\"]})\n\n"
            "IMPORTANT: All numeric fields (approx_word_count, paragraph_count, citation_density, avg_sentence_length) "
            "MUST be plain numbers, NOT strings or descriptions.\n\n"
            "Return ONLY valid JSON, no markdown fences."
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

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        """Convert to float safely — returns default for non-numeric values."""
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return default
        return default

    @staticmethod
    def _safe_int(val: Any, default: int = 0) -> int:
        """Convert to int safely — returns default for non-numeric values."""
        if isinstance(val, int) and not isinstance(val, bool):
            return val
        if isinstance(val, float):
            return int(val)
        if isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                return default
        return default

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
                    approx_word_count=self._safe_int(bp.get("approx_word_count", 0)),
                    paragraph_count=self._safe_int(bp.get("paragraph_count", 0)),
                    subsection_titles=list(bp.get("subsection_titles") or []),
                    role=bp.get("role", ""),
                ))

        sp_data = data.get("style_profile", {})
        style = StyleProfile(
            tone=sp_data.get("tone", "formal"),
            citation_density=self._safe_float(sp_data.get("citation_density", 0.0)),
            avg_sentence_length=self._safe_float(sp_data.get("avg_sentence_length", 0.0)),
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
            - Formats the ExemplarAnalysis into a soft writing-style reference
              tailored to the given section_type.
            - Presents observations as venue conventions and rhetorical
              suggestions, NOT as prescriptive targets.
            - Returns empty string if analysis is None.

        - **Args**:
            - `analysis` (Optional[ExemplarAnalysis]): The analysis result.
            - `section_type` (str): Current section being generated.

        - **Returns**:
            - `str`: Formatted guidance block for prompt injection.
        """
        if analysis is None:
            return ""

        venue = analysis.venue or "this venue"
        parts = [
            f"## Writing Style Reference (from a comparable {venue} paper)",
            f"Observed in: \"{analysis.title}\" ({venue}, {analysis.year})",
        ]

        matching_bp = None
        for bp in analysis.section_blueprint:
            if bp.section_type == section_type:
                matching_bp = bp
                break

        if matching_bp:
            parts.append(f"\n### Rhetorical Strategy for {section_type}")
            parts.append(f"- Strategy: {matching_bp.role}")
            if matching_bp.paragraph_count:
                lo = max(1, matching_bp.paragraph_count - 1)
                hi = matching_bp.paragraph_count + 1
                parts.append(f"- Typical paragraphs: {lo}-{hi} (observed: {matching_bp.paragraph_count})")
            if matching_bp.approx_word_count:
                lo = int(matching_bp.approx_word_count * 0.8)
                hi = int(matching_bp.approx_word_count * 1.2)
                parts.append(f"- Typical length: ~{lo}-{hi} words (observed: {matching_bp.approx_word_count})")
            if matching_bp.subsection_titles:
                parts.append(f"- Common subsection pattern: {', '.join(matching_bp.subsection_titles)}")

        archetypes = analysis.paragraph_archetypes.get(section_type, [])
        if archetypes:
            parts.append(f"\n### Suggested Rhetorical Flow")
            numbered = [f"{i+1}. {a.replace('_', ' ').capitalize()}" for i, a in enumerate(archetypes)]
            parts.extend(numbered)

        sp = analysis.style_profile
        parts.append(f"\n### Venue Conventions (observed)")
        if sp.citation_density:
            parts.append(f"- Citation density: ~{sp.citation_density:.1f} per paragraph is common")
        parts.append(f"- Tone: {sp.tone}")
        if sp.transition_patterns:
            parts.append(f"- Common transitions: {', '.join(sp.transition_patterns[:5])}")

        ap = analysis.argumentation_patterns
        if section_type == "introduction" and ap.intro_hook_type:
            parts.append(f"- Opening strategy: {ap.intro_hook_type.replace('_', ' ')}")
        if section_type in ("discussion", "conclusion") and ap.discussion_closing_strategy:
            parts.append(f"- Closing strategy: {ap.discussion_closing_strategy.replace('_', ' ')}")

        parts.append(
            "\nThese are soft references from a comparable published paper. "
            "Your paper's structure and argumentation should be driven by "
            "YOUR content and research contributions. Borrow rhetorical "
            "strategies and venue conventions, NOT specific claims or framing."
        )

        return "\n".join(parts)
