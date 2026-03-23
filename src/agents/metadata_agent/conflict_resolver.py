"""
Conflict Resolver — extracted from MetaDataAgent for architecture decoupling.
- **Description**:
    - Resolves conflicts between reviewer and VLM/typesetter feedback
    - Builds VLM and typesetter feedback from raw outputs
    - Merges and reconciles section-level feedback from multiple sources
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..reviewer_agent.models import (
    ConflictResolutionRecord,
    FeedbackResult,
    ReviewResult,
    SectionFeedback,
    Severity,
)
from .models import StructuralAction

logger = logging.getLogger(__name__)


LATEX_ERROR_FIXES: Dict[str, str] = {
    "ended by": (
        "Fix unclosed LaTeX environment. Ensure every \\begin{{...}} has a matching \\end{{...}}. "
        "Check figure, table, and equation environments."
    ),
    "misplaced alignment tab character &": (
        "Escape all literal '&' characters as '\\&' in regular text. "
        "Only use bare '&' inside tabular/align environments."
    ),
    "unicode character": (
        "Replace Unicode characters with LaTeX equivalents. "
        "For example: use \\textendash for –, $-$ or $\\minus$ for −, "
        "\\% for %, \\& for &."
    ),
    "missing $ inserted": (
        "Fix math mode errors. Wrap mathematical symbols like _, ^, "
        "\\alpha, \\beta in $...$ when used outside math environments."
    ),
    "undefined control sequence": (
        "Remove or replace undefined LaTeX commands. "
        "Check for typos in command names or missing package imports."
    ),
    "not in outer par mode": (
        "Move float environments (figure, table) out of restricted contexts. "
        "Floats cannot appear inside minipage, parbox, or other floats."
    ),
    "file not found": (
        "Remove or comment out \\includegraphics for missing figure files. "
        "Replace with a placeholder comment if needed."
    ),
    "no output pdf file produced": (
        "Fix critical LaTeX errors that prevent PDF generation. "
        "Check for unclosed environments, invalid commands, and encoding issues."
    ),
}


class ConflictResolver:
    """Resolves conflicts between reviewer, VLM, and typesetter feedback."""

    def __init__(self, host: "MetaDataAgent") -> None:  # noqa: F821
        self.host = host
        self.client = host.client
        self.model_name = host.model_name

    # -----------------------------------------------------------------
    # Static normalisation helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _normalize_target_paragraphs(raw_targets: Any) -> List[int]:
        """
        Normalizes paragraph targets from heterogeneous payloads.
        - **Description**:
         - Normalizes paragraph targets from heterogeneous payload types (int/str/list) into a deduplicated integer list.

        - **Args**:
         - `raw_targets` (Any): Raw paragraph target payload from planner/reviewer output.

        - **Returns**:
         - `targets` (List[int]): Deduplicated paragraph indices in ascending order.
        """
        if raw_targets is None:
            return []
        if isinstance(raw_targets, (int, str)):
            raw_targets = [raw_targets]
        if not isinstance(raw_targets, list):
            return []
        targets: List[int] = []
        for item in raw_targets:
            if isinstance(item, int):
                targets.append(item)
            elif isinstance(item, str) and item.strip().isdigit():
                targets.append(int(item.strip()))
        return sorted(list(set(targets)))

    @staticmethod
    def _normalize_paragraph_instructions(
        raw_instructions: Any,
        target_paragraphs: Optional[List[int]] = None,
        fallback_instruction: str = "",
    ) -> Dict[int, str]:
        """
        Normalizes paragraph instructions into a stable mapping.
        - **Description**:
         - Converts reviewer/planner outputs from `dict`, `list`, or `str` into `{paragraph_index: instruction}`.
         - Supports JSON-like string payloads and generic text instructions with paragraph-target fallback.

        - **Args**:
         - `raw_instructions` (Any): Raw paragraph instruction payload from planner/reviewer output.
         - `target_paragraphs` (Optional[List[int]]): Candidate paragraph indices used for fallback fan-out.
         - `fallback_instruction` (str): Backup instruction applied when parsing fails but targets exist.

        - **Returns**:
         - `normalized` (Dict[int, str]): Normalized paragraph instruction mapping.
        """
        normalized: Dict[int, str] = {}
        targets = target_paragraphs or []

        # dict form
        if isinstance(raw_instructions, dict):
            for k, v in raw_instructions.items():
                if str(k).isdigit():
                    normalized[int(k)] = str(v).strip()
            return normalized

        # list form
        if isinstance(raw_instructions, list):
            for item in raw_instructions:
                if isinstance(item, dict):
                    pidx = item.get("paragraph_index")
                    ins = item.get("instruction") or item.get("suggestion") or item.get("text")
                    if (isinstance(pidx, int) or str(pidx).isdigit()) and ins:
                        normalized[int(pidx)] = str(ins).strip()
                elif isinstance(item, str) and targets:
                    for t in targets:
                        normalized[t] = item.strip()
            return normalized

        # string form (generic or JSON-like)
        if isinstance(raw_instructions, str):
            text = raw_instructions.strip()
            if not text:
                return normalized

            # Try JSON object first
            if text.startswith("{") and text.endswith("}"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        for k, v in parsed.items():
                            if str(k).isdigit():
                                normalized[int(k)] = str(v).strip()
                        if normalized:
                            return normalized
                except Exception:
                    pass

            # Parse "3: xxx; 4: yyy" pattern
            pairs = re.findall(r"(\d+)\s*[:=-]\s*([^;]+)", text)
            for pidx, ins in pairs:
                normalized[int(pidx)] = ins.strip()
            if normalized:
                return normalized

            # Generic sentence => apply to all targets when available
            if targets:
                for t in targets:
                    normalized[t] = text
                return normalized

        # Fallback: if we have targets, use fallback instruction
        if targets and fallback_instruction:
            for t in targets:
                normalized[t] = fallback_instruction
        return normalized

    # -----------------------------------------------------------------
    # Conflict resolution (multi-objective + LLM arbitration)
    # -----------------------------------------------------------------

    async def _resolve_conflicts_with_llm(
        self,
        reviewer_feedbacks: List[SectionFeedback],
        external_feedbacks: List[SectionFeedback],
    ) -> Tuple[List[SectionFeedback], List[ConflictResolutionRecord]]:
        """
        Resolve conflicts between reviewer and external (typesetter/VLM) feedback.
        - **Description**:
            - Uses multi-objective scoring (Logic, Layout, Citation, Style) as the
              primary resolution mechanism via Pareto front selection
            - Falls back to LLM arbitration only when Pareto front has >= 2 members
              with indistinguishable lexicographic scores
        """
        from ..reviewer_agent.objective_scoring import (
            ObjectiveType,
            ActionScorecard,
            ObjectiveScore,
            score_logic,
            score_layout,
            score_citation,
            score_style,
            compute_pareto_front,
            select_action,
            needs_llm_arbitration,
        )

        merged = {sf.section_type: sf for sf in reviewer_feedbacks}
        conflict_records: List[ConflictResolutionRecord] = []
        for ext in external_feedbacks:
            existing = merged.get(ext.section_type)
            if existing is None:
                merged[ext.section_type] = ext
                continue
            if existing.action == ext.action:
                merged[ext.section_type] = self._merge_section_feedbacks(
                    [existing],
                    [ext],
                    prefer_vlm=False,
                )[0]
                continue

            # Build multi-objective scorecards for both candidates
            candidates_map = {"existing": existing, "external": ext}
            scorecards: List[ActionScorecard] = []
            for label, sf_candidate in candidates_map.items():
                details = {}
                for pf in (sf_candidate.paragraph_feedbacks or []):
                    details.setdefault("issues", []).append(pf.model_dump())
                scorecards.append(ActionScorecard(
                    action=sf_candidate.action,
                    section_type=sf_candidate.section_type,
                    source=label,
                    scores=[
                        score_logic(details),
                        score_layout(details),
                        score_citation(details),
                        score_style(details),
                    ],
                ))

            for card in scorecards:
                card.compute_weighted_sum()

            front = compute_pareto_front(scorecards)
            obj_scores_dict = {
                card.source: [s.model_dump() for s in card.scores]
                for card in scorecards
            }

            if len(front) == 1 or not needs_llm_arbitration(front):
                winner = select_action(scorecards)
                chosen = candidates_map[winner.source]
                resolution_method = "pareto_dominant" if len(front) == 1 else "lexicographic"
                conflict_records.append(ConflictResolutionRecord(
                    section_type=ext.section_type,
                    target_id=ext.target_id or existing.target_id or ext.section_type,
                    candidates=[existing.model_dump(), ext.model_dump()],
                    selected_action=chosen.action,
                    selected_source=winner.source,
                    reason=f"Multi-objective {resolution_method} selection (weighted_sum={winner.weighted_sum:.3f})",
                    applied_guardrail=None,
                    objective_scores=obj_scores_dict,
                    pareto_front_size=len(front),
                    resolution_method=resolution_method,
                ))
                merged[ext.section_type] = chosen
            else:
                # LLM arbitration as fallback for ambiguous Pareto front
                chosen = existing
                reason = "Fallback arbitration kept existing action."
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You arbitrate conflicting section revision actions. "
                                    "Prefer preserving compilation validity and page limits. "
                                    "Return JSON only: {selected: 'a'|'b', reason: '...'}."
                                ),
                            },
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "section": ext.section_type,
                                        "candidate_a": existing.model_dump(),
                                        "candidate_b": ext.model_dump(),
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                        ],
                        temperature=0.1,
                        max_tokens=300,
                    )
                    raw = response.choices[0].message.content or ""
                    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
                    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
                    arb = json.loads(cleaned) if cleaned else {}
                    if str(arb.get("selected", "a")).lower() == "b":
                        chosen = ext
                    reason = str(arb.get("reason", "LLM arbitration selected safer action."))
                except Exception:
                    reason = "Fallback arbitration kept existing action."

                conflict_records.append(ConflictResolutionRecord(
                    section_type=ext.section_type,
                    target_id=ext.target_id or existing.target_id or ext.section_type,
                    candidates=[existing.model_dump(), ext.model_dump()],
                    selected_action=chosen.action,
                    selected_source="llm_arbiter",
                    reason=reason,
                    applied_guardrail=None,
                    objective_scores=obj_scores_dict,
                    pareto_front_size=len(front),
                    resolution_method="llm_arbiter",
                ))
                merged[ext.section_type] = chosen

        return list(merged.values()), conflict_records

    # -----------------------------------------------------------------
    # VLM feedback builder
    # -----------------------------------------------------------------

    def _build_vlm_feedback(
        self,
        vlm_result: Dict[str, Any],
        structural_actions: Optional[List[StructuralAction]] = None,
    ) -> Tuple[List[FeedbackResult], List[SectionFeedback]]:
        """
        Build feedback and section revisions from VLM result.
        - **Description**:
            - Converts VLM review output into Reviewer-compatible feedback
            - Maps overflow/underfill to FeedbackResult and section advice to SectionFeedback
            - When structural_actions are provided, enriches revision prompts with
              global strategy context (e.g. what was moved to appendix, resized)

        - **Args**:
            - `vlm_result` (Dict[str, Any]): Raw VLM review result dict
            - `structural_actions` (Optional[List[StructuralAction]]): Planned/executed actions

        - **Returns**:
            - `feedbacks` (List[FeedbackResult]): Aggregated feedback results
            - `section_feedbacks` (List[SectionFeedback]): Per-section revision guidance
        """
        feedbacks: List[FeedbackResult] = []
        section_feedbacks: List[SectionFeedback] = []
        
        if not vlm_result:
            return feedbacks, section_feedbacks

        overflow_pages = vlm_result.get("overflow_pages", 0)
        needs_trim = vlm_result.get("needs_trim", False)
        needs_expand = vlm_result.get("needs_expand", False)
        
        if overflow_pages > 0 or needs_trim:
            feedbacks.append(FeedbackResult(
                checker_name="vlm_review",
                passed=False,
                severity=Severity.ERROR,
                message=vlm_result.get("summary", "Page overflow detected"),
                details={
                    "overflow_pages": overflow_pages,
                    "needs_trim": True,
                    "source": "vlm_review",
                },
            ))
        elif needs_expand:
            feedbacks.append(FeedbackResult(
                checker_name="vlm_review",
                passed=False,
                severity=Severity.WARNING,
                message=vlm_result.get("summary", "Underfill detected"),
                details={
                    "needs_expand": True,
                    "source": "vlm_review",
                },
            ))
        else:
            feedbacks.append(FeedbackResult(
                checker_name="vlm_review",
                passed=True,
                severity=Severity.INFO,
                message=vlm_result.get("summary", "VLM review passed"),
                details={"source": "vlm_review"},
            ))
        
        print(
            "[VLMReview] Summary: "
            f"{vlm_result.get('summary', 'No summary')} | "
            f"overflow_pages={overflow_pages} needs_trim={needs_trim} needs_expand={needs_expand}"
        )
        
        # Build structural context string for enriched revision prompts
        structural_context = None
        if structural_actions:
            ctx_parts = [
                f"This paper exceeds the page limit by {overflow_pages:.1f} pages. "
                "The following structural adjustments have been applied:"
            ]
            moved_count = sum(1 for a in structural_actions if a.action_type in ("move_figure", "move_table"))
            resized_count = sum(1 for a in structural_actions if a.action_type == "resize_figure")
            downgraded_count = sum(1 for a in structural_actions if a.action_type == "downgrade_wide")
            appendix_created = any(a.action_type == "create_appendix" for a in structural_actions)

            if appendix_created:
                ctx_parts.append("- An Appendix section has been created.")
            if moved_count > 0:
                moved_ids = [a.target_id for a in structural_actions if a.action_type in ("move_figure", "move_table")]
                ctx_parts.append(f"- {moved_count} figure(s)/table(s) moved to Appendix: {', '.join(moved_ids)}.")
            if downgraded_count > 0:
                ctx_parts.append(f"- {downgraded_count} wide figure(s) converted to single-column.")
            if resized_count > 0:
                ctx_parts.append(f"- {resized_count} figure(s) resized to smaller width.")

            total_saved = sum(a.estimated_savings for a in structural_actions)
            remaining_trim = max(0, overflow_pages - total_saved)
            if remaining_trim > 0:
                ctx_parts.append(
                    f"After these adjustments, an estimated {remaining_trim:.1f} pages of word-level "
                    "trimming is still needed across sections."
                )
            structural_context = " ".join(ctx_parts)

        section_recommendations = vlm_result.get("section_recommendations", {}) or {}
        for section_type, advice in section_recommendations.items():
            if section_type == "appendix":
                continue
            recommended_action = getattr(advice, "recommended_action", None) or advice.get("recommended_action")
            target_change = getattr(advice, "target_change", None) or advice.get("target_change")
            guidance = getattr(advice, "specific_guidance", None) or advice.get("specific_guidance")
            
            if recommended_action == "trim":
                action = "reduce"
                delta_words = -abs(target_change) if target_change else 0
            elif recommended_action == "expand":
                action = "expand"
                delta_words = abs(target_change) if target_change else 0
            else:
                action = "ok"
                delta_words = 0
            
            # Build per-section structural action descriptors
            section_struct_actions = []
            if structural_actions:
                section_struct_actions = [
                    f"{a.action_type}:{a.target_id}"
                    for a in structural_actions
                    if a.section == section_type
                ]
            
            revision_prompt = self._build_vlm_revision_prompt(
                section_type=section_type,
                action=action,
                delta_words=delta_words,
                guidance=guidance,
                structural_context=structural_context if action == "reduce" else None,
            )
            
            section_feedbacks.append(SectionFeedback(
                section_type=section_type,
                current_word_count=0,
                target_word_count=0,
                action=action,
                delta_words=delta_words,
                revision_prompt=revision_prompt,
                structural_actions=section_struct_actions,
            ))
            print(
                "[VLMReview] Advice: "
                f"section={section_type} action={action} delta_words={delta_words} "
                f"guidance={guidance or 'n/a'}"
                + (f" structural_actions={section_struct_actions}" if section_struct_actions else "")
            )
        
        return feedbacks, section_feedbacks

    # -----------------------------------------------------------------
    # VLM revision prompt builder
    # -----------------------------------------------------------------

    def _build_vlm_revision_prompt(
        self,
        section_type: str,
        action: str,
        delta_words: int,
        guidance: Optional[str] = None,
        structural_context: Optional[str] = None,
    ) -> str:
        """
        Build revision prompt from VLM guidance with optional structural context.
        - **Description**:
            - Produces a detailed revision instruction for the writer agent
            - When structural_context is provided, includes information about
              overall page-reduction strategy so the writer knows what has changed

        - **Args**:
            - `section_type` (str): Section name to revise
            - `action` (str): "reduce" or "expand"
            - `delta_words` (int): Target word change (+/-)
            - `guidance` (Optional[str]): Extra VLM guidance
            - `structural_context` (Optional[str]): Global strategy summary

        - **Returns**:
            - `prompt` (str): Revision instruction prompt
        """
        action_text = "reduce" if action == "reduce" else "expand"
        delta = abs(delta_words) if delta_words else 0

        parts = []

        # Global strategy context (if structural actions were planned)
        if structural_context:
            parts.append(structural_context)

        parts.append(
            f"Revise the {section_type} section to {action_text} by approximately {delta} words."
        )

        if action == "reduce":
            parts.append(
                "Prioritize: (1) removing redundant explanations of figures/tables that may "
                "have been moved to the Appendix, (2) condensing verbose passages, "
                "(3) merging overlapping sentences. Preserve factual consistency, "
                "citations, equations, and LaTeX formatting."
            )
        else:
            parts.append(
                "Prioritize expanding with additional evidence, details, or analysis. "
                "Preserve factual consistency, citations, and LaTeX formatting."
            )

        if guidance:
            parts.append(f"Additional guidance: {guidance}")

        return " ".join(parts)

    # -----------------------------------------------------------------
    # Typesetter feedback builder
    # -----------------------------------------------------------------

    def _build_typesetter_feedback(
        self,
        compile_errors: List[str],
        generated_sections: Dict[str, str],
        section_errors: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[List[FeedbackResult], List[SectionFeedback]]:
        """
        Build feedback from LaTeX compilation errors.
        - **Description**:
            - Converts compilation errors into reviewer-compatible feedback
            - When section_errors is provided (multi-file mode), uses precise
              error-to-section mapping from the LaTeX log file tracking
            - Falls back to heuristic content matching when section_errors is not available

        - **Args**:
            - `compile_errors` (List[str]): Error messages from LaTeX compiler
            - `generated_sections` (Dict[str, str]): Current section contents
            - `section_errors` (Dict[str, List[str]], optional): Pre-mapped section -> errors

        - **Returns**:
            - `feedbacks` (List[FeedbackResult]): Aggregated feedback results
            - `section_feedbacks` (List[SectionFeedback]): Per-section revision guidance
        """
        feedbacks: List[FeedbackResult] = []
        section_feedbacks: List[SectionFeedback] = []
        
        if not compile_errors and not section_errors:
            return feedbacks, section_feedbacks
        
        total_errors = len(compile_errors) if compile_errors else sum(
            len(v) for v in (section_errors or {}).values()
        )
        print(f"[Typesetter] Building feedback from {total_errors} compile errors")
        
        # Build a combined feedback result
        feedbacks.append(FeedbackResult(
            checker_name="typesetter",
            passed=False,
            severity=Severity.ERROR,
            message=f"LaTeX compilation failed with {total_errors} error(s): {'; '.join(compile_errors[:3]) if compile_errors else 'see section_errors'}",
            details={
                "source": "typesetter",
                "compile_errors": compile_errors or [],
                "section_errors": section_errors or {},
            },
        ))
        
        # =================================================================
        # Multi-file mode: precise section_errors mapping available
        # =================================================================
        if section_errors:
            for sec_type, sec_errs in section_errors.items():
                if sec_type not in generated_sections:
                    continue
                if not sec_errs:
                    continue
                
                revision_parts = [
                    "Fix the following LaTeX compilation errors in this section:\n"
                ]
                for err in sec_errs:
                    err_lower = err.lower()
                    matched_fix = False
                    for pattern, fix in LATEX_ERROR_FIXES.items():
                        if pattern in err_lower:
                            revision_parts.append(f"- {err}: {fix}")
                            matched_fix = True
                            break
                    if not matched_fix:
                        revision_parts.append(f"- {err}: Review and correct this LaTeX error.")
                revision_parts.append(
                    "\nOutput ONLY valid LaTeX. Do NOT use unescaped special characters "
                    "(&, %, $, #, _, {, }) in regular text."
                )
                
                section_feedbacks.append(SectionFeedback(
                    section_type=sec_type,
                    current_word_count=len(generated_sections.get(sec_type, "").split()),
                    target_word_count=0,
                    action="fix_latex",
                    delta_words=0,
                    revision_prompt="\n".join(revision_parts),
                ))
                print(f"[Typesetter] Targeted fix (multi-file): section={sec_type} errors={sec_errs[:3]}")
            
            # Handle any errors not attributed to a specific section
            attributed_errors = set()
            for errs in section_errors.values():
                attributed_errors.update(errs)
            unattributed = [e for e in (compile_errors or []) if e not in attributed_errors]
            if unattributed and not section_feedbacks:
                compile_errors = unattributed
            elif section_feedbacks:
                return feedbacks, section_feedbacks
        
        # =================================================================
        # Fallback: heuristic matching or broadcast
        # =================================================================
        if not compile_errors:
            return feedbacks, section_feedbacks
        
        # Collect fix instructions for all errors
        fix_instructions: List[str] = []
        for error in compile_errors:
            error_lower = error.lower()
            for pattern, fix in LATEX_ERROR_FIXES.items():
                if pattern in error_lower:
                    fix_instructions.append(f"Error: {error}\nFix: {fix}")
                    break
            else:
                fix_instructions.append(f"Error: {error}\nFix: Review and correct this LaTeX error.")
        
        # Try to locate errors to specific sections by scanning content
        section_error_map: Dict[str, List[str]] = {}
        for error in compile_errors:
            error_lower = error.lower()
            matched_section = None
            
            if "figure" in error_lower or "includegraphics" in error_lower:
                for section_type, content in generated_sections.items():
                    if "\\begin{figure" in content or "\\includegraphics" in content:
                        matched_section = section_type
                        break
            elif "tabular" in error_lower or "alignment tab" in error_lower:
                for section_type, content in generated_sections.items():
                    if "\\begin{tabular" in content or "\\begin{table" in content or "&" in content:
                        matched_section = section_type
                        break
            
            if matched_section:
                if matched_section not in section_error_map:
                    section_error_map[matched_section] = []
                section_error_map[matched_section].append(error)
        
        if section_error_map:
            for section_type, sec_errs in section_error_map.items():
                revision_parts = [
                    "Fix the following LaTeX compilation errors in this section:\n"
                ]
                for err in sec_errs:
                    err_lower = err.lower()
                    for pattern, fix in LATEX_ERROR_FIXES.items():
                        if pattern in err_lower:
                            revision_parts.append(f"- {err}: {fix}")
                            break
                    else:
                        revision_parts.append(f"- {err}: Review and correct.")
                revision_parts.append(
                    "\nOutput ONLY valid LaTeX. Do NOT use unescaped special characters "
                    "(&, %, $, #, _, {, }) in regular text."
                )
                
                section_feedbacks.append(SectionFeedback(
                    section_type=section_type,
                    current_word_count=len(generated_sections.get(section_type, "").split()),
                    target_word_count=0,
                    action="fix_latex",
                    delta_words=0,
                    revision_prompt="\n".join(revision_parts),
                ))
                print(f"[Typesetter] Targeted fix (heuristic): section={section_type} errors={sec_errs}")
        else:
            # Cannot locate to specific section - broadcast to all sections
            all_fix_prompt = (
                "Fix the following LaTeX compilation errors in this section:\n"
                + "\n".join(f"- {inst}" for inst in fix_instructions)
                + "\n\nOutput ONLY valid LaTeX. Ensure all environments are properly closed. "
                "Escape special characters (&, %, $, #, _, {, }) in regular text."
            )
            for section_type in generated_sections:
                if section_type == "appendix":
                    continue
                section_feedbacks.append(SectionFeedback(
                    section_type=section_type,
                    current_word_count=len(generated_sections.get(section_type, "").split()),
                    target_word_count=0,
                    action="fix_latex",
                    delta_words=0,
                    revision_prompt=all_fix_prompt,
                ))
            print(f"[Typesetter] Broadcast fix to all {len(generated_sections)} sections (excl. appendix)")
        
        return feedbacks, section_feedbacks

    # -----------------------------------------------------------------
    # Merge & resolve helpers
    # -----------------------------------------------------------------

    def _merge_section_feedbacks(
        self,
        base_feedbacks: List[SectionFeedback],
        vlm_feedbacks: List[SectionFeedback],
        prefer_vlm: bool,
    ) -> List[SectionFeedback]:
        """
        Merge section feedbacks with conflict resolution.
        - **Description**:
            - Merges reviewer and VLM section feedback
            - Resolves conflicts based on prefer_vlm flag
        
        - **Args**:
            - `base_feedbacks` (List[SectionFeedback]): Reviewer-driven feedback
            - `vlm_feedbacks` (List[SectionFeedback]): VLM-driven feedback
            - `prefer_vlm` (bool): Whether to override conflicts with VLM advice
        
        - **Returns**:
            - `merged` (List[SectionFeedback]): Merged section feedback list
        """
        merged: Dict[str, SectionFeedback] = {fb.section_type: fb for fb in base_feedbacks}
        
        for fb in vlm_feedbacks:
            existing = merged.get(fb.section_type)
            if not existing:
                merged[fb.section_type] = fb
                continue
            
            # fix_latex always takes priority — compilation must succeed first
            if fb.action == "fix_latex" and existing.action != "fix_latex":
                merged[fb.section_type] = fb
            elif existing.action == "fix_latex":
                # Keep existing fix_latex; append new prompt if also fix_latex
                if fb.action == "fix_latex":
                    existing.revision_prompt += "\n\n" + fb.revision_prompt
            elif existing.action != fb.action and prefer_vlm:
                merged[fb.section_type] = fb
            elif existing.action == fb.action and abs(fb.delta_words) > abs(existing.delta_words):
                merged[fb.section_type] = fb
            else:
                # Preserve paragraph-level targets/instructions from both sources.
                merged_targets = sorted(
                    list(set((existing.target_paragraphs or []) + (fb.target_paragraphs or [])))
                )
                existing.target_paragraphs = merged_targets
                if fb.paragraph_instructions:
                    existing.paragraph_instructions.update(fb.paragraph_instructions)
                if fb.paragraph_feedbacks:
                    existing.paragraph_feedbacks.extend(fb.paragraph_feedbacks)
                if fb.revision_prompt and fb.revision_prompt not in (existing.revision_prompt or ""):
                    if existing.revision_prompt:
                        existing.revision_prompt += "\n\n" + fb.revision_prompt
                    else:
                        existing.revision_prompt = fb.revision_prompt
        
        return list(merged.values())

    def _resolve_section_feedbacks(
        self,
        section_feedbacks: List[SectionFeedback],
        revised_sections: set,
        review_result: ReviewResult,
    ) -> None:
        """
        Mark section feedbacks as resolved after revision.
        - **Description**:
            - Clears revision prompts for sections already revised
            - Updates review_result.requires_revision accordingly
        
        - **Args**:
            - `section_feedbacks` (List[SectionFeedback]): Feedback list to update
            - `revised_sections` (set): Sections that were revised
            - `review_result` (ReviewResult): Review result to update
        
        - **Returns**:
            - `None`
        """
        if not revised_sections:
            return
        
        for sf in section_feedbacks:
            if sf.section_type in revised_sections:
                sf.action = "ok"
                sf.delta_words = 0
                sf.revision_prompt = ""
        
        for section_type in list(review_result.requires_revision.keys()):
            if section_type in revised_sections:
                review_result.requires_revision.pop(section_type, None)
