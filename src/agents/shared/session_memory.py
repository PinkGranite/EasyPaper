"""
Session Memory - Shared state for a single paper generation session.
- **Description**:
    - Provides a unified memory object for cross-agent coordination
    - Stores plan, generated sections, review history, and agent logs
    - Review parts persist to files in the output directory
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field

logger = logging.getLogger("uvicorn.error")


# =========================================================================
# Data models
# =========================================================================

class ReviewEntry(BaseModel):
    """Human-readable review log entry (one per section per iteration)."""
    iteration: int
    target: str
    review_comment: str
    feedback: str


class ReviewRecord(BaseModel):
    """Internal record of a single review iteration (kept for orchestration)."""
    iteration: int
    reviewer: str
    timestamp: str = ""
    passed: bool = False
    feedback_summary: str = ""
    section_feedbacks: Dict[str, Any] = Field(default_factory=dict)
    actions_taken: List[str] = Field(default_factory=list)
    result_snapshot: Dict[str, int] = Field(default_factory=dict)

    def to_review_entries(self) -> List[ReviewEntry]:
        """Flatten this record into human-readable ReviewEntry items."""
        entries: List[ReviewEntry] = []
        for section, fb in self.section_feedbacks.items():
            if not isinstance(fb, dict):
                continue
            action = fb.get("action", "ok")
            comment_parts: List[str] = []
            if self.feedback_summary:
                comment_parts.append(self.feedback_summary)
            para_fbs = fb.get("paragraph_feedbacks", [])
            for pf in para_fbs:
                if isinstance(pf, dict) and pf.get("feedback"):
                    comment_parts.append(pf["feedback"])
            review_comment = "; ".join(comment_parts) if comment_parts else "No issues."
            revised = section in {
                a.replace("revised:", "") for a in self.actions_taken
            }
            if revised:
                feedback_str = f"Revised ({action})"
            elif action == "ok":
                feedback_str = "Passed, no changes needed"
            else:
                feedback_str = f"Flagged ({action}), not revised this iteration"
            entries.append(ReviewEntry(
                iteration=self.iteration,
                target=section,
                review_comment=review_comment[:500],
                feedback=feedback_str,
            ))
        if not entries and self.feedback_summary:
            entries.append(ReviewEntry(
                iteration=self.iteration,
                target="overall",
                review_comment=self.feedback_summary[:500],
                feedback="passed" if self.passed else "issues found",
            ))
        return entries


class AgentLogEntry(BaseModel):
    """Log entry for agent activity tracking."""
    agent: str
    phase: str
    timestamp: str = ""
    action: str = ""
    narrative: str = ""
    communication: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = Field(default_factory=dict)


# =========================================================================
# Session Memory
# =========================================================================

class SessionMemory:
    """
    Shared memory for one paper generation session.
    - **Description**:
        - Created at the start of generate_paper()
        - Passed to all phase methods for cross-agent coordination
        - Review history and logs are persisted to disk at the end
    """

    def __init__(self) -> None:
        self.plan: Optional[Any] = None
        self.generated_sections: Dict[str, str] = {}
        self.contributions: List[str] = []
        self.review_history: List[ReviewRecord] = []
        self.agent_logs: List[AgentLogEntry] = []

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_section(self, section_type: str) -> Optional[str]:
        """Get generated content for a section."""
        return self.generated_sections.get(section_type)

    def get_latest_review(self) -> Optional[ReviewRecord]:
        """Get the most recent review record."""
        return self.review_history[-1] if self.review_history else None

    def get_review_history_for_section(
        self, section_type: str,
    ) -> List[ReviewRecord]:
        """Get review records that mention a specific section."""
        return [
            r for r in self.review_history
            if section_type in r.section_feedbacks
        ]

    def has_been_revised(self, section_type: str) -> bool:
        """Check whether a section has been revised in any review iteration."""
        for record in self.review_history:
            if section_type in record.section_feedbacks:
                fb = record.section_feedbacks[section_type]
                if isinstance(fb, dict) and fb.get("action") not in ("ok", None):
                    return True
        return False

    def get_revision_count(self, section_type: str) -> int:
        """Count how many times a section has been revised."""
        count = 0
        for record in self.review_history:
            if section_type in record.section_feedbacks:
                fb = record.section_feedbacks[section_type]
                if isinstance(fb, dict) and fb.get("action") not in ("ok", None):
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Update interface
    # ------------------------------------------------------------------

    def update_section(self, section_type: str, content: str) -> None:
        """Store or update generated content for a section."""
        self.generated_sections[section_type] = content

    def add_review(self, record: ReviewRecord) -> None:
        """Append a review record."""
        if not record.timestamp:
            record.timestamp = datetime.now().isoformat()
        self.review_history.append(record)

    def log(
        self,
        agent: str,
        phase: str,
        action: str,
        narrative: str = "",
        communication: Optional[Dict[str, Any]] = None,
        **details: Any,
    ) -> None:
        """Log an agent activity with optional human-readable narrative."""
        self.agent_logs.append(AgentLogEntry(
            agent=agent,
            phase=phase,
            timestamp=datetime.now().isoformat(),
            action=action,
            narrative=narrative,
            communication=communication,
            details=details,
        ))

    # ------------------------------------------------------------------
    # Context generation for LLM prompts
    # ------------------------------------------------------------------

    def get_writing_context(self, section_type: str) -> str:
        """
        Build structured context for writing a specific section.
        - **Description**:
            - Aggregates plan guidance, cross-section summaries, and
              contributions into a text block suitable for LLM prompts.

        - **Args**:
            - `section_type` (str): The section being written

        - **Returns**:
            - `context` (str): Multi-line context string
        """
        parts: List[str] = []

        # Plan guidance for this section
        if self.plan is not None:
            for sp in getattr(self.plan, "sections", []):
                if getattr(sp, "section_type", None) == section_type:
                    parts.append("## Plan Guidance for This Section")
                    if getattr(sp, "writing_guidance", ""):
                        parts.append(f"Writing guidance: {sp.writing_guidance}")
                    paras = getattr(sp, "paragraphs", [])
                    if paras:
                        parts.append(f"Expected paragraphs: {len(paras)}")
                        for idx, p in enumerate(paras, 1):
                            kp = getattr(p, "key_point", "")
                            role = getattr(p, "role", "")
                            sents = getattr(p, "approx_sentences", 0)
                            parts.append(
                                f"  P{idx}: [{role}] {kp} (~{sents} sentences)"
                            )
                    break

        # Contributions discovered so far
        if self.contributions:
            parts.append("## Key Contributions")
            for c in self.contributions:
                parts.append(f"- {c}")

        # Summaries of already-written sections
        summaries = self._build_section_summaries(exclude=section_type)
        if summaries:
            parts.append("## Already-Written Sections")
            parts.append(summaries)

        return "\n".join(parts) if parts else ""

    def get_revision_context(self, section_type: str) -> str:
        """
        Build context for revising a section based on review history.
        - **Description**:
            - Provides revision count, prior feedback summaries, and
              known unresolved issues to prevent regression.

        - **Args**:
            - `section_type` (str): The section being revised

        - **Returns**:
            - `context` (str): Multi-line revision history string
        """
        history = self.get_review_history_for_section(section_type)
        if not history:
            return ""

        rev_count = self.get_revision_count(section_type)
        parts: List[str] = [
            f"## Revision History for '{section_type}' (revised {rev_count} time(s) so far)"
        ]

        for rec in history:
            fb = rec.section_feedbacks.get(section_type, {})
            summary_bits: List[str] = []
            if isinstance(fb, dict):
                if fb.get("message"):
                    summary_bits.append(fb["message"])
                for pf in fb.get("paragraph_feedbacks", []):
                    if isinstance(pf, dict) and pf.get("feedback"):
                        summary_bits.append(
                            f"  - P{pf.get('paragraph_index', '?')}: {pf['feedback']}"
                        )
            if summary_bits:
                parts.append(
                    f"Iteration {rec.iteration} ({rec.reviewer}): "
                    + "; ".join(summary_bits[:6])
                )

        parts.append(
            "IMPORTANT: Do NOT regress on issues already fixed in earlier revisions."
        )
        return "\n".join(parts)

    def get_cross_section_summary(self) -> str:
        """
        Return a compact summary of all written sections.
        - **Description**:
            - Shows first and last sentence + word count for each section
            - Useful for synthesis sections (abstract, conclusion)

        - **Returns**:
            - `summary` (str): Multi-line summary string
        """
        return self._build_section_summaries()

    def to_review_context_dict(self) -> Dict[str, Any]:
        """
        Build a serializable memory snapshot for HTTP-based agents.
        - **Description**:
            - Contains plan section summaries, review history digest,
              per-section word counts, and contributions

        - **Returns**:
            - `context` (dict): JSON-serializable dictionary
        """
        # Plan section overview
        plan_sections: List[Dict[str, Any]] = []
        if self.plan is not None:
            for sp in getattr(self.plan, "sections", []):
                plan_sections.append({
                    "section_type": getattr(sp, "section_type", ""),
                    "num_paragraphs": len(getattr(sp, "paragraphs", [])),
                    "estimated_words": (
                        sp.get_estimated_words()
                        if hasattr(sp, "get_estimated_words") else 0
                    ),
                    "key_points": (
                        sp.get_key_points()
                        if hasattr(sp, "get_key_points") else []
                    ),
                })

        # Prior review issues (last two iterations)
        prior_issues: List[Dict[str, Any]] = []
        recent = self.review_history[-2:] if self.review_history else []
        for rec in recent:
            prior_issues.append({
                "iteration": rec.iteration,
                "reviewer": rec.reviewer,
                "passed": rec.passed,
                "feedback_summary": rec.feedback_summary,
                "actions_taken": rec.actions_taken,
            })

        # Per-section word counts
        word_counts: Dict[str, int] = {}
        for stype, content in self.generated_sections.items():
            word_counts[stype] = len(content.split())

        return {
            "plan_sections": plan_sections,
            "prior_issues": prior_issues,
            "word_counts": word_counts,
            "contributions": self.contributions,
        }

    # ------------------------------------------------------------------
    # Unified search (used by AskTool)
    # ------------------------------------------------------------------

    _llm_refine: Optional[Any] = None

    def set_llm_refine(self, refine_fn) -> None:
        """
        Inject an LLM refinement callable for semantic search.
        - **Description**:
            - When set, search() uses a two-stage pipeline: rule-based
              candidate gathering followed by LLM-based semantic refinement.
            - When not set, search() falls back to rule-only candidates.

        - **Args**:
            - `refine_fn` (async callable): Signature
              ``async (question: str, context: str) -> str``
        """
        self._llm_refine = refine_fn

    async def search(self, question: str, scope: str = "all") -> str:
        """
        Two-stage search over session memory.
        - **Description**:
            - Stage 1 (rule filter): gather compact candidate snippets
              via keyword matching with strict token budgets.
            - Stage 2 (LLM refine): if an LLM refine callable has been
              injected via set_llm_refine(), pass the candidates + question
              to the LLM for semantic understanding and precise answers.
            - Falls back to Stage-1-only when no LLM is available.

        - **Args**:
            - `question` (str): Natural-language question
            - `scope` (str): "plan", "sections", "reviews",
              "contributions", or "all"

        - **Returns**:
            - `result` (str): Answer text
        """
        candidates = self._gather_candidates(question, scope)
        if not candidates:
            return ""

        if self._llm_refine is not None:
            try:
                return await self._llm_refine(question, candidates)
            except Exception as e:
                logger.warning("session_memory.llm_refine failed: %s", e)
                return candidates

        return candidates

    # ------------------------------------------------------------------
    # Stage 1: Rule-based candidate gathering (token-budgeted)
    # ------------------------------------------------------------------

    def _gather_candidates(self, question: str, scope: str = "all") -> str:
        """
        Gather compact candidate snippets via keyword matching.
        - **Description**:
            - Each scope has a strict output budget to keep total context
              under ~1000 tokens for a typical 7-section paper.
            - Plan: section_type + paragraph_count + guidance first sentence
            - Sections: first + last sentence + word count
            - Reviews: last 2 iterations, feedback_summary only
            - Contributions: full list (typically short)

        - **Args**:
            - `question` (str): The search question
            - `scope` (str): Which areas to search

        - **Returns**:
            - `text` (str): Compact candidate context
        """
        keywords = [w.lower() for w in question.split() if len(w) > 2]
        parts: List[str] = []

        if scope in ("plan", "all"):
            parts.extend(self._candidates_plan(keywords))

        if scope in ("sections", "all"):
            parts.extend(self._candidates_sections(keywords))

        if scope in ("reviews", "all"):
            parts.extend(self._candidates_reviews(keywords))

        if scope in ("contributions", "all"):
            if self.contributions:
                parts.append("## Contributions")
                for c in self.contributions:
                    parts.append(f"- {c}")

        return "\n".join(parts) if parts else ""

    def _candidates_plan(self, keywords: List[str]) -> List[str]:
        """Compact plan candidates: type + paragraph count + guidance snippet."""
        if self.plan is None:
            return []
        hits: List[str] = []
        for sp in getattr(self.plan, "sections", []):
            stype = getattr(sp, "section_type", "")
            guidance = getattr(sp, "writing_guidance", "") or ""
            paras = getattr(sp, "paragraphs", [])
            para_texts = " ".join(getattr(p, "key_point", "") for p in paras)
            full = f"{stype} {guidance} {para_texts}".lower()
            if not keywords or any(kw in full for kw in keywords):
                guidance_snippet = guidance.split(".")[0] if guidance else ""
                n_paras = len(paras)
                kp_list = ", ".join(
                    getattr(p, "key_point", "")[:60] for p in paras[:4]
                )
                line = f"- {stype}: {n_paras} paragraphs"
                if guidance_snippet:
                    line += f", guidance: \"{guidance_snippet}\""
                if kp_list:
                    line += f", key points: [{kp_list}]"
                figs = getattr(sp, "figure_placements", [])
                if figs:
                    fig_ids = ", ".join(getattr(f, "figure_id", "") for f in figs)
                    line += f", figures: [{fig_ids}]"
                hits.append(line)
        if hits:
            return ["## Plan"] + hits
        return []

    def _candidates_sections(self, keywords: List[str]) -> List[str]:
        """Compact section candidates: first + last sentence + word count."""
        hits: List[str] = []
        for stype, content in self.generated_sections.items():
            if not content.strip():
                continue
            content_lower = content.lower()
            if not keywords or any(kw in content_lower for kw in keywords):
                wc = len(content.split())
                sentences = [
                    s.strip() for s in content.replace("\n", " ").split(".")
                    if s.strip()
                ]
                first = (sentences[0][:120] + ".") if sentences else ""
                last = (sentences[-1][:120] + ".") if len(sentences) > 1 else ""
                preview = first
                if last and last != first:
                    preview += f" ... {last}"
                hits.append(f"- {stype} ({wc} words): {preview}")
        if hits:
            return ["## Sections"] + hits
        return []

    def _candidates_reviews(self, keywords: List[str]) -> List[str]:
        """Compact review candidates: last 2 iterations, summary only."""
        if not self.review_history:
            return []
        recent = self.review_history[-2:]
        hits: List[str] = []
        for rec in recent:
            rec_text = f"{rec.feedback_summary} {' '.join(rec.actions_taken)}".lower()
            if not keywords or any(kw in rec_text for kw in keywords):
                line = (
                    f"- Iter {rec.iteration} ({rec.reviewer}): "
                    f"passed={rec.passed}"
                )
                if rec.feedback_summary:
                    summary = rec.feedback_summary[:200]
                    line += f", \"{summary}\""
                hits.append(line)
        if hits:
            return ["## Reviews"] + hits
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_section_summaries(self, exclude: str = "") -> str:
        """Build short summaries (first+last sentence, word count) of written sections."""
        lines: List[str] = []
        for stype, content in self.generated_sections.items():
            if stype == exclude or not content.strip():
                continue
            wc = len(content.split())
            sentences = [s.strip() for s in content.replace("\n", " ").split(".") if s.strip()]
            first = (sentences[0] + ".") if sentences else ""
            last = (sentences[-1] + ".") if len(sentences) > 1 else ""
            preview = first
            if last and last != first:
                preview += f" ... {last}"
            if len(preview) > 300:
                preview = preview[:297] + "..."
            lines.append(f"- **{stype}** ({wc} words): {preview}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist_reviews(self, output_dir: Path) -> None:
        """
        Save review history as a flat, human-readable list of ReviewEntry items.

        - **Args**:
            - `output_dir` (Path): Paper output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "review_history.json"
        entries: list = []
        for record in self.review_history:
            entries.extend(e.model_dump() for e in record.to_review_entries())
        path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("session_memory.persisted_reviews path=%s entries=%d", path, len(entries))

    def persist_logs(self, output_dir: Path) -> None:
        """
        Save agent logs to output_dir/agent_logs.json.

        - **Args**:
            - `output_dir` (Path): Paper output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "agent_logs.json"
        data = [e.model_dump() for e in self.agent_logs]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("session_memory.persisted_logs path=%s count=%d", path, len(data))

    def persist_all(self, output_dir: Path) -> None:
        """Persist both reviews and logs."""
        self.persist_reviews(output_dir)
        self.persist_logs(output_dir)
