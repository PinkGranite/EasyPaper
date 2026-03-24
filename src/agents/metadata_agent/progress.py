"""
Progress Callback Infrastructure for MetaData Paper Generation
- **Description**:
    - Defines structured event types for real-time progress reporting
    - Provides a ProgressEmitter class that wraps an async callback
    - Used by MetaDataAgent.generate_paper() to emit SSE-compatible events
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional


# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

class EventType:
    GENERATION_STARTED = "generation_started"
    PHASE_START = "phase_start"
    PHASE_COMPLETE = "phase_complete"
    SECTION_START = "section_start"
    SECTION_CONTENT = "section_content"
    THINKING = "thinking"
    PLAN_CREATED = "plan_created"
    REFERENCES_DISCOVERED = "references_discovered"
    REVIEW_START = "review_start"
    REVIEW_RESULT = "review_result"
    REVIEW_FEEDBACK_REQUEST = "review_feedback_request"
    REVIEW_FEEDBACK_RECEIVED = "review_feedback_received"
    REVISION_APPLIED = "revision_applied"
    COMPILE_START = "compile_start"
    COMPILE_COMPLETE = "compile_complete"
    VLM_REVIEW = "vlm_review"
    ERROR = "error"
    COMPLETED = "completed"
    # Fine-grained events for richer streaming
    LLM_RESPONSE = "llm_response"
    SEARCH_STARTED = "search_started"
    SEARCH_RESULT = "search_result"
    TOOL_CALL = "tool_call"
    AGENT_STEP = "agent_step"
    REF_ASSIGNED = "ref_assigned"
    LOG = "log"
    GEN_UI = "gen_ui"
    ARTIFACT_SAVED = "artifact_saved"
    # Paragraph-level DAG progress (decomposed body generation)
    PARAGRAPH_START = "paragraph_start"
    PARAGRAPH_CONTENT = "paragraph_content"
    CLAIM_VERIFY_RESULT = "claim_verify_result"


# Phase identifiers
class Phase:
    CODE_CONTEXT = "code_context"
    PLANNING = "planning"
    REF_DISCOVERY = "ref_discovery"
    REF_ASSIGNMENT = "ref_assignment"
    RESEARCH_CONTEXT = "research_context"
    TABLE_CONVERSION = "table_conversion"
    INTRODUCTION = "introduction"
    BODY_SECTIONS = "body_sections"
    SYNTHESIS = "synthesis"
    REVIEW_LOOP = "review_loop"
    PDF_COMPILE = "pdf_compile"
    VLM_REVIEW = "vlm_review"
    FINALIZE = "finalize"


ProgressCallback = Callable[
    [Dict[str, Any]], Coroutine[Any, Any, None]
]


class ProgressEmitter:
    """
    Wraps an async progress callback for structured event emission.

    - **Description**:
        - Provides typed helper methods for each event kind.
        - Serializes events as dicts suitable for JSON / SSE transmission.
        - No-ops gracefully when no callback is provided.

    - **Args**:
        - `callback` (ProgressCallback | None): Async function receiving event dicts.
    """

    def __init__(self, callback: Optional[ProgressCallback] = None):
        self._callback = callback
        self._start_time = datetime.now(timezone.utc)

    def _ts(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _elapsed_s(self) -> float:
        return (datetime.now(timezone.utc) - self._start_time).total_seconds()

    async def _emit(self, event: Dict[str, Any]) -> None:
        if self._callback is not None:
            event.setdefault("timestamp", self._ts())
            event.setdefault("elapsed_seconds", round(self._elapsed_s(), 1))
            await self._callback(event)

    async def generation_started(
        self, title: str, target_pages: Optional[int] = None, **extra: Any
    ) -> None:
        await self._emit({
            "type": EventType.GENERATION_STARTED,
            "title": title,
            "target_pages": target_pages,
            **extra,
        })

    async def phase_start(
        self, phase: str, description: str = "", **extra: Any
    ) -> None:
        await self._emit({
            "type": EventType.PHASE_START,
            "phase": phase,
            "description": description,
            **extra,
        })

    async def phase_complete(
        self, phase: str, summary: str = "", **extra: Any
    ) -> None:
        await self._emit({
            "type": EventType.PHASE_COMPLETE,
            "phase": phase,
            "summary": summary,
            **extra,
        })

    async def section_start(
        self, section_type: str, phase: str = "", **extra: Any
    ) -> None:
        await self._emit({
            "type": EventType.SECTION_START,
            "section_type": section_type,
            "phase": phase,
            **extra,
        })

    async def section_content(
        self,
        section_type: str,
        content: str,
        word_count: int = 0,
        phase: str = "",
        **extra: Any,
    ) -> None:
        await self._emit({
            "type": EventType.SECTION_CONTENT,
            "section_type": section_type,
            "content": content,
            "word_count": word_count,
            "phase": phase,
            **extra,
        })

    async def paragraph_start(
        self,
        section_type: str,
        paragraph_index: int,
        claim_id: str,
        total_paragraphs: int = 0,
        phase: str = "",
        **extra: Any,
    ) -> None:
        await self._emit({
            "type": EventType.PARAGRAPH_START,
            "section_type": section_type,
            "paragraph_index": paragraph_index,
            "claim_id": claim_id,
            "total_paragraphs": total_paragraphs,
            "phase": phase,
            **extra,
        })

    async def paragraph_content(
        self,
        section_type: str,
        paragraph_index: int,
        claim_id: str,
        content: str,
        word_count: int = 0,
        phase: str = "",
        **extra: Any,
    ) -> None:
        await self._emit({
            "type": EventType.PARAGRAPH_CONTENT,
            "section_type": section_type,
            "paragraph_index": paragraph_index,
            "claim_id": claim_id,
            "content": content,
            "word_count": word_count,
            "phase": phase,
            **extra,
        })

    async def claim_verify_result(
        self,
        section_type: str,
        paragraph_index: int,
        claim_id: str,
        passed: bool,
        attempt: int,
        max_attempts: int,
        feedback_summary: str = "",
        phase: str = "",
        **extra: Any,
    ) -> None:
        await self._emit({
            "type": EventType.CLAIM_VERIFY_RESULT,
            "section_type": section_type,
            "paragraph_index": paragraph_index,
            "claim_id": claim_id,
            "passed": passed,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "feedback_summary": feedback_summary,
            "phase": phase,
            **extra,
        })

    async def thinking(
        self,
        content: str,
        section_type: str = "",
        phase: str = "",
        agent: str = "",
        **extra: Any,
    ) -> None:
        await self._emit({
            "type": EventType.THINKING,
            "content": content,
            "section_type": section_type,
            "phase": phase,
            "agent": agent,
            **extra,
        })

    async def plan_created(
        self,
        sections: int,
        estimated_words: int,
        plan_summary: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        await self._emit({
            "type": EventType.PLAN_CREATED,
            "sections": sections,
            "estimated_words": estimated_words,
            "plan_summary": plan_summary,
            **extra,
        })

    async def references_discovered(
        self, count: int, total_pool: int = 0, **extra: Any
    ) -> None:
        await self._emit({
            "type": EventType.REFERENCES_DISCOVERED,
            "count": count,
            "total_pool": total_pool,
            **extra,
        })

    async def review_start(
        self, iteration: int, max_iterations: int, **extra: Any
    ) -> None:
        await self._emit({
            "type": EventType.REVIEW_START,
            "iteration": iteration,
            "max_iterations": max_iterations,
            **extra,
        })

    async def review_result(
        self,
        iteration: int,
        passed: bool,
        issues_count: int = 0,
        summary: str = "",
        revision_tasks: Optional[List[Dict[str, Any]]] = None,
        **extra: Any,
    ) -> None:
        await self._emit({
            "type": EventType.REVIEW_RESULT,
            "iteration": iteration,
            "passed": passed,
            "issues_count": issues_count,
            "summary": summary,
            "revision_tasks": revision_tasks,
            **extra,
        })

    async def review_feedback_request(
        self,
        iteration: int,
        review_summary: str = "",
        sections_status: Optional[Dict[str, Any]] = None,
        requires_revision: Optional[Dict[str, list]] = None,
        section_feedbacks: Optional[list] = None,
        revision_tasks: Optional[list] = None,
        current_sections: Optional[Dict[str, str]] = None,
        checkpoint_path: Optional[str] = None,
        **extra: Any,
    ) -> None:
        await self._emit({
            "type": EventType.REVIEW_FEEDBACK_REQUEST,
            "iteration": iteration,
            "review_summary": review_summary,
            "sections_status": sections_status,
            "requires_revision": requires_revision,
            "section_feedbacks": section_feedbacks,
            "revision_tasks": revision_tasks,
            "current_sections": current_sections,
            "checkpoint_path": checkpoint_path,
            **extra,
        })

    async def review_feedback_received(
        self, feedback_text: str = "", action: str = "continue", **extra: Any
    ) -> None:
        await self._emit({
            "type": EventType.REVIEW_FEEDBACK_RECEIVED,
            "feedback_text": feedback_text,
            "action": action,
            **extra,
        })

    async def revision_applied(
        self,
        section_type: str,
        iteration: int,
        word_count: int = 0,
        **extra: Any,
    ) -> None:
        await self._emit({
            "type": EventType.REVISION_APPLIED,
            "section_type": section_type,
            "iteration": iteration,
            "word_count": word_count,
            **extra,
        })

    async def compile_start(self, **extra: Any) -> None:
        await self._emit({
            "type": EventType.COMPILE_START,
            **extra,
        })

    async def compile_complete(
        self, success: bool, pdf_path: Optional[str] = None, **extra: Any
    ) -> None:
        await self._emit({
            "type": EventType.COMPILE_COMPLETE,
            "success": success,
            "pdf_path": pdf_path,
            **extra,
        })

    async def vlm_review(
        self, passed: bool, summary: str = "", **extra: Any
    ) -> None:
        await self._emit({
            "type": EventType.VLM_REVIEW,
            "passed": passed,
            "summary": summary,
            **extra,
        })

    async def error(self, message: str, phase: str = "", **extra: Any) -> None:
        await self._emit({
            "type": EventType.ERROR,
            "message": message,
            "phase": phase,
            **extra,
        })

    async def completed(
        self,
        status: str,
        total_words: int = 0,
        review_iterations: int = 0,
        sections_count: int = 0,
        pdf_path: Optional[str] = None,
        paper_dir: Optional[str] = None,
        **extra: Any,
    ) -> None:
        await self._emit({
            "type": EventType.COMPLETED,
            "status": status,
            "total_words": total_words,
            "review_iterations": review_iterations,
            "sections_count": sections_count,
            "pdf_path": pdf_path,
            "paper_dir": paper_dir,
            **extra,
        })

    # ---- Fine-grained events ----

    async def search_started(
        self,
        query: str,
        source: str = "",
        section: str = "",
        **extra: Any,
    ) -> None:
        await self._emit({
            "type": EventType.SEARCH_STARTED,
            "query": query,
            "source": source,
            "section": section,
            **extra,
        })

    async def search_result(
        self,
        found: int,
        new_count: int = 0,
        section: str = "",
        query: str = "",
        **extra: Any,
    ) -> None:
        await self._emit({
            "type": EventType.SEARCH_RESULT,
            "found": found,
            "new_count": new_count,
            "section": section,
            "query": query,
            **extra,
        })

    async def tool_call(
        self,
        tool_name: str,
        result_summary: str = "",
        section: str = "",
        agent: str = "",
        **extra: Any,
    ) -> None:
        await self._emit({
            "type": EventType.TOOL_CALL,
            "tool_name": tool_name,
            "result_summary": result_summary,
            "section": section,
            "agent": agent,
            **extra,
        })

    async def agent_step(
        self,
        agent: str,
        iteration: int = 0,
        description: str = "",
        section: str = "",
        phase: str = "",
        **extra: Any,
    ) -> None:
        await self._emit({
            "type": EventType.AGENT_STEP,
            "agent": agent,
            "iteration": iteration,
            "description": description,
            "section": section,
            "phase": phase,
            **extra,
        })

    async def ref_assigned(
        self,
        section: str,
        count: int = 0,
        **extra: Any,
    ) -> None:
        await self._emit({
            "type": EventType.REF_ASSIGNED,
            "section": section,
            "count": count,
            **extra,
        })

    async def log(
        self,
        message: str,
        level: str = "info",
        phase: str = "",
        **extra: Any,
    ) -> None:
        await self._emit({
            "type": EventType.LOG,
            "message": message,
            "level": level,
            "phase": phase,
            **extra,
        })

    async def artifact_saved(
        self,
        relative_path: str,
        absolute_path: str,
        category: str,
        size: int,
        mime_type: str = "application/octet-stream",
        label: str = "",
        storage_key: str = "",
        **extra: Any,
    ) -> None:
        """
        Emit an event indicating that an artifact file was saved.

        - **Args**:
            - `relative_path` (str): Path relative to paper_dir.
            - `absolute_path` (str): Absolute filesystem path.
            - `category` (str): Artifact category (planning, references, section, etc.).
            - `size` (int): File size in bytes.
            - `mime_type` (str): MIME type of the file.
            - `label` (str): Human-readable label for the artifact.
            - `storage_key` (str): OSS object key if uploaded directly by agentsys.
        """
        await self._emit({
            "type": EventType.ARTIFACT_SAVED,
            "relative_path": relative_path,
            "absolute_path": absolute_path,
            "category": category,
            "size": size,
            "mime_type": mime_type,
            "label": label,
            "storage_key": storage_key,
            **extra,
        })

    async def emit_gen_ui(
        self,
        component: str,
        props: Dict[str, Any],
        **extra: Any,
    ) -> None:
        await self._emit({
            "type": EventType.GEN_UI,
            "component": component,
            "props": props,
            **extra,
        })
