"""
Reviewer Agent
- **Description**:
    - Coordinates feedback checkers for paper review
    - Provides iterative feedback loop support
    - Extensible architecture for adding new checkers
"""
import logging
from typing import List, Dict, Any, Optional, Type, TYPE_CHECKING
from fastapi import APIRouter
from openai import AsyncOpenAI

from ..base import BaseAgent
from ...config.schema import ModelConfig
from .models import (
    ReviewContext,
    ReviewResult,
    FeedbackResult,
    Severity,
    SectionFeedback,
    ParagraphFeedback,
    HierarchicalFeedbackItem,
    FeedbackLevel,
)
from .checkers.base import FeedbackChecker
from .checkers.word_count import WordCountChecker
from .checkers.style_check import StyleChecker
from .checkers.logic_check import LogicChecker

if TYPE_CHECKING:
    from ...skills.registry import SkillRegistry


logger = logging.getLogger("uvicorn.error")


class ReviewerAgent(BaseAgent):
    """
    Reviewer Agent for paper feedback
    - **Description**:
        - Manages multiple feedback checkers
        - Coordinates review process
        - Generates revision guidance
    """
    
    # Default checkers — WordCountChecker removed; word count is now
    # an informational metric only, not a hard constraint.
    DEFAULT_CHECKERS: List[Type[FeedbackChecker]] = []
    
    def __init__(
        self,
        config: ModelConfig,
        skill_registry: Optional["SkillRegistry"] = None,
    ):
        """
        Initialize the Reviewer Agent.

        - **Args**:
            - `config` (ModelConfig): Model configuration
            - `skill_registry` (SkillRegistry, optional): Global skill registry
              for loading checker rules and anti-patterns
        """
        self.config = config
        self.model_name = config.model_name
        self._checkers: List[FeedbackChecker] = []
        self._skill_registry = skill_registry
        self._router = self._create_router()
        
        # Register default checkers
        for checker_cls in self.DEFAULT_CHECKERS:
            self.register_checker(checker_cls())
        
        # Register skill-based checkers
        self._register_skill_checkers()
        
        logger.info(
            "ReviewerAgent initialized with %d checkers: %s",
            len(self._checkers),
            [c.name for c in self._checkers]
        )

    def _register_skill_checkers(self) -> None:
        """
        Dynamically register StyleChecker and LogicChecker.

        - **Description**:
            - StyleChecker is always registered (works with or without registry)
            - LogicChecker is registered only when an LLM client can be created
        """
        # StyleChecker: pure rule-based, always available
        self.register_checker(StyleChecker(skill_registry=self._skill_registry))

        # LogicChecker: needs LLM client
        try:
            llm_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
            self.register_checker(
                LogicChecker(
                    llm_client=llm_client,
                    model_name=self.model_name,
                    skill_registry=self._skill_registry,
                )
            )
        except Exception as e:
            logger.warning(
                "ReviewerAgent: could not initialize LogicChecker: %s", e
            )
    
    @property
    def name(self) -> str:
        return "reviewer"
    
    @property
    def description(self) -> str:
        return "Reviews paper content and provides feedback for improvement"
    
    @property
    def router(self) -> APIRouter:
        return self._router
    
    @property
    def endpoints_info(self) -> List[Dict[str, Any]]:
        return [
            {
                "path": "/agent/reviewer/review",
                "method": "POST",
                "description": "Review paper and provide feedback",
            },
            {
                "path": "/agent/reviewer/checkers",
                "method": "GET",
                "description": "List registered checkers",
            },
            {
                "path": "/agent/reviewer/health",
                "method": "GET",
                "description": "Health check",
            },
        ]
    
    def _create_router(self) -> APIRouter:
        """Create FastAPI router"""
        from .router import create_reviewer_router
        return create_reviewer_router(self)
    
    def register_checker(self, checker: FeedbackChecker) -> None:
        """
        Register a feedback checker
        - **Args**:
            - `checker`: FeedbackChecker instance to register
        """
        # Check for duplicate names
        for existing in self._checkers:
            if existing.name == checker.name:
                logger.warning(
                    "Checker '%s' already registered, skipping",
                    checker.name
                )
                return
        
        self._checkers.append(checker)
        # Sort by priority
        self._checkers.sort(key=lambda c: c.priority)
        logger.info("Registered checker: %s (priority=%d)", checker.name, checker.priority)
    
    def unregister_checker(self, name: str) -> bool:
        """
        Unregister a checker by name
        - **Args**:
            - `name`: Name of checker to remove
        - **Returns**:
            - `bool`: True if removed, False if not found
        """
        for i, checker in enumerate(self._checkers):
            if checker.name == name:
                self._checkers.pop(i)
                logger.info("Unregistered checker: %s", name)
                return True
        return False
    
    def get_checkers(self) -> List[Dict[str, Any]]:
        """Get list of registered checkers"""
        return [
            {
                "name": c.name,
                "priority": c.priority,
                "enabled": c.enabled,
                "class": c.__class__.__name__,
            }
            for c in self._checkers
        ]
    
    async def answer(self, question: str, memory=None) -> str:
        """
        Quick consultation — answer a writing quality or consistency question.
        - **Description**:
            - Uses a lightweight LLM call with focused context extracted
              from SessionMemory.
            - Designed to be called via AskTool during WriterAgent's
              ReAct loop.

        - **Args**:
            - `question` (str): The question to answer
            - `memory` (SessionMemory, optional): Session memory for context

        - **Returns**:
            - `answer` (str): Brief assessment or guidance
        """
        context_parts: List[str] = []
        if memory is not None:
            for stype, content in getattr(memory, "generated_sections", {}).items():
                if content.strip():
                    preview = content[:600] + ("..." if len(content) > 600 else "")
                    context_parts.append(f"[{stype}]: {preview}")
            for rec in getattr(memory, "review_history", [])[-2:]:
                context_parts.append(
                    f"[Review iter {rec.iteration}]: {rec.feedback_summary}"
                )

        context_block = "\n".join(context_parts) if context_parts else "No context available."

        try:
            llm_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
            response = await llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an academic paper reviewer providing brief, "
                            "focused feedback. Answer the question concisely based "
                            "on the provided context. Keep your response under 200 words."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Paper context:\n{context_block}\n\n"
                            f"Question: {question}"
                        ),
                    },
                ],
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content or "No answer generated."
        except Exception as e:
            logger.error("reviewer.answer error: %s", e)
            return f"Could not answer: {e}"

    async def review(
        self,
        context: ReviewContext,
        iteration: int = 0,
        memory=None,
    ) -> ReviewResult:
        """
        Run all enabled checkers on the context.

        - **Args**:
            - `context` (ReviewContext): Review context with paper data
            - `iteration` (int): Current iteration number
            - `memory` (SessionMemory, optional): Shared session memory.
              When provided, checkers can read prior issues and plan
              details directly instead of relying on serialized snapshots.

        - **Returns**:
            - `ReviewResult`: Aggregated review result
        """
        # Inject memory into context for checkers that support it
        if memory is not None and hasattr(context, "memory_context"):
            if context.memory_context is None:
                from ..shared.session_memory import SessionMemory
                if isinstance(memory, SessionMemory):
                    context.memory_context = memory.to_review_context_dict()

        result = ReviewResult(iteration=iteration)

        logger.info(
            "reviewer.review iteration=%d sections=%s total_words=%d",
            iteration,
            list(context.sections.keys()),
            context.total_word_count(),
        )
        
        # Run each enabled checker
        for checker in self._checkers:
            if not checker.enabled:
                continue
            
            try:
                feedback = await checker.check(context)
                result.add_feedback(feedback)
                
                logger.info(
                    "reviewer.checker name=%s passed=%s severity=%s",
                    checker.name,
                    feedback.passed,
                    feedback.severity,
                )
                
                # Extract sections needing revision
                if not feedback.passed:
                    for df in feedback.details.get("document_feedbacks", []):
                        try:
                            result.add_hierarchical_feedback(HierarchicalFeedbackItem(
                                level=FeedbackLevel.DOCUMENT,
                                agent=str(df.get("agent", "reviewer")),
                                checker=str(df.get("checker", checker.name)),
                                target_id=str(df.get("target_id", "document")),
                                severity=Severity(str(df.get("severity", feedback.severity))),
                                issue_type=str(df.get("issue_type", "document_issue")),
                                message=str(df.get("message", feedback.message)),
                                suggested_action=df.get("suggested_action"),
                                evidence=df if isinstance(df, dict) else {},
                            ))
                        except Exception:
                            # Keep review loop resilient even with malformed checker payloads.
                            pass

                    sections_to_revise = feedback.details.get("sections_to_revise", {})
                    section_paragraph_feedbacks = feedback.details.get("paragraph_feedbacks", {})
                    raw_section_feedbacks = feedback.details.get("section_feedbacks", []) or []
                    for section_type, reason in sections_to_revise.items():
                        result.add_section_revision(section_type, reason)
                        
                        # Generate and store revision prompt
                        section_content = context.sections.get(section_type, "")
                        revision_prompt = checker.generate_revision_prompt(
                            section_type,
                            section_content,
                            feedback,
                        )
                        
                        # Find and update section feedback
                        matched_section_feedback = False
                        for sf in raw_section_feedbacks:
                            if sf.get("section_type") == section_type:
                                matched_section_feedback = True
                                raw_para = section_paragraph_feedbacks.get(section_type, []) or []
                                para_feedbacks: List[ParagraphFeedback] = []
                                for p in raw_para:
                                    raw_sev = str(p.get("severity", "warning")).lower()
                                    sev = Severity.WARNING
                                    if raw_sev in ("error", "high", "critical"):
                                        sev = Severity.ERROR
                                    elif raw_sev in ("info", "low"):
                                        sev = Severity.INFO
                                    para_feedbacks.append(ParagraphFeedback(
                                        paragraph_index=int(p.get("paragraph_index", 0)),
                                        paragraph_preview=str(p.get("paragraph_preview", "")),
                                        issues=[str(x) for x in p.get("issues", [])],
                                        severity=sev,
                                        suggestion=str(p.get("suggestion", "")),
                                    ))

                                section_fb = SectionFeedback(
                                    section_type=section_type,
                                    current_word_count=sf.get("current_word_count", 0),
                                    target_word_count=sf.get("target_word_count", 0),
                                    action=sf.get("action", "ok"),
                                    delta_words=sf.get("delta_words", 0),
                                    revision_prompt=revision_prompt,
                                    paragraph_feedbacks=para_feedbacks,
                                    target_paragraphs=sf.get(
                                        "target_paragraphs",
                                        [pf.paragraph_index for pf in para_feedbacks],
                                    ),
                                    paragraph_instructions=sf.get("paragraph_instructions", {}),
                                    feedback_level=FeedbackLevel.SECTION,
                                    target_id=sf.get("target_id", section_type),
                                )
                                result.section_feedbacks.append(section_fb)
                                result.add_hierarchical_feedback(HierarchicalFeedbackItem(
                                    level=FeedbackLevel.SECTION,
                                    agent="reviewer",
                                    checker=checker.name,
                                    target_id=section_fb.target_id or section_type,
                                    section_type=section_type,
                                    severity=feedback.severity,
                                    issue_type=checker.name,
                                    message=reason,
                                    suggested_action=sf.get("action", "revise"),
                                    revision_instruction=revision_prompt,
                                    evidence={
                                        "target_paragraphs": section_fb.target_paragraphs,
                                        "paragraph_feedbacks": [
                                            pf.model_dump() for pf in section_fb.paragraph_feedbacks
                                        ],
                                    },
                                ))
                                for pf in section_fb.paragraph_feedbacks:
                                    result.add_hierarchical_feedback(HierarchicalFeedbackItem(
                                        level=FeedbackLevel.PARAGRAPH,
                                        agent="reviewer",
                                        checker=checker.name,
                                        target_id=f"{section_type}.p{pf.paragraph_index}",
                                        section_type=section_type,
                                        paragraph_index=pf.paragraph_index,
                                        severity=pf.severity,
                                        issue_type=checker.name,
                                        message="; ".join(pf.issues)[:500],
                                        suggested_action="refine_paragraph",
                                        revision_instruction=pf.suggestion,
                                        evidence=pf.model_dump(),
                                    ))
                        if not matched_section_feedback:
                            para_feedbacks: List[ParagraphFeedback] = []
                            raw_para = section_paragraph_feedbacks.get(section_type, []) or []
                            para_indices: List[int] = []
                            para_instructions: Dict[int, str] = {}
                            for p in raw_para:
                                pidx = int(p.get("paragraph_index", 0))
                                para_indices.append(pidx)
                                raw_sev = str(p.get("severity", "warning")).lower()
                                sev = Severity.WARNING
                                if raw_sev in ("error", "high", "critical"):
                                    sev = Severity.ERROR
                                elif raw_sev in ("info", "low"):
                                    sev = Severity.INFO
                                para_feedbacks.append(ParagraphFeedback(
                                    paragraph_index=pidx,
                                    paragraph_preview=str(p.get("paragraph_preview", "")),
                                    issues=[str(x) for x in p.get("issues", [])],
                                    severity=sev,
                                    suggestion=str(p.get("suggestion", "")),
                                ))
                                para_instructions[pidx] = str(p.get("suggestion", "")) or reason
                            section_fb = SectionFeedback(
                                section_type=section_type,
                                current_word_count=context.word_counts.get(section_type, 0),
                                target_word_count=context.get_section_target(section_type) or context.word_counts.get(section_type, 0),
                                action="refine_paragraphs" if para_indices else "revise",
                                delta_words=0,
                                revision_prompt=revision_prompt,
                                paragraph_feedbacks=para_feedbacks,
                                target_paragraphs=sorted(list(set(para_indices))),
                                paragraph_instructions=para_instructions,
                                feedback_level=FeedbackLevel.SECTION,
                                target_id=section_type,
                            )
                            result.section_feedbacks.append(section_fb)
                        
            except Exception as e:
                logger.error("reviewer.checker_error name=%s error=%s", checker.name, str(e))
                result.add_feedback(FeedbackResult(
                    checker_name=checker.name,
                    passed=False,
                    severity=Severity.ERROR,
                    message=f"Checker error: {str(e)}",
                ))
        
        logger.info(
            "reviewer.review.complete passed=%s feedbacks=%d revisions=%d",
            result.passed,
            len(result.feedbacks),
            len(result.requires_revision),
        )
        
        return result
    
    def get_revision_prompt(
        self,
        section_type: str,
        current_content: str,
        review_result: ReviewResult,
    ) -> Optional[str]:
        """
        Get revision prompt for a specific section
        
        - **Args**:
            - `section_type`: Type of section to revise
            - `current_content`: Current section content
            - `review_result`: Review result with feedbacks
            
        - **Returns**:
            - `str`: Revision prompt, or None if no revision needed
        """
        # Find section feedback with revision prompt
        for sf in review_result.section_feedbacks:
            if sf.section_type == section_type and sf.revision_prompt:
                return sf.revision_prompt
        
        return None
