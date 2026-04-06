"""
Review Orchestrator — extracted from MetaDataAgent for architecture decoupling.
- **Description**:
    - Manages the multi-iteration review-revise-compile loop
    - Coordinates between reviewer, VLM reviewer, and typesetter feedback
    - Handles pre-search, planning, and reference coverage enforcement
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)

from ..reviewer_agent.models import (
    ConflictResolutionRecord,
    FeedbackResult,
    ReviewResult,
    SectionFeedback,
    SemanticCheckRecord,
    Severity,
)
from ..shared.session_memory import SessionMemory, ReviewRecord
from ...models.evidence_graph import EvidenceDAG
from ..shared.reference_pool import ReferencePool
from .models import StructuralAction

if TYPE_CHECKING:
    from .models import PaperMetaData, SectionResult, PaperPlan
    from ..planner_agent.models import PaperPlan as PlannerPaperPlan

logger = logging.getLogger("uvicorn.error")

SEARCH_JUDGMENT_PROMPT = """\
You are an academic research assistant. Your task is to analyze whether the existing \
references are sufficient for writing a specific section of a research paper, or whether \
additional references should be searched.

Given the following section context, analyze whether additional references are needed.

SECTION: {section_type} ({section_title})
PAPER TITLE: {paper_title}
KEY POINTS TO COVER:
{key_points}

EXISTING REFERENCES ({n_refs} papers):
{ref_summaries}

RULES:
- If existing references adequately cover the section's claims, set need_search to false.
- If there are gaps (e.g., missing baselines, no empirical evidence for a claim, \
a sub-topic with zero refs), set need_search to true.
- Provide 1-3 focused search queries using specific keywords, not broad topics.
  Good: "R&D tax credit firm productivity panel data"
  Bad: "innovation economics"
- Each query should target a specific gap you identified.

Respond with ONLY a JSON object, no other text:
{{"need_search": true/false, "reason": "...", "queries": ["...", "..."]}}\
"""


class ReviewOrchestrator:
    """Manages the multi-iteration review-revise-compile loop."""

    def __init__(self, host: Any) -> None:
        """
        Initialize ReviewOrchestrator.
        - **Description**:
            - Stores a reference to the host MetaDataAgent.
            - client and model_name are resolved lazily from the host.

        - **Args**:
            - `host`: The MetaDataAgent instance that owns this orchestrator.
        """
        self.host = host
        self._client: Any = None
        self._model_name: Optional[str] = None

    # ------------------------------------------------------------------
    # Lazy accessors
    # ------------------------------------------------------------------

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self.host.client
        return self._client

    @property
    def model_name(self) -> str:
        if self._model_name is None:
            self._model_name = self.host.model_name
        return self._model_name

    # ------------------------------------------------------------------
    # Main review orchestration loop
    # ------------------------------------------------------------------

    async def _run_review_orchestration(
        self,
        generated_sections: Dict[str, str],
        sections_results: List[SectionResult],
        metadata: PaperMetaData,
        parsed_refs: List[Dict[str, Any]],
        paper_plan: Optional[PaperPlan],
        template_path: Optional[str],
        figures_source_dir: Optional[str],
        converted_tables: Dict[str, str],
        max_review_iterations: int,
        enable_review: bool,
        compile_pdf: bool,
        enable_vlm_review: bool,
        target_pages: Optional[int],
        paper_dir: Optional[Path],
        memory: Optional[SessionMemory] = None,
        evidence_dag: Optional[EvidenceDAG] = None,
    ) -> Tuple[Dict[str, str], List[SectionResult], int, Optional[int], Optional[str], List[str]]:
        """
        Run unified review orchestration across reviewer and VLM.
        - **Description**:
            - Executes reviewer checks and VLM review in a single loop
            - Applies revisions, recompiles, and rechecks until pass or limit

        - **Args**:
            - `generated_sections` (Dict[str, str]): Section contents
            - `sections_results` (List[SectionResult]): Section result list
            - `metadata` (PaperMetaData): Paper metadata
            - `parsed_refs` (List[Dict[str, Any]]): Parsed references
            - `paper_plan` (Optional[PaperPlan]): Paper plan with targets
            - `template_path` (Optional[str]): Template path for PDF
            - `figures_source_dir` (Optional[str]): Figure source directory
            - `converted_tables` (Dict[str, str]): Converted table LaTeX
            - `max_review_iterations` (int): Maximum review iterations
            - `enable_review` (bool): Enable ReviewerAgent checks
            - `compile_pdf` (bool): Compile PDF if template_path is provided
            - `enable_vlm_review` (bool): Enable VLM-based PDF review
            - `target_pages` (Optional[int]): Target page count
            - `paper_dir` (Optional[Path]): Output directory

        - **Returns**:
            - `generated_sections` (Dict[str, str]): Updated sections
            - `sections_results` (List[SectionResult]): Updated results
            - `review_iterations` (int): Iteration count used
            - `target_word_count` (Optional[int]): Target word count from reviewer
            - `pdf_path` (Optional[str]): Latest compiled PDF path
            - `errors` (List[str]): Orchestration errors
        """
        errors: List[str] = []
        review_iterations = 0
        target_word_count = None
        pdf_path = None
        last_fingerprint = self.host._executor._get_sections_fingerprint(generated_sections)
        last_compiled_fingerprint = last_fingerprint
        last_vlm_result = None

        if enable_review:
            print(f"[MetaDataAgent] Unified Review Loop (max {max_review_iterations} iterations)...")
        baseline_gap_audit = self._perform_baseline_gap_audit(
            generated_sections=generated_sections,
            enable_review=enable_review,
            enable_vlm_review=enable_vlm_review,
        )
        if memory is not None:
            memory.log(
                "metadata",
                "review_init",
                "baseline_gap_audit",
                narrative="Captured baseline capability-gap audit for LLM enhancement rollout.",
                audit=baseline_gap_audit,
            )

        for iteration in range(max_review_iterations):
            review_iterations = iteration + 1
            iteration_decision_trace: List[Dict[str, Any]] = []
            iteration_conflicts: List[ConflictResolutionRecord] = []
            iteration_semantic_checks: List[SemanticCheckRecord] = []
            iteration_revision_plan: List[Dict[str, Any]] = []
            iteration_writer_response_section: List[Dict[str, Any]] = []
            iteration_writer_response_paragraph: List[Dict[str, Any]] = []
            iteration_reviewer_verification: List[Dict[str, Any]] = []
            print(f"[MetaDataAgent] Review iteration {review_iterations}/{max_review_iterations}")
            print(
                "[MetaDataAgent] Review context: "
                f"target_pages={target_pages or 8} enable_review={enable_review} enable_vlm_review={enable_vlm_review}"
            )

            word_counts = {
                sr.section_type: sr.word_count
                for sr in sections_results
                if sr.status == "ok"
            }

            review_result = ReviewResult(iteration=iteration)

            # Reference coverage pass: proactively integrate uncited assigned refs.
            ref_revised_sections: Set[str] = set()
            if paper_plan:
                ref_revised_sections = await self._enforce_reference_coverage(
                    generated_sections=generated_sections,
                    sections_results=sections_results,
                    paper_plan=paper_plan,
                    metadata=metadata,
                    valid_ref_keys=self.host._extract_valid_citation_keys(parsed_refs),
                    memory=memory,
                    max_sections_to_revise=2,
                )
                if ref_revised_sections:
                    if memory:
                        memory.log(
                            "metadata",
                            f"review_iter_{review_iterations}",
                            "reference_coverage_revised",
                            narrative=(
                                "Applied targeted revisions to improve citation coverage "
                                f"in sections: {', '.join(sorted(ref_revised_sections))}."
                            ),
                            revised_sections=sorted(ref_revised_sections),
                        )

            if enable_review:
                section_targets = None
                if paper_plan and paper_plan.sections:
                    section_targets = {
                        s.section_type: s.get_estimated_words()
                        for s in paper_plan.sections
                    }
                section_structure_signals = None
                if paper_plan and paper_plan.sections:
                    section_structure_signals = {
                        s.section_type: {
                            "topic_clusters": list(s.topic_clusters or []),
                            "transition_intents": list(s.transition_intents or []),
                            "sectioning_recommended": bool(s.sectioning_recommended),
                            "paragraph_count": len(s.paragraphs or []),
                        }
                        for s in paper_plan.sections
                    }
                reviewer_result, target_word_count = await self.host._executor._call_reviewer(
                    sections=generated_sections,
                    word_counts=word_counts,
                    target_pages=target_pages,
                    style_guide=metadata.style_guide,
                    template_path=template_path,
                    iteration=iteration,
                    section_targets=section_targets,
                    section_structure_signals=section_structure_signals,
                    memory=memory,
                    evidence_dag=evidence_dag,
                )
                if reviewer_result is None:
                    print("[MetaDataAgent] Reviewer not available, skipping content review")
                else:
                    review_result = ReviewResult(**reviewer_result)
                    iteration_decision_trace.append({
                        "source": "reviewer_orchestrator",
                        "summary": review_result.orchestrator_summary or "Reviewer orchestration completed.",
                        "num_tasks": len(review_result.revision_tasks),
                    })
                    iteration_revision_plan = await self._llm_plan_revision_tasks(review_result)
                    self._apply_revision_plan_to_feedbacks(review_result, iteration_revision_plan)
                    print(
                        "[Reviewer] Result: "
                        f"passed={review_result.passed} "
                        f"sections_to_revise={list(review_result.requires_revision.keys())}"
                    )

                    if last_vlm_result and last_vlm_result.get("needs_trim"):
                        suppressed = [
                            sf.section_type
                            for sf in review_result.section_feedbacks
                            if sf.action == "expand"
                        ]
                        if suppressed:
                            review_result.section_feedbacks = [
                                sf for sf in review_result.section_feedbacks
                                if sf.action != "expand"
                            ]
                            for sec in suppressed:
                                review_result.requires_revision.pop(sec, None)
                            print(
                                f"[ReviewLoop] Suppressed Reviewer 'expand' for "
                                f"{suppressed} (VLM overflow active)"
                            )

            reviewer_revised_sections = await self.host._executor._apply_revisions(
                review_result=review_result,
                generated_sections=generated_sections,
                sections_results=sections_results,
                valid_citation_keys=self.host._extract_valid_citation_keys(parsed_refs),
                metadata=metadata,
                memory=memory,
                semantic_checks=iteration_semantic_checks,
                decision_trace=iteration_decision_trace,
                writer_response_section=iteration_writer_response_section,
                writer_response_paragraph=iteration_writer_response_paragraph,
                reviewer_verification=iteration_reviewer_verification,
            )
            if reviewer_revised_sections:
                print(f"[ReviewLoop] Reviewer revised: {sorted(reviewer_revised_sections)}")
            self.host._resolver._resolve_section_feedbacks(
                section_feedbacks=review_result.section_feedbacks,
                revised_sections=reviewer_revised_sections,
                review_result=review_result,
            )
            word_counts = {
                sr.section_type: sr.word_count
                for sr in sections_results
                if sr.status == "ok"
            }
            print(f"[ReviewLoop] Word counts: {word_counts}")

            # Compile PDF and run VLM review if enabled
            compile_succeeded = False
            last_compiled_fingerprint = self.host._executor._get_sections_fingerprint(generated_sections)
            if compile_pdf and template_path and paper_dir:
                iteration_dir = paper_dir / f"iteration_{review_iterations:02d}"
                iteration_dir.mkdir(parents=True, exist_ok=True)
                print(f"[ReviewLoop] PDF output dir: {iteration_dir}")
                figure_base_path = os.getcwd()
                figure_paths = self.host._collect_figure_paths(metadata.figures, base_path=figure_base_path)
                pdf_result_path, _, compile_errors, section_errors = await self.host._compile_pdf(
                    generated_sections=generated_sections,
                    template_path=template_path,
                    references=parsed_refs,
                    output_dir=iteration_dir,
                    paper_title=metadata.title,
                    figures_source_dir=figures_source_dir,
                    figure_paths=figure_paths,
                    converted_tables=converted_tables,
                    paper_plan=paper_plan,
                    figures=metadata.figures,
                    metadata_tables=metadata.tables,
                )
                if pdf_result_path:
                    pdf_path = pdf_result_path
                    compile_succeeded = True
                else:
                    print(f"[ReviewLoop] PDF compilation failed, treating as Typesetter review feedback")
                    ts_feedbacks, ts_section_feedbacks = self.host._resolver._build_typesetter_feedback(
                        compile_errors=compile_errors,
                        section_errors=section_errors,
                        generated_sections=generated_sections,
                    )
                    for fb in ts_feedbacks:
                        review_result.add_feedback(fb)

                    merged_seed = self.host._resolver._merge_section_feedbacks(
                        review_result.section_feedbacks,
                        ts_section_feedbacks,
                        prefer_vlm=False,
                    )
                    merged_section_feedbacks, ts_conflicts = await self.host._resolver._resolve_conflicts_with_llm(
                        reviewer_feedbacks=review_result.section_feedbacks,
                        external_feedbacks=merged_seed,
                    )
                    iteration_conflicts.extend(ts_conflicts)
                    review_result.section_feedbacks = merged_section_feedbacks
                    for sf in ts_section_feedbacks:
                        review_result.add_section_revision(sf.section_type, "Typesetter LaTeX fix")

                    review_result.passed = False

                if compile_succeeded and enable_vlm_review and pdf_path:
                    print(f"[MetaDataAgent] VLM Review: pdf_path={pdf_path}")
                    last_vlm_result = await self.host._call_vlm_review(
                        pdf_path=pdf_path,
                        page_limit=target_pages or 8,
                        template_type=metadata.style_guide or "ICML",
                        sections_info={
                            sr.section_type: {"word_count": sr.word_count}
                            for sr in sections_results if sr.word_count
                        },
                        memory=memory,
                    )
                    if last_vlm_result:
                        print(
                            "[VLMReview] Result: "
                            f"passed={last_vlm_result.get('passed', False)} "
                            f"overflow_pages={last_vlm_result.get('overflow_pages', 0)} "
                            f"needs_trim={last_vlm_result.get('needs_trim', False)} "
                            f"needs_expand={last_vlm_result.get('needs_expand', False)} "
                            f"sections={list((last_vlm_result.get('section_recommendations') or {}).keys())}"
                        )

                        planned_structural_actions: List[StructuralAction] = []
                        if last_vlm_result.get("needs_trim"):
                            overflow = last_vlm_result.get("overflow_pages", 0)
                            if overflow > 0:
                                section_order = list(generated_sections.keys())
                                if paper_plan and paper_plan.sections:
                                    section_order = [s.section_type for s in paper_plan.sections]

                                planned_structural_actions = self.host._plan_overflow_strategy(
                                    overflow_pages=overflow,
                                    generated_sections=generated_sections,
                                    paper_plan=paper_plan,
                                    figures=metadata.figures,
                                )
                                if planned_structural_actions:
                                    self.host._execute_structural_actions(
                                        planned_structural_actions,
                                        generated_sections,
                                        section_order,
                                    )
                                    for sr in sections_results:
                                        if sr.section_type in generated_sections:
                                            sr.latex_content = generated_sections[sr.section_type]
                                            sr.word_count = len(generated_sections[sr.section_type].split())
                                    print(
                                        f"[ReviewLoop] Structural actions applied: "
                                        f"{len(planned_structural_actions)} actions, "
                                        f"total estimated savings ~"
                                        f"{sum(a.estimated_savings for a in planned_structural_actions):.1f} pages"
                                    )

                        vlm_feedbacks, vlm_section_feedbacks = self.host._resolver._build_vlm_feedback(
                            last_vlm_result,
                            structural_actions=planned_structural_actions or None,
                        )
                        translated_vlm_plan = await self._translate_vlm_to_revision_plan(
                            vlm_result=last_vlm_result,
                            generated_sections=generated_sections,
                        )
                        for p in translated_vlm_plan:
                            st = str(p.get("section_type", ""))
                            if not st:
                                continue
                            action = str(p.get("action", "ok"))
                            delta = int(p.get("delta_words", 0) or 0)
                            prompt = str(p.get("rationale", "")) or "Apply VLM-guided layout refinements."
                            section_struct_actions = []
                            if planned_structural_actions:
                                section_struct_actions = [
                                    f"{a.action_type}:{a.target_id}"
                                    for a in planned_structural_actions
                                    if a.section == st
                                ]
                            vlm_section_feedbacks.append(SectionFeedback(
                                section_type=st,
                                current_word_count=0,
                                target_word_count=0,
                                action=action,
                                delta_words=delta,
                                revision_prompt=prompt,
                                structural_actions=section_struct_actions,
                                target_paragraphs=self.host._resolver._normalize_target_paragraphs(
                                    p.get("target_paragraphs", [])
                                ),
                                paragraph_instructions=self.host._resolver._normalize_paragraph_instructions(
                                    p.get("paragraph_instructions", {}),
                                    target_paragraphs=self.host._resolver._normalize_target_paragraphs(
                                        p.get("target_paragraphs", [])
                                    ),
                                    fallback_instruction=prompt,
                                ),
                            ))
                        if translated_vlm_plan:
                            iteration_decision_trace.append({
                                "source": "vlm_translator_llm",
                                "summary": f"Translated {len(translated_vlm_plan)} VLM recommendation(s) into revision plan.",
                            })
                        for fb in vlm_feedbacks:
                            review_result.add_feedback(fb)

                        prefer_vlm = bool(last_vlm_result.get("needs_trim") or last_vlm_result.get("needs_expand"))
                        merged_seed = self.host._resolver._merge_section_feedbacks(
                            review_result.section_feedbacks,
                            vlm_section_feedbacks,
                            prefer_vlm=prefer_vlm,
                        )
                        merged_section_feedbacks, conflicts = await self.host._resolver._resolve_conflicts_with_llm(
                            reviewer_feedbacks=review_result.section_feedbacks,
                            external_feedbacks=merged_seed,
                        )
                        iteration_conflicts.extend(conflicts)
                        review_result.section_feedbacks = merged_section_feedbacks
                        for sf in review_result.section_feedbacks:
                            if sf.section_type in word_counts:
                                sf.current_word_count = word_counts.get(sf.section_type, 0)
                            if paper_plan:
                                section_plan = paper_plan.get_section(sf.section_type)
                                if section_plan:
                                    est = section_plan.get_estimated_words()
                                    if est > 0:
                                        sf.target_word_count = est

                        for sf in merged_section_feedbacks:
                            if sf.action != "ok":
                                review_result.add_section_revision(sf.section_type, "VLM adjustment")
                    else:
                        print("[MetaDataAgent] VLM review unavailable, skipping")
            elif enable_vlm_review:
                errors.append("VLM review skipped: PDF not compiled (missing template or output path)")

            # Apply revisions from VLM feedback and/or Typesetter LaTeX-fix feedback
            post_compile_revised = await self.host._executor._apply_revisions(
                review_result=review_result,
                generated_sections=generated_sections,
                sections_results=sections_results,
                valid_citation_keys=self.host._extract_valid_citation_keys(parsed_refs),
                metadata=metadata,
                memory=memory,
                semantic_checks=iteration_semantic_checks,
                decision_trace=iteration_decision_trace,
                writer_response_section=iteration_writer_response_section,
                writer_response_paragraph=iteration_writer_response_paragraph,
                reviewer_verification=iteration_reviewer_verification,
            )
            if post_compile_revised:
                sources = "VLM/Typesetter" if not compile_succeeded else "VLM"
                print(f"[ReviewLoop] {sources} revised: {sorted(post_compile_revised)}")
            self.host._resolver._resolve_section_feedbacks(
                section_feedbacks=review_result.section_feedbacks,
                revised_sections=post_compile_revised,
                review_result=review_result,
            )
            review_result.conflict_resolution = iteration_conflicts
            review_result.semantic_checks = iteration_semantic_checks
            review_result.decision_trace = iteration_decision_trace

            # Record review iteration in session memory
            if memory is not None:
                section_fb_dict = {}
                for sf in review_result.section_feedbacks:
                    section_fb_dict[sf.section_type] = {
                        "action": sf.action,
                        "delta_words": sf.delta_words,
                        "target_paragraphs": sf.target_paragraphs,
                        "paragraph_instructions": sf.paragraph_instructions,
                        "paragraph_feedbacks": [
                            pf.model_dump() if hasattr(pf, "model_dump") else pf
                            for pf in (sf.paragraph_feedbacks if hasattr(sf, "paragraph_feedbacks") else [])
                        ],
                    }
                agent_feedbacks: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
                for fb in review_result.feedbacks:
                    agent_name = str(fb.checker_name or "reviewer")
                    if agent_name not in agent_feedbacks:
                        agent_feedbacks[agent_name] = {
                            "document_feedbacks": [],
                            "section_feedbacks": [],
                            "paragraph_feedbacks": [],
                            "sentence_feedbacks": [],
                        }
                    agent_feedbacks[agent_name]["document_feedbacks"].append({
                        "level": "document",
                        "agent": agent_name,
                        "checker": fb.checker_name,
                        "target_id": "document",
                        "severity": fb.severity.value if hasattr(fb.severity, "value") else str(fb.severity),
                        "issue_type": fb.suggested_action or fb.checker_name,
                        "message": fb.message,
                        "details": fb.details,
                    })
                for hf in (review_result.hierarchical_feedbacks or []):
                    item = hf.model_dump() if hasattr(hf, "model_dump") else hf
                    agent_name = str(item.get("agent", "reviewer"))
                    item["source_agent"] = str(item.get("source_agent") or agent_name)
                    item["source_stage"] = str(item.get("source_stage") or item["source_agent"])
                    if agent_name not in agent_feedbacks:
                        agent_feedbacks[agent_name] = {
                            "document_feedbacks": [],
                            "section_feedbacks": [],
                            "paragraph_feedbacks": [],
                            "sentence_feedbacks": [],
                        }
                    level = str(item.get("level", "section"))
                    bucket = f"{level}_feedbacks"
                    if bucket not in agent_feedbacks[agent_name]:
                        bucket = "section_feedbacks"
                    agent_feedbacks[agent_name][bucket].append(item)
                for sf in review_result.section_feedbacks:
                    source_agent = "reviewer"
                    if sf.action == "fix_latex":
                        source_agent = "typesetter"
                    elif sf.structural_actions:
                        source_agent = "vlm_review"
                    elif "VLM" in (sf.revision_prompt or "") or "page limit" in (sf.revision_prompt or "").lower():
                        source_agent = "vlm_review"
                    if source_agent not in agent_feedbacks:
                        agent_feedbacks[source_agent] = {
                            "document_feedbacks": [],
                            "section_feedbacks": [],
                            "paragraph_feedbacks": [],
                            "sentence_feedbacks": [],
                        }
                    sec_item = {
                        "level": "section",
                        "agent": source_agent,
                        "source_agent": source_agent,
                        "source_stage": source_agent,
                        "checker": source_agent,
                        "target_id": sf.target_id or sf.section_type,
                        "section_type": sf.section_type,
                        "severity": "warning",
                        "issue_type": sf.action,
                        "message": sf.revision_prompt[:400],
                        "target_paragraphs": sf.target_paragraphs,
                    }
                    agent_feedbacks[source_agent]["section_feedbacks"].append(sec_item)
                    for pf in sf.paragraph_feedbacks or []:
                        pf_item = pf.model_dump() if hasattr(pf, "model_dump") else pf
                        agent_feedbacks[source_agent]["paragraph_feedbacks"].append({
                            "level": "paragraph",
                            "agent": source_agent,
                            "source_agent": source_agent,
                            "source_stage": source_agent,
                            "checker": source_agent,
                            "target_id": f"{sf.section_type}.p{pf_item.get('paragraph_index', 0)}",
                            "section_type": sf.section_type,
                            "paragraph_index": pf_item.get("paragraph_index", 0),
                            "severity": str(pf_item.get("severity", "warning")),
                            "issue_type": sf.action,
                            "message": "; ".join(pf_item.get("issues", []))[:400],
                            "suggestion": pf_item.get("suggestion", ""),
                        })
                actions_taken = sorted(
                    reviewer_revised_sections | post_compile_revised
                )
                lifecycle_result = memory.update_issue_lifecycle(
                    iteration=review_iterations,
                    hierarchical_feedbacks=[
                        hf.model_dump() if hasattr(hf, "model_dump") else hf
                        for hf in (review_result.hierarchical_feedbacks or [])
                    ],
                    writer_response_section=iteration_writer_response_section,
                    writer_response_paragraph=iteration_writer_response_paragraph,
                )
                word_snapshot = {
                    sr.section_type: sr.word_count
                    for sr in sections_results if sr.status == "ok"
                }
                iter_hallucination_stats: Dict[str, Any] = {
                    "review_is_final_assert": True,
                }
                for fb in review_result.feedbacks:
                    if fb.checker_name == "evidence_check":
                        ev_stats = fb.details.get("hallucination_stats", {})
                        if ev_stats:
                            iter_hallucination_stats["review_catch_rate"] = (
                                ev_stats.get("drifted_claims", 0) / max(ev_stats.get("total_claims", 1), 1)
                            )
                            iter_hallucination_stats["overall_grounding_rate"] = ev_stats.get(
                                "grounding_rate", 1.0,
                            )
                            iter_hallucination_stats.update(ev_stats)

                record = ReviewRecord(
                    iteration=review_iterations,
                    reviewer="unified",
                    passed=review_result.passed,
                    feedback_summary="; ".join(
                        f.message for f in review_result.feedbacks if not f.passed
                    )[:500],
                    section_feedbacks=section_fb_dict,
                    hierarchical_feedbacks=[
                        hf.model_dump() if hasattr(hf, "model_dump") else hf
                        for hf in (review_result.hierarchical_feedbacks or [])
                    ],
                    agent_feedbacks=agent_feedbacks,
                    decision_trace=iteration_decision_trace + [
                        dt if isinstance(dt, dict) else {"event": str(dt)}
                        for dt in (review_result.decision_trace or [])
                    ],
                    revision_plan=iteration_revision_plan + [
                        t.model_dump() if hasattr(t, "model_dump") else t
                        for t in (review_result.revision_tasks or [])
                    ],
                    before_after_semantic_check=[
                        s.model_dump() if hasattr(s, "model_dump") else s
                        for s in iteration_semantic_checks
                    ],
                    conflict_resolution=[
                        c.model_dump() if hasattr(c, "model_dump") else c
                        for c in iteration_conflicts
                    ],
                    baseline_gap_audit=baseline_gap_audit,
                    issue_lifecycle=lifecycle_result.get("issue_lifecycle", []),
                    hallucination_stats=iter_hallucination_stats,
                    writer_response_section=iteration_writer_response_section,
                    writer_response_paragraph=iteration_writer_response_paragraph,
                    reviewer_verification=iteration_reviewer_verification,
                    regression_report=lifecycle_result.get("regression_report", {}),
                    actions_taken=[f"revised:{s}" for s in actions_taken],
                    result_snapshot=word_snapshot,
                )
                memory.add_review(record)
                review_narr = f"Review iteration {review_iterations}: "
                if review_result.passed:
                    review_narr += "All checks passed."
                else:
                    failed_msgs = [f.message for f in review_result.feedbacks if not f.passed]
                    review_narr += f"Found {len(failed_msgs)} issue(s). "
                    if failed_msgs:
                        review_narr += failed_msgs[0][:150]
                if actions_taken:
                    review_narr += f" Revised sections: {', '.join(actions_taken)}."
                memory.log("metadata", f"review_iter_{review_iterations}",
                           "review_completed",
                           narrative=review_narr,
                           passed=review_result.passed,
                           revised=actions_taken)
                for stype, content in generated_sections.items():
                    memory.update_section(stype, content)

            current_fingerprint = self.host._executor._get_sections_fingerprint(generated_sections)
            if current_fingerprint == last_fingerprint:
                if review_result.passed and (not last_vlm_result or last_vlm_result.get("passed", True)):
                    print("[MetaDataAgent] Review passed with no further changes (fingerprint stable)")
                elif not compile_succeeded:
                    errors.append("LaTeX compilation failed and revisions did not change content")
                    print("[MetaDataAgent] Exiting: compilation failed with no effective revisions (fingerprint stable)")
                else:
                    if last_vlm_result and not last_vlm_result.get("passed", True):
                        errors.append(last_vlm_result.get("summary", "VLM review failed"))
                        print("[MetaDataAgent] Exiting due to VLM failure with no changes")
                break

            last_fingerprint = current_fingerprint
            if not reviewer_revised_sections and not post_compile_revised and review_result.passed and (not last_vlm_result or last_vlm_result.get("passed", True)):
                print("[MetaDataAgent] Exiting: no revisions needed and review passed")
                break

        # =====================================================================
        # Final compilation pass
        # =====================================================================
        if compile_pdf and template_path and paper_dir:
            final_fp = self.host._executor._get_sections_fingerprint(generated_sections)
            if final_fp != last_compiled_fingerprint:
                print("[MetaDataAgent] Final pass: content changed since last compile — recompiling")
                final_dir = paper_dir / f"iteration_{review_iterations:02d}_final"
                final_dir.mkdir(parents=True, exist_ok=True)
                figure_base_path = os.getcwd()
                figure_paths = self.host._collect_figure_paths(metadata.figures, base_path=figure_base_path)
                final_pdf, _, final_errors, _ = await self.host._compile_pdf(
                    generated_sections=generated_sections,
                    template_path=template_path,
                    references=parsed_refs,
                    output_dir=final_dir,
                    paper_title=metadata.title,
                    figures_source_dir=figures_source_dir,
                    figure_paths=figure_paths,
                    converted_tables=converted_tables,
                    paper_plan=paper_plan,
                    figures=metadata.figures,
                    metadata_tables=metadata.tables,
                )
                if final_pdf:
                    pdf_path = final_pdf
                    print(f"[MetaDataAgent] Final pass compiled: {final_pdf}")
                elif final_errors:
                    errors.extend(final_errors)
                    print(f"[MetaDataAgent] Final pass compile errors: {final_errors[:2]}")
            else:
                print("[MetaDataAgent] Final pass: no content changes since last compile — skipping")

        return generated_sections, sections_results, review_iterations, target_word_count, pdf_path, errors

    # ------------------------------------------------------------------
    # Reference coverage enforcement
    # ------------------------------------------------------------------

    async def _enforce_reference_coverage(
        self,
        generated_sections: Dict[str, str],
        sections_results: List[SectionResult],
        paper_plan: Optional[PaperPlan],
        metadata: PaperMetaData,
        valid_ref_keys: Set[str],
        memory: Optional[SessionMemory] = None,
        max_sections_to_revise: int = 2,
    ) -> Set[str]:
        """
        Reference coverage fix pass.
        - **Description**:
            - Finds uncited pooled references.
            - Routes each missing key to one section that has it in assigned_refs.
            - Applies targeted revision prompts to integrate those citations.
        """
        if not paper_plan:
            return set()

        all_content = "\n".join(generated_sections.values())
        cited_keys = ReferencePool.extract_cite_keys(all_content)
        uncited_keys = set(valid_ref_keys) - set(cited_keys)
        if not uncited_keys:
            return set()

        missing_by_section: Dict[str, List[str]] = {}
        for sp in paper_plan.sections:
            st = sp.section_type
            if st in ("abstract", "conclusion"):
                continue
            if st not in generated_sections:
                continue
            assigned = set(getattr(sp, "assigned_refs", []) or [])
            missing = sorted(list(assigned & uncited_keys))
            if missing:
                missing_by_section[st] = missing

        if not missing_by_section:
            return set()

        revised_sections: Set[str] = set()
        targets = sorted(
            missing_by_section.items(),
            key=lambda kv: len(kv[1]),
            reverse=True,
        )[:max_sections_to_revise]

        for section_type, missing_keys in targets:
            prompt = (
                f"Reference coverage fix for section '{section_type}'.\n"
                f"Integrate the following citation keys naturally into relevant claims: "
                f"{', '.join(missing_keys[:6])}.\n"
                "Rules:\n"
                "- Preserve technical meaning and paragraph structure.\n"
                "- Use ONLY these keys via \\cite{key}.\n"
                "- Do not add citations in abstract or conclusion.\n"
                "- Do not fabricate facts; attach citations to existing statements where appropriate."
            )
            revised = await self.host._executor._revise_section(
                section_type=section_type,
                current_content=generated_sections[section_type],
                revision_prompt=prompt,
                metadata=metadata,
                memory=memory,
            )
            if revised and revised.strip():
                generated_sections[section_type] = revised
                for sr in sections_results:
                    if sr.section_type == section_type and sr.status == "ok":
                        sr.latex_content = revised
                        sr.word_count = len(revised.split())
                        break
                revised_sections.add(section_type)

        if revised_sections:
            post_cited = ReferencePool.extract_cite_keys("\n".join(generated_sections.values()))
            post_coverage = (len(set(valid_ref_keys) & set(post_cited)) / len(valid_ref_keys)) if valid_ref_keys else 1.0
            print(
                "[MetaDataAgent] Ref coverage pass revised="
                f"{sorted(revised_sections)} coverage={post_coverage:.0%}"
            )
        return revised_sections

    # ------------------------------------------------------------------
    # Baseline gap audit
    # ------------------------------------------------------------------

    def _perform_baseline_gap_audit(
        self,
        generated_sections: Dict[str, str],
        enable_review: bool,
        enable_vlm_review: bool,
    ) -> Dict[str, Any]:
        """
        Build a module-level baseline gap audit snapshot.
        - **Description**:
            - Captures current capabilities against the LLM-upgrade goals
            - Stored into review export for explainability/audit trails
        """
        has_paragraph_content = any("\n\n" in (c or "") for c in generated_sections.values())
        return {
            "reviewer": {
                "has_hierarchical_feedback": True,
                "has_llm_orchestrator": True,
                "gap": "needs stronger cross-agent conflict arbitration",
            },
            "writer": {
                "writes_at_section_level": True,
                "has_paragraph_units": has_paragraph_content,
                "gap": "needs explicit revision plan execution constraints",
            },
            "metadata": {
                "has_unified_review_loop": bool(enable_review),
                "has_conflict_guardrails": True,
                "gap": "needs explainable LLM arbitration traces",
            },
            "vlm": {
                "enabled": bool(enable_vlm_review),
                "has_structural_strategy": True,
                "gap": "needs text-structure co-planning translation",
            },
            "export": {
                "iteration_centric": True,
                "has_agent_hierarchy": True,
                "gap": "decision/reason traces should be first-class fields",
            },
        }

    # ------------------------------------------------------------------
    # Pre-generation search judgment
    # ------------------------------------------------------------------

    async def _judge_search_need(
        self,
        section_type: str,
        section_title: str,
        paper_title: str,
        key_points: List[str],
        ref_pool: ReferencePool,
    ) -> Dict[str, Any]:
        """
        Ask the LLM whether additional references are needed for a section.

        - **Description**:
            - Phase A of the two-phase generation pattern.
            - Sends a lightweight prompt asking the LLM to analyse the gap
              between the section's requirements and the existing references.
            - The LLM returns a structured JSON response with need_search,
              reason, and queries fields.
            - Does NOT use Function Calling — the LLM directly outputs JSON.

        - **Args**:
            - `section_type` (str): E.g. "introduction", "method".
            - `section_title` (str): Human-readable section title.
            - `paper_title` (str): Title of the paper being generated.
            - `key_points` (List[str]): Key points to cover in this section.
            - `ref_pool` (ReferencePool): Current reference pool.

        - **Returns**:
            - `dict`: {"need_search": bool, "reason": str, "queries": [str]}
              On any error, returns {"need_search": False, ...}.
        """
        ref_summaries_parts = []
        for ref in ref_pool.get_all_refs():
            ref_id = ref.get("ref_id", "unknown")
            title = ref.get("title", "Untitled")
            year = ref.get("year", "?")
            ref_summaries_parts.append(f"  - [{ref_id}] {title} ({year})")
        ref_summaries = "\n".join(ref_summaries_parts) if ref_summaries_parts else "  (none)"

        key_points_str = "\n".join(f"  - {kp}" for kp in key_points) if key_points else "  (not specified)"

        prompt = SEARCH_JUDGMENT_PROMPT.format(
            section_type=section_type,
            section_title=section_title,
            paper_title=paper_title,
            key_points=key_points_str,
            n_refs=len(ref_pool.get_all_refs()),
            ref_summaries=ref_summaries,
        )

        print(f"[SearchJudge] Judging search need for {section_type} ({section_title})...")

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an academic research assistant. Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            raw = response.choices[0].message.content or ""
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

            result = json.loads(raw)

            need = result.get("need_search", False)
            reason = result.get("reason", "")
            queries = result.get("queries", [])

            if not isinstance(queries, list):
                queries = []
            queries = [q for q in queries if isinstance(q, str) and len(q.strip()) > 0]

            print(f"[SearchJudge] need_search={need}, reason={reason}")
            if queries:
                print(f"[SearchJudge] Suggested queries: {queries}")

            return {"need_search": bool(need), "reason": reason, "queries": queries}

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[SearchJudge] Failed to parse LLM response: {e}")
            print(f"[SearchJudge] Raw response: {raw[:300]}")
            return {"need_search": False, "reason": f"parse error: {e}", "queries": []}
        except Exception as e:
            print(f"[SearchJudge] LLM call failed: {e}")
            return {"need_search": False, "reason": f"error: {e}", "queries": []}

    # ------------------------------------------------------------------
    # Pre-generation search execution
    # ------------------------------------------------------------------

    async def _execute_pre_searches(
        self,
        queries: List[str],
        ref_pool: ReferencePool,
    ) -> int:
        """
        Execute paper searches and merge valid results into the reference pool.

        - **Description**:
            - Phase A continuation: after the LLM provides search queries,
              this method directly calls PaperSearchTool.execute() for each
              query (without going through the ReAct loop).
            - Found papers are validated and added to the ref_pool.
            - Returns the count of newly added references.

        - **Args**:
            - `queries` (List[str]): Search queries from the judgment step.
            - `ref_pool` (ReferencePool): Persistent reference pool to update.

        - **Returns**:
            - `int`: Number of new references added to the pool.
        """
        from ..shared.tools.paper_search import PaperSearchTool

        paper_search_cfg = self.host.tools_config.paper_search if self.host.tools_config else None
        tool = PaperSearchTool(
            serpapi_api_key=(
                paper_search_cfg.serpapi_api_key if paper_search_cfg else None
            ),
            semantic_scholar_api_key=(
                paper_search_cfg.semantic_scholar_api_key if paper_search_cfg else None
            ),
            timeout=paper_search_cfg.timeout if paper_search_cfg else 10,
            semantic_scholar_min_results_before_fallback=(
                paper_search_cfg.semantic_scholar_min_results_before_fallback
                if paper_search_cfg else 3
            ),
            enable_query_cache=paper_search_cfg.enable_query_cache if paper_search_cfg else True,
            cache_ttl_hours=paper_search_cfg.cache_ttl_hours if paper_search_cfg else 24,
        )

        added_count = 0
        for i, query in enumerate(queries):
            if i > 0:
                print("[PreSearch] Waiting 1.5s between queries to avoid rate limits...")
                await asyncio.sleep(1.5)

            print(f"[PreSearch] Executing search ({i+1}/{len(queries)}): '{query}'")
            try:
                per_round = paper_search_cfg.search_results_per_round if paper_search_cfg else 5
                result = await tool.execute(query=query, max_results=per_round)
                if not result.success:
                    print(f"[PreSearch] Search failed: {result.message}")
                    continue

                papers = result.data.get("papers", []) if result.data else []
                print(f"[PreSearch] Found {len(papers)} papers for '{query}'")

                for paper in papers:
                    bibtex = paper.get("bibtex", "")
                    cite_key = paper.get("bibtex_key", "") or paper.get("ref_id", "")
                    if not bibtex or not cite_key:
                        continue
                    if ref_pool.has_key(cite_key):
                        continue
                    added = ref_pool.add_discovered(cite_key, bibtex, source="pre_search")
                    if added:
                        added_count += 1
                        title = paper.get("title", "?")
                        print(f"[PreSearch] Added to ref_pool: [{cite_key}] {title}")

            except Exception as e:
                print(f"[PreSearch] Error searching '{query}': {e}")

        print(f"[PreSearch] Total new references added: {added_count} "
              f"(pool: {ref_pool.summary()})")
        return added_count

    # ------------------------------------------------------------------
    # Paper plan creation
    # ------------------------------------------------------------------

    async def _create_paper_plan(
        self,
        metadata: PaperMetaData,
        target_pages: Optional[int],
        style_guide: Optional[str],
        research_context: Optional[Dict[str, Any]] = None,
        code_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[PaperPlan]:
        """
        Create a paper plan by calling the Planner Agent.

        - **Args**:
            - `metadata` (PaperMetaData): Paper metadata
            - `target_pages` (Optional[int]): Target page count
            - `style_guide` (Optional[str]): Writing style guide (e.g., "ICML")

        - **Returns**:
            - `PaperPlan` or None if planning fails
        """
        from ..planner_agent.models import PlanRequest, FigureInfo, TableInfo
        from ..shared.code_context import format_code_context_for_planner

        try:
            figures_info = []
            for fig in metadata.figures:
                figures_info.append({
                    "id": fig.id,
                    "caption": fig.caption,
                    "description": fig.description,
                    "section": fig.section,
                    "wide": fig.wide,
                    "file_path": getattr(fig, "file_path", None) or "",
                })

            tables_info = []
            for tbl in metadata.tables:
                tables_info.append({
                    "id": tbl.id,
                    "caption": tbl.caption,
                    "description": tbl.description,
                    "section": tbl.section,
                    "wide": tbl.wide,
                    "file_path": getattr(tbl, "file_path", None) or "",
                })

            planner_code_brief = format_code_context_for_planner(
                context=code_context,
                style_guide=style_guide,
            ) if code_context else ""

            plan_request = PlanRequest(
                title=metadata.title,
                idea_hypothesis=metadata.idea_hypothesis,
                method=metadata.method,
                data=metadata.data,
                experiments=metadata.experiments,
                references=metadata.references,
                research_context=research_context,
                code_context=code_context,
                code_writing_assets={
                    **((code_context or {}).get("writing_assets", {}) or {}),
                    "planner_brief": planner_code_brief,
                } if code_context else None,
                figures=[FigureInfo(**fi) for fi in figures_info],
                tables=[TableInfo(**ti) for ti in tables_info],
                target_pages=target_pages,
                style_guide=style_guide,
            )

            paper_plan = await self.host._planner.create_plan(plan_request)
            return paper_plan

        except Exception as e:
            print(f"[MetaDataAgent] Planning error: {e}")
            return None

    # ------------------------------------------------------------------
    # LLM revision task planning
    # ------------------------------------------------------------------

    async def _llm_plan_revision_tasks(
        self,
        review_result: ReviewResult,
    ) -> List[Dict[str, Any]]:
        """
        Generate paragraph-level executable revision plans from tasks.
        """
        task_payload = [
            t.model_dump() if hasattr(t, "model_dump") else t
            for t in (review_result.revision_tasks or [])
        ]
        if not task_payload:
            return []

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a paragraph revision planner. Convert tasks into JSON-only "
                            "revision_plan entries with keys: task_id, section_type, target_paragraphs, "
                            "paragraph_instructions, preserve_claims, do_not_change, expected_change, priority, "
                            "issue_type, acceptance_criteria."
                        ),
                    },
                    {"role": "user", "content": json.dumps({"tasks": task_payload}, ensure_ascii=False)},
                ],
                temperature=0.2,
                max_tokens=1800,
            )
            raw = response.choices[0].message.content or ""
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            parsed = json.loads(cleaned) if cleaned else {}
            if isinstance(parsed, list):
                return parsed
            plan = parsed.get("revision_plan", []) if isinstance(parsed, dict) else []
            if isinstance(plan, list):
                return plan
        except Exception as e:
            print(f"[ReviewPlanner] LLM planner fallback: {e}")

        # Deterministic fallback
        fallback: List[Dict[str, Any]] = []
        for idx, task in enumerate(task_payload):
            section_type = str(task.get("section_type", ""))
            if not section_type:
                continue
            tps = [int(x) for x in task.get("paragraph_indices", []) if isinstance(x, int) or str(x).isdigit()]
            instruction = str(task.get("instruction", "")) or str(task.get("rationale", "Revise this section."))
            paragraph_instructions = {p: instruction for p in tps}
            fallback.append({
                "task_id": task.get("task_id", f"fallback_{idx}"),
                "section_type": section_type,
                "target_paragraphs": tps,
                "paragraph_instructions": paragraph_instructions,
                "preserve_claims": task.get("preserve_claims", []),
                "do_not_change": task.get("do_not_change", []),
                "expected_change": task.get("expected_change", ""),
                "priority": int(task.get("priority", 5)),
                "issue_type": str(task.get("issue_type", "other")),
                "acceptance_criteria": [str(x) for x in task.get("acceptance_criteria", [])],
            })
        return fallback

    # ------------------------------------------------------------------
    # Apply revision plan to feedbacks
    # ------------------------------------------------------------------

    def _apply_revision_plan_to_feedbacks(
        self,
        review_result: ReviewResult,
        revision_plan: List[Dict[str, Any]],
    ) -> None:
        """
        Merge revision-plan paragraph instructions into section feedbacks.
        """
        if not revision_plan:
            return
        by_section = {sf.section_type: sf for sf in review_result.section_feedbacks}
        for plan_item in revision_plan:
            section_type = str(plan_item.get("section_type", ""))
            if not section_type:
                continue
            sf = by_section.get(section_type)
            if sf is None:
                sf = SectionFeedback(
                    section_type=section_type,
                    current_word_count=0,
                    target_word_count=0,
                    action="refine_paragraphs",
                    delta_words=0,
                    revision_prompt=str(plan_item.get("expected_change", "")),
                    target_paragraphs=[],
                    paragraph_instructions={},
                )
                review_result.section_feedbacks.append(sf)
                by_section[section_type] = sf
            targets = self.host._resolver._normalize_target_paragraphs(plan_item.get("target_paragraphs", []))
            sf.target_paragraphs = sorted(list(set((sf.target_paragraphs or []) + targets)))
            normalized_instructions = self.host._resolver._normalize_paragraph_instructions(
                plan_item.get("paragraph_instructions", {}),
                target_paragraphs=sf.target_paragraphs,
                fallback_instruction=str(plan_item.get("expected_change", "")).strip(),
            )
            for k, v in normalized_instructions.items():
                try:
                    sf.paragraph_instructions[int(k)] = str(v)
                except Exception:
                    continue
            if sf.target_paragraphs and sf.action == "revise":
                sf.action = "refine_paragraphs"
            if plan_item.get("issue_type"):
                try:
                    sf.issue_type = plan_item.get("issue_type")
                except Exception:
                    pass
            criteria = plan_item.get("acceptance_criteria", []) or []
            if criteria:
                sf.acceptance_criteria = [str(x) for x in criteria]
            preserve_claims = plan_item.get("preserve_claims", [])
            do_not_change = plan_item.get("do_not_change", [])
            expected_change = plan_item.get("expected_change", "")
            extra = []
            if preserve_claims:
                extra.append(f"Preserve claims: {', '.join([str(x) for x in preserve_claims])}.")
            if do_not_change:
                extra.append(f"Do not change: {', '.join([str(x) for x in do_not_change])}.")
            if expected_change:
                extra.append(f"Expected change: {expected_change}.")
            if extra:
                sf.revision_prompt = (sf.revision_prompt + "\n\n" + " ".join(extra)).strip()

    # ------------------------------------------------------------------
    # VLM-to-revision-plan translation
    # ------------------------------------------------------------------

    async def _translate_vlm_to_revision_plan(
        self,
        vlm_result: Dict[str, Any],
        generated_sections: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Translate VLM section advice into joint text/structure revision tasks.
        """
        if not vlm_result:
            return []
        payload = {
            "summary": vlm_result.get("summary", ""),
            "needs_trim": bool(vlm_result.get("needs_trim", False)),
            "needs_expand": bool(vlm_result.get("needs_expand", False)),
            "section_recommendations": {
                sec: (advice.model_dump() if hasattr(advice, "model_dump") else advice)
                for sec, advice in (vlm_result.get("section_recommendations", {}) or {}).items()
            },
            "sections": {k: v[:1800] for k, v in generated_sections.items()},
        }
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You translate VLM layout feedback into executable JSON revision tasks. "
                            "Return JSON only with key vlm_revision_plan as list of objects: "
                            "section_type, action, delta_words, paragraph_instructions, rationale."
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.2,
                max_tokens=1200,
            )
            raw = response.choices[0].message.content or ""
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            parsed = json.loads(cleaned) if cleaned else {}
            plan = parsed.get("vlm_revision_plan", [])
            if isinstance(plan, list):
                return plan
        except Exception as e:
            print(f"[VLMTranslate] LLM translator fallback: {e}")

        fallback = []
        for section_type, advice in (vlm_result.get("section_recommendations", {}) or {}).items():
            action = getattr(advice, "recommended_action", None) or advice.get("recommended_action")
            target_change = getattr(advice, "target_change", None) or advice.get("target_change") or 0
            fallback.append({
                "section_type": section_type,
                "action": "reduce" if action == "trim" else ("expand" if action == "expand" else "ok"),
                "delta_words": int(target_change or 0),
                "paragraph_instructions": {},
                "rationale": "VLM fallback translation",
            })
        return fallback
