"""
Revision Executor — extracted from MetaDataAgent for architecture decoupling.
- **Description**:
    - Executes section, paragraph, and sentence level revisions
    - Manages semantic consistency guards between before/after revision
    - Handles reviewer interaction for review-based revisions
"""
from __future__ import annotations

import json
import re
import logging
import hashlib
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Tuple,
    Set,
    TYPE_CHECKING,
)

from ..reviewer_agent.models import (
    ReviewResult,
    FeedbackResult,
    Severity,
    SectionFeedback,
    SemanticCheckRecord,
    HierarchicalFeedbackItem,
    FeedbackLevel,
)
from ..shared.session_memory import SessionMemory, ReviewRecord
from ...models.evidence_graph import EvidenceDAG

if TYPE_CHECKING:
    from .models import PaperMetaData, SectionResult

logger = logging.getLogger("uvicorn.error")


class RevisionExecutor:
    """
    Encapsulates all revision-execution logic previously embedded in MetaDataAgent.
    - **Description**:
        - Receives the host MetaDataAgent instance and delegates back for
          utilities that remain on the host (e.g. _fix_latex_references).
        - Owns section / paragraph / sentence revision, semantic consistency
          guards, reviewer calls, and fingerprinting helpers.
    """

    def __init__(self, host: Any) -> None:
        """
        Initialize RevisionExecutor.
        - **Description**:
            - Stores a reference to the host MetaDataAgent.
            - client and model_name are resolved lazily from the host.

        - **Args**:
            - `host`: The MetaDataAgent instance that owns this executor.
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
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_acceptance_criteria(issue_type: str) -> List[str]:
        """
        Build default acceptance gates for generalized issue handling.
        - **Description**:
         - Uses issue taxonomy to derive validation gates
         - Keeps contracts stable when reviewer does not supply criteria
        """
        base = ["execution_changed", "semantic_preserved"]
        normalized = str(issue_type or "other").strip().lower()
        if normalized == "logical_contradiction":
            return base + ["contradiction_resolved"]
        if normalized in ("claim_evidence_gap", "unsupported_generalization"):
            return base + ["evidence_sufficient"]
        return base

    @staticmethod
    def _split_section_paragraphs(content: str) -> List[str]:
        """
        Split section content into paragraph units.
        - **Description**:
            - Uses blank lines as paragraph boundaries
            - Preserves paragraph order for stable paragraph IDs
        """
        if not content:
            return []
        return [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]

    @staticmethod
    def _join_section_paragraphs(paragraphs: List[str]) -> str:
        """Join paragraph units back into section LaTeX text."""
        return "\n\n".join([p for p in paragraphs if p is not None]).strip()

    # ------------------------------------------------------------------
    # Semantic consistency guard
    # ------------------------------------------------------------------

    async def _run_semantic_consistency_guard(
        self,
        section_type: str,
        before_text: str,
        after_text: str,
        revision_prompt: str,
    ) -> SemanticCheckRecord:
        """
        Check semantic consistency of before/after revisions.
        """
        if before_text.strip() == after_text.strip():
            return SemanticCheckRecord(
                section_type=section_type,
                passed=True,
                summary="No semantic delta detected.",
                risks=[],
                action_taken="accepted",
            )
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a semantic consistency guard for academic revisions. "
                            "Return JSON only with keys: passed (bool), summary (str), risks (list[str]). "
                            "Fail if key claims are dropped or contradictions are introduced."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "section_type": section_type,
                                "revision_prompt": revision_prompt[:1200],
                                "before": before_text[:7000],
                                "after": after_text[:7000],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=600,
            )
            raw = response.choices[0].message.content or ""
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            parsed = json.loads(cleaned) if cleaned else {}
            passed = bool(parsed.get("passed", True))
            summary = str(parsed.get("summary", "Semantic check completed."))
            risks = [str(x) for x in parsed.get("risks", [])]
            return SemanticCheckRecord(
                section_type=section_type,
                passed=passed,
                summary=summary,
                risks=risks,
                action_taken="accepted" if passed else "rollback",
            )
        except Exception as e:
            return SemanticCheckRecord(
                section_type=section_type,
                passed=False,
                summary=f"Semantic guard failed due to error: {e}",
                risks=["semantic_guard_unavailable"],
                action_taken="rollback",
            )

    # ------------------------------------------------------------------
    # Core revision orchestration
    # ------------------------------------------------------------------

    async def _apply_revisions(
        self,
        review_result: ReviewResult,
        generated_sections: Dict[str, str],
        sections_results: List["SectionResult"],
        valid_citation_keys: set,
        metadata: "PaperMetaData",
        memory: Optional[SessionMemory] = None,
        semantic_checks: Optional[List[SemanticCheckRecord]] = None,
        decision_trace: Optional[List[Dict[str, Any]]] = None,
        writer_response_section: Optional[List[Dict[str, Any]]] = None,
        writer_response_paragraph: Optional[List[Dict[str, Any]]] = None,
        reviewer_verification: Optional[List[Dict[str, Any]]] = None,
    ) -> set:
        """
        Apply revisions based on a unified review result.
        - **Description**:
            - Uses review_result.section_feedbacks to revise sections
            - Updates generated_sections and sections_results in place
            - Injects revision history from memory to prevent regression
        
        - **Args**:
            - `review_result` (ReviewResult): Unified review result
            - `generated_sections` (Dict[str, str]): Section contents
            - `sections_results` (List[SectionResult]): Section results to update
            - `valid_citation_keys` (set): Valid citation keys
            - `metadata` (PaperMetaData): Original metadata for context
            - `memory` (SessionMemory, optional): Session memory for revision context
        
        - **Returns**:
            - `revised_sections` (set): Section types that were revised
        """
        revised_sections: set = set()
        if not review_result or not review_result.section_feedbacks:
            return revised_sections
        
        for sf in review_result.section_feedbacks:
            if sf.action == "ok":
                continue
            if sf.section_type not in generated_sections:
                continue
            
            revision_prompt = sf.revision_prompt
            if not revision_prompt:
                continue
            task_contract = {
                "target": sf.target_id or sf.section_type,
                "target_level": "paragraph" if sf.target_paragraphs else "section",
                "instruction": revision_prompt,
                "issue_type": str(getattr(sf, "issue_type", "other") or "other"),
                "constraints": {
                    "preserve_claims": ["Preserve factual claims and citations."],
                    "do_not_change": [],
                },
                "acceptance_criteria": self._default_acceptance_criteria(
                    str(getattr(sf, "issue_type", "other") or "other")
                ),
                "target_paragraphs": sf.target_paragraphs or [],
                "paragraph_instructions": sf.paragraph_instructions or {},
            }
            if getattr(sf, "acceptance_criteria", None):
                task_contract["acceptance_criteria"] = [str(x) for x in (sf.acceptance_criteria or [])]
            if memory is not None:
                issue_ctx = memory.get_issue_context(limit=60)
                unresolved = issue_ctx.get("unresolved_issues", [])
                hard_rules = []
                soft_hints = []
                for issue in unresolved:
                    sec = str(issue.get("section_type", ""))
                    target_id = str(issue.get("target_id", ""))
                    if sec not in ("", sf.section_type) and not target_id.startswith(f"{sf.section_type}."):
                        continue
                    msg = str(issue.get("message", ""))[:220]
                    if str(issue.get("locked_mode", "soft")) == "hard":
                        hard_rules.append(msg)
                    else:
                        soft_hints.append(msg)
                if hard_rules:
                    revision_prompt = (
                        "HARD LOCK ISSUES (must remain resolved; do not reintroduce):\n"
                        + "\n".join([f"- {x}" for x in hard_rules[:8]])
                        + "\n\n"
                        + revision_prompt
                    )
                if soft_hints:
                    revision_prompt = (
                        revision_prompt
                        + "\n\nSOFT LOCK GUIDANCE (avoid regression if possible):\n"
                        + "\n".join([f"- {x}" for x in soft_hints[:6]])
                    )
            
            print(
                "[ReviewLoop] Applying revision: "
                f"section={sf.section_type} action={sf.action} delta_words={sf.delta_words}"
            )
            
            current_content = generated_sections[sf.section_type]
            before_content = current_content

            if sf.target_paragraphs:
                revised_content = await self._revise_section_paragraphs(
                    section_type=sf.section_type,
                    current_content=current_content,
                    target_paragraphs=sf.target_paragraphs,
                    paragraph_instructions=sf.paragraph_instructions or {},
                    fallback_prompt=revision_prompt,
                    metadata=metadata,
                    memory=memory,
                    task_contract_base=task_contract,
                    valid_citation_keys=valid_citation_keys,
                )
            else:
                revised_content = await self._revise_section(
                    section_type=sf.section_type,
                    current_content=current_content,
                    revision_prompt=revision_prompt,
                    metadata=metadata,
                    memory=memory,
                    task_contract=task_contract,
                    valid_citation_keys=valid_citation_keys,
                )
            
            if revised_content:
                revised_content = self.host._fix_latex_references(revised_content)
                revised_content, invalid_citations, _ = self.host._validate_and_fix_citations(
                    revised_content, valid_citation_keys, remove_invalid=True
                )
                if invalid_citations:
                    print(f"[ReviewLoop] Removed {len(invalid_citations)} invalid citations from {sf.section_type}: {invalid_citations[:3]}{'...' if len(invalid_citations) > 3 else ''}")
                    if decision_trace is not None:
                        decision_trace.append(
                            {
                                "section_type": sf.section_type,
                                "decision": "removed_invalid_citations",
                                "role": "final_assert",
                                "note": "Citation validation at review stage is the FINAL ASSERT; primary prevention happened during generation (ClaimVerifier)",
                                "count": len(invalid_citations),
                                "keys": list(invalid_citations),
                            }
                        )
                semantic_record = await self._run_semantic_consistency_guard(
                    section_type=sf.section_type,
                    before_text=before_content,
                    after_text=revised_content,
                    revision_prompt=revision_prompt,
                )
                if semantic_checks is not None:
                    semantic_checks.append(semantic_record)
                if not semantic_record.passed:
                    if decision_trace is not None:
                        decision_trace.append({
                            "section_type": sf.section_type,
                            "decision": "rollback_revision",
                            "reason": semantic_record.summary,
                            "risks": semantic_record.risks,
                        })
                    fallback_content = await self._revise_section(
                        section_type=sf.section_type,
                        current_content=before_content,
                        revision_prompt=revision_prompt,
                        metadata=metadata,
                        memory=memory,
                        task_contract=task_contract,
                        valid_citation_keys=valid_citation_keys,
                    )
                    if fallback_content:
                        fallback_content = self.host._fix_latex_references(fallback_content)
                        fallback_content, _, _ = self.host._validate_and_fix_citations(
                            fallback_content, valid_citation_keys, remove_invalid=True
                        )
                        fallback_semantic = await self._run_semantic_consistency_guard(
                            section_type=sf.section_type,
                            before_text=before_content,
                            after_text=fallback_content,
                            revision_prompt=revision_prompt,
                        )
                        if semantic_checks is not None:
                            semantic_checks.append(fallback_semantic)
                        if fallback_semantic.passed:
                            revised_content = fallback_content
                            if decision_trace is not None:
                                decision_trace.append({
                                    "section_type": sf.section_type,
                                    "decision": "accept_fallback_section_rewrite",
                                    "reason": fallback_semantic.summary,
                                })
                        else:
                            if decision_trace is not None:
                                decision_trace.append({
                                    "section_type": sf.section_type,
                                    "decision": "keep_original_after_failed_semantic_checks",
                                    "reason": fallback_semantic.summary,
                                    "risks": fallback_semantic.risks,
                                })
                            continue
                    else:
                        continue
                
                generated_sections[sf.section_type] = revised_content
                new_word_count = len(revised_content.split())
                if writer_response_section is not None:
                    writer_response_section.append({
                        "target_id": sf.target_id or sf.section_type,
                        "section_type": sf.section_type,
                        "target": task_contract["target"],
                        "instruction": task_contract["instruction"],
                        "constraints": task_contract["constraints"],
                        "disposition": "executed",
                        "source_agent": "writer",
                        "evidence": {
                            "before_words": len(before_content.split()),
                            "after_words": new_word_count,
                        },
                    })
                if writer_response_paragraph is not None:
                    final_paragraphs = self._split_section_paragraphs(revised_content)
                    final_para_count = len(final_paragraphs)
                    for pidx in (sf.target_paragraphs or []):
                        mapped_to: List[int] = []
                        mapping_strategy = "identity"
                        if 0 <= int(pidx) < final_para_count:
                            mapped_to = [int(pidx)]
                        elif final_para_count > 0:
                            mapped_to = [min(max(int(pidx), 0), final_para_count - 1)]
                            mapping_strategy = "clamped_nearest"
                        else:
                            mapping_strategy = "no_paragraphs"
                        writer_response_paragraph.append({
                            "target_id": f"{sf.section_type}.p{int(pidx)}",
                            "section_type": sf.section_type,
                            "paragraph_index": int(pidx),
                            "target": f"{sf.section_type}.p{int(pidx)}",
                            "disposition": "executed",
                            "instruction": str((task_contract.get("paragraph_instructions", {}) or {}).get(int(pidx), "")),
                            "constraints": task_contract["constraints"],
                            "source_agent": "writer",
                            "evidence": {
                                "content_changed": True,
                                "mapped_to_paragraph_indices": mapped_to,
                                "mapping_strategy": mapping_strategy,
                                "final_paragraph_count": final_para_count,
                            },
                        })
                if reviewer_verification is not None:
                    verify_result: Dict[str, Any] = {
                        "section_type": sf.section_type,
                        "target": task_contract["target"],
                        "passed": semantic_record.passed,
                        "summary": semantic_record.summary,
                        "checker": "reviewer_verifier",
                    }
                    if hasattr(self.host._reviewer, "verify_execution"):
                        try:
                            verify_result = await self.host._reviewer.verify_execution(
                                section_type=sf.section_type,
                                task_contract=task_contract,
                                before_text=before_content,
                                after_text=revised_content,
                                semantic_passed=semantic_record.passed,
                                semantic_summary=semantic_record.summary,
                            )
                        except Exception:
                            pass
                    reviewer_verification.append(verify_result)
                
                for sr in sections_results:
                    if sr.section_type == sf.section_type:
                        sr.latex_content = revised_content
                        sr.word_count = new_word_count
                        break
                
                revised_sections.add(sf.section_type)
                print(f"[MetaDataAgent] Revised {sf.section_type}: {new_word_count} words")

        # Sentence-level revision pass (runs after section/paragraph revisions)
        if review_result.hierarchical_feedbacks:
            sent_fb_by_section: Dict[str, List[Dict[str, Any]]] = {}
            for hf in review_result.hierarchical_feedbacks:
                if hf.level == FeedbackLevel.SENTENCE and hf.section_type:
                    sent_fb_by_section.setdefault(hf.section_type, []).append({
                        "paragraph_index": hf.paragraph_index or 0,
                        "sentence_index": hf.sentence_index or 0,
                        "issue": hf.message,
                        "suggestion": hf.revision_instruction,
                        "severity": hf.severity.value if hasattr(hf.severity, "value") else str(hf.severity),
                    })
            for sec_type, sent_fbs in sent_fb_by_section.items():
                if sec_type not in generated_sections:
                    continue
                current = generated_sections[sec_type]
                revised = await self._revise_section_sentences(
                    section_type=sec_type,
                    current_content=current,
                    sentence_feedbacks=sent_fbs,
                    metadata=metadata,
                    memory=memory,
                    valid_citation_keys=valid_citation_keys,
                )
                if revised and revised.strip():
                    revised = self.host._fix_latex_references(revised)
                    revised, _, _ = self.host._validate_and_fix_citations(
                        revised, valid_citation_keys, remove_invalid=True
                    )
                    generated_sections[sec_type] = revised
                    for sr in sections_results:
                        if sr.section_type == sec_type:
                            sr.latex_content = revised
                            sr.word_count = len(revised.split())
                            break
                    revised_sections.add(sec_type)
                    print(f"[MetaDataAgent] Sentence-level revised {sec_type}")

        return revised_sections

    # ------------------------------------------------------------------
    # Paragraph-level revision
    # ------------------------------------------------------------------

    async def _revise_section_paragraphs(
        self,
        section_type: str,
        current_content: str,
        target_paragraphs: List[int],
        paragraph_instructions: Dict[int, str],
        fallback_prompt: str,
        metadata: "PaperMetaData",
        memory: Optional[SessionMemory] = None,
        task_contract_base: Optional[Dict[str, Any]] = None,
        valid_citation_keys: Optional[set] = None,
    ) -> Optional[str]:
        """
        Revise selected paragraphs, then reassemble the section.
        - **Description**:
            - Applies targeted paragraph-level revision prompts
            - Falls back to whole-section revision if paragraph pass fails
        """
        paragraphs = self._split_section_paragraphs(current_content)
        if not paragraphs:
            return await self._revise_section(
                section_type=section_type,
                current_content=current_content,
                revision_prompt=fallback_prompt,
                metadata=metadata,
                memory=memory,
                task_contract=task_contract_base,
                valid_citation_keys=valid_citation_keys,
            )

        revised_any = False
        for pidx in sorted(set(target_paragraphs)):
            if pidx < 0 or pidx >= len(paragraphs):
                continue
            instruction = paragraph_instructions.get(
                pidx,
                "Improve this paragraph according to reviewer feedback while preserving factual correctness.",
            )
            revised_paragraph = await self._revise_paragraph(
                section_type=section_type,
                paragraph_index=pidx,
                paragraph_text=paragraphs[pidx],
                instruction=instruction,
                memory=memory,
                task_contract=task_contract_base,
                valid_citation_keys=valid_citation_keys,
            )
            if revised_paragraph and revised_paragraph.strip():
                paragraphs[pidx] = revised_paragraph.strip()
                revised_any = True

        if revised_any:
            return self._join_section_paragraphs(paragraphs)

        return await self._revise_section(
            section_type=section_type,
            current_content=current_content,
            revision_prompt=fallback_prompt,
            metadata=metadata,
            memory=memory,
            task_contract=task_contract_base,
            valid_citation_keys=valid_citation_keys,
        )

    # ------------------------------------------------------------------
    # Sentence-level revision
    # ------------------------------------------------------------------

    async def _revise_section_sentences(
        self,
        section_type: str,
        current_content: str,
        sentence_feedbacks: List[Dict[str, Any]],
        metadata: "PaperMetaData",
        memory: Optional[SessionMemory] = None,
        valid_citation_keys: Optional[set] = None,
    ) -> Optional[str]:
        """
        Apply sentence-level revisions within a section.
        - **Description**:
            - Splits section into paragraphs, then sentences
            - Generates focused revision prompts per sentence with paragraph context
            - Reassembles after sentence-level fixes

        - **Args**:
            - `section_type` (str): The section being revised
            - `current_content` (str): Current LaTeX content of the section
            - `sentence_feedbacks` (List[Dict]): Sentence-level feedback items
            - `metadata` (PaperMetaData): Paper metadata
            - `memory` (SessionMemory, optional): Session memory
            - `valid_citation_keys` (set, optional): Valid citation keys

        - **Returns**:
            - Revised section content, or None if no changes made
        """
        paragraphs = self._split_section_paragraphs(current_content)
        if not paragraphs:
            return None

        fb_by_para: Dict[int, List[Dict[str, Any]]] = {}
        for fb in sentence_feedbacks:
            pidx = int(fb.get("paragraph_index", 0))
            fb_by_para.setdefault(pidx, []).append(fb)

        revised_any = False
        for pidx in sorted(fb_by_para.keys()):
            if pidx < 0 or pidx >= len(paragraphs):
                continue
            para_text = paragraphs[pidx]
            sentences = re.split(r'(?<=[.!?])\s+', para_text)
            if not sentences:
                continue

            para_revised = False
            for fb in fb_by_para[pidx]:
                sidx = int(fb.get("sentence_index", 0))
                if sidx < 0 or sidx >= len(sentences):
                    continue

                target_sentence = sentences[sidx]
                context_before = " ".join(sentences[max(0, sidx - 1):sidx])
                context_after = " ".join(sentences[sidx + 1:sidx + 2])

                prompt_parts = [
                    f"Revise the TARGET SENTENCE in the {section_type} section.",
                    f"\nIssue: {fb.get('issue', fb.get('message', ''))}",
                    f"Suggestion: {fb.get('suggestion', fb.get('revision_instruction', ''))}",
                    f"\nContext before: {context_before}" if context_before else "",
                    f"TARGET SENTENCE: {target_sentence}",
                    f"Context after: {context_after}" if context_after else "",
                    "\nReturn ONLY the revised sentence. Preserve LaTeX commands and citations.",
                ]
                revision_prompt = "\n".join(p for p in prompt_parts if p)

                try:
                    revised_sentence = await self.host._writer.run(
                        content=revision_prompt,
                        mode="revision",
                    )
                    if revised_sentence and revised_sentence.strip():
                        sentences[sidx] = revised_sentence.strip()
                        para_revised = True
                except Exception as e:
                    logger.warning(
                        "sentence_revision failed section=%s p=%d s=%d: %s",
                        section_type, pidx, sidx, e,
                    )

            if para_revised:
                paragraphs[pidx] = " ".join(sentences)
                revised_any = True

        if revised_any:
            return self._join_section_paragraphs(paragraphs)
        return None

    # ------------------------------------------------------------------
    # Fingerprinting
    # ------------------------------------------------------------------

    def _get_sections_fingerprint(self, sections: Dict[str, str]) -> str:
        """
        Build a stable fingerprint for section content.
        - **Description**:
            - Generates a hash string from section contents
            - Used to detect no-op revisions
        
        - **Args**:
            - `sections` (Dict[str, str]): Section contents
        
        - **Returns**:
            - `fingerprint` (str): SHA-256 fingerprint
        """
        hasher = hashlib.sha256()
        for section_type in sorted(sections.keys()):
            hasher.update(section_type.encode("utf-8"))
            hasher.update(b"\n")
            hasher.update(sections[section_type].encode("utf-8"))
            hasher.update(b"\n")
        return hasher.hexdigest()

    # ------------------------------------------------------------------
    # Reviewer interaction
    # ------------------------------------------------------------------

    async def _call_reviewer(
        self,
        sections: Dict[str, str],
        word_counts: Dict[str, int],
        target_pages: Optional[int],
        style_guide: Optional[str],
        template_path: Optional[str],
        iteration: int,
        section_targets: Optional[Dict[str, int]] = None,
        section_structure_signals: Optional[Dict[str, Any]] = None,
        memory: Optional[SessionMemory] = None,
        evidence_dag: Optional[EvidenceDAG] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        """
        Call Reviewer Agent directly (no HTTP) to check the paper.
        - **Description**:
            - Builds a ReviewContext and calls self.host._reviewer.review() directly.
            - Passes memory so checkers can read prior issues natively.

        - **Returns**:
            - Tuple of (review_result_dict, target_word_count) or (None, None) on failure
        """
        try:
            from ..reviewer_agent.models import ReviewContext as RC

            review_ctx = RC(
                sections=sections,
                word_counts=word_counts,
                target_pages=target_pages or 8,
                style_guide=style_guide,
                template_path=template_path,
                metadata={},
            )
            review_ctx.metadata["review_structure_gate_enabled"] = bool(
                getattr(self.host.tools_config, "review_structure_gate_enabled", True)
            ) if self.host.tools_config else True
            review_ctx.metadata["structure_gate_min_paragraph_threshold"] = int(
                getattr(self.host.tools_config, "structure_gate_min_paragraph_threshold", 5)
            ) if self.host.tools_config else 5
            if section_structure_signals:
                review_ctx.metadata["section_structure_signals"] = section_structure_signals
            if memory is not None:
                review_ctx.metadata["issue_memory"] = memory.get_issue_context(limit=40)
            if section_targets:
                review_ctx.section_targets = section_targets
            if evidence_dag is not None:
                review_ctx.metadata["evidence_dag"] = evidence_dag.to_serializable()

            review_result = await self.host._reviewer.review(
                context=review_ctx,
                iteration=iteration,
                memory=memory,
            )

            result = review_result.model_dump()

            target_word_count = None
            for fb in result.get("feedbacks", []):
                details = fb.get("details", {})
                if "target_words" in details:
                    target_word_count = details["target_words"]
                    break

            return result, target_word_count

        except Exception as e:
            print(f"[MetaDataAgent] Review error: {e}")
            return None, None

    # ------------------------------------------------------------------
    # Section-level revision
    # ------------------------------------------------------------------

    async def _revise_section(
        self,
        section_type: str,
        current_content: str,
        revision_prompt: str,
        metadata: "PaperMetaData",
        memory: Optional[SessionMemory] = None,
        task_contract: Optional[Dict[str, Any]] = None,
        valid_citation_keys: Optional[set] = None,
    ) -> Optional[str]:
        """
        Revise a section based on feedback — delegates to WriterAgent.
        - **Description**:
            - Packs the revision instructions + current content as a user_prompt
              and delegates to WriterAgent, which can consult memory/planner/reviewer
              via AskTool during the ReAct loop.

        - **Args**:
            - `section_type` (str): Type of section to revise
            - `current_content` (str): Current section LaTeX content
            - `revision_prompt` (str): Instructions for the revision
            - `metadata` (PaperMetaData): Paper metadata for context
            - `memory` (SessionMemory, optional): Session memory

        - **Returns**:
            - Revised content string, or None on failure
        """
        try:
            system_prompt = (
                "You are an expert academic writer revising a paper section.\n"
                "Follow the revision instructions carefully to improve the content.\n"
                "Maintain academic writing quality.\n"
                "Output ONLY the revised LaTeX content, no explanations or preamble."
            )

            revision_ctx = ""
            if memory:
                revision_ctx = memory.get_revision_context(section_type)
                issue_ctx = memory.get_issue_context(limit=20)
                unresolved = issue_ctx.get("unresolved_issues", [])
                if unresolved:
                    pinned = []
                    for item in unresolved[:8]:
                        if str(item.get("section_type", "")) in ("", section_type):
                            pinned.append(
                                f"- [{item.get('locked_mode', 'soft')}] {item.get('target_id', '')}: {str(item.get('message', ''))[:180]}"
                            )
                    if pinned:
                        revision_ctx = (
                            revision_ctx
                            + ("\n\n" if revision_ctx else "")
                            + "## Unresolved Memory Issues\n"
                            + "\n".join(pinned)
                        )

            if "Current content" in revision_prompt or current_content in revision_prompt:
                user_message = revision_prompt
            else:
                user_message = (
                    f"{revision_prompt}\n\n"
                    f"Current content of the {section_type} section to revise:\n"
                    f"{current_content}"
                )

            if revision_ctx:
                user_message = f"{revision_ctx}\n\n{user_message}"
            user_message += (
                "\n\nCitation guardrails:\n"
                "- Preserve all existing valid \\cite{...} commands unless explicitly instructed to remove invalid ones.\n"
                "- Do not drop citations during style or structure edits.\n"
            )

            result = await self.host._writer.run(
                system_prompt=system_prompt,
                user_prompt=user_message,
                section_type=section_type,
                enable_review=False,
                valid_citation_keys=sorted(list(valid_citation_keys or [])),
                mode="revision",
                current_content=current_content,
                revision_plan={
                    **(task_contract or {}),
                    "section_type": section_type,
                    "target": str((task_contract or {}).get("target") or section_type),
                    "target_level": "section",
                    "instruction": str((task_contract or {}).get("instruction") or revision_prompt),
                    "constraints": (
                        (task_contract or {}).get("constraints")
                        or {
                            "preserve_claims": ["Preserve factual claims and citations."],
                            "do_not_change": [],
                        }
                    ),
                },
                memory=memory,
                peers={"planner": self.host._planner, "reviewer": self.host._reviewer},
            )
            revised = result.get("generated_content", "")
            return revised if revised else None

        except Exception as e:
            print(f"[MetaDataAgent] Revision error for {section_type}: {e}")
            return None

    # ------------------------------------------------------------------
    # Paragraph-level single revision
    # ------------------------------------------------------------------

    async def _revise_paragraph(
        self,
        section_type: str,
        paragraph_index: int,
        paragraph_text: str,
        instruction: str,
        memory: Optional[SessionMemory] = None,
        task_contract: Optional[Dict[str, Any]] = None,
        valid_citation_keys: Optional[set] = None,
    ) -> Optional[str]:
        """
        Revise a single paragraph using WriterAgent.
        - **Description**:
            - Uses a focused prompt to avoid unnecessary edits to other paragraphs
            - Returns revised paragraph text only
        """
        try:
            system_prompt = (
                "You are an expert academic editor revising one paragraph.\n"
                "Keep the same scientific meaning and preserve LaTeX correctness.\n"
                "Return ONLY the revised paragraph text."
            )
            revision_ctx = ""
            if memory:
                revision_ctx = memory.get_revision_context(section_type)
                issue_ctx = memory.get_issue_context(limit=20)
                unresolved = issue_ctx.get("unresolved_issues", [])
                if unresolved:
                    pinned = []
                    for item in unresolved[:12]:
                        target_id = str(item.get("target_id", ""))
                        if target_id.endswith(f".p{paragraph_index}") or str(item.get("section_type", "")) == section_type:
                            pinned.append(
                                f"- [{item.get('locked_mode', 'soft')}] {target_id}: {str(item.get('message', ''))[:180]}"
                            )
                    if pinned:
                        revision_ctx = (
                            revision_ctx
                            + ("\n\n" if revision_ctx else "")
                            + "## Memory Issues For This Paragraph\n"
                            + "\n".join(pinned)
                        )

            user_message = (
                f"Section: {section_type}\n"
                f"Paragraph index: {paragraph_index}\n"
                f"Instruction: {instruction}\n\n"
                f"Current paragraph:\n{paragraph_text}"
            )
            if revision_ctx:
                user_message = f"{revision_ctx}\n\n{user_message}"
            user_message += (
                "\n\nCitation guardrails:\n"
                "- Preserve existing valid citations in this paragraph.\n"
                "- Only remove citations when they are explicitly marked invalid.\n"
            )

            result = await self.host._writer.run(
                system_prompt=system_prompt,
                user_prompt=user_message,
                section_type=section_type,
                enable_review=False,
                valid_citation_keys=sorted(list(valid_citation_keys or [])),
                mode="revision",
                current_content=paragraph_text,
                revision_plan={
                    **(task_contract or {}),
                    "section_type": section_type,
                    "target": f"{section_type}.p{paragraph_index}",
                    "target_level": "paragraph",
                    "instruction": instruction,
                    "constraints": (
                        (task_contract or {}).get("constraints")
                        or {
                            "preserve_claims": ["Preserve the original scientific claim and evidence links."],
                            "do_not_change": [],
                        }
                    ),
                    "target_paragraphs": [paragraph_index],
                    "paragraph_instructions": {paragraph_index: instruction},
                },
                memory=memory,
                peers={"planner": self.host._planner, "reviewer": self.host._reviewer},
            )
            revised = result.get("generated_content", "")
            return revised if revised else None
        except Exception as e:
            print(
                f"[MetaDataAgent] Paragraph revision error for {section_type}.p{paragraph_index}: {e}"
            )
            return None
