"""
Claim-Level Verification for Decomposed Generation
- **Description**:
    - Provides immediate (not post-hoc) verification of paragraph-level
      generated content against the EvidenceDAG and citation database.
    - Three checks run in sequence:
      1. Citation validity — reuses CitationValidatorTool logic.
      2. Evidence anchor — ensures bound evidence IDs are referenced.
      3. Key-point coverage — checks that the paragraph's stated goals
         are reflected in the output.
    - Produces a ``VerificationResult`` with structured feedback that can
      be fed back to the Writer for retry, or trigger degradation to
      template-slot filling.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CLAIM_RETRIES: int = 3
TEMPLATE_FALLBACK_ENABLED: bool = True


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class VerificationResult(BaseModel):
    """
    Outcome of claim-level verification on a generated paragraph.
    - **Fields**:
        - ``passed``: Whether all checks passed.
        - ``citation_issues``: Invalid citation keys found.
        - ``unanchored_claims``: Claims not backed by cited evidence.
        - ``missing_evidence_refs``: Bound evidence IDs that should have
          been cited but were not.
        - ``coverage_gaps``: Key points or supporting points not covered.
        - ``feedback_for_retry``: Human-readable feedback to prepend to the
          retry prompt.
    """
    passed: bool = True
    citation_issues: List[str] = Field(default_factory=list)
    unanchored_claims: List[str] = Field(default_factory=list)
    missing_evidence_refs: List[str] = Field(default_factory=list)
    coverage_gaps: List[str] = Field(default_factory=list)
    feedback_for_retry: str = ""


# ---------------------------------------------------------------------------
# ClaimVerifier
# ---------------------------------------------------------------------------

class ClaimVerifier:
    """
    Immediate verifier for paragraph-level generated content.
    - **Description**:
        - Designed for use inside the decomposed generation loop
          (``_generate_section_decomposed`` in metadata_agent).
        - Stateless; all context is passed per call.
    """

    async def verify(
        self,
        generated_text: str,
        paragraph_plan: Any,
        evidence_dag: Optional[Any] = None,
        valid_citation_keys: Optional[Set[str]] = None,
    ) -> VerificationResult:
        """
        Run all verification checks on a generated paragraph.
        - **Args**:
            - ``generated_text`` (str): The LaTeX paragraph output.
            - ``paragraph_plan``: ParagraphPlan with claim_id, bound_evidence_ids,
              key_point, supporting_points.
            - ``evidence_dag``: EvidenceDAG instance (optional).
            - ``valid_citation_keys`` (Set[str]): Allowed citation keys.

        - **Returns**:
            - ``VerificationResult``
        """
        result = VerificationResult()
        valid_keys = valid_citation_keys or set()

        self._check_citations(generated_text, valid_keys, result)
        self._check_evidence_anchoring(generated_text, paragraph_plan, evidence_dag, result)
        self._check_coverage(generated_text, paragraph_plan, result)

        result.passed = (
            not result.citation_issues
            and not result.missing_evidence_refs
            and not result.coverage_gaps
        )

        if not result.passed:
            result.feedback_for_retry = self._build_feedback(result)

        return result

    # ------------------------------------------------------------------
    # Check 1: Citation validity
    # ------------------------------------------------------------------

    @staticmethod
    def _check_citations(
        text: str,
        valid_keys: Set[str],
        result: VerificationResult,
    ) -> None:
        """Verify that every \\cite{} key is in the allowed set."""
        cite_pattern = re.compile(r"\\cite\{([^}]+)\}")
        for m in cite_pattern.finditer(text):
            for key in m.group(1).split(","):
                k = key.strip()
                if k and k not in valid_keys:
                    if k not in result.citation_issues:
                        result.citation_issues.append(k)

    # ------------------------------------------------------------------
    # Check 2: Evidence anchoring
    # ------------------------------------------------------------------

    @staticmethod
    def _check_evidence_anchoring(
        text: str,
        paragraph_plan: Any,
        evidence_dag: Optional[Any],
        result: VerificationResult,
    ) -> None:
        """Ensure bound evidence nodes are actually referenced."""
        bound_ids: List[str] = getattr(paragraph_plan, "bound_evidence_ids", [])
        if not bound_ids:
            return

        text_lower = text.lower()

        cite_pattern = re.compile(r"\\cite\{([^}]+)\}")
        cited_keys: Set[str] = set()
        for m in cite_pattern.finditer(text):
            for key in m.group(1).split(","):
                cited_keys.add(key.strip())

        ref_pattern = re.compile(r"\\ref\{([^}]+)\}")
        ref_labels: Set[str] = set()
        for m in ref_pattern.finditer(text):
            ref_labels.add(m.group(1).strip())

        for eid in bound_ids:
            referenced = False
            if eid in cited_keys or eid in ref_labels:
                referenced = True
            elif evidence_dag:
                try:
                    enode = evidence_dag.evidence_nodes.get(eid)
                    if enode:
                        if enode.source_path and enode.source_path in cited_keys:
                            referenced = True
                        elif enode.content and len(enode.content) > 10:
                            snippet_words = enode.content[:60].lower().split()
                            if any(w in text_lower for w in snippet_words[:3]):
                                referenced = True
                except Exception:
                    pass

            if not referenced:
                result.missing_evidence_refs.append(eid)

    # ------------------------------------------------------------------
    # Check 3: Key-point coverage
    # ------------------------------------------------------------------

    @staticmethod
    def _check_coverage(
        text: str,
        paragraph_plan: Any,
        result: VerificationResult,
    ) -> None:
        """Check that the paragraph covers its planned key point."""
        key_point: str = getattr(paragraph_plan, "key_point", "")
        if not key_point:
            return

        text_lower = text.lower()
        kp_words = [w for w in key_point.lower().split() if len(w) > 3]

        if kp_words:
            overlap = sum(1 for w in kp_words if w in text_lower)
            coverage = overlap / len(kp_words)
            if coverage < 0.3:
                result.coverage_gaps.append(
                    f"Key point '{key_point[:80]}...' has low coverage ({coverage:.0%})"
                )

    # ------------------------------------------------------------------
    # Feedback builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_feedback(result: VerificationResult) -> str:
        """Compose a retry-prompt from the verification failures."""
        parts: List[str] = ["## Verification Feedback — Please Fix These Issues:\n"]

        if result.citation_issues:
            parts.append(
                f"**Invalid citations**: {', '.join(result.citation_issues)}. "
                "Remove or replace these with valid citation keys."
            )
        if result.missing_evidence_refs:
            parts.append(
                f"**Missing evidence references**: Evidence IDs "
                f"{', '.join(result.missing_evidence_refs)} are bound to this "
                "paragraph but not referenced. Include \\cite{{}} or \\ref{{}} "
                "for each."
            )
        if result.coverage_gaps:
            for gap in result.coverage_gaps:
                parts.append(f"**Coverage gap**: {gap}")

        return "\n".join(parts)
