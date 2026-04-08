"""
Evidence Checker (DAG-aware)
- **Description**:
    - Verifies each claim is still anchored to its bound evidence post-revision
    - Cross-validates citations against the Evidence DAG
    - Reports hallucination statistics (grounding rate, drifted claims)
    - Gracefully skips when no DAG is available (fail-open)
"""
import logging
import re
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from .base import FeedbackChecker

if TYPE_CHECKING:
    from ..models import ReviewContext, FeedbackResult

logger = logging.getLogger("uvicorn.error")


class EvidenceChecker(FeedbackChecker):
    """
    DAG-aware evidence grounding checker.
    - **Description**:
        - Verifies each claim is still anchored to its bound evidence post-revision
        - Reports hallucination statistics
        - Fail-open when DAG is unavailable
    """

    @property
    def name(self) -> str:
        return "evidence_check"

    @property
    def priority(self) -> int:
        return 12

    @property
    def enabled(self) -> bool:
        return True

    async def check(self, context: "ReviewContext") -> "FeedbackResult":
        """
        Verify evidence grounding for all claims in the Evidence DAG.

        - **Args**:
            - `context` (ReviewContext): Review context with sections and metadata

        - **Returns**:
            - `FeedbackResult`: Grounding check results with hallucination statistics
        """
        from ..models import FeedbackResult, Severity
        from ....models.evidence_graph import EvidenceDAG

        raw_dag = context.metadata.get("evidence_dag")
        if not raw_dag:
            return FeedbackResult(
                checker_name=self.name,
                passed=True,
                severity=Severity.INFO,
                message="Evidence check skipped: no DAG available.",
            )

        try:
            dag = EvidenceDAG.model_validate(raw_dag)
        except Exception as e:
            logger.warning("evidence_check: DAG deserialization failed: %s", e)
            return FeedbackResult(
                checker_name=self.name,
                passed=True,
                severity=Severity.INFO,
                message=f"Evidence check skipped: DAG deserialization error ({e}).",
            )

        total_claims = 0
        anchored_claims = 0
        drifted_claims_list: List[Dict[str, Any]] = []
        section_feedbacks_list: List[Dict[str, Any]] = []

        for section_type, section_content in context.sections.items():
            claims = dag.get_claims_for_section(section_type)
            for claim in claims:
                total_claims += 1
                bound_evidence_ids = dag.get_bound_evidence_ids_for_claim(claim.node_id)
                if not bound_evidence_ids:
                    anchored_claims += 1
                    continue

                referenced = self._check_evidence_references(
                    section_content, bound_evidence_ids, dag,
                )
                if referenced:
                    anchored_claims += 1
                else:
                    expected = self._resolve_citation_keys(bound_evidence_ids, dag)
                    drifted_claims_list.append({
                        "claim_id": claim.node_id,
                        "claim_text": (claim.statement or "")[:100],
                        "section_type": section_type,
                        "expected_evidence": expected,
                    })
                    section_feedbacks_list.append({
                        "section_type": section_type,
                        "current_word_count": context.word_counts.get(section_type, 0),
                        "target_word_count": (
                            context.get_section_target(section_type)
                            or context.word_counts.get(section_type, 0)
                        ),
                        "action": "refine_paragraphs",
                        "delta_words": 0,
                        "target_paragraphs": [],
                        "paragraph_instructions": {},
                    })

        grounding_rate = (anchored_claims / total_claims) if total_claims > 0 else 1.0
        passed = len(drifted_claims_list) == 0
        severity = Severity.INFO if passed else Severity.WARNING
        if grounding_rate < 0.7:
            severity = Severity.ERROR

        hallucination_stats = {
            "total_claims": total_claims,
            "anchored_claims": anchored_claims,
            "drifted_claims": len(drifted_claims_list),
            "grounding_rate": round(grounding_rate, 4),
        }

        message = (
            f"Evidence grounding: {anchored_claims}/{total_claims} claims anchored "
            f"(rate={grounding_rate:.1%})"
        )

        sections_to_revise: Dict[str, str] = {}
        for dc in drifted_claims_list:
            sec = dc["section_type"]
            if sec not in sections_to_revise:
                sections_to_revise[sec] = (
                    f"Claim '{dc['claim_text'][:50]}...' lost evidence anchoring"
                )

        document_feedbacks = []
        if not passed:
            document_feedbacks.append({
                "level": "document",
                "agent": "reviewer",
                "checker": self.name,
                "target_id": "document",
                "severity": severity.value,
                "issue_type": "evidence_grounding",
                "message": message,
                "suggested_action": "evidence_fix",
            })

        return FeedbackResult(
            checker_name=self.name,
            passed=passed,
            severity=severity,
            message=message,
            suggested_action="evidence_fix" if not passed else None,
            details={
                "hallucination_stats": hallucination_stats,
                "drifted_claims": drifted_claims_list,
                "sections_to_revise": sections_to_revise,
                "section_feedbacks": section_feedbacks_list,
                "document_feedbacks": document_feedbacks,
            },
        )

    @staticmethod
    def _check_evidence_references(
        content: str,
        evidence_ids: List[str],
        dag: Any,
    ) -> bool:
        """
        Check if at least one bound evidence ID is referenced in the content.
        - **Description**:
            - For LITERATURE nodes, checks the actual citation key (source_path)
              in \\cite{} commands rather than the internal node ID (e.g. LIT001).
            - For other node types, searches for evidence IDs in \\ref{} or as
              raw strings, plus the node label if available.
        """
        if not evidence_ids:
            return True
        for eid in evidence_ids:
            ev_node = dag.evidence_nodes.get(eid) if hasattr(dag, "evidence_nodes") else None

            # For LITERATURE nodes, match against the real citation key (source_path)
            if ev_node and hasattr(ev_node, "node_type"):
                from ....models.evidence_graph import EvidenceNodeType
                if ev_node.node_type == EvidenceNodeType.LITERATURE and ev_node.source_path:
                    cite_key = ev_node.source_path
                    cite_pattern = re.compile(
                        r"\\(?:cite|citep|citet|citealp)\{[^}]*"
                        + re.escape(cite_key)
                        + r"[^}]*\}"
                    )
                    if cite_pattern.search(content):
                        return True
                    if cite_key in content:
                        return True
                    continue

            if eid in content:
                return True
            cite_pattern = re.compile(
                r"\\(?:cite|ref|cref|Cref)\{[^}]*" + re.escape(eid) + r"[^}]*\}"
            )
            if cite_pattern.search(content):
                return True
            if ev_node and hasattr(ev_node, "label") and ev_node.label:
                if ev_node.label.lower() in content.lower():
                    return True
        return False

    @staticmethod
    def _resolve_citation_keys(
        evidence_ids: List[str],
        dag: Any,
    ) -> List[str]:
        """
        Convert internal evidence node IDs to user-facing identifiers.
        - **Description**:
            - For LITERATURE nodes, returns the actual citation key (source_path)
              instead of the internal ID (e.g. LIT001) so that downstream prompts
              give the LLM actionable BibTeX keys.
            - For non-LITERATURE nodes (figures, tables, code), keeps the original ID.

        - **Args**:
            - `evidence_ids` (List[str]): Internal evidence node IDs.
            - `dag` (EvidenceDAG): The evidence DAG.

        - **Returns**:
            - `List[str]`: Resolved identifiers (citation keys or original IDs).
        """
        from ....models.evidence_graph import EvidenceNodeType

        resolved: List[str] = []
        evidence_nodes = getattr(dag, "evidence_nodes", {})
        for eid in evidence_ids:
            ev_node = evidence_nodes.get(eid)
            if (
                ev_node
                and getattr(ev_node, "node_type", None) == EvidenceNodeType.LITERATURE
                and ev_node.source_path
            ):
                if ev_node.source_path not in resolved:
                    resolved.append(ev_node.source_path)
            else:
                resolved.append(eid)
        return resolved

    def generate_revision_prompt(
        self,
        section_type: str,
        current_content: str,
        feedback: "FeedbackResult",
    ) -> str:
        """
        Generate a revision prompt to restore evidence anchoring.

        - **Args**:
            - `section_type` (str): Section to revise
            - `current_content` (str): Current LaTeX content
            - `feedback` (FeedbackResult): Evidence check feedback

        - **Returns**:
            - `str`: Revision prompt for the LLM
        """
        drifted = feedback.details.get("drifted_claims", [])
        relevant = [d for d in drifted if d.get("section_type", "").lower() == section_type.lower()]

        if not relevant:
            return ""

        parts = [
            f"The following claims in the {section_type} section have lost their evidence anchoring. "
            "Please restore citations/references to the bound evidence for each claim:\n",
        ]

        for dc in relevant:
            expected = dc.get("expected_evidence", [])
            parts.append(
                f"- Claim: \"{dc.get('claim_text', 'unknown')}\"\n"
                f"  Required citation keys: {expected}\n"
                f"  Ensure this claim references at least one of the listed keys "
                f"via \\cite{{}}."
            )

        parts.append(f"\nCurrent content:\n{current_content}")
        parts.append("\nReturn the revised LaTeX content only.")

        return "\n".join(parts)
