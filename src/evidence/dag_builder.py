"""
DAG Builder
- **Description**:
    - Constructs an EvidenceDAG from three evidence sources:
      code context, literature/research context, and visual assets.
    - Extracts claims from a PaperPlan or raw metadata.
    - Builds edges via heuristic role-matching.
    - Runs greedy bipartite matching to ensure every claim has
      at least one bound evidence node.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..models.evidence_graph import (
    ClaimNode,
    ClaimNodeType,
    DAGEdge,
    EdgeType,
    EvidenceDAG,
    EvidenceNode,
    EvidenceNodeType,
)

logger = logging.getLogger(__name__)

# Mapping from code evidence dominant_role to claim type
_ROLE_TO_CLAIM_TYPE: Dict[str, ClaimNodeType] = {
    "method": ClaimNodeType.METHOD_CLAIM,
    "experiment": ClaimNodeType.METHOD_CLAIM,
    "result": ClaimNodeType.RESULT_CLAIM,
}

# Section-type to relevant dominant_role mapping for edge heuristics
_SECTION_ROLE_AFFINITY: Dict[str, List[str]] = {
    "introduction": ["method", "experiment", "result"],
    "related_work": ["method"],
    "method": ["method"],
    "experiment": ["experiment"],
    "result": ["result"],
    "discussion": ["result", "experiment"],
    "conclusion": ["result", "method"],
    "abstract": ["method", "result"],
}


class DAGBuilder:
    """
    Builds an EvidenceDAG from heterogeneous sources.

    - **Description**:
        - Ingests code evidence, literature evidence, and visual evidence.
        - Extracts claims from PaperPlan paragraphs or metadata fields.
        - Creates heuristic edges based on role/scope affinity.
        - Runs greedy bipartite matching to confirm bindings.

    - **Args**:
        - `code_context` (Dict): Output of CodeContextBuilder.build()
        - `research_context` (Dict): Output of PlannerAgent._generate_research_context()
        - `figures` (List): FigureSpec dicts or objects
        - `tables` (List): TableSpec dicts or objects
        - `paper_plan` (PaperPlan, optional): If available, used for claim extraction
        - `metadata` (Dict, optional): Raw PaperMetaData fields as fallback
    """

    def build(
        self,
        code_context: Optional[Dict[str, Any]] = None,
        research_context: Optional[Dict[str, Any]] = None,
        figures: Optional[List[Any]] = None,
        tables: Optional[List[Any]] = None,
        paper_plan: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        graph_structure: Optional[Any] = None,
    ) -> EvidenceDAG:
        """
        Construct the complete EvidenceDAG.

        - **Args**:
            - `graph_structure` (Optional): CanvasGraphStructure from user's canvas
              for preserving reasoning flow in DAG

        - **Returns**:
            - `EvidenceDAG`: The populated graph with nodes, edges, and bindings.
        """
        dag = EvidenceDAG()

        self._ingest_code_evidence(code_context, dag)
        self._ingest_literature_evidence(research_context, dag)
        self._ingest_visual_evidence(figures or [], tables or [], dag)
        self._ingest_canvas_graph_evidence(graph_structure, dag)
        self._extract_claims(paper_plan, metadata, dag)
        self._build_edges(dag)
        self._run_bipartite_matching(dag)

        logger.info("EvidenceDAG built: %s", dag.summary())
        return dag

    # ------------------------------------------------------------------
    # Evidence ingestion
    # ------------------------------------------------------------------

    def _ingest_code_evidence(
        self,
        code_context: Optional[Dict[str, Any]],
        dag: EvidenceDAG,
    ) -> None:
        """
        Convert code_evidence_graph entries into EvidenceNodes.
        """
        if not code_context:
            return
        graph = code_context.get("code_evidence_graph") or []
        for entry in graph:
            eid = entry.get("evidence_id", "")
            if not eid:
                continue
            dag.add_evidence(EvidenceNode(
                node_id=eid,
                node_type=EvidenceNodeType.CODE,
                content=entry.get("purpose", ""),
                source_path=entry.get("path", ""),
                confidence=float(entry.get("confidence", 0.0)),
                metadata={
                    "symbols": entry.get("symbols", []),
                    "role_tags": entry.get("role_tags", []),
                    "dominant_role": entry.get("dominant_role", ""),
                    "snippet": entry.get("snippet", "")[:400],
                },
            ))

    def _ingest_literature_evidence(
        self,
        research_context: Optional[Dict[str, Any]],
        dag: EvidenceDAG,
    ) -> None:
        """
        Convert claim_evidence_matrix and key_papers into EvidenceNodes.
        """
        if not research_context:
            return

        # Key papers become evidence nodes
        key_papers = research_context.get("key_papers") or []
        for idx, kp in enumerate(key_papers, start=1):
            ref_id = kp.get("ref_id") or kp.get("title", f"lit_{idx}")
            node_id = f"LIT{idx:03d}"
            dag.add_evidence(EvidenceNode(
                node_id=node_id,
                node_type=EvidenceNodeType.LITERATURE,
                content=kp.get("contribution", kp.get("title", "")),
                source_path=ref_id,
                confidence=0.7,
                metadata={
                    "title": kp.get("title", ""),
                    "authors": kp.get("authors", ""),
                    "year": kp.get("year", ""),
                },
            ))

        # claim_evidence_matrix entries may reference refs not yet added
        matrix = research_context.get("claim_evidence_matrix") or []
        existing_paths = {n.source_path for n in dag.evidence_nodes.values()}
        lit_counter = len(key_papers)
        for entry in matrix:
            for ref_key in entry.get("support_refs") or []:
                if ref_key and ref_key not in existing_paths:
                    lit_counter += 1
                    dag.add_evidence(EvidenceNode(
                        node_id=f"LIT{lit_counter:03d}",
                        node_type=EvidenceNodeType.LITERATURE,
                        content=f"Reference: {ref_key}",
                        source_path=ref_key,
                        confidence=0.5,
                    ))
                    existing_paths.add(ref_key)

    def _ingest_visual_evidence(
        self,
        figures: List[Any],
        tables: List[Any],
        dag: EvidenceDAG,
    ) -> None:
        """
        Create EvidenceNodes from figure and table specs.
        """
        for idx, fig in enumerate(figures, start=1):
            fig_id = _get_attr_or_key(fig, "id", f"fig_{idx}")
            caption = _get_attr_or_key(fig, "caption", "")
            desc = _get_attr_or_key(fig, "description", "")
            section = _get_attr_or_key(fig, "section", "")
            dag.add_evidence(EvidenceNode(
                node_id=f"FIG{idx:03d}",
                node_type=EvidenceNodeType.FIGURE,
                content=desc or caption,
                source_path=fig_id,
                confidence=0.9,
                metadata={"caption": caption, "section": section},
            ))

        for idx, tbl in enumerate(tables, start=1):
            tbl_id = _get_attr_or_key(tbl, "id", f"tab_{idx}")
            caption = _get_attr_or_key(tbl, "caption", "")
            desc = _get_attr_or_key(tbl, "description", "")
            section = _get_attr_or_key(tbl, "section", "")
            dag.add_evidence(EvidenceNode(
                node_id=f"TBL{idx:03d}",
                node_type=EvidenceNodeType.TABLE,
                content=desc or caption,
                source_path=tbl_id,
                confidence=0.9,
                metadata={"caption": caption, "section": section},
            ))

    def _ingest_canvas_graph_evidence(
        self,
        graph_structure: Optional[Any],
        dag: EvidenceDAG,
    ) -> None:
        """
        Convert user's canvas graph nodes/edges into DAG evidence nodes and reasoning edges.

        Canvas nodes represent the user's explicit reasoning flow (hypothesis -> method -> result).
        These are added as evidence nodes with high confidence since they represent user input.
        """
        if not graph_structure:
            return

        # Import here to avoid circular imports at module level
        from src.models.evidence_graph import EvidenceNodeType, EdgeType

        nodes = getattr(graph_structure, "nodes", []) or []
        edges = getattr(graph_structure, "edges", []) or []

        if not nodes:
            return

        node_id_map = {}  # canvas_node_id -> dag_node_id

        # Add canvas nodes as evidence nodes
        for cn in nodes:
            node_id = getattr(cn, "node_id", "") or ""
            node_type = getattr(cn, "node_type", "") or ""
            content = getattr(cn, "content", "") or ""
            label = getattr(cn, "label", "") or ""

            ev_node = EvidenceNode(
                node_id=f"CANVAS_{node_id}",
                node_type=EvidenceNodeType.CANVAS,
                content=content,
                source_path=node_id,
                confidence=1.0,  # User-provided content has highest confidence
                metadata={"canvas_node_type": node_type, "label": label},
            )
            dag.add_evidence(ev_node)
            node_id_map[node_id] = ev_node.node_id

        # Add reasoning edges between canvas-derived evidence nodes
        for edge in edges:
            src = getattr(edge, "source_id", "") or ""
            tgt = getattr(edge, "target_id", "") or ""
            edge_type = getattr(edge, "edge_type", "reasoning") or "reasoning"

            if src in node_id_map and tgt in node_id_map:
                dag.add_edge(DAGEdge(
                    source_id=node_id_map[src],
                    target_id=node_id_map[tgt],
                    edge_type=EdgeType.REASONING,
                    weight=0.9,  # High weight - user explicitly connected these
                    reason=f"User reasoning: {edge_type}",
                    is_bound=True,  # Explicit user binding
                ))

    # ------------------------------------------------------------------
    # Claim extraction
    # ------------------------------------------------------------------

    def _extract_claims(
        self,
        paper_plan: Optional[Any],
        metadata: Optional[Dict[str, Any]],
        dag: EvidenceDAG,
    ) -> None:
        """
        Extract ClaimNodes from PaperPlan paragraphs or raw metadata.
        """
        claim_counter = 0

        if paper_plan is not None:
            sections = getattr(paper_plan, "sections", []) or []
            for sp in sections:
                section_type = getattr(sp, "section_type", "")
                paragraphs = getattr(sp, "paragraphs", []) or []
                for pidx, para in enumerate(paragraphs):
                    key_point = getattr(para, "key_point", "")
                    if not key_point:
                        continue
                    claim_counter += 1
                    ctype = self._infer_claim_type(section_type)
                    dag.add_claim(ClaimNode(
                        node_id=f"CLM{claim_counter:03d}",
                        node_type=ctype,
                        statement=key_point,
                        section_scope=[section_type] if section_type else [],
                        metadata={
                            "source": "paper_plan",
                            "section_type": section_type,
                            "paragraph_index": pidx,
                        },
                    ))
        elif metadata:
            # Fallback: derive claims from metadata fields
            field_map = [
                ("idea_hypothesis", ClaimNodeType.HYPOTHESIS, ["introduction"]),
                ("method", ClaimNodeType.METHOD_CLAIM, ["method"]),
                ("data", ClaimNodeType.METHOD_CLAIM, ["experiment"]),
                ("experiments", ClaimNodeType.RESULT_CLAIM, ["result", "experiment"]),
            ]
            for field, ctype, scope in field_map:
                text = metadata.get(field, "")
                if not text:
                    continue
                claim_counter += 1
                dag.add_claim(ClaimNode(
                    node_id=f"CLM{claim_counter:03d}",
                    node_type=ctype,
                    statement=text[:500],
                    section_scope=scope,
                    priority="P0",
                    metadata={"source": "metadata", "field": field},
                ))

        # Also ingest claims from research_context claim_evidence_matrix
        # (these are literature-grounded claims)
        # This is handled during edge building via the matrix.

    @staticmethod
    def _infer_claim_type(section_type: str) -> ClaimNodeType:
        mapping = {
            "introduction": ClaimNodeType.CONTEXT,
            "related_work": ClaimNodeType.CONTEXT,
            "method": ClaimNodeType.METHOD_CLAIM,
            "experiment": ClaimNodeType.METHOD_CLAIM,
            "result": ClaimNodeType.RESULT_CLAIM,
            "discussion": ClaimNodeType.FINDING,
            "conclusion": ClaimNodeType.FINDING,
            "abstract": ClaimNodeType.HYPOTHESIS,
        }
        return mapping.get(section_type, ClaimNodeType.CONTEXT)

    # ------------------------------------------------------------------
    # Edge construction
    # ------------------------------------------------------------------

    def _build_edges(self, dag: EvidenceDAG) -> None:
        """
        Build heuristic edges between evidence and claims based on
        role/scope affinity and metadata cues.
        """
        for claim in dag.claim_nodes.values():
            scopes = claim.section_scope or list(_SECTION_ROLE_AFFINITY.keys())
            relevant_roles: set[str] = set()
            for scope in scopes:
                relevant_roles.update(_SECTION_ROLE_AFFINITY.get(scope, []))

            for ev in dag.evidence_nodes.values():
                weight = self._compute_edge_weight(ev, claim, relevant_roles)
                if weight > 0:
                    dag.add_edge(DAGEdge(
                        source_id=ev.node_id,
                        target_id=claim.node_id,
                        edge_type=EdgeType.SUPPORTS,
                        weight=weight,
                        reason=self._edge_reason(ev, claim),
                    ))

    def _compute_edge_weight(
        self,
        ev: EvidenceNode,
        claim: ClaimNode,
        relevant_roles: set[str],
    ) -> float:
        """
        Score how well an evidence node supports a claim.
        Returns 0 if no plausible connection exists.
        """
        weight = 0.0

        # Code evidence: match dominant_role against section affinity
        if ev.node_type == EvidenceNodeType.CODE:
            dom_role = ev.metadata.get("dominant_role", "")
            if dom_role in relevant_roles:
                weight += 0.4
                weight += ev.confidence * 0.3

        # Literature evidence: always somewhat relevant to scoped claims
        elif ev.node_type == EvidenceNodeType.LITERATURE:
            if claim.section_scope:
                for scope in claim.section_scope:
                    if scope in ("related_work", "introduction", "discussion"):
                        weight += 0.5
                        break
                else:
                    weight += 0.2
            else:
                weight += 0.3

        # Visual evidence: match section hints
        elif ev.node_type in (EvidenceNodeType.FIGURE, EvidenceNodeType.TABLE):
            ev_section = ev.metadata.get("section", "")
            if ev_section and claim.section_scope and ev_section in claim.section_scope:
                weight += 0.7
            elif not ev_section:
                # Figures/tables with result-related claims
                if claim.node_type in (ClaimNodeType.RESULT_CLAIM, ClaimNodeType.FINDING):
                    weight += 0.3

        # Metric evidence aligns with result claims
        elif ev.node_type == EvidenceNodeType.METRIC:
            if claim.node_type in (ClaimNodeType.RESULT_CLAIM, ClaimNodeType.FINDING):
                weight += 0.6

        return min(weight, 1.0)

    @staticmethod
    def _edge_reason(ev: EvidenceNode, claim: ClaimNode) -> str:
        return f"{ev.node_type.value} evidence '{ev.node_id}' supports {claim.node_type.value} claim in {claim.section_scope}"

    # ------------------------------------------------------------------
    # Bipartite matching
    # ------------------------------------------------------------------

    def _run_bipartite_matching(self, dag: EvidenceDAG) -> None:
        """
        Greedy bipartite matching: ensure every claim has at least one
        bound evidence edge. Higher-weight edges are preferred.

        - **Description**:
            - Sorts all edges by weight descending.
            - Greedily binds edges, ensuring each claim gets at least one.
            - Evidence nodes may support multiple claims.
            - Unmatched claims are logged as warnings.
        """
        if not dag.edges:
            return

        # Sort edges by weight descending
        sorted_edges = sorted(
            range(len(dag.edges)),
            key=lambda i: dag.edges[i].weight,
            reverse=True,
        )

        bound_claims: dict[str, int] = {}  # claim_id -> count of bound edges
        max_bindings_per_claim = 5

        # First pass: bind the best edge for each claim
        for idx in sorted_edges:
            edge = dag.edges[idx]
            cid = edge.target_id
            if cid not in bound_claims:
                dag.edges[idx].is_bound = True
                bound_claims[cid] = 1

        # Second pass: allow additional bindings up to the cap
        for idx in sorted_edges:
            edge = dag.edges[idx]
            if edge.is_bound:
                continue
            cid = edge.target_id
            if bound_claims.get(cid, 0) < max_bindings_per_claim and edge.weight >= 0.3:
                dag.edges[idx].is_bound = True
                bound_claims[cid] = bound_claims.get(cid, 0) + 1

        # Report unmatched claims
        all_claim_ids = set(dag.claim_nodes.keys())
        matched = set(bound_claims.keys())
        unmatched = all_claim_ids - matched
        if unmatched:
            logger.warning(
                "Bipartite matching: %d claims have no bound evidence: %s",
                len(unmatched),
                [dag.claim_nodes[c].statement[:60] for c in list(unmatched)[:5]],
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_attr_or_key(obj: Any, name: str, default: Any = "") -> Any:
    """Get attribute or dict key, with fallback."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
