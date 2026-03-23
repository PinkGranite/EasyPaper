"""
Evidence Graph Data Models
- **Description**:
    - Defines the Claim-Evidence DAG used by the Evidence Orchestration Layer.
    - EvidenceNode: a piece of verifiable evidence (code, literature, figure, etc.)
    - ClaimNode: a statement/assertion that requires evidence support
    - DAGEdge: directed relationship from evidence to claim
    - EvidenceDAG: the complete graph with query helpers
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Node and edge type enums
# ---------------------------------------------------------------------------

class EvidenceNodeType(str, Enum):
    """Source category for an evidence node."""
    CODE = "code"
    LITERATURE = "literature"
    FIGURE = "figure"
    TABLE = "table"
    METRIC = "metric"


class ClaimNodeType(str, Enum):
    """Semantic category for a claim node."""
    HYPOTHESIS = "hypothesis"
    METHOD_CLAIM = "method_claim"
    RESULT_CLAIM = "result_claim"
    FINDING = "finding"
    CONTEXT = "context"


class EdgeType(str, Enum):
    """Relationship between an evidence node and a claim node."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXTUALIZES = "contextualizes"
    DERIVED_FROM = "derived_from"


# ---------------------------------------------------------------------------
# Node models
# ---------------------------------------------------------------------------

class EvidenceNode(BaseModel):
    """
    A verifiable piece of evidence.

    - **Description**:
        - Represents a single evidence artifact from code, literature,
          figures, tables, or experimental metrics.

    - **Args**:
        - `node_id` (str): Unique identifier (e.g. ``EV001``, ``LIT003``).
        - `node_type` (EvidenceNodeType): Source category.
        - `content` (str): Descriptive summary of this evidence.
        - `source_path` (str): File path, citation key, or figure ID.
        - `confidence` (float): Estimated reliability in [0, 1].
        - `metadata` (Dict): Arbitrary extra fields (snippet, symbols, …).
    """
    node_id: str
    node_type: EvidenceNodeType
    content: str
    source_path: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClaimNode(BaseModel):
    """
    A statement or assertion that should be backed by evidence.

    - **Description**:
        - Represents a scientific claim made in the paper.
        - ``section_scope`` limits which sections may use this claim.

    - **Args**:
        - `node_id` (str): Unique identifier (e.g. ``CLM001``).
        - `node_type` (ClaimNodeType): Semantic category.
        - `statement` (str): Natural-language claim text.
        - `section_scope` (List[str]): Section types where this claim applies.
        - `priority` (str): Importance tag (``P0`` / ``P1`` / ``P2``).
        - `metadata` (Dict): Arbitrary extra fields.
    """
    node_id: str
    node_type: ClaimNodeType
    statement: str
    section_scope: List[str] = Field(default_factory=list)
    priority: str = "P1"
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Edge model
# ---------------------------------------------------------------------------

class DAGEdge(BaseModel):
    """
    Directed edge from an evidence node to a claim node.

    - **Description**:
        - ``source_id`` references an EvidenceNode, ``target_id`` a ClaimNode.
        - ``weight`` encodes matching strength (higher = stronger support).
        - ``is_bound`` is set to True after bipartite matching confirms this
          edge as a primary binding.

    - **Args**:
        - `source_id` (str): EvidenceNode.node_id
        - `target_id` (str): ClaimNode.node_id
        - `edge_type` (EdgeType): Relationship kind.
        - `weight` (float): Matching strength in [0, 1].
        - `reason` (str): Why this edge exists.
        - `is_bound` (bool): Whether this edge is a confirmed binding.
    """
    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.SUPPORTS
    weight: float = 1.0
    reason: str = ""
    is_bound: bool = False


# ---------------------------------------------------------------------------
# DAG container
# ---------------------------------------------------------------------------

class EvidenceDAG(BaseModel):
    """
    Complete Claim-Evidence directed acyclic graph.

    - **Description**:
        - Central data structure for the Evidence Orchestration Layer.
        - Stores evidence nodes, claim nodes, and directed edges between them.
        - Provides query helpers used by Planner and Writer stages.
    """
    evidence_nodes: Dict[str, EvidenceNode] = Field(default_factory=dict)
    claim_nodes: Dict[str, ClaimNode] = Field(default_factory=dict)
    edges: List[DAGEdge] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_evidence(self, node: EvidenceNode) -> None:
        self.evidence_nodes[node.node_id] = node

    def add_claim(self, node: ClaimNode) -> None:
        self.claim_nodes[node.node_id] = node

    def add_edge(self, edge: DAGEdge) -> None:
        self.edges.append(edge)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_evidence_for_claim(self, claim_id: str) -> List[EvidenceNode]:
        """
        Return all evidence nodes connected to a claim via bound edges.
        Falls back to any edge if no bound edges exist.
        """
        bound = [
            e.source_id for e in self.edges
            if e.target_id == claim_id and e.is_bound
        ]
        if not bound:
            bound = [
                e.source_id for e in self.edges
                if e.target_id == claim_id
            ]
        return [
            self.evidence_nodes[eid]
            for eid in bound
            if eid in self.evidence_nodes
        ]

    def get_claims_for_section(self, section_type: str) -> List[ClaimNode]:
        """Return claims whose scope includes *section_type* (or is empty = global)."""
        return [
            c for c in self.claim_nodes.values()
            if not c.section_scope or section_type in c.section_scope
        ]

    def get_unsupported_claims(self) -> List[ClaimNode]:
        """Return claims with zero bound edges."""
        supported: Set[str] = set()
        for e in self.edges:
            if e.is_bound:
                supported.add(e.target_id)
        if not supported:
            for e in self.edges:
                supported.add(e.target_id)
        return [
            c for c in self.claim_nodes.values()
            if c.node_id not in supported
        ]

    def get_bound_evidence_ids_for_claim(self, claim_id: str) -> List[str]:
        """Return source_ids of bound edges targeting *claim_id*."""
        bound = [
            e.source_id for e in self.edges
            if e.target_id == claim_id and e.is_bound
        ]
        if not bound:
            bound = [
                e.source_id for e in self.edges
                if e.target_id == claim_id
            ]
        return bound

    def get_citation_refs_for_claim(self, claim_id: str) -> List[str]:
        """
        Return literature citation keys (source_path) for evidence nodes
        of type LITERATURE that are bound to *claim_id*.
        """
        refs: List[str] = []
        for ev in self.get_evidence_for_claim(claim_id):
            if ev.node_type == EvidenceNodeType.LITERATURE and ev.source_path:
                if ev.source_path not in refs:
                    refs.append(ev.source_path)
        return refs

    def get_section_evidence_ids(self, section_type: str) -> List[str]:
        """
        Collect unique evidence IDs bound to any claim in *section_type*.
        """
        ids: List[str] = []
        seen: Set[str] = set()
        for claim in self.get_claims_for_section(section_type):
            for eid in self.get_bound_evidence_ids_for_claim(claim.node_id):
                if eid not in seen:
                    ids.append(eid)
                    seen.add(eid)
        return ids

    def summary(self) -> Dict[str, Any]:
        """Compact diagnostic summary."""
        bound_edges = sum(1 for e in self.edges if e.is_bound)
        return {
            "evidence_count": len(self.evidence_nodes),
            "claim_count": len(self.claim_nodes),
            "edge_count": len(self.edges),
            "bound_edge_count": bound_edges,
            "unsupported_claims": len(self.get_unsupported_claims()),
        }

    def to_serializable(self) -> Dict[str, Any]:
        """Convert to a JSON-safe dictionary for storage in PaperPlan."""
        return {
            "evidence_nodes": {
                k: v.model_dump() for k, v in self.evidence_nodes.items()
            },
            "claim_nodes": {
                k: v.model_dump() for k, v in self.claim_nodes.items()
            },
            "edges": [e.model_dump() for e in self.edges],
        }

    @classmethod
    def from_serializable(cls, data: Optional[Dict[str, Any]]) -> Optional["EvidenceDAG"]:
        """Reconstruct from a dictionary produced by ``to_serializable``."""
        if not data:
            return None
        dag = cls()
        for nid, ndata in (data.get("evidence_nodes") or {}).items():
            dag.evidence_nodes[nid] = EvidenceNode(**ndata)
        for nid, ndata in (data.get("claim_nodes") or {}).items():
            dag.claim_nodes[nid] = ClaimNode(**ndata)
        for edata in data.get("edges") or []:
            dag.edges.append(DAGEdge(**edata))
        return dag
