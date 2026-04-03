"""
Tests for Canvas-to-DAG Translation Layer.

Validates the semantic mapping from user's Research Graph (canvas) node types
to Evidence DAG entities (ClaimNode / EvidenceNode / skip).
"""
import pytest
from src.models.evidence_graph import (
    ClaimNode,
    ClaimNodeType,
    DAGEdge,
    EdgeType,
    EvidenceDAG,
    EvidenceNode,
    EvidenceNodeType,
)
from src.agents.commander_agent.models import (
    CanvasGraphNode,
    CanvasGraphEdge,
    CanvasGraphStructure,
)
from src.evidence.dag_builder import DAGBuilder


# ============================================================================
# Helpers
# ============================================================================


def _make_canvas_node(
    node_id: str,
    node_type: str,
    content: str = "test content",
    label: str = "",
    **extra_metadata,
) -> CanvasGraphNode:
    return CanvasGraphNode(
        node_id=node_id,
        node_type=node_type,
        label=label or f"{node_type}_label",
        content=content,
        metadata=extra_metadata,
    )


def _make_canvas_edge(
    edge_id: str,
    source_id: str,
    target_id: str,
    edge_type: str = "reasoning",
) -> CanvasGraphEdge:
    return CanvasGraphEdge(
        edge_id=edge_id,
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
    )


def _make_graph(*nodes, edges=None) -> CanvasGraphStructure:
    return CanvasGraphStructure(
        nodes=list(nodes),
        edges=edges or [],
    )


# ============================================================================
# 1. Node Classification Tests
# ============================================================================


class TestNodeClassification:
    """Verify canvas node types map to correct DAG roles (claim / evidence / skip)."""

    # --- Claim-role nodes ---

    def test_hypothesis_becomes_claim(self):
        """hypothesis canvas node should produce a ClaimNode with HYPOTHESIS type."""
        gs = _make_graph(
            _make_canvas_node("h1", "hypothesis", content="Neural networks can generalise"),
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert "CANVAS_CLM_h1" in dag.claim_nodes
        claim = dag.claim_nodes["CANVAS_CLM_h1"]
        assert claim.node_type == ClaimNodeType.HYPOTHESIS
        assert "Neural networks" in claim.statement
        assert "CANVAS_CLM_h1" not in dag.evidence_nodes

    def test_idea_becomes_claim(self):
        """idea canvas node should produce a ClaimNode with HYPOTHESIS type."""
        gs = _make_graph(
            _make_canvas_node("i1", "idea", content="A new approach to RL"),
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert "CANVAS_CLM_i1" in dag.claim_nodes
        claim = dag.claim_nodes["CANVAS_CLM_i1"]
        assert claim.node_type == ClaimNodeType.HYPOTHESIS

    def test_question_becomes_claim(self):
        """question canvas node should produce a ClaimNode with CONTEXT type."""
        gs = _make_graph(
            _make_canvas_node("q1", "question", content="What is the optimal learning rate?"),
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert "CANVAS_CLM_q1" in dag.claim_nodes
        claim = dag.claim_nodes["CANVAS_CLM_q1"]
        assert claim.node_type == ClaimNodeType.CONTEXT

    # --- Evidence-role nodes ---

    def test_result_becomes_evidence(self):
        """result canvas node should produce an EvidenceNode with METRIC type."""
        gs = _make_graph(
            _make_canvas_node("r1", "result", content="Accuracy 95%"),
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert "CANVAS_EV_r1" in dag.evidence_nodes
        ev = dag.evidence_nodes["CANVAS_EV_r1"]
        assert ev.node_type == EvidenceNodeType.METRIC
        assert "CANVAS_EV_r1" not in dag.claim_nodes

    def test_finding_becomes_evidence(self):
        """finding canvas node should produce an EvidenceNode with METRIC type."""
        gs = _make_graph(
            _make_canvas_node("f1", "finding", content="The model outperforms baseline"),
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert "CANVAS_EV_f1" in dag.evidence_nodes

    def test_literature_becomes_evidence(self):
        """literature canvas node should produce an EvidenceNode with LITERATURE type."""
        gs = _make_graph(
            _make_canvas_node("l1", "literature", content="Smith et al. 2024"),
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert "CANVAS_EV_l1" in dag.evidence_nodes
        ev = dag.evidence_nodes["CANVAS_EV_l1"]
        assert ev.node_type == EvidenceNodeType.LITERATURE

    def test_figure_becomes_evidence(self):
        """figure canvas node should produce an EvidenceNode with FIGURE type."""
        gs = _make_graph(
            _make_canvas_node("fig1", "figure", content="Architecture diagram"),
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert "CANVAS_EV_fig1" in dag.evidence_nodes
        ev = dag.evidence_nodes["CANVAS_EV_fig1"]
        assert ev.node_type == EvidenceNodeType.FIGURE

    def test_table_becomes_evidence(self):
        """table canvas node should produce an EvidenceNode with TABLE type."""
        gs = _make_graph(
            _make_canvas_node("tab1", "table", content="Results table"),
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert "CANVAS_EV_tab1" in dag.evidence_nodes
        ev = dag.evidence_nodes["CANVAS_EV_tab1"]
        assert ev.node_type == EvidenceNodeType.TABLE

    def test_data_becomes_evidence(self):
        """data canvas node should produce an EvidenceNode with CODE type."""
        gs = _make_graph(
            _make_canvas_node("d1", "data", content="ImageNet dataset"),
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert "CANVAS_EV_d1" in dag.evidence_nodes

    def test_metric_becomes_evidence(self):
        """metric canvas node should produce an EvidenceNode with METRIC type."""
        gs = _make_graph(
            _make_canvas_node("m1", "metric", content="F1 Score"),
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert "CANVAS_EV_m1" in dag.evidence_nodes
        ev = dag.evidence_nodes["CANVAS_EV_m1"]
        assert ev.node_type == EvidenceNodeType.METRIC

    # --- Mixed-role nodes (both claim + evidence) ---

    def test_method_produces_both_claim_and_evidence(self):
        """method canvas node should produce both a ClaimNode and an EvidenceNode."""
        gs = _make_graph(
            _make_canvas_node("m1", "method", content="Transformer-based approach"),
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert "CANVAS_CLM_m1" in dag.claim_nodes
        assert "CANVAS_EV_m1" in dag.evidence_nodes
        claim = dag.claim_nodes["CANVAS_CLM_m1"]
        assert claim.node_type == ClaimNodeType.METHOD_CLAIM

    def test_experiment_produces_both_claim_and_evidence(self):
        """experiment canvas node should produce both a ClaimNode and an EvidenceNode."""
        gs = _make_graph(
            _make_canvas_node("e1", "experiment", content="Ablation study"),
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert "CANVAS_CLM_e1" in dag.claim_nodes
        assert "CANVAS_EV_e1" in dag.evidence_nodes

    # --- Skipped nodes ---

    def test_start_is_skipped(self):
        """start node should not produce any DAG entity."""
        gs = _make_graph(_make_canvas_node("s1", "start"))
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert len(dag.claim_nodes) == 0
        assert len(dag.evidence_nodes) == 0

    def test_end_is_skipped(self):
        """end node should not produce any DAG entity."""
        gs = _make_graph(_make_canvas_node("e1", "end"))
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert len(dag.claim_nodes) == 0
        assert len(dag.evidence_nodes) == 0

    def test_note_is_skipped(self):
        """note node should not produce any DAG entity."""
        gs = _make_graph(_make_canvas_node("n1", "note"))
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert len(dag.claim_nodes) == 0
        assert len(dag.evidence_nodes) == 0

    def test_comment_is_skipped(self):
        """comment node should not produce any DAG entity."""
        gs = _make_graph(_make_canvas_node("c1", "comment"))
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert len(dag.claim_nodes) == 0
        assert len(dag.evidence_nodes) == 0

    def test_paper_section_is_skipped(self):
        """paper_section node should not produce any DAG entity."""
        gs = _make_graph(_make_canvas_node("ps1", "paper_section"))
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert len(dag.claim_nodes) == 0
        assert len(dag.evidence_nodes) == 0

    def test_concept_is_skipped(self):
        """concept node should not produce any DAG entity."""
        gs = _make_graph(_make_canvas_node("k1", "concept"))
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert len(dag.claim_nodes) == 0
        assert len(dag.evidence_nodes) == 0

    def test_unknown_type_is_skipped(self):
        """unknown / unrecognised node type should be skipped."""
        gs = _make_graph(_make_canvas_node("x1", "unknown_type"))
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert len(dag.claim_nodes) == 0
        assert len(dag.evidence_nodes) == 0

    def test_empty_content_node_is_skipped(self):
        """node with empty content should be skipped even if type is valid."""
        gs = _make_graph(_make_canvas_node("h1", "hypothesis", content=""))
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert len(dag.claim_nodes) == 0

    # --- Metadata preservation ---

    def test_canvas_origin_metadata_preserved(self):
        """translated nodes should carry canvas_node_type and label in metadata."""
        gs = _make_graph(
            _make_canvas_node("h1", "hypothesis", content="Test", label="My Hypo"),
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        claim = dag.claim_nodes["CANVAS_CLM_h1"]
        assert claim.metadata.get("canvas_node_type") == "hypothesis"
        assert claim.metadata.get("label") == "My Hypo"
        assert claim.metadata.get("source") == "canvas"

    def test_none_graph_structure_is_noop(self):
        """passing None graph_structure should do nothing."""
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(None, dag)

        assert len(dag.claim_nodes) == 0
        assert len(dag.evidence_nodes) == 0


# ============================================================================
# 2. Edge Translation Tests
# ============================================================================


class TestEdgeTranslation:
    """Verify canvas edges become correct DAG edge candidates."""

    def test_evidence_to_claim_edge_becomes_supports(self):
        """Edge from an evidence-role node to a claim-role node creates a SUPPORTS candidate."""
        gs = _make_graph(
            _make_canvas_node("r1", "result", content="Accuracy 95%"),
            _make_canvas_node("h1", "hypothesis", content="Our method works"),
            edges=[_make_canvas_edge("e1", "r1", "h1")],
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)
        builder._translate_canvas_edges(gs, dag)

        supports_edges = [
            e for e in dag.edges
            if e.edge_type == EdgeType.SUPPORTS
        ]
        assert len(supports_edges) == 1
        edge = supports_edges[0]
        assert edge.source_id == "CANVAS_EV_r1"
        assert edge.target_id == "CANVAS_CLM_h1"
        assert edge.is_bound is False  # bipartite matching decides binding

    def test_claim_to_claim_edge_is_skipped(self):
        """Edge between two claim-role nodes should not create a SUPPORTS edge."""
        gs = _make_graph(
            _make_canvas_node("h1", "hypothesis", content="Hypo 1"),
            _make_canvas_node("h2", "hypothesis", content="Hypo 2"),
            edges=[_make_canvas_edge("e1", "h1", "h2")],
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)
        builder._translate_canvas_edges(gs, dag)

        assert len(dag.edges) == 0

    def test_evidence_to_evidence_edge_is_skipped(self):
        """Edge between two evidence-role nodes should not create a SUPPORTS edge."""
        gs = _make_graph(
            _make_canvas_node("r1", "result", content="Result 1"),
            _make_canvas_node("r2", "result", content="Result 2"),
            edges=[_make_canvas_edge("e1", "r1", "r2")],
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)
        builder._translate_canvas_edges(gs, dag)

        assert len(dag.edges) == 0

    def test_edge_referencing_skipped_node_is_ignored(self):
        """Edge referencing a skipped node (start/end/note) should be ignored."""
        gs = _make_graph(
            _make_canvas_node("s1", "start", content="start"),
            _make_canvas_node("h1", "hypothesis", content="Hypo"),
            edges=[_make_canvas_edge("e1", "s1", "h1")],
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)
        builder._translate_canvas_edges(gs, dag)

        assert len(dag.edges) == 0

    def test_mixed_node_evidence_side_creates_supports_edge(self):
        """method node produces both claim and evidence; edge from result to method
        should create a SUPPORTS edge targeting the claim side."""
        gs = _make_graph(
            _make_canvas_node("r1", "result", content="Accuracy 95%"),
            _make_canvas_node("m1", "method", content="Transformer approach"),
            edges=[_make_canvas_edge("e1", "r1", "m1")],
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)
        builder._translate_canvas_edges(gs, dag)

        supports_edges = [e for e in dag.edges if e.edge_type == EdgeType.SUPPORTS]
        assert len(supports_edges) == 1
        edge = supports_edges[0]
        assert edge.source_id == "CANVAS_EV_r1"
        assert edge.target_id == "CANVAS_CLM_m1"

    def test_canvas_edge_weight_is_high_but_not_bound(self):
        """Canvas-derived edges should have high weight (user intent) but is_bound=False."""
        gs = _make_graph(
            _make_canvas_node("f1", "finding", content="Finding X"),
            _make_canvas_node("h1", "hypothesis", content="Hypothesis Y"),
            edges=[_make_canvas_edge("e1", "f1", "h1")],
        )
        dag = EvidenceDAG()
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)
        builder._translate_canvas_edges(gs, dag)

        edge = dag.edges[0]
        assert edge.weight >= 0.8
        assert edge.is_bound is False


# ============================================================================
# 3. Deduplication Tests
# ============================================================================


class TestDeduplication:
    """Verify canvas evidence nodes are deduplicated against existing DAG nodes."""

    def test_literature_dedup_by_source_path(self):
        """Canvas literature node matching existing DAG literature by source_path
        should not create a duplicate."""
        dag = EvidenceDAG()
        dag.add_evidence(EvidenceNode(
            node_id="LIT001",
            node_type=EvidenceNodeType.LITERATURE,
            content="Smith et al. 2024 contribution",
            source_path="smith2024",
            confidence=0.7,
        ))

        gs = _make_graph(
            _make_canvas_node("l1", "literature", content="Smith et al. 2024", label="smith2024"),
        )
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        lit_nodes = [
            n for n in dag.evidence_nodes.values()
            if n.node_type == EvidenceNodeType.LITERATURE
        ]
        assert len(lit_nodes) == 1
        assert lit_nodes[0].node_id == "LIT001"

    def test_figure_dedup_by_source_path(self):
        """Canvas figure node matching existing DAG figure should not duplicate."""
        dag = EvidenceDAG()
        dag.add_evidence(EvidenceNode(
            node_id="FIG001",
            node_type=EvidenceNodeType.FIGURE,
            content="Architecture diagram",
            source_path="fig_arch",
            confidence=0.9,
        ))

        gs = _make_graph(
            _make_canvas_node("fig_arch", "figure", content="Architecture diagram"),
        )
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        fig_nodes = [
            n for n in dag.evidence_nodes.values()
            if n.node_type == EvidenceNodeType.FIGURE
        ]
        assert len(fig_nodes) == 1

    def test_table_dedup_by_source_path(self):
        """Canvas table node matching existing DAG table should not duplicate."""
        dag = EvidenceDAG()
        dag.add_evidence(EvidenceNode(
            node_id="TBL001",
            node_type=EvidenceNodeType.TABLE,
            content="Results table",
            source_path="tab_results",
            confidence=0.9,
        ))

        gs = _make_graph(
            _make_canvas_node("tab_results", "table", content="Results table"),
        )
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        tbl_nodes = [
            n for n in dag.evidence_nodes.values()
            if n.node_type == EvidenceNodeType.TABLE
        ]
        assert len(tbl_nodes) == 1

    def test_no_dedup_for_different_types(self):
        """Nodes of different types with same source_path should not be deduped."""
        dag = EvidenceDAG()
        dag.add_evidence(EvidenceNode(
            node_id="FIG001",
            node_type=EvidenceNodeType.FIGURE,
            content="Figure X",
            source_path="shared_id",
        ))

        gs = _make_graph(
            _make_canvas_node("shared_id", "table", content="Table X"),
        )
        builder = DAGBuilder()
        builder._translate_canvas_nodes(gs, dag)

        assert len(dag.evidence_nodes) == 2


# ============================================================================
# 4. Integration: Full build() pipeline
# ============================================================================


class TestBuildIntegration:
    """Verify the full build() pipeline with canvas translation replacing old ingestion."""

    @pytest.mark.asyncio
    async def test_build_with_canvas_uses_new_translation(self):
        """build() should use _translate_canvas_nodes + _translate_canvas_edges
        instead of old _ingest_canvas_graph_evidence."""
        gs = CanvasGraphStructure(
            nodes=[
                CanvasGraphNode(node_id="h1", node_type="hypothesis",
                                label="Hypo", content="Our main claim"),
                CanvasGraphNode(node_id="r1", node_type="result",
                                label="Result", content="We achieved 95%"),
            ],
            edges=[
                CanvasGraphEdge(edge_id="e1", source_id="r1", target_id="h1"),
            ],
        )

        builder = DAGBuilder()
        dag = await builder.build(graph_structure=gs)

        assert "CANVAS_CLM_h1" in dag.claim_nodes
        assert "CANVAS_EV_r1" in dag.evidence_nodes

        # The old method would have created CANVAS_h1 and CANVAS_r1 as evidence only
        assert "CANVAS_h1" not in dag.evidence_nodes
        assert "CANVAS_r1" not in dag.evidence_nodes

    @pytest.mark.asyncio
    async def test_build_without_canvas_works(self):
        """build() without canvas should work as before."""
        builder = DAGBuilder()
        dag = await builder.build()

        assert len(dag.claim_nodes) == 0
        assert len(dag.evidence_nodes) == 0

    @pytest.mark.asyncio
    async def test_canvas_edges_participate_in_bipartite_matching(self):
        """Canvas SUPPORTS edges should be picked up by bipartite matching."""
        gs = CanvasGraphStructure(
            nodes=[
                CanvasGraphNode(node_id="h1", node_type="hypothesis",
                                label="Hypo", content="Main hypothesis"),
                CanvasGraphNode(node_id="r1", node_type="result",
                                label="Result", content="Experimental result"),
            ],
            edges=[
                CanvasGraphEdge(edge_id="e1", source_id="r1", target_id="h1"),
            ],
        )

        builder = DAGBuilder()
        dag = await builder.build(graph_structure=gs)

        bound_edges = [e for e in dag.edges if e.is_bound]
        assert len(bound_edges) >= 1
        bound_edge = next(
            (e for e in bound_edges
             if e.source_id == "CANVAS_EV_r1" and e.target_id == "CANVAS_CLM_h1"),
            None,
        )
        assert bound_edge is not None

    @pytest.mark.asyncio
    async def test_canvas_does_not_pollute_is_bound(self):
        """Canvas edges should never directly set is_bound=True before bipartite matching."""
        gs = CanvasGraphStructure(
            nodes=[
                CanvasGraphNode(node_id="h1", node_type="hypothesis",
                                label="Hypo", content="Test"),
                CanvasGraphNode(node_id="r1", node_type="result",
                                label="Result", content="Test result"),
            ],
            edges=[
                CanvasGraphEdge(edge_id="e1", source_id="r1", target_id="h1"),
            ],
        )

        builder = DAGBuilder()
        # Manually run the translation steps without bipartite matching
        dag = EvidenceDAG()
        builder._translate_canvas_nodes(gs, dag)
        builder._translate_canvas_edges(gs, dag)

        # Before bipartite matching, no edge should be bound
        for edge in dag.edges:
            assert edge.is_bound is False, (
                f"Edge {edge.source_id}->{edge.target_id} should not be bound "
                "before bipartite matching"
            )
