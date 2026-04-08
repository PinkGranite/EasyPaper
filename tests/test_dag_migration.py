"""
End-to-end tests for the DAG migration (academic fd03af0 → commercial).

Verifies:
  1. Core data models (EvidenceDAG, ActionSpace, DocumentSpec)
  2. Evidence layer (DAGBuilder, ContextPerception)
  3. Generation layer (ClaimVerifier, TemplateSlots)
  4. PromptLoader + prompt_compiler integration
  5. Model extensions (to_unified_action, to_document_input, to_document_spec)
  6. Agent-level integrations (PlannerAgent._generate_sentence_plans)
  7. MetaDataAgent import & orchestrator wiring
  8. Full round-trip: DAG → Plan → Prompt → Verify
"""
import json
import pytest
from typing import Dict, Any, List, Optional


# =========================================================================
# 1. Core Data Models
# =========================================================================


class TestEvidenceDAG:
    """Verify EvidenceDAG CRUD, queries, and serialization."""

    def _build_dag(self):
        from src.models.evidence_graph import (
            EvidenceDAG, EvidenceNode, ClaimNode, DAGEdge,
            EvidenceNodeType, ClaimNodeType, EdgeType,
        )
        dag = EvidenceDAG()

        ev1 = EvidenceNode(
            node_id="EV001", node_type=EvidenceNodeType.CODE,
            content="BERT classifier implementation", source_path="code/classify.py",
            confidence=0.9,
        )
        ev2 = EvidenceNode(
            node_id="EV002", node_type=EvidenceNodeType.LITERATURE,
            content="Devlin et al. 2019", source_path="devlin2019bert",
            confidence=0.95,
        )
        ev3 = EvidenceNode(
            node_id="EV003", node_type=EvidenceNodeType.FIGURE,
            content="Architecture diagram", source_path="fig:arch",
            confidence=0.8,
        )
        dag.add_evidence(ev1)
        dag.add_evidence(ev2)
        dag.add_evidence(ev3)

        clm1 = ClaimNode(
            node_id="CLM001", node_type=ClaimNodeType.METHOD_CLAIM,
            statement="We use a fine-tuned BERT model for classification.",
            section_scope=["method", "introduction"],
        )
        clm2 = ClaimNode(
            node_id="CLM002", node_type=ClaimNodeType.RESULT_CLAIM,
            statement="Our approach outperforms baselines by 5%.",
            section_scope=["result"],
        )
        dag.add_claim(clm1)
        dag.add_claim(clm2)

        dag.add_edge(DAGEdge(
            source_id="EV001", target_id="CLM001",
            edge_type=EdgeType.SUPPORTS, weight=0.9, is_bound=True,
        ))
        dag.add_edge(DAGEdge(
            source_id="EV002", target_id="CLM001",
            edge_type=EdgeType.SUPPORTS, weight=0.85, is_bound=True,
        ))
        dag.add_edge(DAGEdge(
            source_id="EV003", target_id="CLM001",
            edge_type=EdgeType.CONTEXTUALIZES, weight=0.6, is_bound=False,
        ))
        return dag

    def test_add_and_count(self):
        dag = self._build_dag()
        assert len(dag.evidence_nodes) == 3
        assert len(dag.claim_nodes) == 2
        assert len(dag.edges) == 3

    def test_get_evidence_for_claim_bound(self):
        dag = self._build_dag()
        evidence = dag.get_evidence_for_claim("CLM001")
        assert len(evidence) == 2, "Should return 2 bound evidence nodes"
        ids = {ev.node_id for ev in evidence}
        assert "EV001" in ids
        assert "EV002" in ids

    def test_get_claims_for_section(self):
        dag = self._build_dag()
        method_claims = dag.get_claims_for_section("method")
        assert len(method_claims) == 1
        assert method_claims[0].node_id == "CLM001"

        result_claims = dag.get_claims_for_section("result")
        assert len(result_claims) == 1
        assert result_claims[0].node_id == "CLM002"

    def test_get_unsupported_claims(self):
        dag = self._build_dag()
        unsupported = dag.get_unsupported_claims()
        assert len(unsupported) == 1, "CLM002 has no edges"
        assert unsupported[0].node_id == "CLM002"

    def test_get_citation_refs_for_claim(self):
        dag = self._build_dag()
        refs = dag.get_citation_refs_for_claim("CLM001")
        assert refs == ["devlin2019bert"]

    def test_get_section_evidence_ids(self):
        dag = self._build_dag()
        ids = dag.get_section_evidence_ids("method")
        assert "EV001" in ids
        assert "EV002" in ids

    def test_summary(self):
        dag = self._build_dag()
        s = dag.summary()
        assert s["evidence_count"] == 3
        assert s["claim_count"] == 2
        assert s["edge_count"] == 3
        assert s["bound_edge_count"] == 2
        assert s["unsupported_claims"] == 1

    def test_serialization_roundtrip(self):
        dag = self._build_dag()
        data = dag.to_serializable()
        assert isinstance(data, dict)
        assert "evidence_nodes" in data
        assert "claim_nodes" in data
        assert "edges" in data

        from src.models.evidence_graph import EvidenceDAG
        restored = EvidenceDAG.from_serializable(data)
        assert restored is not None
        assert len(restored.evidence_nodes) == 3
        assert len(restored.claim_nodes) == 2
        assert len(restored.edges) == 3
        assert restored.summary() == dag.summary()

    def test_from_serializable_none(self):
        from src.models.evidence_graph import EvidenceDAG
        assert EvidenceDAG.from_serializable(None) is None
        assert EvidenceDAG.from_serializable({}) is None


class TestActionSpace:
    """Verify Action model, from_legacy_action, and property helpers."""

    def test_from_legacy_expand(self):
        from src.models.action_space import from_legacy_action, TextAction
        action = from_legacy_action("expand", section="method", estimated_impact=200.0)
        assert action.action_type == "expand"
        assert action.is_text_action
        assert not action.is_layout_action
        assert action.section == "method"
        assert action.estimated_impact == 200.0
        assert "word_count < target_words" in action.preconditions
        assert action.priority == 20

    def test_from_legacy_resize_figure(self):
        from src.models.action_space import from_legacy_action
        action = from_legacy_action(
            "resize_figure", target_id="fig:arch",
            params={"width": "0.8\\linewidth"},
        )
        assert action.is_layout_action
        assert not action.is_text_action
        assert action.target_id == "fig:arch"
        assert action.params["width"] == "0.8\\linewidth"
        assert action.priority == 90

    def test_legacy_alias_ok_to_keep(self):
        from src.models.action_space import from_legacy_action
        action = from_legacy_action("ok")
        assert action.action_type == "keep"
        assert action.priority == 0


class TestDocumentSpec:
    """Verify DocumentSpec and related models."""

    def test_document_spec_creation(self):
        from src.models.document_spec import DocumentSpec, ContentSection
        sec = ContentSection(
            section_id="method", title="Method",
            paragraphs=[{"key_point": "test"}], order=1,
        )
        spec = DocumentSpec(
            title="Test Paper", document_type="paper",
            sections=[sec], contributions=["Contribution 1"],
        )
        assert spec.title == "Test Paper"
        assert len(spec.sections) == 1
        assert spec.sections[0].section_id == "method"

    def test_document_input_creation(self):
        from src.models.document_spec import DocumentInput, GenerationConstraints
        constraints = GenerationConstraints(
            max_pages=8, style_guide="NeurIPS",
            output_format="latex", citation_format="bibtex",
        )
        di = DocumentInput(
            title="Test", content_brief={"idea": "test idea"},
            references=["@article{test}"], constraints=constraints,
        )
        assert di.title == "Test"
        assert di.constraints.max_pages == 8


# =========================================================================
# 2. Evidence Layer
# =========================================================================


class TestDAGBuilder:
    """Verify DAGBuilder can construct a DAG from scratch."""

    @pytest.mark.asyncio
    async def test_build_empty_dag(self):
        from src.evidence.dag_builder import DAGBuilder
        builder = DAGBuilder()
        dag = await builder.build()
        assert dag is not None
        assert len(dag.evidence_nodes) == 0
        assert len(dag.claim_nodes) == 0

    @pytest.mark.asyncio
    async def test_build_with_code_context(self):
        from src.evidence.dag_builder import DAGBuilder
        builder = DAGBuilder()
        code_context = {
            "code_evidence_graph": [
                {
                    "evidence_id": "CE001",
                    "path": "src/model.py",
                    "symbols": ["train_model", "evaluate"],
                    "dominant_role": "method",
                    "purpose": "Implements BERT fine-tuning pipeline",
                },
                {
                    "evidence_id": "CE002",
                    "path": "src/eval.py",
                    "symbols": ["compute_f1"],
                    "dominant_role": "result",
                    "purpose": "Computes evaluation metrics",
                },
            ],
        }
        dag = await builder.build(code_context=code_context)
        assert len(dag.evidence_nodes) >= 2, "Should have at least 2 code evidence nodes"

    @pytest.mark.asyncio
    async def test_build_with_research_context(self):
        from src.evidence.dag_builder import DAGBuilder
        builder = DAGBuilder()
        research_context = {
            "key_papers": [
                {"ref_id": "devlin2019", "title": "BERT", "contribution": "Pre-training"},
                {"ref_id": "vaswani2017", "title": "Attention", "contribution": "Self-attention"},
            ],
        }
        dag = await builder.build(research_context=research_context)
        assert len(dag.evidence_nodes) >= 2

    @pytest.mark.asyncio
    async def test_build_with_plan(self):
        from src.evidence.dag_builder import DAGBuilder
        from src.agents.planner_agent.models import (
            PaperPlan, SectionPlan, ParagraphPlan,
        )
        plan = PaperPlan(
            title="Test Paper",
            sections=[
                SectionPlan(
                    section_type="method", section_title="Method",
                    paragraphs=[
                        ParagraphPlan(key_point="We fine-tune BERT for NER"),
                        ParagraphPlan(key_point="We use CRF decoding"),
                    ],
                ),
                SectionPlan(
                    section_type="result", section_title="Results",
                    paragraphs=[
                        ParagraphPlan(key_point="Our model achieves 95% F1"),
                    ],
                ),
            ],
        )
        builder = DAGBuilder()
        dag = await builder.build(paper_plan=plan)
        assert len(dag.claim_nodes) >= 3, "Should extract claims from paragraphs"

    @pytest.mark.asyncio
    async def test_full_dag_with_edges(self):
        from src.evidence.dag_builder import DAGBuilder
        from src.agents.planner_agent.models import (
            PaperPlan, SectionPlan, ParagraphPlan,
        )
        builder = DAGBuilder()
        code_context = {
            "code_evidence_graph": [
                {"evidence_id": "CE001", "path": "model.py",
                 "symbols": ["train"], "dominant_role": "method",
                 "purpose": "Training pipeline"},
            ],
        }
        plan = PaperPlan(
            title="Test",
            sections=[
                SectionPlan(
                    section_type="method", section_title="Method",
                    paragraphs=[ParagraphPlan(key_point="Training approach")],
                ),
            ],
        )
        dag = await builder.build(code_context=code_context, paper_plan=plan)
        assert len(dag.edges) > 0, "Should have heuristic edges"


class TestContextPerception:
    """Verify ContextPerceptionModule as the unified entry point."""

    @pytest.mark.asyncio
    async def test_build_dag_basic(self):
        from src.evidence.context_perception import ContextPerceptionModule
        module = ContextPerceptionModule()
        dag = await module.build()
        assert dag is not None
        assert isinstance(dag.summary(), dict)


# =========================================================================
# 3. Generation Layer
# =========================================================================


class TestClaimVerifier:
    """Verify claim-level verification logic."""

    def _make_para_plan(self, claim_id="CLM001", bound_ids=None, key_point="Test"):
        from src.agents.planner_agent.models import ParagraphPlan
        return ParagraphPlan(
            key_point=key_point,
            claim_id=claim_id,
            bound_evidence_ids=bound_ids or [],
        )

    @pytest.mark.asyncio
    async def test_verify_all_valid(self):
        from src.generation.claim_verifier import ClaimVerifier, VerificationResult
        verifier = ClaimVerifier()
        paragraph_content = r"""
        We use a fine-tuned BERT model \cite{devlin2019} for classification.
        The architecture follows \cite{vaswani2017}.
        """
        para_plan = self._make_para_plan(bound_ids=[], key_point="BERT classification")
        result = await verifier.verify(
            generated_text=paragraph_content,
            paragraph_plan=para_plan,
            valid_citation_keys={"devlin2019", "vaswani2017"},
        )
        assert isinstance(result, VerificationResult)
        assert result.passed is True
        assert len(result.citation_issues) == 0

    @pytest.mark.asyncio
    async def test_verify_with_bound_evidence_requires_ref(self):
        """When bound evidence IDs are present, they must be referenced."""
        from src.generation.claim_verifier import ClaimVerifier
        verifier = ClaimVerifier()
        paragraph_content = r"We use BERT \cite{devlin2019}."
        para_plan = self._make_para_plan(bound_ids=["EV001"], key_point="BERT")
        result = await verifier.verify(
            generated_text=paragraph_content,
            paragraph_plan=para_plan,
            valid_citation_keys={"devlin2019"},
        )
        assert "EV001" in result.missing_evidence_refs

    @pytest.mark.asyncio
    async def test_verify_invalid_citation(self):
        from src.generation.claim_verifier import ClaimVerifier
        verifier = ClaimVerifier()
        paragraph_content = r"Results show improvement \cite{fake_paper2025}."
        para_plan = self._make_para_plan(key_point="improvement")
        result = await verifier.verify(
            generated_text=paragraph_content,
            paragraph_plan=para_plan,
            valid_citation_keys={"devlin2019"},
        )
        assert result.passed is False
        assert "fake_paper2025" in result.citation_issues

    @pytest.mark.asyncio
    async def test_verify_missing_evidence(self):
        from src.generation.claim_verifier import ClaimVerifier
        verifier = ClaimVerifier()
        paragraph_content = r"Our method improves accuracy."
        para_plan = self._make_para_plan(bound_ids=["EV001", "EV002"], key_point="accuracy")
        result = await verifier.verify(
            generated_text=paragraph_content,
            paragraph_plan=para_plan,
            valid_citation_keys=set(),
        )
        assert len(result.missing_evidence_refs) > 0


class TestTemplateSlots:
    """Verify template parsing and rendering."""

    def test_paragraph_template_creation(self):
        from src.generation.template_slots import ParagraphTemplate, TemplateSlot, SlotType
        slot = TemplateSlot(
            slot_type=SlotType.METRIC, slot_id="s1",
            evidence_id="EV001", constraints="Report F1 score",
        )
        assert slot.marker == "[METRIC_SLOT:s1]"

        template = ParagraphTemplate(
            skeleton="Our model achieves [METRIC_SLOT:s1] on the test set.",
            slots=[slot],
        )
        assert len(template.slots) == 1

    def test_parse_template_slots(self):
        from src.generation.template_slots import parse_template_slots
        skeleton = "We use [EVIDENCE_SLOT:s1] and achieve [METRIC_SLOT:s2]."
        slots = parse_template_slots(skeleton)
        assert len(slots) == 2
        types = {s.slot_type.value for s in slots}
        assert "EVIDENCE_SLOT" in types
        assert "METRIC_SLOT" in types

    def test_render_filled_template(self):
        from src.generation.template_slots import (
            ParagraphTemplate, TemplateSlot, SlotType, render_filled_template,
        )
        slot = TemplateSlot(
            slot_type=SlotType.METRIC, slot_id="s1",
            filled_content="95.2\\% F1 score",
        )
        template = ParagraphTemplate(
            skeleton="Our model achieves [METRIC_SLOT:s1] on the benchmark.",
            slots=[slot],
        )
        rendered = render_filled_template(template)
        assert "95.2\\% F1 score" in rendered
        assert "[METRIC_SLOT:s1]" not in rendered


# =========================================================================
# 4. PromptLoader + prompt_compiler
# =========================================================================


class TestPromptLoader:
    """Verify PromptLoader can load from files and fall back to defaults."""

    def test_loader_init(self):
        from src.prompts import PromptLoader
        loader = PromptLoader()
        assert loader is not None

    def test_load_with_default(self):
        from src.prompts import PromptLoader
        loader = PromptLoader()
        result = loader.load("nonexistent", "nonexistent", default="FALLBACK")
        assert result == "FALLBACK"

    def test_load_existing_prompt(self):
        from src.prompts import PromptLoader
        loader = PromptLoader()
        result = loader.load("metadata", "generation_system", default="FALLBACK")
        assert result != "FALLBACK", "Should load from src/prompts/metadata/generation_system.txt"
        assert "LaTeX" in result or "latex" in result.lower()

    def test_section_prompts_loaded(self):
        from src.agents.shared.prompt_compiler import SECTION_PROMPTS
        assert "abstract" in SECTION_PROMPTS
        assert "introduction" in SECTION_PROMPTS
        assert "method" in SECTION_PROMPTS
        assert len(SECTION_PROMPTS) >= 8

    def test_compile_paragraph_prompt(self):
        from src.agents.shared.prompt_compiler import compile_paragraph_prompt
        from src.agents.planner_agent.models import ParagraphPlan
        para = ParagraphPlan(
            key_point="Fine-tuned BERT for NER",
            supporting_points=["CRF decoding", "Data augmentation"],
            approx_sentences=5,
            role="evidence",
            references_to_cite=["devlin2019"],
        )
        prompt = compile_paragraph_prompt(
            paragraph_plan=para,
            section_type="method",
            section_title="Methodology",
            paragraph_index=0,
            total_paragraphs=3,
            evidence_snippets=["BERT uses bidirectional attention"],
            valid_refs=["devlin2019", "vaswani2017"],
        )
        assert "Methodology" in prompt or "method" in prompt
        assert "Fine-tuned BERT" in prompt
        assert "devlin2019" in prompt
        assert "BERT uses bidirectional attention" in prompt


# =========================================================================
# 5. Model Extensions
# =========================================================================


class TestModelExtensions:
    """Verify new methods added to existing model classes."""

    def test_paragraph_plan_effective_sentence_count_default(self):
        from src.agents.planner_agent.models import ParagraphPlan
        p = ParagraphPlan(approx_sentences=7)
        assert p.effective_sentence_count == 7

    def test_paragraph_plan_effective_sentence_count_with_plans(self):
        from src.agents.planner_agent.models import ParagraphPlan, SentencePlan
        p = ParagraphPlan(
            approx_sentences=7,
            sentence_plans=[
                SentencePlan(sentence_id="s0"),
                SentencePlan(sentence_id="s1"),
                SentencePlan(sentence_id="s2"),
            ],
        )
        assert p.effective_sentence_count == 3, "Should use sentence_plans length"

    def test_section_plan_get_total_sentences_uses_effective(self):
        from src.agents.planner_agent.models import SectionPlan, ParagraphPlan, SentencePlan
        plan = SectionPlan(
            section_type="method",
            paragraphs=[
                ParagraphPlan(approx_sentences=10, sentence_plans=[
                    SentencePlan(sentence_id="s0"),
                    SentencePlan(sentence_id="s1"),
                ]),
                ParagraphPlan(approx_sentences=5),
            ],
        )
        assert plan.get_total_sentences() == 2 + 5

    def test_paper_plan_evidence_dag_field(self):
        from src.agents.planner_agent.models import PaperPlan
        plan = PaperPlan(
            title="Test",
            evidence_dag={"evidence_nodes": {}, "claim_nodes": {}, "edges": []},
        )
        assert plan.evidence_dag is not None

    def test_paper_plan_to_document_spec(self):
        from src.agents.planner_agent.models import (
            PaperPlan, SectionPlan, ParagraphPlan,
        )
        plan = PaperPlan(
            title="Test Paper",
            contributions=["C1", "C2"],
            sections=[
                SectionPlan(
                    section_type="method", section_title="Method",
                    paragraphs=[ParagraphPlan(key_point="point 1")],
                    order=1,
                ),
                SectionPlan(
                    section_type="result", section_title="Results",
                    paragraphs=[ParagraphPlan(key_point="point 2")],
                    order=2,
                ),
            ],
        )
        spec = plan.to_document_spec()
        assert spec.title == "Test Paper"
        assert spec.document_type == "paper"
        assert len(spec.sections) == 2
        assert spec.sections[0].section_id == "method"
        assert len(spec.contributions) == 2

    def test_paper_metadata_to_document_input(self):
        from src.agents.metadata_agent.models import PaperMetaData
        meta = PaperMetaData(
            title="Test Paper",
            idea_hypothesis="Test hypothesis",
            method="Test method",
            data="Test data",
            experiments="Test experiments",
            references=["@article{test}"],
            style_guide="NeurIPS",
            target_pages=8,
        )
        di = meta.to_document_input()
        assert di.title == "Test Paper"
        assert di.content_brief["idea_hypothesis"] == "Test hypothesis"
        assert di.constraints.max_pages == 8
        assert di.constraints.style_guide == "NeurIPS"

    def test_structural_action_to_unified_action(self):
        from src.agents.metadata_agent.models import StructuralAction
        sa = StructuralAction(
            action_type="resize_figure",
            target_id="fig:arch",
            section="method",
            params={"width": "0.8\\linewidth"},
            estimated_savings=0.3,
        )
        unified = sa.to_unified_action()
        assert unified.action_type == "resize_figure"
        assert unified.target_id == "fig:arch"
        assert unified.is_layout_action
        assert unified.priority == 90

    def test_revision_task_to_unified_action(self):
        from src.agents.reviewer_agent.models import RevisionTask
        rt = RevisionTask(
            section_type="method",
            target_id="method.p1",
            action="revise",
            priority=7,
            rationale="Logic issue",
        )
        unified = rt.to_unified_action()
        assert unified.action_type == "revise"
        assert unified.is_text_action

    def test_section_feedback_to_unified_action(self):
        from src.agents.reviewer_agent.models import SectionFeedback
        sf = SectionFeedback(
            section_type="result",
            current_word_count=300,
            target_word_count=500,
            action="expand",
            delta_words=200,
        )
        unified = sf.to_unified_action()
        assert unified.action_type == "expand"
        assert unified.estimated_impact == 200.0

    def test_conflict_resolution_record_new_fields(self):
        from src.agents.reviewer_agent.models import ConflictResolutionRecord
        crr = ConflictResolutionRecord(
            section_type="method",
            objective_scores={"quality": [{"score": 0.8}]},
            pareto_front_size=3,
            resolution_method="weighted_sum",
        )
        assert crr.pareto_front_size == 3
        assert crr.resolution_method == "weighted_sum"

    def test_paragraph_result_model(self):
        from src.agents.writer_agent.models import ParagraphResult
        pr = ParagraphResult(
            latex_content="\\textbf{test}",
            paragraph_index=0,
            claim_id="CLM001",
            used_citations=["devlin2019"],
            word_count=10,
            claim_coverage=0.95,
            verification_passed=True,
        )
        assert pr.claim_id == "CLM001"
        assert pr.verification_passed

    def test_session_memory_hallucination_stats(self):
        from src.agents.shared.session_memory import ReviewRecord
        rr = ReviewRecord(
            iteration=1,
            reviewer="test_reviewer",
            hallucination_stats={"total_checked": 5, "hallucinated": 0},
        )
        assert rr.hallucination_stats["total_checked"] == 5


# =========================================================================
# 6. Agent Integration — sentence plan generation
# =========================================================================


class TestPlannerSentencePlans:
    """Verify PlannerAgent._generate_sentence_plans."""

    def test_generate_without_dag(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent
        from src.agents.planner_agent.models import ParagraphPlan, SentenceRole

        para = ParagraphPlan(
            key_point="Test point", approx_sentences=4,
        )
        plans = PlannerAgent._generate_sentence_plans(para, evidence_dag=None)
        assert len(plans) == 4
        assert plans[0].role == SentenceRole.TOPIC
        assert plans[-1].role == SentenceRole.CONCLUSION
        for sp in plans[1:-1]:
            assert sp.role == SentenceRole.EVIDENCE

    def test_generate_with_dag(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent
        from src.agents.planner_agent.models import ParagraphPlan, SentenceRole
        from src.models.evidence_graph import (
            EvidenceDAG, EvidenceNode, EvidenceNodeType,
        )

        dag = EvidenceDAG()
        dag.add_evidence(EvidenceNode(
            node_id="EV001", node_type=EvidenceNodeType.CODE,
            content="Model code", confidence=0.9,
        ))
        dag.add_evidence(EvidenceNode(
            node_id="EV002", node_type=EvidenceNodeType.LITERATURE,
            content="Related paper", confidence=0.85,
        ))

        para = ParagraphPlan(
            key_point="Test", approx_sentences=3,
            claim_id="CLM001",
            bound_evidence_ids=["EV001", "EV002"],
        )
        plans = PlannerAgent._generate_sentence_plans(para, evidence_dag=dag)
        assert len(plans) == 4  # TOPIC + 2 EVIDENCE + CONCLUSION
        assert plans[0].role == SentenceRole.TOPIC
        assert plans[1].role == SentenceRole.EVIDENCE
        assert plans[1].evidence_ids == ["EV001"]
        assert plans[2].role == SentenceRole.EVIDENCE
        assert plans[2].evidence_ids == ["EV002"]
        assert plans[3].role == SentenceRole.CONCLUSION


# =========================================================================
# 7. MetaDataAgent import & orchestrator wiring
# =========================================================================


class TestMetaDataAgentWiring:
    """Verify MetaDataAgent imports and has the right structure."""

    def test_import_metadata_agent(self):
        from src.agents.metadata_agent.metadata_agent import MetaDataAgent
        assert MetaDataAgent is not None

    def test_import_via_init(self):
        from src.agents.metadata_agent import MetaDataAgent, ReviewOrchestrator
        assert MetaDataAgent is not None
        assert ReviewOrchestrator is not None

    def test_metadata_agent_has_dag_aware_methods(self):
        from src.agents.metadata_agent.metadata_agent import MetaDataAgent
        assert hasattr(MetaDataAgent, 'generate_paper')
        assert hasattr(MetaDataAgent, '_save_artifact')
        assert hasattr(MetaDataAgent, '_save_compilation_output')

    def test_orchestrator_import(self):
        from src.agents.metadata_agent.orchestrator import ReviewOrchestrator
        assert ReviewOrchestrator is not None

    def test_conflict_resolver_import(self):
        from src.agents.metadata_agent.conflict_resolver import ConflictResolver
        assert ConflictResolver is not None

    def test_revision_executor_import(self):
        from src.agents.metadata_agent.revision_executor import RevisionExecutor
        assert RevisionExecutor is not None

    def test_evidence_checker_import(self):
        from src.agents.reviewer_agent.checkers.evidence_check import EvidenceChecker
        assert EvidenceChecker is not None

    def test_no_event_emitter_references(self):
        """Ensure commercial version does not reference academic EventEmitter."""
        import inspect
        from src.agents.metadata_agent import metadata_agent as ma_module
        source = inspect.getsource(ma_module)
        assert "EventEmitter" not in source, "EventEmitter should not exist in commercial metadata_agent"
        assert "GenerationEvent" not in source, "GenerationEvent should not exist in commercial metadata_agent"

    def test_has_progress_emitter(self):
        """Verify commercial ProgressEmitter is used."""
        import inspect
        from src.agents.metadata_agent import metadata_agent as ma_module
        source = inspect.getsource(ma_module)
        assert "ProgressEmitter" in source


# =========================================================================
# 8. Full round-trip: DAG → Plan → Prompt → Verify
# =========================================================================


class TestFullRoundTrip:
    """End-to-end: build DAG, generate sentence plans, compile prompt, verify."""

    @pytest.mark.asyncio
    async def test_dag_to_prompt_pipeline(self):
        from src.models.evidence_graph import (
            EvidenceDAG, EvidenceNode, ClaimNode, DAGEdge,
            EvidenceNodeType, ClaimNodeType, EdgeType,
        )
        from src.agents.planner_agent.models import (
            PaperPlan, SectionPlan, ParagraphPlan, SentencePlan,
        )
        from src.agents.planner_agent.planner_agent import PlannerAgent
        from src.agents.shared.prompt_compiler import compile_paragraph_prompt
        from src.generation.claim_verifier import ClaimVerifier

        # 1. Build DAG
        dag = EvidenceDAG()
        dag.add_evidence(EvidenceNode(
            node_id="EV001", node_type=EvidenceNodeType.CODE,
            content="BERT classifier with CRF decoding", source_path="model.py",
            confidence=0.9,
        ))
        dag.add_evidence(EvidenceNode(
            node_id="EV002", node_type=EvidenceNodeType.LITERATURE,
            content="Devlin et al. BERT paper", source_path="devlin2019",
            confidence=0.95,
        ))
        dag.add_claim(ClaimNode(
            node_id="CLM001", node_type=ClaimNodeType.METHOD_CLAIM,
            statement="We use BERT with CRF for NER",
            section_scope=["method"],
        ))
        dag.add_edge(DAGEdge(
            source_id="EV001", target_id="CLM001",
            edge_type=EdgeType.SUPPORTS, weight=0.9, is_bound=True,
        ))
        dag.add_edge(DAGEdge(
            source_id="EV002", target_id="CLM001",
            edge_type=EdgeType.SUPPORTS, weight=0.85, is_bound=True,
        ))

        # 2. Create a paragraph plan with DAG binding
        para = ParagraphPlan(
            key_point="We use BERT with CRF for NER",
            supporting_points=["Bidirectional attention", "CRF layer"],
            approx_sentences=4,
            claim_id="CLM001",
            bound_evidence_ids=["EV001", "EV002"],
            references_to_cite=["devlin2019"],
        )

        # 3. Generate sentence plans
        sentence_plans = PlannerAgent._generate_sentence_plans(para, evidence_dag=dag)
        assert len(sentence_plans) >= 3

        # 4. Attach and verify effective_sentence_count
        para.sentence_plans = sentence_plans
        assert para.effective_sentence_count == len(sentence_plans)

        # 5. Compile paragraph prompt
        evidence_snippets = [
            ev.content for ev in dag.get_evidence_for_claim("CLM001")
        ]
        prompt = compile_paragraph_prompt(
            paragraph_plan=para,
            section_type="method",
            section_title="Methodology",
            evidence_snippets=evidence_snippets,
            valid_refs=["devlin2019"],
            paragraph_index=0,
            total_paragraphs=3,
        )
        assert "BERT" in prompt
        assert "devlin2019" in prompt
        assert "Sentence-level Plan" in prompt

        # 6. Verify a well-formed paragraph
        good_paragraph = (
            r"We use a fine-tuned BERT model \cite{devlin2019} for named entity recognition. "
            r"The architecture combines bidirectional attention with a CRF layer."
        )
        verifier = ClaimVerifier()
        result = await verifier.verify(
            generated_text=good_paragraph,
            paragraph_plan=para,
            evidence_dag=dag,
            valid_citation_keys={"devlin2019"},
        )
        assert result.passed is True
        assert len(result.citation_issues) == 0

    def test_dag_serialization_in_paper_plan(self):
        """Verify DAG can be stored in and restored from PaperPlan."""
        from src.models.evidence_graph import (
            EvidenceDAG, EvidenceNode, ClaimNode, DAGEdge,
            EvidenceNodeType, ClaimNodeType, EdgeType,
        )
        from src.agents.planner_agent.models import PaperPlan, SectionPlan, ParagraphPlan

        dag = EvidenceDAG()
        dag.add_evidence(EvidenceNode(
            node_id="EV001", node_type=EvidenceNodeType.CODE,
            content="Test code", confidence=0.9,
        ))
        dag.add_claim(ClaimNode(
            node_id="CLM001", node_type=ClaimNodeType.METHOD_CLAIM,
            statement="Test claim",
        ))
        dag.add_edge(DAGEdge(
            source_id="EV001", target_id="CLM001", is_bound=True,
        ))

        plan = PaperPlan(
            title="Test",
            sections=[SectionPlan(section_type="method")],
            evidence_dag=dag.to_serializable(),
        )

        # Serialize + deserialize
        plan_json = plan.model_dump()
        restored_plan = PaperPlan(**plan_json)

        restored_dag = EvidenceDAG.from_serializable(restored_plan.evidence_dag)
        assert restored_dag is not None
        assert len(restored_dag.evidence_nodes) == 1
        assert len(restored_dag.claim_nodes) == 1
        assert len(restored_dag.edges) == 1
        evidence = restored_dag.get_evidence_for_claim("CLM001")
        assert len(evidence) == 1
        assert evidence[0].node_id == "EV001"

    def test_main_app_import(self):
        """Verify the main FastAPI app loads without import errors."""
        from src.main import app
        assert app is not None


# =========================================================================
# 9. EvidenceChecker: citation key resolution (LIT node ID mismatch fix)
# =========================================================================


class TestEvidenceCheckerCitationKeyResolution:
    """
    Verify that EvidenceChecker resolves LITERATURE node IDs (LIT001)
    to actual BibTeX citation keys (source_path) when checking and prompting.
    """

    def _build_dag_with_literature(self):
        from src.models.evidence_graph import (
            EvidenceDAG, EvidenceNode, ClaimNode, DAGEdge,
            EvidenceNodeType, ClaimNodeType, EdgeType,
        )
        dag = EvidenceDAG()
        dag.add_evidence(EvidenceNode(
            node_id="LIT001",
            node_type=EvidenceNodeType.LITERATURE,
            content="Proposed Flamingo architecture",
            source_path="alayrac2022flamingo",
            confidence=0.9,
        ))
        dag.add_evidence(EvidenceNode(
            node_id="LIT002",
            node_type=EvidenceNodeType.LITERATURE,
            content="BLIP-2 bootstrapping approach",
            source_path="li2023blip2",
            confidence=0.95,
        ))
        dag.add_evidence(EvidenceNode(
            node_id="FIG001",
            node_type=EvidenceNodeType.FIGURE,
            content="Architecture diagram",
            source_path="figures/arch.png",
        ))
        dag.add_claim(ClaimNode(
            node_id="CLM001",
            node_type=ClaimNodeType.CONTEXT,
            statement="Flamingo introduced cross-attention for VL alignment.",
            section_scope=["introduction"],
        ))
        dag.add_claim(ClaimNode(
            node_id="CLM002",
            node_type=ClaimNodeType.METHOD_CLAIM,
            statement="BLIP-2 uses Q-Former for efficient bridging.",
            section_scope=["method"],
        ))
        dag.add_edge(DAGEdge(
            source_id="LIT001", target_id="CLM001", is_bound=True,
        ))
        dag.add_edge(DAGEdge(
            source_id="LIT002", target_id="CLM002", is_bound=True,
        ))
        dag.add_edge(DAGEdge(
            source_id="FIG001", target_id="CLM002", is_bound=True,
        ))
        return dag

    def test_check_refs_matches_citation_key_not_node_id(self):
        """Content with \\cite{alayrac2022flamingo} should match LIT001."""
        from src.agents.reviewer_agent.checkers.evidence_check import EvidenceChecker
        dag = self._build_dag_with_literature()
        content = r"Prior work \cite{alayrac2022flamingo} introduced cross-attention."
        assert EvidenceChecker._check_evidence_references(content, ["LIT001"], dag) is True

    def test_check_refs_rejects_bare_node_id_for_literature(self):
        """Content with \\cite{LIT001} should NOT match — LIT001 is not a real key."""
        from src.agents.reviewer_agent.checkers.evidence_check import EvidenceChecker
        dag = self._build_dag_with_literature()
        content = r"Prior work \cite{LIT001} introduced cross-attention."
        assert EvidenceChecker._check_evidence_references(content, ["LIT001"], dag) is False

    def test_check_refs_no_citation_key_present(self):
        """Content without any matching citation key should fail."""
        from src.agents.reviewer_agent.checkers.evidence_check import EvidenceChecker
        dag = self._build_dag_with_literature()
        content = r"Prior work introduced cross-attention for vision-language."
        assert EvidenceChecker._check_evidence_references(content, ["LIT001"], dag) is False

    def test_check_refs_non_literature_still_uses_node_id(self):
        """FIG001 (non-LITERATURE) should still match by node ID / \\ref{}."""
        from src.agents.reviewer_agent.checkers.evidence_check import EvidenceChecker
        dag = self._build_dag_with_literature()
        content = r"As shown in Figure \ref{FIG001}, the architecture..."
        assert EvidenceChecker._check_evidence_references(content, ["FIG001"], dag) is True

    def test_check_refs_multi_cite_command(self):
        """\\cite{key1,alayrac2022flamingo,key3} should match LIT001."""
        from src.agents.reviewer_agent.checkers.evidence_check import EvidenceChecker
        dag = self._build_dag_with_literature()
        content = r"Related approaches \cite{key1,alayrac2022flamingo,key3} show..."
        assert EvidenceChecker._check_evidence_references(content, ["LIT001"], dag) is True

    def test_resolve_citation_keys_converts_lit_ids(self):
        """_resolve_citation_keys should convert LIT001 → alayrac2022flamingo."""
        from src.agents.reviewer_agent.checkers.evidence_check import EvidenceChecker
        dag = self._build_dag_with_literature()
        resolved = EvidenceChecker._resolve_citation_keys(["LIT001", "LIT002"], dag)
        assert "alayrac2022flamingo" in resolved
        assert "li2023blip2" in resolved
        assert "LIT001" not in resolved
        assert "LIT002" not in resolved

    def test_resolve_citation_keys_keeps_non_literature_ids(self):
        """Non-LITERATURE nodes (FIG001) should keep their original ID."""
        from src.agents.reviewer_agent.checkers.evidence_check import EvidenceChecker
        dag = self._build_dag_with_literature()
        resolved = EvidenceChecker._resolve_citation_keys(["FIG001", "LIT001"], dag)
        assert "FIG001" in resolved
        assert "alayrac2022flamingo" in resolved

    def test_revision_prompt_uses_citation_keys(self):
        """generate_revision_prompt should emit actual citation keys, not LIT IDs."""
        from src.agents.reviewer_agent.checkers.evidence_check import EvidenceChecker
        from src.agents.reviewer_agent.models import FeedbackResult, Severity
        checker = EvidenceChecker()
        feedback = FeedbackResult(
            checker_name="evidence_check",
            passed=False,
            severity=Severity.WARNING,
            message="test",
            details={
                "drifted_claims": [
                    {
                        "claim_id": "CLM001",
                        "claim_text": "Flamingo introduced cross-attention.",
                        "section_type": "introduction",
                        "expected_evidence": ["alayrac2022flamingo"],
                    },
                ],
            },
        )
        prompt = checker.generate_revision_prompt("introduction", "some content", feedback)
        assert "alayrac2022flamingo" in prompt
        assert "LIT001" not in prompt
        assert "Required citation keys" in prompt
