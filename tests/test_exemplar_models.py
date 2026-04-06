"""Tests for Exemplar Paper models (TDD: RED first)."""
from __future__ import annotations

import json

import pytest

from src.agents.metadata_agent.models import (
    ExemplarAnalysis,
    SectionBlueprint,
    StyleProfile,
    ArgumentationPatterns,
    PaperMetaData,
    PaperGenerationRequest,
    PlanResult,
)


# ---------------------------------------------------------------------------
# SectionBlueprint
# ---------------------------------------------------------------------------

class TestSectionBlueprint:
    def test_minimal_construction(self):
        bp = SectionBlueprint(section_type="introduction", title="Introduction")
        assert bp.section_type == "introduction"
        assert bp.title == "Introduction"
        assert bp.approx_word_count == 0
        assert bp.paragraph_count == 0
        assert bp.subsection_titles == []
        assert bp.role == ""

    def test_full_construction(self):
        bp = SectionBlueprint(
            section_type="method",
            title="Methodology",
            approx_word_count=1200,
            paragraph_count=6,
            subsection_titles=["Overview", "Architecture", "Training"],
            role="Detail the proposed approach",
        )
        assert bp.approx_word_count == 1200
        assert len(bp.subsection_titles) == 3

    def test_serialization_roundtrip(self):
        bp = SectionBlueprint(
            section_type="results",
            title="Results",
            approx_word_count=800,
            paragraph_count=4,
        )
        data = bp.model_dump()
        restored = SectionBlueprint(**data)
        assert restored == bp


# ---------------------------------------------------------------------------
# StyleProfile
# ---------------------------------------------------------------------------

class TestStyleProfile:
    def test_defaults(self):
        sp = StyleProfile()
        assert sp.tone == "formal"
        assert sp.citation_density == 0.0
        assert sp.avg_sentence_length == 0.0
        assert sp.hedging_level == "moderate"
        assert sp.transition_patterns == []

    def test_full_construction(self):
        sp = StyleProfile(
            tone="semi-formal",
            citation_density=3.5,
            avg_sentence_length=22.0,
            hedging_level="low",
            transition_patterns=["however", "moreover", "in contrast"],
        )
        assert sp.tone == "semi-formal"
        assert len(sp.transition_patterns) == 3

    def test_serialization_roundtrip(self):
        sp = StyleProfile(
            tone="formal",
            citation_density=2.0,
            avg_sentence_length=18.5,
        )
        data = sp.model_dump()
        restored = StyleProfile(**data)
        assert restored == sp


# ---------------------------------------------------------------------------
# ArgumentationPatterns
# ---------------------------------------------------------------------------

class TestArgumentationPatterns:
    def test_defaults(self):
        ap = ArgumentationPatterns()
        assert ap.intro_hook_type == ""
        assert ap.claim_evidence_structure == ""
        assert ap.discussion_closing_strategy == ""

    def test_full_construction(self):
        ap = ArgumentationPatterns(
            intro_hook_type="broad_significance",
            claim_evidence_structure="claim_first_then_evidence",
            discussion_closing_strategy="limitation_then_future_work",
        )
        assert ap.intro_hook_type == "broad_significance"


# ---------------------------------------------------------------------------
# ExemplarAnalysis
# ---------------------------------------------------------------------------

class TestExemplarAnalysis:
    def test_minimal_construction(self):
        ea = ExemplarAnalysis(ref_id="doe2024", title="A Great Paper", venue="Nature")
        assert ea.ref_id == "doe2024"
        assert ea.year == 0
        assert ea.section_blueprint == []
        assert ea.style_profile is not None
        assert ea.argumentation_patterns is not None
        assert ea.paragraph_archetypes == {}

    def test_full_construction(self):
        ea = ExemplarAnalysis(
            ref_id="doe2024",
            title="A Great Paper",
            venue="Nature",
            year=2024,
            section_blueprint=[
                SectionBlueprint(section_type="introduction", title="Intro", paragraph_count=3),
                SectionBlueprint(section_type="results", title="Results", paragraph_count=8),
            ],
            style_profile=StyleProfile(tone="formal", citation_density=2.5),
            argumentation_patterns=ArgumentationPatterns(intro_hook_type="gap_driven"),
            paragraph_archetypes={
                "introduction": ["broad_hook", "gap_statement", "contribution_list"],
                "results": ["finding_statement", "evidence_detail", "interpretation"],
            },
        )
        assert len(ea.section_blueprint) == 2
        assert ea.paragraph_archetypes["introduction"][0] == "broad_hook"

    def test_serialization_roundtrip(self):
        ea = ExemplarAnalysis(
            ref_id="test2024",
            title="Test",
            venue="ICML",
            year=2024,
            section_blueprint=[
                SectionBlueprint(section_type="method", title="Method"),
            ],
            paragraph_archetypes={"method": ["overview", "detail"]},
        )
        data = ea.model_dump()
        restored = ExemplarAnalysis(**data)
        assert restored.ref_id == ea.ref_id
        assert len(restored.section_blueprint) == 1
        assert restored.paragraph_archetypes == ea.paragraph_archetypes

    def test_json_roundtrip(self):
        ea = ExemplarAnalysis(
            ref_id="json2024",
            title="JSON Test",
            venue="NeurIPS",
            year=2024,
        )
        json_str = ea.model_dump_json()
        parsed = json.loads(json_str)
        restored = ExemplarAnalysis(**parsed)
        assert restored == ea


# ---------------------------------------------------------------------------
# Extended existing models
# ---------------------------------------------------------------------------

class TestPaperMetaDataExemplarField:
    def test_exemplar_paper_path_default_none(self):
        md = PaperMetaData(
            title="T",
            idea_hypothesis="i",
            method="m",
            data="d",
            experiments="e",
        )
        assert md.exemplar_paper_path is None

    def test_exemplar_paper_path_set(self):
        md = PaperMetaData(
            title="T",
            idea_hypothesis="i",
            method="m",
            data="d",
            experiments="e",
            exemplar_paper_path="/tmp/exemplar.pdf",
        )
        assert md.exemplar_paper_path == "/tmp/exemplar.pdf"


class TestPaperGenerationRequestExemplarField:
    def test_enable_exemplar_default_false(self):
        req = PaperGenerationRequest()
        assert req.enable_exemplar is False

    def test_enable_exemplar_set_true(self):
        req = PaperGenerationRequest(enable_exemplar=True)
        assert req.enable_exemplar is True


class TestPlanResultExemplarField:
    def test_exemplar_analysis_default_none(self):
        pr = PlanResult(paper_plan={})
        assert pr.exemplar_analysis is None

    def test_exemplar_analysis_set(self):
        ea_dict = {"ref_id": "x", "title": "X", "venue": "Nature"}
        pr = PlanResult(paper_plan={}, exemplar_analysis=ea_dict)
        assert pr.exemplar_analysis == ea_dict
