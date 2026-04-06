"""Tests for TemplateWriterGuide injection in prompt compiler."""
import pytest
from unittest.mock import MagicMock

from src.agents.shared.prompt_compiler import (
    compile_introduction_prompt,
    compile_body_section_prompt,
    compile_synthesis_prompt,
    compile_paragraph_prompt,
)
from src.agents.shared.template_analyzer import TemplateWriterGuide


@pytest.fixture
def sample_guide():
    return TemplateWriterGuide(
        available_packages=["amsmath", "graphicx", "natbib", "booktabs"],
        document_class="article",
        column_format="double",
        citation_style="citep",
        figure_guidance="Use figure for single-column, figure* for spanning.",
        table_guidance="Use \\toprule, \\midrule, \\bottomrule.",
        algorithm_guidance="",
        general_constraints="Use \\citep{} for parenthetical citations.",
    )


class TestIntroPromptWithGuide:
    def test_guide_injected_into_intro_prompt(self, sample_guide):
        prompt = compile_introduction_prompt(
            paper_title="Test Paper",
            idea_hypothesis="Test hypothesis",
            method_summary="Test method",
            data_summary="Test data",
            experiments_summary="Test experiments",
            template_guide=sample_guide,
        )
        assert "Template Constraints" in prompt
        assert "booktabs" in prompt
        assert "\\toprule" in prompt

    def test_no_guide_no_constraints_block(self):
        prompt = compile_introduction_prompt(
            paper_title="Test Paper",
            idea_hypothesis="Test hypothesis",
            method_summary="Test method",
            data_summary="Test data",
            experiments_summary="Test experiments",
        )
        assert "Template Constraints" not in prompt


class TestBodyPromptWithGuide:
    def test_guide_injected_into_body_prompt(self, sample_guide):
        prompt = compile_body_section_prompt(
            section_type="method",
            metadata_content="Method content",
            intro_context="Intro context",
            template_guide=sample_guide,
        )
        assert "Template Constraints" in prompt
        assert "figure*" in prompt

    def test_no_guide_no_constraints_block(self):
        prompt = compile_body_section_prompt(
            section_type="method",
            metadata_content="Method content",
            intro_context="Intro context",
        )
        assert "Template Constraints" not in prompt


class TestSynthesisPromptWithGuide:
    def test_guide_injected_into_synthesis_prompt(self, sample_guide):
        prompt = compile_synthesis_prompt(
            section_type="conclusion",
            paper_title="Test Paper",
            prior_sections={"introduction": "intro text"},
            template_guide=sample_guide,
        )
        assert "Template Constraints" in prompt

    def test_no_guide_no_constraints_block(self):
        prompt = compile_synthesis_prompt(
            section_type="abstract",
            paper_title="Test Paper",
            prior_sections={"introduction": "intro text"},
        )
        assert "Template Constraints" not in prompt


class TestParagraphPromptWithGuide:
    def test_guide_injected_into_paragraph_prompt(self, sample_guide):
        mock_plan = MagicMock()
        mock_plan.key_point = "Test point"
        mock_plan.supporting_points = []
        mock_plan.role = "evidence"
        mock_plan.sentence_plans = []
        mock_plan.effective_sentence_count = 3
        mock_plan.references_to_cite = []
        mock_plan.figures_to_reference = []
        mock_plan.tables_to_reference = []

        prompt = compile_paragraph_prompt(
            paragraph_plan=mock_plan,
            section_type="method",
            template_guide=sample_guide,
        )
        assert "Template Constraints" in prompt
