"""Tests for exemplar_guidance injection in prompt_compiler (TDD: RED first)."""
from __future__ import annotations

import pytest

from src.agents.shared.prompt_compiler import (
    compile_introduction_prompt,
    compile_body_section_prompt,
    compile_synthesis_prompt,
    compile_paragraph_prompt,
    PROMPT_BUDGETS,
)


SAMPLE_GUIDANCE = (
    "## Exemplar Paper Writing Guide\n"
    'Source: "A Great Paper" (Nature, 2024)\n'
    "### Style Reference\n"
    "- Tone: formal"
)


# ---------------------------------------------------------------------------
# Budget key exists
# ---------------------------------------------------------------------------

def test_exemplar_guidance_budget_key_exists():
    assert "exemplar_guidance_chars" in PROMPT_BUDGETS
    assert PROMPT_BUDGETS["exemplar_guidance_chars"] > 0


# ---------------------------------------------------------------------------
# compile_introduction_prompt
# ---------------------------------------------------------------------------

class TestIntroductionPrompt:
    def test_no_guidance_backward_compat(self):
        prompt = compile_introduction_prompt(
            paper_title="Test",
            idea_hypothesis="H",
            method_summary="M",
            data_summary="D",
            experiments_summary="E",
        )
        assert "Exemplar" not in prompt

    def test_guidance_injected(self):
        prompt = compile_introduction_prompt(
            paper_title="Test",
            idea_hypothesis="H",
            method_summary="M",
            data_summary="D",
            experiments_summary="E",
            exemplar_guidance=SAMPLE_GUIDANCE,
        )
        assert "Exemplar Paper Writing Guide" in prompt
        assert "A Great Paper" in prompt

    def test_long_guidance_truncated(self):
        budget = PROMPT_BUDGETS["exemplar_guidance_chars"]
        long_guidance = "X" * (budget + 500)
        prompt_without = compile_introduction_prompt(
            paper_title="Test",
            idea_hypothesis="H",
            method_summary="M",
            data_summary="D",
            experiments_summary="E",
        )
        prompt_with = compile_introduction_prompt(
            paper_title="Test",
            idea_hypothesis="H",
            method_summary="M",
            data_summary="D",
            experiments_summary="E",
            exemplar_guidance=long_guidance,
        )
        injected_len = len(prompt_with) - len(prompt_without)
        assert injected_len <= budget + 10  # truncation marker overhead


# ---------------------------------------------------------------------------
# compile_body_section_prompt
# ---------------------------------------------------------------------------

class TestBodySectionPrompt:
    def test_no_guidance_backward_compat(self):
        prompt = compile_body_section_prompt(
            section_type="method",
            metadata_content="content",
            intro_context="intro",
        )
        assert "Exemplar" not in prompt

    def test_guidance_injected(self):
        prompt = compile_body_section_prompt(
            section_type="method",
            metadata_content="content",
            intro_context="intro",
            exemplar_guidance=SAMPLE_GUIDANCE,
        )
        assert "Exemplar Paper Writing Guide" in prompt


# ---------------------------------------------------------------------------
# compile_synthesis_prompt
# ---------------------------------------------------------------------------

class TestSynthesisPrompt:
    def test_no_guidance_backward_compat(self):
        prompt = compile_synthesis_prompt(
            section_type="abstract",
            paper_title="Test",
            prior_sections={"introduction": "intro text"},
        )
        assert "Exemplar" not in prompt

    def test_guidance_injected(self):
        prompt = compile_synthesis_prompt(
            section_type="abstract",
            paper_title="Test",
            prior_sections={"introduction": "intro text"},
            exemplar_guidance=SAMPLE_GUIDANCE,
        )
        assert "Exemplar Paper Writing Guide" in prompt


# ---------------------------------------------------------------------------
# compile_paragraph_prompt
# ---------------------------------------------------------------------------

class TestParagraphPrompt:
    def _make_plan(self):
        from unittest.mock import MagicMock
        plan = MagicMock()
        plan.claim = "We propose X"
        plan.claim_id = "c1"
        plan.evidence_keys = []
        plan.target_words = 150
        plan.sentence_plans = []
        plan.bound_evidence_ids = []
        return plan

    def test_no_guidance_backward_compat(self):
        prompt = compile_paragraph_prompt(
            paragraph_plan=self._make_plan(),
            section_type="method",
        )
        assert "Exemplar" not in prompt

    def test_guidance_injected(self):
        prompt = compile_paragraph_prompt(
            paragraph_plan=self._make_plan(),
            section_type="method",
            exemplar_guidance=SAMPLE_GUIDANCE,
        )
        assert "Exemplar Paper Writing Guide" in prompt
