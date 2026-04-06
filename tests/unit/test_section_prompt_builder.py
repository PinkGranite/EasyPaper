"""
Tests for SectionPromptBuilder — declarative prompt composition.

Phase 3: Verifies system/user prompt separation, token budget compliance,
and stable system prompt caching.
"""
import pytest


class TestSectionPromptBuilder:
    """Verify prompt builder output structure and constraints."""

    def _builder(self, section_type="method"):
        from src.agents.shared.section_prompt_builder import SectionPromptBuilder
        return SectionPromptBuilder(section_type)

    def test_produces_system_and_user_prompts(self):
        builder = self._builder()
        system_prompt, user_prompt = builder.build(token_budget=4000)

        assert isinstance(system_prompt, str)
        assert isinstance(user_prompt, str)
        assert len(system_prompt) > 0
        assert len(user_prompt) > 0

    def test_system_prompt_is_stable(self):
        b1 = self._builder("method")
        b2 = self._builder("method")

        sys1, _ = b1.build(token_budget=4000)
        sys2, _ = b2.build(token_budget=4000)

        assert sys1 == sys2

    def test_different_section_types_differ(self):
        b_method = self._builder("method")
        b_intro = self._builder("introduction")

        sys_m, _ = b_method.build(token_budget=4000)
        sys_i, _ = b_intro.build(token_budget=4000)

        assert sys_m != sys_i

    def test_respects_token_budget(self):
        builder = (
            self._builder()
            .with_memory_context("A" * 5000)
            .with_code_context("B" * 5000)
        )
        _, user_prompt = builder.build(token_budget=500)

        estimated_tokens = (len(user_prompt) // 4) + 1
        assert estimated_tokens <= 600  # allow some overhead

    def test_includes_memory_context(self):
        builder = self._builder().with_memory_context(
            "Previously wrote introduction with 500 words."
        )
        _, user_prompt = builder.build(token_budget=4000)

        assert "previously wrote" in user_prompt.lower() or "introduction" in user_prompt.lower()

    def test_includes_style_guide(self):
        builder = self._builder().with_style_guide("NeurIPS 2025")
        _, user_prompt = builder.build(token_budget=4000)

        assert "neurips" in user_prompt.lower()

    def test_builder_chaining(self):
        builder = (
            self._builder()
            .with_memory_context("context")
            .with_code_context("code")
            .with_style_guide("ICML")
        )

        assert builder is not None
        system_prompt, user_prompt = builder.build(token_budget=4000)
        assert len(system_prompt) > 0

    def test_empty_builder(self):
        builder = self._builder()
        system_prompt, user_prompt = builder.build(token_budget=4000)

        assert len(system_prompt) > 0
        assert isinstance(user_prompt, str)
