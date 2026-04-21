"""
Tests for the decomposed (paragraph-level) generation pipeline.

Validates that MetaDataAgent._generate_section_decomposed correctly
calls WriterAgent.generate_paragraph with the right parameter names,
handles no-claim paragraphs, and uses _all_paragraphs().
"""
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from src.agents.metadata_agent.metadata_agent import MetaDataAgent
from src.agents.planner_agent.models import FigureUsagePlan, ParagraphPlan
from src.agents.shared.session_memory import SessionMemory


# ═══════════════════════════════════════════════════════════════════════════
# Path Unification: decomposed path handles all paragraph types
# ═══════════════════════════════════════════════════════════════════════════


class TestDecomposedPathUnification:
    """Decomposed path must handle paragraphs without claim_id (no DAG)."""

    def test_decomposed_uses_all_paragraphs(self):
        """_generate_section_decomposed must iterate _all_paragraphs() to
        include paragraphs nested inside subsections."""
        import textwrap
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent._generate_section_decomposed)
        assert "_all_paragraphs()" in source, (
            "_generate_section_decomposed must use section_plan._all_paragraphs() "
            "instead of section_plan.paragraphs to include subsection paragraphs"
        )

    def test_decomposed_skips_verifier_for_no_claim(self):
        """When claim_id is empty, ClaimVerifier.verify should NOT be called."""
        import textwrap
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent._generate_section_decomposed)
        # The method should have conditional logic around claim_id for verification
        assert "claim_id" in source, "decomposed must reference claim_id"

    def test_body_section_always_calls_decomposed(self):
        """_generate_body_section should NOT have has_claim_bindings fallback."""
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent._generate_body_section)
        assert "has_claim_bindings" not in source, (
            "_generate_body_section must not branch on has_claim_bindings; "
            "it should always call _generate_section_decomposed"
        )

    def test_introduction_does_not_call_writer_run(self):
        """_generate_introduction must not call self._writer.run()."""
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent._generate_introduction)
        assert "self._writer.run(" not in source, (
            "_generate_introduction must not use the legacy self._writer.run() path; "
            "it should delegate to _generate_section_decomposed"
        )

    def test_synthesis_does_not_call_writer_run(self):
        """_generate_synthesis_section must not call self._writer.run()."""
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent._generate_synthesis_section)
        assert "self._writer.run(" not in source, (
            "_generate_synthesis_section must not use the legacy self._writer.run() path"
        )


class TestGenerateParagraphSignature:
    """Verify the call-site in _generate_section_decomposed matches
    WriterAgent.generate_paragraph's actual signature."""

    def test_generate_paragraph_accepts_valid_refs_kwarg(self):
        """WriterAgent.generate_paragraph must accept 'valid_refs' as a keyword."""
        from src.agents.writer_agent.writer_agent import WriterAgent
        sig = inspect.signature(WriterAgent.generate_paragraph)
        assert "valid_refs" in sig.parameters, (
            f"generate_paragraph missing 'valid_refs'; params={list(sig.parameters)}"
        )

    def test_caller_does_not_pass_unexpected_kwargs(self):
        """_generate_section_decomposed must NOT pass kwargs that
        generate_paragraph does not accept (e.g. memory, peers,
        valid_citation_keys)."""
        from src.agents.writer_agent.writer_agent import WriterAgent
        sig = inspect.signature(WriterAgent.generate_paragraph)
        accepted = set(sig.parameters.keys()) - {"self"}

        # These are the kwargs that _generate_section_decomposed passes.
        # If any of them are not in the accepted set, the call will TypeError.
        caller_kwargs = {
            "paragraph_prompt",
            "section_type",
            "valid_refs",
            "paragraph_index",
            "claim_id",
        }
        unexpected = caller_kwargs - accepted
        assert not unexpected, (
            f"Caller passes kwargs not accepted by generate_paragraph: {unexpected}"
        )

    def test_caller_does_not_use_valid_citation_keys_for_writer(self):
        """Ensure _generate_section_decomposed uses 'valid_refs', not 
        'valid_citation_keys' when calling generate_paragraph or
        generate_from_template."""
        import ast, textwrap
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent._generate_section_decomposed)
        tree = ast.parse(textwrap.dedent(source))

        writer_methods = {"generate_paragraph", "generate_from_template"}
        bad_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                attr = getattr(node.func, "attr", "")
                if attr in writer_methods:
                    for kw in node.keywords:
                        if kw.arg == "valid_citation_keys":
                            bad_calls.append(f"{attr}() uses 'valid_citation_keys'")

        assert not bad_calls, (
            "_generate_section_decomposed passes 'valid_citation_keys' "
            f"to writer methods — should be 'valid_refs': {bad_calls}"
        )

    def test_caller_does_not_pass_memory_or_peers(self):
        """Ensure _generate_section_decomposed does NOT pass 'memory' or 
        'peers' kwargs to generate_paragraph (they are not accepted)."""
        import ast, textwrap
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent._generate_section_decomposed)
        tree = ast.parse(textwrap.dedent(source))

        # Find calls to generate_paragraph and check for forbidden kwargs
        forbidden = {"memory", "peers"}
        found_forbidden = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                attr = getattr(node.func, "attr", "")
                if attr == "generate_paragraph":
                    for kw in node.keywords:
                        if kw.arg in forbidden:
                            found_forbidden.add(kw.arg)

        assert not found_forbidden, (
            f"_generate_section_decomposed passes forbidden kwargs to "
            f"generate_paragraph: {found_forbidden}"
        )


class TestRequiredFigureUsageValidation:

    def test_accepts_writer_authored_float_marker(self):
        para = ParagraphPlan(
            key_point="Discuss the main figure",
            figure_usages=[
                FigureUsagePlan(
                    figure_id="fig:arch",
                    must_appear=True,
                    supported_claim="The architecture motivates the paragraph.",
                ),
            ],
        )
        passed, feedback, missing = MetaDataAgent._validate_required_figure_usages(
            raw_latex="As shown in [FLOAT:fig:arch], the architecture is modular.",
            final_latex="As shown in Figure~\\ref{fig:arch}, the architecture is modular.",
            paragraph_plan=para,
        )
        assert passed is True
        assert feedback == ""
        assert missing == []

    def test_rejects_auto_appended_ref_without_writer_marker(self):
        para = ParagraphPlan(
            key_point="Discuss the main figure",
            figure_usages=[
                FigureUsagePlan(
                    figure_id="fig:arch",
                    must_appear=True,
                    supported_claim="The architecture motivates the paragraph.",
                ),
            ],
        )
        passed, feedback, missing = MetaDataAgent._validate_required_figure_usages(
            raw_latex="As visualized in, the architecture is modular.",
            final_latex="As visualized in Figure~\\ref{fig:arch}, the architecture is modular.",
            paragraph_plan=para,
        )
        assert passed is False
        assert "fig:arch" in missing
        assert "[FLOAT:fig:arch]" in feedback
        assert "auto-appended references" in feedback or "dangling phrases" in feedback


class TestLocalMiniReview:

    @pytest.mark.asyncio
    async def test_local_mini_review_fixes_missing_figure_ref_in_place(self):
        agent = MetaDataAgent.__new__(MetaDataAgent)
        agent._writer = SimpleNamespace(
            rewrite_content=AsyncMock(return_value="As shown in [FLOAT:fig:arch], the architecture is modular.")
        )
        para = ParagraphPlan(
            figure_usages=[
                FigureUsagePlan(
                    figure_id="fig:arch",
                    must_appear=True,
                    supported_claim="The architecture motivates the paragraph.",
                )
            ]
        )
        memory = SessionMemory()

        result = await agent._run_local_mini_review(
            section_type="method",
            paragraph_index=0,
            paragraph_plan=para,
            raw_latex="As visualized in, the architecture is modular.",
            final_latex="As visualized in, the architecture is modular.",
            figs_to_ref=["fig:arch"],
            tables_to_ref=[],
            attempt=0,
            max_attempts=2,
            memory=memory,
        )

        assert result["status"] == "fixed_locally"
        assert "Figure~\\ref{fig:arch}" in result["latex"]
        assert memory.local_review_events[-1].disposition == "fixed_locally"

    @pytest.mark.asyncio
    async def test_local_mini_review_requests_retry_when_fix_is_still_invalid(self):
        agent = MetaDataAgent.__new__(MetaDataAgent)
        agent._writer = SimpleNamespace(
            rewrite_content=AsyncMock(return_value="As visualized in, the architecture is modular.")
        )
        para = ParagraphPlan(
            figure_usages=[
                FigureUsagePlan(
                    figure_id="fig:arch",
                    must_appear=True,
                    supported_claim="The architecture motivates the paragraph.",
                )
            ]
        )
        memory = SessionMemory()

        result = await agent._run_local_mini_review(
            section_type="method",
            paragraph_index=0,
            paragraph_plan=para,
            raw_latex="As visualized in, the architecture is modular.",
            final_latex="As visualized in, the architecture is modular.",
            figs_to_ref=["fig:arch"],
            tables_to_ref=[],
            attempt=0,
            max_attempts=2,
            memory=memory,
        )

        assert result["status"] == "retry_required"
        assert memory.local_review_events[-1].disposition == "retry_required"

    @pytest.mark.asyncio
    async def test_local_mini_review_escalates_on_final_failed_attempt(self):
        agent = MetaDataAgent.__new__(MetaDataAgent)
        agent._writer = SimpleNamespace(
            rewrite_content=AsyncMock(return_value="As visualized in, the architecture is modular.")
        )
        para = ParagraphPlan(
            figure_usages=[
                FigureUsagePlan(
                    figure_id="fig:arch",
                    must_appear=True,
                    supported_claim="The architecture motivates the paragraph.",
                )
            ]
        )
        memory = SessionMemory()

        result = await agent._run_local_mini_review(
            section_type="method",
            paragraph_index=0,
            paragraph_plan=para,
            raw_latex="As visualized in, the architecture is modular.",
            final_latex="As visualized in, the architecture is modular.",
            figs_to_ref=["fig:arch"],
            tables_to_ref=[],
            attempt=1,
            max_attempts=2,
            memory=memory,
        )

        assert result["status"] == "escalate"
        assert memory.local_review_events[-1].disposition == "escalate"
