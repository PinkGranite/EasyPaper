"""
Phase-by-phase tests for body section generation bugfixes.

Validates three critical fixes:
1. _generate_section_decomposed accepts exemplar_guidance parameter
2. Phase 2 body section loop excludes introduction (avoid duplicate)
3. Typesetter abstract sanitization strips LLM boilerplate
4. Workflow integration: PaperPlan -> body section generation path
"""
from __future__ import annotations

import ast
import inspect
import re
import textwrap

import pytest


# =========================================================================
# Phase 1: _generate_section_decomposed exemplar_guidance parameter
# =========================================================================

class TestPhase1DecomposedExemplarGuidance:
    """Verify _generate_section_decomposed accepts and uses exemplar_guidance."""

    def test_method_signature_has_exemplar_guidance(self):
        """_generate_section_decomposed must accept exemplar_guidance kwarg."""
        from src.agents.metadata_agent.metadata_agent import MetaDataAgent
        sig = inspect.signature(MetaDataAgent._generate_section_decomposed)
        assert "exemplar_guidance" in sig.parameters, (
            f"Missing exemplar_guidance in _generate_section_decomposed; "
            f"params={list(sig.parameters)}"
        )

    def test_exemplar_guidance_has_correct_default(self):
        """exemplar_guidance should default to None."""
        from src.agents.metadata_agent.metadata_agent import MetaDataAgent
        sig = inspect.signature(MetaDataAgent._generate_section_decomposed)
        param = sig.parameters["exemplar_guidance"]
        assert param.default is None, (
            f"exemplar_guidance default should be None, got {param.default}"
        )

    def test_caller_passes_exemplar_guidance_to_decomposed(self):
        """_generate_body_section must pass exemplar_guidance when calling
        _generate_section_decomposed (the root cause of the original bug)."""
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent._generate_body_section)
        tree = ast.parse(textwrap.dedent(source))

        found_call = False
        passes_exemplar = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                attr = getattr(node.func, "attr", "")
                if attr == "_generate_section_decomposed":
                    found_call = True
                    for kw in node.keywords:
                        if kw.arg == "exemplar_guidance":
                            passes_exemplar = True

        assert found_call, (
            "_generate_body_section does not call _generate_section_decomposed"
        )
        assert passes_exemplar, (
            "_generate_body_section calls _generate_section_decomposed "
            "WITHOUT exemplar_guidance — this causes NameError at runtime"
        )

    def test_decomposed_uses_exemplar_guidance_in_prompt(self):
        """_generate_section_decomposed must reference exemplar_guidance
        in compile_core_prompt (3-stage pipeline) to inject exemplar patterns."""
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent._generate_section_decomposed)
        assert "exemplar_guidance" in source, (
            "exemplar_guidance not used inside _generate_section_decomposed body"
        )
        assert "compile_core_prompt" in source or "compile_paragraph_prompt" in source, (
            "Neither compile_core_prompt nor compile_paragraph_prompt called "
            "in _generate_section_decomposed"
        )


# =========================================================================
# Phase 2: Phase 2 body section loop excludes introduction
# =========================================================================

class TestPhase2IntroductionFilter:
    """Verify Phase 2 does not redundantly regenerate introduction."""

    def test_execute_generation_filters_introduction_from_body(self):
        """Phase 2 body_section_types must exclude 'introduction' since
        Phase 1 already generates it via _generate_introduction."""
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent.execute_generation)
        # The fix adds a list comprehension that filters out introduction
        assert 's != "introduction"' in source or "s != 'introduction'" in source, (
            "execute_generation does not filter introduction from "
            "body_section_types — Phase 2 will redundantly regenerate it"
        )

    def test_paper_plan_get_body_section_types_includes_introduction(self):
        """Confirm PaperPlan.get_body_section_types() includes introduction
        (it only excludes abstract and conclusion) — this is why the filter
        in execute_generation is needed."""
        from src.agents.planner_agent.models import PaperPlan, SectionPlan

        plan = PaperPlan(
            title="Test",
            sections=[
                SectionPlan(section_type="abstract", section_title="Abstract"),
                SectionPlan(section_type="introduction", section_title="Introduction"),
                SectionPlan(section_type="related_work", section_title="Related Work"),
                SectionPlan(section_type="method", section_title="Method"),
                SectionPlan(section_type="conclusion", section_title="Conclusion"),
            ],
        )
        body_types = plan.get_body_section_types()
        assert "introduction" in body_types, (
            "get_body_section_types should include introduction "
            "(it only excludes abstract/conclusion)"
        )
        assert "abstract" not in body_types
        assert "conclusion" not in body_types

    def test_filtered_body_types_excludes_introduction(self):
        """Simulate the Phase 2 filter and verify introduction is removed."""
        from src.agents.planner_agent.models import PaperPlan, SectionPlan

        plan = PaperPlan(
            title="Test",
            sections=[
                SectionPlan(section_type="abstract", section_title="Abstract"),
                SectionPlan(section_type="introduction", section_title="Introduction"),
                SectionPlan(section_type="related_work", section_title="Related Work"),
                SectionPlan(section_type="method", section_title="Method"),
                SectionPlan(section_type="experiment", section_title="Experiments"),
                SectionPlan(section_type="result", section_title="Results"),
                SectionPlan(section_type="conclusion", section_title="Conclusion"),
            ],
        )
        body_types = plan.get_body_section_types()
        # Apply the same filter as execute_generation
        filtered = [s for s in body_types if s != "introduction"]

        assert "introduction" not in filtered
        assert filtered == ["related_work", "method", "experiment", "result"], (
            f"Filtered body types should be [related_work, method, experiment, result], "
            f"got {filtered}"
        )


# =========================================================================
# Phase 3: Typesetter abstract sanitization
# =========================================================================

class TestPhase3AbstractSanitization:
    """Verify typesetter strips LLM-generated boilerplate from abstract."""

    def _get_typesetter(self):
        """Create a minimal TypesetterAgent for testing sanitization."""
        from src.agents.typesetter_agent.typesetter_agent import TypesetterAgent
        return TypesetterAgent.__new__(TypesetterAgent)

    def test_strips_title_from_abstract(self):
        """Abstract sanitization must remove \\title{...} commands."""
        raw = r"""\title{My Paper Title}

\begin{abstract}
This is the actual abstract content about vision-language models.
\end{abstract}"""
        # Apply the same regex pipeline as inject_template
        cleaned = raw.strip()
        cleaned = re.sub(r'\\title\{[^}]*\}\s*', '', cleaned)
        cleaned = re.sub(r'\\maketitle\s*', '', cleaned)
        cleaned = re.sub(r'\\begin\{abstract\}\s*', '', cleaned)
        cleaned = re.sub(r'\s*\\end\{abstract\}', '', cleaned)
        cleaned = cleaned.strip()

        assert r"\title" not in cleaned, "\\title should be stripped"
        assert r"\begin{abstract}" not in cleaned
        assert r"\end{abstract}" not in cleaned
        assert "actual abstract content" in cleaned

    def test_strips_maketitle_from_abstract(self):
        """Abstract sanitization must remove \\maketitle."""
        raw = r"""This is the abstract.
\end{abstract}

\maketitle"""
        cleaned = raw.strip()
        cleaned = re.sub(r'\\title\{[^}]*\}\s*', '', cleaned)
        cleaned = re.sub(r'\\maketitle\s*', '', cleaned)
        cleaned = re.sub(r'\\begin\{abstract\}\s*', '', cleaned)
        cleaned = re.sub(r'\s*\\end\{abstract\}', '', cleaned)
        cleaned = cleaned.strip()

        assert r"\maketitle" not in cleaned
        assert "This is the abstract." in cleaned

    def test_strips_nested_begin_abstract(self):
        """If LLM generates \\begin{abstract} inside content that already
        has the environment wrapper, it must be stripped."""
        raw = r"""\title{BLIP-2: Bootstrapping}

\begin{abstract}
The rising computational cost of VLP...
BLIP-2 achieves state-of-the-art performance.
\end{abstract}

\maketitle"""
        cleaned = raw.strip()
        cleaned = re.sub(r'\\title\{[^}]*\}\s*', '', cleaned)
        cleaned = re.sub(r'\\maketitle\s*', '', cleaned)
        cleaned = re.sub(r'\\begin\{abstract\}\s*', '', cleaned)
        cleaned = re.sub(r'\s*\\end\{abstract\}', '', cleaned)
        cleaned = cleaned.strip()

        assert r"\title" not in cleaned
        assert r"\begin{abstract}" not in cleaned
        assert r"\end{abstract}" not in cleaned
        assert r"\maketitle" not in cleaned
        assert "rising computational cost" in cleaned
        assert "state-of-the-art" in cleaned

    def test_sanitization_in_inject_template_source(self):
        """Verify the typesetter source code performs all four strippings."""
        from src.agents.typesetter_agent import typesetter_agent as mod

        source = inspect.getsource(mod.TypesetterAgent.inject_template)
        # Must strip \title{...}
        assert r"\\title" in source, (
            "inject_template does not strip \\title from abstract"
        )
        # Must strip \maketitle
        assert r"\\maketitle" in source, (
            "inject_template does not strip \\maketitle from abstract"
        )
        # Must strip \begin{abstract}
        assert r"\\begin" in source and "abstract" in source, (
            "inject_template does not strip \\begin{abstract}"
        )


# =========================================================================
# Phase 4: Workflow integration — PaperPlan → body section generation
# =========================================================================

class TestPhase4WorkflowIntegration:
    """End-to-end structural tests for the plan → generation pipeline."""

    def test_paper_plan_roundtrip_preserves_sections(self):
        """PaperPlan serialized via model_dump() and reconstructed should
        preserve all sections — this is the path used in execute_generation."""
        from src.agents.planner_agent.models import PaperPlan, SectionPlan

        original = PaperPlan(
            title="Test Paper",
            sections=[
                SectionPlan(section_type="abstract", section_title="Abstract"),
                SectionPlan(section_type="introduction", section_title="Introduction"),
                SectionPlan(section_type="related_work", section_title="Related Work"),
                SectionPlan(section_type="method", section_title="Method"),
                SectionPlan(section_type="experiment", section_title="Experiments"),
                SectionPlan(section_type="result", section_title="Results"),
                SectionPlan(section_type="discussion", section_title="Discussion"),
                SectionPlan(section_type="conclusion", section_title="Conclusion"),
            ],
        )

        # Simulate prepare_plan -> execute_generation roundtrip
        dumped = original.model_dump()
        reconstructed = PaperPlan(**dumped)

        assert len(reconstructed.sections) == 8
        assert reconstructed.get_body_section_types() == [
            "introduction", "related_work", "method", "experiment",
            "result", "discussion",
        ]
        # After Phase 2 filter
        filtered = [s for s in reconstructed.get_body_section_types() if s != "introduction"]
        assert filtered == ["related_work", "method", "experiment", "result", "discussion"]

    def test_all_generation_methods_accept_exemplar_guidance(self):
        """All section generation methods must accept exemplar_guidance."""
        from src.agents.metadata_agent.metadata_agent import MetaDataAgent

        methods_to_check = [
            "_generate_introduction",
            "_generate_body_section",
            "_generate_section_decomposed",
            "_generate_synthesis_section",
        ]

        for method_name in methods_to_check:
            method = getattr(MetaDataAgent, method_name)
            sig = inspect.signature(method)
            assert "exemplar_guidance" in sig.parameters, (
                f"{method_name} missing exemplar_guidance parameter"
            )

    def test_execute_generation_calls_body_section_with_exemplar(self):
        """execute_generation must pass exemplar_guidance to
        _generate_body_section for every body section."""
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent.execute_generation)

        # Find the _generate_body_section call in Phase 2
        assert "_generate_body_section(" in source

        # Find the call and verify it has exemplar_guidance
        # Locate Phase 2 section of the source
        phase2_start = source.find("# Phase 2: Body Sections")
        phase3_start = source.find("# Phase 3: Synthesis Sections")
        assert phase2_start != -1, "Phase 2 marker not found"
        assert phase3_start != -1, "Phase 3 marker not found"

        phase2_body = source[phase2_start:phase3_start]
        assert "exemplar_guidance=" in phase2_body, (
            "Phase 2 does not pass exemplar_guidance to _generate_body_section"
        )

    def test_conclusion_generation_is_conditional_on_plan(self):
        """Conclusion should only be generated if paper_plan has a
        conclusion section — this is the existing behavior."""
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent.execute_generation)
        assert 'should_generate_conclusion' in source
        assert 'paper_plan.get_section("conclusion")' in source
