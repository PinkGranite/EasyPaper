"""
Tests for smart caption normalization, LLM-based element allocation,
and decomposed writer pipeline (3 stages).
"""
import re
import json
import pytest
from types import SimpleNamespace
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch


# ============================= FEATURE 1 ===================================
# normalize_caption: strip redundant Table/Figure numbering prefixes
# ===========================================================================

class TestNormalizeCaption:

    def test_strip_table_prefix(self):
        from src.agents.shared.table_converter import normalize_caption
        assert normalize_caption("Table 1. Overview of results") == "Overview of results"

    def test_strip_figure_prefix(self):
        from src.agents.shared.table_converter import normalize_caption
        assert normalize_caption("Figure 2. (Left) Model architecture") == "(Left) Model architecture"

    def test_strip_table_prefix_with_colon(self):
        from src.agents.shared.table_converter import normalize_caption
        assert normalize_caption("Table 3: Ablation study") == "Ablation study"

    def test_strip_tab_abbreviation(self):
        from src.agents.shared.table_converter import normalize_caption
        assert normalize_caption("Tab. 4. Comparison") == "Comparison"

    def test_strip_fig_abbreviation(self):
        from src.agents.shared.table_converter import normalize_caption
        assert normalize_caption("Fig. 1. Architecture overview") == "Architecture overview"

    def test_strip_uppercase(self):
        from src.agents.shared.table_converter import normalize_caption
        assert normalize_caption("TABLE 5. Performance") == "Performance"
        assert normalize_caption("FIGURE 3. Results") == "Results"

    def test_no_prefix_unchanged(self):
        from src.agents.shared.table_converter import normalize_caption
        assert normalize_caption("Comparison of BLIP-2 results") == "Comparison of BLIP-2 results"

    def test_empty_string(self):
        from src.agents.shared.table_converter import normalize_caption
        assert normalize_caption("") == ""

    def test_only_prefix(self):
        from src.agents.shared.table_converter import normalize_caption
        result = normalize_caption("Table 1.")
        assert result == "" or result == "Table 1."

    def test_preserves_inner_table_word(self):
        from src.agents.shared.table_converter import normalize_caption
        assert normalize_caption("Comparison table of results") == "Comparison table of results"

    def test_multidigit_number(self):
        from src.agents.shared.table_converter import normalize_caption
        assert normalize_caption("Table 12. Large benchmark") == "Large benchmark"

    def test_figure_parenthetical(self):
        from src.agents.shared.table_converter import normalize_caption
        result = normalize_caption("Figure 5. Effect of vision-language")
        assert result == "Effect of vision-language"


# ============================= FEATURE 2 ===================================
# LLM-based figure/table section allocation
# ===========================================================================

def _make_section_plan(section_type, title, key_points=None):
    paragraphs = []
    for kp in (key_points or []):
        paragraphs.append(SimpleNamespace(key_point=kp))
    return SimpleNamespace(
        section_type=section_type,
        section_title=title,
        paragraphs=paragraphs,
        figures=[],
        tables=[],
        figures_to_reference=[],
        tables_to_reference=[],
    )


def _make_element_info(id, caption, description="", section=""):
    return SimpleNamespace(
        id=id, caption=caption, description=description,
        section=section, wide=False, file_path=None,
    )


class TestLLMAssignElements:

    def test_parses_valid_llm_response(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent

        sections = [
            _make_section_plan("method", "Methodology", ["Describe the Q-Former"]),
            _make_section_plan("result", "Experimental Results", ["Report performance"]),
        ]
        plan = SimpleNamespace(sections=sections, wide_figures=[], wide_tables=[])

        elements = {
            "fig:arch": _make_element_info("fig:arch", "Architecture overview"),
            "tab:results": _make_element_info("tab:results", "Performance comparison"),
        }

        llm_response = json.dumps({
            "fig:arch": "method",
            "tab:results": "result",
        })

        result = PlannerAgent._parse_element_assignment(
            llm_response, elements, plan,
        )
        assert result["fig:arch"] == "method"
        assert result["tab:results"] == "result"

    def test_fallback_on_invalid_section_type(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent

        sections = [
            _make_section_plan("method", "Methodology"),
            _make_section_plan("result", "Results"),
        ]
        plan = SimpleNamespace(sections=sections, wide_figures=[], wide_tables=[])

        elements = {
            "tab:x": _make_element_info("tab:x", "Some table"),
        }

        llm_response = json.dumps({"tab:x": "nonexistent_section"})
        result = PlannerAgent._parse_element_assignment(
            llm_response, elements, plan,
        )
        assert result["tab:x"] in ("method", "result")

    def test_fallback_on_malformed_json(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent

        sections = [
            _make_section_plan("result", "Results"),
        ]
        plan = SimpleNamespace(sections=sections, wide_figures=[], wide_tables=[])
        elements = {"tab:a": _make_element_info("tab:a", "A")}

        result = PlannerAgent._parse_element_assignment(
            "not valid json", elements, plan,
        )
        assert "tab:a" in result

    def test_capacity_limit(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent

        sections = [
            _make_section_plan("result", "Results"),
            _make_section_plan("analysis", "Analysis"),
        ]
        plan = SimpleNamespace(sections=sections, wide_figures=[], wide_tables=[])

        elements = {f"tab:{i}": _make_element_info(f"tab:{i}", f"Table {i}") for i in range(6)}
        llm_response = json.dumps({f"tab:{i}": "result" for i in range(6)})

        result = PlannerAgent._parse_element_assignment(
            llm_response, elements, plan, max_per_section=3,
        )
        result_count = sum(1 for v in result.values() if v == "result")
        assert result_count <= 3

    def test_all_elements_assigned(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent

        sections = [
            _make_section_plan("intro", "Introduction"),
            _make_section_plan("method", "Method"),
        ]
        plan = SimpleNamespace(sections=sections, wide_figures=[], wide_tables=[])

        elements = {
            "fig:a": _make_element_info("fig:a", "Figure A"),
            "tab:b": _make_element_info("tab:b", "Table B"),
        }
        llm_response = json.dumps({"fig:a": "intro", "tab:b": "method"})

        result = PlannerAgent._parse_element_assignment(
            llm_response, elements, plan,
        )
        assert set(result.keys()) == {"fig:a", "tab:b"}

    def test_build_assignment_prompt(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent

        sections = [
            _make_section_plan("method", "Methodology", ["Describe architecture"]),
            _make_section_plan("result", "Results", ["Show performance"]),
        ]
        plan = SimpleNamespace(sections=sections)

        figures = {"fig:arch": _make_element_info("fig:arch", "Architecture diagram")}
        tables = {"tab:perf": _make_element_info("tab:perf", "Performance comparison")}

        prompt = PlannerAgent._build_assignment_prompt(plan, figures, tables)
        assert "method" in prompt
        assert "Methodology" in prompt
        assert "Describe architecture" in prompt
        assert "Show performance" in prompt
        assert "fig:arch" in prompt
        assert "tab:perf" in prompt


# ============================= FEATURE 3 STAGE 1 ===========================
# Core content writing: no citations, CITE/FLOAT markers
# ===========================================================================

class TestCompileCorePrompt:

    def test_excludes_citation_keys(self):
        from src.agents.shared.prompt_compiler import compile_core_prompt

        para = SimpleNamespace(
            key_point="BLIP-2 outperforms Flamingo",
            supporting_points=["8.7% improvement on VQAv2"],
            role="evidence",
            sentence_plans=[],
            approx_sentences=4,
            effective_sentence_count=4,
            references_to_cite=["alayrac2022"],
            figures_to_reference=["fig:arch"],
            tables_to_reference=["tab:results"],
        )
        result = compile_core_prompt(
            paragraph_plan=para,
            section_type="result",
            section_context="Previous paragraph...",
            evidence_snippets=["BLIP-2 achieves 65.0% on VQAv2"],
            section_title="Results",
            paragraph_index=0,
            total_paragraphs=3,
        )
        assert "Valid Citation Keys" not in result
        assert "alayrac2022" not in result
        assert "BLIP-2 outperforms Flamingo" in result
        assert "CITE" in result
        assert "FLOAT" in result

    def test_includes_float_markers_instruction(self):
        from src.agents.shared.prompt_compiler import compile_core_prompt

        para = SimpleNamespace(
            key_point="Results are strong",
            supporting_points=[],
            role="evidence",
            sentence_plans=[],
            approx_sentences=3,
            effective_sentence_count=3,
            references_to_cite=[],
            figures_to_reference=["fig:overview"],
            tables_to_reference=["tab:main"],
        )
        result = compile_core_prompt(
            paragraph_plan=para,
            section_type="result",
            evidence_snippets=[],
            section_title="Results",
        )
        assert "FLOAT" in result or "float" in result.lower()

    def test_includes_evidence_snippets(self):
        from src.agents.shared.prompt_compiler import compile_core_prompt

        para = SimpleNamespace(
            key_point="Test",
            supporting_points=[],
            role="evidence",
            sentence_plans=[],
            approx_sentences=3,
            effective_sentence_count=3,
            references_to_cite=[],
            figures_to_reference=[],
            tables_to_reference=[],
        )
        result = compile_core_prompt(
            paragraph_plan=para,
            section_type="method",
            evidence_snippets=["Evidence about Q-Former architecture"],
            section_title="Method",
        )
        assert "Q-Former" in result


# ============================= FEATURE 3 STAGE 2 ===========================
# Citation injection: LLM-based Method A
# ===========================================================================

class TestCitationModels:

    def test_citation_action_model(self):
        from src.agents.writer_agent.models import CitationAction

        action = CitationAction(
            action="replace_marker",
            marker_or_location="[CITE:contrastive_learning]",
            new_text="contrastive learning \\cite{radford2021clip}",
            cite_keys=["radford2021clip"],
        )
        assert action.action == "replace_marker"
        assert action.cite_keys == ["radford2021clip"]

    def test_citation_edit_result_model(self):
        from src.agents.writer_agent.models import CitationEditResult, CitationAction

        result = CitationEditResult(
            actions=[
                CitationAction(
                    action="replace_marker",
                    marker_or_location="[CITE:x]",
                    new_text="text \\cite{a}",
                    cite_keys=["a"],
                )
            ],
            raw_response="...",
        )
        assert len(result.actions) == 1


class TestApplyCitationEdits:

    def test_replace_marker(self):
        from src.agents.shared.prompt_compiler import apply_citation_edits
        from src.agents.writer_agent.models import CitationAction

        latex = "Vision-language models [CITE:vlm] have improved significantly."
        actions = [
            CitationAction(
                action="replace_marker",
                marker_or_location="[CITE:vlm]",
                new_text="\\cite{radford2021clip,jia2021align}",
                cite_keys=["radford2021clip", "jia2021align"],
            )
        ]
        result = apply_citation_edits(latex, actions, valid_keys={"radford2021clip", "jia2021align"})
        assert "[CITE:" not in result
        assert "\\cite{radford2021clip" in result

    def test_strips_invalid_keys(self):
        from src.agents.shared.prompt_compiler import apply_citation_edits
        from src.agents.writer_agent.models import CitationAction

        latex = "Models [CITE:x] work well."
        actions = [
            CitationAction(
                action="replace_marker",
                marker_or_location="[CITE:x]",
                new_text="\\cite{valid_key,fake_key}",
                cite_keys=["valid_key", "fake_key"],
            )
        ]
        result = apply_citation_edits(latex, actions, valid_keys={"valid_key"})
        assert "valid_key" in result
        assert "fake_key" not in result

    def test_insert_sentence(self):
        from src.agents.shared.prompt_compiler import apply_citation_edits
        from src.agents.writer_agent.models import CitationAction

        latex = "First sentence. Second sentence."
        actions = [
            CitationAction(
                action="insert_sentence",
                marker_or_location="after_sentence:1",
                new_text="Recent work \\cite{new2024} extends this.",
                cite_keys=["new2024"],
            )
        ]
        result = apply_citation_edits(latex, actions, valid_keys={"new2024"})
        assert "Recent work" in result
        assert result.index("Recent work") > result.index("First sentence")

    def test_leftover_markers_cleaned(self):
        from src.agents.shared.prompt_compiler import apply_citation_edits

        latex = "Some text [CITE:orphan] more text."
        result = apply_citation_edits(latex, [], valid_keys=set())
        assert "[CITE:" not in result


# ============================= FEATURE 3 STAGE 3 ===========================
# Float reference injection: mechanical marker replacement
# ===========================================================================

class TestInjectFloatRefs:

    def test_replace_table_marker(self):
        from src.agents.shared.table_converter import inject_float_refs

        latex = "Results in [FLOAT:tab:results] demonstrate improvements."
        result = inject_float_refs(latex, [], ["tab:results"])
        assert "Table~\\ref{tab:results}" in result
        assert "[FLOAT:" not in result

    def test_replace_figure_marker(self):
        from src.agents.shared.table_converter import inject_float_refs

        latex = "As shown in [FLOAT:fig:arch], the architecture is modular."
        result = inject_float_refs(latex, ["fig:arch"], [])
        assert "Figure~\\ref{fig:arch}" in result
        assert "[FLOAT:" not in result

    def test_multiple_markers(self):
        from src.agents.shared.table_converter import inject_float_refs

        latex = "[FLOAT:fig:a] shows architecture. [FLOAT:tab:b] shows results."
        result = inject_float_refs(latex, ["fig:a"], ["tab:b"])
        assert "Figure~\\ref{fig:a}" in result
        assert "Table~\\ref{tab:b}" in result

    def test_no_markers_no_change(self):
        from src.agents.shared.table_converter import inject_float_refs

        latex = "Plain text without markers."
        result = inject_float_refs(latex, [], [])
        assert result == latex

    def test_cleans_orphan_markers(self):
        from src.agents.shared.table_converter import inject_float_refs

        latex = "Text with [FLOAT:unknown_id] orphan."
        result = inject_float_refs(latex, [], [])
        assert "[FLOAT:" not in result


# ============================= INTEGRATION ==================================
# Full pipeline: core -> cite -> float -> verify
# ===========================================================================

class TestCompileCitationPrompt:

    def test_includes_raw_latex_and_refs(self):
        from src.agents.shared.prompt_compiler import compile_citation_prompt

        refs = [
            {"id": "smith2024", "title": "Deep Learning for Vision", "abstract": "We propose..."},
            {"id": "jones2023", "title": "Contrastive Methods", "abstract": "A survey of..."},
        ]
        raw_latex = "Models [CITE:deep_learning] have advanced. Contrastive learning [CITE:contrastive] is key."

        result = compile_citation_prompt(
            raw_latex=raw_latex,
            assigned_refs=refs,
            section_type="related_work",
        )
        assert "smith2024" in result
        assert "jones2023" in result
        assert "[CITE:deep_learning]" in result
        assert "JSON" in result or "json" in result


# ============================= PHASE 4: FIGURE WIDTH =========================
# ===========================================================================


class TestFigureWidthDecision:
    """PlannerAgent._should_be_wide_figure: VLM, aspect ratio, keyword fallback."""

    def _open_context(self, size):
        ctx = MagicMock()
        ctx.__enter__.return_value = MagicMock(size=size)
        ctx.__exit__.return_value = None
        return ctx

    def test_explicit_wide_flag_respected(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent

        fig = SimpleNamespace(
            id="f1", caption="", description="", wide=True, file_path="/any.png",
        )
        assert PlannerAgent._should_be_wide_figure(fig, None) is True

    def test_vlm_is_wide_overrides_aspect_ratio(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent

        fig = SimpleNamespace(
            id="f1", caption="", description="", wide=False, file_path="/x.png",
        )
        vlm = SimpleNamespace(is_wide=True)
        with patch("PIL.Image.open", side_effect=AssertionError("should not open")):
            assert PlannerAgent._should_be_wide_figure(fig, vlm) is True

    def test_vlm_false_returns_false_without_pil(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent

        fig = SimpleNamespace(
            id="f1", caption="architecture overview", description="",
            wide=False, file_path="/x.png",
        )
        vlm = SimpleNamespace(is_wide=False)
        with patch("PIL.Image.open", side_effect=AssertionError("should not open")):
            assert PlannerAgent._should_be_wide_figure(fig, vlm) is False

    def test_wide_aspect_ratio_returns_true(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent

        fig = SimpleNamespace(
            id="f1", caption="", description="", wide=False, file_path="/w.png",
        )
        with patch(
            "PIL.Image.open",
            return_value=self._open_context((1200, 400)),
        ):
            assert PlannerAgent._should_be_wide_figure(fig, None) is True

    def test_tall_aspect_ratio_returns_false(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent

        fig = SimpleNamespace(
            id="f1", caption="", description="", wide=False, file_path="/t.png",
        )
        with patch(
            "PIL.Image.open",
            return_value=self._open_context((400, 800)),
        ):
            assert PlannerAgent._should_be_wide_figure(fig, None) is False

    def test_square_aspect_ratio_uses_keywords(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent

        fig_wide_kw = SimpleNamespace(
            id="f1", caption="Model architecture diagram", description="",
            wide=False, file_path="/s.png",
        )
        fig_narrow_kw = SimpleNamespace(
            id="f2", caption="Simple accuracy plot", description="",
            wide=False, file_path="/s2.png",
        )
        with patch(
            "PIL.Image.open",
            return_value=self._open_context((600, 600)),
        ):
            assert PlannerAgent._should_be_wide_figure(fig_wide_kw, None) is True
            assert PlannerAgent._should_be_wide_figure(fig_narrow_kw, None) is False

    def test_no_file_path_uses_keywords(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent

        fig = SimpleNamespace(
            id="f1", caption="Full pipeline overview", description="",
            wide=False, file_path="",
        )
        assert PlannerAgent._should_be_wide_figure(fig, None) is True


# ============================= PHASE 2: LLM ALLOCATION =========================
# ===========================================================================


class TestLLMAllocationIntegration:
    """Async _assign_figure_table_definitions uses LLM + semantic prompt context."""

    @pytest.mark.asyncio
    async def test_assign_uses_llm_not_keywords(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent
        from src.agents.planner_agent.models import (
            FigureInfo,
            PaperPlan,
            PaperType,
            ParagraphPlan,
            PlanRequest,
            SectionPlan,
            TableInfo,
        )

        cfg = MagicMock()
        cfg.model_name = "m"
        cfg.api_key = "k"
        cfg.base_url = "http://127.0.0.1"
        agent = PlannerAgent(cfg, vlm_service=None)
        agent._llm_json_call = AsyncMock(
            return_value={"fig:a": "method", "tab:b": "result"},
        )

        sections = [
            SectionPlan(
                section_type="introduction",
                section_title="Intro",
                paragraphs=[ParagraphPlan(key_point="Motivation")],
            ),
            SectionPlan(
                section_type="method",
                section_title="Method",
                paragraphs=[ParagraphPlan(key_point="Describe model")],
            ),
            SectionPlan(
                section_type="result",
                section_title="Results",
                paragraphs=[ParagraphPlan(key_point="Show numbers")],
            ),
        ]
        plan = PaperPlan(
            title="T",
            paper_type=PaperType.EMPIRICAL,
            sections=sections,
            contributions=["c"],
        )
        request = PlanRequest(
            idea_hypothesis="i",
            method="m",
            data="d",
            experiments="e",
            figures=[FigureInfo(id="fig:a", caption="Q-Former architecture")],
            tables=[TableInfo(id="tab:b", caption="Zero-shot VQA accuracy")],
        )

        await agent._assign_figure_table_definitions(plan, request, {}, {})

        method_sec = next(s for s in plan.sections if s.section_type == "method")
        result_sec = next(s for s in plan.sections if s.section_type == "result")
        assert any(f.figure_id == "fig:a" for f in method_sec.figures)
        assert any(t.table_id == "tab:b" for t in result_sec.tables)
        agent._llm_json_call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_assign_distributes_tables_across_sections(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent
        from src.agents.planner_agent.models import (
            PaperPlan,
            PaperType,
            ParagraphPlan,
            PlanRequest,
            SectionPlan,
            TableInfo,
        )

        cfg = MagicMock()
        cfg.model_name = "m"
        cfg.api_key = "k"
        cfg.base_url = "http://127.0.0.1"
        agent = PlannerAgent(cfg, vlm_service=None)
        agent._llm_json_call = AsyncMock(
            return_value={
                "tab:main": "result",
                "tab:setup": "experiment_setup",
            },
        )

        sections = [
            SectionPlan(
                section_type="introduction",
                section_title="Intro",
                paragraphs=[ParagraphPlan(key_point="Overview")],
            ),
            SectionPlan(
                section_type="experiment_setup",
                section_title="Experimental setup",
                paragraphs=[ParagraphPlan(key_point="Datasets and baselines")],
            ),
            SectionPlan(
                section_type="result",
                section_title="Results",
                paragraphs=[ParagraphPlan(key_point="Main metrics")],
            ),
        ]
        plan = PaperPlan(
            title="T",
            paper_type=PaperType.EMPIRICAL,
            sections=sections,
            contributions=["c"],
        )
        request = PlanRequest(
            idea_hypothesis="i",
            method="m",
            data="d",
            experiments="e",
            tables=[
                TableInfo(id="tab:main", caption="Main comparison"),
                TableInfo(id="tab:setup", caption="Training hyperparameters"),
            ],
        )

        await agent._assign_figure_table_definitions(plan, request, {}, {})

        res = next(s for s in plan.sections if s.section_type == "result")
        exp = next(s for s in plan.sections if s.section_type == "experiment_setup")
        assert any(t.table_id == "tab:main" for t in res.tables)
        assert any(t.table_id == "tab:setup" for t in exp.tables)

    def test_assign_prompt_includes_vlm_and_normalized_caption(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent

        sections = [
            _make_section_plan("result", "Results", ["Main scores"]),
        ]
        plan = SimpleNamespace(sections=sections)
        fig = _make_element_info("fig:x", "Figure 2. Ablations")
        vlm = SimpleNamespace(
            semantic_role="ablation_study",
            message="Shows component contributions",
            suggested_section="result",
        )
        prompt = PlannerAgent._build_assignment_prompt(
            plan, {"fig:x": fig}, {}, figure_analyses={"fig:x": vlm},
        )
        assert "Main scores" in prompt
        assert "Figure 2." not in prompt
        assert "ablation_study" in prompt
        assert "vlm_summary=" in prompt

    @pytest.mark.asyncio
    async def test_assign_fallback_on_llm_failure(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent
        from src.agents.planner_agent.models import (
            PaperPlan,
            PaperType,
            ParagraphPlan,
            PlanRequest,
            SectionPlan,
            TableInfo,
        )

        cfg = MagicMock()
        cfg.model_name = "m"
        cfg.api_key = "k"
        cfg.base_url = "http://127.0.0.1"
        agent = PlannerAgent(cfg, vlm_service=None)
        agent._llm_json_call = AsyncMock(side_effect=RuntimeError("api down"))

        sections = [
            SectionPlan(
                section_type="introduction",
                section_title="Intro",
                paragraphs=[ParagraphPlan(key_point="Overview")],
            ),
            SectionPlan(
                section_type="result",
                section_title="Results",
                paragraphs=[ParagraphPlan(key_point="Numbers")],
            ),
        ]
        plan = PaperPlan(
            title="T",
            paper_type=PaperType.EMPIRICAL,
            sections=sections,
            contributions=["c"],
        )
        request = PlanRequest(
            idea_hypothesis="i",
            method="m",
            data="d",
            experiments="e",
            tables=[TableInfo(id="tab:z", caption="Performance on benchmark")],
        )

        await agent._assign_figure_table_definitions(plan, request, {}, {})

        placed = []
        for sec in plan.sections:
            placed.extend(t.table_id for t in sec.tables)
        assert "tab:z" in placed

    @pytest.mark.asyncio
    async def test_assign_respects_capacity(self):
        from src.agents.planner_agent.planner_agent import PlannerAgent
        from src.agents.planner_agent.models import (
            PaperPlan,
            PaperType,
            ParagraphPlan,
            PlanRequest,
            SectionPlan,
            TableInfo,
        )

        cfg = MagicMock()
        cfg.model_name = "m"
        cfg.api_key = "k"
        cfg.base_url = "http://127.0.0.1"
        agent = PlannerAgent(cfg, vlm_service=None)
        agent._llm_json_call = AsyncMock(
            return_value={f"tab:{i}": "result" for i in range(6)},
        )

        sections = [
            SectionPlan(
                section_type="introduction",
                section_title="Intro",
                paragraphs=[ParagraphPlan(key_point="I")],
            ),
            SectionPlan(
                section_type="method",
                section_title="Method",
                paragraphs=[ParagraphPlan(key_point="M")],
            ),
            SectionPlan(
                section_type="result",
                section_title="Results",
                paragraphs=[ParagraphPlan(key_point="R")],
            ),
            SectionPlan(
                section_type="analysis",
                section_title="Analysis",
                paragraphs=[ParagraphPlan(key_point="A")],
            ),
        ]
        plan = PaperPlan(
            title="T",
            paper_type=PaperType.EMPIRICAL,
            sections=sections,
            contributions=["c"],
        )
        tables = [
            TableInfo(id=f"tab:{i}", caption=f"Table {i}")
            for i in range(6)
        ]
        request = PlanRequest(
            idea_hypothesis="i",
            method="m",
            data="d",
            experiments="e",
            tables=tables,
        )

        await agent._assign_figure_table_definitions(plan, request, {}, {})

        result_sec = next(s for s in plan.sections if s.section_type == "result")
        assert len(result_sec.tables) <= 3


# ============================= PHASE 1: PROMPT CAPTION NORMALIZATION =========
# ===========================================================================


class TestCaptionNormInPrompts:
    """Prompts must not echo 'Figure N.' / 'Table N.' into model context (duplicate numbering)."""

    def test_compile_section_prompt_normalizes_captions(self):
        from src.agents.shared.prompt_compiler import compile_section_prompt

        fig = SimpleNamespace(id="fig:arch", caption="Figure 2. Architecture overview")
        tbl = SimpleNamespace(id="tab:main", caption="Table 3. Main results")
        out = compile_section_prompt(
            "method",
            thesis="t",
            figures=[fig],
            tables=[tbl],
        )
        assert "Figure 2." not in out
        assert "Architecture overview" in out
        assert "Table 3." not in out
        assert "Main results" in out

    def test_compile_introduction_prompt_normalizes_captions(self):
        from src.agents.shared.prompt_compiler import compile_introduction_prompt

        fig = SimpleNamespace(id="fig:q", caption="Figure 2. (Left) Q-Former")
        tbl = SimpleNamespace(id="tab:x", caption="Table 1. Comparison")
        out = compile_introduction_prompt(
            paper_title="T",
            idea_hypothesis="i",
            method_summary="m",
            data_summary="d",
            experiments_summary="e",
            section_plan=None,
            figures=[fig],
            tables=[tbl],
        )
        assert "Figure 2." not in out
        assert "(Left) Q-Former" in out
        assert "Table 1." not in out
        assert "Comparison" in out

    def test_compile_body_section_prompt_normalizes_captions(self):
        from src.agents.shared.prompt_compiler import compile_body_section_prompt

        fig = SimpleNamespace(id="fig:a", caption="Figure 4. Examples")
        tbl = SimpleNamespace(id="tab:b", caption="Table 5. Stats")
        plan = SimpleNamespace(
            section_type="result",
            section_title="Results",
            paragraphs=[],
            figures=[],
            tables=[],
            figures_to_reference=["fig:a"],
            tables_to_reference=["tab:b"],
            assigned_refs=[],
        )
        out = compile_body_section_prompt(
            section_type="result",
            metadata_content="meta",
            intro_context="intro",
            section_plan=plan,
            figures=[fig],
            tables=[tbl],
        )
        assert "Figure 4." not in out
        assert "Examples" in out
        assert "Table 5." not in out
        assert "Stats" in out

    def test_compile_body_section_prompt_legacy_fallback_normalizes(self):
        from src.agents.shared.prompt_compiler import compile_body_section_prompt

        fig = SimpleNamespace(id="fig:a", caption="Figure 2. Wide diagram")
        tbl = SimpleNamespace(id="tab:b", caption="Table 2. Data")
        out = compile_body_section_prompt(
            section_type="method",
            metadata_content="m",
            intro_context="i",
            section_plan=None,
            figures=[fig],
            tables=[tbl],
        )
        assert "Figure 2." not in out
        assert "Wide diagram" in out
        assert "Table 2." not in out
        assert "Data" in out


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Subsection support — planner, prompt compiler, decomposed gen
# ═══════════════════════════════════════════════════════════════════════════


class TestSubsectionTriggering:
    """Planner should auto-enable subsections when paragraphs >= 5 and clusters >= 2."""

    def test_auto_enable_sectioning_for_long_sections(self):
        """Subsection decisions are now LLM-driven via _decide_section_structure (Step 4)
        instead of a heuristic based on topic_clusters. Verify the new method exists
        and that sectioning_recommended is set in the pipeline."""
        from src.agents.planner_agent.planner_agent import PlannerAgent
        import inspect

        source = inspect.getsource(PlannerAgent.create_plan)
        assert "sectioning_recommended" in source
        assert "_decide_section_structure" in source


class TestSubsectionPromptFormatting:
    """Prompt compiler must format subsections with grouped headings."""

    def test_paragraph_guidance_groups_by_subsection(self):
        """When subsections exist, _format_paragraph_guidance must include
        subsection titles and group paragraphs under them."""
        from src.agents.shared.prompt_compiler import _format_paragraph_guidance
        from src.agents.planner_agent.models import (
            SectionPlan, SubSectionPlan, ParagraphPlan,
        )

        plan = SectionPlan(
            section_type="method",
            section_title="Method",
            subsections=[
                SubSectionPlan(
                    title="Architecture",
                    paragraphs=[
                        ParagraphPlan(key_point="Transformer design"),
                        ParagraphPlan(key_point="Attention mechanism"),
                    ],
                ),
                SubSectionPlan(
                    title="Training",
                    paragraphs=[
                        ParagraphPlan(key_point="Pre-training"),
                        ParagraphPlan(key_point="Fine-tuning"),
                    ],
                ),
            ],
        )
        out = _format_paragraph_guidance(plan)
        assert "Architecture" in out, (
            "_format_paragraph_guidance must mention subsection title 'Architecture'"
        )
        assert "Training" in out, (
            "_format_paragraph_guidance must mention subsection title 'Training'"
        )
        assert "subsection" in out.lower() or "\\subsection" in out, (
            "_format_paragraph_guidance must reference \\subsection when subsections exist"
        )

    def test_structure_contract_allows_subsection_when_present(self):
        """_format_structure_quality_contract must NOT forbid \\subsection{}
        when section_plan.subsections is non-empty."""
        from src.agents.shared.prompt_compiler import _format_structure_quality_contract
        from src.agents.planner_agent.models import (
            SectionPlan, SubSectionPlan, ParagraphPlan,
        )

        plan = SectionPlan(
            section_type="method",
            section_title="Method",
            sectioning_recommended=True,
            subsections=[
                SubSectionPlan(
                    title="Arch",
                    paragraphs=[ParagraphPlan(key_point="x")],
                ),
            ],
        )
        out = _format_structure_quality_contract("method", plan)
        assert "DO NOT use" not in out, (
            "Structure contract must not forbid \\subsection{} when subsections exist"
        )

    def test_structure_contract_forbids_subsection_when_not_present(self):
        """_format_structure_quality_contract should still forbid \\subsection{}
        when there are no subsections and sectioning_recommended=False."""
        from src.agents.shared.prompt_compiler import _format_structure_quality_contract
        from src.agents.planner_agent.models import SectionPlan, ParagraphPlan

        plan = SectionPlan(
            section_type="method",
            section_title="Method",
            sectioning_recommended=False,
            paragraphs=[
                ParagraphPlan(key_point=f"P{i}") for i in range(4)
            ],
        )
        out = _format_structure_quality_contract("method", plan)
        assert "DO NOT use" in out


class TestDecomposedSubsectionHeaders:
    """_generate_section_decomposed must insert \\subsection{} headers."""

    def test_decomposed_inserts_subsection_headers(self):
        """Source of _generate_section_decomposed must contain subsection
        header insertion logic."""
        import inspect
        from src.agents.metadata_agent.metadata_agent import MetaDataAgent

        source = inspect.getsource(MetaDataAgent._generate_section_decomposed)
        assert "\\subsection{" in source or "subsection_title" in source, (
            "_generate_section_decomposed must insert \\subsection{} headers "
            "before each subsection's paragraphs"
        )

    def test_compile_core_prompt_accepts_subsection_title(self):
        """compile_core_prompt should accept a subsection_title parameter."""
        import inspect
        from src.agents.shared.prompt_compiler import compile_core_prompt

        sig = inspect.signature(compile_core_prompt)
        assert "subsection_title" in sig.parameters, (
            "compile_core_prompt must accept subsection_title so the LLM knows "
            "which subsection the paragraph belongs to"
        )

    def test_compile_core_prompt_includes_subsection_context(self):
        """When subsection_title is provided, compile_core_prompt should
        include it in the output."""
        from src.agents.shared.prompt_compiler import compile_core_prompt
        from src.agents.planner_agent.models import ParagraphPlan

        para = ParagraphPlan(key_point="Attention design", approx_sentences=4)
        out = compile_core_prompt(
            paragraph_plan=para,
            section_type="method",
            section_title="Methodology",
            subsection_title="Self-Attention Architecture",
        )
        assert "Self-Attention Architecture" in out
