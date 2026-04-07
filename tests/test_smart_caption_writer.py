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
