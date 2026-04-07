"""
Tests for table direct-injection pipeline.

Instead of having the Writer generate table environments (which leads to
simplified/wrong tables), tables are:
1. Converted by TableConverter (pre-existing)
2. Writer only sees raw data + metadata to write discussion prose
3. Tables are directly injected at the first \\ref location post-generation

Three components tested:
- _format_table_placement_guidance: prompt shows data, forbids \\begin{table}
- strip_writer_tables: defensive removal of Writer-generated table envs
- _ensure_tables_defined (upgraded): primary injection at first \\ref
"""
import re
import pytest
from types import SimpleNamespace
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers to build test fixtures
# ---------------------------------------------------------------------------

def _make_table_spec(
    id: str,
    caption: str = "",
    description: str = "",
    content: Optional[str] = None,
    wide: bool = False,
):
    return SimpleNamespace(
        id=id,
        caption=caption,
        description=description,
        content=content,
        wide=wide,
        file_path=None,
    )


def _make_table_placement(
    table_id: str,
    is_wide: bool = False,
    position_hint: str = "mid",
    message: str = "",
    semantic_role: str = "",
):
    return SimpleNamespace(
        table_id=table_id,
        is_wide=is_wide,
        position_hint=position_hint,
        message=message,
        semantic_role=semantic_role,
    )


def _make_section_plan(
    tables=None,
    tables_to_reference=None,
    tables_to_define=None,
    section_type="result",
):
    def get_table_ids_to_define():
        if tables_to_define is not None:
            return tables_to_define
        if tables:
            return [t.table_id for t in tables]
        return []

    return SimpleNamespace(
        tables=tables or [],
        tables_to_reference=tables_to_reference or [],
        section_type=section_type,
        get_table_ids_to_define=get_table_ids_to_define,
    )


def _make_paper_plan(sections):
    return SimpleNamespace(sections=sections)


# ============================= PHASE 1 =====================================
# _format_table_placement_guidance: show data, forbid \begin{table}
# ===========================================================================

class TestFormatTablePlacementGuidanceDirectInjection:
    """Writer should see raw data for discussion but never be asked to create table envs."""

    def test_no_include_table_env_instruction(self):
        """Prompt must NOT instruct Writer to 'Include the complete table environment'."""
        from src.agents.shared.prompt_compiler import _format_table_placement_guidance

        csv_data = "Model,Acc\nBLIP-2,65.0\nFlamingo,56.3"
        tbl = _make_table_spec("tab:results", caption="Main Results", content=csv_data)
        tp = _make_table_placement("tab:results")
        sp = _make_section_plan(tables=[tp])

        result = _format_table_placement_guidance(sp, [tbl])
        assert "Include the complete table environment" not in result
        assert "Required LaTeX" not in result
        assert "```latex" not in result

    def test_contains_auto_injected_notice(self):
        """Prompt must tell Writer that tables are auto-injected."""
        from src.agents.shared.prompt_compiler import _format_table_placement_guidance

        csv_data = "Model,Acc\nBLIP-2,65.0"
        tbl = _make_table_spec("tab:results", caption="Results", content=csv_data)
        tp = _make_table_placement("tab:results")
        sp = _make_section_plan(tables=[tp])

        result = _format_table_placement_guidance(sp, [tbl])
        assert "auto-inject" in result.lower() or "auto inject" in result.lower()

    def test_contains_do_not_create_table_instruction(self):
        """Prompt must explicitly say DO NOT create \\begin{table}."""
        from src.agents.shared.prompt_compiler import _format_table_placement_guidance

        tbl = _make_table_spec("tab:r", caption="R", content="a,b\n1,2")
        tp = _make_table_placement("tab:r")
        sp = _make_section_plan(tables=[tp])

        result = _format_table_placement_guidance(sp, [tbl])
        assert "DO NOT" in result
        assert "\\begin{table" in result or "begin{table" in result

    def test_shows_raw_data_content(self):
        """Writer should see raw CSV/MD data to write accurate discussion."""
        from src.agents.shared.prompt_compiler import _format_table_placement_guidance

        csv_data = "Model,VQA,NoCaps\nBLIP-2,65.0,121.6\nFlamingo,56.3,--"
        tbl = _make_table_spec("tab:results", caption="Results", content=csv_data)
        tp = _make_table_placement("tab:results")
        sp = _make_section_plan(tables=[tp])

        result = _format_table_placement_guidance(sp, [tbl])
        assert "BLIP-2" in result
        assert "65.0" in result
        assert "121.6" in result

    def test_shows_table_metadata(self):
        """Writer should see caption, description, column/row counts."""
        from src.agents.shared.prompt_compiler import _format_table_placement_guidance

        csv_data = "A,B,C\n1,2,3\n4,5,6\n7,8,9"
        tbl = _make_table_spec(
            "tab:ablation", caption="Ablation Study",
            description="Effect of each component", content=csv_data,
        )
        tp = _make_table_placement("tab:ablation")
        sp = _make_section_plan(tables=[tp])

        result = _format_table_placement_guidance(sp, [tbl])
        assert "Ablation Study" in result
        assert "Effect of each component" in result

    def test_instructs_ref_only(self):
        """Prompt should tell Writer to use Table~\\ref{tab:id} for referencing."""
        from src.agents.shared.prompt_compiler import _format_table_placement_guidance

        tbl = _make_table_spec("tab:x", caption="X", content="a,b\n1,2")
        tp = _make_table_placement("tab:x")
        sp = _make_section_plan(tables=[tp])

        result = _format_table_placement_guidance(sp, [tbl])
        assert "\\ref{tab:x}" in result or "ref{tab:x}" in result

    def test_converted_tables_not_embedded_as_latex(self):
        """Even with converted_tables, should NOT embed LaTeX code in prompt."""
        from src.agents.shared.prompt_compiler import _format_table_placement_guidance

        tbl = _make_table_spec("tab:r", caption="R", content="a,b\n1,2")
        tp = _make_table_placement("tab:r")
        sp = _make_section_plan(tables=[tp])
        converted = {"tab:r": "\\begin{table}[htbp]\n\\centering\n...\\end{table}"}

        result = _format_table_placement_guidance(sp, [tbl], converted_tables=converted)
        assert "```latex" not in result
        assert "Required LaTeX" not in result

    def test_truncates_large_data(self):
        """Very large data should be truncated with indication."""
        from src.agents.shared.prompt_compiler import _format_table_placement_guidance

        big_csv = "col1,col2,col3\n" + "\n".join(
            f"val{i}_a,val{i}_b,val{i}_c" for i in range(200)
        )
        tbl = _make_table_spec("tab:big", caption="Big", content=big_csv)
        tp = _make_table_placement("tab:big")
        sp = _make_section_plan(tables=[tp])

        result = _format_table_placement_guidance(sp, [tbl])
        assert len(result) < len(big_csv)

    def test_reference_only_tables_not_shown_as_define(self):
        """Tables in tables_to_reference should not appear in define section."""
        from src.agents.shared.prompt_compiler import _format_table_placement_guidance

        tbl = _make_table_spec("tab:ref_only", caption="Ref", content="a,b\n1,2")
        tp = _make_table_placement("tab:ref_only")
        sp = _make_section_plan(tables=[tp], tables_to_reference=["tab:ref_only"])

        result = _format_table_placement_guidance(sp, [tbl])
        assert "Data for your reference" not in result or "tab:ref_only" not in result.split("Data for your reference")[0]

    def test_wide_table_noted(self):
        """Wide tables should have a note about spanning columns."""
        from src.agents.shared.prompt_compiler import _format_table_placement_guidance

        tbl = _make_table_spec("tab:wide", caption="Wide", content="a,b\n1,2", wide=True)
        tp = _make_table_placement("tab:wide", is_wide=True)
        sp = _make_section_plan(tables=[tp])

        result = _format_table_placement_guidance(sp, [tbl])
        assert "wide" in result.lower() or "WIDE" in result


# ============================= PHASE 2 =====================================
# strip_writer_tables: defensively remove Writer-generated table environments
# ===========================================================================

class TestStripWriterTables:
    """Remove any \\begin{table}...\\end{table} blocks the Writer generated."""

    def test_strips_simple_table(self):
        from src.agents.shared.table_converter import strip_writer_tables

        content = (
            "Some text before.\n"
            "\\begin{table}[htbp]\n\\centering\n"
            "\\caption{Wrong}\\label{tab:results}\n"
            "\\begin{tabular}{lcc}\na & b & c\\\\\n\\end{tabular}\n"
            "\\end{table}\n"
            "Some text after."
        )
        known_ids = {"tab:results"}
        result = strip_writer_tables(content, known_ids)

        assert "\\begin{table}" not in result
        assert "\\end{table}" not in result
        assert "Some text before." in result
        assert "Some text after." in result

    def test_strips_table_star(self):
        from src.agents.shared.table_converter import strip_writer_tables

        content = (
            "Text.\n"
            "\\begin{table*}[htbp]\n"
            "\\caption{Wide}\\label{tab:wide}\n"
            "\\begin{tabular}{lccccc}\n\\end{tabular}\n"
            "\\end{table*}\n"
            "More text."
        )
        result = strip_writer_tables(content, {"tab:wide"})
        assert "\\begin{table*}" not in result
        assert "More text." in result

    def test_preserves_unknown_tables(self):
        """Tables not in known_ids should be preserved (Writer may legitimately create some)."""
        from src.agents.shared.table_converter import strip_writer_tables

        content = (
            "\\begin{table}[htbp]\n"
            "\\caption{Unknown}\\label{tab:unknown}\n"
            "\\begin{tabular}{lc}\n\\end{tabular}\n"
            "\\end{table}\n"
        )
        result = strip_writer_tables(content, {"tab:results"})
        assert "\\begin{table}" in result

    def test_strips_multiple_known_tables(self):
        from src.agents.shared.table_converter import strip_writer_tables

        content = (
            "Intro.\n"
            "\\begin{table}[htbp]\n\\caption{A}\\label{tab:a}\n\\end{table}\n"
            "Middle.\n"
            "\\begin{table*}[htbp]\n\\caption{B}\\label{tab:b}\n\\end{table*}\n"
            "End."
        )
        result = strip_writer_tables(content, {"tab:a", "tab:b"})
        assert "\\begin{table" not in result
        assert "Intro." in result
        assert "Middle." in result
        assert "End." in result

    def test_empty_known_ids_preserves_all(self):
        from src.agents.shared.table_converter import strip_writer_tables

        content = "\\begin{table}[htbp]\n\\caption{X}\\label{tab:x}\n\\end{table}"
        result = strip_writer_tables(content, set())
        assert "\\begin{table}" in result

    def test_handles_no_tables_in_content(self):
        from src.agents.shared.table_converter import strip_writer_tables

        content = "Just plain text with Table~\\ref{tab:results} reference."
        result = strip_writer_tables(content, {"tab:results"})
        assert result == content

    def test_preserves_table_ref_when_stripping(self):
        """Stripping table env should keep surrounding \\ref mentions intact."""
        from src.agents.shared.table_converter import strip_writer_tables

        content = (
            "As shown in Table~\\ref{tab:r1}, the results are good.\n"
            "\\begin{table}[htbp]\n"
            "\\caption{R1}\\label{tab:r1}\n"
            "\\begin{tabular}{lc}\nA & B\\\\\n\\end{tabular}\n"
            "\\end{table}\n"
            "Further analysis of Table~\\ref{tab:r1} shows..."
        )
        result = strip_writer_tables(content, {"tab:r1"})
        assert "Table~\\ref{tab:r1}" in result
        assert result.count("Table~\\ref{tab:r1}") == 2
        assert "\\begin{table}" not in result


# ============================= PHASE 3 =====================================
# _ensure_tables_defined (upgraded): primary injection at first \ref
# ===========================================================================

class TestInjectTablesAtFirstRef:
    """Tables should be injected at the first \\ref location, not duplicated."""

    def test_injects_at_first_ref(self):
        from src.agents.shared.table_converter import inject_tables

        content = (
            "Results are in Table~\\ref{tab:r1}. We observe improvements.\n"
            "As also shown in Table~\\ref{tab:r1}, the trend continues."
        )
        converted = {
            "tab:r1": (
                "\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Results}\\label{tab:r1}\n"
                "\\begin{tabular}{lcc}\n\\toprule\nA & B & C\\\\\n\\bottomrule\n"
                "\\end{tabular}\n\\end{table}"
            )
        }
        tables = [_make_table_spec("tab:r1", caption="Results")]
        section_plan = _make_section_plan(
            tables=[_make_table_placement("tab:r1")],
            tables_to_define=["tab:r1"],
        )

        result = inject_tables(content, section_plan, tables, converted)

        assert result.count("\\begin{table}") == 1
        first_ref_pos = result.index("Table~\\ref{tab:r1}")
        table_env_pos = result.index("\\begin{table}")
        assert table_env_pos > first_ref_pos

    def test_does_not_inject_already_defined(self):
        """If table env already exists in content, skip injection."""
        from src.agents.shared.table_converter import inject_tables

        content = (
            "See Table~\\ref{tab:r1}.\n"
            "\\begin{table}[htbp]\n\\centering\n"
            "\\caption{Results}\\label{tab:r1}\n"
            "\\begin{tabular}{lcc}\nA & B & C\\\\\n\\end{tabular}\n"
            "\\end{table}"
        )
        converted = {"tab:r1": "\\begin{table}...\\end{table}"}
        tables = [_make_table_spec("tab:r1", caption="Results")]
        section_plan = _make_section_plan(
            tables=[_make_table_placement("tab:r1")],
            tables_to_define=["tab:r1"],
        )

        result = inject_tables(content, section_plan, tables, converted)
        assert result.count("\\begin{table}") == 1

    def test_injects_at_end_if_no_ref(self):
        """If no \\ref found, append table at end of section."""
        from src.agents.shared.table_converter import inject_tables

        content = "This section discusses results without explicit table reference."
        converted = {
            "tab:r1": (
                "\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Results}\\label{tab:r1}\n"
                "\\begin{tabular}{lcc}\nA & B\\\\\n\\end{tabular}\n"
                "\\end{table}"
            )
        }
        tables = [_make_table_spec("tab:r1", caption="Results")]
        section_plan = _make_section_plan(
            tables=[_make_table_placement("tab:r1")],
            tables_to_define=["tab:r1"],
        )

        result = inject_tables(content, section_plan, tables, converted)
        assert "\\begin{table}" in result
        assert result.index("\\begin{table}") > len(content) - 5

    def test_multiple_tables_injected_at_respective_refs(self):
        """Each table injected after its own first \\ref."""
        from src.agents.shared.table_converter import inject_tables

        content = (
            "Table~\\ref{tab:a} shows accuracy. "
            "Table~\\ref{tab:b} shows efficiency.\n"
            "Both Table~\\ref{tab:a} and Table~\\ref{tab:b} confirm our hypothesis."
        )
        converted = {
            "tab:a": "\\begin{table}[htbp]\n\\caption{A}\\label{tab:a}\n\\begin{tabular}{lc}\n\\end{tabular}\n\\end{table}",
            "tab:b": "\\begin{table}[htbp]\n\\caption{B}\\label{tab:b}\n\\begin{tabular}{lc}\n\\end{tabular}\n\\end{table}",
        }
        tables = [
            _make_table_spec("tab:a", caption="A"),
            _make_table_spec("tab:b", caption="B"),
        ]
        section_plan = _make_section_plan(
            tables=[_make_table_placement("tab:a"), _make_table_placement("tab:b")],
            tables_to_define=["tab:a", "tab:b"],
        )

        result = inject_tables(content, section_plan, tables, converted)
        assert result.count("\\begin{table}") == 2
        pos_a = result.index("\\label{tab:a}")
        pos_b = result.index("\\label{tab:b}")
        first_ref_a = result.index("Table~\\ref{tab:a}")
        first_ref_b = result.index("Table~\\ref{tab:b}")
        assert pos_a > first_ref_a
        assert pos_b > first_ref_b

    def test_ensures_label_present(self):
        """Injected table must have \\label."""
        from src.agents.shared.table_converter import inject_tables

        content = "See Table~\\ref{tab:r1}."
        converted = {
            "tab:r1": (
                "\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Results}\n"
                "\\begin{tabular}{lc}\nA & B\\\\\n\\end{tabular}\n"
                "\\end{table}"
            )
        }
        tables = [_make_table_spec("tab:r1", caption="Results")]
        section_plan = _make_section_plan(
            tables=[_make_table_placement("tab:r1")],
            tables_to_define=["tab:r1"],
        )

        result = inject_tables(content, section_plan, tables, converted)
        assert "\\label{tab:r1}" in result

    def test_uses_placeholder_when_not_converted(self):
        """If no converted LaTeX, inject a placeholder table."""
        from src.agents.shared.table_converter import inject_tables

        content = "See Table~\\ref{tab:r1}."
        tables = [_make_table_spec("tab:r1", caption="Results")]
        section_plan = _make_section_plan(
            tables=[_make_table_placement("tab:r1")],
            tables_to_define=["tab:r1"],
        )

        result = inject_tables(content, section_plan, tables, converted_tables={})
        assert "\\begin{table}" in result
        assert "\\label{tab:r1}" in result
        assert "\\caption{Results}" in result

    def test_reference_only_tables_not_injected(self):
        """Tables marked as reference-only should not be injected."""
        from src.agents.shared.table_converter import inject_tables

        content = "See Table~\\ref{tab:r1}."
        converted = {"tab:r1": "\\begin{table}...\\end{table}"}
        tables = [_make_table_spec("tab:r1", caption="Results")]
        section_plan = _make_section_plan(
            tables=[],
            tables_to_define=[],
            tables_to_reference=["tab:r1"],
        )

        result = inject_tables(content, section_plan, tables, converted)
        assert "\\begin{table}" not in result


# ============================= PHASE 4 =====================================
# Integration: full pipeline (strip + inject) in correct order
# ===========================================================================

class TestDirectInjectionPipeline:
    """End-to-end: strip Writer tables, then inject pre-converted tables."""

    def test_strip_then_inject_replaces_simplified_table(self):
        """Writer's simplified table replaced by full pre-converted table."""
        from src.agents.shared.table_converter import strip_writer_tables, inject_tables

        writer_output = (
            "Results in Table~\\ref{tab:r1} are promising.\n"
            "\\begin{table}[htbp]\n\\centering\n"
            "\\caption{Results}\\label{tab:r1}\n"
            "\\begin{tabular}{lcc}\n"
            "\\toprule\nModel & Acc \\\\\n"
            "\\midrule\nBLIP-2 & 65.0 \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
            "Further discussion."
        )
        full_converted = (
            "\\begin{table}[htbp]\n\\centering\n"
            "\\caption{Results}\\label{tab:r1}\n"
            "\\begin{tabular}{lccccccc}\n"
            "\\toprule\n"
            "Model & VQA & NoCaps & Flickr & COCO & GQA & OKVQA & VizWiz \\\\\n"
            "\\midrule\n"
            "BLIP-2 & 65.0 & 121.6 & 97.6 & 85.4 & 44.7 & 45.9 & 19.6 \\\\\n"
            "Flamingo & 56.3 & -- & -- & -- & -- & 50.6 & 28.8 \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}"
        )

        tables = [_make_table_spec("tab:r1", caption="Results")]
        section_plan = _make_section_plan(
            tables=[_make_table_placement("tab:r1")],
            tables_to_define=["tab:r1"],
        )

        stripped = strip_writer_tables(writer_output, {"tab:r1"})
        result = inject_tables(stripped, section_plan, tables, {"tab:r1": full_converted})

        assert result.count("\\begin{table}") == 1
        assert "lccccccc" in result
        assert "VizWiz" in result
        assert "\\begin{tabular}{lcc}" not in result

    def test_pipeline_preserves_prose(self):
        """All discussion text survives strip+inject cycle."""
        from src.agents.shared.table_converter import strip_writer_tables, inject_tables

        writer_output = (
            "Our method achieves state-of-the-art results as shown in Table~\\ref{tab:r1}.\n"
            "\\begin{table}[htbp]\n\\caption{R}\\label{tab:r1}\n"
            "\\begin{tabular}{lc}\nA & B\\\\\n\\end{tabular}\n\\end{table}\n"
            "Specifically, we outperform Flamingo by 8.7% on VQAv2."
        )
        full_converted = (
            "\\begin{table}[htbp]\n\\caption{R}\\label{tab:r1}\n"
            "\\begin{tabular}{lccccc}\nFull data...\n\\end{tabular}\n\\end{table}"
        )
        tables = [_make_table_spec("tab:r1", caption="R")]
        section_plan = _make_section_plan(
            tables=[_make_table_placement("tab:r1")],
            tables_to_define=["tab:r1"],
        )

        stripped = strip_writer_tables(writer_output, {"tab:r1"})
        result = inject_tables(stripped, section_plan, tables, {"tab:r1": full_converted})

        assert "state-of-the-art results" in result
        assert "outperform Flamingo by 8.7%" in result
        assert "Table~\\ref{tab:r1}" in result
