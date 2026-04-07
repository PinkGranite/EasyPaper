"""
Tests for enhanced table converter: TableAnalyzer, enhanced prompt, TableValidator,
smart width decisions, and end-to-end integration.
Covers Phases 1–5 of the Table Converter Enhancement TDD plan.
"""
import pytest
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: TableAnalyzer — CSV/MD structural pre-processing
# ═══════════════════════════════════════════════════════════════════════════

# -- Real CSV data from experiments (inline) --

CSV_TAB1 = """\
Models.,#Trainable Params.,Open- sourced?.,Visual Question Answering VQAv2 (test-dev).VQA acc.,Image Captioning NoCaps (val).CIDEr,Image Captioning NoCaps (val).SPICE,Image-Text Retrieval Flickr (test).TR@1,Image-Text Retrieval Flickr (test).IR@1
"BLIP (Li et al., 2022)",583M,✓,-,113.2,14.8,96.7,86.7
"SimVLM (Wang et al., 2021b)",1.4B,✗,-,112.2,-,-,-
"BEIT-3 (Wang et al., 2022b)",1.9B,✗,-,-,-,94.9,81.5
"Flamingo (Alayrac et al., 2022)",10.2B,✗,56.3,-,-,-,-
BLIP-2,188M,✓,65.0,121.6,15.8,97.6,89.7
"""

CSV_TAB2 = """\
Models.Models,#Trainable Params.#Trainable Params,#Total Params.#Total Params,VQAv2.val,VQAv2.test-dev,OK-VQA test.,GQA test-dev.
VL-T5 no-vqa,224M,269M,13.5,-,5.8,6.3
"FewVLM (Jin et al., 2022)",740M,785M,47.7,-,16.5,29.3
"Frozen (Tsimpoukelli et al., 2021)",40M,7.1B,29.6,-,5.9,-
"VLKD (Dai et al., 2022)",406M,832M,42.6,44.5,13.3,-
"Flamingo3B (Alayrac et al., 2022)",1.4B,3.2B,-,49.2,41.2,-
"Flamingo9B (Alayrac et al., 2022)",1.8B,9.3B,-,51.8,44.7,-
"Flamingo80B (Alayrac et al., 2022)",10.2B,80B,-,56.3,50.6,-
BLIP-2 ViT-L OPT 2.7B,104M,3.1B,50.1,49.7,30.2,33.9
BLIP-2 ViT-g OPT 2.7B,107M,3.8B,53.5,52.3,31.7,34.6
BLIP-2 ViT-g OPT 6.7B,108M,7.8B,54.3,52.6,36.4,36.4
BLIP-2 ViT-L FlanT5 XL,103M,3.4B,62.6,62.3,39.4,44.4
BLIP-2 ViT-g FlanT5 XL,107M,4.1B,63.1,63.0,40.7,44.2
BLIP-2 ViT-g FlanT5 XXL,108M,12.1B,65.2,65.0,45.9,44.7
"""

CSV_TAB5 = """\
Model.,#Trainable Params..,Flickr30K Zero-shot (1K test set).Image →.R@1,Flickr30K Zero-shot (1K test set).Image →.R@5,Flickr30K Zero-shot (1K test set).Text.R@10,Flickr30K Zero-shot (1K test set).Text.R@1,Flickr30K Zero-shot (1K test set).Text → Image.R@5,Flickr30K Zero-shot (1K test set).Text → Image.R@10,COCO Fine-tuned (5K test set).Image → Text.R@1,COCO Fine-tuned (5K test set).Image → Text.R@5,COCO Fine-tuned (5K test set).Image → Text.R@10,COCO Fine-tuned (5K test set).Text → Image.R@1,COCO Fine-tuned (5K test set).Text → Image.R@5,COCO Fine-tuned (5K test set).Text → Image.R@10
Dual-encoder models,,,,,,,,,,,,,
"CLIP (Radford et al., 2021)",428M,88.0,98.7,99.4,68.7,90.6,95.2,-,-,-,-,-,-
"ALIGN (Jia et al., 2021)",820M,88.6,98.7,99.7,75.7,93.8,96.8,77.0,93.5,96.9,59.9,83.3,89.8
"FILIP (Yao et al., 2022)",417M,89.8,99.2,99.8,75.0,93.4,96.3,78.9,94.4,97.4,61.2,84.3,90.6
"Florence (Yuan et al., 2021)",893M,90.9,99.1,-,76.7,93.6,-,81.8,95.2,-,63.2,85.7,-
"BEIT-3(Wang et al., 2022b)",1.9B,94.9,99.9,100.0,81.5,95.6,97.8,84.8,96.5,98.3,67.2,87.7,92.8
"Fusion-encoder models UNITER (Chen et al., 2020)",303M,83.6,95.7,97.7,68.7,89.2,93.9,65.7,88.6,93.8,52.9,79.9,88.0
"OSCAR (Li et al., 2020)",345M,-,-,-,-,-,-,70.0,91.1,95.5,54.0,80.8,88.5
"VinVL (Zhang et al., 2021)",345M,-,-,-,-,-,-,75.4,92.9,96.2,58.8,83.5,90.3
Dual encoder + Fusion encoder reranking,Dual encoder + Fusion encoder reranking,,,,,,,,,,,,
"ALBEF (Li et al., 2021)",233M,94.1,99.5,99.7,82.8,96.3,98.1,77.6,94.3,97.2,60.7,84.3,90.5
"BLIP (Li et al., 2022)",446M,96.7,100.0,100.0,86.7,97.3,98.7,82.4,95.4,97.9,65.1,86.3,91.8
BLIP-2 ViT-L,474M,96.9,100.0,100.0,88.6,97.6,98.9,83.5,96.0,98.0,66.3,86.5,91.8
BLIP-2 ViT-g,1.2B,97.6,100.0,100.0,89.7,98.1,98.9,85.4,97.0,98.5,68.3,87.7,92.6
"""

CSV_TAB6 = """\
COCO finetuning objectives,Image R@1,→ Text R@5,Text → R@1,Image R@5
ITC + ITM,84.5,96.2,67.2,87.1
ITC + ITM + ITG,85.4,97.0,68.3,87.7
"""

MD_TABLE = """\
| Model | Acc | F1 |
|---|---|---|
| A | 0.9 | 0.85 |
| B | 0.92 | 0.88 |
"""


class TestTableAnalyzer:
    """Phase 1: Verify structural analysis of CSV/MD content."""

    def test_basic_csv_structure(self):
        """Small CSV (tab_6) should give exact row/col counts."""
        from src.agents.shared.table_converter import TableAnalyzer

        result = TableAnalyzer.analyze(CSV_TAB6)
        assert result.col_count == 5
        assert result.data_row_count == 2

    def test_large_csv_col_count(self):
        """Wide CSV (tab_5, 14 cols) should be detected."""
        from src.agents.shared.table_converter import TableAnalyzer

        result = TableAnalyzer.analyze(CSV_TAB5)
        assert result.col_count == 14

    def test_large_csv_row_count(self):
        """tab_5 has 14 data rows (including 2 sub-header rows)."""
        from src.agents.shared.table_converter import TableAnalyzer

        result = TableAnalyzer.analyze(CSV_TAB5)
        # 14 data rows (some are sub-headers like "Dual-encoder models")
        assert result.data_row_count >= 12

    def test_multirow_header_detection(self):
        """Dot-separated headers (tab_1) should be flagged as multi-level."""
        from src.agents.shared.table_converter import TableAnalyzer

        result = TableAnalyzer.analyze(CSV_TAB1)
        assert result.has_multirow_header is True

    def test_simple_header_no_multirow(self):
        """Simple headers without dots should not be multi-level."""
        from src.agents.shared.table_converter import TableAnalyzer

        result = TableAnalyzer.analyze(CSV_TAB6)
        assert result.has_multirow_header is False

    def test_width_class_narrow(self):
        """Small 5-col table with short values should be narrow/medium."""
        from src.agents.shared.table_converter import TableAnalyzer

        result = TableAnalyzer.analyze(CSV_TAB6)
        assert result.estimated_width_class in ("narrow", "medium")

    def test_width_class_wide(self):
        """14-col table (tab_5) should be wide or very_wide."""
        from src.agents.shared.table_converter import TableAnalyzer

        result = TableAnalyzer.analyze(CSV_TAB5)
        assert result.estimated_width_class in ("wide", "very_wide")

    def test_markdown_table_parsing(self):
        """Markdown table should be parsed correctly."""
        from src.agents.shared.table_converter import TableAnalyzer

        result = TableAnalyzer.analyze(MD_TABLE)
        assert result.col_count == 3
        assert result.data_row_count == 2

    def test_max_cell_width(self):
        """max_cell_width should reflect the longest cell content."""
        from src.agents.shared.table_converter import TableAnalyzer

        result = TableAnalyzer.analyze(CSV_TAB1)
        # "Visual Question Answering VQAv2 (test-dev).VQA acc." is very long
        assert result.max_cell_width > 20

    def test_header_rows_extracted(self):
        """Header rows should capture the raw header structure."""
        from src.agents.shared.table_converter import TableAnalyzer

        result = TableAnalyzer.analyze(CSV_TAB1)
        assert len(result.header_levels) >= 1
        # First level should contain things like "Models.", "#Trainable Params."
        assert any("Models" in h for h in result.header_levels[0])

    def test_subgroup_rows_detected(self):
        """Rows like 'Dual-encoder models,,,...' should be detected as sub-group rows."""
        from src.agents.shared.table_converter import TableAnalyzer

        result = TableAnalyzer.analyze(CSV_TAB5)
        assert result.subgroup_row_count >= 2

    def test_analyze_returns_table_structure(self):
        """Return type should be TableStructure dataclass."""
        from src.agents.shared.table_converter import TableAnalyzer, TableStructure

        result = TableAnalyzer.analyze(CSV_TAB6)
        assert isinstance(result, TableStructure)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Enhanced prompt — context-aware conversion
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildConversionPrompt:
    """Phase 2: Verify the enhanced prompt includes structural context."""

    def test_prompt_includes_data_preservation_rule(self):
        """Prompt must contain a strict data-preservation instruction."""
        from src.agents.shared.table_converter import (
            TableAnalyzer, build_conversion_prompt,
        )
        structure = TableAnalyzer.analyze(CSV_TAB1)
        prompt = build_conversion_prompt(
            label="tab:test", caption="Test", content=CSV_TAB1,
            structure=structure,
        )
        assert "ALL rows" in prompt or "all rows" in prompt
        assert "ALL columns" in prompt or "all columns" in prompt

    def test_prompt_includes_row_col_counts(self):
        """Prompt should tell the LLM exactly how many rows/cols to expect."""
        from src.agents.shared.table_converter import (
            TableAnalyzer, build_conversion_prompt,
        )
        structure = TableAnalyzer.analyze(CSV_TAB2)
        prompt = build_conversion_prompt(
            label="tab:vqa", caption="VQA", content=CSV_TAB2,
            structure=structure,
        )
        assert str(structure.data_row_count) in prompt
        assert str(structure.col_count) in prompt

    def test_prompt_multirow_header_guidance(self):
        """When multi-level headers detected, prompt must mention \\multicolumn."""
        from src.agents.shared.table_converter import (
            TableAnalyzer, build_conversion_prompt,
        )
        structure = TableAnalyzer.analyze(CSV_TAB1)
        assert structure.has_multirow_header
        prompt = build_conversion_prompt(
            label="tab:overview", caption="Overview", content=CSV_TAB1,
            structure=structure,
        )
        assert "multicolumn" in prompt.lower() or "\\multicolumn" in prompt

    def test_prompt_no_multicolumn_for_simple(self):
        """Simple headers should NOT get multicolumn guidance."""
        from src.agents.shared.table_converter import (
            TableAnalyzer, build_conversion_prompt,
        )
        structure = TableAnalyzer.analyze(CSV_TAB6)
        assert not structure.has_multirow_header
        prompt = build_conversion_prompt(
            label="tab:ablation", caption="Ablation", content=CSV_TAB6,
            structure=structure,
        )
        assert "multicolumn" not in prompt.lower()

    def test_prompt_double_column_layout(self):
        """When column_format='double', prompt should mention table* and resizebox."""
        from src.agents.shared.table_converter import (
            TableAnalyzer, build_conversion_prompt,
        )
        structure = TableAnalyzer.analyze(CSV_TAB5)
        prompt = build_conversion_prompt(
            label="tab:retrieval", caption="Retrieval", content=CSV_TAB5,
            structure=structure, column_format="double",
        )
        assert "table*" in prompt
        assert "resizebox" in prompt.lower() or "\\resizebox" in prompt

    def test_prompt_single_column_narrow_table(self):
        """Single-column narrow table should use \\begin{table}, no resizebox."""
        from src.agents.shared.table_converter import (
            TableAnalyzer, build_conversion_prompt,
        )
        structure = TableAnalyzer.analyze(CSV_TAB6)
        prompt = build_conversion_prompt(
            label="tab:ablation", caption="Ablation", content=CSV_TAB6,
            structure=structure, column_format="single",
        )
        assert "table*" not in prompt
        assert "resizebox" not in prompt.lower()

    def test_prompt_very_wide_table_landscape_or_tiny(self):
        """Very wide table (14 cols) should mention \\tiny or landscape option."""
        from src.agents.shared.table_converter import (
            TableAnalyzer, build_conversion_prompt,
        )
        structure = TableAnalyzer.analyze(CSV_TAB5)
        prompt = build_conversion_prompt(
            label="tab:retrieval", caption="Retrieval", content=CSV_TAB5,
            structure=structure, column_format="double",
        )
        assert "tiny" in prompt.lower() or "small" in prompt.lower() or "scriptsize" in prompt.lower()

    def test_prompt_subgroup_rows_guidance(self):
        """Tables with sub-group rows should get \\midrule guidance."""
        from src.agents.shared.table_converter import (
            TableAnalyzer, build_conversion_prompt,
        )
        structure = TableAnalyzer.analyze(CSV_TAB5)
        assert structure.subgroup_row_count >= 2
        prompt = build_conversion_prompt(
            label="tab:retrieval", caption="Retrieval", content=CSV_TAB5,
            structure=structure,
        )
        assert "sub-group" in prompt.lower() or "subgroup" in prompt.lower() or "midrule" in prompt.lower()

    def test_dynamic_max_tokens(self):
        """build_conversion_prompt should return suggested max_tokens."""
        from src.agents.shared.table_converter import (
            TableAnalyzer, build_conversion_prompt,
        )
        small = TableAnalyzer.analyze(CSV_TAB6)
        large = TableAnalyzer.analyze(CSV_TAB5)
        _, small_tokens = build_conversion_prompt(
            label="t1", caption="c1", content=CSV_TAB6,
            structure=small, return_max_tokens=True,
        )
        _, large_tokens = build_conversion_prompt(
            label="t2", caption="c2", content=CSV_TAB5,
            structure=large, return_max_tokens=True,
        )
        assert large_tokens > small_tokens
        assert small_tokens >= 1500
        assert large_tokens >= 3000


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: TableValidator — post-processing validation & auto-fix
# ═══════════════════════════════════════════════════════════════════════════

GOOD_LATEX = r"""
\begin{table}[htbp]
\centering
\caption{Ablation study.}\label{tab:ablation}
\begin{tabular}{lcccc}
\toprule
Objectives & Image R@1 & Text R@5 & Text R@1 & Image R@5 \\
\midrule
ITC + ITM & 84.5 & 96.2 & 67.2 & 87.1 \\
ITC + ITM + ITG & 85.4 & 97.0 & 68.3 & 87.7 \\
\bottomrule
\end{tabular}
\end{table}
"""

LATEX_MISSING_LABEL = r"""
\begin{table}[htbp]
\centering
\caption{Ablation study.}
\begin{tabular}{lcc}
\toprule
A & B & C \\
\midrule
1 & 2 & 3 \\
\bottomrule
\end{tabular}
\end{table}
"""

LATEX_TRUNCATED_ROWS = r"""
\begin{table}[htbp]
\centering
\caption{Results.}\label{tab:results}
\begin{tabular}{lccc}
\toprule
Model & Acc & F1 & AUC \\
\midrule
A & 0.9 & 0.85 & 0.92 \\
\bottomrule
\end{tabular}
\end{table}
"""

LATEX_WIDE_NO_RESIZE = r"""
\begin{table*}[htbp]
\centering
\caption{Retrieval results.}\label{tab:retrieval}
\begin{tabular}{lccccccccccccc}
\toprule
Model & P & R1 & R5 & R10 & F1 & R2 & R3 & R4 & R6 & R7 & R8 & R9 & R11 \\
\midrule
A & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 & 13 \\
B & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 & 13 \\
\bottomrule
\end{tabular}
\end{table*}
"""


class TestTableValidator:
    """Phase 3: Verify validation and auto-fix of LLM-generated LaTeX."""

    def test_valid_table_passes(self):
        """A correct table should produce no errors."""
        from src.agents.shared.table_converter import (
            TableValidator, TableStructure,
        )
        structure = TableStructure(col_count=5, data_row_count=2)
        result = TableValidator.validate(GOOD_LATEX, structure, "tab:ablation")
        assert result.is_valid
        assert len(result.errors) == 0

    def test_missing_label_detected(self):
        """Missing \\label should be flagged."""
        from src.agents.shared.table_converter import (
            TableValidator, TableStructure,
        )
        structure = TableStructure(col_count=3, data_row_count=1)
        result = TableValidator.validate(
            LATEX_MISSING_LABEL, structure, "tab:test",
        )
        assert not result.is_valid
        assert any("label" in e.lower() for e in result.errors)

    def test_truncated_rows_detected(self):
        """Fewer data rows than expected should be flagged."""
        from src.agents.shared.table_converter import (
            TableValidator, TableStructure,
        )
        structure = TableStructure(col_count=4, data_row_count=5)
        result = TableValidator.validate(
            LATEX_TRUNCATED_ROWS, structure, "tab:results",
        )
        assert not result.is_valid
        assert any("row" in e.lower() for e in result.errors)

    def test_missing_booktabs_detected(self):
        """Table without \\toprule/\\bottomrule should be flagged."""
        from src.agents.shared.table_converter import (
            TableValidator, TableStructure,
        )
        no_booktabs = r"""
\begin{table}[htbp]
\centering
\caption{Test.}\label{tab:test}
\begin{tabular}{lc}
\hline
A & B \\
\hline
1 & 2 \\
\hline
\end{tabular}
\end{table}
"""
        structure = TableStructure(col_count=2, data_row_count=1)
        result = TableValidator.validate(no_booktabs, structure, "tab:test")
        assert any("booktabs" in e.lower() or "toprule" in e.lower()
                    for e in result.warnings)

    def test_auto_fix_missing_label(self):
        """auto_fix should insert missing \\label."""
        from src.agents.shared.table_converter import (
            TableValidator, TableStructure,
        )
        structure = TableStructure(col_count=3, data_row_count=1)
        fixed = TableValidator.auto_fix(
            LATEX_MISSING_LABEL, structure, "tab:test",
        )
        assert r"\label{tab:test}" in fixed

    def test_auto_fix_resizebox_for_wide_table(self):
        """Wide table in double-column format should get \\resizebox wrap."""
        from src.agents.shared.table_converter import (
            TableValidator, TableStructure,
        )
        structure = TableStructure(
            col_count=14, data_row_count=2,
            estimated_width_class="very_wide",
        )
        fixed = TableValidator.auto_fix(
            LATEX_WIDE_NO_RESIZE, structure, "tab:retrieval",
            column_format="double",
        )
        assert r"\resizebox" in fixed

    def test_validation_result_type(self):
        """validate should return a ValidationResult."""
        from src.agents.shared.table_converter import (
            TableValidator, TableStructure, ValidationResult,
        )
        structure = TableStructure(col_count=5, data_row_count=2)
        result = TableValidator.validate(GOOD_LATEX, structure, "tab:ablation")
        assert isinstance(result, ValidationResult)

    def test_col_count_mismatch_warning(self):
        """Significantly fewer columns than expected should warn."""
        from src.agents.shared.table_converter import (
            TableValidator, TableStructure,
        )
        structure = TableStructure(col_count=10, data_row_count=2)
        result = TableValidator.validate(GOOD_LATEX, structure, "tab:ablation")
        assert any("column" in w.lower() for w in result.warnings)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Smart width decision — replace naive _promote_wide_tables
# ═══════════════════════════════════════════════════════════════════════════


class TestSmartPromoteWideTables:
    """Phase 4: Smart width-aware table promotion and resizebox wrapping."""

    def test_narrow_table_unchanged(self):
        """3-column table with short content should not be promoted."""
        from src.agents.shared.table_converter import smart_promote_wide_tables
        content = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Small.}\\label{tab:small}\n"
            "\\begin{tabular}{lcc}\n"
            "\\toprule\n"
            "A & B & C \\\\\n"
            "\\midrule\n"
            "1 & 2 & 3 \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}"
        )
        result = smart_promote_wide_tables(content)
        assert "\\begin{table}" in result
        assert "table*" not in result
        assert "\\resizebox" not in result

    def test_very_wide_table_promoted_and_resized(self):
        """14-column table should be promoted to table* and get resizebox."""
        from src.agents.shared.table_converter import smart_promote_wide_tables
        cols = " & ".join([f"C{i}" for i in range(14)])
        data = " & ".join(["1"] * 14)
        content = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Wide.}\\label{tab:wide}\n"
            f"\\begin{{tabular}}{{{('c' * 14)}}}\n"
            "\\toprule\n"
            f"{cols} \\\\\n"
            "\\midrule\n"
            f"{data} \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}"
        )
        result = smart_promote_wide_tables(content)
        assert "\\begin{table*}" in result
        assert "\\end{table*}" in result
        assert "\\resizebox" in result

    def test_medium_table_gets_resizebox_only(self):
        """6-column table should get resizebox but NOT promoted to table*."""
        from src.agents.shared.table_converter import smart_promote_wide_tables
        content = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Medium.}\\label{tab:med}\n"
            "\\begin{tabular}{lccccc}\n"
            "\\toprule\n"
            "Model & A & B & C & D & E \\\\\n"
            "\\midrule\n"
            "X & 1 & 2 & 3 & 4 & 5 \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}"
        )
        result = smart_promote_wide_tables(content)
        assert "\\begin{table}" in result
        assert "table*" not in result
        assert "\\resizebox" in result

    def test_existing_table_star_gets_resizebox(self):
        """table* that already exists should get resizebox if wide."""
        from src.agents.shared.table_converter import smart_promote_wide_tables
        content = (
            "\\begin{table*}[htbp]\n"
            "\\centering\n"
            "\\caption{Wide.}\\label{tab:wide}\n"
            "\\begin{tabular}{lcccccccc}\n"
            "\\toprule\n"
            "A & B & C & D & E & F & G & H & I \\\\\n"
            "\\midrule\n"
            "1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table*}"
        )
        result = smart_promote_wide_tables(content)
        assert "\\begin{table*}" in result
        assert "\\resizebox" in result

    def test_already_resized_table_not_double_wrapped(self):
        """Table already with \\resizebox should not be wrapped again."""
        from src.agents.shared.table_converter import smart_promote_wide_tables
        content = (
            "\\begin{table*}[htbp]\n"
            "\\centering\n"
            "\\caption{Wide.}\\label{tab:wide}\n"
            "\\resizebox{\\textwidth}{!}{\n"
            "\\begin{tabular}{lcccccccc}\n"
            "\\toprule\n"
            "A & B & C & D & E & F & G & H & I \\\\\n"
            "\\midrule\n"
            "1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "}\n"
            "\\end{table*}"
        )
        result = smart_promote_wide_tables(content)
        assert result.count("\\resizebox") == 1

    def test_wide_content_narrow_cols_gets_resizebox(self):
        """4-col table with very wide content should get resizebox."""
        from src.agents.shared.table_converter import smart_promote_wide_tables
        content = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Long.}\\label{tab:long}\n"
            "\\begin{tabular}{llll}\n"
            "\\toprule\n"
            "Model Name & Very Long Description & Another Long Text & "
            "Even More Text Here \\\\\n"
            "\\midrule\n"
            "SomeVeryLongModelNameHere & This is a description & "
            "Another description text & More text here \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}"
        )
        result = smart_promote_wide_tables(content)
        assert "\\resizebox" in result

    def test_p_columns_counted_correctly(self):
        """Columns with p{width} specs should be counted."""
        from src.agents.shared.table_converter import smart_promote_wide_tables
        content = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Para.}\\label{tab:para}\n"
            "\\begin{tabular}{lp{3cm}p{3cm}p{3cm}p{3cm}p{3cm}}\n"
            "\\toprule\n"
            "A & B & C & D & E & F \\\\\n"
            "\\midrule\n"
            "1 & 2 & 3 & 4 & 5 & 6 \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}"
        )
        result = smart_promote_wide_tables(content)
        assert "\\resizebox" in result

    def test_scriptsize_for_very_wide(self):
        """14+ column tables should get \\scriptsize font."""
        from src.agents.shared.table_converter import smart_promote_wide_tables
        cols = " & ".join([f"C{i}" for i in range(14)])
        data = " & ".join(["val"] * 14)
        content = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Huge.}\\label{tab:huge}\n"
            f"\\begin{{tabular}}{{{('c' * 14)}}}\n"
            "\\toprule\n"
            f"{cols} \\\\\n"
            "\\midrule\n"
            f"{data} \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}"
        )
        result = smart_promote_wide_tables(content)
        assert "\\scriptsize" in result or "\\small" in result or "\\tiny" in result


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: End-to-end integration tests with real CSV data + pipeline wiring
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEndPipeline:
    """Phase 5: Verify the full analyze → prompt → validate → fix pipeline."""

    def test_full_pipeline_small_table(self):
        """Small table (tab_6): analyze → prompt → validate → no fixes needed."""
        from src.agents.shared.table_converter import (
            TableAnalyzer, build_conversion_prompt, TableValidator,
        )
        structure = TableAnalyzer.analyze(CSV_TAB6)
        assert structure.col_count == 5
        assert structure.data_row_count == 2

        prompt = build_conversion_prompt(
            label="tab:itg_ablation",
            caption="ITG ablation study",
            content=CSV_TAB6,
            structure=structure,
            column_format="double",
        )
        assert "5 columns" in prompt
        assert "2 data rows" in prompt

        # Simulate good LLM output
        mock_latex = GOOD_LATEX
        result = TableValidator.validate(mock_latex, structure, "tab:ablation")
        assert result.is_valid

    def test_full_pipeline_wide_table(self):
        """Wide table (tab_5, 14 cols): should trigger table* + resizebox."""
        from src.agents.shared.table_converter import (
            TableAnalyzer, build_conversion_prompt, TableValidator,
            smart_promote_wide_tables,
        )
        structure = TableAnalyzer.analyze(CSV_TAB5)
        assert structure.col_count == 14
        assert structure.estimated_width_class == "very_wide"
        assert structure.has_multirow_header is True
        assert structure.subgroup_row_count >= 2

        prompt, max_tokens = build_conversion_prompt(
            label="tab:retrieval",
            caption="Retrieval results",
            content=CSV_TAB5,
            structure=structure,
            column_format="double",
            return_max_tokens=True,
        )
        assert "table*" in prompt
        assert "resizebox" in prompt.lower()
        assert "multicolumn" in prompt.lower()
        assert "sub-group" in prompt.lower() or "subgroup" in prompt.lower()
        assert max_tokens >= 3000

    def test_full_pipeline_multirow_header(self):
        """Table with multi-level headers (tab_1): prompt includes multicolumn."""
        from src.agents.shared.table_converter import (
            TableAnalyzer, build_conversion_prompt,
        )
        structure = TableAnalyzer.analyze(CSV_TAB1)
        assert structure.has_multirow_header
        assert structure.col_count == 8

        prompt = build_conversion_prompt(
            label="tab:zero_shot_overview",
            caption="Overview results",
            content=CSV_TAB1,
            structure=structure,
            column_format="double",
        )
        assert "multicolumn" in prompt.lower()
        assert "8 columns" in prompt
        assert "5 data rows" in prompt

    def test_full_pipeline_validate_and_fix(self):
        """Pipeline with broken LLM output: validator detects, auto_fix repairs."""
        from src.agents.shared.table_converter import (
            TableAnalyzer, TableValidator,
        )
        structure = TableAnalyzer.analyze(CSV_TAB6)

        # Missing label
        broken = LATEX_MISSING_LABEL
        result = TableValidator.validate(broken, structure, "tab:itg_ablation")
        assert not result.is_valid

        fixed = TableValidator.auto_fix(
            broken, structure, "tab:itg_ablation", column_format="double",
        )
        assert r"\label{tab:itg_ablation}" in fixed

    def test_smart_promote_integration_with_real_structure(self):
        """smart_promote_wide_tables applied to simulated LLM output for tab_5."""
        from src.agents.shared.table_converter import smart_promote_wide_tables

        # Simulated LLM output for a 14-col table (without promotion)
        cols_spec = "l" + "c" * 13
        header = " & ".join(["Model"] + [f"R@{i}" for i in range(1, 14)])
        row = " & ".join(["BLIP-2"] + ["95.0"] * 13)
        latex = (
            f"\\begin{{table}}[htbp]\n"
            f"\\centering\n"
            f"\\caption{{Retrieval.}}\\label{{tab:retrieval}}\n"
            f"\\begin{{tabular}}{{{cols_spec}}}\n"
            f"\\toprule\n"
            f"{header} \\\\\n"
            f"\\midrule\n"
            f"{row} \\\\\n"
            f"\\bottomrule\n"
            f"\\end{{tabular}}\n"
            f"\\end{{table}}"
        )
        result = smart_promote_wide_tables(latex)
        assert "\\begin{table*}" in result
        assert "\\resizebox" in result
        assert "\\scriptsize" in result

    def test_convert_table_to_latex_uses_enhanced_prompt(self):
        """convert_table_to_latex should use enhanced prompt when structure available."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from src.agents.shared.table_converter import convert_table_to_latex

        # Create a mock TableSpec
        table = MagicMock()
        table.id = "tab:test"
        table.caption = "Test table"
        table.auto_generate = False
        table.file_path = None
        table.content = CSV_TAB6
        table.wide = False

        # Mock LLM client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = GOOD_LATEX

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = asyncio.get_event_loop().run_until_complete(
            convert_table_to_latex(
                table=table,
                llm_client=mock_client,
                model_name="test-model",
                column_format="double",
            )
        )

        assert result is not None
        # Verify the enhanced prompt was used
        call_args = mock_client.chat.completions.create.call_args
        prompt_text = call_args.kwargs["messages"][1]["content"]
        assert "ALL rows" in prompt_text or "all rows" in prompt_text
        assert "CRITICAL" in prompt_text

    def test_convert_table_to_latex_validates_output(self):
        """convert_table_to_latex should validate and auto-fix LLM output."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from src.agents.shared.table_converter import convert_table_to_latex

        table = MagicMock()
        table.id = "tab:needs_fix"
        table.caption = "Needs fix"
        table.auto_generate = False
        table.file_path = None
        table.content = CSV_TAB6
        table.wide = False

        # LLM returns latex WITHOUT the label
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = LATEX_MISSING_LABEL

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = asyncio.get_event_loop().run_until_complete(
            convert_table_to_latex(
                table=table,
                llm_client=mock_client,
                model_name="test-model",
                column_format="double",
            )
        )

        assert result is not None
        assert r"\label{tab:needs_fix}" in result

    def test_convert_tables_batch_uses_enhanced(self):
        """convert_tables should process all tables with enhanced pipeline."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from src.agents.shared.table_converter import convert_tables

        tables = []
        for i, csv_data in enumerate([CSV_TAB6, CSV_TAB1]):
            t = MagicMock()
            t.id = f"tab:t{i}"
            t.caption = f"Table {i}"
            t.auto_generate = False
            t.file_path = None
            t.content = csv_data
            t.wide = False
            tables.append(t)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = GOOD_LATEX

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = asyncio.get_event_loop().run_until_complete(
            convert_tables(
                tables=tables,
                llm_client=mock_client,
                model_name="test-model",
                column_format="double",
            )
        )

        assert len(result) == 2
        assert "tab:t0" in result
        assert "tab:t1" in result
