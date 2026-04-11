"""
Table Converter - Convert any readable format to LaTeX tables
- **Description**:
    - Converts CSV, Markdown, plain text tables to LaTeX
    - Uses LLM for intelligent format detection and conversion
    - Handles special characters and academic table formatting
    - Pre-analyzes table structure for layout-aware conversion
    - Post-validates LLM output for data fidelity
"""
import csv
import hashlib
import io
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..metadata_agent.models import TableSpec

logger = logging.getLogger("uvicorn.error")


# ═══════════════════════════════════════════════════════════════════════════
# TableStructure — result of structural analysis
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TableStructure:
    """
    Structural metadata extracted from raw table content before LLM conversion.
    - **Description**:
        - Captures row/col counts, header structure, width estimates,
          and sub-group rows for layout-aware LaTeX generation.
    """
    col_count: int = 0
    data_row_count: int = 0
    has_multirow_header: bool = False
    max_cell_width: int = 0
    estimated_width_class: str = "narrow"
    header_levels: List[List[str]] = field(default_factory=list)
    subgroup_row_count: int = 0
    detected_format: str = "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# TableAnalyzer — pure-code structural pre-processing
# ═══════════════════════════════════════════════════════════════════════════

class TableAnalyzer:
    """
    Analyze CSV/Markdown table content to extract structural metadata.
    - **Description**:
        - Detects format (CSV vs Markdown) automatically.
        - Parses hierarchical (dot-separated) headers.
        - Counts rows, columns, sub-group divider rows.
        - Estimates rendered width class for layout decisions.
    """

    # Width thresholds: total estimated char-width of one row
    _WIDTH_THRESHOLDS = {
        "narrow": 40,
        "medium": 80,
        "wide": 140,
    }

    @classmethod
    def analyze(cls, content: str) -> TableStructure:
        """
        Analyze raw table content and return structural metadata.
        - **Args**:
            - `content` (str): Raw CSV or Markdown table text.
        - **Returns**:
            - `TableStructure`: Extracted structural information.
        """
        content = content.strip()
        if not content:
            return TableStructure()

        fmt = cls._detect_format(content)
        if fmt == "markdown":
            return cls._analyze_markdown(content)
        return cls._analyze_csv(content)

    @staticmethod
    def _detect_format(content: str) -> str:
        """Heuristic: if first non-empty line starts with '|', it's Markdown."""
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("|"):
                return "markdown"
            return "csv"
        return "csv"

    @classmethod
    def _analyze_csv(cls, content: str) -> TableStructure:
        """Parse CSV content and build TableStructure."""
        reader = csv.reader(io.StringIO(content))
        rows: List[List[str]] = []
        for row in reader:
            rows.append(row)

        if not rows:
            return TableStructure(detected_format="csv")

        header_row = rows[0]
        col_count = len(header_row)

        # Detect multi-level headers (dot-separated like "NoCaps (val).CIDEr")
        has_multirow = any("." in h and not h.endswith(".") for h in header_row)
        # Also check for repeated sub-header pattern like "X.X"
        if not has_multirow:
            has_multirow = any(
                "." in h and len(h.split(".")) >= 2 and len(h.split(".")[0]) > 1
                for h in header_row
            )

        # Build header levels
        header_levels: List[List[str]] = []
        if has_multirow:
            max_depth = max(len(h.split(".")) for h in header_row)
            for level in range(max_depth):
                level_headers = []
                for h in header_row:
                    parts = h.split(".")
                    if level < len(parts):
                        level_headers.append(parts[level].strip())
                    else:
                        level_headers.append("")
                header_levels.append(level_headers)
        else:
            header_levels = [header_row]

        # Analyze data rows (skip header)
        data_rows = rows[1:]
        subgroup_count = 0
        actual_data_rows = 0
        max_cell_w = max((len(h) for h in header_row), default=0)

        for row in data_rows:
            # Sub-group row: first cell has content, most others are empty
            non_empty = sum(1 for c in row if c.strip())
            if len(row) > 2 and non_empty <= 2 and row[0].strip():
                subgroup_count += 1
            else:
                actual_data_rows += 1

            for cell in row:
                max_cell_w = max(max_cell_w, len(cell.strip()))

        # Estimate width: average cell width * col_count
        total_cells = sum(len(r) for r in rows)
        total_chars = sum(len(c) for r in rows for c in r)
        avg_cell_w = total_chars / max(total_cells, 1)
        row_width = avg_cell_w * col_count

        width_class = cls._classify_width(row_width, col_count)

        return TableStructure(
            col_count=col_count,
            data_row_count=actual_data_rows,
            has_multirow_header=has_multirow,
            max_cell_width=max_cell_w,
            estimated_width_class=width_class,
            header_levels=header_levels,
            subgroup_row_count=subgroup_count,
            detected_format="csv",
        )

    @classmethod
    def _analyze_markdown(cls, content: str) -> TableStructure:
        """Parse Markdown table and build TableStructure."""
        lines = [l.strip() for l in content.strip().splitlines() if l.strip()]

        # Filter out separator lines (e.g. |---|---|)
        data_lines = []
        header_line = None
        for i, line in enumerate(lines):
            cleaned = line.strip("|").strip()
            if re.match(r'^[\s\-:|]+$', cleaned):
                continue
            if header_line is None:
                header_line = line
            else:
                data_lines.append(line)

        if header_line is None:
            return TableStructure(detected_format="markdown")

        headers = [h.strip() for h in header_line.strip("|").split("|")]
        col_count = len(headers)
        max_cell_w = max((len(h) for h in headers), default=0)

        for line in data_lines:
            cells = [c.strip() for c in line.strip("|").split("|")]
            for c in cells:
                max_cell_w = max(max_cell_w, len(c))

        has_multirow = any("." in h and not h.endswith(".") for h in headers)
        header_levels = [headers]

        avg_w = sum(len(h) for h in headers) / max(len(headers), 1)
        row_width = avg_w * col_count
        width_class = cls._classify_width(row_width, col_count)

        return TableStructure(
            col_count=col_count,
            data_row_count=len(data_lines),
            has_multirow_header=has_multirow,
            max_cell_width=max_cell_w,
            estimated_width_class=width_class,
            header_levels=header_levels,
            subgroup_row_count=0,
            detected_format="markdown",
        )

    @classmethod
    def _classify_width(cls, row_width: float, col_count: int) -> str:
        """
        Classify table width based on estimated row character width and column count.
        - **Description**:
            - Uses both character width and column count for classification.
            - A 14-col table is always at least "wide" regardless of cell widths.
        """
        if col_count >= 10 or row_width >= cls._WIDTH_THRESHOLDS["wide"]:
            return "very_wide"
        if col_count >= 7 or row_width >= cls._WIDTH_THRESHOLDS["medium"]:
            return "wide"
        if col_count >= 5 or row_width >= cls._WIDTH_THRESHOLDS["narrow"]:
            return "medium"
        return "narrow"


# Legacy prompt kept for backward compatibility
TABLE_CONVERSION_PROMPT = """You are an expert LaTeX typesetter. Convert the following table data into a properly formatted LaTeX table.

## Table Information
- **Label**: {label}
- **Caption**: {caption}

## Table Data (in any format)
```
{content}
```

## Requirements
1. Generate a complete LaTeX table environment with \\begin{{table}}...\\end{{table}}
2. Use \\centering for the table
3. Use booktabs style (\\toprule, \\midrule, \\bottomrule)
4. Include the caption and label
5. Use appropriate column alignment (l for text, c for short items, r for numbers)
6. If numbers represent best results, make them bold with \\textbf{{}}
7. Handle any special characters that need escaping in LaTeX
8. Use [t] or [h] for table placement

## Output
Output ONLY the LaTeX code, no explanations or markdown code blocks.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Enhanced prompt builder — context-aware conversion
# ═══════════════════════════════════════════════════════════════════════════

def build_conversion_prompt(
    label: str,
    caption: str,
    content: str,
    structure: TableStructure,
    column_format: str = "double",
    return_max_tokens: bool = False,
) -> Any:
    """
    Build a context-aware LaTeX table conversion prompt.
    - **Description**:
        - Injects structural metadata (row/col counts, width class) into the prompt
          so the LLM produces faithful, layout-correct LaTeX.
        - Dynamically adjusts guidance for multi-level headers, wide tables,
          sub-group rows, and template column format.

    - **Args**:
        - `label` (str): LaTeX label for the table.
        - `caption` (str): Table caption.
        - `content` (str): Raw table data (CSV or Markdown).
        - `structure` (TableStructure): Pre-analyzed structural metadata.
        - `column_format` (str): Template column format ("single" or "double").
        - `return_max_tokens` (bool): If True, return (prompt, max_tokens) tuple.

    - **Returns**:
        - `str`: The formatted prompt, or `(str, int)` if return_max_tokens is True.
    """
    parts: List[str] = []

    caption = normalize_caption(caption)

    parts.append(
        "You are an expert LaTeX typesetter. Convert the following table data "
        "into a properly formatted LaTeX table.\n"
    )

    # -- Table metadata --
    parts.append(f"## Table Information")
    parts.append(f"- **Label**: {label}")
    parts.append(f"- **Caption**: {caption}")
    parts.append(f"- **Source data has {structure.col_count} columns and "
                 f"{structure.data_row_count} data rows**")
    parts.append("")

    # -- Raw data --
    parts.append("## Table Data")
    parts.append("```")
    parts.append(content.strip())
    parts.append("```")
    parts.append("")

    # -- Data fidelity rules --
    parts.append("## CRITICAL — Data Preservation Rules")
    parts.append(
        f"- You MUST include ALL rows ({structure.data_row_count} data rows) "
        f"and ALL columns ({structure.col_count} columns) from the source data."
    )
    parts.append("- Do NOT summarize, truncate, or omit any rows or columns.")
    parts.append("- Preserve all numeric values exactly as given.")
    parts.append("- Use '-' for missing/empty values.")
    parts.append("")

    # -- Layout rules based on structure --
    parts.append("## Layout Requirements")

    # Environment type and sizing
    is_wide = structure.estimated_width_class in ("wide", "very_wide")

    if column_format == "double" and is_wide:
        parts.append(
            f"- Use \\begin{{table*}}...\\end{{table*}} (spans both columns) "
            f"since this table has {structure.col_count} columns."
        )
        parts.append(
            "- Wrap the tabular inside \\resizebox{\\textwidth}{!}{...} "
            "to fit within the page width."
        )
        if structure.estimated_width_class == "very_wide":
            parts.append(
                "- Use \\small or \\scriptsize font size inside the table "
                "to improve readability of this very wide table."
            )
    elif column_format == "double" and not is_wide:
        parts.append(
            "- Use \\begin{table}...\\end{table} (single column width)."
        )
    else:
        parts.append(
            "- Use \\begin{table}...\\end{table} with full page width."
        )

    parts.append("- Use \\centering for the table.")
    parts.append("- Use [htbp] for table placement.")
    parts.append("- Use booktabs style (\\toprule, \\midrule, \\bottomrule).")
    parts.append(f"- Include \\caption{{{caption}}} and \\label{{{label}}}.")
    parts.append(
        "- Use appropriate column alignment: l for text, c for short items, "
        "r for numbers."
    )
    parts.append(
        "- Handle any special characters (%, &, _, #, $, ~, ^) that need "
        "escaping in LaTeX."
    )
    parts.append("")

    # -- Multi-level header guidance --
    if structure.has_multirow_header:
        parts.append("## Multi-Level Header Handling")
        parts.append(
            "- The source data has hierarchical (dot-separated) column headers."
        )
        parts.append(
            "- Use \\multicolumn{n}{c}{Group Header} to merge cells for "
            "top-level header groups."
        )
        parts.append(
            "- Use \\cmidrule{start-end} to separate header groups visually."
        )
        if len(structure.header_levels) >= 2:
            top_groups = []
            seen = set()
            for h in structure.header_levels[0]:
                if h and h not in seen:
                    top_groups.append(h)
                    seen.add(h)
            parts.append(
                f"- Top-level groups detected: {', '.join(top_groups[:8])}"
            )
        parts.append("")

    # -- Sub-group rows --
    if structure.subgroup_row_count > 0:
        parts.append("## Sub-Group Row Handling")
        parts.append(
            f"- The table contains {structure.subgroup_row_count} sub-group "
            f"divider rows (rows where only the first cell has content)."
        )
        parts.append(
            "- Render each sub-group label spanning all columns using "
            "\\multicolumn{N}{l}{\\textit{Sub-group Name}}."
        )
        parts.append(
            "- Insert \\midrule before each sub-group divider for visual separation."
        )
        parts.append("")

    # -- Output --
    parts.append("## Output")
    parts.append(
        "Output ONLY the LaTeX code. No explanations, no markdown code blocks."
    )

    prompt = "\n".join(parts)

    # Dynamic max_tokens based on table size
    base_tokens = 1500
    per_row = 80
    per_col_extra = 50
    multirow_extra = 500 if structure.has_multirow_header else 0
    max_tokens = (
        base_tokens
        + structure.data_row_count * per_row
        + max(0, structure.col_count - 5) * per_col_extra
        + multirow_extra
    )
    max_tokens = min(max_tokens, 8000)

    if return_max_tokens:
        return prompt, max_tokens
    return prompt


# ═══════════════════════════════════════════════════════════════════════════
# ValidationResult & TableValidator — post-processing validation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """
    Result of validating LLM-generated LaTeX table output.
    - **Description**:
        - Captures errors (hard failures) and warnings (soft issues).
        - `is_valid` is True only when there are zero errors.
    """
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


class TableValidator:
    """
    Validate and auto-fix LLM-generated LaTeX table output.
    - **Description**:
        - Checks structural consistency (row/col counts, label, booktabs).
        - Auto-fixes missing labels and adds \\resizebox for wide tables.
    """

    @classmethod
    def validate(
        cls,
        latex: str,
        structure: TableStructure,
        expected_label: str,
    ) -> ValidationResult:
        """
        Validate LaTeX table output against expected structure.
        - **Args**:
            - `latex` (str): LLM-generated LaTeX.
            - `structure` (TableStructure): Expected structure from source data.
            - `expected_label` (str): Expected \\label value.
        - **Returns**:
            - `ValidationResult`: Errors and warnings.
        """
        result = ValidationResult()

        # Check label
        if f"\\label{{{expected_label}}}" not in latex:
            result.errors.append(
                f"Missing \\label{{{expected_label}}} in generated LaTeX."
            )

        # Check caption
        if "\\caption" not in latex:
            result.errors.append("Missing \\caption in generated LaTeX.")

        # Count data rows in the tabular (lines with \\)
        tabular_rows = cls._count_tabular_rows(latex)
        if structure.data_row_count > 0 and tabular_rows > 0:
            # Allow header row(s) + data rows; just check data portion
            # The first row after \toprule/\midrule is usually the header
            expected_min = max(1, structure.data_row_count - 1)
            if tabular_rows < expected_min:
                result.errors.append(
                    f"Row count mismatch: expected at least "
                    f"{structure.data_row_count} data rows, "
                    f"found {tabular_rows} total tabular rows."
                )

        # Check column count in tabular spec
        latex_col_count = cls._count_tabular_cols(latex)
        if (structure.col_count > 0 and latex_col_count > 0
                and abs(latex_col_count - structure.col_count) > 2):
            result.warnings.append(
                f"Column count mismatch: expected {structure.col_count}, "
                f"found {latex_col_count} in tabular spec."
            )

        # Check booktabs
        if "\\toprule" not in latex or "\\bottomrule" not in latex:
            result.warnings.append(
                "Missing booktabs commands (\\toprule/\\bottomrule). "
                "Consider using booktabs style."
            )

        return result

    @classmethod
    def auto_fix(
        cls,
        latex: str,
        structure: TableStructure,
        expected_label: str,
        column_format: str = "double",
    ) -> str:
        """
        Auto-fix common issues in LLM-generated LaTeX table.
        - **Args**:
            - `latex` (str): LLM-generated LaTeX.
            - `structure` (TableStructure): Expected structure.
            - `expected_label` (str): Label to ensure.
            - `column_format` (str): Template column format.
        - **Returns**:
            - `str`: Fixed LaTeX.
        """
        fixed = latex

        # Fix 1: Insert missing label
        if f"\\label{{{expected_label}}}" not in fixed:
            # Insert after \caption{...}
            caption_match = re.search(r'(\\caption\{[^}]*\})', fixed)
            if caption_match:
                insert_pos = caption_match.end()
                fixed = (
                    fixed[:insert_pos]
                    + f"\\label{{{expected_label}}}"
                    + fixed[insert_pos:]
                )
            else:
                # Insert before \end{table...}
                fixed = re.sub(
                    r'(\\end\{table\*?\})',
                    f"\\label{{{expected_label}}}\n\\1",
                    fixed,
                    count=1,
                )

        # Fix 2: Add resizebox for wide tables
        is_wide = structure.estimated_width_class in ("wide", "very_wide")
        if is_wide and column_format == "double" and "\\resizebox" not in fixed:
            # Determine target width
            if "table*" in fixed:
                target_width = "\\textwidth"
            else:
                target_width = "\\columnwidth"

            # Wrap tabular in resizebox
            tabular_pattern = re.compile(
                r'(\\begin\{tabular[*]?\}\{[^}]*\}.*?\\end\{tabular[*]?\})',
                re.DOTALL,
            )
            match = tabular_pattern.search(fixed)
            if match:
                original = match.group(0)
                wrapped = (
                    f"\\resizebox{{{target_width}}}{{!}}{{\n"
                    f"{original}\n"
                    f"}}"
                )
                fixed = fixed[:match.start()] + wrapped + fixed[match.end():]

        return fixed

    @staticmethod
    def _count_tabular_rows(latex: str) -> int:
        """Count rows in tabular environment (lines ending with \\\\)."""
        tabular_match = re.search(
            r'\\begin\{tabular[*]?\}.*?(\\end\{tabular[*]?\})',
            latex, re.DOTALL,
        )
        if not tabular_match:
            return 0
        body = tabular_match.group(0)
        # Count lines with \\ (row terminators), excluding rule commands
        rows = re.findall(r'\\\\', body)
        return len(rows)

    @staticmethod
    def _count_tabular_cols(latex: str) -> int:
        """Count columns from tabular column spec like {lccc} or {lp{3cm}cc}."""
        col_spec = _extract_tabular_col_spec(latex)
        if not col_spec:
            return 0
        return _count_cols_from_spec(col_spec)


# ═══════════════════════════════════════════════════════════════════════════
# smart_promote_wide_tables — content-aware table layout optimization
# ═══════════════════════════════════════════════════════════════════════════

# Thresholds for smart promotion decisions
_PROMOTE_TO_TABLE_STAR_MIN_COLS = 9
_RESIZEBOX_MIN_COLS = 4
_RESIZEBOX_MIN_CONTENT_WIDTH = 55
_MULTICOLUMN_WIDE_MIN_SPAN = 3
_FONT_SHRINK_MIN_COLS = 12


def _extract_tabular_col_spec(content: str) -> Optional[str]:
    """
    Extract the column spec from \\begin{tabular}{...}, handling nested braces.
    - **Description**:
        - Standard regex [^}]* fails for specs like p{3cm}. This function
          uses brace-depth counting to correctly extract the full spec.
    """
    match = re.search(r'\\begin\{tabular[*]?\}\{', content)
    if not match:
        return None

    start = match.end()
    depth = 1
    i = start
    while i < len(content) and depth > 0:
        if content[i] == '{':
            depth += 1
        elif content[i] == '}':
            depth -= 1
        i += 1

    if depth == 0:
        return content[start:i - 1]
    return None


def _count_cols_from_spec(col_spec: str) -> int:
    """
    Count columns from a LaTeX tabular column spec, handling p{width} etc.
    - **Description**:
        - Strips brace content (e.g. {3cm}) before counting column letters
          to avoid false positives from characters inside width specs.
    """
    stripped = re.sub(r'\{[^}]*\}', '', col_spec)
    return len(re.findall(r'[lcrpXm]', stripped, re.IGNORECASE))


def _estimate_max_tabular_row_width(body: str) -> int:
    """
    Estimate max raw text width across tabular rows (no booktabs required).
    - **Description**:
        - Scans lines between ``\\begin{tabular}`` and ``\\end{tabular}`` that
          look like data rows (contain ``&`` and row terminator ``\\\\``).
        - Complements ``_estimate_row_width`` for tables without ``\\toprule``.

    - **Args**:
        - `body` (str): Table environment body (may include caption, etc.).

    - **Returns**:
        - `int`: Maximum cleaned character width among detected rows.
    """
    tab_m = re.search(r'\\begin\{tabular[*]?\}\{[^}]*\}', body)
    if not tab_m:
        return 0
    rest = body[tab_m.end():]
    end_m = re.search(r'\\end\{tabular[*]?\}', rest)
    if not end_m:
        return 0
    chunk = rest[: end_m.start()]
    max_w = 0
    for line in chunk.split("\n"):
        stripped = line.strip()
        if "&" not in stripped or "\\\\" not in stripped:
            continue
        if stripped.startswith("%"):
            continue
        clean = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', stripped)
        clean = re.sub(r'\\[a-zA-Z]+', '', clean)
        clean = clean.replace("\\\\", "").replace("&", "").strip()
        max_w = max(max_w, len(clean))
    return max_w


def _max_multicolumn_span(body: str) -> int:
    """
    Return the largest N in ``\\multicolumn{N}{...}`` within the table body.
    - **Args**:
        - `body` (str): Table environment body.

    - **Returns**:
        - `int`: Maximum span, or 0 if none.
    """
    spans = []
    for m in re.finditer(r'\\multicolumn\{(\d+)\}', body):
        try:
            spans.append(int(m.group(1)))
        except ValueError:
            continue
    return max(spans) if spans else 0


def smart_promote_wide_tables(content: str) -> str:
    """
    Intelligently promote wide tables and add \\resizebox for double-column layouts.
    - **Description**:
        - Replaces the naive column-count-only heuristic with a decision
          based on both column count and content width.
        - Applies three tiers of intervention:
          1. \\resizebox only — for moderately wide tables (5-8 cols or wide content)
          2. Promote to table* + \\resizebox — for very wide tables (9+ cols)
          3. Add \\scriptsize — for extremely wide tables (12+ cols)
        - Skips tables that already have \\resizebox.

    - **Args**:
        - `content` (str): LaTeX content (may contain multiple table envs).

    - **Returns**:
        - `str`: Content with smart table promotions applied.
    """
    # Match both table and table* environments
    env_pattern = re.compile(
        r'(\\begin\{(table\*?)\})(.*?)(\\end\{\2\})',
        re.DOTALL,
    )

    def _process_table(m: re.Match) -> str:
        begin_tag = m.group(1)
        env_name = m.group(2)
        body = m.group(3)
        end_tag = m.group(4)

        # Skip if already has resizebox
        if "\\resizebox" in body:
            return m.group(0)

        # Extract column spec from tabular (handles nested braces like p{3cm})
        col_spec = _extract_tabular_col_spec(body)
        if col_spec is None:
            return m.group(0)

        col_count = _count_cols_from_spec(col_spec)

        # Estimate content width from first data row (booktabs) and any row
        content_width = _estimate_row_width(body)
        loose_width = _estimate_max_tabular_row_width(body)
        row_width = max(content_width, loose_width)
        mc_span = _max_multicolumn_span(body)

        needs_resize = (
            col_count >= _RESIZEBOX_MIN_COLS
            or row_width >= _RESIZEBOX_MIN_CONTENT_WIDTH
            or (
                mc_span >= _MULTICOLUMN_WIDE_MIN_SPAN
                and col_count >= 3
            )
        )
        needs_promote = (
            col_count >= _PROMOTE_TO_TABLE_STAR_MIN_COLS
            and env_name == "table"
        )
        needs_font_shrink = col_count >= _FONT_SHRINK_MIN_COLS

        new_body = body

        # Tier 3: Add font size command for very wide tables
        if needs_font_shrink:
            # Insert \scriptsize after \centering (or at start of body)
            centering_match = re.search(r'(\\centering\s*\n?)', new_body)
            if centering_match and "\\scriptsize" not in new_body and "\\small" not in new_body:
                insert_pos = centering_match.end()
                new_body = (
                    new_body[:insert_pos]
                    + "\\scriptsize\n"
                    + new_body[insert_pos:]
                )

        # Tier 1/2: Add resizebox around tabular
        if needs_resize:
            target_width = "\\textwidth" if (needs_promote or env_name == "table*") else "\\columnwidth"
            tabular_full = re.compile(
                r'(\\begin\{tabular[*]?\}\{[^}]*\}.*?\\end\{tabular[*]?\})',
                re.DOTALL,
            )
            tab_match = tabular_full.search(new_body)
            if tab_match:
                original = tab_match.group(0)
                wrapped = (
                    f"\\resizebox{{{target_width}}}{{!}}{{\n"
                    f"{original}\n"
                    f"}}"
                )
                new_body = new_body[:tab_match.start()] + wrapped + new_body[tab_match.end():]

        # Tier 2: Promote table -> table*
        if needs_promote:
            return f"\\begin{{table*}}{new_body}\\end{{table*}}"

        return f"{begin_tag}{new_body}{end_tag}"

    return env_pattern.sub(_process_table, content)


def add_adjustbox_safety(content: str) -> str:
    """
    Wrap each ``tabular`` in ``\\adjustbox{max width=...}`` when not already scaled.
    - **Description**:
        - For double-column papers, shrinks tables that would overflow the column.
        - Skips environments that already use ``\\resizebox`` or ``\\adjustbox``.
        - Requires ``\\usepackage{adjustbox}`` (auto-injected by Typesetter when missing).

    - **Args**:
        - `content` (str): LaTeX fragment (e.g. one section body).

    - **Returns**:
        - `str`: Content with optional ``adjustbox`` wrappers added.
    """
    env_pattern = re.compile(
        r'(\\begin\{(table\*?)\})(.*?)(\\end\{\2\})',
        re.DOTALL,
    )
    tabular_full = re.compile(
        r'(\\begin\{tabular[*]?\}\{[^}]*\}.*?\\end\{tabular[*]?\})',
        re.DOTALL,
    )

    def _wrap(m: re.Match[str]) -> str:
        begin_tag = m.group(1)
        env_name = m.group(2)
        body = m.group(3)
        end_tag = m.group(4)
        if "\\adjustbox" in body or "\\resizebox" in body:
            return m.group(0)
        tab_match = tabular_full.search(body)
        if not tab_match:
            return m.group(0)
        original = tab_match.group(0)
        width = "\\textwidth" if env_name == "table*" else "\\columnwidth"
        wrapped_tab = (
            f"\\adjustbox{{max width={width},center}}{{\n{original}\n}}"
        )
        new_body = body[: tab_match.start()] + wrapped_tab + body[tab_match.end() :]
        return f"{begin_tag}{new_body}{end_tag}"

    return env_pattern.sub(_wrap, content)


def _estimate_row_width(body: str) -> int:
    """
    Estimate the character width of the widest data row in the tabular.
    - **Description**:
        - Finds the first row after \\midrule (or \\toprule) and measures
          total character length of cell contents.
    """
    lines = body.split('\n')
    in_data = False
    max_width = 0

    for line in lines:
        stripped = line.strip()
        if '\\midrule' in stripped or '\\toprule' in stripped:
            in_data = True
            continue
        if '\\bottomrule' in stripped or '\\end{tabular' in stripped:
            break
        if in_data and '\\\\' in stripped:
            # Remove LaTeX commands for width estimation
            clean = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', stripped)
            clean = re.sub(r'\\[a-zA-Z]+', '', clean)
            clean = clean.replace('\\\\', '').replace('&', '').strip()
            max_width = max(max_width, len(clean))

    return max_width


def _escape_latex_text(text: str) -> str:
    """
    Escape plain text for safe insertion into LaTeX text commands.
    - **Description**:
        - Escapes common LaTeX special characters in user-provided text.
        - Intended for metadata strings rendered in headings or prose.

    - **Args**:
        - `text` (str): Raw text.

    - **Returns**:
        - `escaped_text` (str): LaTeX-safe text.
    """
    raw = text or ""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in raw)


def _parse_table_rows(content: str) -> List[List[str]]:
    """
    Parse raw table content into rows.
    - **Description**:
        - Supports Markdown pipe tables and CSV text.
        - Skips Markdown separator rows like ``|---|---|``.
        - Returns normalized trimmed cells.

    - **Args**:
        - `content` (str): Raw table content.

    - **Returns**:
        - `rows` (List[List[str]]): Parsed rows.
    """
    raw = (content or "").strip()
    if not raw:
        return []

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if lines and lines[0].startswith("|"):
        rows: List[List[str]] = []
        for line in lines:
            body = line.strip("|").strip()
            if re.match(r'^[\s\-:|]+$', body):
                continue
            rows.append([cell.strip() for cell in body.split("|")])
        return rows

    parsed_rows: List[List[str]] = []
    for row in csv.reader(io.StringIO(raw)):
        parsed_rows.append([cell.strip() for cell in row])
    return parsed_rows


def _build_table_latex_from_source(
    table_id: str,
    caption: str,
    content: str,
) -> str:
    """
    Build a deterministic LaTeX table from source content.
    - **Description**:
        - Converts parsed rows directly into a booktabs tabular.
        - Preserves row/column structure without LLM conversion.
        - Uses '-' for missing values.

    - **Args**:
        - `table_id` (str): Table label id.
        - `caption` (str): Raw caption text.
        - `content` (str): Raw table content.

    - **Returns**:
        - `table_latex` (str): Deterministic table LaTeX.
    """
    rows = _parse_table_rows(content)
    if not rows:
        raise ValueError(f"Cannot build table '{table_id}' from empty source content")

    col_count = max(len(row) for row in rows)
    normalized_rows: List[List[str]] = []
    for row in rows:
        padded = row + [""] * max(0, col_count - len(row))
        normalized_rows.append([_escape_latex_text(cell or "-") for cell in padded])

    header = normalized_rows[0]
    data_rows = normalized_rows[1:]
    col_spec = "l" + ("c" * (col_count - 1))
    safe_caption = _escape_latex_text(normalize_caption(caption) or table_id)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{safe_caption}}}",
        f"\\label{{{table_id}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for row in data_rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def build_table_preview_document(
    table_latex: str,
    column_format: str = "double",
    title: str = "Table Preview",
) -> str:
    """
    Build a standalone LaTeX document for single-table visual preview.
    - **Description**:
        - Wraps one table LaTeX block into a minimal compile-ready document.
        - Supports single/double-column preview mode via documentclass options.
        - Preserves the original table environment content as-is.

    - **Args**:
        - `table_latex` (str): Converted LaTeX table environment.
        - `column_format` (str): "single" or "double" layout preview mode.
        - `title` (str): Preview document title.

    - **Returns**:
        - `preview_tex` (str): Standalone .tex source for table-only preview.
    """
    clean_table = _strip_code_fences((table_latex or "").strip())
    if not clean_table:
        clean_table = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Table Preview}\n"
            "\\label{tab:preview}\n"
            "\\begin{tabular}{lc}\n"
            "\\toprule\n"
            "A & B \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}"
        )

    if "\\begin{table" not in clean_table:
        clean_table = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            f"{clean_table}\n"
            "\\end{table}"
        )

    class_opts = "[twocolumn]" if column_format == "double" else ""
    safe_title = _escape_latex_text(title)
    return (
        f"\\documentclass{class_opts}{{article}}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{adjustbox}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\begin{document}\n"
        f"\\section*{{{safe_title}}}\n"
        f"{clean_table}\n"
        "\\end{document}\n"
    )


def build_table_preview_documents(
    tables: list,
    converted_tables: Dict[str, str],
    column_format: str = "double",
    allow_placeholder: bool = False,
    base_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build standalone table preview documents for converted metadata tables.
    - **Description**:
        - Creates one compile-ready LaTeX document per table ID.
        - Uses converted table LaTeX when available.
        - Falls back to a labeled placeholder table if conversion is missing.

    - **Args**:
        - `tables` (list): List of table specs (TableSpec-like objects).
        - `converted_tables` (Dict[str, str]): Mapping table_id -> converted table LaTeX.
        - `column_format` (str): "single" or "double" preview layout mode.
        - `allow_placeholder` (bool): Whether to allow placeholder table fallback.
        - `base_path` (str, optional): Base path for relative `table.file_path` loading.

    - **Returns**:
        - `preview_docs` (Dict[str, str]): Mapping table_id -> standalone preview tex.
    """
    preview_docs: Dict[str, str] = {}
    converted = converted_tables or {}

    for table in tables or []:
        table_id = getattr(table, "id", None) or ""
        if not table_id:
            continue

        table_latex = converted.get(table_id, "")
        if not table_latex:
            source_content = _read_table_content(table, base_path=base_path) or ""
            if source_content.strip():
                table_latex = _build_table_latex_from_source(
                    table_id=table_id,
                    caption=getattr(table, "caption", "") or table_id,
                    content=source_content,
                )
            elif allow_placeholder:
                caption = _escape_latex_text(
                    normalize_caption(getattr(table, "caption", "") or table_id),
                )
                table_latex = (
                    "\\begin{table}[htbp]\n"
                    "\\centering\n"
                    f"\\caption{{{caption}}}\n"
                    f"\\label{{{table_id}}}\n"
                    "\\begin{tabular}{lc}\n"
                    "\\toprule\n"
                    "Column & Value \\\\\n"
                    "\\bottomrule\n"
                    "\\end{tabular}\n"
                    "\\end{table}"
                )
            else:
                raise ValueError(
                    f"Missing converted table and source content for '{table_id}'",
                )

        preview_docs[table_id] = build_table_preview_document(
            table_latex=table_latex,
            column_format=column_format,
            title=f"Preview {table_id}",
        )

    return preview_docs


def _sanitize_table_preview_filename(table_id: str) -> str:
    """
    Convert a table id into a filesystem-safe filename stem.
    - **Description**:
        - Replaces unsupported filename characters with underscores.
        - Preserves alphanumeric, dot, underscore, and dash characters.

    - **Args**:
        - `table_id` (str): Table identifier such as "tab:results".

    - **Returns**:
        - `safe_name` (str): Safe filename stem.
    """
    raw = (table_id or "").strip()
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", raw) or "table_preview"
    suffix = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned}_{suffix}"


def _extract_latex_warnings_from_log(log_content: str) -> List[str]:
    """
    Extract warning messages from LaTeX log content.
    - **Description**:
        - Collects classic LaTeX warning forms.
        - De-duplicates while preserving message order.

    - **Args**:
        - `log_content` (str): Full log text.

    - **Returns**:
        - `warnings` (List[str]): Warning messages.
    """
    candidates: List[str] = []
    patterns = [
        r"LaTeX Warning:\s*(.*?)(?:\n|$)",
        r"Warning:\s*(.*?)(?:\n|$)",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, log_content or ""):
            msg = (match or "").strip()
            if msg:
                candidates.append(msg)

    deduped: List[str] = []
    seen = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _extract_latex_errors_from_log(log_content: str) -> List[str]:
    """
    Extract error messages from LaTeX log content.
    - **Description**:
        - Captures lines beginning with "!" as primary LaTeX errors.
        - Also captures generic "Error:" lines.

    - **Args**:
        - `log_content` (str): Full log text.

    - **Returns**:
        - `errors` (List[str]): Error messages.
    """
    candidates: List[str] = []
    for line in (log_content or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("!"):
            msg = stripped[1:].strip()
            if msg:
                candidates.append(msg)
        elif "Error:" in stripped:
            msg = stripped.split("Error:", 1)[1].strip()
            if msg:
                candidates.append(msg)

    deduped: List[str] = []
    seen = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def compile_table_preview_document(
    table_id: str,
    preview_tex: str,
    output_dir: str,
    max_passes: int = 2,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    """
    Compile one standalone table preview document with pdflatex.
    - **Description**:
        - Writes preview tex into output_dir and runs pdflatex passes.
        - Returns a structured result with artifact paths and diagnostics.
        - Intended for table-only visual validation loops.

    - **Args**:
        - `table_id` (str): Table identifier.
        - `preview_tex` (str): Standalone LaTeX source.
        - `output_dir` (str): Directory for .tex/.pdf/.log artifacts.
        - `max_passes` (int): Number of pdflatex passes.
        - `timeout_seconds` (int): Timeout per pdflatex invocation.

    - **Returns**:
        - `result` (Dict[str, Any]): Compilation result details.
    """
    resolved_output_dir = os.path.abspath(output_dir)
    os.makedirs(resolved_output_dir, exist_ok=True)
    safe_name = _sanitize_table_preview_filename(table_id)
    tex_filename = f"{safe_name}.tex"
    tex_path = os.path.join(resolved_output_dir, tex_filename)
    pdf_path = os.path.join(resolved_output_dir, f"{safe_name}.pdf")
    log_path = os.path.join(resolved_output_dir, f"{safe_name}.log")

    result: Dict[str, Any] = {
        "table_id": table_id,
        "success": False,
        "tex_path": tex_path,
        "pdf_path": pdf_path,
        "log_path": log_path,
        "warnings": [],
        "errors": [],
        "attempts": 0,
    }

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(preview_tex or "")

    try:
        final_returncode = 0
        passes = max(1, int(max_passes or 1))
        for attempt in range(passes):
            result["attempts"] = attempt + 1
            proc = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-output-directory",
                    resolved_output_dir,
                    tex_filename,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                cwd=resolved_output_dir,
            )
            final_returncode = proc.returncode
            if proc.returncode != 0:
                break

        log_content = ""
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                log_content = f.read()
            result["warnings"] = _extract_latex_warnings_from_log(log_content)
            result["errors"] = _extract_latex_errors_from_log(log_content)

        if final_returncode != 0 and not result["errors"]:
            result["errors"].append("pdflatex exited with non-zero return code")

        result["success"] = os.path.exists(pdf_path) and final_returncode == 0
        return result
    except subprocess.TimeoutExpired:
        result["errors"].append("pdflatex timeout")
        return result
    except FileNotFoundError:
        result["errors"].append("pdflatex not found")
        return result
    except Exception as e:
        result["errors"].append(str(e))
        return result


def compile_table_preview_documents(
    preview_docs: Dict[str, str],
    output_dir: str,
    max_passes: int = 2,
    timeout_seconds: int = 60,
) -> Dict[str, Dict[str, Any]]:
    """
    Compile multiple standalone table preview documents.
    - **Description**:
        - Runs table preview compilation per table id.
        - Returns per-table structured outcomes for downstream checks.

    - **Args**:
        - `preview_docs` (Dict[str, str]): Table id to preview tex mapping.
        - `output_dir` (str): Root output directory for artifacts.
        - `max_passes` (int): Number of pdflatex passes per table.
        - `timeout_seconds` (int): Timeout per pdflatex call.

    - **Returns**:
        - `results` (Dict[str, Dict[str, Any]]): Table id to result mapping.
    """
    results: Dict[str, Dict[str, Any]] = {}
    os.makedirs(output_dir, exist_ok=True)

    for table_id, preview_tex in (preview_docs or {}).items():
        results[table_id] = compile_table_preview_document(
            table_id=table_id,
            preview_tex=preview_tex,
            output_dir=output_dir,
            max_passes=max_passes,
            timeout_seconds=timeout_seconds,
        )

    return results


def _read_table_content(
    table: "TableSpec",
    base_path: Optional[str] = None,
) -> Optional[str]:
    """
    Read raw table content from file_path or inline content.
    - **Args**:
        - `table` (TableSpec): Table specification.
        - `base_path` (str, optional): Base path for resolving file_path.
    - **Returns**:
        - `str`: Raw table content, or None.
    """
    if table.file_path:
        if base_path and not os.path.isabs(table.file_path):
            file_path = os.path.join(base_path, table.file_path)
        else:
            file_path = table.file_path

        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info("table_converter.read_file path=%s", file_path)
                return content
            else:
                logger.warning(
                    "table_converter.file_not_found path=%s", file_path,
                )
                if table.content:
                    logger.info(
                        "table_converter.fallback_inline_content id=%s",
                        getattr(table, "id", "unknown"),
                    )
                    return table.content
        except Exception as e:
            logger.error(
                "table_converter.file_read_error path=%s error=%s",
                file_path, str(e),
            )
            if table.content:
                logger.info(
                    "table_converter.fallback_inline_after_error id=%s",
                    getattr(table, "id", "unknown"),
                )
                return table.content
    else:
        return table.content
    return None


def _strip_code_fences(text: str) -> str:
    """Remove markdown code block markers from LLM output."""
    if text.startswith("```"):
        lines = text.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return '\n'.join(lines)
    return text


async def convert_table_to_latex(
    table: "TableSpec",
    llm_client: Any,
    model_name: str,
    base_path: Optional[str] = None,
    column_format: str = "double",
) -> Optional[str]:
    """
    Convert a TableSpec to LaTeX format using LLM with enhanced pipeline.
    - **Description**:
        - Pre-analyzes table structure with TableAnalyzer.
        - Builds context-aware prompt via build_conversion_prompt.
        - Validates and auto-fixes LLM output via TableValidator.

    - **Args**:
        - `table` (TableSpec): Table specification.
        - `llm_client`: OpenAI-compatible async client.
        - `model_name` (str): Model to use for conversion.
        - `base_path` (str, optional): Base path for resolving file_path.
        - `column_format` (str): Template column format ("single" or "double").

    - **Returns**:
        - `str`: Complete LaTeX table code, or None if conversion fails.
    """
    if table.auto_generate:
        logger.warning(
            "table_converter.auto_generate_not_implemented id=%s", table.id,
        )
        return None

    content = _read_table_content(table, base_path)
    if not content:
        logger.warning("table_converter.no_content id=%s", table.id)
        return None

    # Phase 1: Structural analysis
    structure = TableAnalyzer.analyze(content)
    logger.info(
        "table_converter.analyzed id=%s cols=%d rows=%d width=%s multirow=%s",
        table.id, structure.col_count, structure.data_row_count,
        structure.estimated_width_class, structure.has_multirow_header,
    )

    # Phase 2: Build enhanced prompt
    prompt, max_tokens = build_conversion_prompt(
        label=table.id,
        caption=table.caption,
        content=content,
        structure=structure,
        column_format=column_format,
        return_max_tokens=True,
    )

    try:
        response = await llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert LaTeX typesetter specializing in "
                        "academic tables. Preserve ALL data exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )

        latex_content = response.choices[0].message.content.strip()
        latex_content = _strip_code_fences(latex_content)

        # Phase 3: Validate and auto-fix
        validation = TableValidator.validate(
            latex_content, structure, table.id,
        )
        if not validation.is_valid:
            logger.warning(
                "table_converter.validation_errors id=%s errors=%s",
                table.id, validation.errors,
            )
            latex_content = TableValidator.auto_fix(
                latex_content, structure, table.id, column_format,
            )

        if validation.warnings:
            logger.info(
                "table_converter.validation_warnings id=%s warnings=%s",
                table.id, validation.warnings,
            )

        logger.info(
            "table_converter.success id=%s length=%d",
            table.id, len(latex_content),
        )

        return latex_content

    except Exception as e:
        logger.error(
            "table_converter.llm_error id=%s error=%s",
            table.id, str(e),
        )
        return None


async def convert_tables(
    tables: list,
    llm_client: Any,
    model_name: str,
    base_path: Optional[str] = None,
    column_format: str = "double",
) -> dict:
    """
    Convert multiple tables to LaTeX with enhanced pipeline.
    - **Args**:
        - `tables` (List[TableSpec]): List of table specifications.
        - `llm_client`: OpenAI-compatible async client.
        - `model_name` (str): Model to use.
        - `base_path` (str, optional): Base path for file resolution.
        - `column_format` (str): Template column format.
    - **Returns**:
        - `dict`: Mapping of table_id to LaTeX code.
    """
    converted = {}

    for table in tables:
        latex = await convert_table_to_latex(
            table=table,
            llm_client=llm_client,
            model_name=model_name,
            base_path=base_path,
            column_format=column_format,
        )
        if latex:
            converted[table.id] = latex

    logger.info(
        "table_converter.batch_complete total=%d converted=%d",
        len(tables), len(converted),
    )

    return converted


# =========================================================================
# Caption normalization
# =========================================================================

_CAPTION_PREFIX_RE = re.compile(
    r'^(?:Table|Figure|Tab\.|Fig\.|TABLE|FIGURE|FIG\.)\s*\d+[\.:]\s*',
    re.IGNORECASE,
)


def normalize_caption(caption: str) -> str:
    """
    Strip redundant numbering prefixes from captions.
    - **Description**:
        - LaTeX auto-generates "Table N." / "Figure N." via \\caption{}.
          If the source caption already contains such a prefix the rendered
          output will read "Table 1. Table 1. ...", causing duplication.
        - This function strips the leading prefix so \\caption{} produces
          the correct single-numbered caption.

    - **Args**:
        - `caption` (str): Raw caption text, possibly with numbering prefix.

    - **Returns**:
        - `str`: Caption with the redundant prefix removed.
    """
    if not caption:
        return ""
    return _CAPTION_PREFIX_RE.sub('', caption).strip()


# =========================================================================
# Float reference injection (Stage 3 of decomposed writer pipeline)
# =========================================================================

_FLOAT_MARKER_RE = re.compile(r'\[FLOAT:([^\]]+)\]')


def inject_float_refs(
    latex: str,
    figures_to_ref: List[str],
    tables_to_ref: List[str],
) -> str:
    """
    Replace [FLOAT:{id}] markers with proper LaTeX references.
    - **Description**:
        - Stage 3 of the decomposed writer pipeline.
        - Mechanically replaces markers placed by Stage 1 (core content).
        - Cleans up any orphan markers that don't match known IDs.

    - **Args**:
        - `latex` (str): LaTeX content with [FLOAT:...] markers.
        - `figures_to_ref` (List[str]): Figure IDs (e.g. "fig:arch").
        - `tables_to_ref` (List[str]): Table IDs (e.g. "tab:results").

    - **Returns**:
        - `str`: LaTeX with markers replaced by Table~\\ref / Figure~\\ref.
    """
    known_ids = set(figures_to_ref or []) | set(tables_to_ref or [])
    fig_set = set(figures_to_ref or [])

    def _replace(m: re.Match) -> str:
        fid = m.group(1)
        if fid in fig_set:
            return f"Figure~\\ref{{{fid}}}"
        if fid in known_ids:
            return f"Table~\\ref{{{fid}}}"
        return ""

    result = _FLOAT_MARKER_RE.sub(_replace, latex)
    result = re.sub(r'\s{2,}', ' ', result)
    return result


# =========================================================================
# Direct-injection helpers (post-Writer processing)
# =========================================================================

def strip_writer_tables(content: str, known_table_ids: set) -> str:
    """
    Remove \\begin{table}...\\end{table} blocks whose \\label matches a known ID.
    - **Description**:
        - Under the direct-injection model the Writer is told NOT to create table
          environments, but may still do so.  This function defensively strips
          any Writer-generated table environments for tables that will be
          injected from the pre-converted pool.
        - Tables whose label is NOT in *known_table_ids* are preserved (the Writer
          may legitimately create ad-hoc tables not in the metadata).

    - **Args**:
        - `content` (str): Section LaTeX content from the Writer.
        - `known_table_ids` (set): Table IDs that will be injected later.

    - **Returns**:
        - `str`: Content with matching table environments removed.
    """
    if not known_table_ids:
        return content

    for tbl_id in known_table_ids:
        escaped_id = re.escape(tbl_id)
        for env in ("table*", "table"):
            esc_env = re.escape(env)
            pattern = re.compile(
                rf'\\begin{{{esc_env}}}.*?\\label{{{escaped_id}}}.*?\\end{{{esc_env}}}\s*',
                re.DOTALL,
            )
            content = pattern.sub('', content)

    return content.strip() if content.strip() else content


def inject_tables(
    content: str,
    section_plan,
    tables,
    converted_tables: dict,
) -> str:
    """
    Inject pre-converted table environments at the first \\ref location.
    - **Description**:
        - For each table assigned to this section (via section_plan), finds the
          first ``Table~\\ref{tab:id}`` in the content and inserts the full
          table environment after the enclosing sentence.
        - If no \\ref is found, appends the table at the end.
        - Skips tables already defined in the content.
        - Ensures every injected table has a \\label.

    - **Args**:
        - `content` (str): Section LaTeX content (post strip_writer_tables).
        - `section_plan`: Section plan with ``get_table_ids_to_define()``.
        - `tables` (list): Table specifications (TableSpec or SimpleNamespace).
        - `converted_tables` (dict): table_id -> LaTeX code.

    - **Returns**:
        - `str`: Content with tables injected.
    """
    tables_to_define = section_plan.get_table_ids_to_define()
    if not tables_to_define:
        return content

    table_map = {}
    for tbl in (tables or []):
        tbl_id = tbl.id if hasattr(tbl, "id") else tbl.get("id", "")
        if tbl_id:
            table_map[tbl_id] = tbl

    _converted = converted_tables or {}

    for tbl_id in tables_to_define:
        tbl = table_map.get(tbl_id)
        if not tbl:
            continue

        already_pattern = re.compile(
            rf'\\begin{{table\*?}}.*?\\label{{{re.escape(tbl_id)}}}.*?\\end{{table\*?}}',
            re.DOTALL,
        )
        if already_pattern.search(content):
            continue

        env_name = "table*" if getattr(tbl, "wide", False) else "table"

        if tbl_id in _converted:
            table_latex = _converted[tbl_id]
            if f"\\label{{{tbl_id}}}" not in table_latex:
                label_str = f"\\label{{{tbl_id}}}"
                table_latex = re.sub(
                    rf'(\\end{{{env_name}}})',
                    lambda m, ls=label_str: f"{ls}\n{m.group(1)}",
                    table_latex,
                )
        else:
            caption = normalize_caption(getattr(tbl, "caption", "") or tbl_id)
            table_latex = (
                f"\\begin{{{env_name}}}[htbp]\n"
                f"\\centering\n"
                f"\\caption{{{caption}}}\\label{{{tbl_id}}}\n"
                f"\\begin{{tabular}}{{lcc}}\n"
                f"\\hline\n"
                f"Column 1 & Column 2 & Column 3 \\\\\n"
                f"\\hline\n"
                f"-- & -- & -- \\\\\n"
                f"\\hline\n"
                f"\\end{{tabular}}\n"
                f"\\end{{{env_name}}}"
            )

        ref_pattern = re.compile(
            rf'(Table~?\\ref{{{re.escape(tbl_id)}}}[^.]*\.)',
        )
        match = ref_pattern.search(content)
        if match:
            insert_pos = match.end()
            content = content[:insert_pos] + "\n" + table_latex + "\n" + content[insert_pos:]
        else:
            content = content + "\n" + table_latex + "\n"

    return content
