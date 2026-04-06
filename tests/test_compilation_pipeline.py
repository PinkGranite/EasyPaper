"""
TDD tests for the compilation pipeline.

Covers three distinct bugs:
1. _strip_leading_section_command fails on nested braces (e.g. \\section{Some \\textbf{Bold} Title})
2. _copy_to_output_dir only runs on success — iteration directories stay empty on failure
3. _assemble_paper never strips existing \\section{} from LLM content before prepending its own

RED phase: these tests should FAIL until the fixes are applied.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# 1. _strip_leading_section_command — nested brace handling
# ---------------------------------------------------------------------------

class TestStripLeadingSectionCommand:
    """Verify _strip_leading_section_command correctly removes
    leading \\section{...} commands, including titles with nested braces."""

    @staticmethod
    def _strip(content: str) -> str:
        from src.agents.typesetter_agent.typesetter_agent import TypesetterAgent
        return TypesetterAgent._strip_leading_section_command(content)

    def test_simple_section(self):
        """Simple \\section{Title} followed by body text."""
        content = r"\section{Introduction}" + "\nSome body text."
        result = self._strip(content)
        assert result == "Some body text."

    def test_section_star(self):
        """\\section*{Title} variant."""
        content = r"\section*{Appendix A}" + "\nAppendix content."
        result = self._strip(content)
        assert result == "Appendix content."

    def test_multiple_leading(self):
        """Multiple consecutive leading \\section{} should all be removed."""
        content = (
            r"\section{Results}" + "\n"
            r"\section{Result}" + "\n"
            "The results show..."
        )
        result = self._strip(content)
        assert result == "The results show..."

    def test_nested_braces_in_title(self):
        """\\section{Some \\textbf{Bold} Title} — nested braces must be handled."""
        content = r"\section{Some \textbf{Bold} Title}" + "\nBody text here."
        result = self._strip(content)
        assert result == "Body text here.", (
            f"Nested braces not handled correctly; got: {result!r}"
        )

    def test_nested_braces_multiple_levels(self):
        """\\section{\\texorpdfstring{$O(n^{2})$}{O(n2)}} — deeply nested."""
        content = r"\section{\texorpdfstring{$O(n^{2})$}{O(n2)}}" + "\nBody."
        result = self._strip(content)
        assert result == "Body.", (
            f"Deeply nested braces not handled; got: {result!r}"
        )

    def test_section_with_label(self):
        """\\section{Title}\\label{sec:intro} should both be stripped."""
        content = r"\section{Introduction}\label{sec:intro}" + "\nBody."
        result = self._strip(content)
        assert result == "Body."

    def test_subsection_not_stripped(self):
        """\\subsection{...} should NOT be stripped."""
        content = r"\subsection{Details}" + "\nSub content."
        result = self._strip(content)
        assert content.strip() == result

    def test_non_leading_section_preserved(self):
        """\\section{} in the middle of content should stay."""
        content = "First paragraph.\n\n" + r"\section{Next}" + "\nNext section."
        result = self._strip(content)
        assert r"\section{Next}" in result

    def test_empty_content(self):
        """Edge case: empty string."""
        assert self._strip("") == ""

    def test_only_section_command(self):
        """Content is just a bare section command."""
        content = r"\section{Title}"
        result = self._strip(content)
        assert result == ""


# ---------------------------------------------------------------------------
# 1b. _validate_compiled_tex_structure — conference-specific title commands
# ---------------------------------------------------------------------------

class TestValidateCompiledTexStructure:
    """Validation should recognise conference-specific title commands."""

    @staticmethod
    def _validate(tex: str):
        from src.agents.typesetter_agent.typesetter_agent import TypesetterAgent
        return TypesetterAgent._validate_compiled_tex_structure(tex)

    def test_standard_title_passes(self):
        tex = r"\title{My Paper}\begin{abstract}Good abstract.\end{abstract}"
        errors = self._validate(tex)
        assert "missing_or_empty_title" not in errors

    def test_icmltitle_passes(self):
        tex = r"\icmltitle{My ICML Paper}\begin{abstract}Abstract.\end{abstract}"
        errors = self._validate(tex)
        assert "missing_or_empty_title" not in errors, (
            "Validation does not recognise \\icmltitle — "
            "ICML template compilation will always fail."
        )

    def test_empty_title_fails(self):
        tex = r"\title{}\begin{abstract}Abstract.\end{abstract}"
        errors = self._validate(tex)
        assert "missing_or_empty_title" in errors

    def test_no_title_fails(self):
        tex = r"\begin{abstract}Abstract.\end{abstract}"
        errors = self._validate(tex)
        assert "missing_or_empty_title" in errors


# ---------------------------------------------------------------------------
# 2. compile_latex — diagnostic files on failure
# ---------------------------------------------------------------------------

class TestCompileLatexDiagnosticsOnFailure:
    """When pdflatex fails, iteration directories should still contain
    diagnostic files (main.tex, main.log, sections/) for debugging."""

    def _make_work_dir(self) -> str:
        """Create a fake work_dir with main.tex, main.log, and sections/."""
        work_dir = tempfile.mkdtemp(prefix="ts_test_")
        # main.tex
        (Path(work_dir) / "main.tex").write_text(
            r"\documentclass{article}\begin{document}Hello\end{document}",
            encoding="utf-8",
        )
        # main.log
        (Path(work_dir) / "main.log").write_text(
            "! LaTeX Error: Something went wrong.\n", encoding="utf-8",
        )
        # sections/
        sec_dir = Path(work_dir) / "sections"
        sec_dir.mkdir()
        (sec_dir / "introduction.tex").write_text(
            r"\section{Introduction}" + "\nIntro text.", encoding="utf-8",
        )
        (sec_dir / "method.tex").write_text(
            r"\section{Methodology}" + "\nMethod text.", encoding="utf-8",
        )
        return work_dir

    def test_copy_to_output_dir_copies_sections(self):
        """Verify _copy_to_output_dir copies the sections/ subdirectory."""
        from src.agents.typesetter_agent.typesetter_agent import TypesetterAgent

        work_dir = self._make_work_dir()
        output_dir = tempfile.mkdtemp(prefix="ts_out_")
        try:
            # Put a fake main.pdf so _copy_to_output_dir runs
            (Path(work_dir) / "main.pdf").write_bytes(b"%PDF-1.4 fake")
            ts = TypesetterAgent.__new__(TypesetterAgent)
            paths = ts._copy_to_output_dir(work_dir, output_dir)

            assert (Path(output_dir) / "main.tex").exists()
            assert (Path(output_dir) / "main.pdf").exists()
            sec_out = Path(output_dir) / "sections"
            assert sec_out.exists(), "sections/ directory not copied"
            assert (sec_out / "introduction.tex").exists()
            assert (sec_out / "method.tex").exists()
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_save_diagnostics_on_failure(self):
        """Even when compilation fails (no main.pdf), main.tex and main.log
        should be saved to output_dir for post-mortem debugging."""
        from src.agents.typesetter_agent.typesetter_agent import TypesetterAgent

        work_dir = self._make_work_dir()
        output_dir = tempfile.mkdtemp(prefix="ts_out_")
        try:
            ts = TypesetterAgent.__new__(TypesetterAgent)
            ts._save_diagnostics_on_failure(work_dir, output_dir)

            assert (Path(output_dir) / "main.tex").exists(), (
                "main.tex not saved on failure — iteration dir stays empty"
            )
            assert (Path(output_dir) / "main.log").exists(), (
                "main.log not saved on failure — no way to debug"
            )
            sec_out = Path(output_dir) / "sections"
            assert sec_out.exists(), (
                "sections/ not saved on failure"
            )
            assert (sec_out / "introduction.tex").exists()
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_compile_latex_calls_diagnostics_on_failure(self):
        """compile_latex should call _save_diagnostics_on_failure when
        compilation fails and output_dir is set."""
        from src.agents.typesetter_agent.typesetter_agent import TypesetterAgent
        import inspect

        src = inspect.getsource(TypesetterAgent.compile_latex)
        assert "_save_diagnostics_on_failure" in src, (
            "compile_latex does not call _save_diagnostics_on_failure — "
            "iteration directories will be empty on compile failure"
        )

    def test_compile_pdf_saves_error_log_on_exception(self):
        """_compile_pdf should write compile_errors.json to iteration
        directory when TypesetterAgent fails entirely."""
        from src.agents.metadata_agent.metadata_agent import MetaDataAgent
        import inspect

        src = inspect.getsource(MetaDataAgent._compile_pdf)
        assert "_save_compile_error_log" in src, (
            "_compile_pdf does not call _save_compile_error_log — "
            "iteration directories will be empty when TypesetterAgent crashes"
        )

    def test_save_compile_error_log_writes_file(self):
        """_save_compile_error_log should create compile_errors.json."""
        from src.agents.metadata_agent.metadata_agent import MetaDataAgent

        output_dir = Path(tempfile.mkdtemp(prefix="err_test_"))
        try:
            MetaDataAgent._save_compile_error_log(
                output_dir, ["inject_template structure validation failed"]
            )
            err_file = output_dir / "compile_errors.json"
            assert err_file.exists(), "compile_errors.json not created"
            import json
            data = json.loads(err_file.read_text())
            assert "errors" in data
            assert len(data["errors"]) == 1
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3. _assemble_paper — duplicate \section{} prevention
# ---------------------------------------------------------------------------

class TestAssemblePaperSectionDedup:
    """_assemble_paper should strip any existing \\section{} commands
    from LLM-generated content before prepending its own."""

    @staticmethod
    def _assemble(sections: Dict[str, str]) -> str:
        from src.agents.metadata_agent.metadata_agent import MetaDataAgent

        agent = MetaDataAgent.__new__(MetaDataAgent)
        # Provide minimal requirements
        return agent._assemble_paper(
            title="Test Paper",
            sections=sections,
            references=[],
            valid_citation_keys=set(),
        )

    def test_no_duplicate_section_introduction(self):
        """If LLM content starts with \\section{Introduction},
        assembled output must NOT have \\section{Introduction}\\section{Introduction}."""
        sections = {
            "abstract": "This is an abstract.",
            "introduction": r"\section{Introduction}" + "\nIntro body.",
            "conclusion": "Conclusion body.",
        }
        latex = self._assemble(sections)
        count = len(re.findall(r'\\section\{Introduction\}', latex))
        assert count == 1, (
            f"Expected exactly 1 \\section{{Introduction}}, found {count}. "
            "LLM-generated \\section{{}} not stripped before assembly."
        )

    def test_no_duplicate_section_nested_title(self):
        """If LLM content has \\section{Some \\emph{Title}},
        it should be stripped and replaced with the canonical one."""
        sections = {
            "abstract": "Abstract.",
            "method": r"\section{Our \emph{Novel} Method}" + "\nMethod body.",
        }
        latex = self._assemble(sections)
        method_count = latex.count(r"\section{")
        assert method_count == 1, (
            f"Expected 1 \\section{{}} for method, found {method_count}"
        )

    def test_section_content_preserved(self):
        """After stripping, the actual body text should be preserved."""
        sections = {
            "abstract": "Abstract text.",
            "introduction": r"\section{Introduction}" + "\nThis is the intro body.",
        }
        latex = self._assemble(sections)
        assert "This is the intro body." in latex

    def test_section_with_label_stripped(self):
        """\\section{Title}\\label{sec:intro} should be stripped cleanly."""
        sections = {
            "abstract": "Abstract.",
            "introduction": (
                r"\section{Introduction}\label{sec:intro}" + "\nIntro body."
            ),
        }
        latex = self._assemble(sections)
        count = len(re.findall(r'\\section\{Introduction\}', latex))
        assert count == 1

    def test_non_leading_section_also_stripped(self):
        """\\section{} after a figure environment should also be stripped
        (not just leading ones) in _assemble_paper."""
        sections = {
            "abstract": "Abstract.",
            "related_work": (
                r"\begin{figure}[t]\centering\end{figure}" + "\n"
                r"\section{Related Work}" + "\nRelated body."
            ),
        }
        latex = self._assemble(sections)
        count = len(re.findall(r'\\section\{Related Work\}', latex))
        assert count == 1, (
            f"Expected 1 \\section{{Related Work}}, found {count}. "
            "Non-leading \\section{{}} in LLM content not stripped."
        )

    def test_abstract_boilerplate_stripped(self):
        """\\begin{abstract}/\\end{abstract} inside abstract content
        should be stripped to prevent nesting."""
        sections = {
            "abstract": (
                r"\begin{abstract}" + "\n"
                "This is the real abstract text.\n"
                r"\end{abstract}"
            ),
        }
        latex = self._assemble(sections)
        count = latex.count(r"\begin{abstract}")
        assert count == 1, (
            f"Expected 1 \\begin{{abstract}}, found {count}. "
            "Nested abstract environment not cleaned."
        )


# ---------------------------------------------------------------------------
# 4. _write_section_files — integration test for dedup
# ---------------------------------------------------------------------------

class TestWriteSectionFilesDedup:
    """_write_section_files should produce section files with exactly
    one \\section{} command, even if LLM content already has one."""

    def test_no_double_section_in_file(self):
        from src.agents.typesetter_agent.typesetter_agent import TypesetterAgent

        work_dir = tempfile.mkdtemp(prefix="ws_test_")
        try:
            ts = TypesetterAgent.__new__(TypesetterAgent)
            sections = {
                "introduction": r"\section{Introduction}" + "\nIntro body text.",
                "method": "Method body without section command.",
            }
            section_file_map = ts._write_section_files(
                work_dir=work_dir,
                sections=sections,
                section_order=["introduction", "method"],
                section_titles={"introduction": "Introduction", "method": "Methodology"},
            )

            intro_path = Path(work_dir) / "sections" / "introduction.tex"
            assert intro_path.exists()
            intro_content = intro_path.read_text(encoding="utf-8")

            section_count = len(re.findall(r'\\section\*?\{', intro_content))
            assert section_count == 1, (
                f"Expected 1 \\section{{}} in introduction.tex, found {section_count}. "
                f"Content:\n{intro_content}"
            )

            method_path = Path(work_dir) / "sections" / "method.tex"
            method_content = method_path.read_text(encoding="utf-8")
            assert method_content.startswith(r"\section{Methodology}")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def test_nested_brace_title_handled(self):
        """Section content with nested-brace title should still dedup."""
        from src.agents.typesetter_agent.typesetter_agent import TypesetterAgent

        work_dir = tempfile.mkdtemp(prefix="ws_test_")
        try:
            ts = TypesetterAgent.__new__(TypesetterAgent)
            sections = {
                "method": r"\section{Our \textbf{Proposed} Framework}" + "\nActual method.",
            }
            ts._write_section_files(
                work_dir=work_dir,
                sections=sections,
                section_order=["method"],
                section_titles={"method": "Methodology"},
            )
            method_path = Path(work_dir) / "sections" / "method.tex"
            content = method_path.read_text(encoding="utf-8")
            section_count = len(re.findall(r'\\section\*?\{', content))
            assert section_count == 1, (
                f"Expected 1 \\section in method.tex with nested-brace title, "
                f"found {section_count}. Content:\n{content}"
            )
            assert "Actual method." in content
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
