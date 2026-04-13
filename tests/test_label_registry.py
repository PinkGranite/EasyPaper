"""
Tests for cross-section label registry: collecting labels and validating refs.
Phase 5 of the Template Adaptation TDD plan.
"""
import pytest
from src.agents.shared.label_registry import collect_all_labels, validate_and_fix_refs


class TestCollectLabels:
    """Test _collect_all_labels scans sections for \\label{} definitions."""

    def test_collect_from_single_section(self):
        sections = {
            "introduction": (
                "\\begin{figure}[htbp]\n"
                "\\caption{Overview.}\\label{fig:overview}\n"
                "\\end{figure}\n"
                "Some text with \\label{tab:results} in a table."
            ),
        }
        labels = collect_all_labels(sections)
        assert "fig:overview" in labels
        assert "tab:results" in labels

    def test_collect_from_multiple_sections(self):
        sections = {
            "introduction": "Text with \\label{fig:overview}.",
            "method": "Method with \\label{fig:architecture} and \\label{tab:params}.",
            "result": "Results with \\label{fig:ablation}.",
        }
        labels = collect_all_labels(sections)
        assert labels == {"fig:overview", "fig:architecture", "tab:params", "fig:ablation"}

    def test_empty_sections(self):
        labels = collect_all_labels({})
        assert labels == set()

    def test_no_labels(self):
        sections = {"introduction": "Just plain text."}
        labels = collect_all_labels(sections)
        assert labels == set()

    def test_section_label_also_collected(self):
        sections = {
            "method": "\\section{Method}\\label{sec:method}\nContent.",
        }
        labels = collect_all_labels(sections)
        assert "sec:method" in labels


class TestValidateAndFixRefs:
    """Test validate_and_fix_refs removes undefined references."""

    def test_valid_refs_preserved(self):
        content = "See Figure~\\ref{fig:overview} for details."
        valid_labels = {"fig:overview"}
        result = validate_and_fix_refs(content, valid_labels)
        assert "\\ref{fig:overview}" in result

    def test_undefined_figure_ref_removed(self):
        content = "Results are in Figure~\\ref{fig:FIG001} and Table~\\ref{tab:results}."
        valid_labels = {"tab:results"}
        result = validate_and_fix_refs(content, valid_labels)
        assert "\\ref{fig:FIG001}" not in result
        assert "\\ref{tab:results}" in result

    def test_multiple_undefined_refs_removed(self):
        content = (
            "See Figure~\\ref{fig:FIG001}, Figure~\\ref{fig:FIG002}, "
            "and Figure~\\ref{fig:FIG003}."
        )
        valid_labels = set()
        result = validate_and_fix_refs(content, valid_labels)
        assert "\\ref{fig:FIG001}" not in result
        assert "\\ref{fig:FIG002}" not in result
        assert "\\ref{fig:FIG003}" not in result

    def test_cref_also_handled(self):
        content = "As shown in \\cref{fig:nonexistent}."
        valid_labels = set()
        result = validate_and_fix_refs(content, valid_labels)
        assert "\\cref{fig:nonexistent}" not in result

    def test_paragraph_breaks_preserved_after_cleanup(self):
        content = (
            "First paragraph with invalid Figure~\\ref{fig:missing}.\n\n"
            "Second paragraph with valid Table~\\ref{tab:kept}."
        )
        valid_labels = {"tab:kept"}
        result = validate_and_fix_refs(content, valid_labels)
        paragraphs = [p for p in result.split("\n\n") if p.strip()]
        assert len(paragraphs) == 2
        assert "Table~\\ref{tab:kept}" in result

    def test_all_valid_no_changes(self):
        content = "Figure~\\ref{fig:a} and Table~\\ref{tab:b}."
        valid_labels = {"fig:a", "tab:b"}
        result = validate_and_fix_refs(content, valid_labels)
        assert result == content
