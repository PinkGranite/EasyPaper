"""Tests for TemplateAnalyzer, PreambleParser, and TemplateWriterGuide."""
import pytest

from src.agents.shared.template_analyzer import (
    PreambleParser,
    TemplateWriterGuide,
)


# ── Task 1: TemplateWriterGuide model ────────────────────────────────────


class TestTemplateWriterGuide:
    def test_default_construction(self):
        guide = TemplateWriterGuide()
        assert guide.available_packages == []
        assert guide.document_class == "article"
        assert guide.column_format == "single"
        assert guide.citation_style == "cite"
        assert guide.custom_environments == []
        assert guide.custom_commands == []
        assert guide.figure_guidance == ""
        assert guide.table_guidance == ""
        assert guide.algorithm_guidance == ""
        assert guide.math_guidance == ""
        assert guide.general_constraints == ""

    def test_construction_with_packages(self):
        guide = TemplateWriterGuide(
            available_packages=["amsmath", "graphicx", "booktabs"],
            document_class="article",
            column_format="double",
        )
        assert "booktabs" in guide.available_packages
        assert guide.column_format == "double"

    def test_has_package(self):
        guide = TemplateWriterGuide(
            available_packages=["amsmath", "natbib", "booktabs"]
        )
        assert guide.has_package("amsmath") is True
        assert guide.has_package("algorithm2e") is False

    def test_format_for_prompt_non_empty(self):
        guide = TemplateWriterGuide(
            available_packages=["amsmath", "booktabs"],
            column_format="double",
            figure_guidance="Use figure for single-column.",
            table_guidance="Use booktabs commands.",
        )
        prompt_block = guide.format_for_prompt()
        assert "## Template Constraints" in prompt_block
        assert "amsmath" in prompt_block
        assert "booktabs" in prompt_block
        assert "double" in prompt_block
        assert "figure" in prompt_block.lower()

    def test_format_for_prompt_empty_guide_returns_empty(self):
        guide = TemplateWriterGuide()
        assert guide.format_for_prompt() == ""


# ── Task 2: PreambleParser ───────────────────────────────────────────────

SAMPLE_PREAMBLE_ICML = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{natbib}
\usepackage{booktabs}
\usepackage{algorithm2e}
\usepackage{hyperref}
\usepackage[margin=1in]{geometry}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newcommand{\R}{\mathbb{R}}
\newcommand{\email}[1]{\texttt{#1}}
"""

SAMPLE_PREAMBLE_NATURE = r"""
\documentclass[pdflatex,sn-nature]{sn-jnl}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{textcomp}
\usepackage{mathrsfs}
\usepackage{xcolor}
"""

SAMPLE_PREAMBLE_TWOCOL = r"""
\documentclass[twocolumn,10pt]{article}
\usepackage{graphicx}
\usepackage{amsmath}
"""


class TestPreambleParser:
    def test_extract_packages_simple(self):
        result = PreambleParser.extract_packages(SAMPLE_PREAMBLE_ICML)
        assert "amsmath" in result
        assert "amssymb" in result
        assert "booktabs" in result
        assert "algorithm2e" in result
        assert "natbib" in result

    def test_extract_packages_with_options(self):
        result = PreambleParser.extract_packages(SAMPLE_PREAMBLE_ICML)
        assert "inputenc" in result
        assert "geometry" in result

    def test_extract_document_class(self):
        doc_class, options = PreambleParser.extract_document_class(
            SAMPLE_PREAMBLE_ICML
        )
        assert doc_class == "article"
        assert options == []

    def test_extract_document_class_with_options(self):
        doc_class, options = PreambleParser.extract_document_class(
            SAMPLE_PREAMBLE_TWOCOL
        )
        assert doc_class == "article"
        assert "twocolumn" in options
        assert "10pt" in options

    def test_extract_document_class_custom(self):
        doc_class, options = PreambleParser.extract_document_class(
            SAMPLE_PREAMBLE_NATURE
        )
        assert doc_class == "sn-jnl"
        assert "pdflatex" in options
        assert "sn-nature" in options

    def test_detect_column_format_twocolumn(self):
        assert PreambleParser.detect_column_format(
            SAMPLE_PREAMBLE_TWOCOL
        ) == "double"

    def test_detect_column_format_singlecol(self):
        assert PreambleParser.detect_column_format(
            SAMPLE_PREAMBLE_NATURE
        ) == "single"

    def test_extract_custom_environments(self):
        envs = PreambleParser.extract_custom_environments(
            SAMPLE_PREAMBLE_ICML
        )
        assert "theorem" in envs
        assert "lemma" in envs

    def test_extract_custom_commands(self):
        cmds = PreambleParser.extract_custom_commands(SAMPLE_PREAMBLE_ICML)
        assert "\\R" in cmds
        assert "\\email" in cmds

    def test_detect_citation_style_natbib(self):
        assert PreambleParser.detect_citation_style(
            SAMPLE_PREAMBLE_ICML
        ) == "citep"

    def test_detect_citation_style_default(self):
        assert PreambleParser.detect_citation_style(
            SAMPLE_PREAMBLE_NATURE
        ) == "cite"

    def test_extract_preamble_from_full_document(self):
        full_doc = SAMPLE_PREAMBLE_ICML + r"""
\begin{document}
\maketitle
Hello world
\end{document}
"""
        preamble = PreambleParser.extract_preamble(full_doc)
        assert r"\usepackage{booktabs}" in preamble
        assert r"\begin{document}" not in preamble
