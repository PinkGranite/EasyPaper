"""End-to-end tests: zip template → TemplateAnalyzer → prompt injection."""
import zipfile

import pytest

from src.agents.shared.template_analyzer import (
    TemplateAnalyzer,
    TemplateWriterGuide,
    detect_missing_packages,
    inject_missing_packages,
)
from src.agents.shared.prompt_compiler import (
    compile_introduction_prompt,
    compile_body_section_prompt,
    compile_synthesis_prompt,
)


ICML_TEMPLATE = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{natbib}
\usepackage{booktabs}
\usepackage{algorithm2e}
\usepackage{hyperref}
\newtheorem{theorem}{Theorem}
\newcommand{\R}{\mathbb{R}}
\begin{document}
\title{ICML Paper Template}
\maketitle
\begin{abstract}
Abstract here
\end{abstract}
\section{Introduction}
Intro text here
\end{document}
"""


@pytest.fixture
def icml_zip(tmp_path):
    """Create a realistic ICML-style template zip."""
    zip_path = tmp_path / "icml_template.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("main.tex", ICML_TEMPLATE)
        zf.writestr("icml.bst", "% bst file placeholder")
    return str(zip_path)


class TestEndToEnd:
    def test_zip_to_guide_extraction(self, icml_zip):
        guide = TemplateAnalyzer.analyze_zip(icml_zip)

        assert guide.document_class == "article"
        assert guide.citation_style == "citep"
        assert "booktabs" in guide.available_packages
        assert "algorithm2e" in guide.available_packages
        assert "natbib" in guide.available_packages
        assert "theorem" in guide.custom_environments
        assert "\\R" in guide.custom_commands

    def test_guide_produces_meaningful_prompt_block(self, icml_zip):
        guide = TemplateAnalyzer.analyze_zip(icml_zip)
        block = guide.format_for_prompt()

        assert "## Template Constraints" in block
        assert "booktabs" in block
        assert "\\toprule" in block
        assert "algorithm2e" in block
        assert "citep" in block.lower() or "\\citep" in block
        assert "IMPORTANT" in block

    def test_guide_injected_into_intro_prompt(self, icml_zip):
        guide = TemplateAnalyzer.analyze_zip(icml_zip)
        prompt = compile_introduction_prompt(
            paper_title="Test Paper",
            idea_hypothesis="Hypothesis",
            method_summary="Method",
            data_summary="Data",
            experiments_summary="Experiments",
            template_guide=guide,
        )
        assert "Template Constraints" in prompt
        assert "booktabs" in prompt
        assert "algorithm2e" in prompt

    def test_guide_injected_into_body_prompt(self, icml_zip):
        guide = TemplateAnalyzer.analyze_zip(icml_zip)
        prompt = compile_body_section_prompt(
            section_type="method",
            metadata_content="Method content",
            intro_context="Intro context",
            template_guide=guide,
        )
        assert "Template Constraints" in prompt
        assert "\\toprule" in prompt

    def test_guide_injected_into_synthesis_prompt(self, icml_zip):
        guide = TemplateAnalyzer.analyze_zip(icml_zip)
        prompt = compile_synthesis_prompt(
            section_type="conclusion",
            paper_title="Test Paper",
            prior_sections={"introduction": "intro text"},
            template_guide=guide,
        )
        assert "Template Constraints" in prompt

    def test_auto_package_fixes_missing_booktabs(self, icml_zip):
        """Simulate writer using booktabs commands with template lacking booktabs."""
        preamble = r"\documentclass{article}" + "\n" + r"\usepackage{amsmath}"
        body = r"\begin{tabular}{ll}\toprule A & B \\ \bottomrule\end{tabular}"

        missing = detect_missing_packages(preamble, body)
        assert "booktabs" in missing

        full_tex = preamble + "\n" + r"\begin{document}" + "\n" + body + "\n" + r"\end{document}"
        fixed = inject_missing_packages(full_tex, missing)
        assert r"\usepackage{booktabs}" in fixed
        assert fixed.index(r"\usepackage{booktabs}") < fixed.index(r"\begin{document}")

    def test_no_false_positives_when_template_has_packages(self, icml_zip):
        """ICML template has booktabs, so no missing packages for booktabs commands."""
        guide = TemplateAnalyzer.analyze_zip(icml_zip)
        preamble_from_template = ICML_TEMPLATE.split(r"\begin{document}")[0]
        body = r"\toprule A \midrule B \bottomrule"

        missing = detect_missing_packages(preamble_from_template, body)
        assert "booktabs" not in missing

    def test_full_pipeline_guide_prevents_unavailable_commands(self, icml_zip):
        """Guide warns against subfigure when subcaption/subfig not loaded."""
        guide = TemplateAnalyzer.analyze_zip(icml_zip)
        assert not guide.has_package("subcaption")
        assert not guide.has_package("subfig")

        block = guide.format_for_prompt()
        assert "subfigure" in block.lower() or "subfloat" in block.lower()
