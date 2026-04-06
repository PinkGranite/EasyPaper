"""
Tests for TemplateStructureProfile model and _analyze_template_structure().
Covers Phases 1-3 of the Template Adaptation TDD plan.
"""
import pytest
from src.agents.typesetter_agent.models import TemplateStructureProfile
from src.agents.typesetter_agent.typesetter_agent import TypesetterAgent


# ---------------------------------------------------------------------------
# Template fixtures
# ---------------------------------------------------------------------------

STANDARD_ARTICLE_TEMPLATE = r"""
\documentclass[11pt]{article}
\usepackage{amsmath}
\usepackage{graphicx}

\title{My Paper Title}
\author{John Doe}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
This is the abstract.
\end{abstract}

\section{Introduction}
Content here.

\end{document}
"""

ICML_TEMPLATE = r"""
\documentclass{article}
\usepackage{icml2026}
\usepackage{hyperref}
\usepackage{booktabs}

\icmltitlerunning{Short Title}

\begin{document}

\twocolumn[
  \icmltitle{Full Paper Title Here}

  \icmlsetsymbol{equal}{*}

  \begin{icmlauthorlist}
    \icmlauthor{Firstname1 Lastname1}{equal,yyy}
    \icmlauthor{Firstname2 Lastname2}{equal,comp}
  \end{icmlauthorlist}

  \icmlaffiliation{yyy}{Department of XXX, University of YYY}
  \icmlaffiliation{comp}{Company Name}

  \icmlcorrespondingauthor{Firstname1 Lastname1}{first1@xxx.edu}

  \icmlkeywords{Machine Learning, ICML}

  \vskip 0.3in
]

\printAffiliationsAndNotice{}

\begin{abstract}
Template abstract placeholder.
\end{abstract}

\section{Introduction}
Body here.

\bibliographystyle{icml2026}
\bibliography{references}

\end{document}
"""

ICML_ABSTRACT_INSIDE_TWOCOLUMN = r"""
\documentclass{article}
\usepackage{icml2026}

\begin{document}

\twocolumn[
  \icmltitle{Title}

  \begin{icmlauthorlist}
    \icmlauthor{Author}{aff}
  \end{icmlauthorlist}

  \icmlaffiliation{aff}{University}

  \vskip 0.3in

  \begin{abstract}
  Abstract inside twocolumn brackets.
  \end{abstract}
]

\section{Introduction}
Body.

\end{document}
"""

NEURIPS_TEMPLATE = r"""
\documentclass{article}
\usepackage[preprint]{neurips_2025}
\usepackage{booktabs}

\title{NeurIPS Paper}

\author{
  Author One \\
  Department \\
  University \\
  \texttt{author@uni.edu}
}

\begin{document}
\maketitle

\begin{abstract}
Abstract here.
\end{abstract}

\section{Introduction}
Content.

\end{document}
"""

IEEE_TEMPLATE = r"""
\documentclass[conference]{IEEEtran}
\usepackage{graphicx}

\title{IEEE Paper Title}
\author{
  \IEEEauthorblockN{Author One}
  \IEEEauthorblockA{University}
}

\begin{document}
\maketitle

\begin{abstract}
Abstract.
\end{abstract}

\section{Introduction}
Content.

\end{document}
"""

NATURE_TEMPLATE = r"""
\documentclass{sn-jnl}
\usepackage{graphicx}

\title[Short]{Full Title of the Paper}

\author[1]{\fnm{John} \sur{Doe}}
\author[2]{\fnm{Jane} \sur{Smith}}

\affil[1]{Department of Physics, University of Example}
\affil[2]{Institute of Technology}

\abstract{
This is the abstract content for the Nature/Springer paper.
}

\begin{document}
\maketitle

\section{Introduction}
Content.

\end{document}
"""


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: TemplateStructureProfile model + analyzer
# ═══════════════════════════════════════════════════════════════════════════


class TestTemplateStructureProfileModel:
    """Test that TemplateStructureProfile model has all required fields."""

    def test_default_values(self):
        profile = TemplateStructureProfile()
        assert profile.title_command == "\\title"
        assert profile.author_system == "standard"
        assert profile.abstract_format == "environment"
        assert profile.abstract_inside_twocolumn is False
        assert profile.needs_maketitle is True
        assert profile.needs_date is False
        assert profile.has_twocolumn_bracket is False

    def test_icml_values(self):
        profile = TemplateStructureProfile(
            title_command="\\icmltitle",
            author_system="icml",
            abstract_inside_twocolumn=False,
            needs_maketitle=False,
            has_twocolumn_bracket=True,
        )
        assert profile.title_command == "\\icmltitle"
        assert profile.author_system == "icml"
        assert profile.needs_maketitle is False


class TestAnalyzeTemplateStructure:
    """Test _analyze_template_structure static method on various templates."""

    def test_standard_article_template(self):
        profile = TypesetterAgent._analyze_template_structure(STANDARD_ARTICLE_TEMPLATE)
        assert profile.title_command == "\\title"
        assert profile.author_system == "standard"
        assert profile.abstract_format == "environment"
        assert profile.needs_maketitle is True
        assert profile.has_twocolumn_bracket is False
        assert profile.abstract_inside_twocolumn is False

    def test_icml_template(self):
        profile = TypesetterAgent._analyze_template_structure(ICML_TEMPLATE)
        assert profile.title_command == "\\icmltitle"
        assert profile.author_system == "icml"
        assert profile.has_twocolumn_bracket is True
        assert profile.needs_maketitle is False
        assert profile.abstract_format == "environment"
        # Abstract is outside \twocolumn[...] in this variant
        assert profile.abstract_inside_twocolumn is False

    def test_icml_abstract_inside_twocolumn(self):
        profile = TypesetterAgent._analyze_template_structure(ICML_ABSTRACT_INSIDE_TWOCOLUMN)
        assert profile.title_command == "\\icmltitle"
        assert profile.author_system == "icml"
        assert profile.has_twocolumn_bracket is True
        assert profile.abstract_inside_twocolumn is True
        assert profile.needs_maketitle is False

    def test_neurips_template(self):
        profile = TypesetterAgent._analyze_template_structure(NEURIPS_TEMPLATE)
        assert profile.title_command == "\\title"
        assert profile.author_system == "standard"
        assert profile.needs_maketitle is True
        assert profile.abstract_format == "environment"

    def test_ieee_template(self):
        profile = TypesetterAgent._analyze_template_structure(IEEE_TEMPLATE)
        assert profile.title_command == "\\title"
        assert profile.author_system == "ieee"
        assert profile.needs_maketitle is True

    def test_nature_template(self):
        profile = TypesetterAgent._analyze_template_structure(NATURE_TEMPLATE)
        assert profile.title_command == "\\title"
        assert profile.author_system == "nature"
        assert profile.abstract_format == "command"
        assert profile.needs_maketitle is True


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Author/date injection tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAuthorInjection:
    """Test that _smart_inject_content respects profile.author_system."""

    def _make_agent(self):
        agent = TypesetterAgent.__new__(TypesetterAgent)
        return agent

    def test_icml_author_injection_preserves_template(self):
        """ICML templates must NOT get \author{EasyPaper} injected."""
        from src.agents.typesetter_agent.models import TemplateConfig
        agent = self._make_agent()
        profile = TypesetterAgent._analyze_template_structure(ICML_TEMPLATE)
        config = TemplateConfig(paper_title="Test Title", paper_authors="EasyPaper")
        sections = {"abstract": "Test abstract.", "body": "\\input{sections/intro}"}

        result = agent._smart_inject_content(
            ICML_TEMPLATE, sections, config, [], profile=profile,
        )

        assert "\\author{EasyPaper}" not in result
        assert "\\begin{icmlauthorlist}" in result
        assert "\\icmlauthor{" in result

    def test_icml_no_maketitle_inserted(self):
        """ICML templates must NOT get \maketitle injected."""
        from src.agents.typesetter_agent.models import TemplateConfig
        agent = self._make_agent()
        profile = TemplateStructureProfile(needs_maketitle=False)

        text_without_maketitle = r"""
\begin{document}
\begin{abstract}
Test abstract
\end{abstract}
\section{Intro}
\end{document}
"""
        result = TypesetterAgent._ensure_maketitle_present(
            text_without_maketitle, profile=profile,
        )
        assert "\\maketitle" not in result

    def test_standard_maketitle_inserted(self):
        """Standard templates should get \maketitle if missing."""
        profile = TemplateStructureProfile(needs_maketitle=True)
        text = r"""
\begin{document}
\begin{abstract}
Test
\end{abstract}
\end{document}
"""
        result = TypesetterAgent._ensure_maketitle_present(text, profile=profile)
        assert "\\maketitle" in result

    def test_standard_author_replaced(self):
        """Standard templates should have \author{} replaced."""
        from src.agents.typesetter_agent.models import TemplateConfig
        agent = self._make_agent()
        profile = TemplateStructureProfile(author_system="standard")
        config = TemplateConfig(paper_title="Title", paper_authors="New Author")
        sections = {"abstract": "Abstract text.", "body": "Body text."}

        result = agent._smart_inject_content(
            STANDARD_ARTICLE_TEMPLATE, sections, config, [], profile=profile,
        )
        assert "\\author{New Author}" in result
        assert "\\author{John Doe}" not in result


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Abstract formatting tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAbstractFormatting:
    """Test abstract normalization and positioning."""

    def test_normalize_abstract_single_paragraph(self):
        """Multi-paragraph abstract should be collapsed into one."""
        raw = (
            "First part of the abstract about method.\n\n"
            "Second part about results and contributions."
        )
        result = TypesetterAgent._normalize_abstract(raw)
        assert "\n\n" not in result
        assert "method." in result
        assert "contributions." in result

    def test_normalize_abstract_preserves_single(self):
        """Single-paragraph abstract should be unchanged (modulo whitespace)."""
        raw = "This is a single paragraph abstract with no blank lines."
        result = TypesetterAgent._normalize_abstract(raw)
        assert result.strip() == raw.strip()

    def test_icml_abstract_position_after_injection(self):
        """For ICML template (abstract outside twocolumn), abstract must appear
        between \\printAffiliationsAndNotice and first \\input/\\section,
        NOT causing extra page breaks."""
        from src.agents.typesetter_agent.models import TemplateConfig
        agent = self._make_agent()
        profile = TypesetterAgent._analyze_template_structure(ICML_TEMPLATE)
        config = TemplateConfig(paper_title="Test", paper_authors="Author")
        sections = {
            "abstract": "Test abstract single paragraph.",
            "body": "\\input{sections/introduction}",
        }

        result = agent._smart_inject_content(
            ICML_TEMPLATE, sections, config, [], profile=profile,
        )

        # Abstract should exist
        assert "\\begin{abstract}" in result
        assert "Test abstract single paragraph." in result
        # No \maketitle after abstract (ICML doesn't use it)
        abstract_end = result.index("\\end{abstract}")
        body_start = result.index("\\input{sections/introduction}")
        between = result[abstract_end:body_start]
        assert "\\maketitle" not in between

    def _make_agent(self):
        return TypesetterAgent.__new__(TypesetterAgent)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: Table placement enforcement
# ═══════════════════════════════════════════════════════════════════════════


class TestTablePlacementEnforcement:
    """Test that tables are stripped from non-assigned sections."""

    def test_table_not_in_introduction(self):
        """Table assigned to 'experiment' must be removed from 'introduction'."""
        sections = {
            "introduction": (
                "Introduction text.\n\n"
                "\\begin{table}[htbp]\n"
                "\\centering\n"
                "\\caption{Results.}\\label{tab:results}\n"
                "\\begin{tabular}{cc}\n"
                "A & B \\\\\n"
                "\\end{tabular}\n"
                "\\end{table}\n\n"
                "More text."
            ),
            "experiment": "Experiment text.",
        }
        table_assignments = {"tab:results": "experiment"}
        from src.agents.metadata_agent.metadata_agent import MetaDataAgent
        result = MetaDataAgent._enforce_table_placement(sections, table_assignments)
        assert "\\begin{table}" not in result["introduction"]
        assert "More text." in result["introduction"]

    def test_table_in_assigned_section_preserved(self):
        """Table in its assigned section must be kept."""
        sections = {
            "experiment": (
                "Experiment text.\n\n"
                "\\begin{table}[htbp]\n"
                "\\caption{Results.}\\label{tab:results}\n"
                "\\begin{tabular}{cc}\n"
                "A & B \\\\\n"
                "\\end{tabular}\n"
                "\\end{table}\n\n"
                "More text."
            ),
        }
        table_assignments = {"tab:results": "experiment"}
        from src.agents.metadata_agent.metadata_agent import MetaDataAgent
        result = MetaDataAgent._enforce_table_placement(sections, table_assignments)
        assert "\\begin{table}" in result["experiment"]

    def test_unassigned_table_left_alone(self):
        """Tables with no assignment should be left in place."""
        sections = {
            "result": (
                "Results.\n\n"
                "\\begin{table}[htbp]\n"
                "\\caption{Misc.}\n"
                "\\begin{tabular}{c}\n"
                "X\n"
                "\\end{tabular}\n"
                "\\end{table}"
            ),
        }
        table_assignments = {}
        from src.agents.metadata_agent.metadata_agent import MetaDataAgent
        result = MetaDataAgent._enforce_table_placement(sections, table_assignments)
        assert "\\begin{table}" in result["result"]

    def test_wide_table_uses_table_star(self):
        """In double-column layout, wide tables should use table*."""
        content = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Wide results.}\\label{tab:wide}\n"
            "\\begin{tabular}{cccccc}\n"
            "A & B & C & D & E & F \\\\\n"
            "\\end{tabular}\n"
            "\\end{table}"
        )
        from src.agents.typesetter_agent.typesetter_agent import TypesetterAgent
        result = TypesetterAgent._promote_wide_tables(content)
        assert "\\begin{table*}" in result
        assert "\\end{table*}" in result


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Strip ALL section commands (not just leading)
# ═══════════════════════════════════════════════════════════════════════════


class TestStripAllSectionCommands:
    """Test that _strip_all_section_commands removes ALL \\section{} from content."""

    def test_strip_section_after_figure(self):
        content = (
            "\\begin{figure}[htbp]\n"
            "\\centering\n"
            "\\includegraphics[width=0.9\\linewidth]{figures/fig_1}\n"
            "\\caption{Overview.}\\label{fig:overview}\n"
            "\\end{figure}\n\n"
            "\\section{Methodology}\n\n"
            "The proposed approach uses..."
        )
        result = TypesetterAgent._strip_all_section_commands(content)
        assert "\\section{Methodology}" not in result
        assert "\\begin{figure}" in result
        assert "The proposed approach uses..." in result

    def test_strip_multiple_scattered_sections(self):
        content = (
            "\\section{Method}\n"
            "First paragraph.\n\n"
            "\\begin{figure}[htbp]\n"
            "\\end{figure}\n\n"
            "\\section{Results}\n"
            "Second paragraph.\n\n"
            "\\section*{Conclusion}\n"
            "Third paragraph."
        )
        result = TypesetterAgent._strip_all_section_commands(content)
        assert "\\section{Method}" not in result
        assert "\\section{Results}" not in result
        assert "\\section*{Conclusion}" not in result
        assert "First paragraph." in result
        assert "Second paragraph." in result
        assert "Third paragraph." in result

    def test_subsection_preserved(self):
        content = (
            "\\section{Overview}\n"
            "Introduction text.\n\n"
            "\\subsection{Details}\n"
            "Detail text.\n\n"
            "\\subsubsection{Sub-details}\n"
            "Sub text."
        )
        result = TypesetterAgent._strip_all_section_commands(content)
        assert "\\section{Overview}" not in result
        assert "\\subsection{Details}" in result
        assert "\\subsubsection{Sub-details}" in result

    def test_strip_section_with_label(self):
        content = (
            "\\section{Experiments}\\label{sec:exp}\n"
            "We conduct experiments..."
        )
        result = TypesetterAgent._strip_all_section_commands(content)
        assert "\\section{Experiments}" not in result
        assert "\\label{sec:exp}" not in result
        assert "We conduct experiments..." in result

    def test_strip_section_with_nested_braces(self):
        content = (
            "\\section{Some \\textbf{Bold} Title}\n"
            "Content here."
        )
        result = TypesetterAgent._strip_all_section_commands(content)
        assert "\\section{" not in result
        assert "Content here." in result
