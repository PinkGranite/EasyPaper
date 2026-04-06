"""
Template Analyzer — extract LaTeX template constraints for Writer guidance.
- **Description**:
    - Parses LaTeX template preambles to extract packages, environments,
      and commands.
    - Generates semantic writing guidance (figure, table, algorithm conventions).
    - Produces a TemplateWriterGuide that is injected into Writer prompts.
"""
from __future__ import annotations

import logging
import os
import re
import zipfile
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════════════


class TemplateWriterGuide(BaseModel):
    """
    Structured guidance extracted from a LaTeX template for Writer prompts.
    - **Description**:
        - Layer 1 (precise): packages, document class, custom environments/commands.
        - Layer 2 (semantic): natural-language guidance for figures, tables, etc.
    """

    available_packages: List[str] = Field(default_factory=list)
    document_class: str = "article"
    column_format: str = "single"  # single / double
    citation_style: str = "cite"  # cite / citep / citet / autocite
    custom_environments: List[str] = Field(default_factory=list)
    custom_commands: List[str] = Field(default_factory=list)

    figure_guidance: str = ""
    table_guidance: str = ""
    algorithm_guidance: str = ""
    math_guidance: str = ""
    general_constraints: str = ""

    def has_package(self, package_name: str) -> bool:
        """Check whether a specific package is available in the template."""
        return package_name in self.available_packages

    def format_for_prompt(self) -> str:
        """
        Format as a prompt block for injection into Writer prompts.
        - **Returns**:
            - `str`: Markdown-formatted constraint block, or "" if guide is empty.
        """
        if not self.available_packages and not any([
            self.figure_guidance, self.table_guidance,
            self.algorithm_guidance, self.math_guidance,
            self.general_constraints,
        ]):
            return ""

        parts: list[str] = ["## Template Constraints"]

        if self.available_packages:
            parts.append(
                f"**Document class**: `{self.document_class}`  "
                f"**Column format**: {self.column_format}"
            )
            parts.append(
                f"**Available packages**: {', '.join(self.available_packages)}"
            )

        if self.figure_guidance:
            parts.append(f"**Figure writing**: {self.figure_guidance}")
        if self.table_guidance:
            parts.append(f"**Table writing**: {self.table_guidance}")
        if self.algorithm_guidance:
            parts.append(f"**Algorithm writing**: {self.algorithm_guidance}")
        if self.math_guidance:
            parts.append(f"**Math environments**: {self.math_guidance}")
        if self.custom_environments:
            parts.append(
                f"**Custom environments available**: "
                f"{', '.join(self.custom_environments)}"
            )
        if self.general_constraints:
            parts.append(f"\n{self.general_constraints}")

        parts.append(
            "\n**IMPORTANT**: Do NOT use LaTeX commands from packages "
            "not listed above. If you need a command from an unavailable "
            "package, use a standard alternative."
        )

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Preamble parser — precise extraction layer
# ═══════════════════════════════════════════════════════════════════════════


class PreambleParser:
    """Static utilities for extracting structured info from LaTeX preambles."""

    @staticmethod
    def extract_preamble(full_tex: str) -> str:
        """
        Extract the preamble (everything before \\begin{document}).
        - **Args**:
            - `full_tex` (str): Full LaTeX source.
        - **Returns**:
            - `str`: Preamble text, or full text if no \\begin{document}.
        """
        match = re.search(r"\\begin\{document\}", full_tex)
        if match:
            return full_tex[: match.start()]
        return full_tex

    @staticmethod
    def extract_packages(preamble: str) -> List[str]:
        """
        Extract all package names from \\usepackage commands.
        - **Description**:
            - Handles \\usepackage[options]{pkg} and \\usepackage{p1,p2,p3}.
        - **Returns**:
            - `List[str]`: Deduplicated list of package names.
        """
        packages: list[str] = []
        for match in re.finditer(
            r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", preamble
        ):
            raw = match.group(1)
            for pkg in raw.split(","):
                pkg = pkg.strip()
                if pkg:
                    packages.append(pkg)
        return list(dict.fromkeys(packages))

    @staticmethod
    def extract_document_class(preamble: str) -> Tuple[str, List[str]]:
        """
        Extract document class name and options.
        - **Returns**:
            - Tuple of (class_name, [options])
        """
        match = re.search(
            r"\\documentclass(?:\[([^\]]*)\])?\{([^}]+)\}", preamble
        )
        if not match:
            return "article", []
        options_str = match.group(1) or ""
        doc_class = match.group(2).strip()
        options = [o.strip() for o in options_str.split(",") if o.strip()]
        return doc_class, options

    @staticmethod
    def detect_column_format(preamble: str) -> str:
        """
        Detect column format from document class options or twocolumn command.
        - **Returns**:
            - `str`: "single" or "double"
        """
        _, options = PreambleParser.extract_document_class(preamble)
        if "twocolumn" in options:
            return "double"
        if re.search(r"\\twocolumn", preamble):
            return "double"
        return "single"

    @staticmethod
    def extract_custom_environments(preamble: str) -> List[str]:
        """Extract user-defined environments (\\newtheorem, \\newenvironment)."""
        envs: list[str] = []
        for match in re.finditer(r"\\newtheorem\{(\w+)\}", preamble):
            envs.append(match.group(1))
        for match in re.finditer(r"\\newenvironment\{(\w+)\}", preamble):
            envs.append(match.group(1))
        return list(dict.fromkeys(envs))

    @staticmethod
    def extract_custom_commands(preamble: str) -> List[str]:
        """Extract user-defined commands (\\newcommand, \\renewcommand)."""
        cmds: list[str] = []
        for match in re.finditer(
            r"\\(?:re)?newcommand\{?(\\[a-zA-Z]+)\}?", preamble
        ):
            cmds.append(match.group(1))
        return list(dict.fromkeys(cmds))

    @staticmethod
    def detect_citation_style(preamble: str) -> str:
        """
        Detect citation style from loaded packages.
        - **Returns**:
            - `str`: "citep" (natbib), "autocite" (biblatex), or "cite"
        """
        packages = PreambleParser.extract_packages(preamble)
        if "natbib" in packages:
            return "citep"
        if "biblatex" in packages:
            return "autocite"
        return "cite"


# ═══════════════════════════════════════════════════════════════════════════
# Semantic guidance — rule-based mapping
# ═══════════════════════════════════════════════════════════════════════════

_TABLE_GUIDANCE_WITH_BOOKTABS = (
    "Use \\toprule, \\midrule, and \\bottomrule for horizontal rules "
    "(booktabs is available). Do NOT use \\hline."
)
_TABLE_GUIDANCE_WITHOUT_BOOKTABS = (
    "Use \\hline for horizontal rules. Do NOT use \\toprule, \\midrule, "
    "or \\bottomrule (booktabs is not available)."
)

_FIGURE_GUIDANCE_DOUBLE = (
    "Single-column figures: use \\begin{figure}[htbp] with "
    "width=\\columnwidth. "
    "Full-width figures: use \\begin{figure*}[htbp] with "
    "width=\\textwidth."
)
_FIGURE_GUIDANCE_SINGLE = (
    "Use \\begin{figure}[htbp] with width=\\linewidth. "
    "Do NOT use figure* (single-column layout)."
)

_ALGORITHM_GUIDANCE: Dict[str, str] = {
    "algorithm2e": (
        "Use algorithm2e syntax: \\begin{algorithm}, \\SetAlgoLined, "
        "\\KwIn{}, \\KwOut{}, \\KwResult{}, \\If{}, \\While{}, \\For{}."
    ),
    "algorithmic": (
        "Use algorithmic syntax: \\begin{algorithmic}, \\STATE, "
        "\\IF, \\WHILE, \\FOR, \\RETURN."
    ),
    "algorithmicx": (
        "Use algorithmicx syntax: \\begin{algorithmic}[1], \\State, "
        "\\If, \\While, \\For, \\Return."
    ),
    "algorithm": (
        "The algorithm package is loaded. Use \\begin{algorithm}[htbp] "
        "as a float wrapper with algorithmic/algorithmicx inside."
    ),
}


class TemplateAnalyzer:
    """Analyze LaTeX template preambles and produce TemplateWriterGuide."""

    @staticmethod
    def analyze_preamble(preamble_or_full_tex: str) -> TemplateWriterGuide:
        """
        Analyze a LaTeX preamble (or full document) and produce a guide.
        - **Args**:
            - `preamble_or_full_tex` (str): Preamble text or full .tex source.
        - **Returns**:
            - `TemplateWriterGuide`: Structured guidance for Writer prompts.
        """
        preamble = PreambleParser.extract_preamble(preamble_or_full_tex)

        packages = PreambleParser.extract_packages(preamble)
        doc_class, _ = PreambleParser.extract_document_class(preamble)
        column_format = PreambleParser.detect_column_format(preamble)
        citation_style = PreambleParser.detect_citation_style(preamble)
        custom_envs = PreambleParser.extract_custom_environments(preamble)
        custom_cmds = PreambleParser.extract_custom_commands(preamble)

        figure_guidance = (
            _FIGURE_GUIDANCE_DOUBLE
            if column_format == "double"
            else _FIGURE_GUIDANCE_SINGLE
        )

        table_guidance = (
            _TABLE_GUIDANCE_WITH_BOOKTABS
            if "booktabs" in packages
            else _TABLE_GUIDANCE_WITHOUT_BOOKTABS
        )

        algorithm_guidance = ""
        for algo_pkg, guidance in _ALGORITHM_GUIDANCE.items():
            if algo_pkg in packages:
                algorithm_guidance = guidance
                break

        math_guidance = ""
        if custom_envs:
            math_guidance = (
                f"Available theorem-like environments: "
                f"{', '.join(custom_envs)}. "
                f"You may use \\begin{{{custom_envs[0]}}}...\\end{{{custom_envs[0]}}} etc."
            )

        constraint_parts: list[str] = []
        if citation_style == "citep":
            constraint_parts.append(
                "Use \\citep{} for parenthetical citations and "
                "\\citet{} for textual citations (natbib loaded)."
            )
        elif citation_style == "autocite":
            constraint_parts.append(
                "Use \\autocite{} for citations (biblatex loaded)."
            )
        if "hyperref" not in packages:
            constraint_parts.append(
                "Do NOT use \\url{} or \\href{} (hyperref not loaded)."
            )
        if "subcaption" not in packages and "subfig" not in packages:
            constraint_parts.append(
                "Do NOT use \\begin{subfigure} or \\subfloat "
                "(subcaption/subfig not loaded)."
            )

        return TemplateWriterGuide(
            available_packages=packages,
            document_class=doc_class,
            column_format=column_format,
            citation_style=citation_style,
            custom_environments=custom_envs,
            custom_commands=custom_cmds,
            figure_guidance=figure_guidance,
            table_guidance=table_guidance,
            algorithm_guidance=algorithm_guidance,
            math_guidance=math_guidance,
            general_constraints="\n".join(constraint_parts),
        )

    @staticmethod
    def analyze_zip(zip_path: str) -> TemplateWriterGuide:
        """
        Analyze a .zip template file and produce a TemplateWriterGuide.
        - **Description**:
            - Extracts the zip in-memory
            - Finds main.tex (file containing \\documentclass and \\begin{document})
            - Parses the preamble
        - **Args**:
            - `zip_path` (str): Path to .zip template file.
        - **Returns**:
            - `TemplateWriterGuide`: Guidance, or empty guide on failure.
        """
        if not zip_path or not os.path.exists(zip_path):
            logger.warning("template_analyzer.zip_not_found path=%s", zip_path)
            return TemplateWriterGuide()

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                tex_files = [
                    n
                    for n in zf.namelist()
                    if n.endswith(".tex") and not n.startswith("__MACOSX")
                ]
                main_tex_content = None
                for name in sorted(
                    tex_files,
                    key=lambda n: (
                        0 if "main" in os.path.basename(n).lower() else
                        1 if "paper" in os.path.basename(n).lower() else 2,
                        n,
                    ),
                ):
                    content = zf.read(name).decode("utf-8", errors="ignore")
                    if (
                        "\\documentclass" in content
                        and "\\begin{document}" in content
                    ):
                        main_tex_content = content
                        logger.info(
                            "template_analyzer.found_main_tex file=%s", name
                        )
                        break

                if not main_tex_content:
                    logger.warning("template_analyzer.no_main_tex_in_zip")
                    return TemplateWriterGuide()

                return TemplateAnalyzer.analyze_preamble(main_tex_content)

        except Exception as e:
            logger.error(
                "template_analyzer.zip_error path=%s error=%s", zip_path, e
            )
            return TemplateWriterGuide()


# ═══════════════════════════════════════════════════════════════════════════
# Auto-package detection & injection (defense-in-depth for Typesetter)
# ═══════════════════════════════════════════════════════════════════════════

COMMAND_TO_PACKAGE: Dict[str, str] = {
    # booktabs
    "\\toprule": "booktabs",
    "\\midrule": "booktabs",
    "\\bottomrule": "booktabs",
    "\\cmidrule": "booktabs",
    "\\addlinespace": "booktabs",
    # algorithm2e
    "\\SetAlgoLined": "algorithm2e",
    "\\KwIn": "algorithm2e",
    "\\KwOut": "algorithm2e",
    "\\KwResult": "algorithm2e",
    # algorithmicx / algorithmic
    "\\State": "algorithmicx",
    "\\Require": "algorithmicx",
    "\\Ensure": "algorithmicx",
    # subfig / subcaption
    "\\subfloat": "subfig",
    "\\subcaptionbox": "subcaption",
    # url / hyperref
    "\\url": "url",
    "\\href": "hyperref",
    # xcolor
    "\\textcolor": "xcolor",
    "\\colorbox": "xcolor",
    # multirow
    "\\multirow": "multirow",
    # natbib
    "\\citep": "natbib",
    "\\citet": "natbib",
}

# hyperref provides \url, so if hyperref is loaded, \url is available
_PACKAGE_PROVIDES: Dict[str, List[str]] = {
    "hyperref": ["url"],
}


def detect_missing_packages(
    preamble: str, body: str
) -> List[str]:
    """
    Detect LaTeX packages needed by commands in body but absent from preamble.
    - **Args**:
        - `preamble` (str): The LaTeX preamble text.
        - `body` (str): The generated LaTeX body content.
    - **Returns**:
        - `List[str]`: Deduplicated list of missing package names.
    """
    loaded = set(PreambleParser.extract_packages(preamble))
    for pkg, provides in _PACKAGE_PROVIDES.items():
        if pkg in loaded:
            loaded.update(provides)

    missing: list[str] = []
    for cmd, pkg in COMMAND_TO_PACKAGE.items():
        if cmd in body and pkg not in loaded:
            if pkg not in missing:
                missing.append(pkg)
    return missing


def inject_missing_packages(
    full_tex: str, packages: List[str]
) -> List[str]:
    """
    Inject missing \\usepackage lines before \\begin{document}.
    - **Args**:
        - `full_tex` (str): Full LaTeX source.
        - `packages` (List[str]): Package names to inject.
    - **Returns**:
        - `str`: Modified LaTeX source with packages injected.
    """
    if not packages:
        return full_tex

    already = set(PreambleParser.extract_packages(full_tex))
    to_add = [p for p in packages if p not in already]
    if not to_add:
        return full_tex

    injection = "\n".join(f"\\usepackage{{{p}}}" for p in to_add)
    match = re.search(r"\\begin\{document\}", full_tex)
    if match:
        insert_pos = match.start()
        return (
            full_tex[:insert_pos].rstrip()
            + "\n"
            + injection
            + "\n"
            + full_tex[insert_pos:]
        )
    return full_tex + "\n" + injection
