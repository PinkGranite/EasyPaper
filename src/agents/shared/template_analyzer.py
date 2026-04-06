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
