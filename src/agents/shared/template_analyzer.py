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
