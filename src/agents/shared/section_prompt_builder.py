"""
SectionPromptBuilder — Declarative prompt composition for paper sections.
- **Description**:
    - Separates system prompt (cacheable, stable) from user prompt (dynamic).
    - Builder pattern: chain ``.with_memory_context()``, ``.with_style_guide()``,
      ``.with_code_context()`` then call ``.build(token_budget)``.
    - Token budget is enforced on the **user prompt**; system prompt is fixed.
    - Designed to gradually replace ``prompt_compiler.compile_*_prompt()``
      functions.
"""
from __future__ import annotations

from typing import Optional, Tuple

SYSTEM_PROMPTS = {
    "introduction": (
        "You are an expert academic writer specializing in writing introduction "
        "sections for research papers. Write clear, well-structured LaTeX content "
        "that motivates the research problem, surveys related work, and outlines contributions."
    ),
    "method": (
        "You are an expert academic writer specializing in methodology sections. "
        "Write precise, reproducible LaTeX content describing research methods, "
        "algorithms, and theoretical frameworks."
    ),
    "experiment": (
        "You are an expert academic writer specializing in experiment sections. "
        "Write detailed LaTeX content describing experimental setup, datasets, "
        "baselines, evaluation metrics, and implementation details."
    ),
    "result": (
        "You are an expert academic writer specializing in results sections. "
        "Write clear LaTeX content presenting experimental results with proper "
        "tables, figures, and statistical analysis."
    ),
    "discussion": (
        "You are an expert academic writer specializing in discussion sections. "
        "Write insightful LaTeX content analyzing results, limitations, and "
        "future directions."
    ),
    "conclusion": (
        "You are an expert academic writer specializing in conclusion sections. "
        "Write concise LaTeX content summarizing contributions and impact."
    ),
    "abstract": (
        "You are an expert academic writer specializing in abstracts. "
        "Write a concise, self-contained LaTeX abstract summarizing motivation, "
        "method, key results, and significance."
    ),
    "related_work": (
        "You are an expert academic writer specializing in related work sections. "
        "Write comprehensive LaTeX content surveying prior research, identifying "
        "gaps, and positioning the current work."
    ),
}

DEFAULT_SYSTEM = (
    "You are an expert academic writer. Write high-quality LaTeX content "
    "for the specified section of a research paper."
)


class SectionPromptBuilder:
    """
    Builder for structured section prompts.
    - **Description**:
        - Builds a ``(system_prompt, user_prompt)`` pair.
        - System prompt is deterministic per section_type (cacheable).
        - User prompt is assembled from optional context blocks and trimmed
          to fit ``token_budget``.
    """

    def __init__(self, section_type: str) -> None:
        self._section_type = section_type
        self._memory_context: Optional[str] = None
        self._code_context: Optional[str] = None
        self._style_guide: Optional[str] = None

    def with_memory_context(self, ctx: str) -> "SectionPromptBuilder":
        self._memory_context = ctx
        return self

    def with_code_context(self, ctx: str) -> "SectionPromptBuilder":
        self._code_context = ctx
        return self

    def with_style_guide(self, guide: str) -> "SectionPromptBuilder":
        self._style_guide = guide
        return self

    def build(self, token_budget: int) -> Tuple[str, str]:
        """
        Produce ``(system_prompt, user_prompt)``.
        - **Args**:
            - `token_budget` (int): Max tokens for the user prompt.

        - **Returns**:
            - Tuple of (system_prompt, user_prompt).
        """
        system = SYSTEM_PROMPTS.get(self._section_type, DEFAULT_SYSTEM)

        parts = [f"Write the **{self._section_type}** section."]

        if self._style_guide:
            parts.append(f"Style: {self._style_guide}")

        if self._memory_context:
            parts.append(f"Context:\n{self._memory_context}")

        if self._code_context:
            parts.append(f"Additional context:\n{self._code_context}")

        user = "\n\n".join(parts)

        max_chars = token_budget * 4
        if len(user) > max_chars:
            user = user[:max_chars - 3] + "..."

        return system, user
