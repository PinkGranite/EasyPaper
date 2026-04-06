"""
QualityHook pipeline — Pluggable post-generation quality checks.
- **Description**:
    - Replaces hardcoded ``mini_review`` with a composable pipeline.
    - Each hook implements ``check(content, context) -> HookResult``.
    - ``HookPipeline`` runs hooks sequentially; stops early on ``fatal``.
    - Concrete hooks: ``CitationValidationHook``, ``WordCountHook``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Set


@dataclass
class HookResult:
    """
    Result of a single quality check.
    - **Args**:
        - `passed` (bool): Whether the check passed.
        - `severity` (str): "info" | "warning" | "error" | "fatal".
        - `message` (str): Human-readable feedback.
    """

    passed: bool
    severity: Literal["info", "warning", "error", "fatal"] = "info"
    message: str = ""


@dataclass
class HookContext:
    """
    Shared context for hook execution.
    - **Args**:
        - `valid_citation_keys` (set): Known BibTeX keys.
        - `section_type` (str): Section being checked.
        - `target_words` (int | None): Expected word count.
    """

    valid_citation_keys: Set[str] = field(default_factory=set)
    section_type: str = ""
    target_words: Optional[int] = None


class QualityHook(Protocol):
    """Protocol for quality check hooks."""

    name: str

    async def check(self, content: str, context: HookContext) -> HookResult:
        ...


@dataclass
class PipelineResult:
    """Aggregated result from running all hooks."""

    passed: bool
    results: List[HookResult] = field(default_factory=list)


class HookPipeline:
    """
    Runs quality hooks sequentially.
    - **Description**:
        - Stops early on ``fatal`` severity.
        - ``passed`` is ``False`` if any hook fails.
    """

    def __init__(self, hooks: List[Any]) -> None:
        self._hooks = hooks

    async def run(self, content: str, context: HookContext) -> PipelineResult:
        results: List[HookResult] = []
        all_passed = True

        for hook in self._hooks:
            result = await hook.check(content, context)
            results.append(result)

            if not result.passed:
                all_passed = False

            if result.severity == "fatal":
                break

        return PipelineResult(passed=all_passed, results=results)


# -------------------------------------------------------------------------
# Concrete hooks
# -------------------------------------------------------------------------


class CitationValidationHook:
    """
    Check that all ``\\cite{key}`` references exist in the valid key set.
    - **Description**:
        - Extracts all cite keys from LaTeX content.
        - Reports invalid keys in the message.
    """

    name = "citation_validation"

    _CITE_RE = re.compile(r"\\cite\{([^}]+)\}")

    async def check(self, content: str, context: HookContext) -> HookResult:
        if not context.valid_citation_keys:
            return HookResult(passed=True, severity="info", message="No citation keys to validate against.")

        cited_keys: Set[str] = set()
        for match in self._CITE_RE.finditer(content):
            raw = match.group(1)
            for key in raw.split(","):
                cited_keys.add(key.strip())

        if not cited_keys:
            return HookResult(passed=True, severity="info", message="No citations found.")

        invalid = cited_keys - context.valid_citation_keys
        if invalid:
            return HookResult(
                passed=False,
                severity="error",
                message=f"Invalid citation keys: {', '.join(sorted(invalid))}",
            )

        return HookResult(
            passed=True,
            severity="info",
            message=f"All {len(cited_keys)} citations are valid.",
        )


class WordCountHook:
    """
    Check that content meets a minimum word count.
    - **Description**:
        - Uses ``context.target_words`` as the threshold.
        - If ``target_words`` is None, always passes.
    """

    name = "word_count"

    async def check(self, content: str, context: HookContext) -> HookResult:
        if context.target_words is None:
            return HookResult(
                passed=True, severity="info", message="No target word count."
            )

        word_count = len(content.split())
        threshold = int(context.target_words * 0.5)

        if word_count < threshold:
            return HookResult(
                passed=False,
                severity="error",
                message=f"Word count {word_count} below minimum {threshold} (target: {context.target_words}).",
            )

        return HookResult(
            passed=True,
            severity="info",
            message=f"Word count {word_count} meets target {context.target_words}.",
        )
