"""
PaperSession — Unified state container for paper generation.
- **Description**:
    - Replaces the dual-state pattern (local ``generated_sections`` dict +
      ``SessionMemory``) with a single source of truth.
    - ``SectionState`` tracks per-section lifecycle (pending → draft → final).
    - ``PaperSession`` provides token-aware cross-section context via
      ``get_context_for_section()``.
    - ``as_memory()`` returns a backward-compatible ``SessionMemory`` view
      whose ``generated_sections`` dict is kept in sync.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from .session_memory import SessionMemory


@dataclass
class SectionState:
    """
    Per-section lifecycle state.
    - **Description**:
        - Wraps content, status, and word count for a single paper section.
        - Provides summary/full_or_summary helpers for token-aware context.

    - **Args**:
        - `section_type` (str): Section identifier (e.g. "introduction").
        - `status` (str): Lifecycle status.
        - `content` (str): LaTeX content.
    """

    section_type: str
    status: Literal["pending", "writing", "draft", "reviewing", "final"] = "pending"
    content: str = ""
    _word_count: Optional[int] = field(default=None, repr=False)

    @property
    def word_count(self) -> int:
        if self._word_count is None:
            self._word_count = len(self.content.split()) if self.content else 0
        return self._word_count

    def estimate_tokens(self) -> int:
        """Rough token estimate: chars / 4 + 1 (same heuristic as Claude Code)."""
        return (len(self.content) // 4) + 1 if self.content else 0

    def summary(self, max_chars: int = 300) -> str:
        """
        Compact summary: first sentence + last sentence + word count.
        - **Description**:
            - Deterministic (no LLM call).  Suitable for cross-section context.

        - **Args**:
            - `max_chars` (int): Soft cap on summary length.

        - **Returns**:
            - `summary` (str): Human-readable preview.
        """
        if not self.content.strip():
            return ""
        sentences = [
            s.strip()
            for s in self.content.replace("\n", " ").split(".")
            if s.strip()
        ]
        first = (sentences[0] + ".") if sentences else ""
        last = (sentences[-1] + ".") if len(sentences) > 1 else ""
        preview = first
        if last and last != first:
            preview += f" ... {last}"
        if len(preview) > max_chars:
            preview = preview[: max_chars - 3] + "..."
        return f"**{self.section_type}** ({self.word_count} words): {preview}"

    def full_or_summary(self, token_budget: int) -> str:
        """
        Return full content when it fits, otherwise a summary.
        - **Args**:
            - `token_budget` (int): Available token budget.

        - **Returns**:
            - Content string (full or summarized).
        """
        if self.estimate_tokens() <= token_budget:
            return self.content
        budget_chars = max(token_budget * 3, 60)
        return self.summary(max_chars=budget_chars)


class PaperSession:
    """
    Single source of truth for a paper generation session.
    - **Description**:
        - Manages section states, contributions, and review history.
        - Provides ``get_context_for_section()`` with dynamic token budget.
        - ``as_memory()`` returns a live ``SessionMemory`` reference for
          backward compatibility with Writer/Reviewer agents.
    """

    def __init__(self) -> None:
        self.sections: Dict[str, SectionState] = {}
        self.contributions: List[str] = []
        self._memory = SessionMemory()

    # ------------------------------------------------------------------
    # Section management
    # ------------------------------------------------------------------

    def update_section(self, section_type: str, content: str) -> None:
        """
        Store or update section content.
        - **Description**:
            - Creates ``SectionState`` if new, updates content + status.
            - Keeps internal ``SessionMemory.generated_sections`` in sync.
        """
        if section_type in self.sections:
            state = self.sections[section_type]
            state.content = content
            state.status = "draft"
            state._word_count = None
        else:
            self.sections[section_type] = SectionState(
                section_type=section_type,
                status="draft",
                content=content,
            )
        self._memory.update_section(section_type, content)

    def get_section_state(self, section_type: str) -> Optional[SectionState]:
        return self.sections.get(section_type)

    @property
    def generated_sections(self) -> Dict[str, str]:
        """Flat dict view compatible with existing code."""
        return {st: s.content for st, s in self.sections.items() if s.content}

    # ------------------------------------------------------------------
    # Cross-section context
    # ------------------------------------------------------------------

    def get_context_for_section(
        self,
        section_type: str,
        model_context_window: int,
        reserved_tokens: int = 2000,
    ) -> str:
        """
        Build optimally-sized context for writing a specific section.
        - **Description**:
            - Allocates token budget proportionally across context blocks.
            - Priority: plan guidance > contributions > prior section summaries.

        - **Args**:
            - `section_type` (str): Section being written.
            - `model_context_window` (int): Model's total context window.
            - `reserved_tokens` (int): Tokens reserved for system prompt + output.

        - **Returns**:
            - `context` (str): Assembled context string.
        """
        available = max(model_context_window - reserved_tokens, 500)
        parts: List[str] = []

        # Contributions (fixed, small)
        if self.contributions:
            contrib_block = "## Key Contributions\n" + "\n".join(
                f"- {c}" for c in self.contributions
            )
            parts.append(contrib_block)

        # Prior sections (budget-aware)
        section_budget = min(available // 2, 4000)
        prior_parts: List[str] = []
        for stype, state in self.sections.items():
            if stype == section_type or not state.content.strip():
                continue
            per_section_budget = section_budget // max(
                len(self.sections) - 1, 1
            )
            body = state.full_or_summary(token_budget=per_section_budget)
            if body == state.content:
                prior_parts.append(
                    f"**{stype}** ({state.word_count} words):\n{body}"
                )
            else:
                prior_parts.append(body)

        if prior_parts:
            parts.append(
                "## Already-Written Sections\n" + "\n".join(prior_parts)
            )

        return "\n\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    def as_memory(self) -> SessionMemory:
        """
        Return the internal SessionMemory instance (live reference).
        - **Description**:
            - The returned object shares ``generated_sections`` with this
              PaperSession, so updates via either path are visible to both.
        """
        return self._memory

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "sections": {
                st: {
                    "section_type": s.section_type,
                    "status": s.status,
                    "content": s.content,
                }
                for st, s in self.sections.items()
            },
            "contributions": self.contributions,
            "memory": self._memory.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PaperSession":
        session = cls()
        for st, sdata in data.get("sections", {}).items():
            session.sections[st] = SectionState(
                section_type=sdata["section_type"],
                status=sdata.get("status", "draft"),
                content=sdata.get("content", ""),
            )
        session.contributions = data.get("contributions", [])
        mem_data = data.get("memory")
        if mem_data:
            session._memory = SessionMemory.from_dict(mem_data)
        # Ensure memory.generated_sections is in sync
        for st, state in session.sections.items():
            if state.content:
                session._memory.update_section(st, state.content)
        return session
