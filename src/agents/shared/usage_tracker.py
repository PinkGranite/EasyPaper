"""
UsageTracker — LLM call accounting for paper generation sessions.
- **Description**:
    - Records per-call token usage, latency, model, agent, and phase.
    - Provides aggregation queries: ``total_tokens``, ``by_agent()``,
      ``by_phase()``.
    - Thread-safe via append-only list; designed for single-session use.
    - Can be injected into ``LLMClient`` via ``contextvars`` for automatic
      recording of every LLM call.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class LLMCallRecord:
    """
    Record of a single LLM API call.
    - **Args**:
        - `agent` (str): Agent name that made the call.
        - `phase` (str): Pipeline phase (planning / generation / review).
        - `section_type` (str): Which section the call relates to.
        - `model` (str): Model identifier.
        - `prompt_tokens` (int): Input tokens.
        - `completion_tokens` (int): Output tokens.
        - `total_tokens` (int): Total tokens.
        - `latency_ms` (float): Wall-clock latency.
    """

    agent: str
    phase: str
    section_type: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


class UsageTracker:
    """
    Accumulates LLM call records and provides aggregation queries.
    - **Description**:
        - Append-only; designed for a single paper generation run.
        - ``by_agent()`` and ``by_phase()`` return ``{key: total_tokens}``
          breakdowns.
    """

    def __init__(self) -> None:
        self._calls: List[LLMCallRecord] = []

    def record(self, call: LLMCallRecord) -> None:
        """Append a call record."""
        self._calls.append(call)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self._calls)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    def by_agent(self) -> Dict[str, int]:
        result: Dict[str, int] = defaultdict(int)
        for c in self._calls:
            result[c.agent] += c.total_tokens
        return dict(result)

    def by_phase(self) -> Dict[str, int]:
        result: Dict[str, int] = defaultdict(int)
        for c in self._calls:
            result[c.phase] += c.total_tokens
        return dict(result)

    def to_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "by_agent": self.by_agent(),
            "by_phase": self.by_phase(),
            "calls": [c.to_dict() for c in self._calls],
        }
