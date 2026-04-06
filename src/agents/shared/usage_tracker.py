"""
UsageTracker — LLM call accounting for paper generation sessions.
- **Description**:
    - Records per-call token usage, latency, model, agent, and phase.
    - Provides aggregation queries: ``by_agent()``, ``by_phase()``,
      ``by_model()``.
    - Report format (``to_dict()``): top-level summary (with elapsed time),
      per-phase/per-agent/per-model breakdowns, then fine-grained call list.
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


def _aggregate(records: List[LLMCallRecord]) -> dict:
    """Build a subtotal dict from a list of call records."""
    return {
        "call_count": len(records),
        "prompt_tokens": sum(r.prompt_tokens for r in records),
        "completion_tokens": sum(r.completion_tokens for r in records),
        "total_tokens": sum(r.total_tokens for r in records),
        "total_latency_ms": round(sum(r.latency_ms for r in records), 1),
    }


class UsageTracker:
    """
    Accumulates LLM call records and produces a structured report.
    - **Description**:
        - Append-only; designed for a single paper generation run.
        - ``to_dict()`` returns a hierarchical report:
          summary → by_phase → by_agent → by_model → calls.
    """

    def __init__(self) -> None:
        self._calls: List[LLMCallRecord] = []
        self._elapsed_seconds: Optional[float] = None

    def record(self, call: LLMCallRecord) -> None:
        """Append a call record."""
        self._calls.append(call)

    def set_elapsed_time(self, seconds: float) -> None:
        """
        Set the total wall-clock elapsed time for the generation session.
        - **Args**:
            - `seconds` (float): Elapsed seconds.
        """
        self._elapsed_seconds = seconds

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self._calls)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    def by_agent(self) -> Dict[str, dict]:
        """Per-agent subtotals."""
        groups: Dict[str, List[LLMCallRecord]] = defaultdict(list)
        for c in self._calls:
            groups[c.agent].append(c)
        return {k: _aggregate(v) for k, v in groups.items()}

    def by_phase(self) -> Dict[str, dict]:
        """Per-phase subtotals."""
        groups: Dict[str, List[LLMCallRecord]] = defaultdict(list)
        for c in self._calls:
            groups[c.phase].append(c)
        return {k: _aggregate(v) for k, v in groups.items()}

    def by_model(self) -> Dict[str, dict]:
        """Per-model subtotals."""
        groups: Dict[str, List[LLMCallRecord]] = defaultdict(list)
        for c in self._calls:
            groups[c.model].append(c)
        return {k: _aggregate(v) for k, v in groups.items()}

    def to_dict(self) -> dict:
        """
        Build the full structured report.
        - **Returns**:
            - `dict`: Hierarchical report with keys: summary, by_phase,
              by_agent, by_model, calls.
        """
        summary: dict = {
            "total_calls": self.call_count,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": sum(c.prompt_tokens for c in self._calls),
            "total_completion_tokens": sum(c.completion_tokens for c in self._calls),
            "total_latency_ms": round(sum(c.latency_ms for c in self._calls), 1),
        }
        if self._elapsed_seconds is not None:
            summary["elapsed_seconds"] = self._elapsed_seconds

        return {
            "summary": summary,
            "by_phase": self.by_phase(),
            "by_agent": self.by_agent(),
            "by_model": self.by_model(),
            "calls": [c.to_dict() for c in self._calls],
        }
