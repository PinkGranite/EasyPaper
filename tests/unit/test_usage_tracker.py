"""
Tests for UsageTracker — LLM call accounting.

Phase 2: Verifies token tracking, breakdown by agent/phase, and serialization.
"""
import pytest


class TestLLMCallRecord:
    """Verify LLMCallRecord data integrity."""

    def _make(self, **kwargs):
        from src.agents.shared.usage_tracker import LLMCallRecord
        defaults = {
            "agent": "writer",
            "phase": "generation",
            "section_type": "introduction",
            "model": "gpt-4o",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "latency_ms": 1200.0,
        }
        defaults.update(kwargs)
        return LLMCallRecord(**defaults)

    def test_total_tokens(self):
        record = self._make()
        assert record.total_tokens == 150

    def test_to_dict(self):
        record = self._make()
        d = record.to_dict()
        assert d["agent"] == "writer"
        assert d["total_tokens"] == 150


class TestUsageTracker:
    """Verify UsageTracker accumulation and queries."""

    def _tracker(self):
        from src.agents.shared.usage_tracker import UsageTracker
        return UsageTracker()

    def _record(self, **kwargs):
        from src.agents.shared.usage_tracker import LLMCallRecord
        defaults = {
            "agent": "writer",
            "phase": "generation",
            "section_type": "introduction",
            "model": "gpt-4o",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "latency_ms": 1000.0,
        }
        defaults.update(kwargs)
        return LLMCallRecord(**defaults)

    def test_record_and_total(self):
        tracker = self._tracker()
        tracker.record(self._record(total_tokens=100))
        tracker.record(self._record(total_tokens=200))

        assert tracker.total_tokens == 300

    def test_empty_tracker(self):
        tracker = self._tracker()
        assert tracker.total_tokens == 0

    def test_by_agent(self):
        tracker = self._tracker()
        tracker.record(self._record(agent="writer", total_tokens=100))
        tracker.record(self._record(agent="writer", total_tokens=50))
        tracker.record(self._record(agent="planner", total_tokens=200))

        breakdown = tracker.by_agent()
        assert breakdown["writer"] == 150
        assert breakdown["planner"] == 200

    def test_by_phase(self):
        tracker = self._tracker()
        tracker.record(self._record(phase="planning", total_tokens=300))
        tracker.record(self._record(phase="generation", total_tokens=100))
        tracker.record(self._record(phase="generation", total_tokens=150))

        breakdown = tracker.by_phase()
        assert breakdown["planning"] == 300
        assert breakdown["generation"] == 250

    def test_call_count(self):
        tracker = self._tracker()
        tracker.record(self._record())
        tracker.record(self._record())
        tracker.record(self._record())

        assert tracker.call_count == 3

    def test_to_dict(self):
        tracker = self._tracker()
        tracker.record(self._record(agent="writer", total_tokens=100))

        d = tracker.to_dict()
        assert d["total_tokens"] == 100
        assert d["call_count"] == 1
        assert "by_agent" in d
        assert "by_phase" in d
        assert "calls" in d
        assert len(d["calls"]) == 1
