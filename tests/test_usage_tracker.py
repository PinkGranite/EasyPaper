"""
TDD tests for UsageTracker and config loading.

Covers three issues:
1. load_config() ignores --easypaper-config because load_dotenv(override=True)
   overwrites the env var set by the caller.
2. UsageTracker labels all calls as "MetaDataAgent" / "generation" — sub-agent
   calls are never relabelled.
3. Report format lacks: overall summary with elapsed_time, per-phase breakdown,
   per-model breakdown.

RED phase: these tests should FAIL until fixes are applied.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from src.agents.shared.usage_tracker import UsageTracker, LLMCallRecord


# ---------------------------------------------------------------------------
# 1. load_config respects programmatic AGENT_CONFIG_PATH
# ---------------------------------------------------------------------------

class TestConfigOverride:
    """load_config must use the env var set by the caller, not .env override."""

    def test_load_config_does_not_use_lru_cache(self):
        """load_config should NOT be decorated with lru_cache, which prevents
        reloading when the config path changes."""
        from src.config.loader import load_config
        import functools

        wrapper = getattr(load_config, "__wrapped__", None)
        cache_info = getattr(load_config, "cache_info", None)
        assert cache_info is None, (
            "load_config uses @lru_cache — subsequent calls with different "
            "AGENT_CONFIG_PATH will silently return stale config."
        )

    def test_load_dotenv_does_not_override_caller(self):
        """load_dotenv must NOT override=True, otherwise .env values clobber
        programmatically-set AGENT_CONFIG_PATH."""
        import inspect
        from src.config import loader as loader_mod

        source = inspect.getsource(loader_mod.load_config)
        assert "override=True" not in source, (
            "load_config calls load_dotenv(override=True) — this overwrites "
            "AGENT_CONFIG_PATH set by EasyPaper.__init__, causing "
            "--easypaper-config to be silently ignored."
        )


# ---------------------------------------------------------------------------
# 2. UsageTracker — multi-agent / multi-phase labelling
# ---------------------------------------------------------------------------

class TestUsageTrackerLabelling:
    """Sub-agent calls must be labelled with their own agent/phase/section."""

    def _make_tracker_with_mixed_calls(self) -> UsageTracker:
        tracker = UsageTracker()
        tracker.record(LLMCallRecord(
            agent="PlannerAgent", phase="planning", section_type="",
            model="minimax/M2.7", prompt_tokens=1000, completion_tokens=200,
            total_tokens=1200, latency_ms=3000,
        ))
        tracker.record(LLMCallRecord(
            agent="WriterAgent", phase="generation", section_type="introduction",
            model="minimax/M2.7", prompt_tokens=2000, completion_tokens=800,
            total_tokens=2800, latency_ms=5000,
        ))
        tracker.record(LLMCallRecord(
            agent="ReviewerAgent", phase="review", section_type="introduction",
            model="minimax/M2.7", prompt_tokens=3000, completion_tokens=500,
            total_tokens=3500, latency_ms=4000,
        ))
        tracker.record(LLMCallRecord(
            agent="TypesetterAgent", phase="compilation", section_type="",
            model="minimax/M2.7", prompt_tokens=500, completion_tokens=100,
            total_tokens=600, latency_ms=2000,
        ))
        return tracker

    def test_by_agent_distinct(self):
        tracker = self._make_tracker_with_mixed_calls()
        by_agent = tracker.by_agent()
        assert len(by_agent) == 4, f"Expected 4 agents, got {list(by_agent.keys())}"
        assert "PlannerAgent" in by_agent
        assert "WriterAgent" in by_agent
        assert "ReviewerAgent" in by_agent
        assert "TypesetterAgent" in by_agent

    def test_by_phase_distinct(self):
        tracker = self._make_tracker_with_mixed_calls()
        by_phase = tracker.by_phase()
        assert len(by_phase) >= 3, f"Expected >=3 phases, got {list(by_phase.keys())}"
        assert "planning" in by_phase
        assert "generation" in by_phase
        assert "review" in by_phase

    def test_by_model(self):
        """to_dict should contain per-model token breakdown."""
        tracker = self._make_tracker_with_mixed_calls()
        report = tracker.to_dict()
        assert "by_model" in report, (
            "Report missing 'by_model' — need per-model token breakdown."
        )
        assert "minimax/M2.7" in report["by_model"]

    def test_update_context_actually_used(self):
        """MetaDataAgent.execute_generation must call update_usage_tracker_context
        before delegating to sub-agents so calls are labelled correctly."""
        import inspect
        from src.agents.metadata_agent.metadata_agent import MetaDataAgent

        src = inspect.getsource(MetaDataAgent.execute_generation)
        assert "update_usage_tracker_context" in src, (
            "execute_generation never calls update_usage_tracker_context — "
            "all sub-agent calls are labelled as MetaDataAgent."
        )


# ---------------------------------------------------------------------------
# 3. Report format — summary / per-phase / per-call
# ---------------------------------------------------------------------------

class TestUsageReportFormat:
    """Report must have: overall summary (with elapsed), per-phase, per-call."""

    def _make_tracker(self) -> UsageTracker:
        tracker = UsageTracker()
        tracker.record(LLMCallRecord(
            agent="PlannerAgent", phase="planning", section_type="",
            model="minimax/M2.7", prompt_tokens=1000, completion_tokens=200,
            total_tokens=1200, latency_ms=3000,
        ))
        tracker.record(LLMCallRecord(
            agent="WriterAgent", phase="generation", section_type="introduction",
            model="minimax/M2.7", prompt_tokens=2000, completion_tokens=800,
            total_tokens=2800, latency_ms=5000,
        ))
        tracker.record(LLMCallRecord(
            agent="ReviewerAgent", phase="review", section_type="",
            model="openai/gpt-4o", prompt_tokens=500, completion_tokens=100,
            total_tokens=600, latency_ms=2000,
        ))
        return tracker

    def test_summary_section_exists(self):
        tracker = self._make_tracker()
        report = tracker.to_dict()
        assert "summary" in report, "Report missing top-level 'summary'."

    def test_summary_has_elapsed_time(self):
        tracker = self._make_tracker()
        tracker.set_elapsed_time(120.5)
        report = tracker.to_dict()
        summary = report.get("summary", {})
        assert "elapsed_seconds" in summary, (
            "Summary missing 'elapsed_seconds'."
        )
        assert summary["elapsed_seconds"] == 120.5

    def test_summary_has_totals(self):
        tracker = self._make_tracker()
        report = tracker.to_dict()
        summary = report.get("summary", {})
        assert summary.get("total_tokens") == 4600
        assert summary.get("total_calls") == 3
        assert summary.get("total_prompt_tokens") == 3500
        assert summary.get("total_completion_tokens") == 1100

    def test_by_phase_section_exists(self):
        tracker = self._make_tracker()
        report = tracker.to_dict()
        assert "by_phase" in report
        phases = report["by_phase"]
        assert "planning" in phases
        assert "generation" in phases
        assert "review" in phases

    def test_by_phase_has_subtotals(self):
        """Each phase entry should have token subtotals, not just total_tokens."""
        tracker = self._make_tracker()
        report = tracker.to_dict()
        planning = report["by_phase"]["planning"]
        assert isinstance(planning, dict), (
            "by_phase values should be dicts with subtotals, not bare ints."
        )
        assert "total_tokens" in planning
        assert "call_count" in planning
        assert "prompt_tokens" in planning
        assert "completion_tokens" in planning

    def test_by_model_section_exists(self):
        tracker = self._make_tracker()
        report = tracker.to_dict()
        assert "by_model" in report
        assert "minimax/M2.7" in report["by_model"]
        assert "openai/gpt-4o" in report["by_model"]

    def test_by_agent_section_exists(self):
        tracker = self._make_tracker()
        report = tracker.to_dict()
        assert "by_agent" in report
        agents = report["by_agent"]
        assert isinstance(list(agents.values())[0], dict), (
            "by_agent values should be dicts with subtotals, not bare ints."
        )

    def test_calls_section_exists(self):
        tracker = self._make_tracker()
        report = tracker.to_dict()
        assert "calls" in report
        assert len(report["calls"]) == 3

    def test_report_json_serializable(self):
        tracker = self._make_tracker()
        tracker.set_elapsed_time(60.0)
        report = tracker.to_dict()
        serialized = json.dumps(report, indent=2)
        assert len(serialized) > 0
