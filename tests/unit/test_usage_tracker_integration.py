"""
Tests for UsageTracker integration — contextvar injection and result propagation.

TDD RED phase: these tests verify the three integration points:
1. LLMClient records usage via contextvar-injected UsageTracker
2. PaperGenerationResult carries a `usage` field
3. MetaDataAgent lifecycle (create tracker → inject → attach to result)
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# 1. LLMClient contextvar integration
# ---------------------------------------------------------------------------

class TestLLMClientUsageTracking:
    """Verify _CompletionsProxy records token usage to a contextvar tracker."""

    def _make_response(self, prompt_tokens=100, completion_tokens=50):
        choice = MagicMock()
        choice.message.content = "Hello world"
        choice.message._thinking = None
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = MagicMock(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return resp

    def _make_completions(self, response):
        """Build a mock completions object whose .create() returns *response*."""
        completions = MagicMock()
        completions.create = AsyncMock(return_value=response)
        return completions

    @pytest.mark.asyncio
    async def test_records_usage_when_tracker_active(self):
        """When a UsageTracker is set via contextvar, each LLM call appends a record."""
        from src.agents.shared.usage_tracker import UsageTracker
        from src.agents.shared.llm_client import (
            set_usage_tracker_context,
            clear_usage_tracker_context,
            _CompletionsProxy,
        )

        tracker = UsageTracker()
        set_usage_tracker_context(tracker, agent="writer", phase="generation", section="introduction")

        try:
            proxy = _CompletionsProxy(self._make_completions(self._make_response(200, 80)))
            await proxy.create(model="gpt-4o")

            assert tracker.call_count == 1
            assert tracker.total_tokens == 280
            record = tracker.to_dict()["calls"][0]
            assert record["agent"] == "writer"
            assert record["phase"] == "generation"
            assert record["section_type"] == "introduction"
            assert record["model"] == "gpt-4o"
            assert record["prompt_tokens"] == 200
            assert record["completion_tokens"] == 80
        finally:
            clear_usage_tracker_context()

    @pytest.mark.asyncio
    async def test_no_tracker_no_error(self):
        """When no tracker is active, LLM calls work normally without recording."""
        from src.agents.shared.llm_client import clear_usage_tracker_context, _CompletionsProxy

        clear_usage_tracker_context()
        proxy = _CompletionsProxy(self._make_completions(self._make_response()))
        resp = await proxy.create(model="gpt-4o")

        assert resp.choices[0].message.content == "Hello world"

    @pytest.mark.asyncio
    async def test_multiple_calls_accumulate(self):
        """Multiple LLM calls accumulate records in the same tracker."""
        from src.agents.shared.usage_tracker import UsageTracker
        from src.agents.shared.llm_client import (
            set_usage_tracker_context,
            clear_usage_tracker_context,
            _CompletionsProxy,
        )

        tracker = UsageTracker()
        set_usage_tracker_context(tracker, agent="writer", phase="generation", section="method")

        try:
            proxy = _CompletionsProxy(self._make_completions(self._make_response(100, 50)))

            await proxy.create(model="gpt-4o")
            await proxy.create(model="gpt-4o")
            await proxy.create(model="gpt-4o")

            assert tracker.call_count == 3
            assert tracker.total_tokens == 450
        finally:
            clear_usage_tracker_context()

    @pytest.mark.asyncio
    async def test_tracker_records_latency(self):
        """Recorded latency_ms should be positive."""
        from src.agents.shared.usage_tracker import UsageTracker
        from src.agents.shared.llm_client import (
            set_usage_tracker_context,
            clear_usage_tracker_context,
            _CompletionsProxy,
        )

        tracker = UsageTracker()
        set_usage_tracker_context(tracker, agent="planner", phase="planning", section="")

        try:
            proxy = _CompletionsProxy(self._make_completions(self._make_response()))
            await proxy.create(model="gpt-4o")

            record = tracker.to_dict()["calls"][0]
            assert record["latency_ms"] >= 0
        finally:
            clear_usage_tracker_context()

    @pytest.mark.asyncio
    async def test_update_context_changes_metadata(self):
        """update_usage_tracker_context changes agent/phase/section for subsequent calls."""
        from src.agents.shared.usage_tracker import UsageTracker
        from src.agents.shared.llm_client import (
            set_usage_tracker_context,
            update_usage_tracker_context,
            clear_usage_tracker_context,
            _CompletionsProxy,
        )

        tracker = UsageTracker()
        set_usage_tracker_context(tracker, agent="planner", phase="planning", section="")

        try:
            proxy = _CompletionsProxy(self._make_completions(self._make_response(50, 30)))
            await proxy.create(model="gpt-4o")

            update_usage_tracker_context(agent="writer", phase="generation", section="introduction")
            await proxy.create(model="gpt-4o")

            calls = tracker.to_dict()["calls"]
            assert calls[0]["agent"] == "planner"
            assert calls[0]["phase"] == "planning"
            assert calls[1]["agent"] == "writer"
            assert calls[1]["phase"] == "generation"
            assert calls[1]["section_type"] == "introduction"
        finally:
            clear_usage_tracker_context()

    @pytest.mark.asyncio
    async def test_handles_missing_usage_attribute(self):
        """If response.usage is None, records zeros without crashing."""
        from src.agents.shared.usage_tracker import UsageTracker
        from src.agents.shared.llm_client import (
            set_usage_tracker_context,
            clear_usage_tracker_context,
            _CompletionsProxy,
        )

        tracker = UsageTracker()
        set_usage_tracker_context(tracker, agent="writer", phase="generation", section="method")

        try:
            resp = self._make_response()
            resp.usage = None
            proxy = _CompletionsProxy(self._make_completions(resp))
            await proxy.create(model="gpt-4o")

            assert tracker.call_count == 1
            assert tracker.total_tokens == 0
        finally:
            clear_usage_tracker_context()


# ---------------------------------------------------------------------------
# 2. PaperGenerationResult.usage field
# ---------------------------------------------------------------------------

class TestPaperGenerationResultUsage:
    """Verify PaperGenerationResult carries optional usage dict."""

    def test_usage_field_defaults_to_none(self):
        from src.agents.metadata_agent.models import PaperGenerationResult

        result = PaperGenerationResult(status="ok")
        assert result.usage is None

    def test_usage_field_accepts_dict(self):
        from src.agents.metadata_agent.models import PaperGenerationResult

        usage_data = {
            "total_tokens": 5000,
            "call_count": 10,
            "by_agent": {"writer": 3000, "planner": 2000},
            "by_phase": {"planning": 2000, "generation": 3000},
            "calls": [],
        }
        result = PaperGenerationResult(status="ok", usage=usage_data)
        assert result.usage is not None
        assert result.usage["total_tokens"] == 5000
        assert result.usage["call_count"] == 10

    def test_usage_serializes_to_json(self):
        from src.agents.metadata_agent.models import PaperGenerationResult

        usage_data = {"total_tokens": 100, "call_count": 1, "by_agent": {}, "by_phase": {}, "calls": []}
        result = PaperGenerationResult(status="ok", usage=usage_data)
        d = result.model_dump()
        assert d["usage"]["total_tokens"] == 100

    def test_usage_none_serializes_cleanly(self):
        from src.agents.metadata_agent.models import PaperGenerationResult

        result = PaperGenerationResult(status="ok")
        d = result.model_dump()
        assert d["usage"] is None
