"""
Tests for QualityHook pipeline.

Phase 4: Verifies hook execution order, fatal early-stop, result aggregation,
and concrete hook implementations.
"""
import pytest


SAMPLE_LATEX_VALID = r"""
We study the effect of tax policy on innovation \cite{romer1990endogenous}.
Recent work by \cite{bloom2019toolkit} confirms this finding.
Our approach extends previous methods \cite{hall2010patents}.
""".strip()

SAMPLE_LATEX_INVALID_CITE = r"""
We study the effect of tax policy on innovation \cite{romer1990endogenous}.
This is supported by \cite{nonexistent_key_2025}.
""".strip()

VALID_KEYS = {"romer1990endogenous", "bloom2019toolkit", "hall2010patents"}


class TestHookResult:
    def _make(self, **kwargs):
        from src.agents.shared.quality_hooks import HookResult
        defaults = {"passed": True, "severity": "info", "message": "ok"}
        defaults.update(kwargs)
        return HookResult(**defaults)

    def test_creation(self):
        result = self._make()
        assert result.passed is True
        assert result.severity == "info"


class TestHookPipeline:
    """Verify pipeline execution semantics."""

    def _pipeline(self, hooks):
        from src.agents.shared.quality_hooks import HookPipeline
        return HookPipeline(hooks)

    def _context(self, **kwargs):
        from src.agents.shared.quality_hooks import HookContext
        defaults = {"valid_citation_keys": VALID_KEYS, "section_type": "method"}
        defaults.update(kwargs)
        return HookContext(**defaults)

    async def test_runs_all_hooks(self):
        from src.agents.shared.quality_hooks import (
            CitationValidationHook,
            WordCountHook,
        )
        pipeline = self._pipeline([CitationValidationHook(), WordCountHook()])
        ctx = self._context()

        result = await pipeline.run(SAMPLE_LATEX_VALID, ctx)

        assert len(result.results) == 2

    async def test_stops_on_fatal(self):
        from src.agents.shared.quality_hooks import HookResult

        class FatalHook:
            name = "fatal_hook"
            async def check(self, content, context):
                return HookResult(passed=False, severity="fatal", message="fatal error")

        class NeverReachedHook:
            name = "never_reached"
            async def check(self, content, context):
                return HookResult(passed=True, severity="info", message="ok")

        pipeline = self._pipeline([FatalHook(), NeverReachedHook()])
        ctx = self._context()

        result = await pipeline.run(SAMPLE_LATEX_VALID, ctx)

        assert len(result.results) == 1
        assert result.results[0].severity == "fatal"

    async def test_aggregates_results(self):
        from src.agents.shared.quality_hooks import HookResult

        class PassHook:
            name = "pass_hook"
            async def check(self, content, context):
                return HookResult(passed=True, severity="info", message="ok")

        class WarnHook:
            name = "warn_hook"
            async def check(self, content, context):
                return HookResult(passed=True, severity="warning", message="warned")

        pipeline = self._pipeline([PassHook(), WarnHook()])
        ctx = self._context()

        result = await pipeline.run("content", ctx)

        assert result.passed is True
        assert len(result.results) == 2

    async def test_pipeline_fails_if_any_hook_fails(self):
        from src.agents.shared.quality_hooks import HookResult

        class FailHook:
            name = "fail"
            async def check(self, content, context):
                return HookResult(passed=False, severity="error", message="bad")

        class PassHook:
            name = "pass"
            async def check(self, content, context):
                return HookResult(passed=True, severity="info", message="ok")

        pipeline = self._pipeline([PassHook(), FailHook()])
        ctx = self._context()

        result = await pipeline.run("content", ctx)

        assert result.passed is False


class TestCitationValidationHook:
    """Verify citation validation detects invalid keys."""

    def _hook(self):
        from src.agents.shared.quality_hooks import CitationValidationHook
        return CitationValidationHook()

    def _context(self, **kwargs):
        from src.agents.shared.quality_hooks import HookContext
        defaults = {"valid_citation_keys": VALID_KEYS, "section_type": "method"}
        defaults.update(kwargs)
        return HookContext(**defaults)

    async def test_valid_citations_pass(self):
        hook = self._hook()
        ctx = self._context()

        result = await hook.check(SAMPLE_LATEX_VALID, ctx)

        assert result.passed is True

    async def test_invalid_citation_detected(self):
        hook = self._hook()
        ctx = self._context()

        result = await hook.check(SAMPLE_LATEX_INVALID_CITE, ctx)

        assert result.passed is False
        assert "nonexistent_key_2025" in result.message

    async def test_no_citations_passes(self):
        hook = self._hook()
        ctx = self._context()

        result = await hook.check("No citations here.", ctx)

        assert result.passed is True


class TestWordCountHook:
    """Verify word count hook flags short content."""

    def _hook(self):
        from src.agents.shared.quality_hooks import WordCountHook
        return WordCountHook()

    def _context(self, **kwargs):
        from src.agents.shared.quality_hooks import HookContext
        defaults = {
            "valid_citation_keys": set(),
            "section_type": "method",
            "target_words": 200,
        }
        defaults.update(kwargs)
        return HookContext(**defaults)

    async def test_sufficient_words_pass(self):
        hook = self._hook()
        ctx = self._context(target_words=10)
        content = " ".join(["word"] * 20)

        result = await hook.check(content, ctx)

        assert result.passed is True

    async def test_insufficient_words_fail(self):
        hook = self._hook()
        ctx = self._context(target_words=500)
        content = "Too short."

        result = await hook.check(content, ctx)

        assert result.passed is False

    async def test_no_target_always_passes(self):
        hook = self._hook()
        ctx = self._context(target_words=None)

        result = await hook.check("Short.", ctx)

        assert result.passed is True
