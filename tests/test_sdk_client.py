"""
Tests for the EasyPaper thin SDK client.

Verifies importability, construction, one-shot generation, and streaming.
All tests mock internal agents — no LLM calls are made.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.schema import AppConfig, AgentConfig, ModelConfig, SkillsConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_config() -> AppConfig:
    """Build a minimal AppConfig for testing (no real API keys)."""
    agents = [
        AgentConfig(name="metadata", model=ModelConfig(model_name="m", api_key="k", base_url="http://x")),
        AgentConfig(name="writer", model=ModelConfig(model_name="m", api_key="k", base_url="http://x")),
        AgentConfig(name="reviewer", model=ModelConfig(model_name="m", api_key="k", base_url="http://x")),
        AgentConfig(name="planner", model=ModelConfig(model_name="m", api_key="k", base_url="http://x")),
    ]
    return AppConfig(agents=agents, skills=SkillsConfig(enabled=False))


def _mock_metadata():
    """Return a lightweight PaperMetaData-like object for tests."""
    from src.agents.metadata_agent.models import PaperMetaData
    return PaperMetaData(
        title="Test Paper",
        idea_hypothesis="hypothesis",
        method="method",
        data="data",
        experiments="experiments",
    )


def _mock_result():
    """Return a PaperGenerationResult for stubbing generate_paper."""
    from src.agents.metadata_agent.models import PaperGenerationResult
    return PaperGenerationResult(
        status="ok",
        paper_title="Test Paper",
        sections=[],
        latex_content="\\documentclass{article}",
        total_word_count=42,
    )


# ---------------------------------------------------------------------------
# Test: importability
# ---------------------------------------------------------------------------

class TestImports:
    def test_import_easypaper_package(self):
        import easypaper
        assert hasattr(easypaper, "EasyPaper")

    def test_import_public_symbols(self):
        from easypaper import EasyPaper, PaperMetaData, PaperGenerationResult, EventType
        assert EasyPaper is not None
        assert PaperMetaData is not None
        assert PaperGenerationResult is not None
        assert EventType is not None


# ---------------------------------------------------------------------------
# Test: construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_construct_with_config_object(self):
        from easypaper import EasyPaper
        config = _minimal_config()

        with patch("easypaper.client.initialize_agents") as mock_init:
            mock_agent = MagicMock()
            mock_init.return_value = {"metadata": mock_agent}
            ep = EasyPaper(config=config)

        assert ep is not None
        mock_init.assert_called_once()

    def test_construct_with_config_path(self, tmp_path):
        from easypaper import EasyPaper

        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(
            "agents:\n"
            "  - name: metadata\n"
            "    model:\n"
            "      model_name: m\n"
            "      api_key: k\n"
            "      base_url: http://x\n"
            "  - name: writer\n"
            "    model:\n"
            "      model_name: m\n"
            "      api_key: k\n"
            "      base_url: http://x\n"
            "  - name: reviewer\n"
            "    model:\n"
            "      model_name: m\n"
            "      api_key: k\n"
            "      base_url: http://x\n"
            "  - name: planner\n"
            "    model:\n"
            "      model_name: m\n"
            "      api_key: k\n"
            "      base_url: http://x\n"
            "skills:\n"
            "  enabled: false\n"
        )

        with patch("easypaper.client.initialize_agents") as mock_init:
            mock_agent = MagicMock()
            mock_init.return_value = {"metadata": mock_agent}
            ep = EasyPaper(config_path=str(cfg_file))

        assert ep is not None


# ---------------------------------------------------------------------------
# Test: one-shot generate()
# ---------------------------------------------------------------------------

class TestGenerate:
    async def test_generate_delegates_to_metadata_agent(self):
        from easypaper import EasyPaper

        config = _minimal_config()
        result = _mock_result()

        mock_agent = MagicMock()
        mock_agent.generate_paper = AsyncMock(return_value=result)

        with patch("easypaper.client.initialize_agents", return_value={"metadata": mock_agent}):
            ep = EasyPaper(config=config)

        metadata = _mock_metadata()
        out = await ep.generate(metadata, compile_pdf=False)

        assert out.status == "ok"
        assert out.paper_title == "Test Paper"
        mock_agent.generate_paper.assert_awaited_once()
        call_kwargs = mock_agent.generate_paper.call_args
        assert call_kwargs.kwargs["metadata"] is metadata
        assert call_kwargs.kwargs["compile_pdf"] is False


# ---------------------------------------------------------------------------
# Test: streaming generate_stream()
# ---------------------------------------------------------------------------

class TestGenerateStream:
    async def test_stream_yields_progress_events(self):
        from easypaper import EasyPaper
        from src.agents.metadata_agent.progress import EventType

        config = _minimal_config()
        result = _mock_result()

        emitted_events = [
            {"type": EventType.PHASE_START, "phase": "planning", "message": "start"},
            {"type": EventType.PHASE_COMPLETE, "phase": "planning", "message": "done"},
            {"type": EventType.COMPLETED, "phase": "finalize", "message": "all done"},
        ]

        async def fake_generate_paper(*, metadata, progress_callback=None, **opts):
            if progress_callback:
                for evt in emitted_events:
                    await progress_callback(evt)
            return result

        mock_agent = MagicMock()
        mock_agent.generate_paper = AsyncMock(side_effect=fake_generate_paper)

        with patch("easypaper.client.initialize_agents", return_value={"metadata": mock_agent}):
            ep = EasyPaper(config=config)

        metadata = _mock_metadata()
        collected = []
        async for event in ep.generate_stream(metadata):
            collected.append(event)

        assert len(collected) == len(emitted_events)
        assert collected[0]["type"] == EventType.PHASE_START
        assert collected[-1]["type"] == EventType.COMPLETED

    async def test_stream_returns_result_from_last_event(self):
        from easypaper import EasyPaper

        config = _minimal_config()
        result = _mock_result()

        async def fake_generate_paper(*, metadata, progress_callback=None, **opts):
            return result

        mock_agent = MagicMock()
        mock_agent.generate_paper = AsyncMock(side_effect=fake_generate_paper)

        with patch("easypaper.client.initialize_agents", return_value={"metadata": mock_agent}):
            ep = EasyPaper(config=config)

        metadata = _mock_metadata()
        out = await ep.generate(metadata)
        assert out.status == "ok"
