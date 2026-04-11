"""
Tests for EasyPaper SDK generate_metadata_from_folder method.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.config.schema import AppConfig, AgentConfig, ModelConfig, SkillsConfig


LLM_SYNTH_RESPONSE = json.dumps({
    "title": "Generated Title",
    "idea_hypothesis": "Some hypothesis.",
    "method": "Some method.",
    "data": "Some data.",
    "experiments": "Some experiments.",
})


def _minimal_config() -> AppConfig:
    agents = [
        AgentConfig(name="metadata", model=ModelConfig(model_name="m", api_key="k", base_url="http://x")),
        AgentConfig(name="writer", model=ModelConfig(model_name="m", api_key="k", base_url="http://x")),
        AgentConfig(name="reviewer", model=ModelConfig(model_name="m", api_key="k", base_url="http://x")),
        AgentConfig(name="planner", model=ModelConfig(model_name="m", api_key="k", base_url="http://x")),
    ]
    return AppConfig(agents=agents, skills=SkillsConfig(enabled=False))


class TestSDKMetadataGen:
    def test_generate_metadata_from_folder_method_exists(self):
        from easypaper import EasyPaper
        assert hasattr(EasyPaper, "generate_metadata_from_folder")

    @pytest.mark.asyncio
    async def test_generate_metadata_from_folder_returns_metadata(self, tmp_path: Path):
        from easypaper import EasyPaper

        (tmp_path / "notes.md").write_text("# Idea\nSome idea.\n", encoding="utf-8")

        def _mock_response(content):
            choice = MagicMock()
            choice.message.content = content
            resp = MagicMock()
            resp.choices = [choice]
            resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
            return resp

        with patch("easypaper.client.EasyPaper._build_agents") as mock_build:
            mock_meta_agent = MagicMock()
            mock_build.return_value = {"metadata": mock_meta_agent}

            ep = EasyPaper(config=_minimal_config())

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_mock_response(LLM_SYNTH_RESPONSE)
            )
            ep._metadata_agent = mock_meta_agent
            ep._metadata_agent.client = mock_client
            ep._metadata_agent.model_name = "test"

            from src.agents.metadata_agent.metadata_generator import generate_metadata_from_folder
            result = await generate_metadata_from_folder(
                folder_path=str(tmp_path),
                llm_client=mock_client,
                model_name="test",
            )

            from src.agents.metadata_agent.models import PaperMetaData
            assert isinstance(result, PaperMetaData)
