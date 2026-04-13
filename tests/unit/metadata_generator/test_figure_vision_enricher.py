"""Tests for vision-based figure description enrichment."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from PIL import Image

from src.agents.metadata_agent.models import FigureSpec, PaperMetaData
from src.agents.metadata_agent.metadata_generator.figure_vision_enricher import (
    enrich_figure_descriptions_vision,
    _read_cached_description,
)


@pytest.mark.asyncio
async def test_enrich_updates_description(tmp_path: Path) -> None:
    img = tmp_path / "chart.png"
    Image.new("RGB", (40, 30), color=(200, 10, 50)).save(img)
    cache_dir = tmp_path / "vcache"

    md = PaperMetaData(
        title="T",
        idea_hypothesis="i",
        method="m",
        data="d",
        experiments="e",
        materials_root=str(tmp_path),
        figures=[
            FigureSpec(
                id="fig:h1",
                caption="Chart",
                description="Image file: chart.png",
                file_path="chart.png",
            ),
        ],
    )

    choice = MagicMock()
    choice.message.content = "Synthetic test patch: red-dominant rectangle, no axes."
    resp = MagicMock()
    resp.choices = [choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=resp)

    await enrich_figure_descriptions_vision(
        md,
        mock_client,
        "gpt-4o-mini",
        cache_dir=cache_dir,
        max_long_edge=128,
    )

    assert "Synthetic test patch" in md.figures[0].description
    mock_client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_enrich_cache_skips_second_api_call(tmp_path: Path) -> None:
    img = tmp_path / "x.png"
    Image.new("RGB", (8, 8), color="green").save(img)
    cache_dir = tmp_path / "vcache2"

    md = PaperMetaData(
        title="T",
        idea_hypothesis="i",
        method="m",
        data="d",
        experiments="e",
        materials_root=str(tmp_path),
        figures=[
            FigureSpec(
                id="f1",
                caption="X",
                description="old",
                file_path="x.png",
            ),
        ],
    )

    choice = MagicMock()
    choice.message.content = "Cached description text."
    resp = MagicMock()
    resp.choices = [choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=resp)

    await enrich_figure_descriptions_vision(md, mock_client, "m", cache_dir=cache_dir)
    assert mock_client.chat.completions.create.await_count == 1

    md2 = PaperMetaData(
        title="T",
        idea_hypothesis="i",
        method="m",
        data="d",
        experiments="e",
        materials_root=str(tmp_path),
        figures=[
            FigureSpec(
                id="f1",
                caption="X",
                description="old",
                file_path="x.png",
            ),
        ],
    )
    mock_client.chat.completions.create.reset_mock()
    await enrich_figure_descriptions_vision(md2, mock_client, "m", cache_dir=cache_dir)
    assert mock_client.chat.completions.create.await_count == 0
    assert md2.figures[0].description == "Cached description text."


def test_read_cached_missing_returns_none(tmp_path: Path) -> None:
    assert _read_cached_description(tmp_path, "no_such_key") is None
