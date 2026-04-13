"""Tests for phased metadata extraction orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
from unittest.mock import patch

from src.agents.metadata_agent.models import PaperMetaData
from src.agents.metadata_agent.metadata_generator import generate_metadata_from_folder
from src.agents.metadata_agent.metadata_generator.models import ExtractedFragment


@pytest.mark.asyncio
async def test_generate_metadata_adds_phase_tags(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "# Overview\nResearch hypothesis and context.\n",
        encoding="utf-8",
    )
    (tmp_path / "method.py").write_text(
        "def train_model():\n    return 1\n",
        encoding="utf-8",
    )
    (tmp_path / "results.csv").write_text(
        "metric,value\naccuracy,0.91\n",
        encoding="utf-8",
    )

    captured: List[ExtractedFragment] = []

    async def _fake_synthesize(self, fragments, overrides=None):  # type: ignore[no-untyped-def]
        captured.extend(fragments)
        return PaperMetaData(
            title="T",
            idea_hypothesis="I",
            method="M",
            data="D",
            experiments="E",
        )

    with patch(
        "src.agents.metadata_agent.metadata_generator.synthesizer.MetadataSynthesizer.synthesize",
        new=_fake_synthesize,
    ):
        result = await generate_metadata_from_folder(str(tmp_path))

    assert result.title == "T"
    assert captured
    assert any(f.extra.get("phase") for f in captured)
    assert any(f.extra.get("phase") == "phase_1_guidance" for f in captured)
