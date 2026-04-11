"""
Integration: table conversion via the same path as paper generation.

``EasyPaper.generate`` / ``MetaDataAgent.generate_paper`` first calls
``prepare_plan()``, which ends with Phase 0.5 ``convert_tables()`` using
``metadata.tables`` and ``self.client`` — identical wiring to production.

These tests load the BLIP-2 ai_track ``meta.json``, normalize paths so CI
does not require figure/table files on disk, patch ``ReferencePool.create``
to avoid network reference resolution, and mock only the LLM used by
``convert_tables``.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.metadata_agent.metadata_agent import MetaDataAgent
from src.agents.metadata_agent.models import PaperMetaData
from src.agents.shared.reference_pool import ReferencePool
from src.agents.shared.table_converter import _build_table_latex_from_source
from src.config.schema import ModelConfig, ToolsConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
BLIP_META_JSON = (
    REPO_ROOT
    / "experiments"
    / "ai_track"
    / "metadatas"
    / "3f5b31c4f7350dc88002c121aecbdc82f86eb5bb"
    / "meta.json"
)


def _load_blip_metadata_for_prepare_plan() -> PaperMetaData:
    if not BLIP_META_JSON.is_file():
        pytest.skip(f"Fixture meta.json not found: {BLIP_META_JSON}")
    data = json.loads(BLIP_META_JSON.read_text(encoding="utf-8"))
    data["figures"] = []
    for t in data.get("tables") or []:
        t["file_path"] = None
    return PaperMetaData.model_validate(data)


def _openai_completion_with_latex(latex: str) -> MagicMock:
    msg = MagicMock()
    msg.content = latex
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.mark.integration
async def test_prepare_plan_runs_phase05_convert_tables_like_generate_paper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    MetaDataAgent.prepare_plan must populate ``converted_tables`` using the
    same ``convert_tables`` call chain as ``generate_paper`` (Phase 0.5).

    - **Description**:
        - Skips reference search (no SerpAPI / Semantic Scholar).
        - Skips planning and Docling to keep the test local and fast.
        - Mocks only ``LLMClient`` chat completions used by table conversion.

    - **Args**:
        - ``monkeypatch`` (pytest.MonkeyPatch): Pytest monkeypatch fixture.
        - ``tmp_path`` (Path): Temporary output directory for ``paper_dir``.

    - **Returns**:
        - ``None``
    """
    metadata = _load_blip_metadata_for_prepare_plan()
    assert metadata.tables

    async def _instant_ref_pool(
        cls: type[ReferencePool],
        initial_refs: list,
        paper_search_config: object = None,
    ) -> ReferencePool:
        return cls(initial_refs)

    monkeypatch.setattr(
        ReferencePool,
        "create",
        classmethod(_instant_ref_pool),
    )

    latex_sequence = [
        _build_table_latex_from_source(
            table_id=t.id,
            caption=t.caption or t.id,
            content=t.content or "",
        )
        for t in metadata.tables
    ]
    side_effect = [_openai_completion_with_latex(tex) for tex in latex_sequence]

    model = ModelConfig(
        model_name="stub-model",
        api_key="stub-key",
        base_url="http://127.0.0.1:9",
    )
    agent = MetaDataAgent(
        model,
        tools_config=ToolsConfig(enabled=False, available_tools=[]),
    )
    # LLMClient wraps AsyncOpenAI; _CompletionsProxy.create is read-only — patch inner API.
    agent.client._client.chat.completions.create = AsyncMock(side_effect=side_effect)

    out_dir = tmp_path / "paper_gen_flow"
    out_dir.mkdir(parents=True, exist_ok=True)

    tpl = metadata.template_path
    if tpl:
        abs_tpl = (REPO_ROOT / tpl).resolve()
        tpl_arg = str(abs_tpl) if abs_tpl.is_file() else None
    else:
        tpl_arg = None

    plan = await agent.prepare_plan(
        metadata=metadata,
        template_path=tpl_arg,
        target_pages=metadata.target_pages,
        enable_planning=False,
        enable_exemplar=False,
        save_output=True,
        output_dir=str(out_dir),
        artifacts_prefix="integration_test",
    )

    assert not plan.errors, plan.errors
    assert plan.converted_tables
    assert len(plan.converted_tables) == len(metadata.tables)
    for t in metadata.tables:
        assert t.id in plan.converted_tables
        body = plan.converted_tables[t.id]
        assert rf"\label{{{t.id}}}" in body
        assert "\\begin{table" in body


@pytest.mark.integration
async def test_prepare_plan_table_conversion_uses_paper_dir_parent_as_base_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    ``prepare_plan`` must pass ``base_path=str(paper_dir.parent)`` into
    ``convert_tables`` when ``save_output`` is True (production behavior).

    - **Description**:
        - Intercepts ``convert_tables`` and records keyword arguments.

    - **Args**:
        - ``monkeypatch`` (pytest.MonkeyPatch): Pytest monkeypatch fixture.
        - ``tmp_path`` (Path): Temporary output directory.

    - **Returns**:
        - ``None``
    """
    metadata = _load_blip_metadata_for_prepare_plan()
    captured: dict = {}

    async def _spy_convert_tables(*args: object, **kwargs: object) -> dict:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        "src.agents.metadata_agent.metadata_agent.convert_tables",
        _spy_convert_tables,
    )

    async def _instant_ref_pool(
        cls: type[ReferencePool],
        initial_refs: list,
        paper_search_config: object = None,
    ) -> ReferencePool:
        return cls(initial_refs)

    monkeypatch.setattr(
        ReferencePool,
        "create",
        classmethod(_instant_ref_pool),
    )

    model = ModelConfig(
        model_name="stub-model",
        api_key="stub-key",
        base_url="http://127.0.0.1:9",
    )
    agent = MetaDataAgent(
        model,
        tools_config=ToolsConfig(enabled=False, available_tools=[]),
    )

    out_dir = tmp_path / "paper_base_path_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    await agent.prepare_plan(
        metadata=metadata,
        template_path=None,
        enable_planning=False,
        save_output=True,
        output_dir=str(out_dir),
    )

    assert captured.get("base_path") == str(out_dir.parent)
