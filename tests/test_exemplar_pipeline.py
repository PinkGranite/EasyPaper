"""Tests for Phase 0-exemplar in prepare_plan pipeline (TDD: RED first)."""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Source-level ordering: exemplar phase between docling and core analysis
# ---------------------------------------------------------------------------

def test_prepare_plan_runs_exemplar_after_docling():
    """Phase 0-exemplar must appear after Phase 0-docling in source."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "src" / "agents" / "metadata_agent" / "metadata_agent.py").read_text(
        encoding="utf-8",
    )
    docling = text.find("Phase 0-docling:")
    exemplar = text.find("Phase 0-exemplar:")
    assert docling != -1, "Phase 0-docling marker not found"
    assert exemplar != -1, "Phase 0-exemplar marker not found"
    assert docling < exemplar


def test_prepare_plan_runs_exemplar_before_core_analysis():
    """Phase 0-exemplar must appear before Phase 0-core in source."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "src" / "agents" / "metadata_agent" / "metadata_agent.py").read_text(
        encoding="utf-8",
    )
    exemplar = text.find("Phase 0-exemplar:")
    core = text.find("Phase 0-core:")
    assert exemplar != -1, "Phase 0-exemplar marker not found"
    assert core != -1, "Phase 0-core marker not found"
    assert exemplar < core


def test_prepare_plan_has_enable_exemplar_parameter():
    """prepare_plan() signature must include enable_exemplar."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "src" / "agents" / "metadata_agent" / "metadata_agent.py").read_text(
        encoding="utf-8",
    )
    sig_start = text.find("async def prepare_plan(")
    assert sig_start != -1
    sig_end = text.find(") -> PlanResult:", sig_start)
    sig = text[sig_start:sig_end]
    assert "enable_exemplar" in sig


def test_prepare_plan_stores_exemplar_in_plan_result():
    """PlanResult construction must include exemplar_analysis field."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "src" / "agents" / "metadata_agent" / "metadata_agent.py").read_text(
        encoding="utf-8",
    )
    # Find the main success PlanResult return (the one after table conversion)
    table_conv = text.find("Phase 0.5: Convert tables")
    assert table_conv != -1
    after_tables = text[table_conv:]
    return_block = after_tables.find("return PlanResult(")
    assert return_block != -1
    next_chunk = after_tables[return_block:return_block + 800]
    assert "exemplar_analysis" in next_chunk
