"""Lightweight checks that prepare_plan pipeline ordering is documented in source."""
from __future__ import annotations

from pathlib import Path


def test_prepare_plan_runs_core_context_before_create_plan():
    root = Path(__file__).resolve().parents[1]
    text = (root / "src" / "agents" / "metadata_agent" / "metadata_agent.py").read_text(
        encoding="utf-8",
    )
    core = text.find("Phase 0-core:")
    ctx = text.find("Phase 0-ctx:")
    plan_call = text.find("_create_paper_plan(")
    assert core != -1 and ctx != -1 and plan_call != -1
    assert core < ctx < plan_call


def test_prepare_plan_runs_docling_before_core_analysis():
    """Docling enrichment must run before CoreRefAnalyzer."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "src" / "agents" / "metadata_agent" / "metadata_agent.py").read_text(
        encoding="utf-8",
    )
    docling = text.find("Phase 0-docling:")
    core = text.find("Phase 0-core:")
    assert docling != -1 and core != -1
    assert docling < core
