"""Tests for exemplar guidance wiring in execute_generation (TDD: RED first)."""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Source-level checks: execute_generation reconstructs and passes exemplar
# ---------------------------------------------------------------------------

def test_execute_generation_reconstructs_exemplar():
    """execute_generation must read exemplar_analysis from plan_result."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "src" / "agents" / "metadata_agent" / "metadata_agent.py").read_text(
        encoding="utf-8",
    )
    exec_gen = text.find("async def execute_generation(")
    assert exec_gen != -1
    body = text[exec_gen:exec_gen + 5000]
    assert "exemplar_analysis" in body
    assert "ExemplarAnalysis" in body or "exemplar_analysis" in body


def test_generate_introduction_accepts_exemplar_guidance():
    """_generate_introduction must accept exemplar_guidance parameter."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "src" / "agents" / "metadata_agent" / "metadata_agent.py").read_text(
        encoding="utf-8",
    )
    intro_def = text.find("def _generate_introduction(")
    assert intro_def != -1
    sig_end = text.find(") -> SectionResult:", intro_def)
    sig = text[intro_def:sig_end]
    assert "exemplar_guidance" in sig


def test_generate_body_section_accepts_exemplar_guidance():
    """_generate_body_section must accept exemplar_guidance parameter."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "src" / "agents" / "metadata_agent" / "metadata_agent.py").read_text(
        encoding="utf-8",
    )
    body_def = text.find("def _generate_body_section(")
    assert body_def != -1
    sig_end = text.find(") -> SectionResult:", body_def)
    sig = text[body_def:sig_end]
    assert "exemplar_guidance" in sig


def test_generate_synthesis_section_accepts_exemplar_guidance():
    """_generate_synthesis_section must accept exemplar_guidance parameter."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "src" / "agents" / "metadata_agent" / "metadata_agent.py").read_text(
        encoding="utf-8",
    )
    synth_def = text.find("def _generate_synthesis_section(")
    assert synth_def != -1
    sig_end = text.find(") -> SectionResult:", synth_def)
    sig = text[synth_def:sig_end]
    assert "exemplar_guidance" in sig


def test_execute_generation_passes_exemplar_to_intro():
    """execute_generation must pass exemplar_guidance to _generate_introduction."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "src" / "agents" / "metadata_agent" / "metadata_agent.py").read_text(
        encoding="utf-8",
    )
    exec_gen = text.find("async def execute_generation(")
    assert exec_gen != -1
    body = text[exec_gen:]
    intro_call = body.find("_generate_introduction(")
    assert intro_call != -1
    call_block = body[intro_call:intro_call + 600]
    assert "exemplar_guidance" in call_block


def test_execute_generation_passes_exemplar_to_body():
    """execute_generation must pass exemplar_guidance to _generate_body_section."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "src" / "agents" / "metadata_agent" / "metadata_agent.py").read_text(
        encoding="utf-8",
    )
    exec_gen = text.find("async def execute_generation(")
    assert exec_gen != -1
    body = text[exec_gen:]
    body_call = body.find("_generate_body_section(")
    assert body_call != -1
    call_block = body[body_call:body_call + 800]
    assert "exemplar_guidance" in call_block


def test_execute_generation_passes_exemplar_to_synthesis():
    """execute_generation must pass exemplar_guidance to _generate_synthesis_section."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "src" / "agents" / "metadata_agent" / "metadata_agent.py").read_text(
        encoding="utf-8",
    )
    exec_gen = text.find("async def execute_generation(")
    assert exec_gen != -1
    body = text[exec_gen:]
    synth_call = body.find("_generate_synthesis_section(")
    assert synth_call != -1
    call_block = body[synth_call:synth_call + 600]
    assert "exemplar_guidance" in call_block
