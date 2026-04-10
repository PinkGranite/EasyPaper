"""Tests for TeX PATH bootstrap helpers."""
from __future__ import annotations

from pathlib import Path

from src.agents.shared.tex_path_bootstrap import _candidate_tex_bin_dirs


def test_easypaper_tex_bin_is_first_candidate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EASYPAPER_TEX_BIN", str(tmp_path))
    dirs = _candidate_tex_bin_dirs()
    assert dirs[0] == tmp_path
