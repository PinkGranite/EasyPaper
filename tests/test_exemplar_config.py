"""Tests for ExemplarConfig (TDD: RED first)."""
from __future__ import annotations

import pytest

from src.config.schema import ExemplarConfig, ToolsConfig


class TestExemplarConfig:
    def test_default_disabled(self):
        cfg = ExemplarConfig()
        assert cfg.enabled is False

    def test_defaults(self):
        cfg = ExemplarConfig()
        assert cfg.prefer_core_refs is True
        assert cfg.max_external_candidates == 10
        assert cfg.max_analysis_chars == 8000
        assert cfg.venue_match_required is True
        assert cfg.recency_years == 5

    def test_enabled_override(self):
        cfg = ExemplarConfig(enabled=True, recency_years=3)
        assert cfg.enabled is True
        assert cfg.recency_years == 3

    def test_serialization_roundtrip(self):
        cfg = ExemplarConfig(enabled=True, max_external_candidates=5)
        data = cfg.model_dump()
        restored = ExemplarConfig(**data)
        assert restored == cfg


class TestToolsConfigExemplarField:
    def test_exemplar_default_none(self):
        tc = ToolsConfig()
        assert tc.exemplar is None

    def test_exemplar_set(self):
        tc = ToolsConfig(exemplar=ExemplarConfig(enabled=True))
        assert tc.exemplar is not None
        assert tc.exemplar.enabled is True

    def test_exemplar_from_dict(self):
        tc = ToolsConfig(exemplar={"enabled": True, "recency_years": 3})
        assert tc.exemplar.enabled is True
        assert tc.exemplar.recency_years == 3
