"""Tests for DoclingConfig and its integration into ToolsConfig."""
from __future__ import annotations

from src.config.schema import DoclingConfig, ToolsConfig


class TestDoclingConfig:
    """Tests for DoclingConfig model."""

    def test_docling_config_defaults(self):
        cfg = DoclingConfig()
        assert cfg.enabled is False
        assert cfg.device == "auto"
        assert cfg.do_ocr is False
        assert cfg.do_table_structure is True
        assert cfg.do_formula_enrichment is False
        assert cfg.images_scale == 2.0
        assert cfg.document_timeout == 120.0
        assert cfg.max_pages == 30
        assert cfg.download_timeout == 30.0
        assert cfg.cleanup_after_analysis is True
        assert cfg.move_to_output is False

    def test_docling_config_custom_values(self):
        cfg = DoclingConfig(
            enabled=True,
            device="cuda",
            do_ocr=True,
            max_pages=50,
            cleanup_after_analysis=False,
            move_to_output=True,
        )
        assert cfg.enabled is True
        assert cfg.device == "cuda"
        assert cfg.do_ocr is True
        assert cfg.max_pages == 50
        assert cfg.cleanup_after_analysis is False
        assert cfg.move_to_output is True

    def test_tools_config_includes_docling(self):
        tc = ToolsConfig(
            docling=DoclingConfig(enabled=True, device="mps"),
        )
        assert tc.docling is not None
        assert tc.docling.enabled is True
        assert tc.docling.device == "mps"

    def test_tools_config_docling_none_by_default(self):
        tc = ToolsConfig()
        assert tc.docling is None
