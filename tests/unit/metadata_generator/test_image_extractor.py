"""Tests for ImageExtractor."""
import pytest
from pathlib import Path

from src.agents.metadata_agent.metadata_generator.models import FileCategory, ExtractedFragment
from src.agents.metadata_agent.metadata_generator.extractors.image_extractor import ImageExtractor


@pytest.fixture
def image_folder(tmp_path: Path) -> Path:
    (tmp_path / "fig1.png").write_bytes(b"\x89PNG" + b"\x00" * 100)
    (tmp_path / "architecture.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
    (tmp_path / "results_plot.pdf").write_bytes(b"%PDF" + b"\x00" * 50)
    return tmp_path


class TestImageExtractor:
    def test_discover_images(self, image_folder: Path):
        ext = ImageExtractor()
        fragments = ext.extract_from_folder(str(image_folder))
        assert len(fragments) >= 2
        assert all(f.file_category == FileCategory.IMAGE for f in fragments)
        assert all(f.metadata_field == "figures" for f in fragments)

    def test_auto_generate_id(self, image_folder: Path):
        ext = ImageExtractor()
        fragments = ext.extract_from_folder(str(image_folder))
        ids = [f.extra.get("figure_id") for f in fragments]
        assert all(fid is not None and fid.startswith("fig:") for fid in ids)

    def test_auto_generate_caption(self, image_folder: Path):
        ext = ImageExtractor()
        fragments = ext.extract_from_folder(str(image_folder))
        for f in fragments:
            assert f.extra.get("caption", "") != ""

    def test_file_path_in_extra(self, image_folder: Path):
        ext = ImageExtractor()
        fragments = ext.extract_from_folder(str(image_folder))
        for f in fragments:
            assert "file_path" in f.extra

    def test_empty_folder(self, tmp_path: Path):
        ext = ImageExtractor()
        fragments = ext.extract_from_folder(str(tmp_path))
        assert fragments == []
