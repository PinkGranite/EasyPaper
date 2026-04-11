"""Tests for DataExtractor."""
import pytest
from pathlib import Path

from src.agents.metadata_agent.metadata_generator.models import FileCategory, ExtractedFragment
from src.agents.metadata_agent.metadata_generator.extractors.data_extractor import DataExtractor


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    content = "model,accuracy,f1\nours,0.95,0.93\nbaseline,0.88,0.85\nablation,0.91,0.89\n"
    f = tmp_path / "results.csv"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def json_data_file(tmp_path: Path) -> Path:
    import json
    data = {"dataset": "CIFAR-10", "samples": 50000, "classes": 10}
    f = tmp_path / "dataset_info.json"
    f.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return f


class TestDataExtractor:
    def test_csv_extraction(self, csv_file: Path):
        ext = DataExtractor()
        fragments = ext.extract(str(csv_file))
        assert len(fragments) >= 1
        frag = fragments[0]
        assert frag.file_category == FileCategory.DATA
        assert "model" in frag.content
        assert "accuracy" in frag.content

    def test_csv_table_spec_extra(self, csv_file: Path):
        ext = DataExtractor()
        fragments = ext.extract(str(csv_file))
        frag = fragments[0]
        assert "table_id" in frag.extra
        assert "caption" in frag.extra
        assert frag.extra["table_id"].startswith("tab:")

    def test_json_extraction(self, json_data_file: Path):
        ext = DataExtractor()
        fragments = ext.extract(str(json_data_file))
        assert len(fragments) >= 1
        frag = fragments[0]
        assert frag.file_category == FileCategory.DATA
        assert "CIFAR-10" in frag.content or "dataset" in frag.content

    def test_empty_csv(self, tmp_path: Path):
        f = tmp_path / "empty.csv"
        f.write_text("", encoding="utf-8")
        ext = DataExtractor()
        fragments = ext.extract(str(f))
        assert fragments == []

    def test_metadata_field(self, csv_file: Path):
        ext = DataExtractor()
        fragments = ext.extract(str(csv_file))
        for frag in fragments:
            assert frag.metadata_field in ("data", "experiments", "tables")
