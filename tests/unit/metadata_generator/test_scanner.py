"""
Tests for FolderScanner.
RED phase: defines folder scanning and file classification contract.
"""
import pytest
from pathlib import Path

from src.agents.metadata_agent.metadata_generator.models import FileCategory, FolderScanResult
from src.agents.metadata_agent.metadata_generator.scanner import FolderScanner


@pytest.fixture
def sample_folder(tmp_path: Path) -> Path:
    """Create a realistic research materials folder."""
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4 fake pdf content")
    (tmp_path / "model.py").write_text("class MyModel:\n    pass\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# Research Notes\nSome ideas.\n", encoding="utf-8")
    (tmp_path / "refs.bib").write_text(
        "@article{test2024,\n  title={Test},\n  author={A},\n  year={2024}\n}\n",
        encoding="utf-8",
    )
    (tmp_path / "results.csv").write_text("model,accuracy\nours,0.95\nbaseline,0.88\n", encoding="utf-8")
    (tmp_path / "fig1.png").write_bytes(b"\x89PNG fake image")
    (tmp_path / "config.yaml").write_text("learning_rate: 0.001\n", encoding="utf-8")
    (tmp_path / "random.xyz").write_text("unknown format", encoding="utf-8")
    return tmp_path


@pytest.fixture
def nested_folder(tmp_path: Path) -> Path:
    """Create a folder with subdirectories."""
    sub = tmp_path / "code" / "src"
    sub.mkdir(parents=True)
    (sub / "train.py").write_text("def train(): pass\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "design.md").write_text("# Design\n", encoding="utf-8")
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "arch.png").write_bytes(b"\x89PNG")
    (tmp_path / "figures" / "arch.jpg").write_bytes(b"\xff\xd8\xff")
    return tmp_path


class TestFolderScanner:
    def test_scan_classifies_all_types(self, sample_folder: Path):
        scanner = FolderScanner()
        result = scanner.scan(str(sample_folder))

        assert isinstance(result, FolderScanResult)
        assert result.folder_path == str(sample_folder)

        cats = result.files_by_category
        assert any("paper.pdf" in f for f in cats.get(FileCategory.PDF, []))
        assert any("model.py" in f for f in cats.get(FileCategory.CODE, []))
        assert any("notes.md" in f for f in cats.get(FileCategory.TEXT, []))
        assert any("refs.bib" in f for f in cats.get(FileCategory.BIB, []))
        assert any("results.csv" in f for f in cats.get(FileCategory.DATA, []))
        assert any("fig1.png" in f for f in cats.get(FileCategory.IMAGE, []))
        assert any("config.yaml" in f for f in cats.get(FileCategory.CONFIG, []))

    def test_scan_total_files(self, sample_folder: Path):
        scanner = FolderScanner()
        result = scanner.scan(str(sample_folder))
        assert result.total_files == 8  # 7 known + 1 unknown

    def test_scan_empty_folder(self, tmp_path: Path):
        scanner = FolderScanner()
        result = scanner.scan(str(tmp_path))
        assert result.total_files == 0
        assert result.files_by_category == {}

    def test_scan_nested_directories(self, nested_folder: Path):
        scanner = FolderScanner()
        result = scanner.scan(str(nested_folder))
        assert result.total_files == 4
        code_files = result.files_by_category.get(FileCategory.CODE, [])
        assert any("train.py" in f for f in code_files)

    def test_scan_with_include_globs(self, sample_folder: Path):
        scanner = FolderScanner(include_globs=["**/*.py", "**/*.md"])
        result = scanner.scan(str(sample_folder))
        all_files = []
        for files in result.files_by_category.values():
            all_files.extend(files)
        assert any("model.py" in f for f in all_files)
        assert any("notes.md" in f for f in all_files)
        assert not any("paper.pdf" in f for f in all_files)

    def test_scan_with_exclude_globs(self, nested_folder: Path):
        scanner = FolderScanner(exclude_globs=["**/figures/**"])
        result = scanner.scan(str(nested_folder))
        all_files = []
        for files in result.files_by_category.values():
            all_files.extend(files)
        assert not any("arch.png" in f for f in all_files)

    def test_scan_nonexistent_folder_raises(self):
        scanner = FolderScanner()
        with pytest.raises(FileNotFoundError):
            scanner.scan("/nonexistent/folder/path")

    def test_unknown_extensions_classified(self, sample_folder: Path):
        scanner = FolderScanner()
        result = scanner.scan(str(sample_folder))
        unknown = result.files_by_category.get(FileCategory.UNKNOWN, [])
        assert any("random.xyz" in f for f in unknown)
