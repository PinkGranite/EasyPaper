"""
Tests for metadata_generator data models.
RED phase: these tests define the contract before implementation.
"""
import pytest
from src.agents.metadata_agent.metadata_generator.models import (
    FileCategory,
    ExtractedFragment,
    FolderScanResult,
)


class TestFileCategory:
    def test_all_categories_exist(self):
        expected = {"PDF", "CODE", "TEXT", "BIB", "DATA", "IMAGE", "CONFIG", "UNKNOWN"}
        actual = {c.name for c in FileCategory}
        assert actual == expected

    def test_category_values_are_strings(self):
        for cat in FileCategory:
            assert isinstance(cat.value, str)


class TestExtractedFragment:
    def test_create_minimal(self):
        frag = ExtractedFragment(
            source_file="notes.md",
            file_category=FileCategory.TEXT,
            content="Some research notes",
        )
        assert frag.source_file == "notes.md"
        assert frag.file_category == FileCategory.TEXT
        assert frag.content == "Some research notes"
        assert frag.metadata_field is None
        assert frag.confidence == 0.5

    def test_create_with_all_fields(self):
        frag = ExtractedFragment(
            source_file="paper.pdf",
            file_category=FileCategory.PDF,
            content="We propose a novel method...",
            metadata_field="method",
            confidence=0.9,
            extra={"page": 3},
        )
        assert frag.metadata_field == "method"
        assert frag.confidence == 0.9
        assert frag.extra == {"page": 3}

    def test_valid_metadata_fields(self):
        valid_fields = [
            "title", "idea_hypothesis", "method", "data",
            "experiments", "references", "figures", "tables",
        ]
        for field_name in valid_fields:
            frag = ExtractedFragment(
                source_file="x.txt",
                file_category=FileCategory.TEXT,
                content="content",
                metadata_field=field_name,
            )
            assert frag.metadata_field == field_name

    def test_serialization_roundtrip(self):
        frag = ExtractedFragment(
            source_file="data.csv",
            file_category=FileCategory.DATA,
            content="col1,col2",
            metadata_field="data",
            confidence=0.8,
        )
        data = frag.model_dump()
        restored = ExtractedFragment(**data)
        assert restored == frag


class TestFolderScanResult:
    def test_create_empty(self):
        result = FolderScanResult(folder_path="/tmp/empty", files_by_category={})
        assert result.folder_path == "/tmp/empty"
        assert result.total_files == 0
        assert result.files_by_category == {}

    def test_total_files_computed(self):
        result = FolderScanResult(
            folder_path="/tmp/test",
            files_by_category={
                FileCategory.PDF: ["a.pdf", "b.pdf"],
                FileCategory.CODE: ["main.py"],
                FileCategory.IMAGE: ["fig1.png", "fig2.png", "fig3.png"],
            },
        )
        assert result.total_files == 6

    def test_category_counts(self):
        result = FolderScanResult(
            folder_path="/tmp/test",
            files_by_category={
                FileCategory.PDF: ["a.pdf"],
                FileCategory.BIB: ["refs.bib"],
            },
        )
        counts = result.category_counts
        assert counts[FileCategory.PDF] == 1
        assert counts[FileCategory.BIB] == 1
