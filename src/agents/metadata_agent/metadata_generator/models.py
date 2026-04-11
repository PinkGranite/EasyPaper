"""
Data models for the universal metadata generator.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FileCategory(str, Enum):
    """Supported file type categories for research material classification."""
    PDF = "pdf"
    CODE = "code"
    TEXT = "text"
    BIB = "bib"
    DATA = "data"
    IMAGE = "image"
    CONFIG = "config"
    UNKNOWN = "unknown"


class ExtractedFragment(BaseModel):
    """
    A piece of structured information extracted from a single source file.

    - **Args**:
        - `source_file` (str): Relative path of the source file.
        - `file_category` (FileCategory): Category the file was classified as.
        - `content` (str): Extracted textual content or description.
        - `metadata_field` (str, optional): Target PaperMetaData field
          (e.g. "idea_hypothesis", "method", "references").
        - `confidence` (float): Extraction confidence score in [0, 1].
        - `extra` (dict, optional): Arbitrary payload (page numbers, symbols, etc.).
    """
    source_file: str
    file_category: FileCategory
    content: str
    metadata_field: Optional[str] = None
    confidence: float = 0.5
    extra: Dict[str, Any] = Field(default_factory=dict)


class FolderScanResult(BaseModel):
    """
    Result of scanning a research materials folder.

    - **Args**:
        - `folder_path` (str): Absolute path to the scanned folder.
        - `files_by_category` (dict): Mapping from FileCategory to list of
          relative file paths classified under that category.
    """
    folder_path: str
    files_by_category: Dict[FileCategory, List[str]] = Field(default_factory=dict)

    @property
    def total_files(self) -> int:
        return sum(len(v) for v in self.files_by_category.values())

    @property
    def category_counts(self) -> Dict[FileCategory, int]:
        return {cat: len(files) for cat, files in self.files_by_category.items()}
