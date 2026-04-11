"""
FolderScanner: traverse a research materials folder and classify files by type.
"""
from __future__ import annotations

import fnmatch
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .models import FileCategory, FolderScanResult

EXTENSION_MAP: Dict[str, FileCategory] = {
    # PDF
    ".pdf": FileCategory.PDF,
    # Code
    ".py": FileCategory.CODE,
    ".ipynb": FileCategory.CODE,
    ".jl": FileCategory.CODE,
    ".r": FileCategory.CODE,
    ".c": FileCategory.CODE,
    ".cc": FileCategory.CODE,
    ".cpp": FileCategory.CODE,
    ".h": FileCategory.CODE,
    ".hpp": FileCategory.CODE,
    ".java": FileCategory.CODE,
    ".go": FileCategory.CODE,
    ".rs": FileCategory.CODE,
    ".sh": FileCategory.CODE,
    # Text / notes
    ".md": FileCategory.TEXT,
    ".markdown": FileCategory.TEXT,
    ".txt": FileCategory.TEXT,
    ".tex": FileCategory.TEXT,
    ".rst": FileCategory.TEXT,
    # BibTeX
    ".bib": FileCategory.BIB,
    # Data
    ".csv": FileCategory.DATA,
    ".tsv": FileCategory.DATA,
    ".json": FileCategory.DATA,
    ".jsonl": FileCategory.DATA,
    ".xlsx": FileCategory.DATA,
    ".xls": FileCategory.DATA,
    ".parquet": FileCategory.DATA,
    # Image
    ".png": FileCategory.IMAGE,
    ".jpg": FileCategory.IMAGE,
    ".jpeg": FileCategory.IMAGE,
    ".gif": FileCategory.IMAGE,
    ".svg": FileCategory.IMAGE,
    ".webp": FileCategory.IMAGE,
    ".bmp": FileCategory.IMAGE,
    # Config
    ".yaml": FileCategory.CONFIG,
    ".yml": FileCategory.CONFIG,
    ".toml": FileCategory.CONFIG,
    ".ini": FileCategory.CONFIG,
    ".cfg": FileCategory.CONFIG,
}

DEFAULT_EXCLUDE_GLOBS = [
    "**/.git/**",
    "**/node_modules/**",
    "**/venv/**",
    "**/.venv/**",
    "**/__pycache__/**",
    "**/build/**",
    "**/dist/**",
    "**/.idea/**",
    "**/.vscode/**",
]


def _matches_any_glob(rel_path: str, patterns: List[str]) -> bool:
    """
    Check if *rel_path* matches any of the glob *patterns*.
    Supports ``**`` for recursive directory matching.
    """
    normalized = rel_path.replace("\\", "/")
    for pat in patterns:
        if _glob_match(normalized, pat.replace("\\", "/")):
            return True
    return False


def _glob_match(path: str, pattern: str) -> bool:
    """Match a single *path* against a single glob *pattern* with ``**`` support."""
    if "**" not in pattern:
        return fnmatch.fnmatch(path, pattern)

    # Split pattern on "**" segments and match greedily.
    # e.g. "**/*.py" -> ["", "*.py"] meaning: any prefix + *.py at the end
    # e.g. "**/figures/**" -> ["", "figures", ""] meaning: any prefix, literal figures dir, any suffix
    segments = pattern.split("**")
    if pattern.startswith("**/"):
        # "**/*.py" matches "foo.py" or "a/b/foo.py"
        remainder = pattern[3:]  # strip leading "**/""
        if "**" not in remainder:
            # Simple case: match the filename or any sub-path suffix
            return fnmatch.fnmatch(path, remainder) or fnmatch.fnmatch(path, "*/" + remainder) or fnmatch.fnmatch(path, "*/*/" + remainder) or _suffix_match(path, remainder)
        # Nested **: recurse
        parts = path.split("/")
        for i in range(len(parts)):
            sub = "/".join(parts[i:])
            if _glob_match(sub, remainder):
                return True
        return False

    # Pattern like "figures/**" or "a/**/b"
    parts = path.split("/")
    pat_parts = pattern.split("/")
    return _recursive_match(parts, pat_parts)


def _suffix_match(path: str, pattern: str) -> bool:
    """Check if *path* ends with something matching *pattern*."""
    parts = path.split("/")
    pat_parts = pattern.split("/")
    if len(pat_parts) > len(parts):
        return False
    suffix = "/".join(parts[-len(pat_parts):])
    return fnmatch.fnmatch(suffix, pattern)


def _recursive_match(path_parts: List[str], pat_parts: List[str]) -> bool:
    if not pat_parts:
        return not path_parts
    if not path_parts:
        return all(p == "**" for p in pat_parts)

    if pat_parts[0] == "**":
        rest_pat = pat_parts[1:]
        # ** can match zero or more path segments
        for i in range(len(path_parts) + 1):
            if _recursive_match(path_parts[i:], rest_pat):
                return True
        return False

    if fnmatch.fnmatch(path_parts[0], pat_parts[0]):
        return _recursive_match(path_parts[1:], pat_parts[1:])

    return False


class FolderScanner:
    """
    Scan a folder of research materials and classify every file by type.

    - **Args**:
        - `include_globs` (list, optional): If provided, only files matching
          at least one pattern are included.
        - `exclude_globs` (list, optional): Additional patterns to exclude
          (merged with built-in defaults).
    """

    def __init__(
        self,
        include_globs: Optional[List[str]] = None,
        exclude_globs: Optional[List[str]] = None,
    ) -> None:
        self._include_globs = include_globs or []
        self._exclude_globs = list(DEFAULT_EXCLUDE_GLOBS)
        if exclude_globs:
            self._exclude_globs.extend(exclude_globs)

    def scan(self, folder_path: str) -> FolderScanResult:
        """
        Walk *folder_path* recursively and classify every file.

        - **Args**:
            - `folder_path` (str): Absolute or relative path to the folder.

        - **Returns**:
            - `FolderScanResult`: Files grouped by category.

        - **Raises**:
            - `FileNotFoundError`: If the folder does not exist.
        """
        root = Path(folder_path).resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        by_category: Dict[FileCategory, List[str]] = defaultdict(list)

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()

            if _matches_any_glob(rel, self._exclude_globs):
                continue
            if self._include_globs and not _matches_any_glob(rel, self._include_globs):
                continue

            category = self._classify(path)
            by_category[category].append(rel)

        return FolderScanResult(
            folder_path=str(root),
            files_by_category=dict(by_category),
        )

    @staticmethod
    def _classify(path: Path) -> FileCategory:
        ext = path.suffix.lower()
        return EXTENSION_MAP.get(ext, FileCategory.UNKNOWN)
