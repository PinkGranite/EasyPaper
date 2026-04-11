"""
Image file extractor: discover image files and produce FigureSpec-compatible fragments.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

from ..models import ExtractedFragment, FileCategory
from .base import BaseExtractor

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"}


class ImageExtractor(BaseExtractor):
    """
    Discover image files in a folder and produce one ExtractedFragment per image,
    with auto-generated figure ids and placeholder captions.
    """

    def extract(self, file_path: str) -> List[ExtractedFragment]:
        """Extract from a single known image file."""
        return self._make_fragment(Path(file_path))

    def extract_from_folder(self, folder_path: str) -> List[ExtractedFragment]:
        """
        Scan *folder_path* recursively for image files.

        - **Args**:
            - `folder_path` (str): Root directory to scan.

        - **Returns**:
            - `List[ExtractedFragment]`: One fragment per discovered image.
        """
        root = Path(folder_path)
        if not root.is_dir():
            return []

        fragments: List[ExtractedFragment] = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                fragments.extend(self._make_fragment(path, root))
        return fragments

    @staticmethod
    def _make_fragment(
        path: Path,
        root: Path | None = None,
    ) -> List[ExtractedFragment]:
        rel = path.relative_to(root).as_posix() if root else path.name
        stem = path.stem
        fig_id = "fig:" + re.sub(r"[^a-z0-9_]", "_", stem.lower())
        caption = stem.replace("_", " ").replace("-", " ").title()

        return [
            ExtractedFragment(
                source_file=rel,
                file_category=FileCategory.IMAGE,
                content=f"Image file: {rel}",
                metadata_field="figures",
                confidence=0.7,
                extra={
                    "figure_id": fig_id,
                    "caption": caption,
                    "file_path": rel,
                },
            )
        ]
