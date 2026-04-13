"""
Base class for all material extractors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models import ExtractedFragment


class BaseExtractor(ABC):
    """
    Abstract base for extracting structured fragments from research material files.

    - **Args**: (none at base level)

    Subclasses implement ``extract()`` to handle their specific file type.
    """

    @abstractmethod
    def extract(self, file_path: str, *, materials_root: Optional[str] = None) -> List[ExtractedFragment]:
        """
        Extract fragments from a single file.

        - **Args**:
            - `file_path` (str): Absolute or relative path to the file.
            - `materials_root` (str, optional): Root folder for resolving relative
              paths in ``source_file`` / ``extra.file_path`` (CSV/TSV).

        - **Returns**:
            - `List[ExtractedFragment]`: Extracted fragments (may be empty).
        """
        ...
