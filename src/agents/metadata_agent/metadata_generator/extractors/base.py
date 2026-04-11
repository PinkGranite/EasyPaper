"""
Base class for all material extractors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..models import ExtractedFragment


class BaseExtractor(ABC):
    """
    Abstract base for extracting structured fragments from research material files.

    - **Args**: (none at base level)

    Subclasses implement ``extract()`` to handle their specific file type.
    """

    @abstractmethod
    def extract(self, file_path: str) -> List[ExtractedFragment]:
        """
        Extract fragments from a single file.

        - **Args**:
            - `file_path` (str): Absolute or relative path to the file.

        - **Returns**:
            - `List[ExtractedFragment]`: Extracted fragments (may be empty).
        """
        ...
