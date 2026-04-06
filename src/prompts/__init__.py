"""
Prompt template management for EasyPaper.
- **Description**:
    - Loads prompt templates from external .txt files
    - Falls back to inline defaults when files are missing
    - Supports versioning for A/B testing
    - Caches loaded templates to avoid repeated IO
"""
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent


class PromptLoader:
    """
    Loads and caches prompt templates from the prompts/ directory.
    - **Description**:
        - Templates use Python str.format() placeholders (e.g. {section_type})
        - Falls back to provided default when the template file is missing
        - Thread-safe via simple dict caching (no mutation during reads)

    - **Args**:
        - `prompts_dir` (Path, optional): Root directory for prompt files.
        - `version` (str): Version subdirectory for A/B testing.
    """

    def __init__(self, prompts_dir: Optional[Path] = None, version: str = "v1") -> None:
        self._dir = prompts_dir or _PROMPTS_DIR
        self._version = version
        self._cache: Dict[str, str] = {}

    def load(
        self,
        category: str,
        name: str,
        default: str = "",
        **format_kwargs,
    ) -> str:
        """
        Load a prompt template, optionally formatting it.
        - **Description**:
            - Tries {prompts_dir}/{category}/{name}.txt first
            - Falls back to inline default if file not found
            - Caches raw template text after first load
            - Applies str.format(**format_kwargs) if kwargs provided

        - **Args**:
            - `category` (str): Subdirectory (e.g. "metadata", "writer")
            - `name` (str): Template name without extension
            - `default` (str): Fallback text if file not found
            - `**format_kwargs`: Placeholder values for str.format()

        - **Returns**:
            - `str`: The (optionally formatted) prompt text
        """
        cache_key = f"{category}/{name}"
        if cache_key not in self._cache:
            file_path = self._dir / category / f"{name}.txt"
            if file_path.is_file():
                try:
                    self._cache[cache_key] = file_path.read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning("Failed to load prompt %s: %s", file_path, e)
                    self._cache[cache_key] = default
            else:
                self._cache[cache_key] = default

        template = self._cache[cache_key]
        if format_kwargs:
            try:
                return template.format(**format_kwargs)
            except KeyError:
                return template
        return template

    def load_section_prompt(self, section_type: str, default: str = "") -> str:
        """
        Load a section-specific writing instruction.
        - **Description**:
            - Convenience wrapper for load("sections", section_type)

        - **Args**:
            - `section_type` (str): e.g. "introduction", "method"
            - `default` (str): Fallback text

        - **Returns**:
            - `str`: The section prompt text
        """
        return self.load("sections", section_type, default=default)

    def clear_cache(self) -> None:
        """Clear the template cache to force re-reading from disk."""
        self._cache.clear()
