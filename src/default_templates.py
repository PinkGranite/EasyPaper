"""
Built-in LaTeX template resolver.
- **Description**:
    - Maps style_guide names to bundled template .zip files.
    - Templates are stored in the project-level ``templates/`` directory
      and ship with the Docker image, so they work in all environments.

- **Usage**::

    from src.default_templates import resolve_default_template
    path = resolve_default_template("Nature")  # -> "/app/templates/nature.zip" or None
"""
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger("uvicorn.error")

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

_STYLE_TO_TEMPLATE: dict[str, str] = {
    "nature": "nature.zip",
    "science": "nature.zip",
}


def resolve_default_template(style_guide: Optional[str]) -> Optional[str]:
    """
    Resolve a style_guide string to a built-in template path.
    - **Args**:
        - `style_guide` (str | None): Venue or style name (case-insensitive).

    - **Returns**:
        - `str | None`: Absolute path to the .zip template, or None if no match.
    """
    if not style_guide:
        return None

    key = style_guide.strip().lower()
    filename = _STYLE_TO_TEMPLATE.get(key)
    if not filename:
        return None

    path = _TEMPLATES_DIR / filename
    if path.is_file():
        logger.info("default_templates.resolved style_guide=%s -> %s", style_guide, path)
        return str(path)

    logger.warning("default_templates.missing style_guide=%s expected=%s", style_guide, path)
    return None


def list_available_templates() -> list[str]:
    """
    List all style_guide names that have a built-in template.
    - **Returns**:
        - `list[str]`: Available style guide names.
    """
    available = []
    for key, filename in _STYLE_TO_TEMPLATE.items():
        if (_TEMPLATES_DIR / filename).is_file():
            available.append(key)
    return available
