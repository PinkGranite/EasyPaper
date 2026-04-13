"""
Cross-section label registry for LaTeX reference validation.
- **Description**:
    - Collects all \\label{} definitions across generated sections.
    - Validates \\ref{} / \\cref{} commands against defined labels.
    - Removes undefined references to prevent ?? in compiled PDF.
"""
import re
from typing import Dict, Set


def collect_all_labels(sections: Dict[str, str]) -> Set[str]:
    """
    Scan all section contents and collect every \\label{key} definition.
    - **Args**:
        - `sections` (Dict[str, str]): section_type -> LaTeX content

    - **Returns**:
        - `Set[str]`: Set of all defined label keys.
    """
    labels: set[str] = set()
    for content in sections.values():
        for m in re.finditer(r'\\label\{([^}]+)\}', content):
            labels.add(m.group(1))
    return labels


def validate_and_fix_refs(content: str, valid_labels: Set[str]) -> str:
    """
    Remove or clean undefined \\ref{} and \\cref{} from content.
    - **Description**:
        - For each \\ref{key} or \\cref{key}, if key is not in valid_labels,
          removes the reference pattern including the preceding qualifier
          (e.g. "Figure~\\ref{...}" becomes "").
        - Preserves valid references unchanged.

    - **Args**:
        - `content` (str): LaTeX content with references.
        - `valid_labels` (Set[str]): Set of defined label keys.

    - **Returns**:
        - `str`: Content with undefined references removed.
    """
    def _replace_ref(match: re.Match) -> str:
        key = match.group("key")
        if key in valid_labels:
            return match.group(0)
        return ""

    # Match patterns like "Figure~\ref{...}", "Table~\ref{...}", standalone \ref{...}, \cref{...}
    # The optional prefix captures "Figure~", "Table~", "Fig.~", "Tab.~", etc.
    pattern = (
        r'(?:(?:Figure|Table|Fig\.|Tab\.|Figures|Tables|Section|Sec\.)'
        r'[~ \t]*)?'
        r'\\(?:c?ref)\{(?P<key>[^}]+)\}'
    )

    result = re.sub(pattern, _replace_ref, content)

    # Clean up inline spacing artifacts while preserving paragraph breaks.
    result = re.sub(r',[ \t]*,', ',', result)
    result = re.sub(r'[ \t]{2,}', ' ', result)
    result = re.sub(r'[ \t]+([.,;])', r'\1', result)
    result = re.sub(r'[ \t]+\n', '\n', result)
    result = re.sub(r'\n[ \t]+', '\n', result)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result
