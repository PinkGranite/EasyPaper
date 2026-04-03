"""
Pure helpers for citation assignment (no LLM / heavy agent imports).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def claim_matrix_refs_for_section(
    research_context: Optional[Dict[str, Any]],
    section_type: str,
) -> List[str]:
    """
    Collect citation keys from claim_evidence_matrix rows for this section.

    - **Description**:
        - Includes rows where ``section_type`` matches the section, or is
          ``global`` / empty / None.
    """
    out: List[str] = []
    matrix = (research_context or {}).get("claim_evidence_matrix") or []
    if not isinstance(matrix, list):
        return out
    for row in matrix:
        if not isinstance(row, dict):
            continue
        st = row.get("section_type")
        if st not in (section_type, "global", "", None):
            continue
        refs_raw = row.get("support_refs") or []
        if not isinstance(refs_raw, list):
            continue
        for r in refs_raw:
            if r and r not in out:
                out.append(str(r))
    return out
