"""
Template-Slot Filling for Degraded Generation
- **Description**:
    - Defines slot types and template structures for constrained paragraph
      generation when decomposed free-form generation fails repeatedly.
    - The Planner produces a skeleton with typed placeholders (e.g.
      ``[METRIC_SLOT:s1]``); the Writer fills each slot with evidence-bound
      content.
    - ``parse_template_slots`` extracts slot markers from a skeleton string.
    - ``render_filled_template`` substitutes filled content back in.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Slot taxonomy
# ---------------------------------------------------------------------------

class SlotType(str, Enum):
    """Types of evidence slots that can appear in a paragraph template."""
    METRIC = "METRIC_SLOT"
    FIGURE_REF = "FIGURE_SLOT"
    TABLE_REF = "TABLE_SLOT"
    CITATION = "CITATION_SLOT"
    EVIDENCE = "EVIDENCE_SLOT"
    TRANSITION = "TRANSITION_SLOT"


# ---------------------------------------------------------------------------
# Slot and template models
# ---------------------------------------------------------------------------

class TemplateSlot(BaseModel):
    """
    A single fillable slot inside a paragraph template.
    - **Fields**:
        - ``slot_type``: Category of expected content.
        - ``slot_id``: Unique identifier within the template (e.g. ``s1``).
        - ``evidence_id``: The EvidenceDAG node ID bound to this slot.
        - ``constraints``: Natural-language description of what the slot expects.
        - ``filled_content``: Content after the Writer fills the slot.
    """
    slot_type: SlotType
    slot_id: str
    evidence_id: str = ""
    constraints: str = ""
    filled_content: str = ""

    @property
    def marker(self) -> str:
        """The placeholder string as it appears in the skeleton."""
        return f"[{self.slot_type.value}:{self.slot_id}]"


class ParagraphTemplate(BaseModel):
    """
    A skeleton paragraph with typed slot placeholders.
    - **Description**:
        - ``skeleton`` contains natural-language text interspersed with
          slot markers in the form ``[SLOT_TYPE:slot_id]``.
        - ``slots`` lists all slots with their metadata and constraints.
    """
    skeleton: str = ""
    slots: List[TemplateSlot] = Field(default_factory=list)

    @property
    def is_fully_filled(self) -> bool:
        return all(s.filled_content for s in self.slots)

    @property
    def unfilled_slot_ids(self) -> List[str]:
        return [s.slot_id for s in self.slots if not s.filled_content]


# ---------------------------------------------------------------------------
# Parsing and rendering helpers
# ---------------------------------------------------------------------------

_SLOT_PATTERN = re.compile(
    r"\[("
    + "|".join(re.escape(st.value) for st in SlotType)
    + r"):([a-zA-Z0-9_]+)\]"
)


def parse_template_slots(skeleton: str) -> List[TemplateSlot]:
    """
    Extract TemplateSlot instances from a skeleton string.
    - **Description**:
        - Finds all ``[SLOT_TYPE:slot_id]`` markers and returns
          corresponding TemplateSlot objects.

    - **Args**:
        - ``skeleton`` (str): Template text with slot markers.

    - **Returns**:
        - ``List[TemplateSlot]``: Parsed slots in order of appearance.
    """
    slots: List[TemplateSlot] = []
    seen: set = set()
    for m in _SLOT_PATTERN.finditer(skeleton):
        slot_type_str, slot_id = m.group(1), m.group(2)
        if slot_id in seen:
            continue
        seen.add(slot_id)
        try:
            st = SlotType(slot_type_str)
        except ValueError:
            continue
        slots.append(TemplateSlot(slot_type=st, slot_id=slot_id))
    return slots


def render_filled_template(template: ParagraphTemplate) -> str:
    """
    Substitute filled content into the skeleton, returning final text.
    - **Description**:
        - Replaces each ``[SLOT_TYPE:slot_id]`` marker with the
          corresponding ``filled_content``.
        - Unfilled slots are left as-is (marker preserved).

    - **Args**:
        - ``template`` (ParagraphTemplate): Template with (partially) filled slots.

    - **Returns**:
        - ``str``: Rendered paragraph text.
    """
    result = template.skeleton
    for slot in template.slots:
        if slot.filled_content:
            result = result.replace(slot.marker, slot.filled_content)
    return result


def build_template_fill_prompt(
    template: ParagraphTemplate,
    evidence_snippets: Optional[Dict[str, str]] = None,
    valid_refs: Optional[List[str]] = None,
) -> str:
    """
    Compile a prompt instructing the Writer to fill template slots.
    - **Description**:
        - Lists each unfilled slot with its type, constraints, and
          bound evidence.
        - The Writer should output a JSON mapping ``slot_id -> filled text``
          or inline LaTeX with the slots replaced.

    - **Args**:
        - ``template`` (ParagraphTemplate): The template to fill.
        - ``evidence_snippets`` (Dict[str, str]): evidence_id -> snippet text.
        - ``valid_refs`` (List[str]): Allowed citation keys.

    - **Returns**:
        - ``str``: Compiled prompt.
    """
    evidence_snippets = evidence_snippets or {}
    valid_refs = valid_refs or []
    parts: List[str] = [
        "## Task: Fill the template slots in the following paragraph skeleton.\n",
        f"### Skeleton\n```\n{template.skeleton}\n```\n",
        "### Slots to Fill",
    ]
    for slot in template.slots:
        if slot.filled_content:
            continue
        ev_text = evidence_snippets.get(slot.evidence_id, "—")
        parts.append(
            f"- **[{slot.slot_type.value}:{slot.slot_id}]**\n"
            f"  Type: {slot.slot_type.value}\n"
            f"  Evidence: {ev_text}\n"
            f"  Constraints: {slot.constraints or 'none'}"
        )

    if valid_refs:
        parts.append(f"\n### Valid Citation Keys\n{', '.join(valid_refs[:10])}")

    parts.append(
        "\n### Output Requirements\n"
        "- Output the COMPLETE paragraph with all slots filled.\n"
        "- Replace each [SLOT_TYPE:slot_id] marker with the appropriate LaTeX content.\n"
        "- Every factual claim must use \\cite{{}} with a valid key.\n"
        "- Do NOT output the skeleton markers in the final output."
    )
    return "\n".join(parts)
