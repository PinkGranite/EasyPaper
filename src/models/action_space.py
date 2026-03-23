"""
Unified Action Space
- **Description**:
    - Formalises the hierarchical action space for constrained generation.
    - TextAction (A_text): continuous actions involving LLM text generation
      (expand, reduce, revise, etc.)
    - LayoutAction (A_layout): discrete structural/LaTeX operations
      (resize figures, move elements, fix formatting, etc.)
    - Action: unified model with preconditions, postconditions and metadata.
    - Provides ``from_legacy_action`` for backward-compatible conversion from
      the old string-based action representations.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action enums
# ---------------------------------------------------------------------------

class TextAction(str, Enum):
    """Continuous actions that invoke LLM text generation (A_text)."""
    EXPAND = "expand"
    REDUCE = "reduce"
    REVISE = "revise"
    REFINE_PARAGRAPHS = "refine_paragraphs"
    REWRITE_CLAIM = "rewrite_claim"
    LOGIC_FIX = "logic_fix"

    # VLM-originated text actions
    TRIM = "trim"
    KEEP = "keep"


class LayoutAction(str, Enum):
    """Discrete structural / LaTeX operations (A_layout)."""
    RESIZE_FIGURE = "resize_figure"
    DOWNGRADE_WIDE = "downgrade_wide"
    MOVE_FIGURE = "move_figure"
    MOVE_TABLE = "move_table"
    CREATE_APPENDIX = "create_appendix"
    FIX_LATEX = "fix_latex"
    RESIZE_FIGURES = "resize_figures"
    MOVE_TO_APPENDIX = "move_to_appendix"


class ActionStatus(str, Enum):
    """Lifecycle state of an action."""
    NOOP = "noop"
    PENDING = "pending"
    APPLIED = "applied"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Unified priority table
# ---------------------------------------------------------------------------

ACTION_PRIORITY: Dict[str, int] = {
    # Layout actions (highest priority — structural correctness first)
    "fix_latex": 100,
    "resize_figure": 90,
    "resize_figures": 90,
    "downgrade_wide": 85,
    "move_figure": 80,
    "move_table": 80,
    "move_to_appendix": 80,
    "create_appendix": 75,
    # Text actions
    "reduce": 80,
    "trim": 80,
    "refine_paragraphs": 40,
    "logic_fix": 35,
    "revise": 30,
    "rewrite_claim": 30,
    "expand": 20,
    "keep": 0,
    "noop": 0,
    "ok": 0,
}


# ---------------------------------------------------------------------------
# Precondition / postcondition registry
# ---------------------------------------------------------------------------

_ACTION_CONTRACTS: Dict[str, Dict[str, List[str]]] = {
    "expand": {
        "preconditions": ["word_count < target_words", "section_exists"],
        "postconditions": ["word_count >= target_words"],
    },
    "reduce": {
        "preconditions": ["word_count > target_words", "section_exists"],
        "postconditions": ["word_count <= target_words"],
    },
    "trim": {
        "preconditions": ["word_count > target_words"],
        "postconditions": ["word_count <= target_words"],
    },
    "revise": {
        "preconditions": ["section_exists", "feedback_available"],
        "postconditions": ["issues_addressed"],
    },
    "refine_paragraphs": {
        "preconditions": ["paragraph_feedbacks_available"],
        "postconditions": ["paragraph_issues_addressed"],
    },
    "rewrite_claim": {
        "preconditions": ["claim_id_valid", "evidence_bound"],
        "postconditions": ["claim_anchored_to_evidence"],
    },
    "logic_fix": {
        "preconditions": ["logic_issue_detected"],
        "postconditions": ["logic_consistency_restored"],
    },
    "resize_figure": {
        "preconditions": ["figure_exists", "new_width_valid"],
        "postconditions": ["figure_width_updated"],
    },
    "downgrade_wide": {
        "preconditions": ["figure_is_wide"],
        "postconditions": ["figure_is_narrow"],
    },
    "move_figure": {
        "preconditions": ["figure_exists"],
        "postconditions": ["figure_in_appendix"],
    },
    "move_table": {
        "preconditions": ["table_exists"],
        "postconditions": ["table_in_appendix"],
    },
    "move_to_appendix": {
        "preconditions": ["element_exists"],
        "postconditions": ["element_in_appendix"],
    },
    "create_appendix": {
        "preconditions": ["appendix_not_exists"],
        "postconditions": ["appendix_created"],
    },
    "fix_latex": {
        "preconditions": ["latex_error_detected"],
        "postconditions": ["latex_compiles_clean"],
    },
    "resize_figures": {
        "preconditions": ["figures_exist"],
        "postconditions": ["figures_resized"],
    },
}


# ---------------------------------------------------------------------------
# Unified Action model
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    Unified action representation spanning both text and layout operations.
    - **Description**:
        - Replaces the scattered action_type / action string fields across
          StructuralAction, SectionFeedback, and RevisionTask.
        - Carries formal preconditions and postconditions for each action.

    - **Fields**:
        - ``action_type``: One of TextAction or LayoutAction.
        - ``status``: Current lifecycle state.
        - ``target_id``: LaTeX label or section identifier the action targets.
        - ``section``: Section where the action applies.
        - ``params``: Action-specific parameters.
        - ``estimated_impact``: Estimated effect (e.g. word delta, page savings).
        - ``preconditions``: Conditions that must hold before execution.
        - ``postconditions``: Conditions that should hold after execution.
        - ``priority``: Numeric priority (higher = execute first).
    """
    action_type: str
    status: ActionStatus = ActionStatus.PENDING
    target_id: str = ""
    section: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)
    estimated_impact: float = 0.0
    preconditions: List[str] = Field(default_factory=list)
    postconditions: List[str] = Field(default_factory=list)
    priority: int = 0

    @property
    def is_text_action(self) -> bool:
        try:
            TextAction(self.action_type)
            return True
        except ValueError:
            return False

    @property
    def is_layout_action(self) -> bool:
        try:
            LayoutAction(self.action_type)
            return True
        except ValueError:
            return False


# ---------------------------------------------------------------------------
# Legacy conversion
# ---------------------------------------------------------------------------

_LEGACY_ALIAS: Dict[str, str] = {
    "ok": "keep",
}


def from_legacy_action(
    action_str: str,
    *,
    target_id: str = "",
    section: str = "",
    params: Optional[Dict[str, Any]] = None,
    estimated_impact: float = 0.0,
) -> Action:
    """
    Convert a legacy action string into a unified Action.
    - **Description**:
        - Maps old string-based action representations (from StructuralAction,
          SectionFeedback, RevisionTask, SectionAdvice) into the new model.

    - **Args**:
        - ``action_str`` (str): The legacy action string.
        - ``target_id`` (str): Target element identifier.
        - ``section`` (str): Section context.
        - ``params`` (Dict): Extra parameters.
        - ``estimated_impact`` (float): Estimated effect magnitude.

    - **Returns**:
        - ``Action``: Unified action instance.
    """
    canonical = _LEGACY_ALIAS.get(action_str, action_str)
    contract = _ACTION_CONTRACTS.get(canonical, {})
    return Action(
        action_type=canonical,
        target_id=target_id,
        section=section,
        params=params or {},
        estimated_impact=estimated_impact,
        preconditions=contract.get("preconditions", []),
        postconditions=contract.get("postconditions", []),
        priority=ACTION_PRIORITY.get(canonical, 0),
    )
