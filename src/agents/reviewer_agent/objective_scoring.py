"""
Multi-Objective Scoring for Conflict Resolution
- **Description**:
    - Defines four optimization objectives (Logic, Layout, Citation, Style)
    - Provides scoring functions for each objective
    - Implements Pareto front selection and lexicographic tiebreaking
    - Used by _resolve_conflicts_with_llm() as primary resolution mechanism
"""
from enum import Enum
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, Field


class ObjectiveType(str, Enum):
    """Optimization dimensions for multi-objective conflict resolution."""
    LOGIC = "logic"
    LAYOUT = "layout"
    CITATION = "citation"
    STYLE = "style"


DEFAULT_PRIORITY_ORDER: List[ObjectiveType] = [
    ObjectiveType.LAYOUT,
    ObjectiveType.CITATION,
    ObjectiveType.LOGIC,
    ObjectiveType.STYLE,
]


class ObjectiveScore(BaseModel):
    """Single-dimension score for a candidate action."""
    objective: ObjectiveType
    score: float = 0.0
    weight: float = 1.0
    details: str = ""


class ActionScorecard(BaseModel):
    """Aggregate multi-objective scorecard for a candidate action."""
    action: str
    section_type: str = ""
    source: str = ""
    scores: List[ObjectiveScore] = Field(default_factory=list)
    weighted_sum: float = 0.0
    dominates: List[str] = Field(default_factory=list)

    def compute_weighted_sum(self) -> float:
        """Compute and cache the weighted sum of all objective scores."""
        self.weighted_sum = sum(s.score * s.weight for s in self.scores)
        return self.weighted_sum

    def get_score(self, obj_type: ObjectiveType) -> float:
        """Retrieve the score for a specific objective, defaulting to 0."""
        for s in self.scores:
            if s.objective == obj_type:
                return s.score
        return 0.0


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_logic(
    feedback_details: Dict[str, Any],
    section_content: str = "",
) -> ObjectiveScore:
    """
    Score logical consistency based on issue count and severity.
    - **Args**:
        - `feedback_details`: Checker output details containing 'issues' list
        - `section_content`: Current section text (unused, reserved)
    - **Returns**:
        - `ObjectiveScore` with 0.0 (many critical issues) to 1.0 (no issues)
    """
    issues = feedback_details.get("issues", [])
    if not issues:
        return ObjectiveScore(objective=ObjectiveType.LOGIC, score=1.0, details="no issues")

    severity_weights = {"high": 3, "critical": 4, "medium": 2, "low": 1}
    total_penalty = sum(severity_weights.get(str(i.get("severity", "medium")), 2) for i in issues)
    max_penalty = len(issues) * 4
    score = max(0.0, 1.0 - (total_penalty / max(max_penalty, 1)))
    return ObjectiveScore(
        objective=ObjectiveType.LOGIC,
        score=round(score, 4),
        details=f"{len(issues)} issues, penalty={total_penalty}/{max_penalty}",
    )


def score_layout(
    feedback_details: Dict[str, Any],
    vlm_result: Optional[Dict[str, Any]] = None,
) -> ObjectiveScore:
    """
    Score physical layout quality from VLM review data.
    - **Args**:
        - `feedback_details`: Checker output details
        - `vlm_result`: VLM review output with 'sections' containing fill_percentage
    - **Returns**:
        - `ObjectiveScore` with 0.0 (severe overflow/underfill) to 1.0 (ideal)
    """
    if not vlm_result:
        return ObjectiveScore(objective=ObjectiveType.LAYOUT, score=0.5, details="no VLM data")

    sections = vlm_result.get("sections", {})
    if not sections:
        return ObjectiveScore(objective=ObjectiveType.LAYOUT, score=0.5, details="no section data")

    penalties = 0.0
    count = 0
    for sec_data in sections.values():
        if not isinstance(sec_data, dict):
            continue
        fill = sec_data.get("fill_percentage", 100)
        count += 1
        if fill > 100:
            penalties += min((fill - 100) / 50.0, 1.0)
        elif fill < 30:
            penalties += (30 - fill) / 30.0

    score = max(0.0, 1.0 - (penalties / max(count, 1)))
    return ObjectiveScore(
        objective=ObjectiveType.LAYOUT,
        score=round(score, 4),
        details=f"{count} sections evaluated, penalty={penalties:.2f}",
    )


def score_citation(
    feedback_details: Dict[str, Any],
    citation_stats: Optional[Dict[str, Any]] = None,
) -> ObjectiveScore:
    """
    Score citation correctness.
    - **Args**:
        - `feedback_details`: Checker output details
        - `citation_stats`: Dict with 'total' and 'invalid' citation counts
    - **Returns**:
        - `ObjectiveScore` with 0.0 (all citations invalid) to 1.0 (all valid)
    """
    if not citation_stats:
        return ObjectiveScore(objective=ObjectiveType.CITATION, score=0.5, details="no citation data")

    total = citation_stats.get("total", 0)
    invalid = citation_stats.get("invalid", 0)
    if total == 0:
        return ObjectiveScore(objective=ObjectiveType.CITATION, score=1.0, details="no citations")

    score = max(0.0, 1.0 - (invalid / total))
    return ObjectiveScore(
        objective=ObjectiveType.CITATION,
        score=round(score, 4),
        details=f"{invalid}/{total} invalid",
    )


def score_style(
    feedback_details: Dict[str, Any],
    style_issues: Optional[List[Dict[str, Any]]] = None,
) -> ObjectiveScore:
    """
    Score writing quality based on style checker issues.
    - **Args**:
        - `feedback_details`: Checker output details
        - `style_issues`: List of style issues from StyleChecker
    - **Returns**:
        - `ObjectiveScore` with 0.0 (many issues) to 1.0 (clean)
    """
    issues = style_issues or feedback_details.get("style_issues", [])
    if not issues:
        return ObjectiveScore(objective=ObjectiveType.STYLE, score=1.0, details="no style issues")

    severity_weights = {"high": 3, "medium": 2, "low": 1}
    total_penalty = sum(severity_weights.get(str(i.get("severity", "medium")), 2) for i in issues)
    max_penalty = len(issues) * 3
    score = max(0.0, 1.0 - (total_penalty / max(max_penalty, 1)))
    return ObjectiveScore(
        objective=ObjectiveType.STYLE,
        score=round(score, 4),
        details=f"{len(issues)} style issues",
    )


# ---------------------------------------------------------------------------
# Pareto front computation
# ---------------------------------------------------------------------------

def _dominates(a: ActionScorecard, b: ActionScorecard) -> bool:
    """Return True if scorecard `a` Pareto-dominates `b` (>= on all, > on at least one)."""
    all_objectives = set(s.objective for s in a.scores) | set(s.objective for s in b.scores)
    at_least_one_better = False
    for obj in all_objectives:
        sa = a.get_score(obj)
        sb = b.get_score(obj)
        if sa < sb:
            return False
        if sa > sb:
            at_least_one_better = True
    return at_least_one_better


def compute_pareto_front(scorecards: List[ActionScorecard]) -> List[ActionScorecard]:
    """
    Return the Pareto-optimal actions (non-dominated set).
    - **Args**:
        - `scorecards`: All candidate scorecards
    - **Returns**:
        - List of non-dominated ActionScorecards
    """
    if len(scorecards) <= 1:
        return list(scorecards)

    front: List[ActionScorecard] = []
    for card in scorecards:
        dominated = False
        for other in scorecards:
            if other is card:
                continue
            if _dominates(other, card):
                dominated = True
                card.dominates = []
                break
        if not dominated:
            card.dominates = [
                other.action for other in scorecards
                if other is not card and _dominates(card, other)
            ]
            front.append(card)
    return front


def select_action(
    scorecards: List[ActionScorecard],
    priority_order: Optional[List[ObjectiveType]] = None,
    lexicographic_threshold: float = 0.05,
) -> ActionScorecard:
    """
    Select the best action: Pareto front -> lexicographic by priority.
    - **Args**:
        - `scorecards`: All candidate scorecards
        - `priority_order`: Objective priority for tiebreaking (default: LAYOUT > CITATION > LOGIC > STYLE)
        - `lexicographic_threshold`: Min score gap to consider meaningfully different
    - **Returns**:
        - The selected ActionScorecard
    """
    if not scorecards:
        raise ValueError("No scorecards to select from")
    if len(scorecards) == 1:
        return scorecards[0]

    for card in scorecards:
        card.compute_weighted_sum()

    front = compute_pareto_front(scorecards)
    if len(front) == 1:
        return front[0]

    order = priority_order or DEFAULT_PRIORITY_ORDER
    remaining = list(front)
    for obj_type in order:
        if len(remaining) <= 1:
            break
        best_score = max(c.get_score(obj_type) for c in remaining)
        narrowed = [c for c in remaining if best_score - c.get_score(obj_type) < lexicographic_threshold]
        if narrowed:
            remaining = narrowed

    remaining.sort(key=lambda c: c.weighted_sum, reverse=True)
    return remaining[0]


def needs_llm_arbitration(
    front: List[ActionScorecard],
    threshold: float = 0.05,
    priority_order: Optional[List[ObjectiveType]] = None,
) -> bool:
    """
    Determine if LLM arbitration is needed as a fallback.
    - **Description**:
        - Returns True when the Pareto front has >= 2 members and
          lexicographic scores are within threshold across all priority dimensions.
    """
    if len(front) < 2:
        return False
    order = priority_order or DEFAULT_PRIORITY_ORDER
    for obj_type in order:
        scores = [c.get_score(obj_type) for c in front]
        if max(scores) - min(scores) >= threshold:
            return False
    return True
