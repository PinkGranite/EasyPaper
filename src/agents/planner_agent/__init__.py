"""
Planner Agent Package
- **Description**:
    - Creates detailed paragraph-level paper plans
    - Plans structure, content allocation, and style
"""
from .planner_agent import PlannerAgent
from .models import (
    PaperPlan,
    SectionPlan,
    ParagraphPlan,
    FigurePlacement,
    TablePlacement,
    PlanRequest,
    PlanResult,
    PaperType,
    NarrativeStyle,
    FigureInfo,
    TableInfo,
    calculate_total_words,
    estimate_target_paragraphs,
)

__all__ = [
    "PlannerAgent",
    "PaperPlan",
    "SectionPlan",
    "ParagraphPlan",
    "FigurePlacement",
    "TablePlacement",
    "PlanRequest",
    "PlanResult",
    "PaperType",
    "NarrativeStyle",
    "FigureInfo",
    "TableInfo",
    "calculate_total_words",
    "estimate_target_paragraphs",
]
