"""
Reviewer Agent Package
- **Description**:
    - Provides iterative feedback for paper writing
    - Extensible checker system for different review aspects
"""
from .reviewer_agent import ReviewerAgent
from .models import (
    ReviewContext,
    ReviewResult,
    FeedbackResult,
    SectionFeedback,
    ParagraphFeedback,
    Severity,
    ReviewRequest,
)
from .checkers import FeedbackChecker, WordCountChecker

__all__ = [
    "ReviewerAgent",
    "ReviewContext",
    "ReviewResult", 
    "FeedbackResult",
    "SectionFeedback",
    "ParagraphFeedback",
    "Severity",
    "ReviewRequest",
    "FeedbackChecker",
    "WordCountChecker",
]
