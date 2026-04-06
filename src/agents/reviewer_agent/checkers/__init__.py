"""
Feedback Checkers Package
- **Description**:
    - Contains all feedback checker implementations
    - Each checker evaluates a specific aspect of the paper
"""
from .base import FeedbackChecker
from .word_count import WordCountChecker

__all__ = [
    "FeedbackChecker",
    "WordCountChecker",
]
