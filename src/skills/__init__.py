"""
Skills System for EasyPaper
- **Description**:
    - Provides pluggable writing constraints, reviewer rules, and venue profiles
    - Skills are loaded from YAML files and registered in a global registry
    - The PromptCompiler and ReviewerAgent consume skills at runtime
"""
from .models import WritingSkill
from .loader import SkillLoader
from .registry import SkillRegistry

__all__ = ["WritingSkill", "SkillLoader", "SkillRegistry"]
