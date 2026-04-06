"""
Tests for PlannerAgent._format_research_context_for_planning
- **Description**:
    - Validates type-safe handling of LLM-generated research context fields
    - Ensures dict/non-list values for list-expected fields do not cause
      'unhashable type: slice' errors
"""
import pytest
from unittest.mock import MagicMock

from src.agents.planner_agent.planner_agent import PlannerAgent


@pytest.fixture
def planner():
    """Create a PlannerAgent with mocked config (no real LLM needed)."""
    config = MagicMock()
    config.model_name = "test-model"
    config.api_key = "test-key"
    config.base_url = "http://localhost:9999"
    agent = PlannerAgent.__new__(PlannerAgent)
    agent.config = config
    agent.model_name = config.model_name
    agent.vlm_service = None
    agent._last_plan = None
    return agent


class TestFormatResearchContext:
    """Tests for _format_research_context_for_planning type safety."""

    def test_none_context(self, planner):
        result = planner._format_research_context_for_planning(None)
        assert result == "Not available."

    def test_empty_context(self, planner):
        result = planner._format_research_context_for_planning({})
        assert result == "Not available."

    def test_normal_list_fields(self, planner):
        ctx = {
            "research_area": "Machine Learning",
            "summary": "Overview of ML research.",
            "research_trends": ["trend1", "trend2", "trend3", "trend4"],
            "gaps": ["gap1", "gap2"],
            "contribution_ranking": {
                "P0": [{"contribution": "Main contribution"}],
                "P1": [],
                "P2": [],
            },
        }
        result = planner._format_research_context_for_planning(ctx)
        assert "Machine Learning" in result
        assert "trend1" in result
        assert "trend3" in result
        assert "trend4" not in result  # capped at 3
        assert "gap1" in result
        assert "Main contribution" in result

    def test_trends_as_dict_no_crash(self, planner):
        """LLM may return research_trends as a dict instead of a list."""
        ctx = {
            "research_area": "NLP",
            "research_trends": {"trend_a": "description_a", "trend_b": "description_b"},
            "gaps": ["gap1"],
        }
        result = planner._format_research_context_for_planning(ctx)
        assert "NLP" in result
        assert "gap1" in result
        assert "trends" not in result.lower() or "Key trends" not in result

    def test_gaps_as_dict_no_crash(self, planner):
        """LLM may return gaps as a dict instead of a list."""
        ctx = {
            "research_area": "CV",
            "research_trends": ["trend1"],
            "gaps": {"gap_a": "desc_a"},
        }
        result = planner._format_research_context_for_planning(ctx)
        assert "CV" in result
        assert "trend1" in result

    def test_contribution_ranking_as_list_no_crash(self, planner):
        """LLM may return contribution_ranking as a list instead of dict."""
        ctx = {
            "research_area": "RL",
            "contribution_ranking": ["contribution1", "contribution2"],
        }
        result = planner._format_research_context_for_planning(ctx)
        assert "RL" in result

    def test_ranking_items_as_dict_no_crash(self, planner):
        """LLM may return P0/P1 items as a dict instead of a list."""
        ctx = {
            "research_area": "AI",
            "contribution_ranking": {
                "P0": {"contribution": "top contribution"},
                "P1": [],
                "P2": [],
            },
        }
        result = planner._format_research_context_for_planning(ctx)
        assert "AI" in result

    def test_trends_as_string_no_crash(self, planner):
        """LLM may return research_trends as a plain string."""
        ctx = {
            "research_area": "Robotics",
            "research_trends": "single trend string",
            "gaps": "single gap",
        }
        result = planner._format_research_context_for_planning(ctx)
        assert "Robotics" in result

    def test_all_fields_malformed(self, planner):
        """Worst case: every list/dict field has wrong types."""
        ctx = {
            "research_area": 42,
            "summary": None,
            "research_trends": {"a": 1},
            "gaps": {"b": 2},
            "contribution_ranking": "not a dict",
        }
        result = planner._format_research_context_for_planning(ctx)
        assert "42" in result
        assert isinstance(result, str)
