"""
Compatibility tests for the public writer surface after internal migration.
"""
import inspect
from unittest.mock import AsyncMock

import pytest


def test_writer_run_no_longer_uses_agent_ainvoke():
    from src.agents.writer_agent.writer_agent import WriterAgent

    source = inspect.getsource(WriterAgent.run)
    assert "self.agent.ainvoke" not in source, (
        "WriterAgent.run() should no longer execute the deprecated LangGraph "
        "graph directly."
    )


def test_writer_init_no_longer_builds_legacy_graph():
    from src.agents.writer_agent.writer_agent import WriterAgent

    source = inspect.getsource(WriterAgent.__init__)
    assert "self.init_agent()" not in source, (
        "WriterAgent should not eagerly initialize the deprecated LangGraph "
        "workflow during normal startup."
    )


def test_writer_no_longer_defines_init_agent():
    from src.agents.writer_agent.writer_agent import WriterAgent

    assert not hasattr(WriterAgent, "init_agent"), (
        "WriterAgent should no longer carry the dead LangGraph init_agent() "
        "constructor after cleanup."
    )


@pytest.mark.asyncio
async def test_writer_run_preserves_basic_result_shape():
    from src.agents.writer_agent.writer_agent import WriterAgent

    agent = WriterAgent.__new__(WriterAgent)
    agent.generate_content = AsyncMock(return_value={
        "generated_content": "Draft text",
        "llm_calls": 1,
        "iteration": 1,
    })
    agent.mini_review = AsyncMock(return_value={
        "generated_content": "Draft text",
        "review_result": {"passed": True},
        "review_history": [{"passed": True, "issues": [], "warnings": []}],
        "invalid_citations_removed": [],
    })
    agent.revise_content = AsyncMock()
    agent.extract_references = AsyncMock(return_value={
        "citation_ids": ["smith2024"],
        "figure_ids": [],
        "table_ids": [],
        "paragraph_units": [],
        "writer_response_section": [],
        "writer_response_paragraph": [],
    })
    agent._should_revise = lambda state: "done"

    result = await WriterAgent.run(
        agent,
        system_prompt="sys",
        user_prompt="user",
        section_type="introduction",
        enable_review=True,
    )

    assert result["generated_content"] == "Draft text"
    assert result["review_result"]["passed"] is True
    assert result["citation_ids"] == ["smith2024"]
    assert isinstance(result["review_history"], list)
    agent.generate_content.assert_awaited_once()
    agent.mini_review.assert_awaited_once()
    agent.extract_references.assert_awaited_once()
    agent.revise_content.assert_not_called()
