"""
Guards for migrating internal revision execution off WriterAgent.run().
"""
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def test_revision_executor_does_not_call_writer_run():
    from src.agents.metadata_agent.revision_executor import RevisionExecutor

    source = inspect.getsource(RevisionExecutor)
    assert "._writer.run(" not in source, (
        "RevisionExecutor must not depend on the deprecated WriterAgent.run() "
        "path after migration."
    )


def test_revision_executor_uses_direct_rewrite_helper():
    from src.agents.metadata_agent.revision_executor import RevisionExecutor

    source = inspect.getsource(RevisionExecutor)
    assert "._writer.rewrite_content(" in source, (
        "RevisionExecutor should use WriterAgent.rewrite_content() for "
        "internal revision flows."
    )


@pytest.mark.asyncio
async def test_revise_section_uses_rewrite_content_and_returns_text():
    from src.agents.metadata_agent.revision_executor import RevisionExecutor

    host = SimpleNamespace(
        _writer=SimpleNamespace(rewrite_content=AsyncMock(return_value="Revised section text.")),
    )
    executor = RevisionExecutor(host)

    revised = await executor._revise_section(
        section_type="method",
        current_content="Original section text.",
        revision_prompt="Improve clarity.",
        metadata=None,
    )

    assert revised == "Revised section text."
    host._writer.rewrite_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_revise_paragraph_uses_rewrite_content_and_returns_text():
    from src.agents.metadata_agent.revision_executor import RevisionExecutor

    host = SimpleNamespace(
        _writer=SimpleNamespace(rewrite_content=AsyncMock(return_value="Revised paragraph text.")),
    )
    executor = RevisionExecutor(host)

    revised = await executor._revise_paragraph(
        section_type="result",
        paragraph_index=2,
        paragraph_text="Original paragraph text.",
        instruction="Tighten the claim.",
    )

    assert revised == "Revised paragraph text."
    host._writer.rewrite_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_revise_section_sentences_rewrites_target_sentence_only():
    from src.agents.metadata_agent.revision_executor import RevisionExecutor

    host = SimpleNamespace(
        _writer=SimpleNamespace(rewrite_content=AsyncMock(return_value="Revised second sentence.")),
    )
    executor = RevisionExecutor(host)

    revised = await executor._revise_section_sentences(
        section_type="discussion",
        current_content="First sentence. Second sentence. Third sentence.",
        sentence_feedbacks=[
            {
                "paragraph_index": 0,
                "sentence_index": 1,
                "issue": "Too vague",
                "suggestion": "Make the claim concrete",
            }
        ],
        metadata=None,
    )

    assert revised == "First sentence. Revised second sentence. Third sentence."
    host._writer.rewrite_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_revise_section_returns_none_when_rewrite_is_empty():
    from src.agents.metadata_agent.revision_executor import RevisionExecutor

    host = SimpleNamespace(
        _writer=SimpleNamespace(rewrite_content=AsyncMock(return_value="")),
    )
    executor = RevisionExecutor(host)

    revised = await executor._revise_section(
        section_type="method",
        current_content="Original section text.",
        revision_prompt="Improve clarity.",
        metadata=None,
    )

    assert revised is None
