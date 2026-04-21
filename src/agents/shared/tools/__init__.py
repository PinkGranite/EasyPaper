"""
Agent Tools Package.

Provides tool base classes, registry, concrete tool implementations,
and a global tool factory for config-driven tool instantiation.
"""

from typing import Any, Callable, Dict

from .base import WriterTool, ToolResult
from .registry import ToolRegistry, get_default_registry, register_default_tools
from .citation_tools import CitationValidatorTool, WordCountTool, KeyPointCoverageTool
from .ui_tools import ShowMarkdownTool, ShowJsonTool


def _create_show_markdown(ctx: Dict[str, Any]) -> ShowMarkdownTool:
    return ShowMarkdownTool()


def _create_show_json_data(ctx: Dict[str, Any]) -> ShowJsonTool:
    return ShowJsonTool()


def _create_citation_validator(ctx: Dict[str, Any]) -> CitationValidatorTool:
    """Factory for CitationValidatorTool."""
    return CitationValidatorTool(ctx.get("valid_keys", set()))


def _create_word_count(ctx: Dict[str, Any]) -> WordCountTool:
    """Factory for WordCountTool."""
    return WordCountTool()


def _create_key_point_coverage(ctx: Dict[str, Any]) -> KeyPointCoverageTool:
    """Factory for KeyPointCoverageTool."""
    return KeyPointCoverageTool(ctx.get("key_points", []))


def _create_paper_search(ctx: Dict[str, Any]) -> "WriterTool":
    """Factory for PaperSearchTool (lazy import to avoid circular deps)."""
    from .paper_search import PaperSearchTool
    return PaperSearchTool(
        serpapi_api_key=ctx.get("serpapi_api_key"),
        semantic_scholar_api_key=ctx.get("semantic_scholar_api_key"),
        timeout=ctx.get("timeout", 10),
    )


def _create_ask_tool(ctx: Dict[str, Any]) -> "WriterTool":
    """Factory for AskTool — unified routing consultation tool."""
    from .ask_tool import AskTool
    handlers: Dict[str, Any] = {}

    memory = ctx.get("memory")
    if memory and hasattr(memory, "search"):
        handlers["memory"] = memory.search

    planner = ctx.get("planner")
    if planner and hasattr(planner, "answer"):
        handlers["planner"] = planner.answer

    reviewer = ctx.get("reviewer")
    if reviewer and hasattr(reviewer, "answer"):
        async def _reviewer_answer(q: str) -> str:
            return await reviewer.answer(q, memory=memory)
        handlers["reviewer"] = _reviewer_answer

    # Inject LLM refine callable into SessionMemory for two-stage search.
    # Reuses reviewer's LLM config so SessionMemory stays a pure data class.
    if memory and hasattr(memory, "set_llm_refine") and reviewer:
        _inject_llm_refine(memory, reviewer)

    if not handlers:
        raise ValueError("AskTool requires at least one handler (memory/planner/reviewer)")
    return AskTool(handlers, memory=memory)


def _inject_llm_refine(memory, reviewer) -> None:
    """
    Build an LLM refine callable from reviewer's config and inject it into memory.
    - **Description**:
        - Constructs a lightweight async callable that sends
          (question, candidate_context) to the reviewer's LLM endpoint
          for semantic refinement.
        - Keeps SessionMemory free of direct LLM dependencies.

    - **Args**:
        - `memory`: SessionMemory instance with set_llm_refine()
        - `reviewer`: Agent with config.api_key, config.base_url, model_name
    """
    try:
        from .llm_client import LLMClient

        api_key = getattr(reviewer.config, "api_key", None)
        base_url = getattr(reviewer.config, "base_url", None)
        model_name = getattr(reviewer, "model_name", None)
        if not api_key or not model_name:
            return

        client = LLMClient(api_key=api_key, base_url=base_url)

        async def _llm_refine(question: str, context: str) -> str:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an academic writing assistant. Based on the "
                            "session memory context below, answer the question "
                            "concisely and precisely. Keep your response under "
                            "200 words. If the context does not contain relevant "
                            "information, say so briefly."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Memory context:\n{context}\n\n"
                            f"Question: {question}"
                        ),
                    },
                ],
                temperature=0.3,
            )
            return response.choices[0].message.content or ""

        memory.set_llm_refine(_llm_refine)
    except Exception:
        pass


# Global tool factory registry.
# Maps tool name -> factory function that takes a context dict and returns a WriterTool.
# To add a new tool, define the tool class and add an entry here.
TOOL_FACTORY: Dict[str, Callable[[Dict[str, Any]], WriterTool]] = {
    "validate_citations": _create_citation_validator,
    "count_words": _create_word_count,
    "check_key_points": _create_key_point_coverage,
    "search_papers": _create_paper_search,
    "ask": _create_ask_tool,
    "show_markdown": _create_show_markdown,
    "show_json_data": _create_show_json_data,
}


__all__ = [
    "WriterTool",
    "ToolResult",
    "ToolRegistry",
    "get_default_registry",
    "register_default_tools",
    "CitationValidatorTool",
    "WordCountTool",
    "KeyPointCoverageTool",
    "TOOL_FACTORY",
]
