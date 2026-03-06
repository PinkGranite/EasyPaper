"""
UI Rendering tools for agents.

These tools allow the LLM to autonomously decide to render specific UI components
on the frontend Canvas.
"""

import asyncio
from typing import Any, Dict

from .base import WriterTool, ToolResult
from ..llm_client import _progress_ctx


class UIRenderTool(WriterTool):
    """Base class for tools that render UI components on the frontend."""

    async def _emit_ui(self, component: str, props: Dict[str, Any]) -> None:
        """Emit a GEN_UI event using the active progress context."""
        ctx = _progress_ctx.get(None)
        if ctx and ctx.get("callback"):
            event = {
                "type": "gen_ui",
                "component": component,
                "props": props,
            }
            try:
                # Ensure callback is awaited
                await ctx["callback"](event)
            except Exception as e:
                print(f"[UIRenderTool] Error emitting UI event: {e}")


class ShowMarkdownTool(UIRenderTool):
    """Tool to render arbitrary markdown content in the Canvas."""

    @property
    def name(self) -> str:
        return "show_markdown"

    @property
    def description(self) -> str:
        return (
            "Renders markdown content in the user's UI Canvas. "
            "Use this tool to show intermediate thoughts, analyses, or "
            "formatted information to the user."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The markdown content to render"
                }
            },
            "required": ["content"]
        }

    async def execute(self, content: str, **kwargs) -> ToolResult:
        print(f"[Tool:show_markdown] Rendering {len(content)} chars of markdown")
        await self._emit_ui("MarkdownBlock", {"content": content})
        return ToolResult(
            success=True,
            data={"rendered": True},
            message="Successfully rendered markdown to the user's Canvas."
        )


class ShowJsonTool(UIRenderTool):
    """Tool to render JSON data in the Canvas."""

    @property
    def name(self) -> str:
        return "show_json_data"

    @property
    def description(self) -> str:
        return (
            "Renders structured data (JSON) in the user's UI Canvas. "
            "Use this tool to display data tables, configurations, or raw output."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "description": "The JSON object/dict to render"
                }
            },
            "required": ["data"]
        }

    async def execute(self, data: Dict[str, Any], **kwargs) -> ToolResult:
        print(f"[Tool:show_json_data] Rendering JSON data")
        # For now, fallback to DynamicRenderer's JSON logic or use a generic Markdown format
        json_str = str(data)
        markdown_content = f"```json\n{json_str}\n```"
        await self._emit_ui("MarkdownBlock", {"content": markdown_content})
        return ToolResult(
            success=True,
            data={"rendered": True},
            message="Successfully rendered JSON data to the user's Canvas."
        )
