"""
Thinking-Aware LLM Client
- **Description**:
    - Drop-in replacement for AsyncOpenAI that transparently separates
      thinking / reasoning blocks from model responses.
    - Supports <think>, <thinking>, <reasoning> tag families as well as
      orphaned closing tags (e.g. K2-Think API style).
    - Non-thinking models are unaffected — if no tags are detected the
      original content is returned as-is.
    - Thinking content is preserved on ``choice.message._thinking``
      for callers that need it (e.g. streaming progress callbacks).
    - Supports automatic progress event emission via contextvars so
      every LLM call can report thinking/response to a central emitter.
"""

import asyncio
import contextvars
import re
import time
from typing import Any, Callable, Coroutine, Optional, Tuple

from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# Thinking-content extraction
# ---------------------------------------------------------------------------

_THINKING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>(.*?)</(?:think|thinking|reasoning)>",
    re.DOTALL,
)

_ORPHAN_CLOSING_TAGS = ("</think>", "</thinking>", "</reasoning>")

_OPENING_TAGS = ("<think>", "<thinking>", "<reasoning>")


def extract_thinking(text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Separate thinking blocks from clean content.

    - **Description**:
        - Extracts matched ``<think>…</think>``, ``<thinking>…</thinking>``,
          and ``<reasoning>…</reasoning>`` blocks (including multi-line).
        - Handles orphaned closing tags where the API omits the opening tag
          (e.g. K2-Think returns ``thought…\\n</think>\\nactual answer``).
        - Returns ``(None, original_text)`` when no thinking markers are
          present, so non-thinking models work without side-effects.

    - **Args**:
        - `text` (str | None): Raw LLM response content.

    - **Returns**:
        - `(thinking, clean)` (Tuple[str | None, str | None]):
          The extracted thinking content and the cleaned output.
    """
    if not text:
        return None, text

    thinking_parts: list[str] = []
    for match in _THINKING_BLOCK_RE.finditer(text):
        thinking_parts.append(match.group(1).strip())

    clean = _THINKING_BLOCK_RE.sub("", text)

    for tag in _ORPHAN_CLOSING_TAGS:
        if tag in clean:
            orphan_thinking = clean.split(tag, 1)[0]
            stripped_orphan = orphan_thinking.strip()
            for opening in _OPENING_TAGS:
                stripped_orphan = stripped_orphan.removeprefix(opening)
            stripped_orphan = stripped_orphan.strip()
            if stripped_orphan:
                thinking_parts.append(stripped_orphan)
            clean = clean.split(tag, 1)[-1]

    thinking = "\n\n".join(thinking_parts).strip() or None
    clean = clean.strip() or text
    return thinking, clean


def strip_thinking(text: Optional[str]) -> Optional[str]:
    """
    Remove thinking / reasoning blocks from LLM output (convenience wrapper).

    - **Args**:
        - `text` (str | None): Raw LLM response content.

    - **Returns**:
        - `str | None`: Cleaned content with thinking blocks removed.
    """
    _, clean = extract_thinking(text)
    return clean


def get_thinking(response) -> Optional[str]:
    """
    Retrieve preserved thinking content from a chat completion response.

    - **Args**:
        - `response`: OpenAI ChatCompletion response object.

    - **Returns**:
        - `str | None`: Thinking content from the first choice, or None.
    """
    if not response or not response.choices:
        return None
    msg = response.choices[0].message
    return getattr(msg, "_thinking", None)


# ---------------------------------------------------------------------------
# Contextvars-based progress tracking
# ---------------------------------------------------------------------------

ProgressCallback = Callable[..., Coroutine[Any, Any, None]]

_progress_ctx: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "llm_progress", default=None
)


def set_llm_progress_context(
    callback: ProgressCallback,
    agent: str,
    phase: str = "",
    section: str = "",
) -> None:
    """
    Set the LLM progress context for automatic event emission.

    - **Description**:
        - Stores a progress callback plus agent/phase/section metadata
          in a contextvar so ``_CompletionsProxy.create()`` can auto-emit
          ``llm_response`` events after every LLM call.

    - **Args**:
        - `callback` (ProgressCallback): Async callback that accepts event dicts.
        - `agent` (str): Name of the calling agent (e.g. "WriterAgent").
        - `phase` (str): Current pipeline phase.
        - `section` (str): Current section being generated.
    """
    _progress_ctx.set({
        "callback": callback,
        "agent": agent,
        "phase": phase,
        "section": section,
    })


def clear_llm_progress_context() -> None:
    """Clear the LLM progress context."""
    _progress_ctx.set(None)


# ---------------------------------------------------------------------------
# Transparent AsyncOpenAI wrapper
# ---------------------------------------------------------------------------


class _CompletionsProxy:
    """Intercepts ``chat.completions.create`` to separate thinking content
    and auto-emit progress events when a context is active."""

    __slots__ = ("_completions",)

    def __init__(self, completions):
        self._completions = completions

    async def create(self, **kwargs):
        t0 = time.monotonic()
        response = await self._completions.create(**kwargs)
        latency = round(time.monotonic() - t0, 2)

        thinking = None
        clean = None
        for choice in response.choices:
            if choice.message and choice.message.content:
                thinking, clean = extract_thinking(choice.message.content)
                choice.message.content = clean
                try:
                    choice.message._thinking = thinking
                except (AttributeError, TypeError):
                    pass

        ctx = _progress_ctx.get(None)
        if ctx and ctx.get("callback"):
            try:
                model_name = kwargs.get("model", "")
                preview = ((clean or "")[:200] + ("…" if len(clean) > 200 else "")) if clean else ""
                event = {
                    "type": "llm_response",
                    "agent": ctx.get("agent", ""),
                    "phase": ctx.get("phase", ""),
                    "section_type": ctx.get("section", ""),
                    "model": model_name,
                    "latency_seconds": latency,
                    "response_chars": len(clean) if clean else 0,
                    "response_preview": preview,
                }
                if thinking:
                    event["thinking"] = thinking
                asyncio.ensure_future(ctx["callback"](event))
            except Exception:
                pass

        return response

    def __getattr__(self, name):
        return getattr(self._completions, name)


class _ChatProxy:
    """Proxy that replaces ``chat.completions`` with the thinking-aware variant."""

    __slots__ = ("_chat", "completions")

    def __init__(self, chat):
        self._chat = chat
        self.completions = _CompletionsProxy(chat.completions)

    def __getattr__(self, name):
        return getattr(self._chat, name)


class LLMClient:
    """
    Drop-in replacement for ``AsyncOpenAI``.

    - **Description**:
        - Wraps ``AsyncOpenAI`` and transparently separates thinking /
          reasoning blocks from every ``chat.completions.create`` response.
        - Clean content is set on ``message.content``; thinking content
          is preserved on ``message._thinking`` for callers that need it.
        - When a progress context is active (via ``set_llm_progress_context``),
          automatically emits ``llm_response`` events with thinking content,
          agent attribution, and latency information.
        - All other attributes and methods are delegated to the inner client.
        - Non-thinking models are unaffected.

    - **Args**:
        - Same keyword arguments as ``AsyncOpenAI``.
    """

    __slots__ = ("_client", "chat")

    def __init__(self, **kwargs):
        self._client = AsyncOpenAI(**kwargs)
        self.chat = _ChatProxy(self._client.chat)

    def __getattr__(self, name):
        return getattr(self._client, name)
