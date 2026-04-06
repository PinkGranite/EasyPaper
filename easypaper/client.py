"""
EasyPaper SDK client -- public entry point for programmatic paper generation.

- **Description**:
    - Wraps MetaDataAgent.generate_paper() behind a simple interface.
    - Supports one-shot ``generate()`` and streaming ``generate_stream()``.
    - Loads configuration from a YAML file and wires up internal agents
      automatically -- callers never touch agent internals.

Usage::

    from easypaper import EasyPaper, PaperMetaData

    ep = EasyPaper(config_path="configs/my.yaml")
    result = await ep.generate(PaperMetaData(
        title="My Paper",
        idea_hypothesis="...",
        method="...",
        data="...",
        experiments="...",
    ))
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

from src.config.loader import load_config
from src.config.schema import AppConfig, SkillsConfig
from src.agents import initialize_agents

logger = logging.getLogger(__name__)

_SENTINEL = object()


class EasyPaper:
    """
    High-level SDK client for EasyPaper paper generation.

    - **Args**:
        - `config_path` (str | Path, optional): Path to a YAML config file.
            If omitted, falls back to ``AGENT_CONFIG_PATH`` env var /
            ``./configs/dev.yaml`` (same logic as the server).
        - `config` (AppConfig, optional): Pre-built config object.  Takes
            precedence over *config_path* when both are given.
    """

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            if config_path is not None:
                import os
                os.environ["AGENT_CONFIG_PATH"] = str(config_path)
            self._config = load_config()

        self._agents = self._build_agents(self._config)
        self._metadata_agent = self._agents["metadata"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        metadata: Any,
        **options: Any,
    ) -> Any:
        """
        One-shot paper generation.

        - **Args**:
            - `metadata` (PaperMetaData): Paper input (title, idea, method, ...).
            - `**options`: Forwarded to ``MetaDataAgent.generate_paper()``
                (e.g. ``compile_pdf``, ``enable_review``, ``output_dir``).

        - **Returns**:
            - `PaperGenerationResult`: The final generation result.
        """
        return await self._metadata_agent.generate_paper(
            metadata=metadata,
            **options,
        )

    async def generate_stream(
        self,
        metadata: Any,
        **options: Any,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming paper generation via async generator.

        Yields progress event dicts emitted by ``MetaDataAgent.generate_paper``
        through the ``progress_callback`` mechanism.  The final result is
        returned by ``generate()``; this method is for observing progress.

        - **Args**:
            - `metadata` (PaperMetaData): Paper input.
            - `**options`: Forwarded to ``MetaDataAgent.generate_paper()``.

        - **Yields**:
            - `Dict[str, Any]`: Progress event dicts (same schema as SSE events).
        """
        queue: asyncio.Queue[Dict[str, Any] | object] = asyncio.Queue()

        async def _callback(event: Dict[str, Any]) -> None:
            await queue.put(event)

        async def _run() -> None:
            try:
                await self._metadata_agent.generate_paper(
                    metadata=metadata,
                    progress_callback=_callback,
                    **options,
                )
            finally:
                await queue.put(_SENTINEL)

        task = asyncio.create_task(_run())

        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                yield item  # type: ignore[misc]
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    # ------------------------------------------------------------------
    # Internal wiring
    # ------------------------------------------------------------------

    @staticmethod
    def _build_agents(config: AppConfig) -> Dict[str, Any]:
        """
        Instantiate all agents from *config*, mirroring the server lifespan.

        - **Args**:
            - `config` (AppConfig): Parsed application config.

        - **Returns**:
            - `Dict[str, BaseAgent]`: Agent name -> instance mapping.
        """
        skill_registry = None
        skills_config = config.skills or SkillsConfig(enabled=False)
        if skills_config.enabled:
            try:
                from src.skills.loader import SkillLoader
                from src.skills.registry import SkillRegistry
                skill_registry = SkillRegistry()
                loader = SkillLoader()
                for skill in loader.load_directory(Path(skills_config.skills_dir)):
                    skill_registry.register(skill)
                logger.info("Skills: loaded %d skills", len(skill_registry))
            except Exception:
                logger.warning("Skills loading failed; continuing without skills")
                skill_registry = None

        agents = initialize_agents(
            config.agents,
            skill_registry=skill_registry,
            global_tools_config=config.tools,
            vlm_service_config=config.vlm_service,
        )

        if "metadata" not in agents:
            raise RuntimeError(
                "MetaDataAgent not found in config. "
                "Ensure an agent with name='metadata' is defined."
            )

        return agents
