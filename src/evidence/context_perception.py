"""
Context Perception Module
- **Description**:
    - Unified entry point for building an EvidenceDAG from all available sources.
    - Encapsulates CodeContextBuilder and research-context ingestion behind
      a single async interface.
    - Designed for extensibility: new evidence sources can be added by
      implementing additional ``_ingest_*`` methods in DAGBuilder.

- **Usage**::

    module = ContextPerceptionModule()
    dag = await module.build(
        metadata=paper_metadata,
        code_context=code_ctx,
        research_context=research_ctx,
        paper_plan=plan,
    )
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..models.evidence_graph import EvidenceDAG
from .dag_builder import DAGBuilder

logger = logging.getLogger(__name__)


class ContextPerceptionModule:
    """
    Unified facade that assembles an EvidenceDAG from heterogeneous sources.

    - **Description**:
        - Accepts pre-built contexts (code_context, research_context) or raw
          metadata and delegates to DAGBuilder.
        - Serves as the single call-site for the metadata agent, replacing
          scattered builder/formatter invocations with one coherent interface.
        - Provides a pluggable architecture: callers can supply any subset
          of sources and the module gracefully degrades.
    """

    def __init__(self, dag_builder: Optional[DAGBuilder] = None) -> None:
        self._dag_builder = dag_builder or DAGBuilder()

    async def build(
        self,
        metadata: Optional[Any] = None,
        code_context: Optional[Dict[str, Any]] = None,
        research_context: Optional[Dict[str, Any]] = None,
        paper_plan: Optional[Any] = None,
        figures: Optional[List[Any]] = None,
        tables: Optional[List[Any]] = None,
    ) -> EvidenceDAG:
        """
        Build a unified EvidenceDAG from all available evidence sources.

        - **Description**:
            - If ``figures`` / ``tables`` are not explicitly provided but
              ``metadata`` is, they are extracted from metadata automatically.
            - If ``metadata`` is provided without a ``paper_plan``, raw
              metadata fields are used as fallback claim sources.

        - **Args**:
            - `metadata`: PaperMetaData (or compatible object with .figures, .tables).
            - `code_context` (Dict): Output of CodeContextBuilder.build().
            - `research_context` (Dict): Unified research context from
              ``ResearchContextBuilder`` / ``prepare_plan`` (includes ``core_ref_analysis``).
            - `paper_plan`: PaperPlan for claim extraction.
            - `figures` (List): FigureSpec list (overrides metadata.figures if given).
            - `tables` (List): TableSpec list (overrides metadata.tables if given).

        - **Returns**:
            - `EvidenceDAG`: The fully populated graph.
        """
        if figures is None and metadata is not None:
            figures = getattr(metadata, "figures", []) or []
        if tables is None and metadata is not None:
            tables = getattr(metadata, "tables", []) or []

        metadata_dict: Optional[Dict[str, Any]] = None
        if metadata is not None and paper_plan is None:
            metadata_dict = {
                "idea_hypothesis": getattr(metadata, "idea_hypothesis", ""),
                "method": getattr(metadata, "method", ""),
                "data": getattr(metadata, "data", ""),
                "experiments": getattr(metadata, "experiments", ""),
            }

        dag = await self._dag_builder.build(
            code_context=code_context,
            research_context=research_context,
            figures=figures or [],
            tables=tables or [],
            paper_plan=paper_plan,
            metadata=metadata_dict,
        )

        logger.info(
            "ContextPerceptionModule.build complete: %s",
            dag.summary(),
        )
        return dag
