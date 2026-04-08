"""
MetaData Agent - Simple Mode Paper Generation
- **Description**:
    - Generates complete papers from simplified MetaData input
    - Multi-phase generation with persistent ReferencePool:
        0. Planning - core-ref analysis, landscape search, unified research context,
          then paper plan, section reference discovery, utility citation discovery,
          assignment, evidence DAG
        1. Introduction (Leader) - sets tone, extracts contributions
        2. Body Sections - Method, Experiment, Results, Related Work
        3. Synthesis Sections - Abstract and Conclusion from prior sections
        3.5. Review Loop - iterative feedback and revision
        4. PDF Compilation - via Typesetter Agent
    - Two-phase content generation pattern:
        - Phase A (Judgment + Search): LLM judges whether the section needs
          additional references; if yes, PaperSearchTool is called system-side
          and results are merged into the ReferencePool before writing.
        - Phase B (Pure Writing): LLM generates content with all available
          refs (core + discovered) in the prompt, no tools attached.
    - Fixed-sequence review:
        - Mini-review executes citation validation, word count, and key point
          checks in deterministic order (Type 2 tools).
    - ReferencePool accumulates references across all phases:
        user's core refs + discovered refs -> final .bib
    - Independent API, no frontend dependency
"""
import asyncio
import json
import mimetypes
import re
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from functools import partial
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path

import httpx
from pydantic import BaseModel
from fastapi import APIRouter

from .progress import ProgressEmitter, ProgressCallback, EventType, Phase
from ..shared.llm_client import (
    set_llm_progress_context,
    clear_llm_progress_context,
    set_usage_tracker_context,
    update_usage_tracker_context,
    clear_usage_tracker_context,
)
from ..shared.usage_tracker import UsageTracker
from ..react_base import ReActAgent
from ...config.schema import ModelConfig, ToolsConfig
from ..shared.reference_pool import ReferencePool
from ..shared.core_ref_analyzer import CoreRefAnalyzer
from ..shared.research_context_builder import ResearchContextBuilder
from .models import (
    PaperMetaData,
    PaperGenerationRequest,
    PaperGenerationResult,
    PlanResult,
    SectionResult,
    SectionGenerationRequest,
    BODY_SECTION_SOURCES,
    SYNTHESIS_SECTIONS,
    DEFAULT_SECTION_ORDER,
    CodeRepoOnError,
)
from ..shared.prompt_compiler import (
    compile_introduction_prompt,
    compile_body_section_prompt,
    compile_paragraph_prompt,
    compile_core_prompt,
    compile_citation_prompt,
    apply_citation_edits,
    compile_synthesis_prompt,
    extract_contributions_from_intro,
    SECTION_PROMPTS,
)
from ..shared.template_analyzer import TemplateAnalyzer, TemplateWriterGuide
from ..writer_agent.section_models import (
    ReferenceInfo,
    SimpleSectionInput,
    SynthesisSectionInput,
)
from ..planner_agent.models import (
    PaperPlan,
    SectionPlan,
    PlanRequest,
    calculate_total_words,
)
from ..shared.table_converter import convert_tables, inject_float_refs
from ..shared.session_memory import SessionMemory, ReviewRecord
from ..shared.code_context import (
    CodeContextBuilder,
    format_code_context_for_prompt,
    format_code_context_for_planner,
    render_code_repository_summary_markdown,
)
from ...evidence.dag_builder import DAGBuilder
from ...models.evidence_graph import EvidenceDAG
from ..reviewer_agent.models import (
    ReviewResult,
    FeedbackResult,
    Severity,
    SectionFeedback,
    RevisionTask,
    ConflictResolutionRecord,
    SemanticCheckRecord,
    HierarchicalFeedbackItem,
    FeedbackLevel,
)
from .models import FigureSpec, TableSpec, StructuralAction, SpaceEstimate
from .orchestrator import ReviewOrchestrator
from .revision_executor import RevisionExecutor
from .conflict_resolver import ConflictResolver, LATEX_ERROR_FIXES as _CR_LATEX_ERROR_FIXES
from ...prompts import PromptLoader as _PromptLoader

_prompt_loader = _PromptLoader()

# Original inline constant kept as fallback
_GENERATION_SYSTEM_PROMPT_DEFAULT = """\
You are an expert academic writer specializing in research paper composition.
Use present tense for methods, no contractions (it is, do not, cannot),
no possessives on method names (the performance of X, not X's performance).
Place key information at sentence end. Output pure LaTeX only.

CITATION RULES:
- Use ONLY the citation keys listed in the provided references.
- NEVER invent or hallucinate citation keys.
- If a claim needs a reference but none of the provided keys fit, omit the \\cite command.

OUTPUT FORMAT:
Return ONLY the LaTeX content for the section. Do not include explanations outside the LaTeX."""

GENERATION_SYSTEM_PROMPT = _prompt_loader.load(
    "metadata", "generation_system", default=_GENERATION_SYSTEM_PROMPT_DEFAULT
)

_SEARCH_JUDGMENT_PROMPT_DEFAULT = """\
You are an academic research assistant. Your task is to analyze whether the existing \
references are sufficient for writing a specific section of a research paper, or whether \
additional references should be searched.

Given the following section context, analyze whether additional references are needed.

SECTION: {section_type} ({section_title})
PAPER TITLE: {paper_title}
KEY POINTS TO COVER:
{key_points}

EXISTING REFERENCES ({n_refs} papers):
{ref_summaries}

RULES:
- If existing references adequately cover the section's claims, set need_search to false.
- If there are gaps (e.g., missing baselines, no empirical evidence for a claim, \
a sub-topic with zero refs), set need_search to true.
- Provide 1-3 focused search queries using specific keywords, not broad topics.
  Good: "R&D tax credit firm productivity panel data"
  Bad: "innovation economics"
- Each query should target a specific gap you identified.

Respond with ONLY a JSON object, no other text:
{{"need_search": true/false, "reason": "...", "queries": ["...", "..."]}}\
"""

SEARCH_JUDGMENT_PROMPT = _prompt_loader.load(
    "metadata", "search_judgment", default=_SEARCH_JUDGMENT_PROMPT_DEFAULT
)


class MetaDataAgent(ReActAgent):
    """
    MetaData Agent for simple-mode paper generation

    - **Description**:
        - Inherits from ReActAgent for access to react_loop and setup_tools.
        - Accepts 5 natural language fields + BibTeX references.
        - Manages a persistent ReferencePool throughout paper generation:
            - User's core references (~5 papers) initialize the pool.
            - During content generation, LLM may call search_papers (ReAct)
              to discover additional references.
            - Discovered papers undergo two-layer validation (LLM judgment +
              system cross-reference) before being added to the pool.
            - The pool's valid_citation_keys grows across phases.
        - Dual-mode tool invocation:
            - Type 1 (ReAct): _generate_introduction / _generate_body_section
              use react_loop with search_papers for autonomous reference search.
            - Type 2 (Delegated): WriterAgent handles iterative mini-review
              (citation validation, word count, key point coverage) internally.
        - Independent API, can be called directly via curl/Postman.
    """

    def __init__(self, config: ModelConfig, tools_config: Optional[ToolsConfig] = None):
        # Use default tools config if not provided
        if tools_config is None:
            tools_config = ToolsConfig(
                enabled=True,
                available_tools=[
                    "validate_citations",
                    "count_words",
                    "check_key_points",
                    "search_papers",
                ],
                max_react_iterations=3,
            )
        super().__init__(config, tools_config)
        self.results_dir = Path(__file__).parent.parent.parent.parent / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._router = self._create_router()
        except Exception:
            self._router = None
        # Skill registry — injected post-construction by agents/__init__.py
        self._skill_registry = None
        # Peer agent references — injected post-construction via set_peers()
        self._writer = None
        self._reviewer = None
        self._planner = None
        self._vlm_reviewer = None
        self._typesetter = None
        # Sub-modules — initialised here; they access peers lazily via self (host)
        self._orchestrator = ReviewOrchestrator(self)
        self._executor = RevisionExecutor(self)
        self._resolver = ConflictResolver(self)

    def set_peers(self, agents: Dict[str, "BaseAgent"]) -> None:
        """
        Inject references to peer agents for direct method calls.
        - **Description**:
            - Called after all agents are initialized in initialize_agents().
            - Enables MetaDataAgent to call WriterAgent, ReviewerAgent, etc.
              directly instead of via HTTP.

        - **Args**:
            - `agents` (Dict[str, BaseAgent]): The full agent dictionary.
        """
        self._writer = agents.get("writer")
        self._reviewer = agents.get("reviewer")
        self._planner = agents.get("planner")
        self._vlm_reviewer = agents.get("vlm_review")
        self._typesetter = agents.get("typesetter")

    @staticmethod
    async def _save_artifact(
        paper_dir: Path,
        relative_path: str,
        content: Any,
        emitter: "ProgressEmitter",
        category: str,
        label: str = "",
        artifacts_prefix: str = "",
    ) -> Path:
        """
        Write an artifact file, optionally upload to OSS, and emit artifact_saved.
        """
        target = paper_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, bytes):
            target.write_bytes(content)
        elif isinstance(content, str):
            target.write_text(content, encoding="utf-8")
        else:
            target.write_text(
                json.dumps(content, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        mime, _ = mimetypes.guess_type(str(target))
        if mime is None:
            ext = target.suffix.lower()
            mime = {
                ".json": "application/json",
                ".tex": "text/x-latex",
                ".bib": "text/x-bibtex",
                ".md": "text/markdown",
                ".bst": "text/plain",
                ".cls": "text/plain",
            }.get(ext, "application/octet-stream")

        size = target.stat().st_size

        storage_key = ""
        if artifacts_prefix:
            candidate_key = f"{artifacts_prefix}/{relative_path}"
            from ...utils.storage_client import storage_client
            if await storage_client.upload(candidate_key, target.read_bytes()):
                storage_key = candidate_key

        await emitter.artifact_saved(
            relative_path=relative_path,
            absolute_path=str(target),
            category=category,
            size=size,
            mime_type=mime,
            label=label or target.name,
            storage_key=storage_key,
        )
        return target

    _COMPILE_EXTS = {
        ".tex", ".bib", ".bst", ".cls", ".sty", ".bbl",
        ".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg", ".gif",
    }

    @staticmethod
    async def _save_compilation_output(
        compile_dir: Path,
        paper_dir: Path,
        emitter: "ProgressEmitter",
        artifacts_prefix: str = "",
    ) -> None:
        """
        Walk a typesetter output directory and emit artifact_saved for every
        file that is needed for a self-contained LaTeX compilation.
        """
        if not compile_dir or not compile_dir.is_dir():
            return

        _cat_map = {
            ".tex": ("section", "Section"),
            ".bib": ("references", "BibTeX"),
            ".bst": ("template", "BibTeX Style"),
            ".cls": ("template", "Document Class"),
            ".sty": ("template", "Style Package"),
            ".bbl": ("build", "Compiled Bibliography"),
            ".pdf": ("output", "PDF"),
            ".png": ("figure", "Figure"),
            ".jpg": ("figure", "Figure"),
            ".jpeg": ("figure", "Figure"),
            ".eps": ("figure", "Figure"),
            ".svg": ("figure", "Figure"),
            ".gif": ("figure", "Figure"),
        }

        from functools import partial as _partial

        _sa = _partial(
            MetaDataAgent._save_artifact,
            artifacts_prefix=artifacts_prefix,
        )

        for fpath in sorted(compile_dir.rglob("*")):
            if not fpath.is_file():
                continue
            ext = fpath.suffix.lower()
            if ext not in MetaDataAgent._COMPILE_EXTS:
                continue
            rel = str(fpath.relative_to(paper_dir))
            cat, lbl = _cat_map.get(ext, ("other", fpath.name))
            if fpath.name == "main.pdf":
                cat, lbl = "output", "Paper PDF"
            elif fpath.name == "main.tex":
                cat, lbl = "root", "Main LaTeX"
            elif fpath.name.endswith(".bib") and "reference" in fpath.name.lower():
                cat, lbl = "references", "References BibTeX"

            content = fpath.read_bytes()
            await _sa(paper_dir, rel, content, emitter, cat, lbl)

    @property
    def name(self) -> str:
        """Agent name identifier"""
        return "metadata"
    
    @property
    def description(self) -> str:
        """Agent description"""
        return "MetaData-based paper generation (Simple Mode) - generates complete papers from 5 natural language fields + BibTeX references"
    
    @property
    def router(self) -> APIRouter:
        """Return the FastAPI router for this agent"""
        return self._router
    
    @property
    def endpoints_info(self) -> List[Dict[str, Any]]:
        """Return endpoint metadata for list_agents"""
        return [
            {
                "path": "/metadata/generate",
                "method": "POST",
                "description": "Generate complete paper from MetaData (5 fields + references)",
            },
            {
                "path": "/metadata/generate/section",
                "method": "POST",
                "description": "Generate a single section (for debugging or incremental generation)",
            },
            {
                "path": "/metadata/health",
                "method": "GET",
                "description": "Health check endpoint",
            },
            {
                "path": "/metadata/schema",
                "method": "GET",
                "description": "Get input schema for paper generation",
            },
        ]
    
    def _create_router(self) -> APIRouter:
        """Create FastAPI router for this agent"""
        from .router import create_metadata_router
        return create_metadata_router(self)

    def _get_active_skills(self, section_type: str, style_guide: str = None):
        """
        Retrieve active writing skills from the skill registry.

        - **Args**:
            - `section_type` (str): Current section being generated
            - `style_guide` (str, optional): Venue name for venue-profile matching

        - **Returns**:
            - `list` of WritingSkill or None
        """
        if self._skill_registry is None or len(self._skill_registry) == 0:
            return None
        # Runtime decision: select skills using the current request's style_guide.
        # This avoids binding venue profiles at service startup time.
        return self._skill_registry.get_writing_skills(
            section_type=section_type,
            venue=style_guide,
        )

    async def _build_code_repository_context(
        self,
        metadata: PaperMetaData,
    ) -> Optional[Dict[str, Any]]:
        """
        Build code repository context for section-aware writing.
        - **Description**:
            - Resolves and scans optional code repository input.
            - Produces repository overview, section evidence packs, and index.

        - **Args**:
            - `metadata` (PaperMetaData): User metadata payload.

        - **Returns**:
            - `Dict[str, Any] | None`: Built context or None when not provided.
        """
        if not metadata.code_repository:
            return None

        builder = CodeContextBuilder(workspace_root=str(Path.cwd()))
        return await builder.build(
            code_repo=metadata.code_repository,
            paper_title=metadata.title,
        )

    def _retrieve_runtime_code_evidence(
        self,
        code_context: Optional[Dict[str, Any]],
        section_type: str,
        metadata: PaperMetaData,
        contributions: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve section-specific runtime evidence as fallback.
        - **Description**:
            - Uses lightweight query bundles per section.
            - Runs before writing to enrich prompt context.

        - **Args**:
            - `code_context` (Dict | None): Built repository context.
            - `section_type` (str): Target section.
            - `metadata` (PaperMetaData): User metadata.
            - `contributions` (List[str], optional): Known contributions.
            - `top_k` (int): Max snippets to return.

        - **Returns**:
            - `List[Dict]`: Retrieved evidence list.
        """
        if not code_context:
            return []

        query_bundle: List[str] = []
        if section_type == "method":
            query_bundle = ["algorithm", "model", "module", "implementation", "forward", "pipeline"]
            query_bundle.extend(metadata.method.split()[:12])
        elif section_type == "experiment":
            query_bundle = ["train", "eval", "dataset", "metric", "configuration", "ablation"]
            query_bundle.extend(metadata.data.split()[:12])
            query_bundle.extend(metadata.experiments.split()[:12])
        elif section_type == "result":
            query_bundle = ["result", "analysis", "benchmark", "compare", "metric", "report"]
            query_bundle.extend(metadata.experiments.split()[:12])
        else:
            query_bundle = ["project", "workflow", "pipeline"]
            query_bundle.extend(metadata.idea_hypothesis.split()[:10])

        if contributions:
            query_bundle.extend(" ".join(contributions[:2]).split()[:8])

        builder = CodeContextBuilder(workspace_root=str(Path.cwd()))
        return builder.retrieve_for_section(
            context=code_context,
            section_type=section_type,
            query_bundle=query_bundle,
            top_k=top_k,
        )

    def _format_research_context_for_prompt(
        self,
        research_context: Optional[Dict[str, Any]],
        section_type: str,
        evidence_dag: Optional[EvidenceDAG] = None,
    ) -> str:
        """
        Format research context into a compact writer-consumable brief.

        - **Description**:
            - When an EvidenceDAG is available, appends a structured
              claim-evidence binding section that shows exactly which
              evidence nodes support each claim in this section.
        """
        if not research_context and not evidence_dag:
            return ""

        lines: List[str] = ["## Research Context Brief"]

        if research_context:
            area = str(research_context.get("research_area", "")).strip()
            summary = str(research_context.get("summary", "")).strip()
            if area:
                lines.append(f"- Area: {area}")
            if summary:
                lines.append(f"- Landscape: {summary}")

            trends = research_context.get("research_trends", []) or []
            if trends:
                lines.append("- Trends:")
                for t in trends[:3]:
                    lines.append(f"  - {t}")

            gaps = research_context.get("gaps", []) or []
            if gaps:
                lines.append("- Gaps to address:")
                for g in gaps[:3]:
                    lines.append(f"  - {g}")

            key_papers = research_context.get("key_papers", []) or []
            if key_papers:
                lines.append("- Key papers and why they matter:")
                for kp in key_papers[:5]:
                    title = kp.get("title", "")
                    contribution = kp.get("contribution", "")
                    if title:
                        lines.append(f"  - {title}: {contribution}")

            cra = research_context.get("core_ref_analysis")
            if isinstance(cra, dict):
                cra_items = cra.get("items") or []
                if cra_items:
                    lines.append("- Core references (user anchors):")
                    for it in cra_items[:6]:
                        if not isinstance(it, dict):
                            continue
                        rid = it.get("ref_id", "")
                        tit = it.get("title", "")
                        rel = str(it.get("relationship_to_ours", ""))[:220]
                        if tit or rid:
                            lines.append(f"  - [{rid}] {tit}: {rel}")
                    pos = str(cra.get("positioning_statement", "")).strip()
                    if pos:
                        lines.append(f"- Positioning vs core refs: {pos[:400]}")

            claim_matrix = research_context.get("claim_evidence_matrix", []) or []
            section_claims = [
                c for c in claim_matrix
                if c.get("section_type") in {section_type, "global", "", None}
            ]
            if section_claims and evidence_dag is None:
                lines.append("- Claim-evidence priorities:")
                for c in section_claims[:6]:
                    claim = c.get("claim", "")
                    refs = c.get("support_refs", []) or []
                    priority = c.get("priority", "")
                    reason = c.get("reason", "")
                    ref_text = ", ".join(refs[:4]) if refs else "none"
                    lines.append(
                        f"  - [{priority}] Claim: {claim} | Evidence refs: {ref_text} | Why: {reason}"
                    )

            ranking = research_context.get("contribution_ranking", {}) or {}
            if ranking:
                lines.append("- Contribution ranking:")
                for band in ("P0", "P1", "P2"):
                    items = ranking.get(band, []) or []
                    if not items:
                        continue
                    for item in items[:3]:
                        contribution = item.get("contribution", "")
                        why = item.get("why_it_matters", "")
                        sections = item.get("suggested_sections", []) or []
                        section_hint = ", ".join(sections[:3]) if sections else "n/a"
                        lines.append(
                            f"  - {band}: {contribution} | Why: {why} | Suggested sections: {section_hint}"
                        )

        # --- DAG-based claim-evidence bindings (replaces raw matrix when available) ---
        if evidence_dag is not None:
            section_claims = evidence_dag.get_claims_for_section(section_type)
            if section_claims:
                lines.append("")
                lines.append("## Claim-Evidence Bindings (from Evidence DAG)")
                lines.append(
                    "(Each claim MUST be supported ONLY by its bound evidence. "
                    "Do NOT introduce unsupported claims.)"
                )
                for claim in section_claims[:10]:
                    ev_nodes = evidence_dag.get_evidence_for_claim(claim.node_id)
                    ev_desc_parts: List[str] = []
                    for ev in ev_nodes[:5]:
                        label = f"{ev.node_id}({ev.node_type.value})"
                        if ev.source_path:
                            label += f"[{ev.source_path}]"
                        ev_desc_parts.append(label)
                    ev_text = ", ".join(ev_desc_parts) if ev_desc_parts else "UNSUPPORTED"
                    lines.append(
                        f"  - [{claim.priority}] {claim.node_id}: "
                        f"{claim.statement[:200]} "
                        f"| Bound evidence: {ev_text}"
                    )

        if len(lines) <= 1:
            return ""
        return "\n".join(lines)

    def _collect_section_citation_budget_usage(
        self,
        *,
        section_type: str,
        content: str,
        section_plan: Optional[SectionPlan],
        writer_valid_keys: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Collect soft-cap citation usage stats for one section.
        """
        if not section_plan:
            return None

        valid_set = set(writer_valid_keys or [])
        _, _, used_keys = self._validate_and_fix_citations(
            content,
            valid_set,
            remove_invalid=False,
        )
        selected_refs = list(
            section_plan.budget_selected_refs
            or section_plan.assigned_refs
            or []
        )
        selected_set = set(selected_refs)
        used_budget_keys = [k for k in used_keys if k in selected_set]
        overflow_keys = [k for k in used_keys if k not in selected_set]
        budget = section_plan.citation_budget or {}
        return {
            "section_type": section_type,
            "min_refs": budget.get("min_refs"),
            "target_refs": budget.get("target_refs"),
            "max_refs": budget.get("max_refs"),
            "selected_refs": selected_refs,
            "reserve_refs": list(section_plan.budget_reserve_refs or []),
            "writer_valid_keys_count": len(valid_set),
            "used_keys": used_keys,
            "used_count": len(used_keys),
            "used_budget_keys": used_budget_keys,
            "used_budget_count": len(used_budget_keys),
            "overflow_keys": overflow_keys,
            "overflow_count": len(overflow_keys),
        }

    @staticmethod
    def _upsert_section_budget_usage(
        usage_rows: List[Dict[str, Any]],
        usage_row: Optional[Dict[str, Any]],
    ) -> None:
        """
        Upserts one section citation usage row by section_type.
        - **Description**:
         - Updates existing section stats if present, otherwise appends a new row.
         - Ensures one section appears once in exported citation_budget_usage.

        - **Args**:
         - `usage_rows` (List[Dict[str, Any]]): Collected usage rows.
         - `usage_row` (Optional[Dict[str, Any]]): New usage row to merge.

        - **Returns**:
         - `None` (None): In-place update only.
        """
        if not usage_row:
            return
        section_type = usage_row.get("section_type")
        if not section_type:
            return
        for idx, row in enumerate(usage_rows):
            if row.get("section_type") == section_type:
                usage_rows[idx] = usage_row
                return
        usage_rows.append(usage_row)

    @staticmethod
    def _build_citation_plan_alignment_stats(
        paper_plan: Optional[PaperPlan],
        usage_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Builds citation alignment stats between plan and final manuscript.
        - **Description**:
         - Compares planned citation budget and selected keys with actual used keys.
         - Produces per-section and overall metrics for citation usage auditing.

        - **Args**:
         - `paper_plan` (Optional[PaperPlan]): Planner output containing citation budgets.
         - `usage_rows` (List[Dict[str, Any]]): Real usage extracted from final section content.

        - **Returns**:
         - `stats` (Dict[str, Any]): Aggregated alignment report payload.
        """
        usage_by_section: Dict[str, Dict[str, Any]] = {
            str(r.get("section_type")): r
            for r in usage_rows
            if r.get("section_type")
        }

        per_section: List[Dict[str, Any]] = []
        total_selected = 0
        total_used = 0
        total_used_budget = 0
        total_overflow = 0
        total_missing = 0
        sections_with_budget = 0
        sections_meeting_min = 0

        for sp in (paper_plan.sections if paper_plan else []):
            budget = sp.citation_budget or {}
            selected_refs = list(sp.budget_selected_refs or sp.assigned_refs or [])
            selected_set = set(selected_refs)
            usage = usage_by_section.get(sp.section_type, {})
            used_keys = list(usage.get("used_keys", []) or [])
            used_budget_keys = list(usage.get("used_budget_keys", []) or [])
            overflow_keys = list(usage.get("overflow_keys", []) or [])
            missing_selected_refs = [k for k in selected_refs if k not in set(used_budget_keys)]

            min_refs = int(budget.get("min_refs") or 0)
            target_refs = int(budget.get("target_refs") or 0)
            max_refs = int(budget.get("max_refs") or 0)

            used_count = len(used_keys)
            used_budget_count = len(used_budget_keys)
            overflow_count = len(overflow_keys)
            selected_count = len(selected_refs)
            coverage_rate = (
                round(used_budget_count / selected_count, 4) if selected_count > 0 else None
            )
            min_met = (used_count >= min_refs) if min_refs > 0 else True
            target_met = (used_count >= target_refs) if target_refs > 0 else True
            max_exceeded = (used_count > max_refs) if max_refs > 0 else False

            if selected_count > 0:
                sections_with_budget += 1
            if min_met:
                sections_meeting_min += 1

            total_selected += selected_count
            total_used += used_count
            total_used_budget += used_budget_count
            total_overflow += overflow_count
            total_missing += len(missing_selected_refs)

            per_section.append(
                {
                    "section_type": sp.section_type,
                    "plan": {
                        "min_refs": min_refs,
                        "target_refs": target_refs,
                        "max_refs": max_refs,
                        "selected_count": selected_count,
                        "selected_refs": selected_refs,
                    },
                    "final": {
                        "used_count": used_count,
                        "used_keys": used_keys,
                        "used_budget_count": used_budget_count,
                        "used_budget_keys": used_budget_keys,
                        "overflow_count": overflow_count,
                        "overflow_keys": overflow_keys,
                    },
                    "delta": {
                        "coverage_rate": coverage_rate,
                        "missing_selected_count": len(missing_selected_refs),
                        "missing_selected_refs": missing_selected_refs,
                        "extra_non_budget_count": overflow_count,
                    },
                    "status": {
                        "min_met": min_met,
                        "target_met": target_met,
                        "max_exceeded": max_exceeded,
                    },
                }
            )

        overall = {
            "sections_total": len(per_section),
            "sections_with_budget": sections_with_budget,
            "sections_meeting_min": sections_meeting_min,
            "total_selected_refs": total_selected,
            "total_used_refs": total_used,
            "total_used_budget_refs": total_used_budget,
            "total_overflow_refs": total_overflow,
            "total_missing_selected_refs": total_missing,
            "overall_budget_coverage_rate": (
                round(total_used_budget / total_selected, 4) if total_selected > 0 else None
            ),
        }

        return {
            "overall": overall,
            "sections": per_section,
        }

    def _build_structure_alignment_stats(
        self,
        *,
        paper_plan: Optional[PaperPlan],
        generated_sections: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Build plan-vs-final structure alignment statistics.
        """
        threshold = int(
            getattr(self.tools_config, "structure_gate_min_paragraph_threshold", 5)
        ) if self.tools_config else 5
        per_section: List[Dict[str, Any]] = []
        gate_expected_count = 0
        gate_pass_count = 0

        for sp in (paper_plan.sections if paper_plan else []):
            final_content = generated_sections.get(sp.section_type, "") or ""
            paragraphs = [
                p.strip() for p in re.split(r"\n\s*\n", final_content) if p.strip()
            ]
            paragraph_count = len(paragraphs)
            subsection_count = len(
                re.findall(r"\\subsection\{.+?\}|\\subsubsection\{.+?\}", final_content)
            )
            transition_count = 0
            for para in paragraphs[1:]:
                lower = para.lower()
                if lower.startswith((
                    "however", "therefore", "in contrast", "meanwhile",
                    "moreover", "furthermore", "additionally", "by contrast",
                    "in summary",
                )):
                    transition_count += 1

            expected_gate = bool(
                sp.sectioning_recommended
                or paragraph_count >= threshold
            ) and sp.section_type not in {"abstract", "conclusion"}
            passed_gate = bool(
                subsection_count > 0
                or (paragraph_count >= 4 and transition_count >= 1)
            )
            if expected_gate:
                gate_expected_count += 1
                if passed_gate:
                    gate_pass_count += 1

            per_section.append(
                {
                    "section_type": sp.section_type,
                    "plan": {
                        "paragraph_count": len(sp.paragraphs or []),
                        "topic_clusters": list(sp.topic_clusters or []),
                        "transition_intents": list(sp.transition_intents or []),
                        "sectioning_recommended": bool(sp.sectioning_recommended),
                    },
                    "final": {
                        "paragraph_count": paragraph_count,
                        "explicit_subsection_count": subsection_count,
                        "transition_marker_count": transition_count,
                    },
                    "status": {
                        "structure_gate_expected": expected_gate,
                        "structure_gate_passed": passed_gate if expected_gate else None,
                    },
                }
            )

        return {
            "overall": {
                "sections_total": len(per_section),
                "gate_expected_sections": gate_expected_count,
                "gate_passed_sections": gate_pass_count,
                "gate_pass_rate": (
                    round(gate_pass_count / gate_expected_count, 4)
                    if gate_expected_count > 0
                    else None
                ),
            },
            "sections": per_section,
        }

    def _build_paragraph_feedback_alignment_report(
        self,
        *,
        memory: Optional[SessionMemory],
        generated_sections: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Build paragraph feedback alignment report against final section paragraphs.
        """
        if memory is None or not getattr(memory, "review_history", None):
            return {"sections": [], "overall": {"records": 0}}

        latest = memory.review_history[-1]
        rows: List[Dict[str, Any]] = []
        total_targets = 0
        total_mapped = 0
        total_out_of_range = 0

        for section_type, feedback in (latest.section_feedbacks or {}).items():
            if not isinstance(feedback, dict):
                continue
            target_paragraphs = [
                int(x) for x in (feedback.get("target_paragraphs", []) or [])
                if str(x).strip().lstrip("-").isdigit()
            ]
            final_paragraphs = self._executor._split_section_paragraphs(
                generated_sections.get(section_type, "") or ""
            )
            final_count = len(final_paragraphs)
            mapped = []
            out_of_range = []
            for pidx in target_paragraphs:
                if 0 <= pidx < final_count:
                    mapped.append({"from": pidx, "to": [pidx], "strategy": "identity"})
                elif final_count > 0:
                    nearest = min(max(pidx, 0), final_count - 1)
                    mapped.append({"from": pidx, "to": [nearest], "strategy": "clamped_nearest"})
                    out_of_range.append(pidx)
                else:
                    mapped.append({"from": pidx, "to": [], "strategy": "no_paragraphs"})
                    out_of_range.append(pidx)

            total_targets += len(target_paragraphs)
            total_mapped += len([m for m in mapped if m.get("to")])
            total_out_of_range += len(out_of_range)

            rows.append(
                {
                    "section_type": section_type,
                    "target_paragraphs": target_paragraphs,
                    "final_paragraph_count": final_count,
                    "mappings": mapped,
                    "out_of_range_targets": out_of_range,
                }
            )

        return {
            "overall": {
                "records": len(rows),
                "total_targets": total_targets,
                "mapped_targets": total_mapped,
                "out_of_range_targets": total_out_of_range,
            },
            "sections": rows,
        }

    def _rebuild_citation_budget_usage_from_final_sections(
        self,
        *,
        paper_plan: Optional[PaperPlan],
        generated_sections: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Rebuild citation budget usage from final post-review section contents.
        - **Description**:
         - Extracts citation usage from final manuscript sections after review iterations.
         - Ensures exported usage/alignment reflects the final delivered content.

        - **Args**:
         - `paper_plan` (Optional[PaperPlan]): Planner output containing section plans.
         - `generated_sections` (Dict[str, str]): Final section content map.

        - **Returns**:
         - `usage_rows` (List[Dict[str, Any]]): Final per-section citation usage rows.
        """
        if not paper_plan:
            return []

        usage_rows: List[Dict[str, Any]] = []
        for sp in paper_plan.sections:
            final_content = generated_sections.get(sp.section_type, "")
            writer_valid_keys = list(
                dict.fromkeys(
                    list(sp.assigned_refs or [])
                    + list(sp.budget_reserve_refs or [])
                )
            )
            usage_row = self._collect_section_citation_budget_usage(
                section_type=sp.section_type,
                content=final_content,
                section_plan=sp,
                writer_valid_keys=writer_valid_keys,
            )
            self._upsert_section_budget_usage(usage_rows, usage_row)
        return usage_rows

    def _build_reviewer_acceptance_stats(
        self,
        *,
        memory: Optional[SessionMemory],
    ) -> Dict[str, Any]:
        """
        Build acceptance statistics from reviewer verification records.
        """
        if memory is None or not getattr(memory, "review_history", None):
            return {"overall": {"total": 0}, "by_iteration": []}

        by_iteration: List[Dict[str, Any]] = []
        total = 0
        passed = 0
        failed = 0
        noop_accepted = 0
        changed_accepted = 0

        for rec in memory.review_history:
            verifications = list(getattr(rec, "reviewer_verification", []) or [])
            iter_total = len(verifications)
            iter_passed = 0
            iter_failed = 0
            iter_noop_accepted = 0
            iter_changed_accepted = 0
            for v in verifications:
                if not isinstance(v, dict):
                    continue
                ok = bool(v.get("passed", False))
                changed_flag = bool(v.get("changed", False))
                if ok:
                    iter_passed += 1
                    if changed_flag:
                        iter_changed_accepted += 1
                    else:
                        iter_noop_accepted += 1
                else:
                    iter_failed += 1

            total += iter_total
            passed += iter_passed
            failed += iter_failed
            noop_accepted += iter_noop_accepted
            changed_accepted += iter_changed_accepted
            by_iteration.append(
                {
                    "iteration": int(getattr(rec, "iteration", 0)),
                    "total": iter_total,
                    "passed": iter_passed,
                    "failed": iter_failed,
                    "noop_accepted": iter_noop_accepted,
                    "changed_accepted": iter_changed_accepted,
                }
            )

        return {
            "overall": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": round(passed / total, 4) if total > 0 else None,
                "noop_accepted": noop_accepted,
                "changed_accepted": changed_accepted,
            },
            "by_iteration": by_iteration,
        }

    def _build_citation_repair_stats(
        self,
        *,
        memory: Optional[SessionMemory],
    ) -> Dict[str, Any]:
        """
        Build invalid-citation removal statistics from decision traces.
        """
        if memory is None or not getattr(memory, "review_history", None):
            return {"overall": {"removed_total": 0}, "by_section": {}, "events": []}

        by_section: Dict[str, int] = {}
        events: List[Dict[str, Any]] = []
        removed_total = 0
        for rec in memory.review_history:
            trace_rows = list(getattr(rec, "decision_trace", []) or [])
            for row in trace_rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("decision", "")) != "removed_invalid_citations":
                    continue
                section_type = str(row.get("section_type", "unknown"))
                count = int(row.get("count", 0) or 0)
                removed_total += count
                by_section[section_type] = by_section.get(section_type, 0) + count
                events.append(
                    {
                        "iteration": int(getattr(rec, "iteration", 0)),
                        "section_type": section_type,
                        "count": count,
                        "keys": list(row.get("keys", []) or []),
                    }
                )
        return {
            "overall": {
                "removed_total": removed_total,
                "events": len(events),
            },
            "by_section": by_section,
            "events": events,
        }

    def _build_explicit_subsection_coverage(
        self,
        *,
        paper_plan: Optional[PaperPlan],
        generated_sections: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Build explicit subsection coverage for recommended sections.
        """
        rows: List[Dict[str, Any]] = []
        recommended_total = 0
        recommended_with_explicit = 0
        for sp in (paper_plan.sections if paper_plan else []):
            if sp.section_type in {"abstract", "conclusion"}:
                continue
            if not bool(sp.sectioning_recommended):
                continue
            recommended_total += 1
            content = generated_sections.get(sp.section_type, "") or ""
            explicit_count = len(
                re.findall(r"\\subsection\{.+?\}|\\subsubsection\{.+?\}", content)
            )
            has_explicit = explicit_count > 0
            if has_explicit:
                recommended_with_explicit += 1
            rows.append(
                {
                    "section_type": sp.section_type,
                    "section_title": sp.section_title,
                    "explicit_subsection_count": explicit_count,
                    "has_explicit_subsection": has_explicit,
                }
            )
        return {
            "overall": {
                "recommended_sections": recommended_total,
                "recommended_with_explicit_subsection": recommended_with_explicit,
                "coverage_rate": (
                    round(recommended_with_explicit / recommended_total, 4)
                    if recommended_total > 0
                    else None
                ),
            },
            "sections": rows,
        }

    # ------------------------------------------------------------------
    # prepare_plan: Phase 0 only — returns a serializable PlanResult
    # ------------------------------------------------------------------

    async def prepare_plan(
        self,
        metadata: PaperMetaData,
        template_path: Optional[str] = None,
        target_pages: Optional[int] = None,
        enable_planning: bool = True,
        enable_exemplar: bool = False,
        save_output: bool = True,
        output_dir: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
        artifacts_prefix: str = "",
    ) -> PlanResult:
        """
        Execute planning phases (0 through 0.5) and return a serializable snapshot.
        - **Description**:
            - Runs template resolution, reference pool init, file validation,
              figure conversion, planning, DAG construction, code context,
              and table conversion.
            - Does NOT start content generation.
            - The returned PlanResult can be sent to the frontend for review,
              optionally modified, then passed to ``execute_generation()``.

        - **Args**:
            - `metadata` (PaperMetaData): Paper metadata with 5 fields + references.
            - `template_path` (str, optional): Path to .zip template file.
            - `target_pages` (int, optional): Target page count.
            - `enable_planning` (bool): Whether to create a paper plan.
            - `save_output` (bool): Whether to save intermediate files.
            - `output_dir` (str, optional): Directory for output files.
            - `progress_callback` (ProgressCallback, optional): SSE callback.
            - `artifacts_prefix` (str): Storage prefix for artifacts.

        - **Returns**:
            - `PlanResult`: Serializable planning snapshot.
        """
        if template_path is None:
            template_path = metadata.template_path
        if not template_path and not metadata.template_path:
            try:
                from src.default_templates import resolve_default_template
                resolved = resolve_default_template(metadata.style_guide)
                if resolved:
                    template_path = resolved
            except ImportError:
                pass
        if target_pages is None:
            target_pages = metadata.target_pages

        emitter = ProgressEmitter(callback=progress_callback)
        set_llm_progress_context(emitter, agent="MetaDataAgent")
        await emitter.generation_started(title=metadata.title, target_pages=target_pages)

        errors: List[str] = []
        paper_plan: Optional[PaperPlan] = None
        research_context: Optional[Dict[str, Any]] = None
        code_context: Optional[Dict[str, Any]] = None
        code_summary_markdown: Optional[str] = None
        evidence_dag: Optional[EvidenceDAG] = None
        docling_temp_dir: Optional[Path] = None
        docling_cfg = (
            self.tools_config.docling
            if self.tools_config and getattr(self.tools_config, "docling", None)
            else None
        )

        search_cfg_for_pool = {}
        if self.tools_config and self.tools_config.paper_search:
            ps = self.tools_config.paper_search
            search_cfg_for_pool = {
                "serpapi_api_key": ps.serpapi_api_key,
                "semantic_scholar_api_key": ps.semantic_scholar_api_key,
                "timeout": ps.timeout,
                "semantic_scholar_min_results_before_fallback": ps.semantic_scholar_min_results_before_fallback,
                "enable_query_cache": ps.enable_query_cache,
                "cache_ttl_hours": ps.cache_ttl_hours,
            }
        ref_pool = await ReferencePool.create(
            metadata.references, paper_search_config=search_cfg_for_pool,
        )
        print(f"[MetaDataAgent] Reference pool initialized: {ref_pool.summary()}")

        validation_errors = self._validate_file_paths(metadata)
        if validation_errors:
            return PlanResult(
                paper_plan={},
                metadata_input=metadata.model_dump(),
                errors=validation_errors,
                template_path=template_path,
                target_pages=target_pages,
                artifacts_prefix=artifacts_prefix,
                ref_pool_snapshot=ref_pool.to_dict(),
            )

        if metadata.figures:
            n_converted = self._convert_figures_for_latex(metadata)
            if n_converted:
                print(f"[MetaDataAgent] Converted {n_converted} figure(s) to LaTeX-compatible format")

        paper_dir_str: Optional[str] = None
        if save_output:
            if output_dir:
                paper_dir = Path(output_dir)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_title = re.sub(r'[^\w\-]', '_', metadata.title)[:50]
                paper_dir = self.results_dir / f"{safe_title}_{timestamp}"
            paper_dir.mkdir(parents=True, exist_ok=True)
            paper_dir_str = str(paper_dir)
        else:
            paper_dir = None

        try:
            # Phase 0e-pre: Code repository context (optional, pre-plan)
            if metadata.code_repository:
                print("[MetaDataAgent] Phase 0e-pre: Building code repository context...")
                await emitter.phase_start(Phase.CODE_CONTEXT, "Building code repository context")
                try:
                    code_context = await self._build_code_repository_context(metadata)
                    if code_context:
                        code_summary_markdown = render_code_repository_summary_markdown(code_context)
                except Exception as e:
                    on_error = metadata.code_repository.on_error
                    msg = f"Code repository ingestion failed: {e}"
                    print(f"[MetaDataAgent] Warning: {msg}")
                    if (
                        metadata.code_repository.type.value == "local_dir"
                        and isinstance(e, (FileNotFoundError, ValueError))
                    ):
                        return PlanResult(
                            paper_plan={}, metadata_input=metadata.model_dump(),
                            errors=[msg], ref_pool_snapshot=ref_pool.to_dict(),
                            template_path=template_path, target_pages=target_pages,
                            artifacts_prefix=artifacts_prefix, paper_dir=paper_dir_str,
                        )
                    if on_error == CodeRepoOnError.STRICT:
                        return PlanResult(
                            paper_plan={}, metadata_input=metadata.model_dump(),
                            errors=[msg], ref_pool_snapshot=ref_pool.to_dict(),
                            template_path=template_path, target_pages=target_pages,
                            artifacts_prefix=artifacts_prefix, paper_dir=paper_dir_str,
                        )
                    errors.append(msg)
                    code_context = None
                    code_summary_markdown = None

            # Phase 0-docling: Deep reference analysis via Docling (optional)
            if docling_cfg and docling_cfg.enabled:
                print("[MetaDataAgent] Phase 0-docling: Deep reference analysis with Docling...")
                try:
                    from ..shared.docling_service import DoclingService

                    docling_temp_dir = (paper_dir or Path(tempfile.mkdtemp())) / "_docling_tmp"
                    docling_svc = DoclingService(config=docling_cfg)
                    ref_pool._core_refs = await docling_svc.enrich_refs(
                        ref_pool._core_refs,
                        dest_dir=docling_temp_dir,
                        cleanup=False,
                    )
                    docling_count = sum(
                        1 for r in ref_pool._core_refs if r.get("docling_sections")
                    )
                    print(
                        f"[MetaDataAgent] Docling enriched {docling_count} / "
                        f"{len(ref_pool._core_refs)} core references"
                    )
                except ImportError:
                    print(
                        "[MetaDataAgent] Warning: Docling not installed. "
                        "Install with: pip install easypaper[docling]"
                    )
                    errors.append("Docling enabled but not installed")
                except Exception as e:
                    print(f"[MetaDataAgent] Warning: Docling enrichment failed: {e}")
                    errors.append(f"Docling enrichment failed: {e}")

            # Phase 0-exemplar: Exemplar paper selection and analysis (optional)
            exemplar_analysis_dict: Optional[Dict[str, Any]] = None
            exemplar_cfg = (
                self.tools_config.exemplar
                if self.tools_config and getattr(self.tools_config, "exemplar", None)
                else None
            )
            if enable_exemplar and exemplar_cfg and exemplar_cfg.enabled:
                print("[MetaDataAgent] Phase 0-exemplar: Selecting and analyzing exemplar paper...")
                try:
                    from ..shared.exemplar_selector import ExemplarSelector
                    from ..shared.exemplar_analyzer import ExemplarAnalyzer
                    from ..shared.docling_service import DoclingService as _DocSvc

                    if metadata.exemplar_paper_path:
                        _dsvc = _DocSvc(config=docling_cfg if docling_cfg else None)
                        parsed = _dsvc.parse_pdf(metadata.exemplar_paper_path)
                        exemplar_analyzer = ExemplarAnalyzer(
                            self.client, self.model_name,
                            max_chars=exemplar_cfg.max_analysis_chars,
                        )
                        ref_info = {
                            "ref_id": "user_exemplar",
                            "title": metadata.title + " (user-provided exemplar)",
                            "venue": metadata.style_guide or "",
                            "year": 0,
                        }
                        ea = await exemplar_analyzer.analyze(
                            full_text=parsed.full_text,
                            sections=parsed.sections,
                            metadata=metadata,
                            ref_info=ref_info,
                        )
                        exemplar_analysis_dict = ea.model_dump(mode="json")
                        print(f"[MetaDataAgent] Exemplar analysis from user-provided PDF: {len(ea.section_blueprint)} sections")
                    else:
                        selector = ExemplarSelector(self.client, self.model_name)
                        selected = await selector.select(
                            core_refs=list(ref_pool.core_refs),
                            metadata=metadata,
                            config=exemplar_cfg,
                            paper_search_config=search_cfg_for_pool,
                        )
                        if selected:
                            exemplar_analyzer = ExemplarAnalyzer(
                                self.client, self.model_name,
                                max_chars=exemplar_cfg.max_analysis_chars,
                            )
                            ea = await exemplar_analyzer.analyze(
                                full_text=selected.get("docling_full_text", ""),
                                sections=selected.get("docling_sections", {}),
                                metadata=metadata,
                                ref_info={
                                    "ref_id": selected.get("ref_id", ""),
                                    "title": selected.get("title", ""),
                                    "venue": selected.get("venue", ""),
                                    "year": selected.get("year", 0),
                                },
                            )
                            exemplar_analysis_dict = ea.model_dump(mode="json")
                            print(f"[MetaDataAgent] Exemplar selected: {selected.get('ref_id', '')} ({len(ea.section_blueprint)} sections)")
                        else:
                            print("[MetaDataAgent] No suitable exemplar found among core refs")
                except Exception as e:
                    print(f"[MetaDataAgent] Warning: Exemplar analysis failed: {e}")
                    errors.append(f"Exemplar analysis failed: {e}")

            # Phase 0: Planning
            if enable_planning:
                print("[MetaDataAgent] Phase 0: Creating Paper Plan...")
                await emitter.phase_start(Phase.PLANNING, "Creating paper plan")
                search_cfg = {}
                if self.tools_config and self.tools_config.paper_search:
                    ps = self.tools_config.paper_search
                    search_cfg = {
                        "serpapi_api_key": ps.serpapi_api_key,
                        "semantic_scholar_api_key": ps.semantic_scholar_api_key,
                        "timeout": ps.timeout,
                        "search_results_per_round": ps.search_results_per_round,
                        "planner_max_queries_per_section": ps.planner_max_queries_per_section,
                        "planner_inter_round_delay_sec": ps.planner_inter_round_delay_sec,
                        "planner_min_target_papers_per_section": ps.planner_min_target_papers_per_section,
                        "semantic_scholar_min_results_before_fallback": ps.semantic_scholar_min_results_before_fallback,
                        "enable_query_cache": ps.enable_query_cache,
                        "cache_ttl_hours": ps.cache_ttl_hours,
                        "citation_budget_enabled": ps.citation_budget_enabled,
                        "citation_budget_soft_cap": ps.citation_budget_soft_cap,
                        "citation_budget_export": ps.citation_budget_export,
                        "citation_budget_reserve_size": ps.citation_budget_reserve_size,
                        "style_guide": metadata.style_guide,
                        "planner_landscape_max_queries": getattr(
                            ps, "planner_landscape_max_queries", 8,
                        ),
                        "planner_max_utility_searches": getattr(
                            ps, "planner_max_utility_searches", 12,
                        ),
                    }

                rc_cfg = (
                    self.tools_config.research_context
                    if self.tools_config and self.tools_config.research_context
                    else None
                )
                rc_enabled = rc_cfg.enabled if rc_cfg else False

                # Phase 0-core + 0-ctx: core ref analysis, landscape search, unified research context (pre-plan)
                if rc_enabled:
                    print("[MetaDataAgent] Phase 0-core: Analyzing core references...")
                    try:
                        _analyzer = CoreRefAnalyzer.from_tools_config(
                            self.client, self.model_name, self.tools_config,
                        )
                        _core_analysis = await _analyzer.analyze(
                            list(ref_pool.core_refs), metadata,
                        )
                    except Exception as e:
                        print(f"[MetaDataAgent] Warning: Core ref analysis failed: {e}")
                        from .models import CoreRefAnalysis as _CoreRefAnalysis
                        _core_analysis = _CoreRefAnalysis()

                    print("[MetaDataAgent] Phase 0-ctx: Landscape discovery + research context...")
                    landscape_papers: List[Dict[str, Any]] = []
                    try:
                        landscape_papers = await self._planner.discover_landscape_references(
                            core_analysis=_core_analysis,
                            title=metadata.title,
                            idea_hypothesis=metadata.idea_hypothesis,
                            paper_search_config=search_cfg,
                        )
                        for paper in landscape_papers:
                            ref_pool.add_discovered(
                                paper.get("ref_id", ""),
                                paper.get("bibtex", ""),
                                source="landscape_discovery",
                            )
                    except Exception as e:
                        print(f"[MetaDataAgent] Warning: Landscape discovery failed: {e}")

                    try:
                        _builder = ResearchContextBuilder(self.client, self.model_name)
                        _landscape_top_k = 24
                        if rc_cfg is not None and getattr(rc_cfg, "top_k_key_papers", None):
                            _landscape_top_k = max(8, int(rc_cfg.top_k_key_papers))

                        async def _score_landscape(topic: str, papers: List[Dict[str, Any]]):
                            return await self._planner._score_papers_by_relevance(topic, papers)

                        _rc_model = await _builder.build(
                            core_analysis=_core_analysis,
                            landscape_papers=landscape_papers,
                            paper_metadata=metadata,
                            score_papers_fn=_score_landscape if landscape_papers else None,
                            top_k_landscape=_landscape_top_k,
                        )
                        research_context = _rc_model.to_research_context_dict()
                    except Exception as e:
                        print(f"[MetaDataAgent] Warning: Research context build failed: {e}")

                paper_plan = await self._orchestrator._create_paper_plan(
                    metadata=metadata, target_pages=target_pages,
                    style_guide=metadata.style_guide, research_context=research_context,
                    code_context=code_context,
                )
                if paper_plan:
                    if self.tools_config and not getattr(
                        self.tools_config, "planner_structure_signals_enabled", True,
                    ):
                        for sp in paper_plan.sections:
                            sp.topic_clusters = []
                            sp.transition_intents = []
                            sp.sectioning_recommended = False

                    if paper_plan.wide_figures:
                        for fig in metadata.figures:
                            if fig.id in paper_plan.wide_figures and not fig.wide:
                                fig.wide = True
                    if paper_plan.wide_tables:
                        for tbl in metadata.tables:
                            if tbl.id in paper_plan.wide_tables and not tbl.wide:
                                tbl.wide = True

                    if save_output and paper_dir:
                        planning_dir = paper_dir / "analysis" / "planning"
                        planning_dir.mkdir(parents=True, exist_ok=True)
                        plan_path = planning_dir / "paper_plan.json"
                        plan_path.write_text(paper_plan.model_dump_json(indent=2), encoding="utf-8")

                    # Phase 0b: Reference discovery
                    print("[MetaDataAgent] Phase 0b: Discovering references...")
                    discovered = await self._planner.discover_references(
                        plan=paper_plan,
                        existing_ref_keys=list(ref_pool.valid_citation_keys),
                        paper_search_config=search_cfg,
                    )
                    disc_count = 0
                    for sec_type, papers in discovered.items():
                        for paper in papers:
                            if ref_pool.add_discovered(paper["ref_id"], paper["bibtex"], source="planner_discovery"):
                                disc_count += 1
                    if disc_count:
                        print(f"[MetaDataAgent] Discovered {disc_count} new references")

                    # Phase 0b-utility: dataset / metric / framework papers mentioned in plan text
                    try:
                        utility_refs = await self._planner.discover_utility_references(
                            plan=paper_plan,
                            existing_ref_keys=list(ref_pool.valid_citation_keys),
                            paper_search_config=search_cfg,
                        )
                        util_added = 0
                        for sec_type, papers in utility_refs.items():
                            for paper in papers:
                                if ref_pool.add_discovered(
                                    paper.get("ref_id", ""),
                                    paper.get("bibtex", ""),
                                    source="utility_discovery",
                                ):
                                    discovered.setdefault(sec_type, []).append(paper)
                                    util_added += 1
                        if util_added:
                            print(f"[MetaDataAgent] Utility discovery added {util_added} reference(s)")
                    except Exception as e:
                        print(f"[MetaDataAgent] Warning: Utility reference discovery failed: {e}")

                    # Merge section-level paper assignments into pre-plan research context
                    if rc_enabled and research_context is not None:
                        research_context = dict(research_context)
                        research_context["paper_assignments"] = (
                            self._planner._assign_papers_to_sections(paper_plan, discovered)
                        )

                    # Phase 0c: Assign references to sections
                    print("[MetaDataAgent] Phase 0c: Assigning references to sections...")
                    self._planner.assign_references(
                        plan=paper_plan,
                        discovered=discovered,
                        core_ref_keys=list(ref_pool.valid_citation_keys
                                           - {p["ref_id"] for papers in discovered.values() for p in papers}),
                        paper_search_config=search_cfg,
                        research_context=research_context,
                    )
                    for sp in paper_plan.sections:
                        if sp.assigned_refs:
                            print(f"  [{sp.section_type}] {len(sp.assigned_refs)} refs assigned")
                    await emitter.plan_created(
                        sections=len(paper_plan.sections),
                        estimated_words=paper_plan.get_total_estimated_words(),
                    )
                    await emitter.phase_complete(Phase.PLANNING, f"Plan created with {len(paper_plan.sections)} sections")
                else:
                    print("[MetaDataAgent] Planning skipped or failed, using defaults")

            # Phase 0d.5: Build Evidence DAG
            if paper_plan:
                try:
                    dag_builder = DAGBuilder(llm_client=self.client)
                    evidence_dag = await dag_builder.build(
                        code_context=code_context,
                        research_context=research_context,
                        figures=metadata.figures,
                        tables=metadata.tables,
                        paper_plan=paper_plan,
                        graph_structure=metadata.graph_structure,
                    )
                    paper_plan.evidence_dag = evidence_dag.to_serializable()

                    for sp in paper_plan.sections:
                        for pidx, para in enumerate(sp.paragraphs):
                            if not para.claim_id:
                                for claim in evidence_dag.claim_nodes.values():
                                    meta = claim.metadata
                                    if meta.get("section_type") == sp.section_type and meta.get("paragraph_index") == pidx:
                                        para.claim_id = claim.node_id
                                        para.bound_evidence_ids = evidence_dag.get_bound_evidence_ids_for_claim(claim.node_id)
                                        break

                    from ..planner_agent.planner_agent import PlannerAgent as _PA
                    total_sp = 0
                    for sp in paper_plan.sections:
                        for pidx, para in enumerate(sp.paragraphs):
                            if para.claim_id and not para.sentence_plans:
                                para.sentence_plans = _PA._generate_sentence_plans(para, evidence_dag=evidence_dag)
                                total_sp += len(para.sentence_plans)
                    if total_sp:
                        print(f"[MetaDataAgent] Generated {total_sp} sentence plans from DAG bindings")

                    if save_output and paper_dir:
                        dag_path = paper_dir / "analysis" / "planning" / "evidence_dag.json"
                        dag_path.parent.mkdir(parents=True, exist_ok=True)
                        dag_path.write_text(
                            json.dumps(evidence_dag.to_serializable(), indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                except Exception as e:
                    print(f"[MetaDataAgent] Warning: Evidence DAG construction failed: {e}")
                    evidence_dag = None

            # Phase 0e: Code repository context (post-plan fallback)
            if metadata.code_repository and not code_context:
                try:
                    code_context = await self._build_code_repository_context(metadata)
                    if code_context:
                        code_summary_markdown = render_code_repository_summary_markdown(code_context)
                except Exception as e:
                    on_error = metadata.code_repository.on_error
                    msg = f"Code repository ingestion failed: {e}"
                    if (
                        metadata.code_repository.type.value == "local_dir"
                        and isinstance(e, (FileNotFoundError, ValueError))
                    ):
                        return PlanResult(
                            paper_plan=paper_plan.model_dump() if paper_plan else {},
                            metadata_input=metadata.model_dump(), errors=[msg],
                            ref_pool_snapshot=ref_pool.to_dict(),
                            template_path=template_path, target_pages=target_pages,
                            artifacts_prefix=artifacts_prefix, paper_dir=paper_dir_str,
                        )
                    if on_error == CodeRepoOnError.STRICT:
                        return PlanResult(
                            paper_plan=paper_plan.model_dump() if paper_plan else {},
                            metadata_input=metadata.model_dump(), errors=[msg],
                            ref_pool_snapshot=ref_pool.to_dict(),
                            template_path=template_path, target_pages=target_pages,
                            artifacts_prefix=artifacts_prefix, paper_dir=paper_dir_str,
                        )
                    errors.append(msg)
                    code_context = None
                    code_summary_markdown = None

            # Phase 0.5: Convert tables
            converted_tables: Dict[str, str] = {}
            if metadata.tables:
                print(f"[MetaDataAgent] Phase 0.5: Converting {len(metadata.tables)} tables...")
                await emitter.phase_start(Phase.TABLE_CONVERSION, f"Converting {len(metadata.tables)} tables to LaTeX")
                base_path = str(paper_dir.parent) if (save_output and paper_dir) else None
                converted_tables = await convert_tables(
                    tables=metadata.tables, llm_client=self.client,
                    model_name=self.model_name, base_path=base_path,
                )

            return PlanResult(
                paper_plan=paper_plan.model_dump() if paper_plan else {},
                evidence_dag=evidence_dag.to_serializable() if evidence_dag else None,
                research_context=research_context,
                code_context=code_context,
                code_summary_markdown=code_summary_markdown,
                ref_pool_snapshot=ref_pool.to_dict(),
                converted_tables=converted_tables,
                metadata_input=metadata.model_dump(),
                errors=errors,
                template_path=template_path,
                target_pages=target_pages,
                exemplar_analysis=exemplar_analysis_dict,
                artifacts_prefix=artifacts_prefix,
                paper_dir=paper_dir_str,
            )

        except Exception as e:
            print(f"[MetaDataAgent] prepare_plan error: {e}")
            await emitter.error(message=str(e), phase="prepare_plan")
            return PlanResult(
                paper_plan={}, metadata_input=metadata.model_dump(),
                errors=[str(e)], ref_pool_snapshot=ref_pool.to_dict(),
                template_path=template_path, target_pages=target_pages,
                artifacts_prefix=artifacts_prefix, paper_dir=paper_dir_str,
            )
        finally:
            # Docling temp file cleanup
            if docling_temp_dir and docling_temp_dir.exists():
                if docling_cfg and docling_cfg.move_to_output and paper_dir:
                    dest = paper_dir / "reference_pdfs"
                    if not dest.exists():
                        shutil.move(str(docling_temp_dir), str(dest))
                elif docling_cfg and docling_cfg.cleanup_after_analysis:
                    shutil.rmtree(docling_temp_dir, ignore_errors=True)
            clear_llm_progress_context()

    # ------------------------------------------------------------------
    # execute_generation: Phases 1-5 from a PlanResult
    # ------------------------------------------------------------------

    async def execute_generation(
        self,
        plan_result: PlanResult,
        enable_review: bool = True,
        max_review_iterations: int = 3,
        compile_pdf: bool = True,
        enable_vlm_review: bool = False,
        enable_user_feedback: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        feedback_queue: Optional[asyncio.Queue] = None,
        feedback_timeout: float = 300.0,
        save_output: bool = True,
        output_dir: Optional[str] = None,
        figures_source_dir: Optional[str] = None,
    ) -> PaperGenerationResult:
        """
        Execute content generation from a previously computed PlanResult.
        - **Description**:
            - Deserializes PlanResult, reconstructs PaperPlan/EvidenceDAG/ReferencePool,
              then runs Phases 1 through 5 (introduction, body, synthesis, review,
              compilation, assembly).

        - **Args**:
            - `plan_result` (PlanResult): Output of ``prepare_plan()`` (possibly user-modified).
            - `enable_review` (bool): Whether to enable the review loop.
            - `max_review_iterations` (int): Maximum review iterations.
            - `compile_pdf` (bool): Whether to compile to PDF.
            - `enable_vlm_review` (bool): Whether to run VLM review.
            - `enable_user_feedback` (bool): Pause at review for user feedback.
            - `progress_callback` (ProgressCallback, optional): SSE callback.
            - `feedback_queue` (asyncio.Queue, optional): User feedback queue.
            - `feedback_timeout` (float): Seconds to wait for feedback.
            - `save_output` (bool): Whether to save output files.
            - `output_dir` (str, optional): Override output directory.
            - `figures_source_dir` (str, optional): Directory with figure files.

        - **Returns**:
            - `PaperGenerationResult`: Complete generation result.
        """
        # Reconstruct state from PlanResult
        metadata = PaperMetaData(**plan_result.metadata_input)
        ref_pool = ReferencePool.from_dict(plan_result.ref_pool_snapshot)
        template_path = plan_result.template_path
        target_pages = plan_result.target_pages
        artifacts_prefix = plan_result.artifacts_prefix
        converted_tables = plan_result.converted_tables
        research_context = plan_result.research_context
        code_context = plan_result.code_context
        code_summary_markdown = plan_result.code_summary_markdown
        errors = list(plan_result.errors)

        # Analyze template for Writer constraints
        template_guide: Optional[TemplateWriterGuide] = None
        if template_path:
            template_guide = TemplateAnalyzer.analyze_zip(template_path)
            if template_guide.available_packages:
                print(
                    f"[MetaDataAgent] Template analyzed: "
                    f"{len(template_guide.available_packages)} packages, "
                    f"column={template_guide.column_format}, "
                    f"citation={template_guide.citation_style}"
                )

        paper_plan: Optional[PaperPlan] = None
        if plan_result.paper_plan:
            paper_plan = PaperPlan(**plan_result.paper_plan)

        evidence_dag: Optional[EvidenceDAG] = None
        if plan_result.evidence_dag:
            evidence_dag = EvidenceDAG.from_serializable(plan_result.evidence_dag)

        # Reconstruct ExemplarAnalysis for prompt guidance
        from .models import ExemplarAnalysis as _ExemplarAnalysis
        from ..shared.exemplar_analyzer import ExemplarAnalyzer as _ExemplarAnalyzerCls
        _exemplar_analysis: Optional[_ExemplarAnalysis] = None
        if plan_result.exemplar_analysis:
            try:
                _exemplar_analysis = _ExemplarAnalysis(**plan_result.exemplar_analysis)
            except Exception:
                _exemplar_analysis = None

        # Resolve output directory
        if output_dir:
            paper_dir: Optional[Path] = Path(output_dir)
        elif plan_result.paper_dir:
            paper_dir = Path(plan_result.paper_dir)
        else:
            paper_dir = None
        if paper_dir and save_output:
            paper_dir.mkdir(parents=True, exist_ok=True)

        emitter = ProgressEmitter(callback=progress_callback)
        set_llm_progress_context(emitter, agent="MetaDataAgent")
        await emitter.generation_started(title=metadata.title, target_pages=target_pages)

        usage_tracker = UsageTracker()
        set_usage_tracker_context(usage_tracker, agent="MetaDataAgent", phase="generation")

        _sa = partial(self._save_artifact, artifacts_prefix=artifacts_prefix)

        sections_results: List[SectionResult] = []
        generated_sections: Dict[str, str] = {}
        review_iterations = 0
        target_word_count = None
        prompt_traces: List[Dict[str, Any]] = []
        citation_budget_usage: List[Dict[str, Any]] = []
        pdf_path: Optional[str] = None
        parsed_refs = ref_pool.get_all_refs()

        memory = SessionMemory()
        memory.log(
            "metadata", "init", "session_started",
            narrative=f"Resumed generation for '{metadata.title}' targeting {target_pages} pages.",
            title=metadata.title, target_pages=target_pages,
        )
        if paper_plan:
            memory.plan = paper_plan

        def _sec_filename(section_type: str) -> str:
            if paper_plan:
                titles = paper_plan.get_section_titles()
                if section_type in titles:
                    return titles[section_type]
            return section_type.replace("_", " ").title()

        try:
            # Phase 1: Introduction
            print("[MetaDataAgent] Phase 1: Generating Introduction...")
            update_usage_tracker_context(agent="WriterAgent", phase="introduction")
            await emitter.phase_start(Phase.INTRODUCTION, "Generating introduction")
            await emitter.section_start("introduction", phase=Phase.INTRODUCTION)
            await emitter.agent_step(agent="WriterAgent", description="Generating introduction section", section="introduction", phase=Phase.INTRODUCTION)
            intro_plan = paper_plan.get_section("introduction") if paper_plan else None
            intro_result = await self._generate_introduction(
                metadata, ref_pool, section_plan=intro_plan,
                figures=metadata.figures, tables=metadata.tables,
                code_context=code_context, research_context=research_context,
                prompt_traces=prompt_traces, memory=memory, evidence_dag=evidence_dag,
                template_guide=template_guide,
                exemplar_guidance=_ExemplarAnalyzerCls.format_for_prompt(_exemplar_analysis, "introduction"),
            )
            sections_results.append(intro_result)
            print(f"[MetaDataAgent] After introduction: {ref_pool.summary()}")

            if intro_result.status == "ok":
                generated_sections["introduction"] = intro_result.latex_content
                memory.update_section("introduction", intro_result.latex_content)
                memory.log("metadata", "phase1", "introduction_generated",
                           narrative=f"Writer completed the introduction section ({intro_result.word_count} words).",
                           word_count=intro_result.word_count)
                await emitter.section_content(
                    section_type="introduction", content=intro_result.latex_content,
                    word_count=intro_result.word_count, phase=Phase.INTRODUCTION,
                )
                if intro_plan:
                    intro_valid_keys = list(dict.fromkeys(
                        list(intro_plan.assigned_refs or []) + list(intro_plan.budget_reserve_refs or [])
                    ))
                    intro_budget_usage = self._collect_section_citation_budget_usage(
                        section_type="introduction", content=intro_result.latex_content,
                        section_plan=intro_plan, writer_valid_keys=intro_valid_keys,
                    )
                    self._upsert_section_budget_usage(citation_budget_usage, intro_budget_usage)
                contributions = extract_contributions_from_intro(intro_result.latex_content)
                if not contributions:
                    contributions = [f"We propose {metadata.title}", f"Novel approach: {metadata.method[:100]}..."]
            else:
                errors.append(f"Introduction generation failed: {intro_result.error}")
                return PaperGenerationResult(
                    status="error", paper_title=metadata.title,
                    sections=sections_results, errors=errors,
                    usage=usage_tracker.to_dict(),
                )

            if paper_plan and paper_plan.contributions:
                contributions = paper_plan.contributions
            memory.contributions = contributions

            # Phase 2: Body Sections
            print("[MetaDataAgent] Phase 2: Generating Body Sections...")
            update_usage_tracker_context(agent="WriterAgent", phase="body_sections")
            await emitter.phase_start(Phase.BODY_SECTIONS, "Generating body sections")
            body_section_types = paper_plan.get_body_section_types() if paper_plan else ["related_work", "method", "experiment", "result"]
            # Skip introduction — already generated in Phase 1 via _generate_introduction
            body_section_types = [s for s in body_section_types if s != "introduction"]
            for section_type in body_section_types:
                section_plan = paper_plan.get_section(section_type) if paper_plan else None
                section_figures = [f for f in metadata.figures if f.section == section_type or not f.section]
                section_tables = [t for t in metadata.tables if t.section == section_type or not t.section]
                update_usage_tracker_context(section=section_type)
                try:
                    result = await self._generate_body_section(
                        section_type=section_type, metadata=metadata,
                        intro_context=generated_sections.get("introduction", ""),
                        contributions=contributions, ref_pool=ref_pool,
                        section_plan=section_plan, figures=section_figures,
                        tables=section_tables, converted_tables=converted_tables,
                        code_context=code_context, research_context=research_context,
                        prompt_traces=prompt_traces, memory=memory, evidence_dag=evidence_dag,
                        emitter=emitter, template_guide=template_guide,
                        exemplar_guidance=_ExemplarAnalyzerCls.format_for_prompt(_exemplar_analysis, section_type),
                    )
                except Exception as e:
                    result = SectionResult(section_type=section_type, status="error", error=str(e))

                sections_results.append(result)
                if result.status == "ok":
                    generated_sections[section_type] = result.latex_content
                    memory.update_section(section_type, result.latex_content)
                    memory.log("metadata", "phase2", f"{section_type}_generated",
                               narrative=f"Writer completed the {section_type} section ({result.word_count} words).",
                               word_count=result.word_count)
                    await emitter.section_content(
                        section_type=section_type, content=result.latex_content,
                        word_count=result.word_count, phase=Phase.BODY_SECTIONS,
                    )
                    if section_plan:
                        section_valid_keys = list(dict.fromkeys(
                            list(section_plan.assigned_refs or []) + list(section_plan.budget_reserve_refs or [])
                        ))
                        section_budget_usage = self._collect_section_citation_budget_usage(
                            section_type=section_type, content=result.latex_content,
                            section_plan=section_plan, writer_valid_keys=section_valid_keys,
                        )
                        self._upsert_section_budget_usage(citation_budget_usage, section_budget_usage)
                else:
                    errors.append(f"{section_type} generation failed: {result.error}")

            # Phase 3: Synthesis Sections
            print("[MetaDataAgent] Phase 3: Generating Synthesis Sections...")
            update_usage_tracker_context(agent="WriterAgent", phase="synthesis")
            await emitter.phase_start(Phase.SYNTHESIS, "Generating synthesis sections (abstract, conclusion)")
            abstract_result = await self._generate_synthesis_section(
                section_type="abstract", paper_title=metadata.title,
                prior_sections=generated_sections, contributions=contributions,
                style_guide=metadata.style_guide,
                section_plan=paper_plan.get_section("abstract") if paper_plan else None,
                prompt_traces=prompt_traces, memory=memory,
                template_guide=template_guide,
                exemplar_guidance=_ExemplarAnalyzerCls.format_for_prompt(_exemplar_analysis, "abstract"),
            )
            sections_results.insert(0, abstract_result)
            if abstract_result.status == "ok":
                generated_sections["abstract"] = abstract_result.latex_content
                memory.update_section("abstract", abstract_result.latex_content)
                await emitter.section_content(
                    section_type="abstract", content=abstract_result.latex_content,
                    word_count=abstract_result.word_count, phase=Phase.SYNTHESIS,
                )
            else:
                errors.append(f"Abstract generation failed: {abstract_result.error}")

            should_generate_conclusion = bool(paper_plan and paper_plan.get_section("conclusion") is not None)
            if should_generate_conclusion:
                conclusion_result = await self._generate_synthesis_section(
                    section_type="conclusion", paper_title=metadata.title,
                    prior_sections=generated_sections, contributions=contributions,
                    style_guide=metadata.style_guide,
                    section_plan=paper_plan.get_section("conclusion") if paper_plan else None,
                    prompt_traces=prompt_traces, memory=memory,
                    template_guide=template_guide,
                    exemplar_guidance=_ExemplarAnalyzerCls.format_for_prompt(_exemplar_analysis, "conclusion"),
                )
                sections_results.append(conclusion_result)
                if conclusion_result.status == "ok":
                    generated_sections["conclusion"] = conclusion_result.latex_content
                    memory.update_section("conclusion", conclusion_result.latex_content)
                    await emitter.section_content(
                        section_type="conclusion", content=conclusion_result.latex_content,
                        word_count=conclusion_result.word_count, phase=Phase.SYNTHESIS,
                    )
                else:
                    errors.append(f"Conclusion generation failed: {conclusion_result.error}")

            # Reference Usage Validation
            self._validate_ref_usage(generated_sections, ref_pool)

            # Review Orchestration
            update_usage_tracker_context(agent="ReviewerAgent", phase="review")
            if enable_review:
                await emitter.phase_start(Phase.REVIEW_LOOP, "Starting review loop")
            (
                generated_sections, sections_results,
                review_iterations, target_word_count, pdf_path, orchestration_errors,
            ) = await self._orchestrator._run_review_orchestration(
                generated_sections=generated_sections, sections_results=sections_results,
                metadata=metadata, parsed_refs=ref_pool.get_all_refs(),
                paper_plan=paper_plan, template_path=template_path,
                figures_source_dir=figures_source_dir, converted_tables=converted_tables,
                max_review_iterations=max_review_iterations, enable_review=enable_review,
                compile_pdf=compile_pdf, enable_vlm_review=enable_vlm_review,
                target_pages=target_pages, paper_dir=paper_dir,
                memory=memory, evidence_dag=evidence_dag,
            )
            if orchestration_errors:
                errors.extend(orchestration_errors)
            if enable_review:
                await emitter.phase_complete(Phase.REVIEW_LOOP, f"Review completed ({review_iterations} iterations)")

            if paper_plan:
                citation_budget_usage = self._rebuild_citation_budget_usage_from_final_sections(
                    paper_plan=paper_plan, generated_sections=generated_sections,
                )

            # Assemble Paper
            update_usage_tracker_context(agent="MetaDataAgent", phase="assembly")
            print("[MetaDataAgent] Assembling paper...")
            latex_content = self._assemble_paper(
                title=metadata.title, sections=generated_sections,
                references=ref_pool.get_all_refs(),
                valid_citation_keys=ref_pool.valid_citation_keys,
            )
            total_words = sum(r.word_count for r in sections_results if r.word_count)

            # Save output
            output_path = None
            if save_output and paper_dir:
                output_path = str(paper_dir)
                for d_name in ("analysis/planning", "analysis/research_context",
                               "analysis/citations", "analysis/structure",
                               "analysis/review", "analysis/references",
                               "analysis/code_context", "logs/traces"):
                    (paper_dir / d_name).mkdir(parents=True, exist_ok=True)

                (paper_dir / "main.tex").write_text(latex_content, encoding="utf-8")
                (paper_dir / "references.bib").write_text(ref_pool.to_bibtex(), encoding="utf-8")
                (paper_dir / "metadata.json").write_text(
                    json.dumps(metadata.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8",
                )
                memory.log("metadata", "final", "paper_assembled",
                           narrative=f"Paper assembled successfully with {total_words} total words.",
                           total_words=total_words, status="assembled")
                memory.persist_all(paper_dir)

            status = "ok" if not errors else ("partial" if len(errors) < len(sections_results) else "error")
            result = PaperGenerationResult(
                status=status, paper_title=metadata.title,
                sections=sections_results, latex_content=latex_content,
                output_path=output_path, pdf_path=pdf_path,
                total_word_count=total_words, target_word_count=target_word_count,
                review_iterations=review_iterations, errors=errors,
                usage=usage_tracker.to_dict(),
            )
            await emitter.completed(
                status=status, total_words=total_words,
                review_iterations=review_iterations,
                sections_count=len([s for s in sections_results if s.status == "ok"]),
                pdf_path=pdf_path,
                paper_dir=str(paper_dir) if paper_dir else None,
            )
            return result

        except Exception as e:
            print(f"[MetaDataAgent] execute_generation error: {e}")
            await emitter.error(message=str(e), phase="execute_generation")
            return PaperGenerationResult(
                status="error", paper_title=metadata.title,
                sections=sections_results, errors=[str(e)],
                usage=usage_tracker.to_dict(),
            )
        finally:
            clear_usage_tracker_context()
            clear_llm_progress_context()

    # ------------------------------------------------------------------
    # generate_paper: backward-compatible wrapper
    # ------------------------------------------------------------------

    async def generate_paper(
        self,
        metadata: PaperMetaData,
        output_dir: Optional[str] = None,
        save_output: bool = True,
        compile_pdf: bool = True,
        template_path: Optional[str] = None,
        figures_source_dir: Optional[str] = None,
        target_pages: Optional[int] = None,
        enable_review: bool = True,
        max_review_iterations: int = 3,
        enable_planning: bool = True,
        enable_exemplar: bool = False,
        enable_vlm_review: bool = False,
        enable_user_feedback: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        feedback_queue: Optional[asyncio.Queue] = None,
        feedback_timeout: float = 300.0,
        artifacts_prefix: str = "",
    ) -> PaperGenerationResult:
        """
        Generate complete paper from MetaData (backward-compatible wrapper).
        - **Description**:
            - Calls ``prepare_plan()`` then ``execute_generation()`` sequentially.
            - Preserves the original single-call interface used by
              ``/metadata/generate/stream`` and ``/metadata/generate``.
        """
        import time as _time
        _t0 = _time.monotonic()

        _gp_tracker = UsageTracker()
        set_usage_tracker_context(_gp_tracker, agent="PlannerAgent", phase="planning")

        plan_result = await self.prepare_plan(
            metadata=metadata,
            template_path=template_path,
            target_pages=target_pages,
            enable_planning=enable_planning,
            enable_exemplar=enable_exemplar,
            save_output=save_output,
            output_dir=output_dir,
            progress_callback=progress_callback,
            artifacts_prefix=artifacts_prefix,
        )

        if plan_result.errors and not plan_result.paper_plan:
            clear_usage_tracker_context()
            _gp_tracker.set_elapsed_time(round(_time.monotonic() - _t0, 2))
            return PaperGenerationResult(
                status="error",
                paper_title=metadata.title,
                errors=plan_result.errors,
                usage=_gp_tracker.to_dict(),
            )

        clear_usage_tracker_context()

        result = await self.execute_generation(
            plan_result=plan_result,
            enable_review=enable_review,
            max_review_iterations=max_review_iterations,
            compile_pdf=compile_pdf,
            enable_vlm_review=enable_vlm_review,
            enable_user_feedback=enable_user_feedback,
            progress_callback=progress_callback,
            feedback_queue=feedback_queue,
            feedback_timeout=feedback_timeout,
            save_output=save_output,
            output_dir=output_dir,
            figures_source_dir=figures_source_dir,
        )

        elapsed = round(_time.monotonic() - _t0, 2)
        merged_usage = self._merge_usage_reports(_gp_tracker.to_dict(), result.usage or {})
        merged_usage.setdefault("summary", {})["elapsed_seconds"] = elapsed
        result.usage = merged_usage
        return result

    @staticmethod
    def _merge_usage_reports(plan_report: dict, gen_report: dict) -> dict:
        """
        Merge usage reports from prepare_plan and execute_generation phases.
        - **Description**:
            - Combines calls lists and recalculates all aggregate fields.
        """
        plan_calls = plan_report.get("calls", [])
        gen_calls = gen_report.get("calls", [])
        all_calls = plan_calls + gen_calls

        from ..shared.usage_tracker import UsageTracker, LLMCallRecord
        merged = UsageTracker()
        for c in all_calls:
            merged.record(LLMCallRecord(**c))
        return merged.to_dict()

    async def generate_single_section(
        self,
        request: SectionGenerationRequest,
    ) -> SectionResult:
        """Generate a single section (for debugging or incremental generation)"""
        metadata = request.metadata
        ref_pool = ReferencePool(metadata.references)
        
        if request.section_type == "introduction":
            return await self._generate_introduction(metadata, ref_pool)
        elif request.section_type in SYNTHESIS_SECTIONS:
            prior = request.prior_sections or {}
            contributions = extract_contributions_from_intro(prior.get("introduction", ""))
            return await self._generate_synthesis_section(
                section_type=request.section_type,
                paper_title=metadata.title,
                prior_sections=prior,
                contributions=contributions,
                style_guide=metadata.style_guide,
            )
        else:
            contributions = []
            if request.intro_context:
                contributions = extract_contributions_from_intro(request.intro_context)
            return await self._generate_body_section(
                section_type=request.section_type,
                metadata=metadata,
                intro_context=request.intro_context or "",
                contributions=contributions,
                ref_pool=ref_pool,
            )
    
    # =========================================================================
    # Phase 1: Introduction Generation
    # =========================================================================
    
    async def _generate_introduction(
        self,
        metadata: PaperMetaData,
        ref_pool: ReferencePool,
        section_plan: Optional[SectionPlan] = None,
        figures: Optional[List[FigureSpec]] = None,
        tables: Optional[List[TableSpec]] = None,
        code_context: Optional[Dict[str, Any]] = None,
        research_context: Optional[Dict[str, Any]] = None,
        prompt_traces: Optional[List[Dict[str, Any]]] = None,
        memory: Optional[SessionMemory] = None,
        evidence_dag: Optional[EvidenceDAG] = None,
        template_guide: Optional[TemplateWriterGuide] = None,
        exemplar_guidance: Optional[str] = None,
    ) -> SectionResult:
        """
        Generate Introduction section — delegates to WriterAgent.

        - **Description**:
            - Prompt compilation, then WriterAgent.run() with
              ReAct AskTool + iterative mini-review.
            - Reference discovery is done centrally in Planner (Phase 0b).

        - **Args**:
            - `metadata` (PaperMetaData): Paper metadata.
            - `ref_pool` (ReferencePool): Persistent reference pool.
            - `section_plan` (SectionPlan, optional): Plan for this section.
            - `figures` (List[FigureSpec], optional): Figures.
            - `tables` (List[TableSpec], optional): Tables.
            - `memory` (SessionMemory, optional): Shared session memory.

        - **Returns**:
            - `SectionResult`: Generation result.
        """
        try:
            key_points: List[str] = []
            if section_plan:
                key_points = section_plan.get_key_points()

            # Compile prompt then delegate to WriterAgent
            intro_runtime_evidence = self._retrieve_runtime_code_evidence(
                code_context=code_context,
                section_type="introduction",
                metadata=metadata,
                top_k=2,
            )
            intro_code_context = format_code_context_for_prompt(
                context=code_context,
                section_type="introduction",
                retrieved_evidence=intro_runtime_evidence,
                top_k=4,
            )
            intro_research_context = self._format_research_context_for_prompt(
                research_context=research_context,
                section_type="introduction",
                evidence_dag=evidence_dag,
            )
            prompt = compile_introduction_prompt(
                paper_title=metadata.title,
                idea_hypothesis=metadata.idea_hypothesis,
                method_summary=metadata.method,
                data_summary=metadata.data,
                experiments_summary=metadata.experiments,
                references=ref_pool.get_all_refs(),
                style_guide=metadata.style_guide,
                section_plan=section_plan,
                figures=figures,
                tables=tables,
                active_skills=self._get_active_skills("introduction", metadata.style_guide),
                code_context=intro_code_context,
                research_context=intro_research_context,
                enable_structure_contract=bool(
                    self.tools_config is None
                    or getattr(self.tools_config, "writer_structure_contract_enabled", True)
                ),
                template_guide=template_guide,
                exemplar_guidance=exemplar_guidance or None,
            )
            if prompt_traces is not None:
                prompt_traces.append(
                    {
                        "section_type": "introduction",
                        "phase": "generation",
                        "code_context_used": bool(intro_code_context),
                        "research_context_used": bool(intro_research_context),
                        "runtime_evidence": intro_runtime_evidence,
                        "prompt": prompt,
                    }
                )

            # Use section-assigned refs if available, else fall back to full pool
            section_keys = (
                section_plan.assigned_refs
                if section_plan and section_plan.assigned_refs
                else list(ref_pool.valid_citation_keys)
            )
            reserve_keys = (
                section_plan.budget_reserve_refs
                if section_plan and section_plan.budget_reserve_refs
                else []
            )
            writer_valid_keys = list(dict.fromkeys(list(section_keys) + list(reserve_keys)))
            result = await self._writer.run(
                system_prompt=GENERATION_SYSTEM_PROMPT,
                user_prompt=prompt,
                section_type="introduction",
                valid_citation_keys=writer_valid_keys,
                key_points=key_points or None,
                memory=memory,
                peers={"planner": self._planner, "reviewer": self._reviewer},
            )
            content = result.get("generated_content", "")
            if not content.strip():
                return SectionResult(
                    section_type="introduction",
                    section_title=section_plan.section_title if section_plan else "Introduction",
                    status="error",
                    error="Writer returned empty introduction content",
                )
            word_count = len(content.split())

            return SectionResult(
                section_type="introduction",
                section_title=section_plan.section_title if section_plan else "Introduction",
                status="ok",
                latex_content=content,
                word_count=word_count,
            )
        except Exception as e:
            return SectionResult(
                section_type="introduction",
                status="error",
                error=str(e),
            )
    
    # =========================================================================
    # Phase 2: Body Section Generation
    # =========================================================================
    
    async def _generate_body_section(
        self,
        section_type: str,
        metadata: PaperMetaData,
        intro_context: str,
        contributions: List[str],
        ref_pool: ReferencePool,
        section_plan: Optional[SectionPlan] = None,
        figures: Optional[List[FigureSpec]] = None,
        tables: Optional[List[TableSpec]] = None,
        converted_tables: Optional[Dict[str, str]] = None,
        code_context: Optional[Dict[str, Any]] = None,
        research_context: Optional[Dict[str, Any]] = None,
        prompt_traces: Optional[List[Dict[str, Any]]] = None,
        memory: Optional[SessionMemory] = None,
        evidence_dag: Optional[EvidenceDAG] = None,
        template_guide: Optional[TemplateWriterGuide] = None,
        emitter: Optional[ProgressEmitter] = None,
        exemplar_guidance: Optional[str] = None,
    ) -> SectionResult:
        """
        Generate a body section using two-phase pattern.

        - **Description**:
            - Phase A (Judgment): LLM analyses whether existing references
              cover the section needs and suggests search queries if not.
            - Phase A (Search): If needed, PaperSearchTool is called directly
              (system-side) and results are merged into ref_pool.
            - Phase B (Writing): Pure LLM call with no tools. All available
              refs (core + newly discovered) are included in the prompt.

        - **Args**:
            - `section_type` (str): Type of section (method, experiment, etc.).
            - `metadata` (PaperMetaData): Paper metadata.
            - `intro_context` (str): Introduction content for context.
            - `contributions` (List[str]): Paper contributions.
            - `ref_pool` (ReferencePool): Persistent reference pool.
            - `section_plan` (SectionPlan, optional): Plan for this section.
            - `figures` (List[FigureSpec], optional): Figures for this section.
            - `tables` (List[TableSpec], optional): Tables for this section.
            - `converted_tables` (Dict[str, str], optional): Converted table LaTeX.

        - **Returns**:
            - `SectionResult`: Generation result.
        """
        try:
            key_points = []
            if section_plan:
                key_points = section_plan.get_key_points()

            section_title_str = (
                section_plan.section_title
                if section_plan and section_plan.section_title
                else section_type.replace("_", " ").title()
            )
            # Get relevant content from metadata based on section type
            if section_plan and section_plan.content_sources:
                sources = section_plan.content_sources
            else:
                sources = BODY_SECTION_SOURCES.get(section_type, [])

            content_parts = []
            for source in sources:
                if source == "references":
                    continue  # References are handled separately
                value = getattr(metadata, source, "")
                if value:
                    content_parts.append(f"### {source.title()}\n{value}")

            metadata_content = "\n\n".join(content_parts) if content_parts else metadata.method

            # Build memory context for cross-section awareness
            memory_context = memory.get_writing_context(section_type) if memory else ""

            runtime_evidence = self._retrieve_runtime_code_evidence(
                code_context=code_context,
                section_type=section_type,
                metadata=metadata,
                contributions=contributions,
                top_k=3,
            )
            section_code_context = format_code_context_for_prompt(
                context=code_context,
                section_type=section_type,
                retrieved_evidence=runtime_evidence,
                top_k=6,
            )
            section_research_context = self._format_research_context_for_prompt(
                research_context=research_context,
                section_type=section_type,
                evidence_dag=evidence_dag,
            )

            # Build prompt AFTER search so it includes any new refs
            prompt = compile_body_section_prompt(
                section_type=section_type,
                metadata_content=metadata_content,
                intro_context=intro_context,
                contributions=contributions,
                references=ref_pool.get_all_refs(),
                style_guide=metadata.style_guide,
                section_plan=section_plan,
                figures=figures,
                tables=tables,
                converted_tables=converted_tables,
                active_skills=self._get_active_skills(section_type, metadata.style_guide),
                memory_context=memory_context,
                code_context=section_code_context,
                research_context=section_research_context,
                enable_structure_contract=bool(
                    self.tools_config is None
                    or getattr(self.tools_config, "writer_structure_contract_enabled", True)
                ),
                template_guide=template_guide,
                exemplar_guidance=exemplar_guidance or None,
            )
            if prompt_traces is not None:
                prompt_traces.append(
                    {
                        "section_type": section_type,
                        "phase": "generation",
                        "code_context_used": bool(section_code_context),
                        "research_context_used": bool(section_research_context),
                        "runtime_evidence": runtime_evidence,
                        "prompt": prompt,
                    }
                )

            section_keys = (
                section_plan.assigned_refs
                if section_plan and section_plan.assigned_refs
                else list(ref_pool.valid_citation_keys)
            )
            reserve_keys = (
                section_plan.budget_reserve_refs
                if section_plan and section_plan.budget_reserve_refs
                else []
            )
            writer_valid_keys = list(dict.fromkeys(list(section_keys) + list(reserve_keys)))

            # ---------------------------------------------------------------
            # Decomposed generation: paragraph-by-paragraph when DAG bindings
            # are available.  Falls back to section-level generation otherwise.
            # ---------------------------------------------------------------
            has_claim_bindings = (
                evidence_dag is not None
                and section_plan is not None
                and any(p.claim_id for p in section_plan.paragraphs)
            )
            if has_claim_bindings:
                content = await self._generate_section_decomposed(
                    section_type=section_type,
                    section_plan=section_plan,
                    evidence_dag=evidence_dag,
                    writer_valid_keys=writer_valid_keys,
                    section_title_str=section_title_str,
                    memory=memory,
                    emitter=emitter,
                    template_guide=template_guide,
                    exemplar_guidance=exemplar_guidance,
                )
            else:
                result = await self._writer.run(
                    system_prompt=GENERATION_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    section_type=section_type,
                    valid_citation_keys=writer_valid_keys,
                    key_points=key_points or None,
                    memory=memory,
                    peers={"planner": self._planner, "reviewer": self._reviewer},
                )
                content = result.get("generated_content", "")
            word_count = len(content.split())

            # Use plan's title if available
            if section_plan and section_plan.section_title:
                section_title = section_plan.section_title
            else:
                titles = {
                    "related_work": "Related Work",
                    "method": "Methodology",
                    "experiment": "Experiments",
                    "result": "Results",
                    "discussion": "Discussion",
                }
                section_title = titles.get(section_type, section_type.title())

            return SectionResult(
                section_type=section_type,
                section_title=section_title,
                status="ok",
                latex_content=content,
                word_count=word_count,
            )
        except Exception as e:
            return SectionResult(
                section_type=section_type,
                status="error",
                error=str(e),
            )
    
    # =========================================================================
    # Decomposed (claim-level) generation helper
    # =========================================================================

    async def _generate_section_decomposed(
        self,
        section_type: str,
        section_plan: SectionPlan,
        evidence_dag: EvidenceDAG,
        writer_valid_keys: List[str],
        section_title_str: str = "",
        memory: Optional[SessionMemory] = None,
        emitter: Optional[ProgressEmitter] = None,
        template_guide: Optional[TemplateWriterGuide] = None,
        exemplar_guidance: Optional[str] = None,
    ) -> str:
        """
        Generate a section paragraph-by-paragraph with verify-retry-degrade.
        - **Description**:
            - Iterates over ``section_plan.paragraphs``.
            - For each paragraph with a ``claim_id``, compiles a focused prompt
              and calls ``writer.generate_paragraph()``.
            - Immediately verifies the output using ``ClaimVerifier``.
            - On failure, retries up to ``MAX_CLAIM_RETRIES`` with feedback.
            - After exhausting retries, degrades to template-slot filling
              if ``TEMPLATE_FALLBACK_ENABLED`` and a template is available.
            - Accumulates ``section_context`` for coherence.

        - **Args**:
            - ``section_type`` (str): Section identifier.
            - ``section_plan`` (SectionPlan): Plan with paragraph-level structure.
            - ``evidence_dag`` (EvidenceDAG): The evidence graph.
            - ``writer_valid_keys`` (List[str]): Superset of allowed citation keys.
            - ``section_title_str`` (str): Display title for prompt context.
            - ``memory`` (SessionMemory, optional): Shared memory.
            - ``emitter`` (ProgressEmitter, optional): Emits paragraph-level SSE events.

        - **Returns**:
            - ``str``: Assembled LaTeX content for the entire section.
        """
        from ...generation.claim_verifier import (
            ClaimVerifier,
            MAX_CLAIM_RETRIES,
            TEMPLATE_FALLBACK_ENABLED,
        )
        from ...generation.template_slots import ParagraphTemplate

        verifier = ClaimVerifier()
        paragraph_outputs: List[str] = []
        section_context = ""
        total_paras = len(section_plan.paragraphs)
        verify_stats = {"passed": 0, "retried": 0, "degraded": 0, "skipped": 0}

        for pidx, para in enumerate(section_plan.paragraphs):
            if not para.claim_id:
                paragraph_outputs.append("")
                verify_stats["skipped"] += 1
                continue

            if emitter is not None:
                await emitter.paragraph_start(
                    section_type=section_type,
                    paragraph_index=pidx,
                    claim_id=para.claim_id,
                    total_paragraphs=total_paras,
                    phase=Phase.BODY_SECTIONS,
                )

            # Gather evidence for this paragraph's claim
            evidence_nodes = evidence_dag.get_evidence_for_claim(para.claim_id)
            evidence_snippets: List[str] = []
            evidence_snippet_map: Dict[str, str] = {}
            para_valid_refs: List[str] = list(writer_valid_keys)
            for enode in evidence_nodes:
                snippet = enode.content or enode.source_path or enode.node_id
                evidence_snippets.append(snippet)
                evidence_snippet_map[enode.node_id] = snippet
                if enode.source_path and enode.source_path not in para_valid_refs:
                    para_valid_refs.append(enode.source_path)

            for r in para.references_to_cite:
                if r not in para_valid_refs:
                    para_valid_refs.append(r)

            valid_keys_set = set(para_valid_refs)
            latex = ""
            verification_feedback = ""

            figs_to_ref = getattr(para, "figures_to_reference", []) or []
            tables_to_ref = getattr(para, "tables_to_reference", []) or []

            for attempt in range(MAX_CLAIM_RETRIES):
                # === Stage 1: Core content (no citations, CITE/FLOAT markers) ===
                core_prompt = compile_core_prompt(
                    paragraph_plan=para,
                    section_type=section_type,
                    section_context=section_context,
                    evidence_snippets=evidence_snippets,
                    section_title=section_title_str,
                    paragraph_index=pidx,
                    total_paragraphs=total_paras,
                )
                if attempt > 0 and verification_feedback:
                    core_prompt += f"\n\n### Revision Guidance\n{verification_feedback}"

                core_result = await self._writer.generate_core_content(
                    core_prompt=core_prompt,
                    section_type=section_type,
                    paragraph_index=pidx,
                )
                raw_latex = core_result.raw_latex

                # === Stage 2: Citation injection ===
                assigned_refs_for_prompt = []
                for rkey in para_valid_refs:
                    assigned_refs_for_prompt.append({"id": rkey, "title": "", "abstract": ""})

                cite_prompt = compile_citation_prompt(
                    raw_latex=raw_latex,
                    assigned_refs=assigned_refs_for_prompt,
                    section_type=section_type,
                )
                cite_result = await self._writer.inject_citations(
                    citation_prompt=cite_prompt,
                    valid_refs=para_valid_refs,
                )
                latex = apply_citation_edits(
                    raw_latex, cite_result.actions, valid_keys=valid_keys_set,
                )

                # === Stage 3: Float reference injection ===
                latex = inject_float_refs(latex, figs_to_ref, tables_to_ref)

                vr = await verifier.verify(
                    generated_text=latex,
                    paragraph_plan=para,
                    evidence_dag=evidence_dag,
                    valid_citation_keys=valid_keys_set,
                )

                if emitter is not None:
                    fb = (vr.feedback_for_retry or "")[:500]
                    await emitter.claim_verify_result(
                        section_type=section_type,
                        paragraph_index=pidx,
                        claim_id=para.claim_id,
                        passed=vr.passed,
                        attempt=attempt + 1,
                        max_attempts=MAX_CLAIM_RETRIES,
                        feedback_summary=fb,
                        phase=Phase.BODY_SECTIONS,
                    )

                if vr.passed:
                    verify_stats["passed"] += 1
                    break

                verification_feedback = vr.feedback_for_retry
                verify_stats["retried"] += 1
                print(
                    f"[MetaDataAgent] Paragraph {pidx} attempt {attempt + 1} "
                    f"failed verification: {len(vr.citation_issues)} citation issues, "
                    f"{len(vr.missing_evidence_refs)} missing refs, "
                    f"{len(vr.coverage_gaps)} coverage gaps"
                )
            else:
                # Exhausted retries — try template-slot filling as fallback
                if TEMPLATE_FALLBACK_ENABLED and para.paragraph_template:
                    try:
                        tmpl_data = para.paragraph_template
                        if isinstance(tmpl_data, dict):
                            tmpl = ParagraphTemplate(**tmpl_data)
                        else:
                            tmpl = tmpl_data

                        from ...generation.template_slots import build_template_fill_prompt
                        tmpl_prompt = build_template_fill_prompt(
                            template=tmpl,
                            evidence_snippets=evidence_snippet_map,
                            valid_refs=para_valid_refs,
                        )
                        tmpl_result = await self._writer.generate_from_template(
                            template_prompt=tmpl_prompt,
                            section_type=section_type,
                            valid_refs=para_valid_refs,
                            paragraph_index=pidx,
                        )
                        latex = tmpl_result.latex_content if hasattr(tmpl_result, "latex_content") else tmpl_result.get("latex_content", latex)
                        verify_stats["degraded"] += 1
                        print(
                            f"[MetaDataAgent] Paragraph {pidx} degraded to template-slot filling"
                        )
                    except Exception as tmpl_err:
                        print(
                            f"[MetaDataAgent] Template fallback failed for paragraph {pidx}: {tmpl_err}"
                        )

            wc = len(latex.split()) if latex else 0
            if emitter is not None:
                await emitter.paragraph_content(
                    section_type=section_type,
                    paragraph_index=pidx,
                    claim_id=para.claim_id,
                    content=latex,
                    word_count=wc,
                    phase=Phase.BODY_SECTIONS,
                )

            paragraph_outputs.append(latex)
            if latex:
                section_context += "\n\n" + latex

        content = "\n\n".join(p for p in paragraph_outputs if p)
        print(
            f"[MetaDataAgent] Decomposed generation for '{section_type}': "
            f"{verify_stats['passed']} passed, {verify_stats['retried']} retried, "
            f"{verify_stats['degraded']} degraded, {verify_stats['skipped']} skipped "
            f"(total {total_paras} paragraphs)"
        )
        return content

    # =========================================================================
    # Phase 3: Synthesis Section Generation
    # =========================================================================
    
    async def _generate_synthesis_section(
        self,
        section_type: str,
        paper_title: str,
        prior_sections: Dict[str, str],
        contributions: List[str],
        style_guide: Optional[str] = None,
        section_plan: Optional[SectionPlan] = None,
        prompt_traces: Optional[List[Dict[str, Any]]] = None,
        memory: Optional[SessionMemory] = None,
        template_guide: Optional[TemplateWriterGuide] = None,
        exemplar_guidance: Optional[str] = None,
    ) -> SectionResult:
        """Generate synthesis section (Abstract or Conclusion) via WriterAgent."""
        try:
            memory_context = memory.get_cross_section_summary() if memory else ""

            prompt = compile_synthesis_prompt(
                section_type=section_type,
                paper_title=paper_title,
                prior_sections=prior_sections,
                key_contributions=contributions,
                style_guide=style_guide,
                section_plan=section_plan,
                active_skills=self._get_active_skills(section_type, style_guide),
                memory_context=memory_context,
                template_guide=template_guide,
                exemplar_guidance=exemplar_guidance or None,
            )
            if prompt_traces is not None:
                prompt_traces.append(
                    {
                        "section_type": section_type,
                        "phase": "generation",
                        "code_context_used": False,
                        "runtime_evidence": [],
                        "prompt": prompt,
                    }
                )

            synthesis_system = (
                "You are an expert academic writer. Use present tense for methods, "
                "no contractions (it is, do not, cannot), no possessives on method "
                "names (the performance of X, not X's performance). "
                "Place key information at sentence end. Output pure LaTeX only."
            )

            result = await self._writer.run(
                system_prompt=synthesis_system,
                user_prompt=prompt,
                section_type=section_type,
                enable_review=False,
                memory=memory,
                peers={"planner": self._planner, "reviewer": self._reviewer},
            )
            content = result.get("generated_content", "")

            # Hard rule: strip ALL citations and cross-references from
            # abstract and conclusion — these must be self-contained.
            if section_type in ("abstract", "conclusion"):
                content = re.sub(r'~?\\cite\{[^}]*\}', '', content)
                # Strip "Figure~\ref{...}", "Table~\ref{...}", "Section~\ref{...}"
                content = re.sub(
                    r'(?:Figure|Fig\.|Table|Tab\.|Section|Sec\.|Equation|Eq\.)~?\\ref\{[^}]*\}',
                    '', content,
                )
                # Strip any remaining bare \ref{...}
                content = re.sub(r'~?\\ref\{[^}]*\}', '', content)
                # Clean orphaned parentheses like "(, )" or "( )"
                content = re.sub(r'\(\s*[,;]?\s*\)', '', content)
                content = re.sub(r'  +', ' ', content)

            word_count = len(content.split())
            
            # Use plan's title if available
            if section_plan and section_plan.section_title:
                section_title = section_plan.section_title
            else:
                section_title = "Abstract" if section_type == "abstract" else "Conclusion"
            
            return SectionResult(
                section_type=section_type,
                section_title=section_title,
                status="ok",
                latex_content=content,
                word_count=word_count,
            )
        except Exception as e:
            return SectionResult(
                section_type=section_type,
                status="error",
                error=str(e),
            )
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _parse_references(self, bibtex_list: List[str]) -> List[Dict[str, Any]]:
        """Parse BibTeX entries into structured format"""
        parsed = []
        
        for bibtex in bibtex_list:
            try:
                # Extract key fields using regex
                ref_id_match = re.search(r'@\w+{([^,]+),', bibtex)
                title_match = re.search(r'title\s*=\s*[{"]([^}"]+)[}"]', bibtex, re.IGNORECASE)
                author_match = re.search(r'author\s*=\s*[{"]([^}"]+)[}"]', bibtex, re.IGNORECASE)
                year_match = re.search(r'year\s*=\s*[{"]?(\d{4})[}"]?', bibtex, re.IGNORECASE)
                
                ref = {
                    "ref_id": ref_id_match.group(1) if ref_id_match else f"ref_{len(parsed)+1}",
                    "title": title_match.group(1) if title_match else "",
                    "authors": author_match.group(1) if author_match else "",
                    "year": int(year_match.group(1)) if year_match else None,
                    "bibtex": bibtex,
                }
                parsed.append(ref)
            except Exception:
                # If parsing fails, create a minimal entry
                parsed.append({
                    "ref_id": f"ref_{len(parsed)+1}",
                    "bibtex": bibtex,
                })
        
        return parsed
    
    @staticmethod
    def _validate_ref_usage(
        generated_sections: Dict[str, str],
        ref_pool: "ReferencePool",
    ) -> Dict[str, Any]:
        """
        Check that every reference in the pool is cited at least once.
        Logs warnings for uncited references and returns structured coverage.
        """
        all_content = "\n".join(generated_sections.values())
        cited_keys = ReferencePool.extract_cite_keys(all_content)
        pool_keys = ref_pool.valid_citation_keys
        uncited = pool_keys - cited_keys
        if uncited:
            print(f"[MetaDataAgent] WARNING: {len(uncited)} uncited reference(s): "
                  + ", ".join(sorted(uncited)[:10])
                  + ("..." if len(uncited) > 10 else ""))
        else:
            print(f"[MetaDataAgent] All {len(pool_keys)} pooled references are cited.")
        return {
            "cited_keys": sorted(cited_keys),
            "pool_keys": sorted(pool_keys),
            "uncited_keys": sorted(uncited),
            "coverage": (len(cited_keys & pool_keys) / len(pool_keys)) if pool_keys else 1.0,
        }

    async def _enforce_reference_coverage(
        self,
        generated_sections: Dict[str, str],
        sections_results: List[SectionResult],
        paper_plan: Optional[PaperPlan],
        metadata: PaperMetaData,
        valid_ref_keys: Set[str],
        memory: Optional[SessionMemory] = None,
        max_sections_to_revise: int = 2,
    ) -> Set[str]:
        """Delegation stub — see orchestrator.py."""
        return await self._orchestrator._enforce_reference_coverage(
            generated_sections=generated_sections,
            sections_results=sections_results,
            paper_plan=paper_plan,
            metadata=metadata,
            valid_ref_keys=valid_ref_keys,
            memory=memory,
            max_sections_to_revise=max_sections_to_revise,
        )

    def _validate_file_paths(self, metadata: PaperMetaData) -> List[str]:
        """
        Validate that all provided file paths exist before generation.
        
        Args:
            metadata: Paper metadata with figures and tables
            
        Returns:
            List of error messages (empty if all valid)
        """
        errors = []
        base_path = os.getcwd()
        
        # Validate figure file paths
        for fig in metadata.figures:
            if fig.auto_generate:
                continue  # Skip auto-generate figures
            if fig.file_path:
                # Resolve relative paths
                if os.path.isabs(fig.file_path):
                    resolved_path = fig.file_path
                else:
                    resolved_path = os.path.join(base_path, fig.file_path)
                resolved_path = os.path.normpath(resolved_path)
                
                if not os.path.exists(resolved_path):
                    errors.append(f"Figure file not found: {fig.file_path} (resolved: {resolved_path})")
        
        # Validate table file paths
        for tbl in metadata.tables:
            if tbl.auto_generate:
                continue  # Skip auto-generate tables
            if tbl.file_path:
                # Resolve relative paths
                if os.path.isabs(tbl.file_path):
                    resolved_path = tbl.file_path
                else:
                    resolved_path = os.path.join(base_path, tbl.file_path)
                resolved_path = os.path.normpath(resolved_path)
                
                if not os.path.exists(resolved_path):
                    errors.append(f"Table file not found: {tbl.file_path} (resolved: {resolved_path})")
            # Tables without file_path should have content
            elif not tbl.content and not tbl.auto_generate:
                errors.append(f"Table {tbl.id} has no file_path or content")
        
        return errors

    @staticmethod
    def _convert_figures_for_latex(metadata: "PaperMetaData") -> int:
        """
        Convert figure files to LaTeX-compatible formats (PDF preferred, then PNG).
        Mutates FigureSpec.file_path in-place if conversion is performed.

        - **Returns**:
            - `int`: Number of figures converted.
        """
        from PIL import Image as PILImage

        LATEX_OK = {".pdf", ".png", ".jpg", ".jpeg", ".eps"}
        converted = 0
        base_path = os.getcwd()

        for fig in metadata.figures:
            if fig.auto_generate or not fig.file_path:
                continue
            resolved = (
                fig.file_path if os.path.isabs(fig.file_path)
                else os.path.join(base_path, fig.file_path)
            )
            resolved = os.path.normpath(resolved)
            if not os.path.exists(resolved):
                continue
            ext = os.path.splitext(resolved)[1].lower()
            if ext in LATEX_OK:
                continue

            # Convert to PDF (preferred); fall back to PNG if PDF conversion fails
            pdf_path = os.path.splitext(resolved)[0] + ".pdf"
            try:
                img = PILImage.open(resolved)
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(pdf_path, "PDF")
                # Update FigureSpec so downstream sees the new path
                if os.path.isabs(fig.file_path):
                    fig.file_path = pdf_path
                else:
                    fig.file_path = os.path.relpath(pdf_path, base_path)
                converted += 1
                print(f"[MetaDataAgent] Converted figure {fig.id}: {ext} -> .pdf")
            except Exception as exc:
                png_path = os.path.splitext(resolved)[0] + ".png"
                try:
                    img = PILImage.open(resolved)
                    img.save(png_path, "PNG")
                    if os.path.isabs(fig.file_path):
                        fig.file_path = png_path
                    else:
                        fig.file_path = os.path.relpath(png_path, base_path)
                    converted += 1
                    print(f"[MetaDataAgent] Converted figure {fig.id}: {ext} -> .png")
                except Exception as png_exc:
                    print(f"[MetaDataAgent] WARNING: Cannot convert {fig.id} "
                          f"({ext}): pdf={exc}, png={png_exc}")
        return converted

    def _collect_figure_paths(
        self, 
        figures: List[FigureSpec], 
        base_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Collect figure file paths from FigureSpec list
        
        Args:
            figures: List of FigureSpec objects
            base_path: Base directory for resolving relative paths
                       (typically the directory containing the metadata JSON)
            
        Returns:
            Dict mapping figure ID to absolute file path
        """
        import os
        paths = {}
        for fig in figures:
            if fig.auto_generate:
                print(f"[MetaDataAgent] Figure auto-generation not implemented: {fig.id}")
                continue
            if fig.file_path:
                # Resolve relative paths against base_path
                if base_path and not os.path.isabs(fig.file_path):
                    resolved_path = os.path.join(base_path, fig.file_path)
                else:
                    resolved_path = fig.file_path
                
                # Normalize the path
                resolved_path = os.path.normpath(resolved_path)
                
                if os.path.exists(resolved_path):
                    paths[fig.id] = resolved_path
                else:
                    print(f"[MetaDataAgent] Warning: Figure file not found: {resolved_path}")
        return paths
    
    def _assemble_paper(
        self,
        title: str,
        sections: Dict[str, str],
        references: List[Dict[str, Any]],
        valid_citation_keys: set = None,
    ) -> str:
        """Assemble complete LaTeX document with final citation validation"""
        
        # Extract valid keys if not provided
        if valid_citation_keys is None:
            valid_citation_keys = self._extract_valid_citation_keys(references)
        
        # Basic LaTeX template
        latex = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{natbib}

\title{""" + self._escape_latex(title) + r"""}
\author{Author Names}
\date{\today}

\begin{document}

\maketitle

"""
        # Add abstract — strip any LLM-generated abstract boilerplate
        if "abstract" in sections:
            abstract_text = sections["abstract"].strip()
            abstract_text = re.sub(r'\\title\{[^}]*\}\s*', '', abstract_text)
            abstract_text = re.sub(r'\\maketitle\s*', '', abstract_text)
            abstract_text = re.sub(r'\\begin\{abstract\}\s*', '', abstract_text)
            abstract_text = re.sub(r'\s*\\end\{abstract\}', '', abstract_text)
            abstract_text = abstract_text.strip()
            latex += r"\begin{abstract}" + "\n"
            latex += abstract_text + "\n"
            latex += r"\end{abstract}" + "\n\n"
        
        # Add sections in order (handle dynamic sections)
        _default_order = ["introduction", "related_work", "method", "experiment", "result", "discussion", "conclusion"]
        _default_titles = {
            "introduction": "Introduction",
            "related_work": "Related Work",
            "method": "Methodology",
            "experiment": "Experiments",
            "result": "Results",
            "discussion": "Discussion",
            "conclusion": "Conclusion",
        }
        # Build final order: known sections first, then any extras
        section_order = [s for s in _default_order if s in sections]
        for s in sections:
            if s != "abstract" and s not in section_order:
                section_order.append(s)

        for section_type in section_order:
            if section_type in sections and sections[section_type]:
                title = _default_titles.get(section_type, section_type.replace("_", " ").title())
                latex += f"\\section{{{title}}}\n"
                content = re.sub(r'\\section\*?\s*\{[^}]*\}\s*(?:\\label\{[^}]*\}\s*)?', '', sections[section_type])
                content = self._fix_latex_references(content)
                content, invalid, valid = self._validate_and_fix_citations(
                    content, valid_citation_keys, remove_invalid=True
                )
                if invalid:
                    print(f"[Assemble] Removed {len(invalid)} invalid citations from {section_type}: {invalid[:5]}")
                latex += content + "\n\n"
        
        # Add bibliography
        latex += r"""
\bibliographystyle{plainnat}
\bibliography{references}

\end{document}
"""
        
        return latex
    
    def _fix_latex_references(self, content: str) -> str:
        """
        Fix common LaTeX reference syntax errors generated by LLM.
        
        Fixes:
        - \\reftab{id} -> \\ref{tab:id}
        - \\reffig{id} -> \\ref{fig:id}
        - \\ref{id} (without prefix) -> \\ref{tab:id} or \\ref{fig:id} based on context
        - \\%---... comment dividers (escaped percent) -> removed entirely
        """
        import re
        
        # Remove escaped comment dividers (\%--- or \%===)
        # These appear as literal text in PDF because \% is an escaped percent sign
        content = re.sub(r'\\%[-=]+\s*\n?', '', content)
        
        # Also remove lines that are just escaped percent signs with dashes
        content = re.sub(r'^\s*\\%[-=]+.*$', '', content, flags=re.MULTILINE)
        
        # Clean up multiple blank lines that may result
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Fix \reftab{id} -> \ref{tab:id}
        content = re.sub(r'\\reftab\{([^}]+)\}', r'\\ref{tab:\1}', content)
        
        # Fix \reffig{id} -> \ref{fig:id}
        content = re.sub(r'\\reffig\{([^}]+)\}', r'\\ref{fig:\1}', content)
        
        # Fix Table~\ref{id} -> Table~\ref{tab:id} (when missing tab: prefix)
        # Only if not already prefixed with tab: or fig:
        content = re.sub(
            r'(Table~?\\ref\{)(?!tab:|fig:)([^}]+)\}',
            r'\1tab:\2}',
            content
        )
        
        # Fix Figure~\ref{id} -> Figure~\ref{fig:id} (when missing fig: prefix)
        content = re.sub(
            r'(Figure~?\\ref\{)(?!tab:|fig:)([^}]+)\}',
            r'\1fig:\2}',
            content
        )
        
        return content
    
    def _extract_valid_citation_keys(self, parsed_refs: List[Dict[str, Any]]) -> set:
        """
        Extract valid citation keys from parsed references.
        
        Args:
            parsed_refs: List of parsed reference dictionaries
            
        Returns:
            Set of valid citation keys
        """
        keys = set()
        for ref in parsed_refs:
            ref_id = ref.get("ref_id", "")
            if ref_id:
                keys.add(ref_id)
        return keys
    
    def _validate_and_fix_citations(
        self, 
        content: str, 
        valid_keys: set,
        remove_invalid: bool = True
    ) -> tuple:
        """
        Validate citations in content and optionally remove invalid ones.
        
        Args:
            content: LaTeX content with \cite{} commands
            valid_keys: Set of valid citation keys
            remove_invalid: Whether to remove invalid citations
            
        Returns:
            Tuple of (fixed_content, list_of_invalid_keys, list_of_valid_keys_used)
        """
        import re
        
        # Find all citation keys in the content
        # Handle \cite{key1, key2}, \citep{key}, \citet{key}, etc.
        # Captures group(1) = command name, group(2) = keys string
        cite_pattern = r'\\(cite[pt]?|citeauthor|citeyear|citealt|citealp)\{([^}]+)\}'
        
        invalid_keys = []
        valid_keys_used = []
        
        def process_cite(match):
            cmd_name = match.group(1)   # e.g. "cite", "citep", "citet"
            cite_content = match.group(2)
            keys = [k.strip() for k in cite_content.split(',')]
            
            valid_in_cite = []
            for key in keys:
                if key in valid_keys:
                    valid_in_cite.append(key)
                    if key not in valid_keys_used:
                        valid_keys_used.append(key)
                else:
                    if key not in invalid_keys:
                        invalid_keys.append(key)
            
            if remove_invalid:
                if valid_in_cite:
                    return f'\\{cmd_name}{{{", ".join(valid_in_cite)}}}'
                else:
                    # All keys invalid, remove entire citation
                    return ''
            else:
                return match.group(0)
        
        fixed_content = re.sub(cite_pattern, process_cite, content)
        
        # Clean up empty citations and dangling text
        # Remove empty citation commands: \cite{}, \citep{}, \citet{}, etc.
        fixed_content = re.sub(r'\\(?:cite[pt]?|citeauthor|citeyear|citealt|citealp)\{\s*\}', '', fixed_content)
        # Clean up double spaces
        fixed_content = re.sub(r'  +', ' ', fixed_content)
        # Clean up space before punctuation
        fixed_content = re.sub(r' +([.,;:])', r'\1', fixed_content)
        
        return fixed_content, invalid_keys, valid_keys_used
    
    def _ensure_figures_defined(
        self,
        generated_sections: Dict[str, str],
        paper_plan: Optional[PaperPlan],
        figures: Optional[List[FigureSpec]],
    ) -> Dict[str, str]:
        """
        Ensure all figures assigned for definition have their environments created.
        
        If a figure is in section_plan.figures_to_define but no \\begin{figure} 
        exists with matching label, inject the figure environment.
        
        Args:
            generated_sections: Dict of section_type -> latex_content
            paper_plan: Paper plan with figure assignments
            figures: List of figure specifications
            
        Returns:
            Updated generated_sections dict
        """
        import re
        
        if not paper_plan or not figures:
            return generated_sections
        
        # Build figure lookup
        figure_map = {fig.id: fig for fig in figures}
        
        # Pre-scan ALL sections (including appendix) to find which figures
        # already have their environments defined somewhere in the paper.
        # This prevents re-injecting figures that were moved to the appendix
        # by structural overflow actions.
        globally_defined_figs: set = set()
        globally_used_targets: set = set()
        all_content = "\n".join(generated_sections.values())
        target_pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
        for tgt in re.findall(target_pattern, all_content):
            norm = str(tgt).strip()
            if norm:
                globally_used_targets.add(norm)
                globally_used_targets.add(Path(norm).name)
                globally_used_targets.add(Path(norm).stem)
        for fig in figures:
            fig_pattern = rf'\\begin{{figure\*?}}.*?\\label{{{re.escape(fig.id)}}}.*?\\end{{figure\*?}}'
            if re.search(fig_pattern, all_content, re.DOTALL):
                globally_defined_figs.add(fig.id)
        
        for section in paper_plan.sections:
            section_type = section.section_type
            figures_to_define = section.get_figure_ids_to_define()
            
            if not figures_to_define or section_type not in generated_sections:
                continue
            
            content = generated_sections[section_type]
            
            for fig_id in figures_to_define:
                fig = figure_map.get(fig_id)
                if not fig:
                    continue
                
                # Skip if this figure is already defined ANYWHERE in the paper
                # (including appendix — it may have been moved there deliberately)
                if fig_id in globally_defined_figs:
                    continue
                # Also skip if the same underlying image target already appears
                # via includegraphics with a different label/alias.
                aliases = {fig_id, Path(fig_id).name, Path(fig_id).stem}
                if getattr(fig, "file_path", ""):
                    fp = str(fig.file_path)
                    aliases.update({Path(fp).name, Path(fp).stem})
                if any(a for a in aliases if a in globally_used_targets):
                    continue
                
                # Figure not defined anywhere - inject it
                print(f"[EnsureFigures] Injecting missing figure '{fig_id}' in '{section_type}'")
                
                # Determine environment and width based on wide flag
                env_name = "figure*" if fig.wide else "figure"
                width = "\\textwidth" if fig.wide else "0.9\\linewidth"
                
                # Build figure LaTeX
                figure_latex = f"""
\\begin{{{env_name}}}[htbp]
\\centering
\\includegraphics[width={width}]{{{fig_id}}}
\\caption{{{fig.caption}}}\\label{{{fig_id}}}
\\end{{{env_name}}}
"""
                
                # Find a good insertion point:
                # 1. After the first paragraph that mentions this figure
                # 2. Or at the end of the section
                ref_pattern = rf'(Figure~?\\ref{{{re.escape(fig_id)}}}[^.]*\.)'
                match = re.search(ref_pattern, content)
                if match:
                    # Insert after the sentence that references the figure
                    insert_pos = match.end()
                    content = content[:insert_pos] + "\n" + figure_latex + content[insert_pos:]
                else:
                    # Insert at the beginning of the section
                    content = figure_latex + "\n" + content
                
                generated_sections[section_type] = content
        
        return generated_sections

    @staticmethod
    def _deduplicate_figure_environments(
        generated_sections: Dict[str, str],
        section_order: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Remove duplicate figure/table environments across and within sections.
        - **Description**:
            - Scans all sections for \\begin{figure}...\\end{figure} blocks
            - Keeps only the FIRST occurrence of each \\label{...}
            - Removes subsequent duplicates (both within and across sections)
            - Preserves \\ref{} references to figures defined elsewhere

        - **Args**:
            - `generated_sections` (Dict[str, str]): section_type -> LaTeX content
            - `section_order` (List[str], optional): Preferred ordering for priority

        - **Returns**:
            - Updated generated_sections with duplicates removed
        """
        import re

        # Determine processing order: use section_order if given
        if section_order:
            ordered_keys = [k for k in section_order if k in generated_sections]
            ordered_keys += [k for k in generated_sections if k not in ordered_keys]
        else:
            ordered_keys = list(generated_sections.keys())

        seen_labels: set = set()
        total_removed = 0

        for section_type in ordered_keys:
            content = generated_sections.get(section_type, "")
            if not content:
                continue

            # Match figure and figure* environments
            env_pattern = re.compile(
                r'\\begin\{(figure\*?)\}.*?\\end\{\1\}',
                re.DOTALL,
            )

            new_content = content
            offset = 0

            for m in env_pattern.finditer(content):
                block = m.group(0)
                labels = re.findall(r'\\label\{([^}]+)\}', block)
                label = labels[0] if labels else None

                if label and label in seen_labels:
                    # Duplicate — remove the entire figure environment
                    start = m.start() + offset
                    end = m.end() + offset
                    new_content = new_content[:start] + new_content[end:]
                    offset += -(m.end() - m.start())
                    total_removed += 1
                elif label:
                    seen_labels.add(label)

            # Also remove within-section duplicates for unlabeled figures
            # by checking includegraphics targets
            generated_sections[section_type] = new_content.strip()

        if total_removed > 0:
            print(f"[DeduplicateFigures] Removed {total_removed} duplicate figure environments")

        return generated_sections

    @staticmethod
    def _enforce_table_placement(
        sections: Dict[str, str],
        table_assignments: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Remove table environments from sections they are not assigned to.
        - **Description**:
            - For each \\begin{table}...\\end{table} block, checks its \\label{}.
            - If the label has an assignment in table_assignments and the
              current section is NOT the assigned section, the block is removed.
            - Tables with no assignment or no label are left in place.

        - **Args**:
            - `sections` (Dict[str, str]): section_type -> LaTeX content
            - `table_assignments` (Dict[str, str]): label -> assigned section_type

        - **Returns**:
            - `Dict[str, str]`: Sections with misplaced tables removed.
        """
        import re

        if not table_assignments:
            return sections

        result = dict(sections)
        total_removed = 0

        for section_type, content in sections.items():
            if not content:
                continue

            env_pattern = re.compile(
                r'\\begin\{(table\*?)\}.*?\\end\{\1\}',
                re.DOTALL,
            )

            new_content = content
            offset = 0

            for m in env_pattern.finditer(content):
                block = m.group(0)
                labels = re.findall(r'\\label\{([^}]+)\}', block)
                label = labels[0] if labels else None

                if label and label in table_assignments:
                    assigned_section = table_assignments[label]
                    if section_type != assigned_section:
                        start = m.start() + offset
                        end = m.end() + offset
                        new_content = new_content[:start] + new_content[end:]
                        offset -= (m.end() - m.start())
                        total_removed += 1

            result[section_type] = new_content.strip()

        if total_removed > 0:
            print(f"[EnforceTablePlacement] Removed {total_removed} misplaced table(s)")

        return result

    @staticmethod
    def _strip_code_path_references(
        generated_sections: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Remove raw code file path references from all generated sections.
        - **Description**:
            - Strips \\texttt{code/...} and \\texttt{*.py/.c/.cpp/.R/.jl} patterns
            - Removes surrounding prose like "implemented in \\texttt{...}"
            - Acts as a safety net in case the Writer ignores prompt rules

        - **Args**:
            - `generated_sections` (Dict[str, str]): section_type -> LaTeX content

        - **Returns**:
            - Updated generated_sections with code paths removed
        """
        import re as _re

        total_stripped = 0
        for section_type in list(generated_sections.keys()):
            content = generated_sections[section_type]
            original = content

            # Pattern 1: \texttt{code/...} or \texttt{path/to/file.py}
            content = _re.sub(
                r'(?:,?\s*(?:implemented|defined|derived|found|coded|written|specified|described)'
                r'\s+(?:in|from|within|via|using)\s+)?'
                r'\\texttt\{[^}]*(?:\.py|\.c|\.cc|\.cpp|\.R|\.jl|\.ipynb|code/)[^}]*\}',
                '', content,
                flags=_re.IGNORECASE,
            )
            # Pattern 2: \texttt{function_name} derived from \texttt{code/file.py}
            content = _re.sub(
                r'\s*\((?:derived\s+from|from|in|see)\s+\\texttt\{[^}]*(?:\.py|\.c|\.cpp|code/)[^}]*\}\)',
                '', content,
                flags=_re.IGNORECASE,
            )
            # Clean double spaces and orphaned commas
            content = _re.sub(r'\s*,\s*,', ',', content)
            content = _re.sub(r'  +', ' ', content)

            if content != original:
                total_stripped += 1
                generated_sections[section_type] = content

        if total_stripped > 0:
            print(f"[StripCodePaths] Cleaned code path references in {total_stripped} section(s)")

        return generated_sections

    def _ensure_tables_defined(
        self,
        generated_sections: Dict[str, str],
        paper_plan: Optional[PaperPlan],
        tables: Optional[List[TableSpec]],
        converted_tables: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Direct-injection pipeline: strip Writer-generated tables, then inject
        pre-converted tables at the first \\ref location.
        - **Description**:
            - Step 1: Strip any table environments the Writer created for tables
              that have pre-converted LaTeX (defensive).
            - Step 2: Inject the authoritative pre-converted table at the first
              ``Table~\\ref{tab:id}`` in each assigned section.
            - This replaces the old "safety-net" approach; tables are now
              always injected from the converted pool, never left to the Writer.

        - **Args**:
            - `generated_sections` (Dict[str, str]): Section contents keyed by type.
            - `paper_plan` (Optional[PaperPlan]): Paper plan with table assignments.
            - `tables` (Optional[List[TableSpec]]): Table specifications.
            - `converted_tables` (Optional[Dict[str, str]]): table_id -> LaTeX code.

        - **Returns**:
            - `generated_sections` (Dict[str, str]): Updated sections dict.
        """
        from ..shared.table_converter import strip_writer_tables, inject_tables

        if not paper_plan or not tables:
            return generated_sections

        _converted = converted_tables or {}
        known_ids = set(_converted.keys())

        print(f"[DirectInject] Starting: {len(_converted)} converted tables, "
              f"sections={list(generated_sections.keys())}")

        for section in paper_plan.sections:
            section_type = section.section_type
            if section_type not in generated_sections:
                continue

            tables_to_define = section.get_table_ids_to_define()
            if not tables_to_define:
                continue

            print(f"[DirectInject] Section '{section_type}' has tables_to_define={tables_to_define}")

            content = generated_sections[section_type]

            stripped = strip_writer_tables(content, known_ids)
            if stripped != content:
                stripped_ids = known_ids & set(tables_to_define)
                if stripped_ids:
                    print(f"[DirectInject] Stripped Writer tables {stripped_ids} in '{section_type}'")

            result = inject_tables(stripped, section, tables, _converted)
            if result != stripped:
                print(f"[DirectInject] Injected tables in '{section_type}' "
                      f"(content grew {len(stripped)}->{len(result)} chars)")
            generated_sections[section_type] = result

        return generated_sections

    @staticmethod
    def _normalize_float_placement(content: str) -> str:
        """
        Normalize figure/table placement hints to reduce end-of-document float piles.
        - **Description**:
            - Rewrites strict top-only placement ([t]) to [htbp].
            - Leaves existing broader placement options unchanged.

        - **Args**:
            - `content` (str): Section LaTeX content.

        - **Returns**:
            - `str`: Content with normalized float placement.
        """
        if not content:
            return content
        content = re.sub(r'\\begin\{figure\*?\}\[t\]', lambda m: m.group(0).replace('[t]', '[htbp]'), content)
        content = re.sub(r'\\begin\{table\*?\}\[t\]', lambda m: m.group(0).replace('[t]', '[htbp]'), content)
        return content

    def _collect_typesetter_figure_ids(
        self,
        generated_sections: Dict[str, str],
        figures: Optional[List[FigureSpec]],
        figure_paths: Optional[Dict[str, str]],
    ) -> List[str]:
        """
        Collect figure identifiers needed by the typesetter.
        - **Description**:
            - Combines IDs from metadata, includegraphics tags, and figure-path keys.
            - Keeps only likely IDs (not explicit file paths).

        - **Args**:
            - `generated_sections` (Dict[str, str]): Section contents.
            - `figures` (Optional[List[FigureSpec]]): Figure specs from metadata.
            - `figure_paths` (Optional[Dict[str, str]]): Explicit figure-id to file-path mapping.

        - **Returns**:
            - `figure_ids` (List[str]): Deduplicated figure IDs for typesetter resource resolution.
        """
        ids: set[str] = set()
        for fig in (figures or []):
            if getattr(fig, "id", None):
                ids.add(str(fig.id))
        for key in (figure_paths or {}).keys():
            if key:
                ids.add(str(key))

        pattern = r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}"
        for content in generated_sections.values():
            for raw in re.findall(pattern, content or ""):
                token = str(raw).strip()
                if not token:
                    continue
                if "/" in token or token.startswith(".") or token.endswith(
                    (".png", ".jpg", ".jpeg", ".pdf", ".svg")
                ):
                    continue
                ids.add(token)
        return sorted(ids)

    # =========================================================================
    # Pre-Generation Search Judgment (Phase A)
    # =========================================================================

    async def _judge_search_need(
        self,
        section_type: str,
        section_title: str,
        paper_title: str,
        key_points: List[str],
        ref_pool: ReferencePool,
    ) -> Dict[str, Any]:
        """Delegation stub — see orchestrator.py."""
        return await self._orchestrator._judge_search_need(
            section_type=section_type,
            section_title=section_title,
            paper_title=paper_title,
            key_points=key_points,
            ref_pool=ref_pool,
        )

    async def _execute_pre_searches(
        self,
        queries: List[str],
        ref_pool: ReferencePool,
    ) -> int:
        """Delegation stub — see orchestrator.py."""
        return await self._orchestrator._execute_pre_searches(
            queries=queries,
            ref_pool=ref_pool,
        )

    def _validate_and_merge_new_references(
        self,
        content: str,
        msg_history: List[Dict[str, Any]],
        ref_pool: ReferencePool,
    ) -> str:
        """
        Two-layer validation of references discovered via search_papers.

        - **Description**:
            - Layer 1 (LLM judgment): The LLM already decided which papers
              to cite during the ReAct loop. We extract those decisions.
            - Layer 2 (System cross-reference): We verify that every \\cite{}
              key in the generated content either:
              (a) already exists in ref_pool (core or previously discovered), or
              (b) matches a BibTeX entry returned by search_papers — in which
                  case we add it to ref_pool as a discovered reference.
            - Keys that match neither (hallucinated) are removed from the content.

        - **Args**:
            - `content` (str): Generated LaTeX content.
            - `msg_history` (List[dict]): Full message history from react_loop.
            - `ref_pool` (ReferencePool): Persistent reference pool to update.

        - **Returns**:
            - `str`: Content with hallucinated citations removed.
        """
        # Extract BibTeX entries from search_papers tool results in message history
        search_results = ReferencePool.extract_search_results_from_history(msg_history)
        if search_results:
            print(f"[ValidateRefs] Found {len(search_results)} papers from search results")

        # Extract all \cite{} keys from the generated content
        cited_keys = ReferencePool.extract_cite_keys(content)
        if not cited_keys:
            return content

        print(f"[ValidateRefs] Content cites {len(cited_keys)} keys: {cited_keys}")

        # Check each cited key
        hallucinated_keys = []
        for key in cited_keys:
            if ref_pool.has_key(key):
                # Already in pool (core or previously discovered)
                continue

            if key in search_results:
                # Found in search results — add to pool
                added = ref_pool.add_discovered(key, search_results[key], source="search")
                if added:
                    print(f"[ValidateRefs] Added discovered ref: {key}")
                else:
                    print(f"[ValidateRefs] Duplicate discovered ref (skipped): {key}")
            else:
                # Not in pool and not from search — hallucinated
                hallucinated_keys.append(key)
                print(f"[ValidateRefs] Hallucinated key removed: {key}")

        # Remove hallucinated citations from content
        for key in hallucinated_keys:
            content = ReferencePool.remove_citation(content, key)

        if hallucinated_keys:
            print(f"[ValidateRefs] Removed {len(hallucinated_keys)} hallucinated citations")

        return content

    def _generate_bib_file(self, references: List[Dict[str, Any]]) -> str:
        """Generate .bib file content from parsed references"""
        bib_entries = []
        for ref in references:
            if ref.get("bibtex"):
                bib_entries.append(ref["bibtex"])
            else:
                # Generate a minimal entry
                ref_id = ref.get("ref_id", "unknown")
                title = ref.get("title", "Unknown Title")
                authors = ref.get("authors", "Unknown Author")
                year = ref.get("year", 2024)
                
                entry = f"""@article{{{ref_id},
  title = {{{title}}},
  author = {{{authors}}},
  year = {{{year}}},
}}"""
                bib_entries.append(entry)
        
        return "\n\n".join(bib_entries)
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters"""
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    # =========================================================================
    # Phase 4: PDF Compilation
    # =========================================================================
    
    async def _compile_pdf(
        self,
        generated_sections: Dict[str, str],
        template_path: str,
        references: List[Dict[str, Any]],
        output_dir: Path,
        paper_title: str,
        figures_source_dir: Optional[str] = None,
        figure_paths: Optional[Dict[str, str]] = None,
        converted_tables: Optional[Dict[str, str]] = None,
        paper_plan: Optional[PaperPlan] = None,
        figures: Optional[List[FigureSpec]] = None,
        metadata_tables: Optional[List[TableSpec]] = None,
    ) -> Tuple[Optional[str], Optional[str], List[str], Dict[str, List[str]]]:
        """
        Compile PDF using Typesetter Agent (multi-file mode).
        - **Description**:
            - Passes sections as a dict to the TypesetterAgent, which writes each
              section to its own .tex file and uses \\input{} in main.tex
            - This enables precise error-to-section mapping from LaTeX logs

        - **Args**:
            - `generated_sections` (Dict[str, str]): Section contents
            - `template_path` (str): Path to .zip template file
            - `references` (List[Dict[str, Any]]): Parsed reference list
            - `output_dir` (Path): Output directory
            - `paper_title` (str): Paper title
            - `figures_source_dir` (Optional[str]): Directory with figure files (legacy)
            - `figure_paths` (Optional[Dict[str, str]]): figure_id -> file_path
            - `converted_tables` (Optional[Dict[str, str]]): table_id -> LaTeX table code
            - `paper_plan` (Optional[PaperPlan]): Paper plan with figure/table assignments
            - `figures` (Optional[List[FigureSpec]]): Figure specifications
            - `metadata_tables` (Optional[List[TableSpec]]): Table specifications

        - **Returns**:
            - Tuple of (pdf_path, latex_path, compile_errors, section_errors)
            - On success: (pdf_path, latex_path, [], {})
            - On failure: (None, None, [error1, ...], {"section_type": [errors]})
        """
        print(f"[MetaDataAgent] Phase 4: Compiling PDF with template: {template_path}")
        if not (paper_title or "").strip():
            return None, None, ["missing_or_empty_paper_title"], {}
        
        try:
            # Dynamic: read section order and titles from the plan
            if paper_plan:
                section_order = paper_plan.get_compile_section_order()
                section_titles = paper_plan.get_section_titles()
            else:
                section_order = ["introduction", "related_work", "method", "experiment", "result", "conclusion"]
                section_titles = {
                    "introduction": "Introduction",
                    "related_work": "Related Work",
                    "method": "Methodology",
                    "experiment": "Experiments",
                    "result": "Results",
                    "conclusion": "Conclusion",
                }
            # Include any generated sections not in plan (e.g. appendix from review loop)
            for st in generated_sections:
                if st != "abstract" and st not in section_order:
                    section_order.append(st)
                if st not in section_titles:
                    section_titles[st] = st.replace("_", " ").title()
            
            # Final citation validation pass (safety net)
            valid_citation_keys = self._extract_valid_citation_keys(references)
            total_invalid_removed = 0
            for section_type in list(generated_sections.keys()):
                content = generated_sections[section_type]
                content = self._fix_latex_references(content)
                content = self._normalize_float_placement(content)
                fixed_content, invalid, valid = self._validate_and_fix_citations(
                    content, valid_citation_keys, remove_invalid=True
                )
                if invalid:
                    print(f"[CompilePDF] Removed {len(invalid)} invalid citations from {section_type}: {invalid[:3]}{'...' if len(invalid) > 3 else ''}")
                    total_invalid_removed += len(invalid)
                generated_sections[section_type] = fixed_content

            # Cross-section label validation: remove \ref{} to undefined labels
            from ..shared.label_registry import collect_all_labels, validate_and_fix_refs
            valid_labels = collect_all_labels(generated_sections)
            for section_type in list(generated_sections.keys()):
                generated_sections[section_type] = validate_and_fix_refs(
                    generated_sections[section_type], valid_labels
                )
            
            if total_invalid_removed > 0:
                print(f"[CompilePDF] Total invalid citations removed: {total_invalid_removed}")

            # Strip raw code file path references from all sections
            generated_sections = self._strip_code_path_references(generated_sections)

            # Ensure all assigned figures have their environments created
            if paper_plan and figures:
                generated_sections = self._ensure_figures_defined(
                    generated_sections=generated_sections,
                    paper_plan=paper_plan,
                    figures=figures,
                )

            # Deduplicate figure environments (cross-section and within-section)
            generated_sections = self._deduplicate_figure_environments(
                generated_sections,
                section_order=section_order,
            )

            # Enforce table placement according to planner assignments
            if paper_plan:
                table_assignments: Dict[str, str] = {}
                for sec in paper_plan.sections:
                    if sec.tables:
                        for tdef in sec.tables:
                            label = getattr(tdef, "label", None) or getattr(tdef, "table_id", None)
                            if label:
                                table_assignments[label] = sec.section_type
                if table_assignments:
                    generated_sections = self._enforce_table_placement(
                        generated_sections, table_assignments,
                    )

            # Ensure all assigned tables have their environments created
            if paper_plan and metadata_tables:
                generated_sections = self._ensure_tables_defined(
                    generated_sections=generated_sections,
                    paper_plan=paper_plan,
                    tables=metadata_tables,
                    converted_tables=converted_tables,
                )
            
            # Prepare references for Typesetter
            typesetter_refs = []
            for ref in references:
                if ref.get("bibtex"):
                    typesetter_refs.append({
                        "ref_id": ref.get("ref_id", ""),
                        "bibtex": ref.get("bibtex"),
                    })
            figure_ids = self._collect_typesetter_figure_ids(
                generated_sections=generated_sections,
                figures=figures,
                figure_paths=figure_paths,
            )
            
            from ..typesetter_agent.models import TemplateConfig
            ts_template_config = TemplateConfig(
                paper_title=paper_title,
                paper_authors="EasyPaper",
            )

            # Promote wide tables to table* in double-column templates
            if ts_template_config.column_format == "double":
                from ..shared.table_converter import (
                    add_adjustbox_safety,
                    smart_promote_wide_tables,
                )
                for sec_type in list(generated_sections.keys()):
                    tex = smart_promote_wide_tables(generated_sections[sec_type])
                    generated_sections[sec_type] = add_adjustbox_safety(tex)

            # Prefer in-process peer TypesetterAgent (SDK mode); fall back to HTTP.
            if self._typesetter is not None:
                print("[MetaDataAgent] Using in-process Typesetter Agent")
                ts_state = await self._typesetter.run(
                    sections=generated_sections,
                    section_order=section_order,
                    section_titles=section_titles,
                    template_path=template_path,
                    template_config=ts_template_config,
                    references=typesetter_refs,
                    figure_ids=figure_ids,
                    output_dir=str(output_dir),
                    figures_source_dir=figures_source_dir,
                    figure_paths=figure_paths or {},
                    converted_tables=converted_tables or {},
                )
                return self._parse_typesetter_result(ts_state, output_dir)

            # HTTP fallback (server mode — TypesetterAgent running as a separate service)
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{os.getenv('AGENTSYS_SELF_URL', 'http://127.0.0.1:8000')}/agent/typesetter/compile",
                    json={
                        "request_id": str(uuid.uuid4()),
                        "payload": {
                            "sections": generated_sections,
                            "section_order": section_order,
                            "section_titles": section_titles,
                            "template_path": template_path,
                            "template_config": {
                                "paper_title": paper_title,
                                "paper_authors": "EasyPaper",
                            },
                            "references": typesetter_refs,
                            "figure_ids": figure_ids,
                            "output_dir": str(output_dir),
                            "figures_source_dir": figures_source_dir,
                            "figure_paths": figure_paths or {},
                            "converted_tables": converted_tables or {},
                        }
                    }
                )
                
                if response.status_code != 200:
                    print(f"[MetaDataAgent] Typesetter error: {response.status_code} - {response.text}")
                    return None, None, [f"Typesetter HTTP {response.status_code}"], {}
                
                result = response.json()
                
                if result.get("status") == "ok" and result.get("result"):
                    compilation_result = result["result"]
                    pdf_path = compilation_result.get("pdf_path")
                    latex_path = compilation_result.get("source_path")
                    compile_warnings = compilation_result.get("warnings", [])
                    section_errors = compilation_result.get("section_errors", {})
                    
                    if pdf_path:
                        print(f"[MetaDataAgent] PDF compiled successfully: {pdf_path}")
                        if compile_warnings:
                            print(f"[MetaDataAgent] Compile warnings: {compile_warnings[:5]}")
                        if section_errors:
                            print(f"[MetaDataAgent] Section errors (on success): {section_errors}")
                    else:
                        print(f"[MetaDataAgent] PDF compilation failed: compilation result has no pdf_path")

                    # Guard: ensure final main.tex contains required structure.
                    if latex_path:
                        main_tex_path = Path(latex_path) / "main.tex"
                        structure_errors = self._validate_main_tex_structure(main_tex_path)
                        if structure_errors:
                            print(f"[MetaDataAgent] main.tex structure validation failed: {structure_errors}")
                            return None, None, structure_errors, section_errors

                    return pdf_path, latex_path, [], section_errors
                else:
                    compile_errors: List[str] = []
                    section_errors: Dict[str, List[str]] = {}
                    error_msg = result.get("error", "Unknown error")
                    if result.get("result"):
                        compile_errors = result["result"].get("errors", [])
                        section_errors = result["result"].get("section_errors", {})
                    if not compile_errors:
                        compile_errors = [e.strip() for e in error_msg.split(";") if e.strip()]
                    print(f"[MetaDataAgent] PDF compilation failed: {error_msg}")
                    print(f"[Typesetter] Compile errors: {compile_errors}")
                    if section_errors:
                        print(f"[Typesetter] Section errors: {section_errors}")
                    return None, None, compile_errors, section_errors
                
        except httpx.ConnectError:
            print("[MetaDataAgent] Error: Could not connect to Typesetter Agent")
            self._save_compile_error_log(output_dir, ["Could not connect to Typesetter Agent"])
            return None, None, ["Could not connect to Typesetter Agent"], {}
        except Exception as e:
            print(f"[MetaDataAgent] PDF compilation error: {e}")
            self._save_compile_error_log(output_dir, [str(e)])
            return None, None, [str(e)], {}

    @staticmethod
    def _save_compile_error_log(
        output_dir: Path,
        errors: List[str],
    ) -> None:
        """
        Write a compile_errors.json to the iteration directory so failures
        are diagnosable even when the TypesetterAgent crashes before
        producing any output files.
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            error_path = output_dir / "compile_errors.json"
            error_path.write_text(
                json.dumps({"errors": errors}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _parse_typesetter_result(
        self,
        ts_state: Dict[str, Any],
        output_dir: Path,
    ) -> Tuple[Optional[str], Optional[str], List[str], Dict[str, List[str]]]:
        """
        Extract (pdf_path, latex_path, errors, section_errors) from a
        TypesetterAgent in-process LangGraph state dict.
        """
        cr = ts_state.get("compilation_result")
        if cr is None:
            print("[MetaDataAgent] Typesetter returned no compilation_result")
            return None, None, ["Typesetter returned no compilation_result"], {}

        pdf_path = cr.pdf_path if hasattr(cr, "pdf_path") else cr.get("pdf_path")
        latex_path = cr.source_path if hasattr(cr, "source_path") else cr.get("source_path")
        errors = (cr.errors if hasattr(cr, "errors") else cr.get("errors", [])) or []
        section_errors = (
            cr.section_errors if hasattr(cr, "section_errors")
            else cr.get("section_errors", {})
        ) or {}
        warnings = (cr.warnings if hasattr(cr, "warnings") else cr.get("warnings", [])) or []
        success = cr.success if hasattr(cr, "success") else cr.get("success", False)

        if success and pdf_path:
            print(f"[MetaDataAgent] PDF compiled successfully: {pdf_path}")
            if warnings:
                print(f"[MetaDataAgent] Compile warnings: {warnings[:5]}")
            if section_errors:
                print(f"[MetaDataAgent] Section errors (on success): {section_errors}")
            if latex_path:
                main_tex_path = Path(latex_path) / "main.tex"
                structure_errors = self._validate_main_tex_structure(main_tex_path)
                if structure_errors:
                    print(f"[MetaDataAgent] main.tex structure validation failed: {structure_errors}")
                    return None, None, structure_errors, section_errors
            return pdf_path, latex_path, [], section_errors

        print(f"[MetaDataAgent] PDF compilation failed: {errors}")
        if section_errors:
            print(f"[Typesetter] Section errors: {section_errors}")
        return None, None, errors or ["Typesetter compilation failed"], section_errors

    @staticmethod
    def _validate_main_tex_structure(main_tex_path: Path) -> List[str]:
        """
        Validate that compiled main.tex contains non-empty title/abstract.
        - **Returns**:
            - `List[str]`: Validation errors; empty list means pass.
        """
        if not main_tex_path.exists():
            return [f"main.tex not found: {main_tex_path}"]
        try:
            text = main_tex_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return [f"cannot read main.tex: {e}"]

        errors: List[str] = []

        # Title
        title_match = re.search(r'\\title(?:\[[^\]]*\])?\{([^}]*)\}', text, flags=re.DOTALL)
        if not title_match or not title_match.group(1).strip():
            errors.append("missing_or_empty_title")

        # Abstract content: supports \abstract{...} and \begin{abstract}...\end{abstract}
        abstract_cmd = re.search(r'\\abstract\{([^}]*)\}', text, flags=re.DOTALL)
        abstract_env = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', text, flags=re.DOTALL)
        abstract_text = ""
        if abstract_cmd:
            abstract_text = abstract_cmd.group(1)
        elif abstract_env:
            abstract_text = abstract_env.group(1)
        if not abstract_text.strip():
            errors.append("missing_or_empty_abstract")

        return errors
    
    # =========================================================================
    # Phase 5: VLM Review
    # =========================================================================
    
    async def _call_vlm_review(
        self,
        pdf_path: str,
        page_limit: int = 8,
        template_type: str = "ICML",
        sections_info: Optional[Dict[str, Any]] = None,
        memory: Optional[SessionMemory] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Call VLM Review Agent directly (no HTTP) to check the PDF.
        - **Description**:
            - Builds a VLMReviewRequest and calls self._vlm_reviewer.review().
            - Memory context is injected automatically inside review().

        - **Args**:
            - `pdf_path` (str): Path to compiled PDF
            - `page_limit` (int): Maximum allowed pages
            - `template_type` (str): Template type for context
            - `sections_info` (Dict, optional): Section word counts
            - `memory` (SessionMemory, optional): Shared session memory

        - **Returns**:
            - VLM review result dict or None on failure
        """
        if self._vlm_reviewer is None:
            print("[MetaDataAgent] VLM Review Agent not available, skipping")
            return None
        try:
            from ..vlm_review_agent.models import VLMReviewRequest

            request = VLMReviewRequest(
                pdf_path=pdf_path,
                page_limit=page_limit,
                template_type=template_type,
                check_overflow=True,
                check_underfill=True,
                check_layout=False,
                sections_info=sections_info or {},
            )

            result = await self._vlm_reviewer.review(request, memory=memory)
            return result.model_dump()

        except Exception as e:
            print(f"[MetaDataAgent] VLM Review error: {e}")
            return None
    
    # =========================================================================
    # Phase 0: Planning Methods
    # =========================================================================
    
    async def _create_paper_plan(
        self,
        metadata: PaperMetaData,
        target_pages: Optional[int],
        style_guide: Optional[str],
        research_context: Optional[Dict[str, Any]] = None,
        code_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[PaperPlan]:
        """Delegation stub — see orchestrator.py."""
        return await self._orchestrator._create_paper_plan(
            metadata=metadata,
            target_pages=target_pages,
            style_guide=style_guide,
            research_context=research_context,
            code_context=code_context,
        )
    
    # =========================================================================
    # Phase 3.5: Review Loop Methods
    # =========================================================================

    def _perform_baseline_gap_audit(
        self,
        generated_sections: Dict[str, str],
        enable_review: bool,
        enable_vlm_review: bool,
    ) -> Dict[str, Any]:
        """Delegation stub — see orchestrator.py."""
        return self._orchestrator._perform_baseline_gap_audit(
            generated_sections=generated_sections,
            enable_review=enable_review,
            enable_vlm_review=enable_vlm_review,
        )

    async def _llm_plan_revision_tasks(
        self,
        review_result: ReviewResult,
    ) -> List[Dict[str, Any]]:
        """Delegation stub — see orchestrator.py."""
        return await self._orchestrator._llm_plan_revision_tasks(
            review_result=review_result,
        )

    def _apply_revision_plan_to_feedbacks(
        self,
        review_result: ReviewResult,
        revision_plan: List[Dict[str, Any]],
    ) -> None:
        """Delegation stub — see orchestrator.py."""
        return self._orchestrator._apply_revision_plan_to_feedbacks(
            review_result=review_result,
            revision_plan=revision_plan,
        )

    @staticmethod
    def _normalize_target_paragraphs(raw_targets: Any) -> List[int]:
        """Delegation stub — see conflict_resolver.py."""
        return ConflictResolver._normalize_target_paragraphs(raw_targets)

    @staticmethod
    def _normalize_paragraph_instructions(
        raw_instructions: Any,
        target_paragraphs: Optional[List[int]] = None,
        fallback_instruction: str = "",
    ) -> Dict[int, str]:
        """Delegation stub — see conflict_resolver.py."""
        return ConflictResolver._normalize_paragraph_instructions(
            raw_instructions, target_paragraphs, fallback_instruction,
        )

    @staticmethod
    def _default_acceptance_criteria(issue_type: str) -> List[str]:
        """Delegation stub — see revision_executor.py."""
        return RevisionExecutor._default_acceptance_criteria(issue_type)

    async def _run_semantic_consistency_guard(
        self,
        section_type: str,
        before_text: str,
        after_text: str,
        revision_prompt: str,
    ) -> SemanticCheckRecord:
        """Delegation stub — see revision_executor.py."""
        return await self._executor._run_semantic_consistency_guard(
            section_type=section_type,
            before_text=before_text,
            after_text=after_text,
            revision_prompt=revision_prompt,
        )

    async def _translate_vlm_to_revision_plan(
        self,
        vlm_result: Dict[str, Any],
        generated_sections: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Delegation stub — see orchestrator.py."""
        return await self._orchestrator._translate_vlm_to_revision_plan(
            vlm_result=vlm_result,
            generated_sections=generated_sections,
        )

    async def _resolve_conflicts_with_llm(
        self,
        reviewer_feedbacks: List[SectionFeedback],
        external_feedbacks: List[SectionFeedback],
    ) -> Tuple[List[SectionFeedback], List[ConflictResolutionRecord]]:
        """Delegation stub — see conflict_resolver.py."""
        return await self._resolver._resolve_conflicts_with_llm(
            reviewer_feedbacks=reviewer_feedbacks,
            external_feedbacks=external_feedbacks,
        )
    
    def _build_vlm_feedback(
        self,
        vlm_result: Dict[str, Any],
        structural_actions: Optional[List[StructuralAction]] = None,
    ) -> Tuple[List[FeedbackResult], List[SectionFeedback]]:
        """Delegation stub — see conflict_resolver.py."""
        return self._resolver._build_vlm_feedback(
            vlm_result=vlm_result,
            structural_actions=structural_actions,
        )
    
    def _build_vlm_revision_prompt(
        self,
        section_type: str,
        action: str,
        delta_words: int,
        guidance: Optional[str] = None,
        structural_context: Optional[str] = None,
    ) -> str:
        """Delegation stub — see conflict_resolver.py."""
        return self._resolver._build_vlm_revision_prompt(
            section_type=section_type,
            action=action,
            delta_words=delta_words,
            guidance=guidance,
            structural_context=structural_context,
        )
    
    # =====================================================================
    # Smart page-limit control: structural overflow strategy
    # =====================================================================

    # Estimated page cost for non-text elements
    _ELEMENT_PAGE_COST = {
        "figure*": 0.4,
        "figure": 0.2,
        "table*": 0.3,
        "table": 0.15,
    }

    def _estimate_section_space(
        self,
        section_type: str,
        content: str,
    ) -> SpaceEstimate:
        """
        Estimate non-text space usage in a section.
        - **Description**:
            - Counts figure/table environments and their LaTeX labels
            - Returns a SpaceEstimate with per-type counts and total page cost

        - **Args**:
            - `section_type` (str): Section name (for logging)
            - `content` (str): LaTeX content of the section

        - **Returns**:
            - `SpaceEstimate` with element counts, ids, and estimated pages
        """
        est = SpaceEstimate()

        # Count wide/narrow figures
        est.wide_figures = len(re.findall(r"\\begin\{figure\*\}", content))
        est.narrow_figures = (
            len(re.findall(r"\\begin\{figure\}", content)) - est.wide_figures
        )
        if est.narrow_figures < 0:
            est.narrow_figures = 0

        # Count wide/narrow tables
        est.wide_tables = len(re.findall(r"\\begin\{table\*\}", content))
        est.narrow_tables = (
            len(re.findall(r"\\begin\{table\}", content)) - est.wide_tables
        )
        if est.narrow_tables < 0:
            est.narrow_tables = 0

        # Extract labels
        est.figure_ids = re.findall(r"\\label\{(fig:[^}]+)\}", content)
        est.table_ids = re.findall(r"\\label\{(tab:[^}]+)\}", content)

        # Estimate total pages consumed by non-text elements
        est.total_pages = (
            est.wide_figures * self._ELEMENT_PAGE_COST["figure*"]
            + est.narrow_figures * self._ELEMENT_PAGE_COST["figure"]
            + est.wide_tables * self._ELEMENT_PAGE_COST["table*"]
            + est.narrow_tables * self._ELEMENT_PAGE_COST["table"]
        )
        return est

    def _plan_overflow_strategy(
        self,
        overflow_pages: float,
        generated_sections: Dict[str, str],
        paper_plan: Optional[PaperPlan],
        figures: Optional[List[FigureSpec]],
    ) -> List[StructuralAction]:
        """
        Decide which structural actions to take based on overflow severity.
        - **Description**:
            - Level 1 (< 0.5 pages): word trimming only (no structural actions)
            - Level 2 (0.5-1.5 pages): downgrade figure* -> figure, resize images
            - Level 3 (> 1.5 pages): create appendix, move low-priority figures/tables
            - IMPORTANT: appendix section is excluded from scanning — it is the
              destination for moved elements, not a source.
            - Each figure/table ID is acted upon at most once (deduplication).

        - **Args**:
            - `overflow_pages` (float): How many pages over the limit
            - `generated_sections` (Dict[str, str]): Section contents
            - `paper_plan` (Optional[PaperPlan]): Paper plan (for targets)
            - `figures` (Optional[List[FigureSpec]]): Figure specs from metadata

        - **Returns**:
            - List of StructuralAction to execute before word-level revisions
        """
        from ..vlm_review_agent.models import SECTION_TRIM_PRIORITY

        actions: List[StructuralAction] = []
        remaining = overflow_pages

        if remaining <= 0:
            return actions

        # Step 1: gather space estimates per BODY section only (exclude appendix)
        space_map: Dict[str, SpaceEstimate] = {}
        for sec, content in generated_sections.items():
            if sec == "appendix":
                continue  # Never scan appendix — it is the move destination
            space_map[sec] = self._estimate_section_space(sec, content)

        print(
            f"[OverflowStrategy] overflow={overflow_pages:.1f} pages, "
            f"Level={'1' if overflow_pages < 0.5 else '2' if overflow_pages <= 1.5 else '3'}"
        )
        for sec, est in space_map.items():
            if est.total_pages > 0:
                print(
                    f"  {sec}: {est.wide_figures} figure*, {est.narrow_figures} figure, "
                    f"{est.wide_tables} table*, {est.narrow_tables} table  "
                    f"=> ~{est.total_pages:.2f} pages"
                )

        # Level 1: < 0.5 pages — word trim only (handled outside, return empty)
        if overflow_pages < 0.5:
            return actions

        # Track processed IDs to avoid duplicate actions on the same element
        processed_ids: set = set()

        # =================================================================
        # Level 2 actions: downgrade wide figures/tables + resize
        # =================================================================
        # Sort sections by trim priority (high -> low) for deterministic order
        sorted_sections = sorted(
            space_map.keys(),
            key=lambda s: SECTION_TRIM_PRIORITY.get(s, 5),
            reverse=True,
        )

        # 2a. Downgrade figure* -> figure (only sections that actually have wide figures)
        for sec in sorted_sections:
            est = space_map.get(sec)
            if not est or est.wide_figures == 0:
                continue
            for fid in est.figure_ids:
                if fid in processed_ids:
                    continue
                savings = 0.2
                actions.append(StructuralAction(
                    action_type="downgrade_wide",
                    target_id=fid,
                    section=sec,
                    estimated_savings=savings,
                ))
                processed_ids.add(fid)
                remaining -= savings
                print(f"  -> downgrade_wide figure {fid} in {sec} (save ~{savings:.1f}p)")
                if remaining <= 0:
                    break
            if remaining <= 0:
                break

        # 2b. Resize remaining figures (shrink width) — only if not already processed
        if remaining > 0:
            for sec in sorted_sections:
                est = space_map.get(sec)
                if not est or (est.wide_figures + est.narrow_figures) == 0:
                    continue
                for fid in est.figure_ids:
                    resize_key = f"resize:{fid}"
                    if resize_key in processed_ids:
                        continue
                    savings = 0.05
                    actions.append(StructuralAction(
                        action_type="resize_figure",
                        target_id=fid,
                        section=sec,
                        params={"width": "0.8\\linewidth"},
                        estimated_savings=savings,
                    ))
                    processed_ids.add(resize_key)
                    remaining -= savings
                    if remaining <= 0:
                        break
                if remaining <= 0:
                    break

        # =================================================================
        # Level 3 actions: create appendix + move low-priority figures/tables
        # =================================================================
        if overflow_pages > 1.5 and remaining > 0:
            if "appendix" not in generated_sections:
                actions.append(StructuralAction(
                    action_type="create_appendix",
                    estimated_savings=0,
                ))

            for sec in sorted_sections:
                est = space_map.get(sec)
                if not est:
                    continue
                # Move figures
                for fid in est.figure_ids:
                    move_key = f"move:{fid}"
                    if move_key in processed_ids:
                        continue
                    savings = (
                        self._ELEMENT_PAGE_COST["figure*"]
                        if est.wide_figures > 0
                        else self._ELEMENT_PAGE_COST["figure"]
                    )
                    actions.append(StructuralAction(
                        action_type="move_figure",
                        target_id=fid,
                        section=sec,
                        estimated_savings=savings,
                    ))
                    processed_ids.add(move_key)
                    remaining -= savings
                    print(f"  -> move_figure {fid} from {sec} to appendix (save ~{savings:.1f}p)")
                    if remaining <= 0:
                        break
                # Move tables
                if remaining > 0:
                    for tid in est.table_ids:
                        move_key = f"move:{tid}"
                        if move_key in processed_ids:
                            continue
                        savings = (
                            self._ELEMENT_PAGE_COST["table*"]
                            if est.wide_tables > 0
                            else self._ELEMENT_PAGE_COST["table"]
                        )
                        actions.append(StructuralAction(
                            action_type="move_table",
                            target_id=tid,
                            section=sec,
                            estimated_savings=savings,
                        ))
                        processed_ids.add(move_key)
                        remaining -= savings
                        print(f"  -> move_table {tid} from {sec} to appendix (save ~{savings:.1f}p)")
                        if remaining <= 0:
                            break
                if remaining <= 0:
                    break

        total_savings = sum(a.estimated_savings for a in actions)
        print(
            f"[OverflowStrategy] Planned {len(actions)} structural actions, "
            f"estimated savings ~{total_savings:.1f} pages"
        )
        return actions

    # =====================================================================
    # Structural operation execution methods
    # =====================================================================

    def _resize_figures_in_section(
        self,
        section_content: str,
        actions: List[StructuralAction],
    ) -> str:
        """
        Apply resize and downgrade-wide actions to a section's LaTeX content.
        - **Description**:
            - downgrade_wide: figure* -> figure, end{figure*} -> end{figure}
            - resize_figure: adjusts includegraphics width parameter
            - Pure regex operations, no LLM call needed

        - **Args**:
            - `section_content` (str): Current LaTeX content of the section
            - `actions` (List[StructuralAction]): Actions targeting this section

        - **Returns**:
            - Modified section content with figure adjustments applied
        """
        modified = section_content

        for act in actions:
            if act.action_type == "downgrade_wide":
                # Replace figure* -> figure (all occurrences in this section)
                modified = modified.replace("\\begin{figure*}", "\\begin{figure}")
                modified = modified.replace("\\end{figure*}", "\\end{figure}")
                print(f"  [Structural] Downgraded figure* -> figure for {act.target_id}")

            elif act.action_type == "resize_figure":
                target_width = act.params.get("width", "0.8\\linewidth")
                # Use lambda replacements to avoid regex interpreting
                # backslashes in LaTeX commands like \linewidth
                modified = re.sub(
                    r"\\includegraphics\[([^\]]*?)width\s*=\s*\\textwidth",
                    lambda m: f"\\includegraphics[{m.group(1)}width={target_width}",
                    modified,
                )
                modified = re.sub(
                    r"\\includegraphics\[([^\]]*?)width\s*=\s*\\linewidth",
                    lambda m: f"\\includegraphics[{m.group(1)}width={target_width}",
                    modified,
                )
                modified = re.sub(
                    r"\\includegraphics\[([^\]]*?)width\s*=\s*\\columnwidth",
                    lambda m: f"\\includegraphics[{m.group(1)}width={target_width}",
                    modified,
                )
                print(f"  [Structural] Resized figures to {target_width} for {act.target_id}")

        return modified

    def _move_figures_to_appendix(
        self,
        generated_sections: Dict[str, str],
        actions: List[StructuralAction],
    ) -> None:
        """
        Move figure/table environments from source sections to the appendix.
        - **Description**:
            - Extracts the full \\begin{figure}...\\end{figure} block containing the target label
            - Replaces it in the source section with a cross-reference note
            - Appends the extracted block to generated_sections["appendix"]
            - Operates in-place on generated_sections dict

        - **Args**:
            - `generated_sections` (Dict[str, str]): All section contents (mutated)
            - `actions` (List[StructuralAction]): move_figure / move_table actions
        """
        # Ensure appendix section exists
        if "appendix" not in generated_sections:
            generated_sections["appendix"] = ""

        appendix_parts: List[str] = []
        if generated_sections["appendix"]:
            appendix_parts.append(generated_sections["appendix"])

        # Track labels already in appendix to prevent duplicates
        existing_labels = set(re.findall(
            r'\\label\{([^}]+)\}', generated_sections["appendix"],
        ))

        for act in actions:
            if act.action_type not in ("move_figure", "move_table"):
                continue

            sec = act.section
            content = generated_sections.get(sec, "")
            if not content:
                continue

            target_label = act.target_id  # e.g. "fig:arch"

            if target_label in existing_labels:
                print(f"  [Structural] Skipping {target_label} — already in appendix")
                continue

            # Determine environment type
            if act.action_type == "move_figure":
                env_names = ["figure\\*", "figure"]
                ref_text = f"Figure~\\\\ref{{{target_label}}}"
            else:
                env_names = ["table\\*", "table"]
                ref_text = f"Table~\\\\ref{{{target_label}}}"

            extracted = False
            for env in env_names:
                # Match the environment that contains this specific label
                pattern = re.compile(
                    r"(\\begin\{" + env + r"\}.*?\\label\{" + re.escape(target_label) + r"\}.*?\\end\{" + env + r"\})",
                    re.DOTALL,
                )
                m = pattern.search(content)
                if m:
                    block = m.group(1)
                    # Replace the block with a cross-reference
                    replacement = f"% [{target_label} moved to Appendix]\n(see {ref_text} in the Appendix)"
                    content = content[:m.start()] + replacement + content[m.end():]
                    generated_sections[sec] = content
                    appendix_parts.append(block)
                    existing_labels.add(target_label)
                    extracted = True
                    print(f"  [Structural] Moved {target_label} from {sec} to appendix")
                    break

            if not extracted:
                # Try reverse order: label might appear before the begin
                for env in env_names:
                    pattern = re.compile(
                        r"(\\begin\{" + env + r"\}.*?\\end\{" + env + r"\})",
                        re.DOTALL,
                    )
                    for m in pattern.finditer(content):
                        if target_label in m.group(1):
                            block = m.group(1)
                            replacement = f"% [{target_label} moved to Appendix]\n(see {ref_text} in the Appendix)"
                            content = content[:m.start()] + replacement + content[m.end():]
                            generated_sections[sec] = content
                            appendix_parts.append(block)
                            existing_labels.add(target_label)
                            extracted = True
                            print(f"  [Structural] Moved {target_label} from {sec} to appendix (alt)")
                            break
                    if extracted:
                        break

            if not extracted:
                print(f"  [Structural] WARNING: Could not find {target_label} in {sec}")

        generated_sections["appendix"] = "\n\n".join(appendix_parts)

    def _create_appendix_section(
        self,
        generated_sections: Dict[str, str],
        section_order: List[str],
    ) -> None:
        """
        Create an appendix section if it doesn't already exist.
        - **Description**:
            - Adds "appendix" key to generated_sections
            - Inserts "appendix" after "conclusion" in section_order

        - **Args**:
            - `generated_sections` (Dict[str, str]): Section dict (mutated)
            - `section_order` (List[str]): Section ordering list (mutated)
        """
        if "appendix" not in generated_sections:
            generated_sections["appendix"] = ""
            print("[Structural] Created appendix section")

        if "appendix" not in section_order:
            # Insert after conclusion if it exists, else at the end
            if "conclusion" in section_order:
                idx = section_order.index("conclusion") + 1
                section_order.insert(idx, "appendix")
            else:
                section_order.append("appendix")
            print(f"[Structural] Added appendix to section_order at position {section_order.index('appendix')}")

    def _execute_structural_actions(
        self,
        actions: List[StructuralAction],
        generated_sections: Dict[str, str],
        section_order: List[str],
    ) -> None:
        """
        Execute all structural actions in order: create appendix, then move, then resize.
        - **Description**:
            - Groups actions by type and applies them in the correct order
            - Mutates generated_sections and section_order in place

        - **Args**:
            - `actions` (List[StructuralAction]): Planned structural actions
            - `generated_sections` (Dict[str, str]): Section contents (mutated)
            - `section_order` (List[str]): Section ordering (mutated)
        """
        if not actions:
            return

        print(f"[Structural] Executing {len(actions)} structural actions...")

        # Phase 1: create appendix if needed
        create_actions = [a for a in actions if a.action_type == "create_appendix"]
        if create_actions:
            self._create_appendix_section(generated_sections, section_order)

        # Phase 2: move figures/tables to appendix
        move_actions = [a for a in actions if a.action_type in ("move_figure", "move_table")]
        if move_actions:
            # Ensure appendix exists before moving anything
            if "appendix" not in generated_sections:
                self._create_appendix_section(generated_sections, section_order)
            self._move_figures_to_appendix(generated_sections, move_actions)

        # Phase 3: resize/downgrade in remaining sections
        resize_actions = [a for a in actions if a.action_type in ("resize_figure", "downgrade_wide")]
        if resize_actions:
            # Group by section
            by_section: Dict[str, List[StructuralAction]] = {}
            for a in resize_actions:
                by_section.setdefault(a.section, []).append(a)
            for sec, sec_actions in by_section.items():
                if sec in generated_sections:
                    generated_sections[sec] = self._resize_figures_in_section(
                        generated_sections[sec], sec_actions
                    )

        print("[Structural] All structural actions executed")

    # Moved to conflict_resolver.py; kept here for backward compat
    LATEX_ERROR_FIXES = _CR_LATEX_ERROR_FIXES
    
    def _build_typesetter_feedback(
        self,
        compile_errors: List[str],
        generated_sections: Dict[str, str],
        section_errors: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[List[FeedbackResult], List[SectionFeedback]]:
        """Delegation stub — see conflict_resolver.py."""
        return self._resolver._build_typesetter_feedback(
            compile_errors=compile_errors,
            generated_sections=generated_sections,
            section_errors=section_errors,
        )
    
    def _merge_section_feedbacks(
        self,
        base_feedbacks: List[SectionFeedback],
        vlm_feedbacks: List[SectionFeedback],
        prefer_vlm: bool,
    ) -> List[SectionFeedback]:
        """Delegation stub — see conflict_resolver.py."""
        return self._resolver._merge_section_feedbacks(
            base_feedbacks=base_feedbacks,
            vlm_feedbacks=vlm_feedbacks,
            prefer_vlm=prefer_vlm,
        )
    
    def _resolve_section_feedbacks(
        self,
        section_feedbacks: List[SectionFeedback],
        revised_sections: set,
        review_result: ReviewResult,
    ) -> None:
        """Delegation stub — see conflict_resolver.py."""
        return self._resolver._resolve_section_feedbacks(
            section_feedbacks=section_feedbacks,
            revised_sections=revised_sections,
            review_result=review_result,
        )
    
    async def _apply_revisions(
        self,
        review_result: ReviewResult,
        generated_sections: Dict[str, str],
        sections_results: List[SectionResult],
        valid_citation_keys: set,
        metadata: PaperMetaData,
        memory: Optional[SessionMemory] = None,
        semantic_checks: Optional[List[SemanticCheckRecord]] = None,
        decision_trace: Optional[List[Dict[str, Any]]] = None,
        writer_response_section: Optional[List[Dict[str, Any]]] = None,
        writer_response_paragraph: Optional[List[Dict[str, Any]]] = None,
        reviewer_verification: Optional[List[Dict[str, Any]]] = None,
    ) -> set:
        """Delegation stub — see revision_executor.py."""
        return await self._executor._apply_revisions(
            review_result=review_result,
            generated_sections=generated_sections,
            sections_results=sections_results,
            valid_citation_keys=valid_citation_keys,
            metadata=metadata,
            memory=memory,
            semantic_checks=semantic_checks,
            decision_trace=decision_trace,
            writer_response_section=writer_response_section,
            writer_response_paragraph=writer_response_paragraph,
            reviewer_verification=reviewer_verification,
        )

    @staticmethod
    def _split_section_paragraphs(content: str) -> List[str]:
        """Delegation stub — see revision_executor.py."""
        return RevisionExecutor._split_section_paragraphs(content)

    @staticmethod
    def _join_section_paragraphs(paragraphs: List[str]) -> str:
        """Delegation stub — see revision_executor.py."""
        return RevisionExecutor._join_section_paragraphs(paragraphs)

    async def _revise_section_paragraphs(
        self,
        section_type: str,
        current_content: str,
        target_paragraphs: List[int],
        paragraph_instructions: Dict[int, str],
        fallback_prompt: str,
        metadata: PaperMetaData,
        memory: Optional[SessionMemory] = None,
        task_contract_base: Optional[Dict[str, Any]] = None,
        valid_citation_keys: Optional[set] = None,
    ) -> Optional[str]:
        """Delegation stub — see revision_executor.py."""
        return await self._executor._revise_section_paragraphs(
            section_type=section_type,
            current_content=current_content,
            target_paragraphs=target_paragraphs,
            paragraph_instructions=paragraph_instructions,
            fallback_prompt=fallback_prompt,
            metadata=metadata,
            memory=memory,
            task_contract_base=task_contract_base,
            valid_citation_keys=valid_citation_keys,
        )

    async def _revise_section_sentences(
        self,
        section_type: str,
        current_content: str,
        sentence_feedbacks: List[Dict[str, Any]],
        metadata: PaperMetaData,
        memory: Optional[SessionMemory] = None,
        valid_citation_keys: Optional[set] = None,
    ) -> Optional[str]:
        """Delegation stub — see revision_executor.py."""
        return await self._executor._revise_section_sentences(
            section_type=section_type,
            current_content=current_content,
            sentence_feedbacks=sentence_feedbacks,
            metadata=metadata,
            memory=memory,
            valid_citation_keys=valid_citation_keys,
        )

    def _get_sections_fingerprint(self, sections: Dict[str, str]) -> str:
        """Delegation stub — see revision_executor.py."""
        return self._executor._get_sections_fingerprint(sections=sections)
    
    async def _run_review_orchestration(
        self,
        generated_sections: Dict[str, str],
        sections_results: List[SectionResult],
        metadata: PaperMetaData,
        parsed_refs: List[Dict[str, Any]],
        paper_plan: Optional[PaperPlan],
        template_path: Optional[str],
        figures_source_dir: Optional[str],
        converted_tables: Dict[str, str],
        max_review_iterations: int,
        enable_review: bool,
        compile_pdf: bool,
        enable_vlm_review: bool,
        target_pages: Optional[int],
        paper_dir: Optional[Path],
        memory: Optional[SessionMemory] = None,
        evidence_dag: Optional[EvidenceDAG] = None,
    ) -> Tuple[Dict[str, str], List[SectionResult], int, Optional[int], Optional[str], List[str]]:
        """Delegation stub — see orchestrator.py."""
        return await self._orchestrator._run_review_orchestration(
            generated_sections=generated_sections,
            sections_results=sections_results,
            metadata=metadata,
            parsed_refs=parsed_refs,
            paper_plan=paper_plan,
            template_path=template_path,
            figures_source_dir=figures_source_dir,
            converted_tables=converted_tables,
            max_review_iterations=max_review_iterations,
            enable_review=enable_review,
            compile_pdf=compile_pdf,
            enable_vlm_review=enable_vlm_review,
            target_pages=target_pages,
            paper_dir=paper_dir,
            memory=memory,
            evidence_dag=evidence_dag,
        )
    
    
    async def _call_reviewer(
        self,
        sections: Dict[str, str],
        word_counts: Dict[str, int],
        target_pages: Optional[int],
        style_guide: Optional[str],
        template_path: Optional[str],
        iteration: int,
        section_targets: Optional[Dict[str, int]] = None,
        section_structure_signals: Optional[Dict[str, Any]] = None,
        memory: Optional[SessionMemory] = None,
        evidence_dag: Optional[EvidenceDAG] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        """Delegation stub — see revision_executor.py."""
        return await self._executor._call_reviewer(
            sections=sections,
            word_counts=word_counts,
            target_pages=target_pages,
            style_guide=style_guide,
            template_path=template_path,
            iteration=iteration,
            section_targets=section_targets,
            section_structure_signals=section_structure_signals,
            memory=memory,
            evidence_dag=evidence_dag,
        )
    
    async def _revise_section(
        self,
        section_type: str,
        current_content: str,
        revision_prompt: str,
        metadata: PaperMetaData,
        memory: Optional[SessionMemory] = None,
        task_contract: Optional[Dict[str, Any]] = None,
        valid_citation_keys: Optional[set] = None,
    ) -> Optional[str]:
        """Delegation stub — see revision_executor.py."""
        return await self._executor._revise_section(
            section_type=section_type,
            current_content=current_content,
            revision_prompt=revision_prompt,
            metadata=metadata,
            memory=memory,
            task_contract=task_contract,
            valid_citation_keys=valid_citation_keys,
        )

    async def _revise_paragraph(
        self,
        section_type: str,
        paragraph_index: int,
        paragraph_text: str,
        instruction: str,
        memory: Optional[SessionMemory] = None,
        task_contract: Optional[Dict[str, Any]] = None,
        valid_citation_keys: Optional[set] = None,
    ) -> Optional[str]:
        """Delegation stub — see revision_executor.py."""
        return await self._executor._revise_paragraph(
            section_type=section_type,
            paragraph_index=paragraph_index,
            paragraph_text=paragraph_text,
            instruction=instruction,
            memory=memory,
            task_contract=task_contract,
            valid_citation_keys=valid_citation_keys,
        )
