"""
Planner Agent
- **Description**:
    - Creates detailed paper plans before generation
    - Paragraph-level planning with VLM-informed figure/table placement
    - Outputs PaperPlan to guide Writers and Reviewers
"""
import json
import logging
import re
import random
from typing import List, Dict, Any, Optional

from fastapi import APIRouter
from openai import AsyncOpenAI

from ..base import BaseAgent
from ...config.schema import ModelConfig
from .models import (
    PaperPlan,
    SectionPlan,
    PlanRequest,
    PlanResult,
    ParagraphPlan,
    FigurePlacement,
    TablePlacement,
    PaperType,
    NarrativeStyle,
    SECTION_RATIOS_BY_TYPE,
    DEFAULT_EMPIRICAL_SECTIONS,
    VENUE_WORD_LIMITS,
    ELEMENT_PAGE_COST,
    WORDS_PER_SENTENCE,
    calculate_total_words,
)


logger = logging.getLogger("uvicorn.error")


# =========================================================================
# LLM Prompts
# =========================================================================

PLANNING_SYSTEM_PROMPT = """You are an expert academic paper planner. Your job is to analyze paper metadata and create a detailed paragraph-level writing plan.

Given the paper's idea/hypothesis, method, data, experiments, references, figures, and tables, you must:
1. Determine the paper type (empirical, theoretical, survey, position, system, benchmark)
2. Identify the key contributions (usually 2-4)
3. Design the section structure (you choose which sections to include)
   - "abstract" and "conclusion" are REQUIRED
   - For the body, use standard sections or merge/rename as needed
4. For EACH section, plan INDIVIDUAL PARAGRAPHS:
   - Each paragraph has a key_point, supporting_points, approximate sentence count, and a role
   - Roles: "motivation", "problem_statement", "definition", "evidence", "comparison", "transition", "summary"
5. Assign figures and tables to sections with placement hints
6. Suggest which references to cite in each paragraph

Output a JSON object with this structure:
{
    "paper_type": "empirical",
    "contributions": ["Contribution 1", "Contribution 2"],
    "narrative_style": "technical",
    "terminology": {"key_term": "definition"},
    "structure_rationale": "Why this structure works",
    "abstract_focus": "What the abstract should emphasize",
    "sections": [
        {
            "section_type": "introduction",
            "section_title": "Introduction",
            "paragraphs": [
                {
                    "key_point": "Research context and motivation",
                    "supporting_points": ["Background info", "Why this matters"],
                    "approx_sentences": 5,
                    "role": "motivation",
                    "references_to_cite": ["ref_key1"],
                    "figures_to_reference": [],
                    "tables_to_reference": []
                },
                {
                    "key_point": "Problem statement and gap",
                    "supporting_points": ["Current limitations"],
                    "approx_sentences": 4,
                    "role": "problem_statement",
                    "references_to_cite": ["ref_key2"]
                }
            ],
            "figures": [
                {
                    "figure_id": "fig:architecture",
                    "position_hint": "early",
                    "caption_guidance": "Show overall framework"
                }
            ],
            "tables": [],
            "content_sources": ["idea_hypothesis", "method"],
            "citation_budget": {
                "min_refs": 6,
                "target_refs": 10,
                "max_refs": 14,
                "rationale": "Why this section needs this citation depth"
            },
            "topic_clusters": ["Theme A", "Theme B"],
            "transition_intents": ["Move from context to gap", "Bridge method to evidence"],
            "sectioning_recommended": true,
            "writing_guidance": "Specific guidance"
        }
    ]
}

IMPORTANT:
- Each paragraph should have 3-8 sentences (approx_sentences)
- Sections with figures/tables need FEWER text paragraphs (visuals take space)
- "abstract" typically needs 1-2 paragraphs; "conclusion" needs 2-3 paragraphs
- Assign each figure/table to exactly ONE section for definition
- Decide citation budgets per section based on section goals, evidence needs,
  and venue rigor. Do not use a global fixed number.
- Add soft structure signals for each body section:
  - topic_clusters: 1-4 thematic blocks
  - transition_intents: 1-3 intended transitions across blocks
  - sectioning_recommended: whether explicit sub-structure may improve readability
- Be specific and actionable"""


PLANNING_USER_PROMPT_TEMPLATE = """Create a detailed paragraph-level paper plan for:

**Title**: {title}

**Idea/Hypothesis**:
{idea_hypothesis}

**Method**:
{method}

**Data**:
{data}

**Experiments**:
{experiments}

**Research Context (use this to decide emphasis, trade-offs, and style)**:
{research_context_summary}

**Available References** (BibTeX keys):
{reference_keys}

**Available Figures**:
{figure_info}

**Available Tables**:
{table_info}

**Space Budget**:
- Target: {target_pages} pages for {style_guide}
- Estimated total paragraphs: ~{total_paragraphs}
- Effective word budget: ~{total_words} words

Plan each section with specific paragraphs. Each paragraph should have:
- A clear key_point (the main argument)
- Supporting points (evidence, examples)
- Approximate sentence count (3-8)
- A role (motivation, problem_statement, definition, evidence, comparison, transition, summary)
- Which references to cite
- Also provide soft structure signals (topic_clusters, transition_intents,
  sectioning_recommended) for Writer and Reviewer coordination.

Output valid JSON only."""


# =========================================================================
# Planner Agent
# =========================================================================

class PlannerAgent(BaseAgent):
    """
    Planner Agent for paper planning.
    - **Description**:
        - Creates comprehensive paragraph-level plans
        - Optionally uses VLM for intelligent figure/table analysis
        - Directly encapsulates all planning logic (no Strategy pattern)
    """

    def __init__(
        self,
        config: ModelConfig,
        vlm_service: Optional[Any] = None,
    ):
        """
        Initialize the Planner Agent.

        - **Args**:
            - `config` (ModelConfig): LLM configuration
            - `vlm_service` (VLMService, optional): Shared VLM service for figure analysis
        """
        self.config = config
        self.model_name = config.model_name
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.vlm_service = vlm_service
        self._last_plan: Optional[PaperPlan] = None
        self._router = self._create_router()

        logger.info("PlannerAgent initialized (vlm=%s)", vlm_service is not None)

    @property
    def name(self) -> str:
        return "planner"

    @property
    def description(self) -> str:
        return "Creates detailed paragraph-level paper plans"

    @property
    def router(self) -> APIRouter:
        return self._router

    @property
    def endpoints_info(self) -> List[Dict[str, Any]]:
        return [
            {
                "path": "/agent/planner/plan",
                "method": "POST",
                "description": "Create a paper plan from metadata",
            },
            {
                "path": "/agent/planner/health",
                "method": "GET",
                "description": "Health check",
            },
        ]

    def _create_router(self) -> APIRouter:
        from .router import create_planner_router
        return create_planner_router(self)

    # =====================================================================
    # AskTool consultation interface
    # =====================================================================

    async def answer(self, question: str) -> str:
        """
        Two-stage answer about the paper plan.
        - **Description**:
            - Stage 1: Rule-based keyword filtering over the cached
              PaperPlan to gather compact candidate snippets.
            - Stage 2: LLM refinement — passes the candidates + question
              to ``self.client`` for a concise, semantically precise answer.
            - If the LLM call fails, falls back to Stage 1 output.

        - **Args**:
            - `question` (str): Natural-language question about the plan

        - **Returns**:
            - `result` (str): Precise answer about the plan
        """
        if self._last_plan is None:
            return "No plan available yet."

        candidates = self._gather_plan_candidates(question)
        if not candidates:
            return "No matching plan information found."

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a paper-planning assistant. Based on the "
                            "plan context below, answer the question concisely "
                            "and precisely. Keep your response under 200 words."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Plan context:\n{candidates}\n\n"
                            f"Question: {question}"
                        ),
                    },
                ],
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content or candidates
        except Exception as e:
            logger.warning("planner.answer LLM refine failed: %s", e)
            return candidates

    def _gather_plan_candidates(self, question: str) -> str:
        """
        Stage 1: Gather compact plan snippets via keyword matching.
        - **Description**:
            - Produces token-budgeted output: section_type + paragraph
              count + guidance first sentence + key point summaries.
            - Roughly ~50 tokens per matching section.

        - **Args**:
            - `question` (str): The question driving keyword extraction

        - **Returns**:
            - `text` (str): Compact matching plan context
        """
        keywords = [w.lower() for w in question.split() if len(w) > 2]
        hits: List[str] = []

        for sp in self._last_plan.sections:
            stype = sp.section_type
            guidance = sp.writing_guidance or ""
            para_texts = " ".join(
                getattr(p, "key_point", "") for p in (sp.paragraphs or [])
            )
            fig_texts = " ".join(
                getattr(fp, "figure_id", "") + " " + getattr(fp, "purpose", "")
                for fp in (sp.figure_placements or [])
            )
            tbl_texts = " ".join(
                getattr(tp, "table_id", "") + " " + getattr(tp, "purpose", "")
                for tp in (sp.table_placements or [])
            )
            full = f"{stype} {guidance} {para_texts} {fig_texts} {tbl_texts}".lower()

            if not keywords or any(kw in full for kw in keywords):
                guidance_snippet = guidance.split(".")[0] if guidance else ""
                n_paras = len(sp.paragraphs or [])
                kp_list = ", ".join(
                    getattr(p, "key_point", "")[:60]
                    for p in (sp.paragraphs or [])[:4]
                )
                line = f"- {stype}: {n_paras} paragraphs"
                est = sp.get_estimated_words() if hasattr(sp, "get_estimated_words") else 0
                if est:
                    line += f", ~{est} words"
                if guidance_snippet:
                    line += f", guidance: \"{guidance_snippet}\""
                if kp_list:
                    line += f", key points: [{kp_list}]"
                for fp in (sp.figure_placements or []):
                    fid = getattr(fp, "figure_id", "")
                    purpose = getattr(fp, "purpose", "")[:60]
                    line += f", fig {fid}: {purpose}"
                for tp in (sp.table_placements or []):
                    tid = getattr(tp, "table_id", "")
                    purpose = getattr(tp, "purpose", "")[:60]
                    line += f", tbl {tid}: {purpose}"
                hits.append(line)

        return "\n".join(hits) if hits else ""

    # =====================================================================
    # Core planning
    # =====================================================================

    def _format_research_context_for_planning(
        self,
        research_context: Optional[Dict[str, Any]],
    ) -> str:
        """
        Format compact research context for planning input.
        """
        if not research_context:
            return "Not available."

        lines: List[str] = []
        area = str(research_context.get("research_area", "")).strip()
        summary = str(research_context.get("summary", "")).strip()
        if area:
            lines.append(f"- Research area: {area}")
        if summary:
            lines.append(f"- Landscape summary: {summary}")

        trends = research_context.get("research_trends", []) or []
        if trends:
            lines.append("- Key trends:")
            for t in trends[:3]:
                lines.append(f"  - {t}")

        gaps = research_context.get("gaps", []) or []
        if gaps:
            lines.append("- Key gaps/opportunities:")
            for g in gaps[:3]:
                lines.append(f"  - {g}")

        ranking = research_context.get("contribution_ranking", {}) or {}
        if ranking:
            lines.append("- Contribution ranking hints:")
            for band in ("P0", "P1", "P2"):
                items = ranking.get(band, []) or []
                if not items:
                    continue
                top_text = ", ".join(
                    [str(x.get("contribution", "")).strip() for x in items[:3] if isinstance(x, dict)]
                )
                if top_text:
                    lines.append(f"  - {band}: {top_text}")

        return "\n".join(lines) if lines else "Not available."

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """
        Strip common markdown code fences from model outputs.
        """
        raw = (text or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        return raw.strip()

    @staticmethod
    def _extract_balanced_json_block(text: str, start_char: str) -> Optional[str]:
        """
        Extract first balanced JSON object/array block from text.
        """
        end_char = "}" if start_char == "{" else "]"
        start_idx = text.find(start_char)
        if start_idx < 0:
            return None

        depth = 0
        in_string = False
        escaped = False
        for i in range(start_idx, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == "\"":
                    in_string = False
                continue

            if ch == "\"":
                in_string = True
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx:i + 1]
        return None

    @classmethod
    def _safe_load_json(
        cls,
        raw: str,
        expected: Optional[type] = None,
    ) -> Optional[Any]:
        """
        Parse JSON robustly from model outputs with optional type check.
        """
        cleaned = cls._strip_code_fence(raw)
        candidates: List[str] = [cleaned]
        obj_block = cls._extract_balanced_json_block(cleaned, "{")
        arr_block = cls._extract_balanced_json_block(cleaned, "[")
        if obj_block:
            candidates.append(obj_block)
        if arr_block:
            candidates.append(arr_block)

        for cand in candidates:
            if not cand:
                continue
            try:
                parsed = json.loads(cand)
                if expected is not None and not isinstance(parsed, expected):
                    continue
                return parsed
            except Exception:
                continue
        return None

    async def create_plan(self, request: PlanRequest) -> PaperPlan:
        """
        Create a paper plan from metadata.

        - **Args**:
            - `request` (PlanRequest): Planning request with metadata

        - **Returns**:
            - `PaperPlan`: Complete paragraph-level paper plan
        """
        n_figures = len(request.figures) if request.figures else 0
        n_tables = len(request.tables) if request.tables else 0
        n_wide_figures = sum(
            1 for f in (request.figures or []) if self._should_be_wide_figure(f)
        )
        n_wide_tables = sum(
            1 for t in (request.tables or []) if self._should_be_wide_table(t)
        )

        total_words = calculate_total_words(
            request.target_pages,
            request.style_guide,
            n_figures=n_figures,
            n_tables=n_tables,
            n_wide_figures=n_wide_figures,
            n_wide_tables=n_wide_tables,
        )
        target_pages = request.target_pages or 8
        style_guide = request.style_guide or "DEFAULT"
        total_paragraphs = max(1, total_words // 100)

        # VLM analysis for figures and tables (if service available)
        figure_analyses: Dict[str, Any] = {}
        table_analyses: Dict[str, Any] = {}
        if self.vlm_service:
            figure_analyses = await self._analyze_figures(request.figures or [])
            table_analyses = await self._analyze_tables(request.tables or [])

        reference_keys = self._extract_reference_keys(request.references)

        figure_info = self._format_figure_info(request.figures or [], figure_analyses)
        table_info = self._format_table_info(request.tables or [], table_analyses)

        user_prompt = PLANNING_USER_PROMPT_TEMPLATE.format(
            title=request.title,
            idea_hypothesis=request.idea_hypothesis[:2000],
            method=request.method[:2000],
            data=request.data[:1500],
            experiments=request.experiments[:2000],
            research_context_summary=self._format_research_context_for_planning(
                request.research_context
            ),
            reference_keys=", ".join(reference_keys) if reference_keys else "None provided",
            figure_info=figure_info,
            table_info=table_info,
            target_pages=target_pages,
            total_words=total_words,
            total_paragraphs=total_paragraphs,
            style_guide=style_guide,
        )

        logger.info(
            "planner.create_plan title=%s words=%d paragraphs=%d vlm=%s",
            request.title[:30], total_words, total_paragraphs,
            bool(figure_analyses or table_analyses),
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            plan_text = response.choices[0].message.content.strip()
            plan_data = self._parse_plan_json(plan_text)

            paper_plan = self._build_paper_plan(
                plan_data=plan_data,
                request=request,
                total_words=total_words,
                figure_analyses=figure_analyses,
                table_analyses=table_analyses,
            )

            logger.info(
                "planner.plan_created sections=%d sentences=%d",
                len(paper_plan.sections),
                paper_plan.get_total_sentences(),
            )
            self._last_plan = paper_plan
            return paper_plan

        except Exception as e:
            logger.error("planner.llm_error: %s", str(e))
            fallback = self._create_default_plan(request, total_words)
            self._last_plan = fallback
            return fallback

    async def discover_seed_references(
        self,
        title: str,
        idea_hypothesis: str,
        method: str,
        data: str,
        experiments: str,
        existing_ref_keys: List[str],
        paper_search_config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Discover global references before section-level planning.
        """
        import asyncio
        from ..shared.tools.paper_search import PaperSearchTool

        cfg = paper_search_config or {}
        tool = PaperSearchTool(
            serpapi_api_key=cfg.get("serpapi_api_key"),
            semantic_scholar_api_key=cfg.get("semantic_scholar_api_key"),
            timeout=cfg.get("timeout", 10),
            semantic_scholar_min_results_before_fallback=cfg.get(
                "semantic_scholar_min_results_before_fallback", 3
            ),
            enable_query_cache=cfg.get("enable_query_cache", True),
            cache_ttl_hours=cfg.get("cache_ttl_hours", 24),
        )

        max_queries = max(3, min(6, int(cfg.get("planner_max_queries_per_section", 5))))
        per_round = max(3, int(cfg.get("search_results_per_round", 5)))
        delay_sec = max(0.5, float(cfg.get("planner_inter_round_delay_sec", 1.5)))

        seeds = [
            title,
            f"{title} {idea_hypothesis[:160]}",
            f"{title} {method[:180]}",
            f"{title} {data[:160]}",
            f"{title} {experiments[:180]}",
            f"{method[:120]} {experiments[:120]}",
        ]
        queries: List[str] = []
        seen_q = set()
        for q in seeds:
            qq = " ".join(str(q).split()).strip()
            if len(qq) < 8 or qq in seen_q:
                continue
            seen_q.add(qq)
            queries.append(qq)
            if len(queries) >= max_queries:
                break

        discovered: List[Dict[str, Any]] = []
        seen_keys = set(existing_ref_keys)
        for i, query in enumerate(queries):
            if i > 0:
                await asyncio.sleep(delay_sec)
            try:
                result = await tool.execute(query=query, max_results=per_round)
                if not result.success:
                    continue
                papers = result.data.get("papers", []) if result.data else []
                for paper in papers:
                    bkey = paper.get("bibtex_key", "")
                    bibtex = paper.get("bibtex", "")
                    if bkey and bibtex and bkey not in seen_keys:
                        seen_keys.add(bkey)
                        discovered.append(
                            {
                                "ref_id": bkey,
                                "bibtex": bibtex,
                                "title": paper.get("title", ""),
                                "year": paper.get("year"),
                                "abstract": paper.get("abstract", ""),
                                "venue": paper.get("venue", ""),
                                "citation_count": paper.get("citation_count"),
                                "source": paper.get("source", ""),
                            }
                        )
            except Exception as e:
                logger.warning("planner.seed_search_error query='%s': %s", query[:80], e)

        logger.info("planner.seed_reference_discovery count=%d", len(discovered))
        return discovered

    # =====================================================================
    # Reference discovery
    # =====================================================================

    async def discover_references(
        self,
        plan: PaperPlan,
        existing_ref_keys: List[str],
        paper_search_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Discover additional references for each section based on the plan.
        - **Description**:
            - Analyzes each section's key points and generates search queries.
            - Executes multi-round searches via PaperSearchTool.
            - Supports loop searching until target count is reached or no more queries.
            - Returns discovered papers grouped by section_type.
            - Called once during planning, replacing per-section search judgment.

        - **Args**:
            - `plan` (PaperPlan): The paper plan with section structures.
            - `existing_ref_keys` (List[str]): Already-available citation keys.
            - `paper_search_config` (dict, optional): PaperSearchTool config.

        - **Returns**:
            - `Dict[str, List[Dict]]`: section_type -> list of discovered papers
              (each with ref_id, bibtex, title, etc.)
        """
        import asyncio
        from ..shared.tools.paper_search import PaperSearchTool

        cfg = paper_search_config or {}
        tool = PaperSearchTool(
            serpapi_api_key=cfg.get("serpapi_api_key"),
            semantic_scholar_api_key=cfg.get("semantic_scholar_api_key"),
            timeout=cfg.get("timeout", 10),
            semantic_scholar_min_results_before_fallback=cfg.get(
                "semantic_scholar_min_results_before_fallback", 3
            ),
            enable_query_cache=cfg.get("enable_query_cache", True),
            cache_ttl_hours=cfg.get("cache_ttl_hours", 24),
        )

        # Read configuration for multi-round search
        results_per_round = cfg.get("search_results_per_round", 5)
        max_queries_per_section = cfg.get("planner_max_queries_per_section", 5)
        inter_round_delay = cfg.get("planner_inter_round_delay_sec", 1.5)
        min_target_papers = cfg.get("planner_min_target_papers_per_section", 3)

        # Build search queries from plan — multiple per section for multi-round search
        section_queries: Dict[str, List[str]] = {}
        section_targets: Dict[str, int] = {}  # Target paper count per section

        for sp in plan.sections:
            if sp.section_type in ("abstract", "conclusion"):
                continue
            key_points = sp.get_key_points()
            if not key_points:
                continue

            # Generate multiple search queries for this section
            queries = await self._generate_search_queries(
                sp.section_type, key_points, existing_ref_keys, plan.title,
            )

            # Store up to N queries per section for multi-round search
            if queries:
                section_queries[sp.section_type] = queries[:max_queries_per_section]
                # Target priority:
                # 1) planner-provided section citation budget
                # 2) fallback to paragraph-derived heuristic
                planner_budget = sp.citation_budget if isinstance(sp.citation_budget, dict) else {}
                planner_target = planner_budget.get("target_refs")
                if planner_target is not None:
                    try:
                        section_targets[sp.section_type] = max(1, int(planner_target))
                    except Exception:
                        n_paras = len(sp.paragraphs) if sp.paragraphs else 0
                        section_targets[sp.section_type] = max(n_paras, min_target_papers)
                else:
                    n_paras = len(sp.paragraphs) if sp.paragraphs else 0
                    section_targets[sp.section_type] = max(n_paras, min_target_papers)

        discovered: Dict[str, List[Dict[str, Any]]] = {}
        seen_keys: set = set(existing_ref_keys)

        for section_type, queries in section_queries.items():
            section_papers: List[Dict[str, Any]] = []
            target_count = section_targets.get(section_type, 3)
            round_num = 0

            # Collect candidates from all planned rounds, then filter/select to target_count.
            while round_num < len(queries):
                query = queries[round_num]

                if round_num > 0:
                    # Rate limiting between rounds with small jitter to reduce burst contention
                    jitter = random.uniform(0, 0.4)
                    await asyncio.sleep(max(0.0, inter_round_delay + jitter))

                try:
                    result = await tool.execute(query=query, max_results=results_per_round)
                    if not result.success:
                        round_num += 1
                        continue

                    papers = result.data.get("papers", []) if result.data else []

                    for paper in papers:
                        bkey = paper.get("bibtex_key", "")
                        bibtex = paper.get("bibtex", "")
                        if bkey and bibtex and bkey not in seen_keys:
                            seen_keys.add(bkey)
                            section_papers.append({
                                "ref_id": bkey,
                                "bibtex": bibtex,
                                "title": paper.get("title", ""),
                                "year": paper.get("year"),
                                "abstract": paper.get("abstract", ""),
                                "venue": paper.get("venue", ""),
                                "citation_count": paper.get("citation_count"),
                            })

                    logger.info(
                        "planner.search_round section=%s round=%d query='%s' found=%d total=%d",
                        section_type, round_num, query[:50], len(papers), len(section_papers),
                    )
                except Exception as e:
                    logger.warning("planner.search_error query='%s': %s", query, e)

                round_num += 1

            # Filter papers by relevance before storing
            if section_papers:
                raw_count = len(section_papers)
                # Get key points for this section from the plan
                section_key_points = []
                for sp in plan.sections:
                    if sp.section_type == section_type:
                        section_key_points = sp.get_key_points()
                        break

                # Filter by relevance using LLM
                filtered_papers = await self._filter_papers_by_relevance(
                    papers=section_papers,
                    section_type=section_type,
                    key_points=section_key_points,
                    paper_title=plan.title,
                )
                filtered_count = len(filtered_papers)

                # Keep exactly N when possible: select top target_count by quality.
                filtered_sorted = sorted(
                    filtered_papers,
                    key=lambda p: (
                        float(p.get("relevance_score") or 0.0),
                        int(p.get("citation_count") or 0),
                        int(p.get("year") or 0),
                    ),
                    reverse=True,
                )
                selected_papers = filtered_sorted[:target_count] if target_count > 0 else filtered_sorted

                if selected_papers:
                    discovered[section_type] = selected_papers
                    logger.info(
                        "planner.discovered_refs section=%s target=%d raw=%d filtered=%d selected=%d",
                        section_type, target_count, raw_count, filtered_count, len(selected_papers),
                    )

        total = sum(len(v) for v in discovered.values())
        logger.info("planner.reference_discovery_complete total=%d", total)
        return discovered

    def assign_references(
        self,
        plan: "PaperPlan",
        discovered: Dict[str, List[Dict[str, Any]]],
        core_ref_keys: List[str],
        paper_search_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Distribute references to sections, populating SectionPlan.assigned_refs.

        - **Description**:
            - Discovered refs are assigned to the section they were found for.
            - Core (user-provided) refs are assigned to all body sections
              so every section can cite them.
            - Abstract and conclusion get NO refs (citations forbidden there).
            - A single ref can appear in multiple sections.

        - **Args**:
            - `plan` (PaperPlan): The paper plan to mutate in-place.
            - `discovered` (Dict[str, List[Dict]]): section_type -> papers from
              discover_references().
            - `core_ref_keys` (List[str]): Citation keys of user-provided refs.
        """
        cfg = paper_search_config or {}
        budget_enabled = cfg.get("citation_budget_enabled", True)
        soft_cap = cfg.get("citation_budget_soft_cap", True)
        reserve_size = max(1, int(cfg.get("citation_budget_reserve_size", 4)))

        no_cite_sections = {"abstract", "conclusion"}
        for sp in plan.sections:
            if sp.section_type in no_cite_sections:
                sp.assigned_refs = []
                sp.budget_selected_refs = []
                sp.budget_reserve_refs = []
                sp.budget_must_use_refs = []
                sp.citation_budget = {
                    "enabled": budget_enabled,
                    "min_refs": 0,
                    "target_refs": 0,
                    "max_refs": 0,
                    "candidate_count": 0,
                    "selected_count": 0,
                    "soft_cap": soft_cap,
                }
                continue

            discovered_for_section = discovered.get(sp.section_type, [])
            discovered_ranked = self._rank_references_for_section(discovered_for_section)
            planner_hint_refs = [r for r in sp.get_all_references() if r]
            budget = self._infer_section_citation_budget(
                section_type=sp.section_type,
                paragraph_count=len(sp.paragraphs),
                candidate_refs=discovered_ranked,
                planner_hint_refs=planner_hint_refs,
                core_ref_keys=core_ref_keys,
                planner_budget=sp.citation_budget if isinstance(sp.citation_budget, dict) else {},
            )

            if not budget_enabled:
                refs: List[str] = list(core_ref_keys)
                for paper in discovered_ranked:
                    rid = paper.get("ref_id", "")
                    if rid and rid not in refs:
                        refs.append(rid)
                sp.assigned_refs = refs
                sp.budget_selected_refs = refs
                sp.budget_reserve_refs = []
                sp.budget_must_use_refs = planner_hint_refs[:3]
                budget["enabled"] = False
                budget["selected_count"] = len(refs)
                sp.citation_budget = budget
                continue

            selected_refs = list(budget.get("selected_refs", []))
            reserve_refs = list(budget.get("reserve_refs", []))
            must_use_refs = list(budget.get("must_use_refs", []))

            if not selected_refs:
                fallback = [k for k in planner_hint_refs if k in core_ref_keys]
                selected_refs = fallback[: max(1, budget.get("target_refs", 1))]
            sp.assigned_refs = selected_refs
            sp.budget_selected_refs = selected_refs
            sp.budget_reserve_refs = reserve_refs[:reserve_size]
            sp.budget_must_use_refs = must_use_refs
            budget["enabled"] = True
            budget["selected_count"] = len(selected_refs)
            budget["soft_cap"] = soft_cap
            sp.citation_budget = budget

        assigned_counts = {
            sp.section_type: len(sp.assigned_refs)
            for sp in plan.sections if sp.assigned_refs
        }
        logger.info("planner.assign_references result=%s", assigned_counts)

    def _rank_references_for_section(
        self,
        papers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rank section candidate papers by relevance, quality and recency.
        """
        return sorted(
            papers,
            key=lambda p: (
                float(p.get("relevance_score") or 0.0),
                int(p.get("citation_count") or 0),
                int(p.get("year") or 0),
            ),
            reverse=True,
        )

    def _infer_section_citation_budget(
        self,
        section_type: str,
        paragraph_count: int,
        candidate_refs: List[Dict[str, Any]],
        planner_hint_refs: List[str],
        core_ref_keys: List[str],
        planner_budget: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Infer per-section citation budget, prioritizing planner-provided budget.
        """
        candidate_keys = [p.get("ref_id", "") for p in candidate_refs if p.get("ref_id")]
        allowed_key_set = set(candidate_keys) | set(core_ref_keys)
        planner_hint_refs = [r for r in planner_hint_refs if r in allowed_key_set]
        high_quality = [
            p for p in candidate_refs
            if float(p.get("relevance_score") or 0.0) >= 8.0
            or int(p.get("citation_count") or 0) >= 80
        ]
        candidate_count = len(candidate_keys) + len(core_ref_keys)
        normalized_budget = planner_budget or {}
        budget_min = normalized_budget.get("min_refs")
        budget_target = normalized_budget.get("target_refs")
        budget_max = normalized_budget.get("max_refs")

        if budget_target is not None:
            target_refs = max(0, int(budget_target))
        else:
            # Fallback only when planner omitted budget: no section-specific hardcoding.
            complexity_signal = max(1, paragraph_count)
            evidence_signal = len(planner_hint_refs) + max(0, len(high_quality) // 2)
            target_refs = max(1, complexity_signal + evidence_signal)

        if budget_min is not None:
            min_refs = max(0, int(budget_min))
        else:
            min_refs = max(1, min(target_refs, paragraph_count))

        if budget_max is not None:
            max_refs = max(int(budget_max), target_refs, min_refs)
        else:
            # No fixed cap; upper bound is naturally candidate pool size.
            max_refs = max(target_refs, min_refs, candidate_count)

        must_use_refs: List[str] = []
        for rid in planner_hint_refs:
            if rid and rid not in must_use_refs:
                must_use_refs.append(rid)
            if len(must_use_refs) >= 3:
                break
        for p in high_quality:
            rid = p.get("ref_id", "")
            if rid and rid in allowed_key_set and rid not in must_use_refs:
                must_use_refs.append(rid)
            if len(must_use_refs) >= 4:
                break

        selected_refs: List[str] = []
        for rid in must_use_refs:
            if rid and rid not in selected_refs:
                selected_refs.append(rid)

        for rid in candidate_keys:
            if len(selected_refs) >= target_refs:
                break
            if rid and rid not in selected_refs:
                selected_refs.append(rid)

        for rid in core_ref_keys:
            if len(selected_refs) >= min_refs:
                break
            if rid and rid not in selected_refs:
                selected_refs.append(rid)

        reserve_refs: List[str] = []
        for rid in candidate_keys:
            if rid and rid not in selected_refs and rid not in reserve_refs:
                reserve_refs.append(rid)
            if len(reserve_refs) >= 8:
                break

        return {
            "section_type": section_type,
            "min_refs": min_refs,
            "target_refs": target_refs,
            "max_refs": max_refs,
            "candidate_count": candidate_count,
            "must_use_refs": must_use_refs,
            "selected_refs": selected_refs[:max_refs] if max_refs > 0 else selected_refs,
            "reserve_refs": reserve_refs,
            "planner_hint_refs": planner_hint_refs[:8],
            "planner_budget_used": bool(budget_target is not None),
        }

    def _build_context_fallback_payload(
        self,
        *,
        plan: "PaperPlan",
        discovered: Dict[str, List[Dict[str, Any]]],
        all_papers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build non-empty fallback research context when LLM JSON parsing fails.
        """
        paper_assignments = self._assign_papers_to_sections(plan, discovered)
        claim_evidence_matrix: List[Dict[str, Any]] = []
        for section_type, refs in paper_assignments.items():
            if section_type in {"abstract", "conclusion"}:
                continue
            if not refs:
                continue
            claim_evidence_matrix.append(
                {
                    "section_type": section_type,
                    "claim": f"Key findings and arguments in {section_type} should be supported by assigned evidence.",
                    "support_refs": refs[:4],
                    "reason": "Fallback mapping from section assignment after context parsing failure.",
                    "priority": "P1",
                }
            )

        contribs = list(plan.contributions or [])
        p0 = contribs[:2]
        p1 = contribs[2:4]
        p2 = contribs[4:6]
        contribution_ranking = {
            "P0": [
                {
                    "contribution": c,
                    "why_it_matters": "Core contribution from planner output.",
                    "suggested_sections": ["introduction", "methods", "results"],
                    "suggested_result_focus": "Highlight primary quantitative gains.",
                }
                for c in p0
            ],
            "P1": [
                {
                    "contribution": c,
                    "why_it_matters": "Important but secondary contribution.",
                    "suggested_sections": ["discussion", "results"],
                    "suggested_result_focus": "Position as supporting evidence.",
                }
                for c in p1
            ],
            "P2": [
                {
                    "contribution": c,
                    "why_it_matters": "Optional or auxiliary contribution.",
                    "suggested_sections": ["discussion"],
                    "suggested_result_focus": "Mention briefly if space allows.",
                }
                for c in p2
            ],
        }

        return {
            "research_area": "Research area analysis",
            "summary": f"Found {len(all_papers)} relevant papers across {len(discovered)} sections.",
            "key_papers": [],
            "research_trends": [],
            "gaps": [],
            "claim_evidence_matrix": claim_evidence_matrix,
            "contribution_ranking": contribution_ranking,
            "planning_decision_trace": [
                "Used heuristic fallback context because structured JSON parsing failed."
            ],
            "paper_assignments": paper_assignments,
        }

    async def _generate_search_queries(
        self,
        section_type: str,
        key_points: List[str],
        existing_refs: List[str],
        paper_title: str,
    ) -> List[str]:
        """
        Generate search queries for a section using a lightweight LLM call.
        - **Description**:
            - Asks the LLM to suggest 1-2 search queries based on the
              section's key points and gaps in existing references.

        - **Args**:
            - `section_type` (str): The section type.
            - `key_points` (List[str]): Key points from the plan.
            - `existing_refs` (List[str]): Available citation keys.
            - `paper_title` (str): Paper title for context.

        - **Returns**:
            - `List[str]`: Search queries (1-2 per section).
        """
        kp_text = "; ".join(key_points[:4])
        refs_text = ", ".join(existing_refs[:10]) if existing_refs else "none"
        prompt = (
            f"Paper: {paper_title}\n"
            f"Section: {section_type}\n"
            f"Key points: {kp_text}\n"
            f"Existing references: {refs_text}\n\n"
            "Generate 1-2 academic search queries to find relevant papers "
            "for this section. Output JSON: {\"queries\": [\"...\"]}"
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an academic research assistant. Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            raw = response.choices[0].message.content or ""
            data = self._safe_load_json(raw, expected=dict)
            if data is None:
                raise ValueError("Could not parse JSON object from query generation output")
            queries = data.get("queries", [])
            return [q for q in queries if isinstance(q, str) and len(q.strip()) > 3]
        except Exception as e:
            logger.warning("planner.query_generation_error section=%s: %s", section_type, e)
            return []

    async def _filter_papers_by_relevance(
        self,
        papers: List[Dict[str, Any]],
        section_type: str,
        key_points: List[str],
        paper_title: str,
    ) -> List[Dict[str, Any]]:
        """
        Filter discovered papers by relevance to a specific section.

        - **Description**:
            - Uses LLM to evaluate each paper's relevance, quality, and timeliness.
            - Scores papers on relevance (0-10), quality, and recency.
            - Returns filtered list with relevance scores.

        - **Args**:
            - `papers` (List[Dict]): List of discovered papers.
            - `section_type` (str): The section type.
            - `key_points` (List[str]): Key points from the plan.
            - `paper_title` (str): Paper title for context.

        - **Returns**:
            - `List[Dict]`: Filtered papers with relevance scores.
        """
        if not papers:
            return []

        # Prepare paper information for LLM evaluation
        paper_list = []
        for i, p in enumerate(papers):
            paper_list.append({
                "index": i,
                "title": p.get("title", ""),
                "year": p.get("year", ""),
                "venue": p.get("venue", ""),
                "abstract": p.get("abstract", "")[:300] if p.get("abstract") else "",
            })

        kp_text = "; ".join(key_points[:4])
        papers_json = json.dumps(paper_list, ensure_ascii=False)

        prompt = (
            f"Paper: {paper_title}\n"
            f"Section: {section_type}\n"
            f"Key points: {kp_text}\n\n"
            f"Discovered papers:\n{papers_json}\n\n"
            "Evaluate each paper's relevance to this section on a scale of 0-10. "
            "Consider: (1) relevance to key points, (2) paper quality (venue, citations), "
            "(3) recency (prefer papers from the last 5 years). "
            "Output JSON array with format: "
            "[{\"index\": 0, \"relevance_score\": 8, \"reason\": \"brief justification\"}]"
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an academic research assistant. Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            raw = response.choices[0].message.content or ""
            evaluations = self._safe_load_json(raw, expected=list)
            if evaluations is None:
                raise ValueError("Could not parse JSON array from relevance output")

            # Build score lookup
            score_map: Dict[int, Dict[str, Any]] = {}
            for ev in evaluations:
                idx = ev.get("index")
                if idx is not None and 0 <= idx < len(papers):
                    score_map[idx] = {
                        "relevance_score": ev.get("relevance_score", 0),
                        "reason": ev.get("reason", ""),
                    }

            # Filter papers with score >= 5
            filtered = []
            for i, paper in enumerate(papers):
                score_info = score_map.get(i, {})
                score = score_info.get("relevance_score", 0)
                if score >= 5:
                    paper["relevance_score"] = score
                    paper["relevance_reason"] = score_info.get("reason", "")
                    filtered.append(paper)

            logger.info(
                "planner.filter_papers section=%s input=%d output=%d",
                section_type, len(papers), len(filtered),
            )
            return filtered

        except Exception as e:
            logger.warning("planner.filter_error section=%s: %s", section_type, e)
            # Return all papers if filtering fails
            return papers

    async def _generate_research_context(
        self,
        plan: "PaperPlan",
        discovered: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Generate a research context summary from discovered papers.

        - **Description**:
            - Analyzes all discovered papers to generate a research overview.
            - Identifies key papers, research trends, and gaps.
            - Assigns papers to appropriate sections.
            - Returns a structured context dictionary.

        - **Args**:
            - `plan` (PaperPlan): The paper plan.
            - `discovered` (Dict[str, List[Dict]]): section_type -> papers.

        - **Returns**:
            - `Dict`: Research context with summary, key_papers, paper_assignments, etc.
        """
        # Collect all discovered papers
        all_papers: List[Dict[str, Any]] = []
        for section_papers in discovered.values():
            all_papers.extend(section_papers)

        if not all_papers:
            return {
                "research_area": "",
                "summary": "No papers discovered.",
                "key_papers": [],
                "research_trends": [],
                "gaps": [],
                "paper_assignments": {},
            }

        # Prepare paper information for LLM (cap volume to reduce malformed/truncated outputs)
        ranked_papers = sorted(
            all_papers,
            key=lambda p: (
                int(p.get("citation_count") or 0),
                int(p.get("year") or 0),
            ),
            reverse=True,
        )
        analysis_papers = ranked_papers[:24]
        paper_summaries = []
        for p in analysis_papers:
            paper_summaries.append({
                "title": p.get("title", ""),
                "year": p.get("year"),
                "venue": p.get("venue", ""),
                "citation_count": p.get("citation_count"),
                "abstract": p.get("abstract", "")[:200] if p.get("abstract") else "",
            })

        papers_json = json.dumps(paper_summaries, ensure_ascii=False)

        prompt = (
            f"Paper title: {plan.title}\n\n"
            f"Discovered papers:\n{papers_json}\n\n"
            "Analyze these papers and provide a JSON response with:\n"
            "1. research_area: Main research area/topic (brief)\n"
            "2. summary: Overview of the research landscape (2-3 sentences)\n"
            "3. key_papers: Top 5 most important papers with their contributions\n"
            "4. research_trends: 2-3 key research trends identified\n"
            "5. gaps: 2-3 research gaps or opportunities\n"
            "6. claim_evidence_matrix: 6-10 records with {section_type, claim, support_refs, reason, priority}\n"
            "7. contribution_ranking: object with keys P0/P1/P2, each item has "
            "{contribution, why_it_matters, suggested_sections, suggested_result_focus}\n"
            "8. planning_decision_trace: short list of explicit trade-off decisions\n"
            "Output ONLY JSON with this structure:\n"
            "{\"research_area\": \"...\", \"summary\": \"...\", "
            "\"key_papers\": [{\"title\": \"...\", \"contribution\": \"...\"}], "
            "\"research_trends\": [\"...\"], \"gaps\": [\"...\"], "
            "\"claim_evidence_matrix\": [{\"section_type\": \"method\", \"claim\": \"...\", "
            "\"support_refs\": [\"ref1\"], \"reason\": \"...\", \"priority\": \"P0\"}], "
            "\"contribution_ranking\": {\"P0\": [], \"P1\": [], \"P2\": []}, "
            "\"planning_decision_trace\": [\"...\"]}"
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an academic research analyst. Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1400,
            )
            raw = response.choices[0].message.content or ""
            context = self._safe_load_json(raw, expected=dict)
            if context is None:
                # Retry once with a dedicated JSON-repair pass to reduce fallback rate.
                repair_prompt = (
                    "Convert the following model output into a STRICT valid JSON object. "
                    "Keep only these keys: research_area, summary, key_papers, research_trends, gaps, "
                    "claim_evidence_matrix, contribution_ranking, planning_decision_trace.\n"
                    "Rules:\n"
                    "- Output ONLY JSON object, no markdown/code fences.\n"
                    "- If a field is missing, fill with empty default.\n"
                    "- contribution_ranking must be an object with keys P0/P1/P2 (arrays).\n"
                    "- claim_evidence_matrix and planning_decision_trace must be arrays.\n\n"
                    f"Raw output:\n{raw[:12000]}"
                )
                repair_resp = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a strict JSON fixer. Return JSON object only.",
                        },
                        {"role": "user", "content": repair_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=1400,
                )
                repaired_raw = repair_resp.choices[0].message.content or ""
                context = self._safe_load_json(repaired_raw, expected=dict)
            if context is None:
                raise ValueError("Could not parse JSON object from research context output")

            # Generate paper assignments
            paper_assignments = self._assign_papers_to_sections(plan, discovered)
            fallback_context = self._build_context_fallback_payload(
                plan=plan,
                discovered=discovered,
                all_papers=all_papers,
            )

            parsed_claim_matrix = context.get("claim_evidence_matrix", [])
            parsed_ranking = context.get("contribution_ranking", {"P0": [], "P1": [], "P2": []})
            ranking_empty = (
                not isinstance(parsed_ranking, dict)
                or (
                    not parsed_ranking.get("P0")
                    and not parsed_ranking.get("P1")
                    and not parsed_ranking.get("P2")
                )
            )

            return {
                "research_area": context.get("research_area", ""),
                "summary": context.get("summary", ""),
                "key_papers": context.get("key_papers", [])[:10],
                "research_trends": context.get("research_trends", []),
                "gaps": context.get("gaps", []),
                "claim_evidence_matrix": (
                    parsed_claim_matrix
                    if parsed_claim_matrix
                    else fallback_context.get("claim_evidence_matrix", [])
                ),
                "contribution_ranking": (
                    parsed_ranking
                    if not ranking_empty
                    else fallback_context.get("contribution_ranking", {"P0": [], "P1": [], "P2": []})
                ),
                "planning_decision_trace": context.get("planning_decision_trace", []),
                "paper_assignments": paper_assignments,
            }

        except Exception as e:
            logger.warning("planner.context_generation_error: %s", e)
            return self._build_context_fallback_payload(
                plan=plan,
                discovered=discovered,
                all_papers=all_papers,
            )

    def _assign_papers_to_sections(
        self,
        plan: "PaperPlan",
        discovered: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[str]]:
        """
        Assign papers to sections based on where they were discovered.

        - **Args**:
            - `plan` (PaperPlan): The paper plan.
            - `discovered` (Dict[str, List[Dict]]): section_type -> papers.

        - **Returns**:
            - `Dict[str, List[str]]`: section_type -> citation keys.
        """
        assignments: Dict[str, List[str]] = {}

        for section_type, papers in discovered.items():
            citation_keys = [p.get("ref_id", "") for p in papers if p.get("ref_id")]
            assignments[section_type] = citation_keys

        return assignments

    # =====================================================================
    # VLM analysis
    # =====================================================================

    async def _analyze_figures(
        self, figures: List[Any],
    ) -> Dict[str, Any]:
        """Analyze figure images with VLM (returns {fig_id: FigureAnalysis})."""
        results = {}
        if not self.vlm_service:
            return results
        for fig in figures:
            file_path = getattr(fig, "file_path", "") or ""
            if not file_path:
                continue
            try:
                analysis = await self.vlm_service.analyze_figure(file_path)
                results[fig.id] = analysis
                logger.info("planner.vlm_figure id=%s role=%s", fig.id, analysis.semantic_role)
            except Exception as e:
                logger.warning("planner.vlm_figure_error id=%s: %s", fig.id, e)
        return results

    async def _analyze_tables(
        self, tables: List[Any],
    ) -> Dict[str, Any]:
        """Analyze table images with VLM (returns {tbl_id: TableAnalysis})."""
        results = {}
        if not self.vlm_service:
            return results
        for tbl in tables:
            file_path = getattr(tbl, "file_path", "") or ""
            if not file_path:
                continue
            try:
                analysis = await self.vlm_service.analyze_table_image(file_path)
                results[tbl.id] = analysis
                logger.info("planner.vlm_table id=%s role=%s", tbl.id, analysis.semantic_role)
            except Exception as e:
                logger.warning("planner.vlm_table_error id=%s: %s", tbl.id, e)
        return results

    # =====================================================================
    # Building PaperPlan from LLM output
    # =====================================================================

    def _build_paper_plan(
        self,
        plan_data: Dict[str, Any],
        request: PlanRequest,
        total_words: int,
        figure_analyses: Optional[Dict[str, Any]] = None,
        table_analyses: Optional[Dict[str, Any]] = None,
    ) -> PaperPlan:
        """Build PaperPlan from LLM JSON output and VLM analyses."""

        paper_type_str = plan_data.get("paper_type", "empirical").lower()
        try:
            paper_type = PaperType(paper_type_str)
        except ValueError:
            paper_type = PaperType.EMPIRICAL

        style_str = plan_data.get("narrative_style", "technical").lower()
        try:
            narrative_style = NarrativeStyle(style_str)
        except ValueError:
            narrative_style = NarrativeStyle.TECHNICAL

        ratios = SECTION_RATIOS_BY_TYPE.get(
            paper_type, SECTION_RATIOS_BY_TYPE[PaperType.EMPIRICAL],
        )

        # Determine section ordering from LLM output
        llm_sections = plan_data.get("sections", [])
        section_map: Dict[str, Dict[str, Any]] = {}
        llm_section_order: List[str] = []
        for s in llm_sections:
            st = s.get("section_type")
            if st and st not in section_map:
                section_map[st] = s
                llm_section_order.append(st)

        if len(llm_section_order) >= 3 and "abstract" not in llm_section_order:
            llm_section_order.insert(0, "abstract")
        if len(llm_section_order) >= 3 and "conclusion" not in llm_section_order:
            llm_section_order.append("conclusion")

        use_llm_structure = len(llm_section_order) >= 3
        section_type_order = llm_section_order if use_llm_structure else list(DEFAULT_EMPIRICAL_SECTIONS)

        # Distribute sentence budget across sections
        total_sentences = total_words // WORDS_PER_SENTENCE
        known_ratio_sum = sum(ratios.get(st, 0) for st in section_type_order if st in ratios)
        unknown_sections = [st for st in section_type_order if st not in ratios]
        remaining_ratio = max(0.0, 1.0 - known_ratio_sum)
        per_unknown_ratio = (remaining_ratio / len(unknown_sections)) if unknown_sections else 0.0

        sections: List[SectionPlan] = []
        for order, section_type in enumerate(section_type_order):
            ratio = ratios.get(section_type, per_unknown_ratio)
            section_sentences = max(3, int(total_sentences * ratio))
            llm_section = section_map.get(section_type, {})

            # Parse paragraphs from LLM output
            raw_paragraphs = llm_section.get("paragraphs", [])
            paragraphs = self._parse_paragraph_plans(raw_paragraphs)
            if not paragraphs:
                paragraphs = self._generate_default_paragraphs(
                    section_type, section_sentences, llm_section,
                )

            # Parse figure/table placements
            raw_figures = llm_section.get("figures", [])
            raw_tables = llm_section.get("tables", [])
            figure_placements = self._build_figure_placements(
                raw_figures, figure_analyses or {},
            )
            table_placements = self._build_table_placements(
                raw_tables, table_analyses or {},
            )

            # Cross-section references
            figs_to_ref = llm_section.get("figures_to_reference", [])
            tbls_to_ref = llm_section.get("tables_to_reference", [])

            sections.append(SectionPlan(
                section_type=section_type,
                section_title=llm_section.get(
                    "section_title", self._get_section_title(section_type),
                ),
                paragraphs=paragraphs,
                figures=figure_placements,
                tables=table_placements,
                figures_to_reference=figs_to_ref,
                tables_to_reference=tbls_to_ref,
                content_sources=llm_section.get(
                    "content_sources", self._get_default_sources(section_type),
                ),
                depends_on=llm_section.get(
                    "depends_on", self._get_dependencies(section_type),
                ),
                citation_budget=llm_section.get("citation_budget", {}),
                topic_clusters=self._normalize_string_list(
                    llm_section.get("topic_clusters", []),
                    max_items=4,
                ),
                transition_intents=self._normalize_string_list(
                    llm_section.get("transition_intents", []),
                    max_items=3,
                ),
                sectioning_recommended=self._coerce_bool(
                    llm_section.get("sectioning_recommended", False),
                ),
                writing_guidance=llm_section.get("writing_guidance", ""),
                order=order,
            ))

        paper_plan = PaperPlan(
            title=request.title,
            paper_type=paper_type,
            sections=sections,
            contributions=plan_data.get("contributions", []),
            narrative_style=narrative_style,
            terminology=plan_data.get("terminology", {}),
            structure_rationale=plan_data.get("structure_rationale", ""),
            abstract_focus=plan_data.get("abstract_focus", ""),
        )

        # Assign any unassigned figures/tables to sections
        self._assign_figure_table_definitions(paper_plan, request, figure_analyses, table_analyses)

        return paper_plan

    def _parse_paragraph_plans(
        self, raw_paragraphs: List[Dict[str, Any]],
    ) -> List[ParagraphPlan]:
        """Parse paragraph plans from LLM JSON output."""
        paragraphs = []
        for raw in raw_paragraphs:
            if not isinstance(raw, dict):
                continue
            paragraphs.append(ParagraphPlan(
                key_point=raw.get("key_point", ""),
                supporting_points=raw.get("supporting_points", []),
                approx_sentences=raw.get("approx_sentences", 5),
                role=raw.get("role", "evidence"),
                references_to_cite=raw.get("references_to_cite", []),
                figures_to_reference=raw.get("figures_to_reference", []),
                tables_to_reference=raw.get("tables_to_reference", []),
            ))
        return paragraphs

    @staticmethod
    def _normalize_string_list(raw: Any, max_items: int = 5) -> List[str]:
        """
        Normalize mixed list/string into a clean bounded string list.
        """
        if isinstance(raw, str):
            items = [x.strip() for x in raw.split(",") if x.strip()]
        elif isinstance(raw, list):
            items = [str(x).strip() for x in raw if str(x).strip()]
        else:
            items = []
        # De-duplicate while preserving order
        deduped: List[str] = []
        for item in items:
            if item not in deduped:
                deduped.append(item)
            if len(deduped) >= max_items:
                break
        return deduped

    @staticmethod
    def _coerce_bool(raw: Any) -> bool:
        """
        Coerce bool from permissive raw JSON value.
        """
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        text = str(raw).strip().lower()
        return text in {"1", "true", "yes", "y", "recommended"}

    def _generate_default_paragraphs(
        self,
        section_type: str,
        section_sentences: int,
        llm_section: Dict[str, Any],
    ) -> List[ParagraphPlan]:
        """Generate default paragraph structure when LLM doesn't provide one."""
        # Try to use old-style key_points if available
        key_points = llm_section.get("key_points", [])
        refs = llm_section.get("references_to_cite", [])

        default_structures = {
            "abstract": [
                ("Research problem and motivation", "motivation", 2),
                ("Method and key results", "summary", 3),
            ],
            "introduction": [
                ("Research context and motivation", "motivation", 5),
                ("Problem statement and gap", "problem_statement", 4),
                ("Contributions", "summary", 4),
                ("Paper organization", "transition", 2),
            ],
            "related_work": [
                ("Prior work overview", "evidence", 5),
                ("Comparison and gaps", "comparison", 4),
            ],
            "method": [
                ("Overview of approach", "definition", 4),
                ("Technical details", "evidence", 6),
                ("Implementation specifics", "evidence", 4),
            ],
            "experiment": [
                ("Experimental setup", "definition", 4),
                ("Datasets and baselines", "evidence", 4),
            ],
            "result": [
                ("Main results", "evidence", 5),
                ("Analysis and discussion", "comparison", 4),
            ],
            "conclusion": [
                ("Summary of contributions", "summary", 4),
                ("Future work", "transition", 3),
            ],
        }

        if key_points:
            n_paragraphs = len(key_points)
            sentences_per = max(3, section_sentences // n_paragraphs)
            return [
                ParagraphPlan(
                    key_point=kp,
                    approx_sentences=sentences_per,
                    role="evidence",
                    references_to_cite=refs[:2] if i == 0 else [],
                )
                for i, kp in enumerate(key_points)
            ]

        structure = default_structures.get(section_type, [
            ("Main content", "evidence", max(3, section_sentences)),
        ])

        return [
            ParagraphPlan(
                key_point=kp, role=role, approx_sentences=sents,
            )
            for kp, role, sents in structure
        ]

    def _build_figure_placements(
        self,
        raw_figures: List[Dict[str, Any]],
        figure_analyses: Dict[str, Any],
    ) -> List[FigurePlacement]:
        """Build FigurePlacement objects from LLM output + VLM analysis."""
        placements = []
        for raw in raw_figures:
            if not isinstance(raw, dict):
                continue
            fig_id = raw.get("figure_id", "")
            if not fig_id:
                continue
            vlm = figure_analyses.get(fig_id)
            placements.append(FigurePlacement(
                figure_id=fig_id,
                semantic_role=(
                    getattr(vlm, "semantic_role", "") if vlm
                    else raw.get("semantic_role", "")
                ),
                message=(
                    getattr(vlm, "message", "") if vlm
                    else raw.get("message", "")
                ),
                is_wide=(
                    getattr(vlm, "is_wide", False) if vlm
                    else raw.get("is_wide", False)
                ),
                position_hint=raw.get("position_hint", "mid"),
                caption_guidance=(
                    getattr(vlm, "caption_guidance", "") if vlm
                    else raw.get("caption_guidance", "")
                ),
            ))
        return placements

    def _build_table_placements(
        self,
        raw_tables: List[Dict[str, Any]],
        table_analyses: Dict[str, Any],
    ) -> List[TablePlacement]:
        """Build TablePlacement objects from LLM output + VLM analysis."""
        placements = []
        for raw in raw_tables:
            if not isinstance(raw, dict):
                continue
            tbl_id = raw.get("table_id", "")
            if not tbl_id:
                continue
            vlm = table_analyses.get(tbl_id)
            placements.append(TablePlacement(
                table_id=tbl_id,
                semantic_role=(
                    getattr(vlm, "semantic_role", "") if vlm
                    else raw.get("semantic_role", "")
                ),
                message=(
                    getattr(vlm, "message", "") if vlm
                    else raw.get("message", "")
                ),
                is_wide=(
                    getattr(vlm, "is_wide", False) if vlm
                    else raw.get("is_wide", False)
                ),
                position_hint=raw.get("position_hint", "mid"),
            ))
        return placements

    # =====================================================================
    # Figure/Table assignment
    # =====================================================================

    def _assign_figure_table_definitions(
        self,
        paper_plan: PaperPlan,
        request: PlanRequest,
        figure_analyses: Optional[Dict[str, Any]] = None,
        table_analyses: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Ensure each figure/table is DEFINED in exactly one section."""
        all_figures = {f.id: f for f in (request.figures or [])}
        all_tables = {t.id: t for t in (request.tables or [])}

        if not all_figures and not all_tables:
            return

        figures_defined = set()
        tables_defined = set()
        for section in paper_plan.sections:
            figures_defined.update(f.figure_id for f in section.figures)
            tables_defined.update(t.table_id for t in section.tables)

        fa = figure_analyses or {}
        ta = table_analyses or {}

        # Assign unassigned figures
        for fig_id, fig_info in all_figures.items():
            if self._should_be_wide_figure(fig_info):
                if fig_id not in paper_plan.wide_figures:
                    paper_plan.wide_figures.append(fig_id)

            if fig_id in figures_defined:
                continue

            target = self._find_best_section(
                paper_plan, fig_id, fig_info,
                fa.get(fig_id),
                {
                    "architecture": "method", "overview": "method",
                    "framework": "method", "model": "method",
                    "pipeline": "method", "result": "result",
                    "ablation": "result", "comparison": "experiment",
                    "performance": "experiment",
                },
                fallback="method",
            )
            if target:
                vlm_data = fa.get(fig_id)
                target.figures.append(FigurePlacement(
                    figure_id=fig_id,
                    semantic_role=getattr(vlm_data, "semantic_role", "") if vlm_data else "",
                    message=getattr(vlm_data, "message", "") if vlm_data else "",
                    is_wide=getattr(vlm_data, "is_wide", False) if vlm_data else self._should_be_wide_figure(fig_info),
                    position_hint="mid",
                    caption_guidance=getattr(vlm_data, "caption_guidance", "") if vlm_data else "",
                ))
                figures_defined.add(fig_id)

        # Assign unassigned tables
        for tbl_id, tbl_info in all_tables.items():
            if self._should_be_wide_table(tbl_info):
                if tbl_id not in paper_plan.wide_tables:
                    paper_plan.wide_tables.append(tbl_id)

            if tbl_id in tables_defined:
                continue

            target = self._find_best_section(
                paper_plan, tbl_id, tbl_info,
                ta.get(tbl_id),
                {
                    "main": "experiment", "result": "experiment",
                    "comparison": "experiment", "ablation": "result",
                    "hyperparameter": "experiment", "statistics": "experiment",
                    "dataset": "experiment",
                },
                fallback="experiment",
            )
            if target:
                vlm_data = ta.get(tbl_id)
                target.tables.append(TablePlacement(
                    table_id=tbl_id,
                    semantic_role=getattr(vlm_data, "semantic_role", "") if vlm_data else "",
                    message=getattr(vlm_data, "message", "") if vlm_data else "",
                    is_wide=getattr(vlm_data, "is_wide", False) if vlm_data else self._should_be_wide_table(tbl_info),
                    position_hint="mid",
                ))
                tables_defined.add(tbl_id)

    def _find_best_section(
        self,
        plan: PaperPlan,
        element_id: str,
        element_info: Any,
        vlm_analysis: Optional[Any],
        hint_map: Dict[str, str],
        fallback: str,
    ) -> Optional[SectionPlan]:
        """Find the best section for a figure/table using VLM then keyword fallback."""
        # Try VLM suggestion first
        if vlm_analysis:
            suggested = getattr(vlm_analysis, "suggested_section", "")
            if suggested:
                for section in plan.sections:
                    if section.section_type == suggested:
                        return section

        # User-suggested section
        user_section = getattr(element_info, "section", "")
        if user_section:
            for section in plan.sections:
                if section.section_type == user_section:
                    return section

        # Keyword heuristic fallback
        text = (
            (element_info.id if hasattr(element_info, "id") else "")
            + " " + (element_info.caption if hasattr(element_info, "caption") else "")
            + " " + (element_info.description if hasattr(element_info, "description") else "")
        ).lower()
        for hint, section_type in hint_map.items():
            if hint in text:
                for section in plan.sections:
                    if section.section_type == section_type:
                        return section

        # Fallback
        body = plan.get_body_sections()
        for section in body:
            if section.section_type == fallback:
                return section
        return body[0] if body else None

    # =====================================================================
    # Default plan (fallback)
    # =====================================================================

    def _create_default_plan(
        self, request: PlanRequest, total_words: int,
    ) -> PaperPlan:
        """Create a default plan when LLM fails."""
        ratios = SECTION_RATIOS_BY_TYPE[PaperType.EMPIRICAL]
        total_sentences = total_words // WORDS_PER_SENTENCE

        sections = []
        for order, section_type in enumerate(DEFAULT_EMPIRICAL_SECTIONS):
            ratio = ratios.get(section_type, 0.1)
            section_sentences = max(3, int(total_sentences * ratio))
            paragraphs = self._generate_default_paragraphs(
                section_type, section_sentences, {},
            )
            sections.append(SectionPlan(
                section_type=section_type,
                section_title=self._get_section_title(section_type),
                paragraphs=paragraphs,
                content_sources=self._get_default_sources(section_type),
                depends_on=self._get_dependencies(section_type),
                order=order,
            ))

        plan = PaperPlan(
            title=request.title,
            paper_type=PaperType.EMPIRICAL,
            sections=sections,
            contributions=[f"We propose {request.title}"],
        )
        self._assign_figure_table_definitions(plan, request)
        return plan

    # =====================================================================
    # Helpers
    # =====================================================================

    @staticmethod
    def _extract_reference_keys(references: List[str]) -> List[str]:
        keys = []
        for ref in references:
            match = re.search(r"@\w+\{([^,]+)", ref)
            if match:
                keys.append(match.group(1).strip())
        return keys

    @staticmethod
    def _parse_plan_json(text: str) -> Dict[str, Any]:
        parsed = PlannerAgent._safe_load_json(text, expected=dict)
        if parsed is None:
            logger.warning("planner.json_parse_error, using defaults")
            return {}
        return parsed

    @staticmethod
    def _format_figure_info(
        figures: List[Any], analyses: Dict[str, Any],
    ) -> str:
        if not figures:
            return "None provided"
        lines = []
        for fig in figures:
            line = f"- {fig.id}: {fig.caption}"
            if fig.description:
                line += f" ({fig.description})"
            if fig.section:
                line += f" [suggested: {fig.section}]"
            vlm = analyses.get(fig.id)
            if vlm:
                line += f" [VLM: role={getattr(vlm, 'semantic_role', '')}, message={getattr(vlm, 'message', '')}]"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _format_table_info(
        tables: List[Any], analyses: Dict[str, Any],
    ) -> str:
        if not tables:
            return "None provided"
        lines = []
        for tbl in tables:
            line = f"- {tbl.id}: {tbl.caption}"
            if tbl.description:
                line += f" ({tbl.description})"
            if tbl.section:
                line += f" [suggested: {tbl.section}]"
            vlm = analyses.get(tbl.id)
            if vlm:
                line += f" [VLM: role={getattr(vlm, 'semantic_role', '')}, message={getattr(vlm, 'message', '')}]"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _get_section_title(section_type: str) -> str:
        titles = {
            "abstract": "Abstract",
            "introduction": "Introduction",
            "related_work": "Related Work",
            "method": "Method",
            "experiment": "Experiments",
            "result": "Results",
            "discussion": "Discussion",
            "conclusion": "Conclusion",
        }
        return titles.get(section_type, section_type.replace("_", " ").title())

    @staticmethod
    def _get_default_sources(section_type: str) -> List[str]:
        mapping = {
            "introduction": ["idea_hypothesis", "method"],
            "related_work": ["idea_hypothesis"],
            "method": ["method"],
            "experiment": ["experiments", "data"],
            "result": ["experiments"],
            "discussion": ["experiments", "method"],
            "conclusion": ["idea_hypothesis", "experiments"],
            "abstract": ["idea_hypothesis", "method", "experiments"],
        }
        return mapping.get(section_type, [])

    @staticmethod
    def _get_dependencies(section_type: str) -> List[str]:
        deps = {
            "related_work": ["introduction"],
            "method": ["introduction"],
            "experiment": ["method"],
            "result": ["experiment"],
            "discussion": ["result"],
            "conclusion": ["introduction", "result"],
            "abstract": ["introduction", "conclusion"],
        }
        return deps.get(section_type, [])

    @staticmethod
    def _should_be_wide_figure(fig_info: Any) -> bool:
        if getattr(fig_info, "wide", False):
            return True
        wide_keywords = [
            "comparison", "overview", "architecture", "pipeline",
            "framework", "full", "complete", "main", "overall",
            "workflow", "system",
        ]
        text = (
            (getattr(fig_info, "id", "") or "")
            + " " + (getattr(fig_info, "caption", "") or "")
            + " " + (getattr(fig_info, "description", "") or "")
        ).lower()
        return any(kw in text for kw in wide_keywords)

    @staticmethod
    def _should_be_wide_table(tbl_info: Any) -> bool:
        if getattr(tbl_info, "wide", False):
            return True
        wide_keywords = [
            "main", "comparison", "full", "complete", "all",
            "overall", "summary", "comprehensive",
        ]
        text = (
            (getattr(tbl_info, "id", "") or "")
            + " " + (getattr(tbl_info, "caption", "") or "")
            + " " + (getattr(tbl_info, "description", "") or "")
        ).lower()
        if any(kw in text for kw in wide_keywords):
            return True
        content = getattr(tbl_info, "content", None)
        if content:
            first_line = content.strip().split("\n")[0] if content.strip() else ""
            for sep, adj in [
                ("|", -1), (",", 1), ("\t", 1),
            ]:
                if sep in first_line:
                    col_count = first_line.count(sep) + adj
                    if col_count > 5:
                        return True
        return False

    async def create_plan_from_metadata(
        self,
        title: str,
        idea_hypothesis: str,
        method: str,
        data: str,
        experiments: str,
        references: List[str],
        target_pages: Optional[int] = None,
        style_guide: Optional[str] = None,
    ) -> PaperPlan:
        """Convenience method to create plan from individual fields."""
        request = PlanRequest(
            title=title,
            idea_hypothesis=idea_hypothesis,
            method=method,
            data=data,
            experiments=experiments,
            references=references,
            target_pages=target_pages,
            style_guide=style_guide,
        )
        return await self.create_plan(request)
