"""
Standard Planning Strategy
- **Description**:
    - Default planning strategy for empirical papers
    - Uses LLM to analyze metadata and create detailed plan
"""
import json
import logging
from typing import List, Dict, Any, Optional

from .base import PlanningStrategy
from ..models import (
    PaperPlan,
    SectionPlan,
    PlanRequest,
    PaperType,
    NarrativeStyle,
    SECTION_RATIOS_BY_TYPE,
    DEFAULT_EMPIRICAL_SECTIONS,
    VENUE_WORD_LIMITS,
    ELEMENT_PAGE_COST,
    calculate_total_words,
)


logger = logging.getLogger("uvicorn.error")


PLANNING_SYSTEM_PROMPT = """You are an expert academic paper planner. Your job is to analyze paper metadata and create a detailed writing plan.

Given the paper's idea/hypothesis, method, data, experiments, references, figures, and tables, you must:
1. Determine the paper type (empirical, theoretical, survey, position, system, benchmark)
2. Identify the key contributions (usually 2-4)
3. **Design the section structure** — you are free to choose which sections to include.
   - "abstract" and "conclusion" are REQUIRED.
   - For the body, you may use standard sections like "introduction", "related_work", "method", "experiment", "result",
     or you may merge / rename / add sections (e.g. "experiments_and_results", "discussion", "case_study").
   - Choose whatever structure best fits this particular paper.
4. Plan what each section should cover
5. Suggest which references to cite in each section
6. Assign figures and tables to appropriate sections
7. Provide a per-section word budget that sums to the total effective word budget

Output a JSON object with this structure:
{
    "paper_type": "empirical",
    "contributions": [
        "Contribution 1 statement",
        "Contribution 2 statement"
    ],
    "narrative_style": "technical",
    "terminology": {
        "key_term": "definition to use consistently"
    },
    "structure_rationale": "Why this structure works for this paper",
    "abstract_focus": "What the abstract should emphasize",
    "sections": [
        {
            "section_type": "introduction",
            "section_title": "Introduction",
            "target_words": 800,
            "key_points": ["Point 1", "Point 2"],
            "content_sources": ["idea_hypothesis", "method"],
            "references_to_cite": ["ref_key1", "ref_key2"],
            "figures_to_define": ["fig:architecture"],
            "tables_to_define": ["tab:results"],
            "writing_guidance": "Specific guidance for this section"
        }
    ]
}

IMPORTANT:
- The sum of all section target_words should equal the effective word budget.
- Sections that contain figures/tables should have FEWER words because those elements take space.
  Roughly deduct ~170 words per narrow figure, ~340 per wide figure, ~120 per narrow table, ~240 per wide table.
- "abstract" typically needs 100-200 words; "conclusion" needs 200-350 words.
- Be specific and actionable. The plan will guide an AI writer."""


PLANNING_USER_PROMPT_TEMPLATE = """Create a detailed paper plan for:

**Title**: {title}

**Idea/Hypothesis**:
{idea_hypothesis}

**Method**:
{method}

**Data**:
{data}

**Experiments**:
{experiments}

**Available References** (BibTeX keys):
{reference_keys}

**Available Figures**:
{figure_info}

**Available Tables**:
{table_info}

**Space Budget**:
- Target: {target_pages} pages for {style_guide}
- Non-text elements (figures + tables) occupy approximately {non_text_pages:.1f} pages
- Effective text space: approximately {effective_text_pages:.1f} pages ({total_words} words)
- When allocating words per section, subtract ~170 words per narrow figure and ~340 per wide figure defined in that section

Analyze this content and create a comprehensive writing plan. Focus on:
1. What are the 2-4 key contributions?
2. Which sections should the paper have? (you decide the structure)
3. How should word budget be distributed? (each section gets a target_words value)
4. Which references should be cited where?
5. Which figures and tables should be DEFINED in which sections?
6. What specific guidance helps each section?

Output valid JSON only."""


class StandardPlanningStrategy(PlanningStrategy):
    """
    Standard planning strategy for empirical papers
    - **Description**:
        - Uses LLM to analyze metadata
        - Creates detailed section plans
        - Allocates word budgets based on paper type
    """
    
    @property
    def name(self) -> str:
        return "standard"
    
    @property
    def description(self) -> str:
        return "Standard planning for empirical research papers"
    
    async def create_plan(
        self,
        request: PlanRequest,
        llm_client: Any,
        model_name: str,
    ) -> PaperPlan:
        """Create a paper plan using LLM analysis"""
        
        # Count figures/tables and detect wide elements
        n_figures = len(request.figures) if request.figures else 0
        n_tables = len(request.tables) if request.tables else 0
        n_wide_figures = sum(
            1 for f in (request.figures or []) if self._should_be_wide_figure(f)
        )
        n_wide_tables = sum(
            1 for t in (request.tables or []) if self._should_be_wide_table(t)
        )

        # Calculate total word budget (subtracting figure/table space)
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

        # Calculate raw words for display (without figure/table deduction)
        venue_key = (style_guide).upper().split()[0]
        _cfg = VENUE_WORD_LIMITS.get(venue_key, VENUE_WORD_LIMITS["DEFAULT"])
        raw_total = (target_pages) * _cfg["words_per_page"]
        words_per_page = _cfg["words_per_page"]

        # Estimate non-text element space for the prompt
        n_narrow_fig = max(0, n_figures - n_wide_figures)
        n_narrow_tbl = max(0, n_tables - n_wide_tables)
        fig_pages = (
            n_wide_figures * ELEMENT_PAGE_COST["figure*"]
            + n_narrow_fig * ELEMENT_PAGE_COST["figure"]
        )
        tbl_pages = (
            n_wide_tables * ELEMENT_PAGE_COST["table*"]
            + n_narrow_tbl * ELEMENT_PAGE_COST["table"]
        )
        non_text_pages = fig_pages + tbl_pages

        logger.info(
            "planner.word_budget raw=%d adjusted=%d non_text_pages=%.1f "
            "figs=%d(wide=%d) tables=%d(wide=%d)",
            raw_total, total_words, non_text_pages,
            n_figures, n_wide_figures, n_tables, n_wide_tables,
        )
        
        # Extract reference keys from BibTeX
        reference_keys = self._extract_reference_keys(request.references)
        
        # Format figure info
        figure_info = "None provided"
        if request.figures:
            fig_lines = []
            for fig in request.figures:
                line = f"- {fig.id}: {fig.caption}"
                if fig.description:
                    line += f" ({fig.description})"
                if fig.section:
                    line += f" [suggested: {fig.section}]"
                fig_lines.append(line)
            figure_info = "\n".join(fig_lines)
        
        # Format table info
        table_info = "None provided"
        if request.tables:
            tbl_lines = []
            for tbl in request.tables:
                line = f"- {tbl.id}: {tbl.caption}"
                if tbl.description:
                    line += f" ({tbl.description})"
                if tbl.section:
                    line += f" [suggested: {tbl.section}]"
                tbl_lines.append(line)
            table_info = "\n".join(tbl_lines)
        
        # Calculate effective text pages for the prompt
        effective_text_pages = target_pages - non_text_pages
        effective_text_pages = max(effective_text_pages, target_pages * 0.4)

        # Build prompt
        user_prompt = PLANNING_USER_PROMPT_TEMPLATE.format(
            title=request.title,
            idea_hypothesis=request.idea_hypothesis[:2000],
            method=request.method[:2000],
            data=request.data[:1500],
            experiments=request.experiments[:2000],
            reference_keys=", ".join(reference_keys) if reference_keys else "None provided",
            figure_info=figure_info,
            table_info=table_info,
            target_pages=target_pages,
            total_words=total_words,
            style_guide=style_guide,
            non_text_pages=non_text_pages,
            effective_text_pages=effective_text_pages,
        )
        
        try:
            # Call LLM for planning
            response = await llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Lower temperature for consistent planning
            )
            
            plan_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            plan_data = self._parse_plan_json(plan_text)
            
            # Build PaperPlan from LLM output
            paper_plan = self._build_paper_plan(
                plan_data=plan_data,
                request=request,
                total_words=total_words,
            )
            
            logger.info(
                "planner.plan_created title=%s sections=%d words=%d",
                request.title[:30],
                len(paper_plan.sections),
                paper_plan.total_target_words,
            )
            
            return paper_plan
            
        except Exception as e:
            logger.error("planner.llm_error: %s", str(e))
            # Fall back to default plan
            return self._create_default_plan(request, total_words)
    
    def _extract_reference_keys(self, references: List[str]) -> List[str]:
        """Extract BibTeX keys from reference entries"""
        import re
        keys = []
        for ref in references:
            match = re.search(r'@\w+\{([^,]+)', ref)
            if match:
                keys.append(match.group(1).strip())
        return keys
    
    def _parse_plan_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        # Try to extract JSON from markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("planner.json_parse_error, using defaults")
            return {}
    
    def _build_paper_plan(
        self,
        plan_data: Dict[str, Any],
        request: PlanRequest,
        total_words: int,
    ) -> PaperPlan:
        """
        Build PaperPlan from LLM output.
        - **Description**:
            - Uses LLM-returned section list as primary structure.
            - Ensures 'abstract' and 'conclusion' always exist.
            - Falls back to DEFAULT_EMPIRICAL_SECTIONS only when LLM
              returns empty / invalid sections.
        """
        
        # Determine paper type
        paper_type_str = plan_data.get("paper_type", "empirical").lower()
        try:
            paper_type = PaperType(paper_type_str)
        except ValueError:
            paper_type = PaperType.EMPIRICAL
        
        # Get word ratios for this paper type (used as fallback for unknown sections)
        ratios = SECTION_RATIOS_BY_TYPE.get(paper_type, SECTION_RATIOS_BY_TYPE[PaperType.EMPIRICAL])
        
        # Determine narrative style
        style_str = plan_data.get("narrative_style", "technical").lower()
        try:
            narrative_style = NarrativeStyle(style_str)
        except ValueError:
            narrative_style = NarrativeStyle.TECHNICAL
        
        # Resolve words_per_page for per-section figure deduction
        venue_key = ((request.style_guide or "DEFAULT").upper().split()[0])
        _wpp_cfg = VENUE_WORD_LIMITS.get(venue_key, VENUE_WORD_LIMITS["DEFAULT"])
        words_per_page = _wpp_cfg["words_per_page"]

        # --- Determine section ordering ---
        # Primary: use the LLM-returned sections list
        llm_sections = plan_data.get("sections", [])
        section_map: Dict[str, Dict[str, Any]] = {}
        llm_section_order: List[str] = []
        for s in llm_sections:
            st = s.get("section_type")
            if st and st not in section_map:
                section_map[st] = s
                llm_section_order.append(st)

        # Validate: must contain at least 3 sections including abstract & conclusion
        if len(llm_section_order) >= 3 and "abstract" not in llm_section_order:
            llm_section_order.insert(0, "abstract")
        if len(llm_section_order) >= 3 and "conclusion" not in llm_section_order:
            llm_section_order.append("conclusion")

        use_llm_structure = len(llm_section_order) >= 3
        if use_llm_structure:
            section_type_order = llm_section_order
            logger.info(
                "planner.dynamic_sections using LLM structure: %s",
                section_type_order,
            )
        else:
            section_type_order = list(DEFAULT_EMPIRICAL_SECTIONS)
            logger.info(
                "planner.dynamic_sections fallback to default: %s",
                section_type_order,
            )

        # --- Distribute word budget ---
        # If LLM provided custom sections, distribute remaining budget evenly
        # among sections not in the predefined ratios.
        known_ratio_sum = sum(
            ratios.get(st, 0) for st in section_type_order if st in ratios
        )
        unknown_sections = [st for st in section_type_order if st not in ratios]
        # Reserve remaining ratio for unknown sections
        remaining_ratio = max(0.0, 1.0 - known_ratio_sum)
        per_unknown_ratio = (remaining_ratio / len(unknown_sections)) if unknown_sections else 0.0

        sections = []
        for order, section_type in enumerate(section_type_order):
            # Get ratio: predefined or evenly distributed remainder
            ratio = ratios.get(section_type, per_unknown_ratio)
            # Allow LLM to override target_words directly
            llm_section = section_map.get(section_type, {})
            llm_target = llm_section.get("target_words")
            if isinstance(llm_target, (int, float)) and llm_target > 0:
                target_words_section = int(llm_target)
            else:
                target_words_section = int(total_words * ratio)
            
            # Determine default content sources
            default_sources = self._get_default_sources(section_type)
            
            # Handle figures: distinguish define vs reference
            figures_to_define = llm_section.get("figures_to_define", [])
            figures_to_reference = llm_section.get("figures_to_reference", [])
            # Backward compat: if only figures_to_use provided, treat as reference
            if not figures_to_define and not figures_to_reference:
                figures_to_reference = llm_section.get("figures_to_use", [])
            
            # Handle tables: distinguish define vs reference
            tables_to_define = llm_section.get("tables_to_define", [])
            tables_to_reference = llm_section.get("tables_to_reference", [])
            # Backward compat: if only tables_to_use provided, treat as reference
            if not tables_to_define and not tables_to_reference:
                tables_to_reference = llm_section.get("tables_to_use", [])
            
            # Deduct per-section space consumed by figures/tables defined here
            n_figs_in_section = len(figures_to_define)
            n_tbls_in_section = len(tables_to_define)
            if n_figs_in_section > 0 or n_tbls_in_section > 0:
                fig_page_cost = n_figs_in_section * ELEMENT_PAGE_COST["figure"]
                tbl_page_cost = n_tbls_in_section * ELEMENT_PAGE_COST["table"]
                deduction = int((fig_page_cost + tbl_page_cost) * words_per_page)
                # Don't reduce below 30% of original allocation
                target_words_section = max(
                    target_words_section - deduction,
                    int(target_words_section * 0.3),
                )
            
            # Use LLM-provided title or auto-generate
            section_title = llm_section.get("section_title") or self._get_section_title(section_type)

            section_plan = SectionPlan(
                section_type=section_type,
                section_title=section_title,
                target_words=target_words_section,
                key_points=llm_section.get("key_points", []),
                content_sources=llm_section.get("content_sources", default_sources),
                references_to_cite=llm_section.get("references_to_cite", []),
                figures_to_use=llm_section.get("figures_to_use", []),  # Backward compat
                figures_to_define=figures_to_define,
                figures_to_reference=figures_to_reference,
                tables_to_use=llm_section.get("tables_to_use", []),  # Backward compat
                tables_to_define=tables_to_define,
                tables_to_reference=tables_to_reference,
                depends_on=self._get_dependencies(section_type),
                writing_guidance=llm_section.get("writing_guidance", ""),
                order=order,
            )
            sections.append(section_plan)
        
        paper_plan = PaperPlan(
            title=request.title,
            paper_type=paper_type,
            total_target_words=total_words,
            sections=sections,
            contributions=plan_data.get("contributions", []),
            narrative_style=narrative_style,
            terminology=plan_data.get("terminology", {}),
            structure_rationale=plan_data.get("structure_rationale", ""),
            abstract_focus=plan_data.get("abstract_focus", ""),
        )
        
        # Ensure each figure/table is defined in exactly one section
        self._assign_figure_table_definitions(paper_plan, request)
        
        return paper_plan
    
    def _create_default_plan(self, request: PlanRequest, total_words: int) -> PaperPlan:
        """Create a default plan when LLM fails"""
        ratios = SECTION_RATIOS_BY_TYPE[PaperType.EMPIRICAL]
        
        sections = []
        for order, section_type in enumerate(DEFAULT_EMPIRICAL_SECTIONS):
            ratio = ratios.get(section_type, 0.1)
            section_plan = SectionPlan(
                section_type=section_type,
                section_title=self._get_section_title(section_type),
                target_words=int(total_words * ratio),
                content_sources=self._get_default_sources(section_type),
                depends_on=self._get_dependencies(section_type),
                order=order,
            )
            sections.append(section_plan)
        
        return PaperPlan(
            title=request.title,
            paper_type=PaperType.EMPIRICAL,
            total_target_words=total_words,
            sections=sections,
            narrative_style=NarrativeStyle.TECHNICAL,
        )
    
    def _get_section_title(self, section_type: str) -> str:
        """Get display title for section type"""
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
    
    def _get_default_sources(self, section_type: str) -> List[str]:
        """Get default metadata fields for each section"""
        sources = {
            "abstract": ["idea_hypothesis", "method", "experiments"],
            "introduction": ["idea_hypothesis", "method", "data", "experiments"],
            "related_work": ["references"],
            "method": ["method"],
            "experiment": ["data", "experiments"],
            "result": ["experiments"],
            "discussion": ["experiments"],
            "conclusion": ["idea_hypothesis", "experiments"],
        }
        return sources.get(section_type, [])
    
    def _get_dependencies(self, section_type: str) -> List[str]:
        """Get section dependencies"""
        deps = {
            "abstract": ["introduction", "method", "experiment", "result", "conclusion"],
            "introduction": [],
            "related_work": ["introduction"],
            "method": ["introduction"],
            "experiment": ["method"],
            "result": ["experiment"],
            "discussion": ["result"],
            "conclusion": ["introduction", "result"],
        }
        return deps.get(section_type, [])
    
    def _should_be_wide_figure(self, fig_info) -> bool:
        """
        Determine if a figure should use figure* (double-column spanning).
        
        Rules:
        - User-specified wide=True always takes precedence
        - Keywords in caption/description suggesting wide layout
        - "overview", "comparison", "architecture", "pipeline" figures often need wide
        """
        # User explicitly set wide
        if getattr(fig_info, 'wide', False):
            return True
        
        # Check for keywords suggesting wide figure
        wide_keywords = [
            "overview", "comparison", "architecture", "pipeline", 
            "framework", "full", "complete", "main", "overall",
            "workflow", "system"
        ]
        
        text = (
            (fig_info.id if hasattr(fig_info, 'id') else "") + " " +
            (fig_info.caption if hasattr(fig_info, 'caption') else "") + " " +
            (fig_info.description if hasattr(fig_info, 'description') else "")
        ).lower()
        
        for keyword in wide_keywords:
            if keyword in text:
                return True
        
        return False
    
    def _should_be_wide_table(self, tbl_info) -> bool:
        """
        Determine if a table should use table* (double-column spanning).
        
        Rules:
        - User-specified wide=True always takes precedence
        - Tables with many columns (>5) should be wide
        - "main", "comparison", "full" tables often need wide
        - Check content for column count if available
        """
        # User explicitly set wide
        if getattr(tbl_info, 'wide', False):
            return True
        
        # Check for keywords suggesting wide table
        wide_keywords = [
            "main", "comparison", "full", "complete", "all",
            "overall", "summary", "comprehensive"
        ]
        
        text = (
            (tbl_info.id if hasattr(tbl_info, 'id') else "") + " " +
            (tbl_info.caption if hasattr(tbl_info, 'caption') else "") + " " +
            (tbl_info.description if hasattr(tbl_info, 'description') else "")
        ).lower()
        
        for keyword in wide_keywords:
            if keyword in text:
                return True
        
        # Try to estimate column count from content if available
        content = getattr(tbl_info, 'content', None)
        if content:
            # Check first line for column separators
            first_line = content.strip().split('\n')[0] if content.strip() else ""
            # Count pipes for markdown tables
            if '|' in first_line:
                col_count = first_line.count('|') - 1
                if col_count > 5:
                    return True
            # Count commas for CSV
            elif ',' in first_line:
                col_count = first_line.count(',') + 1
                if col_count > 5:
                    return True
            # Count tabs
            elif '\t' in first_line:
                col_count = first_line.count('\t') + 1
                if col_count > 5:
                    return True
        
        return False
    
    def _assign_figure_table_definitions(
        self, 
        paper_plan: PaperPlan, 
        request: PlanRequest
    ) -> None:
        """
        Ensure each figure/table is DEFINED in exactly one section.
        
        This method:
        1. Collects all figures/tables from the request
        2. Checks which sections already have them assigned for definition
        3. For any unassigned, assigns to the most appropriate section
        4. Ensures other sections only REFERENCE them
        5. Auto-detects and sets 'wide' flag for double-column spanning
        
        Default assignment rules:
        - Architecture/overview figures -> method
        - Result/ablation figures -> result or experiment
        - Main results tables -> experiment or result
        - Ablation tables -> result
        
        Wide detection rules:
        - Figures: "comparison", "overview", "architecture" keywords
        - Tables: > 5 columns or "main", "comparison", "full" keywords
        """
        # Collect all available figures and tables
        all_figures = {f.id: f for f in request.figures}
        all_tables = {t.id: t for t in request.tables}
        
        if not all_figures and not all_tables:
            return
        
        # Track which are already assigned for definition
        figures_defined = set()
        tables_defined = set()
        
        for section in paper_plan.sections:
            figures_defined.update(section.figures_to_define)
            tables_defined.update(section.tables_to_define)
        
        # Default section assignments for figures
        figure_section_hints = {
            "architecture": "method",
            "overview": "method",
            "framework": "method",
            "model": "method",
            "pipeline": "method",
            "result": "result",
            "ablation": "result",
            "comparison": "experiment",
            "performance": "experiment",
        }
        
        # Default section assignments for tables
        table_section_hints = {
            "main": "experiment",
            "result": "experiment",
            "comparison": "experiment",
            "ablation": "result",
            "hyperparameter": "experiment",
            "statistics": "experiment",
            "dataset": "experiment",
        }
        
        # Assign unassigned figures and auto-detect wide
        for fig_id, fig_info in all_figures.items():
            # Auto-detect wide for all figures
            if self._should_be_wide_figure(fig_info):
                if fig_id not in paper_plan.wide_figures:
                    paper_plan.wide_figures.append(fig_id)
                    print(f"[Planner] Auto-detected figure '{fig_id}' as WIDE (double-column)")
            
            if fig_id in figures_defined:
                continue
            
            # Try to find the best section
            target_section = None
            
            # First, check user-suggested section
            if fig_info.section:
                for section in paper_plan.sections:
                    if section.section_type == fig_info.section:
                        target_section = section
                        break
            
            # Second, infer from figure ID/caption
            if not target_section:
                fig_lower = (fig_id + " " + fig_info.caption + " " + fig_info.description).lower()
                for hint, section_type in figure_section_hints.items():
                    if hint in fig_lower:
                        for section in paper_plan.sections:
                            if section.section_type == section_type:
                                target_section = section
                                break
                    if target_section:
                        break
            
            # Default: assign to method section, or first body section if method missing
            if not target_section:
                body = paper_plan.get_body_sections()
                for section in body:
                    if section.section_type == "method":
                        target_section = section
                        break
                if not target_section and body:
                    target_section = body[0]
            
            # Assign to definition
            if target_section:
                target_section.figures_to_define.append(fig_id)
                figures_defined.add(fig_id)
                print(f"[Planner] Assigned figure '{fig_id}' to be DEFINED in '{target_section.section_type}'")
        
        # Assign unassigned tables and auto-detect wide
        for tbl_id, tbl_info in all_tables.items():
            # Auto-detect wide for all tables
            if self._should_be_wide_table(tbl_info):
                if tbl_id not in paper_plan.wide_tables:
                    paper_plan.wide_tables.append(tbl_id)
                    print(f"[Planner] Auto-detected table '{tbl_id}' as WIDE (double-column)")
            
            if tbl_id in tables_defined:
                continue
            
            target_section = None
            
            # First, check user-suggested section
            if tbl_info.section:
                for section in paper_plan.sections:
                    if section.section_type == tbl_info.section:
                        target_section = section
                        break
            
            # Second, infer from table ID/caption
            if not target_section:
                tbl_lower = (tbl_id + " " + tbl_info.caption + " " + tbl_info.description).lower()
                for hint, section_type in table_section_hints.items():
                    if hint in tbl_lower:
                        for section in paper_plan.sections:
                            if section.section_type == section_type:
                                target_section = section
                                break
                    if target_section:
                        break
            
            # Default: assign to experiment section, or last body section
            if not target_section:
                body = paper_plan.get_body_sections()
                for section in body:
                    if section.section_type == "experiment":
                        target_section = section
                        break
                if not target_section and body:
                    target_section = body[-1]
            
            if target_section:
                target_section.tables_to_define.append(tbl_id)
                tables_defined.add(tbl_id)
                print(f"[Planner] Assigned table '{tbl_id}' to be DEFINED in '{target_section.section_type}'")
        
        # Now, for sections that reference but don't define, move to reference list
        for section in paper_plan.sections:
            # Handle figures_to_use (legacy)
            for fig_id in list(section.figures_to_use):
                if fig_id not in section.figures_to_define:
                    if fig_id not in section.figures_to_reference:
                        section.figures_to_reference.append(fig_id)
            
            # Handle tables_to_use (legacy)
            for tbl_id in list(section.tables_to_use):
                if tbl_id not in section.tables_to_define:
                    if tbl_id not in section.tables_to_reference:
                        section.tables_to_reference.append(tbl_id)
