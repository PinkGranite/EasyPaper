"""
Prompt Compiler - Shared prompt generation utilities
- **Description**:
    - Compiles section plans into LLM prompts
    - Uses paragraph-level structure from PaperPlan
    - Provides section-specific prompt templates
    - Used by both Commander Agent and MetaData Agent
"""
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import json
import os

if TYPE_CHECKING:
    from ...skills.models import WritingSkill


def _inject_skill_constraints(
    prompt_parts: list,
    active_skills: Optional[List["WritingSkill"]],
    section_type: str,
) -> None:
    """
    Inject writing-style constraints from active skills into prompt_parts (in-place).

    - **Args**:
        - `prompt_parts` (list): Mutable list of prompt segments to append to
        - `active_skills` (List[WritingSkill] | None): Skills loaded from the registry
        - `section_type` (str): Current section being written
    """
    if not active_skills:
        return
    matched = [
        s for s in active_skills
        if "*" in s.target_sections or section_type in s.target_sections
    ]
    matched.sort(key=lambda s: s.priority)
    if matched:
        constraints = "\n\n".join(
            s.system_prompt_append for s in matched if s.system_prompt_append
        )
        if constraints:
            prompt_parts.append(f"\n## Writing Style Constraints\n{constraints}")


# =============================================================================
# Section Prompt Templates
# =============================================================================

SECTION_PROMPTS: Dict[str, str] = {
    "abstract": """You are writing the Abstract section of a research paper.
The abstract should:
- Summarize the research problem and motivation (1-2 sentences)
- Describe the methodology briefly (1-2 sentences)
- Present key results and findings (1-2 sentences)
- State the main conclusions and implications (1-2 sentences)
Keep it concise, typically 150-250 words.""",

    "introduction": """You are writing the Introduction section of a research paper.
The introduction should:
- Establish the research context and background
- Identify the problem or gap in current knowledge
- State the research objectives and contributions
- Outline the paper structure
Use a clear narrative flow from general to specific.""",

    "related_work": """You are writing the Related Work section of a research paper.
This section should:
- Survey relevant prior work systematically
- Group related works by theme or approach
- Identify gaps that your work addresses
- Clearly differentiate your contribution from existing work
Use proper citations throughout.""",

    "method": """You are writing the Method/Methodology section of a research paper.
This section should:
- Describe your approach in sufficient detail for reproduction
- Explain the rationale behind methodological choices
- Include formal definitions, algorithms, or models as needed
- Use clear notation and terminology consistently.""",

    "experiment": """You are writing the Experiment section of a research paper.
This section should:
- Describe the experimental setup and configuration
- Specify datasets, metrics, and baselines used
- Explain evaluation protocols and procedures
- Provide implementation details as necessary.""",

    "result": """You are writing the Results section of a research paper.
This section should:
- Present experimental results clearly and objectively
- Use tables and figures to support key findings
- Compare against baselines and prior work
- Highlight statistically significant results.""",

    "discussion": """You are writing the Discussion section of a research paper.
This section should:
- Interpret the results in context of research questions
- Discuss implications and significance of findings
- Address limitations and potential threats to validity
- Suggest directions for future work.""",

    "conclusion": """You are writing the Conclusion section of a research paper.
This section should:
- Summarize the main contributions concisely
- Restate key findings and their significance
- Discuss broader impact and applications
- End with forward-looking perspective.""",
}


# =============================================================================
# Paragraph-level writing structure
# =============================================================================

def _format_paragraph_guidance(section_plan: Any) -> str:
    """
    Format paragraph-level writing guidance from a SectionPlan.

    - **Args**:
        - `section_plan`: SectionPlan object with paragraphs list

    - **Returns**:
        - `str`: Formatted paragraph guidance for the LLM prompt
    """
    paragraphs = getattr(section_plan, "paragraphs", None)
    if not paragraphs:
        # Backward-compat: fall back to key_points + target_words
        parts = []
        key_points = getattr(section_plan, "key_points", None)
        if callable(key_points):
            key_points = None
        if key_points:
            points_str = "\n".join(f"- {p}" for p in key_points)
            parts.append(f"**Key Points to Cover**:\n{points_str}")
        refs = getattr(section_plan, "references_to_cite", None)
        if callable(refs):
            refs = None
        if refs:
            parts.append(f"**References to Cite**: {', '.join(refs)}")
        guidance = getattr(section_plan, "writing_guidance", "")
        if guidance:
            parts.append(f"**Writing Guidance**: {guidance}")
        return "\n".join(parts) if parts else ""

    n = len(paragraphs)
    total_sentences = sum(getattr(p, "approx_sentences", 5) for p in paragraphs)

    lines = [f"Write this section with **{n} paragraphs** (~{total_sentences} sentences total):\n"]

    for i, para in enumerate(paragraphs, 1):
        role = getattr(para, "role", "evidence")
        sents = getattr(para, "approx_sentences", 5)
        kp = getattr(para, "key_point", "")
        supporting = getattr(para, "supporting_points", [])
        refs = getattr(para, "references_to_cite", [])
        fig_refs = getattr(para, "figures_to_reference", [])
        tbl_refs = getattr(para, "tables_to_reference", [])

        lines.append(f"**Paragraph {i}** (role: {role}, ~{sents} sentences):")
        if kp:
            lines.append(f"  - Key point: {kp}")
        for sp in supporting:
            lines.append(f"  - Supporting: {sp}")
        if refs:
            lines.append(f"  - Cite: {', '.join(refs)}")
        if fig_refs:
            lines.append(f"  - Reference figures: {', '.join(fig_refs)}")
        if tbl_refs:
            lines.append(f"  - Reference tables: {', '.join(tbl_refs)}")
        lines.append("")

    guidance = getattr(section_plan, "writing_guidance", "")
    if guidance:
        lines.append(f"**Writing Guidance**: {guidance}")

    return "\n".join(lines)


def _format_figure_placement_guidance(section_plan: Any, figures: List[Any]) -> str:
    """
    Format figure placement guidance using FigurePlacement semantics.

    - **Args**:
        - `section_plan`: SectionPlan with figures (FigurePlacement list)
        - `figures`: Available FigureSpec objects

    - **Returns**:
        - `str`: Formatted figure guidance for the prompt
    """
    placements = getattr(section_plan, "figures", None)
    if not placements:
        return ""

    figure_map = {}
    for fig in (figures or []):
        fig_id = fig.id if hasattr(fig, "id") else fig.get("id", "")
        if fig_id:
            figure_map[fig_id] = fig

    parts = ["\n## Figures to DEFINE in this section"]
    parts.append("**CREATE the complete figure environment for each figure below.**\n")

    for fp in placements:
        fig_id = fp.figure_id
        fig = figure_map.get(fig_id)
        if not fig:
            continue

        caption = fig.caption if hasattr(fig, "caption") else fig.get("caption", "")
        desc = fig.description if hasattr(fig, "description") else fig.get("description", "")
        file_path = fig.file_path if hasattr(fig, "file_path") else fig.get("file_path", "")
        wide = fp.is_wide

        filename = os.path.basename(file_path) if file_path else f"{fig_id.replace('fig:', '')}.pdf"
        env_name = "figure*" if wide else "figure"
        width = "\\\\textwidth" if wide else "0.9\\\\linewidth"

        parts.append(f"- **{fig_id}**: {caption}")
        if desc:
            parts.append(f"  Description: {desc}")
        if fp.message:
            parts.append(f"  Message: {fp.message}")
        if fp.caption_guidance:
            parts.append(f"  Caption guidance: {fp.caption_guidance}")
        if wide:
            parts.append(f"  **Note: WIDE figure - use {env_name} to span both columns.**")
        parts.append(f"  Position: {fp.position_hint} in the section")
        parts.append(f"  **Required LaTeX:**")
        parts.append(f"  ```latex")
        parts.append(f"  \\\\begin{{{env_name}}}[t]")
        parts.append(f"  \\\\centering")
        parts.append(f"  \\\\includegraphics[width={width}]{{figures/{filename}}}")
        parts.append(f"  \\\\caption{{{caption}}}\\\\label{{{fig_id}}}")
        parts.append(f"  \\\\end{{{env_name}}}")
        parts.append(f"  ```\n")

    return "\n".join(parts)


def _format_table_placement_guidance(
    section_plan: Any,
    tables: List[Any],
    converted_tables: Optional[Dict[str, str]] = None,
) -> str:
    """Format table placement guidance using TablePlacement semantics."""
    placements = getattr(section_plan, "tables", None)
    if not placements:
        return ""

    table_map = {}
    for tbl in (tables or []):
        tbl_id = tbl.id if hasattr(tbl, "id") else tbl.get("id", "")
        if tbl_id:
            table_map[tbl_id] = tbl

    _converted = converted_tables or {}
    parts = ["\n## Tables to DEFINE in this section"]
    parts.append("**Include the complete table environment for each table below.**\n")

    for tp in placements:
        tbl_id = tp.table_id
        tbl = table_map.get(tbl_id)
        if not tbl:
            continue

        caption = tbl.caption if hasattr(tbl, "caption") else tbl.get("caption", "")
        desc = tbl.description if hasattr(tbl, "description") else tbl.get("description", "")
        wide = tp.is_wide
        env_name = "table*" if wide else "table"

        parts.append(f"- **{tbl_id}**: {caption}")
        if desc:
            parts.append(f"  Description: {desc}")
        if tp.message:
            parts.append(f"  Message: {tp.message}")
        if wide:
            parts.append(f"  **Note: WIDE table - use {env_name} to span both columns.**")
        parts.append(f"  Position: {tp.position_hint} in the section")

        if tbl_id in _converted:
            parts.append(f"  **Required LaTeX (include this exact table):**")
            parts.append(f"  ```latex")
            parts.append(f"  {_converted[tbl_id]}")
            parts.append(f"  ```\n")
        else:
            content = tbl.content if hasattr(tbl, "content") else tbl.get("content", "")
            if content:
                parts.append(f"  Data:\n  {content[:500]}")
            parts.append(f"  **Required: Create \\\\begin{{{env_name}}}...\\\\end{{{env_name}}} with \\\\label{{{tbl_id}}}**\n")

    return "\n".join(parts)


# =============================================================================
# Prompt Compilation Functions
# =============================================================================

def compile_section_prompt(
    section_type: str,
    thesis: str = "",
    content_points: List[str] = None,
    references: List[Any] = None,
    figures: List[Any] = None,
    tables: List[Any] = None,
    word_limit: Optional[int] = None,
    style_guide: Optional[str] = None,
    intro_context: Optional[str] = None,
    active_skills: Optional[List["WritingSkill"]] = None,
) -> str:
    """
    Compile a prompt for section generation (generic fallback).

    - **Args**:
        - `section_type` (str): Type of section
        - `thesis` (str): Core thesis/theme
        - `content_points` (List[str]): Key points to express
        - `references` (List[Any]): Available references
        - `figures` (List[Any]): Available figures
        - `tables` (List[Any]): Available tables
        - `word_limit` (Optional[int]): Word limit
        - `style_guide` (Optional[str]): Target venue style
        - `intro_context` (Optional[str]): Introduction content for context
        - `active_skills` (Optional[List[WritingSkill]]): Active writing skills

    - **Returns**:
        - `str`: Compiled prompt string for LLM
    """
    content_points = content_points or []
    references = references or []
    figures = figures or []
    tables = tables or []

    base_prompt = SECTION_PROMPTS.get(section_type, SECTION_PROMPTS.get("method", ""))
    prompt_parts = [base_prompt]

    if thesis:
        prompt_parts.append(f"\n## Core Theme\n{thesis}")

    if content_points:
        points_str = "\n".join(f"- {p}" for p in content_points)
        prompt_parts.append(f"\n## Key Points to Address\n{points_str}")

    if intro_context and section_type not in ["introduction", "abstract"]:
        context = intro_context[:1500] + "..." if len(intro_context) > 1500 else intro_context
        prompt_parts.append(f"\n## Paper Introduction (for context)\n{context}")

    if references:
        refs_info = []
        for ref in references[:20]:
            if hasattr(ref, "ref_id"):
                ref_str = f"- [{ref.ref_id}]"
                if hasattr(ref, "title") and ref.title:
                    ref_str += f": {ref.title}"
                if hasattr(ref, "authors") and ref.authors:
                    ref_str += f" ({ref.authors})"
                refs_info.append(ref_str)
            elif isinstance(ref, dict):
                ref_id = ref.get("ref_id", ref.get("id", "unknown"))
                ref_str = f"- [{ref_id}]"
                if ref.get("title"):
                    ref_str += f": {ref.get('title')}"
                refs_info.append(ref_str)
        if refs_info:
            prompt_parts.append(f"\n## Available References\n" + "\n".join(refs_info))

    if figures:
        figs_info = []
        for fig in figures:
            if hasattr(fig, "figure_id"):
                figs_info.append(f"- {fig.figure_id}")
            elif hasattr(fig, "id"):
                fig_str = f"- {fig.id}"
                if hasattr(fig, "caption") and fig.caption:
                    fig_str += f": {fig.caption}"
                figs_info.append(fig_str)
            elif isinstance(fig, dict):
                fig_id = fig.get("figure_id", fig.get("id", "unknown"))
                figs_info.append(f"- {fig_id}")
        if figs_info:
            prompt_parts.append(f"\n## Available Figures\n" + "\n".join(figs_info))

    if tables:
        tables_info = []
        for tbl in tables:
            if hasattr(tbl, "table_id"):
                tables_info.append(f"- {tbl.table_id}")
            elif hasattr(tbl, "id"):
                tbl_str = f"- {tbl.id}"
                if hasattr(tbl, "caption") and tbl.caption:
                    tbl_str += f": {tbl.caption}"
                tables_info.append(tbl_str)
            elif isinstance(tbl, dict):
                tbl_id = tbl.get("table_id", tbl.get("id", "unknown"))
                tables_info.append(f"- {tbl_id}")
        if tables_info:
            prompt_parts.append(f"\n## Available Tables\n" + "\n".join(tables_info))

    constraints = []
    if style_guide:
        constraints.append(f"- Style guide: {style_guide}")
    if constraints:
        prompt_parts.append(f"\n## Constraints\n" + "\n".join(constraints))

    _inject_skill_constraints(prompt_parts, active_skills, section_type)

    prompt_parts.append("""
## Output Instructions
- Generate LaTeX content for the section body only
- Do NOT include \\section{} command - just the content
- Use \\cite{key} for citations
- Use \\ref{fig:id} for figure references
- Use \\ref{tab:id} for table references
- Write in academic English with clear, precise language
""")

    return "\n".join(prompt_parts)


def compile_introduction_prompt(
    paper_title: str,
    idea_hypothesis: str,
    method_summary: str,
    data_summary: str,
    experiments_summary: str,
    references: List[Any] = None,
    style_guide: Optional[str] = None,
    section_plan: Any = None,
    figures: List[Any] = None,
    tables: List[Any] = None,
    active_skills: Optional[List["WritingSkill"]] = None,
) -> str:
    """
    Compile prompt for Introduction generation (Phase 1 - Leader section).

    - **Args**:
        - `section_plan`: SectionPlan with paragraph-level structure
        - `figures`: Available FigureSpec list
        - `tables`: Available TableSpec list
    """
    references = references or []
    figures = figures or []
    tables = tables or []

    prompt = f"""You are writing the Introduction section for a research paper titled: "{paper_title}"

## Role of Introduction
The Introduction is the LEADER section that:
1. Establishes the research context and motivation
2. Identifies the problem or gap being addressed
3. States the key contributions (typically 3-4 bullet points)
4. Outlines the paper structure

## Research Content

### Idea/Hypothesis
{idea_hypothesis}

### Method Overview
{method_summary}

### Data/Validation
{data_summary}

### Experiments Overview
{experiments_summary}
"""

    # Paragraph-level planning guidance
    if section_plan:
        guidance = _format_paragraph_guidance(section_plan)
        if guidance:
            prompt += f"\n## Writing Structure\n{guidance}\n"

    # References with citation rules
    if references:
        refs_info = []
        valid_keys = []
        for ref in references[:15]:
            if isinstance(ref, dict):
                ref_id = ref.get("ref_id", ref.get("id", ""))
                title = ref.get("title", "")
                if ref_id:
                    valid_keys.append(ref_id)
                    refs_info.append(
                        f"- \\cite{{{ref_id}}}: {title[:80]}" if title
                        else f"- \\cite{{{ref_id}}}"
                    )
        if refs_info:
            prompt += f"\n### CRITICAL: Citation Rules\n"
            prompt += f"**ONLY use these citation keys. DO NOT invent or hallucinate citations.**\n"
            prompt += f"**Valid keys**: {', '.join(valid_keys)}\n\n"
            prompt += "Available references:\n" + "\n".join(refs_info)
            prompt += "\n\n**WARNING**: Any citation not in the above list will be automatically removed.\n"
            assigned_refs = getattr(section_plan, "assigned_refs", []) if section_plan else []
            if assigned_refs:
                prompt += (
                    "\n\n**Coverage priority for this section**:\n"
                    "Prioritize integrating these assigned citation keys in this section where relevant:\n"
                    + ", ".join(assigned_refs[:8])
                    + "\nDo not force unrelated citations; integrate naturally with matching claims.\n"
                )

    # Figure placement guidance
    if section_plan:
        fig_guidance = _format_figure_placement_guidance(section_plan, figures)
        if fig_guidance:
            prompt += fig_guidance

        # Cross-section figure references
        figs_to_ref = getattr(section_plan, "figures_to_reference", [])
        if figs_to_ref:
            prompt += f"\n## Figures to REFERENCE (already defined elsewhere)\n"
            prompt += "**DO NOT create \\\\begin{{figure}} - just reference with Figure~\\\\ref{{fig:id}}.**\n"
            for fig_id in figs_to_ref:
                prompt += f"- {fig_id}\n"
    elif figures:
        figs_info = []
        for fig in figures:
            fig_id = fig.id if hasattr(fig, "id") else fig.get("id", "")
            caption = fig.caption if hasattr(fig, "caption") else fig.get("caption", "")
            if fig_id:
                figs_info.append(f"- \\ref{{{fig_id}}}: {caption}")
        if figs_info:
            prompt += f"\n### Available Figures\n" + "\n".join(figs_info)

    # Table guidance
    if section_plan:
        tbl_guidance = _format_table_placement_guidance(section_plan, tables)
        if tbl_guidance:
            prompt += tbl_guidance
    elif tables:
        tables_info = []
        for tbl in tables:
            tbl_id = tbl.id if hasattr(tbl, "id") else tbl.get("id", "")
            caption = tbl.caption if hasattr(tbl, "caption") else tbl.get("caption", "")
            if tbl_id:
                tables_info.append(f"- \\ref{{{tbl_id}}}: {caption}")
        if tables_info:
            prompt += f"\n### Available Tables\n" + "\n".join(tables_info)

    if style_guide:
        prompt += f"\n\n## Target Venue: {style_guide}"

    if active_skills:
        intro_parts: list = []
        _inject_skill_constraints(intro_parts, active_skills, "introduction")
        if intro_parts:
            prompt += "\n" + "\n".join(intro_parts)

    prompt += """

## Output Requirements
1. Generate LaTeX content for the Introduction section body
2. Do NOT include \\section{Introduction} - just the content
3. Structure the introduction with clear paragraphs as specified above
4. Use \\cite{key} for citations
5. Use \\ref{fig:id} for figure references and \\ref{tab:id} for table references
6. Write in formal academic English

## Important
At the end, clearly state the contributions using:
\\begin{itemize}
\\item Contribution 1...
\\item Contribution 2...
\\end{itemize}

This helps maintain consistency across the paper.
"""

    return prompt


def compile_body_section_prompt(
    section_type: str,
    metadata_content: str,
    intro_context: str,
    contributions: List[str] = None,
    references: List[Any] = None,
    style_guide: Optional[str] = None,
    section_plan: Any = None,
    figures: List[Any] = None,
    tables: List[Any] = None,
    converted_tables: Optional[Dict[str, str]] = None,
    active_skills: Optional[List["WritingSkill"]] = None,
    memory_context: Optional[str] = None,
) -> str:
    """
    Compile prompt for Body section generation (Phase 2).

    - **Args**:
        - `section_plan`: SectionPlan with paragraph-level structure and FigurePlacement
        - `figures`: Available FigureSpec list
        - `tables`: Available TableSpec list
        - `converted_tables`: table_id -> LaTeX code mapping
        - `memory_context` (str, optional): Cross-section context from SessionMemory
    """
    contributions = contributions or []
    references = references or []
    figures = figures or []
    tables = tables or []

    base_prompt = SECTION_PROMPTS.get(section_type, "")

    prompt = f"""{base_prompt}

## Section Content Source
{metadata_content}

## Introduction Context (maintain consistency)
{intro_context[:2000]}{"..." if len(intro_context) > 2000 else ""}

## Key Contributions to Support
"""
    for i, contrib in enumerate(contributions, 1):
        prompt += f"{i}. {contrib}\n"

    # Memory-provided cross-section coordination context
    if memory_context:
        prompt += f"\n## Coordination Context (from Session Memory)\n{memory_context}\n"

    # Paragraph-level planning guidance
    if section_plan:
        guidance = _format_paragraph_guidance(section_plan)
        if guidance:
            prompt += f"\n## Writing Structure\n{guidance}\n"

    # References
    if references:
        refs_info = []
        valid_keys = []
        for ref in references[:10]:
            if isinstance(ref, dict):
                ref_id = ref.get("ref_id", ref.get("id", ""))
                title = ref.get("title", "")
                if ref_id:
                    valid_keys.append(ref_id)
                    refs_info.append(
                        f"- \\cite{{{ref_id}}}: {title[:60]}" if title
                        else f"- \\cite{{{ref_id}}}"
                    )
        if refs_info:
            prompt += f"\n## CRITICAL: Citation Rules\n"
            prompt += f"**ONLY use these citation keys. DO NOT invent citations.**\n"
            prompt += f"**Valid keys**: {', '.join(valid_keys)}\n\n"
            prompt += "\n".join(refs_info)
            assigned_refs = getattr(section_plan, "assigned_refs", []) if section_plan else []
            if assigned_refs:
                prompt += (
                    "\n\n## Citation Coverage Priority\n"
                    "To improve reference coverage, prioritize citing these assigned keys in this section when relevant:\n"
                    + ", ".join(assigned_refs[:10])
                    + "\nUse each key only if it supports an actual statement in the text.\n"
                )

    # Figure placement guidance (using FigurePlacement semantics)
    if section_plan:
        fig_guidance = _format_figure_placement_guidance(section_plan, figures)
        if fig_guidance:
            prompt += fig_guidance

        figs_to_ref = getattr(section_plan, "figures_to_reference", [])
        if figs_to_ref:
            prompt += f"\n## Figures to REFERENCE (already defined elsewhere)\n"
            prompt += "**DO NOT create \\\\begin{{figure}} for these - just reference them with Figure~\\\\ref{{fig:id}}.**\n"
            figure_map = {}
            for fig in figures:
                fid = fig.id if hasattr(fig, "id") else fig.get("id", "")
                if fid:
                    figure_map[fid] = fig
            for fig_id in figs_to_ref:
                fig = figure_map.get(fig_id)
                caption = ""
                if fig:
                    caption = fig.caption if hasattr(fig, "caption") else fig.get("caption", "")
                prompt += f"- {fig_id}: {caption} -> use `Figure~\\\\ref{{{fig_id}}}`\n"

        # Table placement guidance
        tbl_guidance = _format_table_placement_guidance(section_plan, tables, converted_tables)
        if tbl_guidance:
            prompt += tbl_guidance

        tbls_to_ref = getattr(section_plan, "tables_to_reference", [])
        if tbls_to_ref:
            prompt += f"\n## Tables to REFERENCE (already defined elsewhere)\n"
            prompt += "**DO NOT create \\\\begin{{table}} for these - just reference them with Table~\\\\ref{{tab:id}}.**\n"
            table_map = {}
            for tbl in tables:
                tid = tbl.id if hasattr(tbl, "id") else tbl.get("id", "")
                if tid:
                    table_map[tid] = tbl
            for tbl_id in tbls_to_ref:
                tbl = table_map.get(tbl_id)
                caption = ""
                if tbl:
                    caption = tbl.caption if hasattr(tbl, "caption") else tbl.get("caption", "")
                prompt += f"- {tbl_id}: {caption} -> use `Table~\\\\ref{{{tbl_id}}}`\n"

    else:
        # Legacy fallback: no section_plan, show all figures/tables as available
        if figures:
            figs_info = []
            for fig in figures:
                fig_id = fig.id if hasattr(fig, "id") else fig.get("id", "")
                caption = fig.caption if hasattr(fig, "caption") else fig.get("caption", "")
                if fig_id:
                    figs_info.append(f"- {fig_id}: {caption}")
            if figs_info:
                prompt += f"\n## Available Figures (reference only with \\\\ref{{}})\n" + "\n".join(figs_info)

        if tables:
            tables_info = []
            for tbl in tables:
                tbl_id = tbl.id if hasattr(tbl, "id") else tbl.get("id", "")
                caption = tbl.caption if hasattr(tbl, "caption") else tbl.get("caption", "")
                if tbl_id:
                    tables_info.append(f"- {tbl_id}: {caption}")
            if tables_info:
                prompt += f"\n## Available Tables (reference only with \\\\ref{{}})\n" + "\n".join(tables_info)

    if style_guide:
        prompt += f"\n\n## Target Venue: {style_guide}"

    if active_skills:
        body_parts: list = []
        _inject_skill_constraints(body_parts, active_skills, section_type)
        if body_parts:
            prompt += "\n" + "\n".join(body_parts)

    prompt += """

## Output Requirements
1. Generate LaTeX content for the section body only
2. Do NOT include \\section{} command
3. Follow the paragraph structure specified above
4. Maintain consistency with the Introduction's framing
5. Support the stated contributions where relevant
6. Use \\cite{key} for citations
7. Use \\ref{fig:id} for figure references and \\ref{tab:id} for table references
8. Use clear academic writing style
"""

    return prompt


def compile_synthesis_prompt(
    section_type: str,
    paper_title: str,
    prior_sections: Dict[str, str],
    key_contributions: List[str] = None,
    word_limit: Optional[int] = None,
    style_guide: Optional[str] = None,
    section_plan: Any = None,
    active_skills: Optional[List["WritingSkill"]] = None,
    memory_context: Optional[str] = None,
) -> str:
    """
    Compile prompt for Synthesis sections (Abstract/Conclusion - Phase 3).

    - **Args**:
        - `section_plan`: SectionPlan with paragraph-level structure
        - `memory_context` (str, optional): Cross-section summary from SessionMemory
    """
    key_contributions = key_contributions or []

    # Extract plan guidance
    plan_guidance = ""
    plan_writing_guidance = ""
    if section_plan:
        plan_guidance = _format_paragraph_guidance(section_plan)
        plan_writing_guidance = getattr(section_plan, "writing_guidance", "")

    if section_type == "abstract":
        prompt = f"""You are writing the Abstract for a research paper titled: "{paper_title}"

## Task
Synthesize a concise abstract (150-250 words) from the following paper sections.

## Introduction
{prior_sections.get('introduction', '')[:1500]}{"..." if len(prior_sections.get('introduction', '')) > 1500 else ""}

## Method (summary)
{prior_sections.get('method', '')[:800]}{"..." if len(prior_sections.get('method', '')) > 800 else ""}

## Key Results
{prior_sections.get('result', prior_sections.get('experiment', ''))[:800]}{"..." if len(prior_sections.get('result', prior_sections.get('experiment', ''))) > 800 else ""}

## Key Contributions
"""
        for contrib in key_contributions:
            prompt += f"- {contrib}\n"

        if plan_guidance:
            prompt += f"\n## Writing Structure (from Planner)\n{plan_guidance}\n"

        prompt += """
## Abstract Structure
1. Problem/Motivation (1-2 sentences)
2. Method/Approach (1-2 sentences)
3. Key Results (1-2 sentences)
4. Conclusions/Impact (1 sentence)

## Output Requirements
- Generate ONLY the abstract text
- Do NOT include \\begin{abstract} or any LaTeX commands
- Do NOT include any citations (\\cite{...}) — abstracts must be self-contained
- Write in third person, present/past tense
- Be specific about results (include numbers if available)
"""
        if plan_writing_guidance:
            prompt += f"\n## Writing Guidance (IMPORTANT - follow strictly)\n{plan_writing_guidance}\n"

    elif section_type == "conclusion":
        prompt = f"""You are writing the Conclusion for a research paper titled: "{paper_title}"

## Task
Write a conclusion that synthesizes the paper's contributions and findings.

## Paper Sections for Reference

### Introduction
{prior_sections.get('introduction', '')[:1000]}...

### Method
{prior_sections.get('method', '')[:800]}...

### Results
{prior_sections.get('result', prior_sections.get('experiment', ''))[:1000]}...

## Key Contributions
"""
        for contrib in key_contributions:
            prompt += f"- {contrib}\n"

        if plan_guidance:
            prompt += f"\n## Writing Structure (from Planner)\n{plan_guidance}\n"

        prompt += """
## Conclusion Structure
1. Summary of contributions (1 paragraph)
2. Key findings and their significance (1 paragraph)
3. Limitations (brief, 2-3 sentences)
4. Future work (2-3 sentences)

## Output Requirements
- Generate LaTeX content for the Conclusion section body
- Do NOT include \\section{Conclusion}
- Do NOT include any citations (\\cite{...}) — conclusions must stand alone
- Be concise but comprehensive
- End on a forward-looking note
"""
        if plan_writing_guidance:
            prompt += f"\n## Writing Guidance (IMPORTANT - follow strictly)\n{plan_writing_guidance}\n"
    else:
        prompt = f"""Synthesize content for the {section_type} section based on:

{json.dumps(prior_sections, indent=2)[:3000]}

Key contributions: {key_contributions}
"""

    # Memory-provided global context for synthesis
    if memory_context:
        prompt += f"\n## Section Overview (from Session Memory)\n{memory_context}\n"

    if active_skills:
        synth_parts: list = []
        _inject_skill_constraints(synth_parts, active_skills, section_type)
        if synth_parts:
            prompt += "\n" + "\n".join(synth_parts)

    if style_guide:
        prompt += f"\n- Style guide: {style_guide}"

    return prompt


def extract_contributions_from_intro(intro_content: str) -> List[str]:
    """
    Extract contribution statements from Introduction content.
    Looks for itemize environments or numbered contributions.
    """
    contributions = []
    import re

    item_pattern = r"\\item\s*(.+?)(?=\\item|\\end{itemize}|$)"
    itemize_pattern = r"\\begin{itemize}(.*?)\\end{itemize}"
    itemize_matches = re.findall(itemize_pattern, intro_content, re.DOTALL)

    for block in itemize_matches:
        items = re.findall(item_pattern, block, re.DOTALL)
        for item in items:
            clean_item = item.strip()
            clean_item = re.sub(r"\\[a-zA-Z]+{([^}]*)}", r"\1", clean_item)
            clean_item = re.sub(r"\s+", " ", clean_item)
            if clean_item and len(clean_item) > 10:
                contributions.append(clean_item[:200])

    if not contributions:
        contrib_pattern = r"(?:contribution|we propose|we introduce|our approach)\s*[:\-]?\s*(.+?)(?:\.|$)"
        matches = re.findall(contrib_pattern, intro_content.lower(), re.IGNORECASE)
        for match in matches[:5]:
            if len(match) > 10:
                contributions.append(match.strip()[:200])

    return contributions[:5]
