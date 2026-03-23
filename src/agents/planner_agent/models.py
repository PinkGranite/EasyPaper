"""
Planner Agent Models
- **Description**:
    - Defines data models for paper planning
    - ParagraphPlan: Per-paragraph structure and guidance
    - FigurePlacement / TablePlacement: VLM-informed visual element planning
    - SectionPlan: Per-section planning details (paragraph-level)
    - PaperPlan: Complete planning output
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum


class PaperType(str, Enum):
    """Type of academic paper"""
    EMPIRICAL = "empirical"
    THEORETICAL = "theoretical"
    SURVEY = "survey"
    POSITION = "position"
    SYSTEM = "system"
    BENCHMARK = "benchmark"


class NarrativeStyle(str, Enum):
    """Writing style for the paper"""
    TECHNICAL = "technical"
    TUTORIAL = "tutorial"
    CONCISE = "concise"
    COMPREHENSIVE = "comprehensive"


# =========================================================================
# Sentence-level models
# =========================================================================

class SentenceRole(str, Enum):
    """Functional role of a sentence within a paragraph."""
    TOPIC = "topic"
    EVIDENCE = "evidence"
    ANALYSIS = "analysis"
    TRANSITION = "transition"
    CONCLUSION = "conclusion"


class SentencePlan(BaseModel):
    """
    Explicit plan for a single sentence within a paragraph.
    - **Description**:
        - Upgrades the coarse ``approx_sentences`` integer into a list of
          concrete sentence-level instructions.
        - Each sentence carries its own claim/evidence binding and role.
        - The ``template`` field is reserved for template-slot filling
          (Phase 2, Task 2.4).
    """
    sentence_id: str = ""
    claim_id: str = ""
    evidence_ids: List[str] = Field(default_factory=list)
    role: SentenceRole = SentenceRole.EVIDENCE
    approx_words: int = 20
    template: str = ""


# =========================================================================
# Paragraph-level models
# =========================================================================

class ParagraphPlan(BaseModel):
    """
    Planning details for a single paragraph.
    - **Description**:
        - Replaces flat key_points + target_words with fine-grained guidance
        - Each paragraph has a clear role and estimated length
        - claim_id and bound_evidence_ids link to the EvidenceDAG
        - sentence_plans provides explicit per-sentence instructions when the
          EvidenceDAG is available; approx_sentences serves as fallback.
    """
    key_point: str = ""
    supporting_points: List[str] = Field(default_factory=list)
    approx_sentences: int = 5
    role: str = "evidence"
    references_to_cite: List[str] = Field(default_factory=list)
    figures_to_reference: List[str] = Field(default_factory=list)
    tables_to_reference: List[str] = Field(default_factory=list)
    # Evidence DAG bindings (populated by DAGBuilder)
    claim_id: str = ""
    bound_evidence_ids: List[str] = Field(default_factory=list)
    # Explicit sentence plans (populated when DAG is available)
    sentence_plans: List[SentencePlan] = Field(default_factory=list)
    # Template for degraded generation (populated by template-slot filling)
    paragraph_template: Optional[Dict[str, Any]] = None

    @property
    def effective_sentence_count(self) -> int:
        """Return explicit plan length when available, else the estimate."""
        return len(self.sentence_plans) if self.sentence_plans else self.approx_sentences


class FigurePlacement(BaseModel):
    """
    VLM-informed figure placement decision.
    - **Description**:
        - Replaces simple figures_to_define lists
        - Contains semantic analysis from VLM about the figure's role and content
    """
    figure_id: str
    semantic_role: str = ""
    message: str = ""
    is_wide: bool = False
    position_hint: str = "mid"
    caption_guidance: str = ""


class TablePlacement(BaseModel):
    """
    VLM-informed table placement decision.
    - **Description**:
        - Replaces simple tables_to_define lists
        - Contains semantic analysis about the table's role
    """
    table_id: str
    semantic_role: str = ""
    message: str = ""
    is_wide: bool = False
    position_hint: str = "mid"


# =========================================================================
# Section and Paper Plan
# =========================================================================

WORDS_PER_SENTENCE = 20  # rough estimate for word budget calculations


class SectionPlan(BaseModel):
    """
    Planning details for a single section (paragraph-level granularity).
    - **Description**:
        - Contains paragraph-level structure instead of word counts
        - Figures/Tables use placement objects with semantic info
    """
    section_type: str
    section_title: str = ""
    paragraphs: List[ParagraphPlan] = Field(default_factory=list)
    figures: List[FigurePlacement] = Field(default_factory=list)
    tables: List[TablePlacement] = Field(default_factory=list)
    figures_to_reference: List[str] = Field(default_factory=list)
    tables_to_reference: List[str] = Field(default_factory=list)
    content_sources: List[str] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    assigned_refs: List[str] = Field(default_factory=list)
    budget_selected_refs: List[str] = Field(default_factory=list)
    budget_reserve_refs: List[str] = Field(default_factory=list)
    budget_must_use_refs: List[str] = Field(default_factory=list)
    citation_budget: Dict[str, Any] = Field(default_factory=dict)
    # Soft structure signals for writer/reviewer coordination.
    topic_clusters: List[str] = Field(default_factory=list)
    transition_intents: List[str] = Field(default_factory=list)
    sectioning_recommended: bool = False
    code_focus: Dict[str, Any] = Field(default_factory=dict)
    writing_guidance: str = ""
    order: int = 0

    def get_total_sentences(self) -> int:
        """Sum of effective sentence counts across all paragraphs."""
        return sum(p.effective_sentence_count for p in self.paragraphs)

    def get_estimated_words(self) -> int:
        """Rough word estimate from sentence count."""
        return self.get_total_sentences() * WORDS_PER_SENTENCE

    def get_key_points(self) -> List[str]:
        """Collect key_point from each paragraph."""
        return [p.key_point for p in self.paragraphs if p.key_point]

    def get_all_references(self) -> List[str]:
        """Collect unique references across all paragraphs."""
        refs: List[str] = []
        for p in self.paragraphs:
            for r in p.references_to_cite:
                if r not in refs:
                    refs.append(r)
        return refs

    def get_figure_ids_to_define(self) -> List[str]:
        """Figure IDs that should be DEFINED in this section."""
        return [f.figure_id for f in self.figures]

    def get_table_ids_to_define(self) -> List[str]:
        """Table IDs that should be DEFINED in this section."""
        return [t.table_id for t in self.tables]


class PaperPlan(BaseModel):
    """
    Complete paper planning output.
    - **Description**:
        - Contains all planning decisions for the entire paper
        - Guides all phases of paper generation
        - Uses paragraph-level granularity instead of word counts
    """
    title: str = ""
    paper_type: PaperType = PaperType.EMPIRICAL
    sections: List[SectionPlan] = Field(default_factory=list)
    contributions: List[str] = Field(default_factory=list)
    narrative_style: NarrativeStyle = NarrativeStyle.TECHNICAL
    terminology: Dict[str, str] = Field(default_factory=dict)
    structure_rationale: str = ""
    abstract_focus: str = ""
    wide_figures: List[str] = Field(default_factory=list)
    wide_tables: List[str] = Field(default_factory=list)
    citation_strategy: Dict[str, Any] = Field(default_factory=dict)
    # Serialised EvidenceDAG (populated by DAGBuilder, use EvidenceDAG.from_serializable to restore)
    evidence_dag: Optional[Dict[str, Any]] = None

    def get_section(self, section_type: str) -> Optional[SectionPlan]:
        """Get section plan by type."""
        for section in self.sections:
            if section.section_type == section_type:
                return section
        return None

    def get_section_types(self) -> List[str]:
        """Get ordered list of section types."""
        return [s.section_type for s in self.sections]

    def get_body_sections(self) -> List[SectionPlan]:
        """Get non-abstract, non-conclusion sections."""
        excluded = {"abstract", "conclusion"}
        return [s for s in self.sections if s.section_type not in excluded]

    def get_body_section_types(self) -> List[str]:
        """Get ordered list of body section type strings."""
        return [s.section_type for s in self.get_body_sections()]

    def get_compile_section_order(self) -> List[str]:
        """Section order for LaTeX compilation (excludes abstract)."""
        return [
            s.section_type for s in self.sections
            if s.section_type != "abstract"
        ]

    def get_section_titles(self) -> Dict[str, str]:
        """Mapping from section_type -> display title."""
        return {s.section_type: s.section_title for s in self.sections}

    def get_total_sentences(self) -> int:
        """Total sentence estimate across all sections."""
        return sum(s.get_total_sentences() for s in self.sections)

    def get_total_estimated_words(self) -> int:
        """Total word estimate from sentence counts."""
        return self.get_total_sentences() * WORDS_PER_SENTENCE

    def to_document_spec(self) -> "DocumentSpec":
        """
        Convert paper-specific plan to the generic DocumentSpec interface.
        - **Description**:
            - Maps SectionPlan → ContentSection with paragraph dicts
            - Preserves contributions, terminology, evidence_dag, and rationale
            - Enables the generation pipeline to work with any document type

        - **Returns**:
            - `DocumentSpec`: Generic document specification
        """
        from ...models.document_spec import DocumentSpec, ContentSection

        doc_sections = []
        for sp in self.sections:
            para_dicts = [p.model_dump() for p in sp.paragraphs]
            doc_sections.append(ContentSection(
                section_id=sp.section_type,
                title=sp.section_title,
                content_sources=list(sp.content_sources),
                paragraphs=para_dicts,
                depends_on=list(sp.depends_on),
                figures=[f.model_dump() for f in sp.figures],
                tables=[t.model_dump() for t in sp.tables],
                order=sp.order,
            ))
        return DocumentSpec(
            title=self.title,
            document_type="paper",
            sections=doc_sections,
            contributions=list(self.contributions),
            terminology=dict(self.terminology),
            structure_rationale=self.structure_rationale,
            evidence_dag=self.evidence_dag,
        )


# =========================================================================
# Input models
# =========================================================================

class FigureInfo(BaseModel):
    """Simplified figure info for planning."""
    id: str
    caption: str
    description: str = ""
    section: str = ""
    wide: bool = False
    file_path: str = ""


class TableInfo(BaseModel):
    """Simplified table info for planning."""
    id: str
    caption: str
    description: str = ""
    section: str = ""
    wide: bool = False
    file_path: str = ""


class PlanRequest(BaseModel):
    """Request to create a paper plan."""
    title: str = "Untitled Paper"
    idea_hypothesis: str
    method: str
    data: str
    experiments: str
    references: List[str] = Field(default_factory=list)
    research_context: Optional[Dict[str, Any]] = None
    code_context: Optional[Dict[str, Any]] = None
    code_writing_assets: Optional[Dict[str, Any]] = None
    figures: List[FigureInfo] = Field(default_factory=list)
    tables: List[TableInfo] = Field(default_factory=list)
    target_pages: Optional[int] = None
    style_guide: Optional[str] = None


class PlanResult(BaseModel):
    """Result of paper planning."""
    status: str
    plan: Optional[PaperPlan] = None
    error: Optional[str] = None


# =========================================================================
# Constants
# =========================================================================

DEFAULT_EMPIRICAL_SECTIONS = [
    "abstract",
    "introduction",
    "related_work",
    "method",
    "experiment",
    "result",
    "conclusion",
]

WORDS_PER_PAGE_DEFAULT = 600

ELEMENT_PAGE_COST = {
    "figure*": 0.4,
    "figure": 0.2,
    "table*": 0.3,
    "table": 0.15,
}

WORDS_PER_PARAGRAPH = 200


def calculate_total_words(
    target_pages: Optional[int],
    style_guide: Optional[str] = None,
    n_figures: int = 0,
    n_tables: int = 0,
    n_wide_figures: int = 0,
    n_wide_tables: int = 0,
) -> int:
    """
    Estimate total word budget from target pages and non-text element count.

    - **Description**:
      - Uses a single reasonable words-per-page estimate (~600) rather than
        a large venue-specific lookup table, since the user's target_pages
        is the authoritative length signal and the LLM + skills system
        handle venue-specific style.
      - Subtracts estimated page space consumed by figures/tables.

    - **Args**:
      - `target_pages` (Optional[int]): User-specified target page count.
      - `style_guide` (Optional[str]): Venue hint (unused for word calc,
        kept for API compatibility).
      - `n_figures` / `n_tables` / `n_wide_*`: Visual element counts.

    - **Returns**:
      - `int`: Effective word budget for text content.
    """
    pages = target_pages or 10

    n_narrow_figures = max(0, n_figures - n_wide_figures)
    n_narrow_tables = max(0, n_tables - n_wide_tables)
    figure_pages = (
        n_wide_figures * ELEMENT_PAGE_COST["figure*"]
        + n_narrow_figures * ELEMENT_PAGE_COST["figure"]
    )
    table_pages = (
        n_wide_tables * ELEMENT_PAGE_COST["table*"]
        + n_narrow_tables * ELEMENT_PAGE_COST["table"]
    )
    non_text_pages = figure_pages + table_pages
    text_pages = max(pages - non_text_pages, pages * 0.4)
    return int(text_pages * WORDS_PER_PAGE_DEFAULT)


def estimate_target_paragraphs(total_words: int) -> int:
    """
    Estimate total paragraph count from word budget.

    - **Returns**:
      - `int`: Estimated paragraph count (~200 words/paragraph for academic text).
    """
    return max(1, total_words // WORDS_PER_PARAGRAPH)
