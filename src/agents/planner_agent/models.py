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
# New paragraph-level models
# =========================================================================

class ParagraphPlan(BaseModel):
    """
    Planning details for a single paragraph.
    - **Description**:
        - Replaces flat key_points + target_words with fine-grained guidance
        - Each paragraph has a clear role and estimated length
    """
    key_point: str = ""
    supporting_points: List[str] = Field(default_factory=list)
    approx_sentences: int = 5
    role: str = "evidence"
    references_to_cite: List[str] = Field(default_factory=list)
    figures_to_reference: List[str] = Field(default_factory=list)
    tables_to_reference: List[str] = Field(default_factory=list)


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
    writing_guidance: str = ""
    order: int = 0

    def get_total_sentences(self) -> int:
        """Sum of approx_sentences across all paragraphs."""
        return sum(p.approx_sentences for p in self.paragraphs)

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

SECTION_RATIOS_BY_TYPE = {
    PaperType.EMPIRICAL: {
        "abstract": 0.025,
        "introduction": 0.12,
        "related_work": 0.10,
        "method": 0.22,
        "experiment": 0.20,
        "result": 0.20,
        "conclusion": 0.035,
    },
    PaperType.THEORETICAL: {
        "abstract": 0.025,
        "introduction": 0.12,
        "related_work": 0.08,
        "method": 0.35,
        "experiment": 0.10,
        "result": 0.15,
        "conclusion": 0.06,
    },
    PaperType.SURVEY: {
        "abstract": 0.025,
        "introduction": 0.10,
        "related_work": 0.50,
        "method": 0.05,
        "experiment": 0.05,
        "result": 0.10,
        "conclusion": 0.08,
    },
}

VENUE_WORD_LIMITS = {
    "ICML": {"pages": 8, "words_per_page": 850},
    "NEURIPS": {"pages": 8, "words_per_page": 700},
    "NIPS": {"pages": 8, "words_per_page": 700},
    "ICLR": {"pages": 8, "words_per_page": 750},
    "ACL": {"pages": 8, "words_per_page": 750},
    "EMNLP": {"pages": 8, "words_per_page": 750},
    "AAAI": {"pages": 7, "words_per_page": 800},
    "IJCAI": {"pages": 7, "words_per_page": 800},
    "CVPR": {"pages": 8, "words_per_page": 800},
    "ICCV": {"pages": 8, "words_per_page": 800},
    "DEFAULT": {"pages": 8, "words_per_page": 750},
}

ELEMENT_PAGE_COST = {
    "figure*": 0.4,
    "figure": 0.2,
    "table*": 0.3,
    "table": 0.15,
}


def calculate_total_words(
    target_pages: Optional[int],
    style_guide: Optional[str],
    n_figures: int = 0,
    n_tables: int = 0,
    n_wide_figures: int = 0,
    n_wide_tables: int = 0,
) -> int:
    """
    Calculate total word budget from pages, venue, and non-text elements.
    - **Description**:
        - Estimates the page space consumed by figures and tables
        - Subtracts that from total pages to get effective text pages
        - Ensures at least 40% of pages are reserved for text

    - **Args**:
        - `target_pages` (Optional[int]): Target page count
        - `style_guide` (Optional[str]): Venue name
        - `n_figures` (int): Total number of figures
        - `n_tables` (int): Total number of tables
        - `n_wide_figures` (int): Number of wide figures
        - `n_wide_tables` (int): Number of wide tables

    - **Returns**:
        - `int`: Effective word budget for text content
    """
    venue_key = (style_guide or "DEFAULT").upper().split()[0]
    config = VENUE_WORD_LIMITS.get(venue_key, VENUE_WORD_LIMITS["DEFAULT"])
    pages = target_pages or config["pages"]
    words_per_page = config["words_per_page"]

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
    return int(text_pages * words_per_page)


def estimate_target_paragraphs(total_words: int) -> int:
    """
    Estimate total paragraph count from word budget.
    - **Args**:
        - `total_words` (int): Word budget

    - **Returns**:
        - `int`: Estimated paragraph count (avg ~100 words/paragraph)
    """
    return max(1, total_words // 100)
