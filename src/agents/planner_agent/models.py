"""
Planner Agent Models
- **Description**:
    - Defines data models for paper planning
    - PaperPlan: Complete planning output
    - SectionPlan: Per-section planning details
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum


class PaperType(str, Enum):
    """Type of academic paper"""
    EMPIRICAL = "empirical"       # Experiments and results
    THEORETICAL = "theoretical"   # Proofs and analysis
    SURVEY = "survey"             # Literature review
    POSITION = "position"         # Opinion/vision paper
    SYSTEM = "system"             # System description
    BENCHMARK = "benchmark"       # Dataset/benchmark paper


class NarrativeStyle(str, Enum):
    """Writing style for the paper"""
    TECHNICAL = "technical"       # Dense, precise
    TUTORIAL = "tutorial"         # Explanatory, accessible
    CONCISE = "concise"           # Brief, to the point
    COMPREHENSIVE = "comprehensive"  # Thorough coverage


class SectionPlan(BaseModel):
    """
    Planning details for a single section
    - **Description**:
        - Contains all planning decisions for one section
        - Guides the Writer during generation
        - Serves as reference for Reviewer
        
    - **Fields**:
        - `section_type` (str): Section identifier (introduction, method, etc.)
        - `section_title` (str): Display title for the section
        - `target_words` (int): Word budget for this section
        - `key_points` (List[str]): Main points to cover
        - `content_sources` (List[str]): Which metadata fields to draw from
        - `references_to_cite` (List[str]): BibTeX keys to cite in this section
        - `figures_to_use` (List[str]): Figure IDs to include in this section
        - `tables_to_use` (List[str]): Table IDs to include in this section
        - `depends_on` (List[str]): Prior sections needed for context
        - `writing_guidance` (str): Specific instructions for the writer
        - `order` (int): Position in the paper
    """
    section_type: str
    section_title: str = ""
    target_words: int = 500
    key_points: List[str] = Field(default_factory=list)
    content_sources: List[str] = Field(default_factory=list)
    references_to_cite: List[str] = Field(default_factory=list)
    # Figures: distinguish between defining and referencing
    figures_to_use: List[str] = Field(default_factory=list)  # Figure IDs (deprecated, for backward compat)
    figures_to_define: List[str] = Field(default_factory=list)  # Figures to CREATE with \begin{figure} in this section
    figures_to_reference: List[str] = Field(default_factory=list)  # Figures to REFERENCE with \ref only
    # Tables: distinguish between defining and referencing
    tables_to_use: List[str] = Field(default_factory=list)   # Table IDs (deprecated, for backward compat)
    tables_to_define: List[str] = Field(default_factory=list)  # Tables to CREATE with \begin{table} in this section
    tables_to_reference: List[str] = Field(default_factory=list)  # Tables to REFERENCE with \ref only
    depends_on: List[str] = Field(default_factory=list)
    writing_guidance: str = ""
    order: int = 0
    
    def get_word_tolerance(self, percent: float = 0.15) -> tuple:
        """Get min/max word counts with tolerance"""
        min_words = int(self.target_words * (1 - percent))
        max_words = int(self.target_words * (1 + percent))
        return min_words, max_words


class PaperPlan(BaseModel):
    """
    Complete paper planning output
    - **Description**:
        - Contains all planning decisions for the entire paper
        - Guides all phases of paper generation
        - Serves as reference for review
        
    - **Fields**:
        - `title` (str): Paper title
        - `paper_type` (PaperType): Type of paper
        - `total_target_words` (int): Total word budget
        - `sections` (List[SectionPlan]): Ordered list of section plans
        - `contributions` (List[str]): Unified contribution statements
        - `narrative_style` (NarrativeStyle): Writing style
        - `terminology` (Dict[str, str]): Key terms and definitions
        - `structure_rationale` (str): Why this structure was chosen
        - `abstract_focus` (str): What abstract should emphasize
    """
    title: str = ""
    paper_type: PaperType = PaperType.EMPIRICAL
    total_target_words: int = 6000
    sections: List[SectionPlan] = Field(default_factory=list)
    contributions: List[str] = Field(default_factory=list)
    narrative_style: NarrativeStyle = NarrativeStyle.TECHNICAL
    terminology: Dict[str, str] = Field(default_factory=dict)
    structure_rationale: str = ""
    abstract_focus: str = ""
    
    # Auto-detected wide figures/tables (for double-column spanning)
    wide_figures: List[str] = Field(default_factory=list)  # Figure IDs needing figure*
    wide_tables: List[str] = Field(default_factory=list)   # Table IDs needing table*
    
    def get_section(self, section_type: str) -> Optional[SectionPlan]:
        """Get section plan by type"""
        for section in self.sections:
            if section.section_type == section_type:
                return section
        return None
    
    def get_section_types(self) -> List[str]:
        """Get ordered list of section types"""
        return [s.section_type for s in self.sections]
    
    def get_body_sections(self) -> List[SectionPlan]:
        """Get non-abstract, non-conclusion sections (the parallel-writable body)"""
        excluded = {"abstract", "conclusion"}
        return [s for s in self.sections if s.section_type not in excluded]

    def get_body_section_types(self) -> List[str]:
        """Get ordered list of body section type strings (no abstract/conclusion)"""
        return [s.section_type for s in self.get_body_sections()]

    def get_compile_section_order(self) -> List[str]:
        """
        Get section order for LaTeX compilation.
        - **Description**:
            - Returns the ordered list of section types excluding abstract
              (abstract is handled separately in the template).
            - Includes all body sections and conclusion.
        """
        return [
            s.section_type for s in self.sections
            if s.section_type != "abstract"
        ]

    def get_section_titles(self) -> Dict[str, str]:
        """Get a mapping from section_type -> display title"""
        return {s.section_type: s.section_title for s in self.sections}
    
    def validate_word_budget(self) -> bool:
        """Check if section budgets sum to total"""
        section_total = sum(s.target_words for s in self.sections)
        tolerance = self.total_target_words * 0.1
        return abs(section_total - self.total_target_words) <= tolerance


class FigureInfo(BaseModel):
    """Simplified figure info for planning"""
    id: str
    caption: str
    description: str = ""
    section: str = ""  # User-suggested section
    wide: bool = False  # If True, use figure* for double-column spanning


class TableInfo(BaseModel):
    """Simplified table info for planning"""
    id: str
    caption: str
    description: str = ""
    section: str = ""  # User-suggested section
    wide: bool = False  # If True, use table* for double-column spanning


class PlanRequest(BaseModel):
    """
    Request to create a paper plan
    - **Description**:
        - Input for the /agent/planner/plan endpoint
    """
    title: str = "Untitled Paper"
    idea_hypothesis: str
    method: str
    data: str
    experiments: str
    references: List[str] = Field(default_factory=list)
    figures: List[FigureInfo] = Field(default_factory=list)  # Available figures
    tables: List[TableInfo] = Field(default_factory=list)    # Available tables
    target_pages: Optional[int] = None
    style_guide: Optional[str] = None  # e.g., "ICML", "NeurIPS"


class PlanResult(BaseModel):
    """
    Result of paper planning
    - **Description**:
        - Output from the planner agent
    """
    status: str  # 'ok', 'error'
    plan: Optional[PaperPlan] = None
    error: Optional[str] = None


# Default section order for empirical papers
DEFAULT_EMPIRICAL_SECTIONS = [
    "abstract",
    "introduction",
    "related_work",
    "method",
    "experiment",
    "result",
    "conclusion",
]

# Word allocation ratios by paper type
SECTION_RATIOS_BY_TYPE = {
    PaperType.EMPIRICAL: {
        "abstract": 0.025,
        "introduction": 0.12,
        "related_work": 0.10,
        "method": 0.22,
        "experiment": 0.20,  # Increased from 0.18
        "result": 0.20,      # Increased from 0.18
        "conclusion": 0.035, # Reduced from 0.06 (~225 words for 8-page paper)
    },
    PaperType.THEORETICAL: {
        "abstract": 0.025,
        "introduction": 0.12,
        "related_work": 0.08,
        "method": 0.35,  # More space for proofs
        "experiment": 0.10,
        "result": 0.15,
        "conclusion": 0.06,
    },
    PaperType.SURVEY: {
        "abstract": 0.025,
        "introduction": 0.10,
        "related_work": 0.50,  # Main content
        "method": 0.05,
        "experiment": 0.05,
        "result": 0.10,
        "conclusion": 0.08,
    },
}

# Venue-specific configurations
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


# Estimated page cost for non-text elements (consistent with MetaDataAgent)
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
        - `style_guide` (Optional[str]): Venue name (ICML, NeurIPS, etc.)
        - `n_figures` (int): Total number of figures
        - `n_tables` (int): Total number of tables
        - `n_wide_figures` (int): Number of wide (figure*) figures
        - `n_wide_tables` (int): Number of wide (table*) tables

    - **Returns**:
        - `int`: Effective word budget for text content
    """
    venue_key = (style_guide or "DEFAULT").upper().split()[0]
    config = VENUE_WORD_LIMITS.get(venue_key, VENUE_WORD_LIMITS["DEFAULT"])
    pages = target_pages or config["pages"]
    words_per_page = config["words_per_page"]

    # Estimate pages consumed by non-text elements
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

    # Effective text pages (at least 40% of total)
    text_pages = max(pages - non_text_pages, pages * 0.4)

    return int(text_pages * words_per_page)
