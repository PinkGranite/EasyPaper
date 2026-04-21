"""
MetaData Agent Models
- **Description**:
    - Defines input/output models for MetaData-based paper generation
    - PaperMetaData: User's simplified input (5 strings + references)
    - PaperGenerationResult: Complete generation result
"""
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any
from enum import Enum


class OutputFormat(str, Enum):
    """Output format options"""
    LATEX = "latex"
    PDF = "pdf"


class CodeRepositoryType(str, Enum):
    """Code repository source type."""
    LOCAL_DIR = "local_dir"
    GIT_REPO = "git_repo"


class CodeRepoOnError(str, Enum):
    """Error behavior when code repository ingestion fails."""
    FALLBACK = "fallback"
    STRICT = "strict"


class CodeRepositorySpec(BaseModel):
    """
    Code repository input specification.
    - **Description**:
        - Defines where to load project code/docs from for writing support.
        - Supports local directory or remote git repository.

    - **Args**:
        - `type` (CodeRepositoryType): Source type (`local_dir` or `git_repo`)
        - `path` (str, optional): Local directory path (required for `local_dir`)
        - `url` (str, optional): Git URL (required for `git_repo`)
        - `ref` (str, optional): Git branch/tag/commit (default: "main")
        - `subdir` (str, optional): Sub-directory inside repository to scope scanning
        - `include_globs` (List[str], optional): Include patterns
        - `exclude_globs` (List[str], optional): Exclude patterns
        - `max_files` (int): Maximum files to ingest
        - `max_total_bytes` (int): Maximum total bytes to ingest
        - `on_error` (CodeRepoOnError): Error policy (`fallback` or `strict`)
    """
    type: CodeRepositoryType
    path: Optional[str] = None
    url: Optional[str] = None
    ref: Optional[str] = "main"
    subdir: Optional[str] = None
    include_globs: List[str] = Field(default_factory=list)
    exclude_globs: List[str] = Field(default_factory=list)
    max_files: int = 5000
    max_total_bytes: int = 200_000_000
    on_error: CodeRepoOnError = CodeRepoOnError.FALLBACK

    @model_validator(mode="after")
    def validate_source_fields(self) -> "CodeRepositorySpec":
        """
        Validate source-specific required fields.
        - **Description**:
            - Ensures `path` is provided for local source.
            - Ensures `url` is provided for git source.

        - **Returns**:
            - `CodeRepositorySpec`: Validated spec
        """
        if self.type == CodeRepositoryType.LOCAL_DIR and not self.path:
            raise ValueError("code_repository.path is required when type='local_dir'")
        if self.type == CodeRepositoryType.GIT_REPO and not self.url:
            raise ValueError("code_repository.url is required when type='git_repo'")
        return self


class FigureSpec(BaseModel):
    """
    Figure specification for paper generation
    - **Description**:
        - Defines a figure to be included in the paper
        - All fields except id and caption are optional
        
    - **Args**:
        - `id` (str): LaTeX label (e.g., "fig:architecture")
        - `caption` (str): Figure caption
        - `description` (str): Helps Writer understand/reference the figure
        - `section` (str): Suggested section placement
        - `file_path` (str, optional): Path to figure file (PDF, PNG, JPEG)
        - `wide` (bool): If True, use figure* for double-column spanning
        - `auto_generate` (bool): Mark for future auto-generation
        - `generation_prompt` (str, optional): Dreamer idea/prompt. Explicit value wins.
        - `style` (str, optional): Dreamer style or venue hint. Explicit value wins.
        - `target_type` (str, optional): Dreamer target figure type. Explicit value wins.
    """
    id: str                              # LaTeX label: "fig:architecture"
    caption: str                         # Figure caption
    description: str = ""                # Helps Writer understand/reference
    section: str = ""                    # Suggested section placement
    
    # Source
    file_path: Optional[str] = None      # "figures/model.pdf"
    
    # Layout
    wide: bool = False                   # If True, use figure* for double-column spanning
    
    # Future: auto-generation
    auto_generate: bool = False          # Mark for future auto-generation
    generation_prompt: Optional[str] = None  # Dreamer idea/prompt; explicit value wins
    style: Optional[str] = None              # Dreamer style/venue; explicit value wins
    target_type: Optional[str] = None        # Dreamer target type; explicit value wins


class TableSpec(BaseModel):
    """
    Table specification for paper generation
    - **Description**:
        - Defines a table to be included in the paper
        - Content can be provided via file_path or inline content
        - Any readable format (CSV, Markdown, plain text) is converted to LaTeX
        
    - **Args**:
        - `id` (str): LaTeX label (e.g., "tab:results")
        - `caption` (str): Table caption
        - `description` (str): Helps Writer understand/reference the table
        - `section` (str): Suggested section placement
        - `file_path` (str, optional): Path to data file (CSV, MD, TXT)
        - `content` (str, optional): Inline data in any readable format
        - `wide` (bool): If True, use table* for double-column spanning
        - `auto_generate` (bool): Mark for future auto-generation
        - `data_source` (str, optional): Metadata field to extract from (future)
    """
    id: str                              # LaTeX label: "tab:results"
    caption: str                         # Table caption
    description: str = ""                # Helps Writer understand/reference
    section: str = ""                    # Suggested section placement
    
    # Source (choose one, or auto_generate)
    file_path: Optional[str] = None      # "data/results.csv" (CSV, MD, TXT)
    content: Optional[str] = None        # Inline data (any readable format)
    
    # Layout
    wide: bool = False                   # If True, use table* for double-column spanning
    
    # Future: auto-generation
    auto_generate: bool = False          # Mark for future auto-generation
    data_source: Optional[str] = None    # Metadata field to extract from


class PaperMetaData(BaseModel):
    """
    User's paper metadata - simplified input
    - **Description**:
        - Minimal input for paper generation
        - 5 natural language fields + BibTeX references
        - Optional figures and tables

    - **Args**:
        - `title` (str): Paper title
        - `idea_hypothesis` (str): Research idea or hypothesis
        - `method` (str): Method description
        - `data` (str): Data or validation method description
        - `experiments` (str): Experiment design, results, findings
        - `references` (List[str]): BibTeX entries
        - `template_path` (str, optional): Path to .zip template file
        - `style_guide` (str, optional): Writing style guide (e.g., "ICML", "NeurIPS")
        - `target_pages` (int, optional): Target page count (uses venue default if not set)
        - `figures` (List[FigureSpec], optional): Figure specifications
        - `tables` (List[TableSpec], optional): Table specifications
        - `code_repository` (CodeRepositorySpec, optional): External project code/docs source
        - `export_prompt_traces` (bool): Whether to export prompt/evidence traces
        - `graph_structure` (CanvasGraphStructure, optional): User's canvas graph for DAG
    """
    title: str = "Untitled Paper"
    idea_hypothesis: str
    method: str
    data: str
    experiments: str
    references: List[str] = Field(default_factory=list)
    template_path: Optional[str] = None  # Path to .zip template file
    style_guide: Optional[str] = None    # Writing style (can be extracted from template)
    target_pages: Optional[int] = None   # Target page count (overrides venue default)

    # Figures and tables (optional)
    figures: List[FigureSpec] = Field(default_factory=list)  # Optional figures
    tables: List[TableSpec] = Field(default_factory=list)    # Optional tables
    # Root folder for resolving relative figure/table paths in downstream generation
    materials_root: Optional[str] = None
    code_repository: Optional[CodeRepositorySpec] = None
    export_prompt_traces: bool = False

    # Canvas graph structure for DAG construction (optional)
    graph_structure: Optional[Any] = None  # CanvasGraphStructure from commander

    # Exemplar paper (optional, user-provided path to PDF)
    exemplar_paper_path: Optional[str] = None

    def to_document_input(self) -> "DocumentInput":
        """
        Convert paper-specific metadata to the generic DocumentInput interface.
        - **Description**:
            - Maps the five natural-language paper fields into a content_brief dict
            - Preserves figures, tables, references, template, and code_repository
            - Attaches GenerationConstraints from style_guide and target_pages

        - **Returns**:
            - `DocumentInput`: Generic document input
        """
        from ...models.document_spec import DocumentInput, GenerationConstraints

        constraints = GenerationConstraints(
            max_pages=self.target_pages,
            style_guide=self.style_guide,
            output_format="latex",
            citation_format="bibtex",
        )
        return DocumentInput(
            title=self.title,
            content_brief={
                "idea_hypothesis": self.idea_hypothesis,
                "method": self.method,
                "data": self.data,
                "experiments": self.experiments,
            },
            references=list(self.references),
            figures=[f.model_dump() for f in self.figures],
            tables=[t.model_dump() for t in self.tables],
            template_path=self.template_path,
            code_repository=self.code_repository,
            constraints=constraints,
        )


class PaperGenerationRequest(BaseModel):
    """
    Request for paper generation
    - **Description**:
        - Wraps PaperMetaData with generation options
        
    - **Args**:
        - `title` (str): Paper title
        - `idea_hypothesis` (str): Research idea or hypothesis
        - `method` (str): Method description
        - `data` (str): Data description
        - `experiments` (str): Experiments description
        - `references` (List[str]): BibTeX entries
        - `template_path` (str, optional): Path to .zip template file
        - `style_guide` (str, optional): Writing style guide
        - `target_pages` (int, optional): Target page count
        - `compile_pdf` (bool): Whether to compile PDF (default: True if template provided)
        - `figures_source_dir` (str, optional): Directory containing figure files
        - `save_output` (bool): Whether to save output to disk
        - `output_dir` (str, optional): Directory for output files
        - `enable_review` (bool): Whether to enable review loop
        - `max_review_iterations` (int): Maximum review iterations
        - `code_repository` (CodeRepositorySpec, optional): External project code/docs source
        - `export_prompt_traces` (bool): Whether to export prompt/evidence traces
    """
    # Metadata fields (can pass directly)
    title: str = "Untitled Paper"
    idea_hypothesis: str = ""
    method: str = ""
    data: str = ""
    experiments: str = ""
    references: List[str] = Field(default_factory=list)
    
    # Figures and tables (optional)
    figures: List[FigureSpec] = Field(default_factory=list)
    tables: List[TableSpec] = Field(default_factory=list)
    code_repository: Optional[CodeRepositorySpec] = None
    export_prompt_traces: bool = False
    
    # Template and style
    template_path: Optional[str] = None      # Path to .zip template file
    style_guide: Optional[str] = None        # Writing style (ICML, NeurIPS, etc.)
    target_pages: Optional[int] = None       # Target page count
    
    # Compilation options
    compile_pdf: bool = True                 # Compile to PDF if template provided
    figures_source_dir: Optional[str] = None # Directory with figure files (legacy)
    
    # Review options
    enable_review: bool = True               # Enable review loop
    max_review_iterations: int = 3           # Maximum review iterations
    
    # Planning options
    enable_planning: bool = True             # Enable planning phase
    
    # VLM Review options
    enable_vlm_review: bool = False          # Enable VLM-based PDF review (page overflow detection)

    # Exemplar paper options
    enable_exemplar: bool = False            # Enable exemplar (benchmark) paper strategy

    # User feedback options
    enable_user_feedback: bool = False       # Pause at review for user feedback (checkpoint-resume)
    
    # Output options
    save_output: bool = True
    output_dir: Optional[str] = None

    # Storage: when set, artifacts are uploaded directly to OSS under this prefix
    artifacts_prefix: Optional[str] = None
    
    def to_metadata(self) -> PaperMetaData:
        """Convert request to PaperMetaData"""
        return PaperMetaData(
            title=self.title,
            idea_hypothesis=self.idea_hypothesis,
            method=self.method,
            data=self.data,
            experiments=self.experiments,
            references=self.references,
            template_path=self.template_path,
            style_guide=self.style_guide,
            target_pages=self.target_pages,
            figures=self.figures,
            tables=self.tables,
            code_repository=self.code_repository,
            export_prompt_traces=self.export_prompt_traces,
        )


class SectionResult(BaseModel):
    """Result for a single section"""
    section_type: str
    section_title: str = ""
    status: str  # 'ok', 'error'
    latex_content: str = ""
    word_count: int = 0
    error: Optional[str] = None


class PaperGenerationResult(BaseModel):
    """
    Result of paper generation
    - **Description**:
        - Contains generated paper content and metadata
        
    - **Returns**:
        - `status` (str): 'ok', 'partial', 'error'
        - `paper_title` (str): Paper title
        - `sections` (List[SectionResult]): Results for each section
        - `latex_content` (str): Complete assembled LaTeX
        - `output_path` (str, optional): Directory where files are saved
        - `pdf_path` (str, optional): Path to PDF if generated
        - `total_word_count` (int): Total word count
        - `target_word_count` (int, optional): Target word count
        - `review_iterations` (int): Number of review iterations performed
        - `errors` (List[str]): List of errors
        - `usage` (Dict, optional): Token consumption breakdown from UsageTracker
    """
    status: str  # 'ok', 'partial', 'error'
    paper_title: str = ""
    sections: List[SectionResult] = Field(default_factory=list)
    latex_content: str = ""
    output_path: Optional[str] = None
    pdf_path: Optional[str] = None
    total_word_count: int = 0
    target_word_count: Optional[int] = None
    review_iterations: int = 0
    errors: List[str] = Field(default_factory=list)
    usage: Optional[Dict[str, Any]] = None


class CoreRefAnalysisItem(BaseModel):
    """
    Deep analysis of a single core reference.
    - **Description**:
        - Structured interpretation of one user-provided paper (not generic scoring).
    """

    ref_id: str
    title: str
    core_contributions: List[str] = Field(default_factory=list)
    methodology: str = ""
    limitations: List[str] = Field(default_factory=list)
    relationship_to_ours: str = ""
    key_results: List[str] = Field(default_factory=list)
    relevance_score: float = 1.0


class CoreRefAnalysis(BaseModel):
    """
    Cross-cutting analysis of all core references.
    - **Description**:
        - Per-paper items plus synthesis across the user's anchor literature.
    """

    items: List[CoreRefAnalysisItem] = Field(default_factory=list)
    shared_gaps: List[str] = Field(default_factory=list)
    research_lineage: str = ""
    positioning_statement: str = ""


class ResearchContextModel(BaseModel):
    """
    Typed research context for planning and writing.
    - **Description**:
        - Mirrors legacy dict keys for serialization into ``PlanResult.research_context``.
    """

    research_area: str = ""
    summary: str = ""
    key_papers: List[Dict[str, Any]] = Field(default_factory=list)
    research_trends: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    claim_evidence_matrix: List[Dict[str, Any]] = Field(default_factory=list)
    contribution_ranking: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    planning_decision_trace: List[str] = Field(default_factory=list)
    paper_assignments: Dict[str, List[str]] = Field(default_factory=dict)
    core_ref_analysis: Optional[CoreRefAnalysis] = None

    def to_research_context_dict(self) -> Dict[str, Any]:
        """
        Serialize for ``PlanResult.research_context`` and planner consumers.
        - **Returns**:
            - `dict`: JSON-serializable research context.
        """
        d = self.model_dump(mode="json", exclude_none=True)
        return d


class ReviewCheckpoint(BaseModel):
    """
    Serialized generation state at a review feedback pause point.
    - **Description**:
        - Captures all data needed to resume generation after user feedback.
        - Saved to disk when ``enable_user_feedback=True`` and the review loop
          finds issues that need user approval before revision.
    """
    task_id: str
    generated_sections: Dict[str, str]
    sections_results: List[SectionResult]
    review_result: Dict[str, Any]
    metadata: Dict[str, Any]
    paper_plan: Optional[Dict[str, Any]] = None
    ref_pool: Dict[str, Any] = Field(default_factory=dict)
    memory: Dict[str, Any] = Field(default_factory=dict)
    converted_tables: Dict[str, str] = Field(default_factory=dict)
    review_iterations: int = 0
    max_review_iterations: int = 3
    target_word_count: Optional[int] = None
    pdf_path: Optional[str] = None
    template_path: Optional[str] = None
    figures_source_dir: Optional[str] = None
    enable_review: bool = True
    compile_pdf: bool = True
    enable_vlm_review: bool = False
    target_pages: Optional[int] = None
    paper_dir: Optional[str] = None
    last_vlm_result: Optional[Dict[str, Any]] = None
    config_hash: str = ""


class PlanResult(BaseModel):
    """
    Serializable snapshot of the planning phase output.
    - **Description**:
        - Captures all intermediate state produced by ``prepare_plan()``
          so that ``execute_generation()`` can resume from this checkpoint.
        - Designed to be returned to the frontend, optionally modified,
          then sent back to start content generation.

    - **Args**:
        - `paper_plan` (Dict[str, Any]): Serialized PaperPlan (``PaperPlan.model_dump()``).
        - `evidence_dag` (Dict[str, Any], optional): Serialized EvidenceDAG.
        - `research_context` (Dict[str, Any], optional): Research context dict.
        - `code_context` (Dict[str, Any], optional): Code repository context.
        - `code_summary_markdown` (str, optional): Rendered code summary.
        - `ref_pool_snapshot` (Dict[str, Any]): Serialized ReferencePool (``to_dict()``).
        - `converted_tables` (Dict[str, str]): Table ID -> LaTeX mapping.
        - `metadata_input` (Dict[str, Any]): Serialized PaperMetaData.
        - `errors` (List[str]): Non-fatal errors collected during planning.
        - `template_path` (str, optional): Resolved template path.
        - `target_pages` (int, optional): Target page count.
        - `artifacts_prefix` (str): Storage prefix for artifacts.
        - `paper_dir` (str, optional): Output directory path.
        - `plan_review` (Dict[str, Any], optional): Planner review summary.
        - `plan_review_iterations` (List[Dict[str, Any]]): Per-iteration review traces.
    """
    paper_plan: Dict[str, Any]
    evidence_dag: Optional[Dict[str, Any]] = None
    research_context: Optional[Dict[str, Any]] = None
    code_context: Optional[Dict[str, Any]] = None
    code_summary_markdown: Optional[str] = None
    ref_pool_snapshot: Dict[str, Any] = Field(default_factory=dict)
    converted_tables: Dict[str, str] = Field(default_factory=dict)
    metadata_input: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    template_path: Optional[str] = None
    target_pages: Optional[int] = None
    exemplar_analysis: Optional[Dict[str, Any]] = None
    artifacts_prefix: str = ""
    paper_dir: Optional[str] = None
    plan_review: Optional[Dict[str, Any]] = None
    plan_review_iterations: List[Dict[str, Any]] = Field(default_factory=list)


class SectionBlueprint(BaseModel):
    """
    Per-section structure extracted from an exemplar paper.
    - **Description**:
        - Captures the structural pattern of a single section in the exemplar.
    """
    section_type: str
    title: str = ""
    approx_word_count: int = 0
    paragraph_count: int = 0
    subsection_titles: List[str] = Field(default_factory=list)
    role: str = ""


class StyleProfile(BaseModel):
    """
    Writing style features extracted from an exemplar paper.
    - **Description**:
        - Quantifies stylistic characteristics for prompt injection.
    """
    tone: str = "formal"
    citation_density: float = 0.0
    avg_sentence_length: float = 0.0
    hedging_level: str = "moderate"
    transition_patterns: List[str] = Field(default_factory=list)


class ArgumentationPatterns(BaseModel):
    """
    Argumentation and rhetorical patterns from an exemplar paper.
    - **Description**:
        - Describes how the exemplar constructs its narrative arc.
    """
    intro_hook_type: str = ""
    claim_evidence_structure: str = ""
    discussion_closing_strategy: str = ""


class ExemplarAnalysis(BaseModel):
    """
    Structured decomposition of an exemplar (benchmark) paper.
    - **Description**:
        - Top-level container holding all patterns extracted from the exemplar.
        - Used to inject writing guidance into prompt compilation.
    """
    ref_id: str
    title: str
    venue: str
    year: int = 0
    section_blueprint: List[SectionBlueprint] = Field(default_factory=list)
    style_profile: StyleProfile = Field(default_factory=StyleProfile)
    argumentation_patterns: ArgumentationPatterns = Field(default_factory=ArgumentationPatterns)
    paragraph_archetypes: Dict[str, List[str]] = Field(default_factory=dict)


class SectionGenerationRequest(BaseModel):
    """
    Request to generate a single section
    - **Description**:
        - For debugging or incremental generation
        
    - **Args**:
        - `section_type` (str): Type of section to generate
        - `metadata` (PaperMetaData): Paper metadata
        - `intro_context` (str, optional): Introduction content for context
        - `prior_sections` (Dict[str, str], optional): Already generated sections
    """
    section_type: str
    metadata: PaperMetaData
    intro_context: Optional[str] = None
    prior_sections: Optional[Dict[str, str]] = None


class StructuralAction(BaseModel):
    """
    A single structural adjustment action for page-limit control.
    - **Description**:
        - Represents one concrete operation (resize, move, create appendix, etc.)
        - Generated by _plan_overflow_strategy and executed before word-level revisions

    - **Fields**:
        - `action_type` (str): One of 'resize_figure', 'downgrade_wide', 'move_figure',
          'move_table', 'create_appendix'
        - `target_id` (str): LaTeX label of the target element (e.g. "fig:arch")
        - `section` (str): Which section the element currently lives in
        - `params` (Dict): Action-specific parameters
          - resize_figure: {"width": "0.8\\linewidth"}
          - downgrade_wide: {}  (figure* -> figure)
          - move_figure/move_table: {}  (move to appendix)
          - create_appendix: {}
        - `estimated_savings` (float): Estimated page savings from this action
    """
    action_type: str
    target_id: str = ""
    section: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)
    estimated_savings: float = 0.0

    def to_unified_action(self) -> "Action":
        """Convert to the unified Action model from ``src.models``."""
        from ...models.action_space import from_legacy_action
        return from_legacy_action(
            self.action_type,
            target_id=self.target_id,
            section=self.section,
            params=self.params,
            estimated_impact=self.estimated_savings,
        )


class SpaceEstimate(BaseModel):
    """
    Estimated space usage of non-text elements in a section.
    - **Description**:
        - Summarises how many figures/tables a section contains and the estimated page cost

    - **Fields**:
        - `wide_figures` (int): Number of figure* environments
        - `narrow_figures` (int): Number of figure environments
        - `wide_tables` (int): Number of table* environments
        - `narrow_tables` (int): Number of table environments
        - `total_pages` (float): Estimated total pages consumed by these elements
        - `figure_ids` (List[str]): LaTeX labels extracted from \\label{fig:...}
        - `table_ids` (List[str]): LaTeX labels extracted from \\label{tab:...}
    """
    wide_figures: int = 0
    narrow_figures: int = 0
    wide_tables: int = 0
    narrow_tables: int = 0
    total_pages: float = 0.0
    figure_ids: List[str] = Field(default_factory=list)
    table_ids: List[str] = Field(default_factory=list)


# Section source mapping
BODY_SECTION_SOURCES: Dict[str, List[str]] = {
    "related_work": ["references"],
    "method": ["method"],
    "experiment": ["data", "experiments"],
    "result": ["experiments"],
    "discussion": ["experiments"],
}

SYNTHESIS_SECTIONS = ["abstract", "conclusion"]

INTRODUCTION_SOURCES = ["idea_hypothesis", "method", "data", "experiments", "references"]

# Default section order for paper assembly
DEFAULT_SECTION_ORDER = [
    "abstract",
    "introduction",
    "related_work",
    "method",
    "experiment",
    "result",
    "conclusion",
]
