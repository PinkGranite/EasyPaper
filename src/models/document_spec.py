"""
Generalized document specification models.
- **Description**:
    - Defines generic abstractions for long-form document generation
    - Paper writing becomes one concrete implementation of these interfaces
    - Enables the same generation pipeline to handle reports, proposals, etc.
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class ContentSection(BaseModel):
    """
    Generic section specification for any long-form document.
    - **Description**:
        - Maps to SectionPlan in the paper-specific layer
        - Carries paragraph-level plans via opaque dicts for generality

    - **Args**:
        - `section_id` (str): Unique section identifier
        - `title` (str): Display title
        - `content_sources` (List[str]): Where to draw content from
        - `paragraphs` (List[Dict]): Paragraph-level plans (opaque dicts)
        - `depends_on` (List[str]): Sections that must be generated first
        - `figures` (List[Any]): Figure placement specs
        - `tables` (List[Any]): Table placement specs
        - `order` (int): Rendering order
    """
    section_id: str
    title: str = ""
    content_sources: List[str] = Field(default_factory=list)
    paragraphs: List[Dict[str, Any]] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    figures: List[Any] = Field(default_factory=list)
    tables: List[Any] = Field(default_factory=list)
    order: int = 0


class DocumentSpec(BaseModel):
    """
    Generic document specification — the universal planning interface.
    - **Description**:
        - Replaces paper-specific PaperPlan for the purpose of defining
          a universal long-form generation interface
        - Paper plans can convert to DocumentSpec via PaperPlan.to_document_spec()
        - Other document types (reports, proposals) can produce DocumentSpec directly

    - **Args**:
        - `title` (str): Document title
        - `document_type` (str): "paper", "report", "proposal", etc.
        - `sections` (List[ContentSection]): Ordered list of sections
        - `contributions` (List[str]): Key contributions or findings
        - `terminology` (Dict[str, str]): Domain-specific term definitions
        - `structure_rationale` (str): Why this structure was chosen
        - `evidence_dag` (Optional[Dict]): Serialized EvidenceDAG if available
    """
    title: str = ""
    document_type: str = "paper"
    sections: List[ContentSection] = Field(default_factory=list)
    contributions: List[str] = Field(default_factory=list)
    terminology: Dict[str, str] = Field(default_factory=dict)
    structure_rationale: str = ""
    evidence_dag: Optional[Dict[str, Any]] = None


class GenerationConstraints(BaseModel):
    """
    Constraint specification for controlled long-form generation.
    - **Description**:
        - Captures output format, citation style, page/word limits, and
          style guide references
        - Decoupled from any specific document type

    - **Args**:
        - `max_pages` (Optional[int]): Page limit
        - `max_words` (Optional[int]): Word limit
        - `style_guide` (Optional[str]): e.g. "ICML", "NeurIPS", "APA"
        - `output_format` (str): "latex" | "markdown" | "html"
        - `citation_format` (str): "bibtex" | "apa" | "inline"
        - `section_word_targets` (Dict[str, int]): Per-section word targets
        - `required_sections` (List[str]): Sections that must appear
        - `forbidden_patterns` (List[str]): Regex patterns to reject
    """
    max_pages: Optional[int] = None
    max_words: Optional[int] = None
    style_guide: Optional[str] = None
    output_format: str = "latex"
    citation_format: str = "bibtex"
    section_word_targets: Dict[str, int] = Field(default_factory=dict)
    required_sections: List[str] = Field(default_factory=list)
    forbidden_patterns: List[str] = Field(default_factory=list)


class DocumentInput(BaseModel):
    """
    Generic input for document generation — replaces PaperMetaData as the
    universal generation interface.
    - **Description**:
        - Provides a document-type-agnostic way to supply generation inputs
        - PaperMetaData can convert via .to_document_input()

    - **Args**:
        - `title` (str): Document title
        - `content_brief` (Dict[str, str]): Free-form content descriptions
        - `references` (List[str]): Reference entries (BibTeX or otherwise)
        - `figures` (List[Any]): Figure specs
        - `tables` (List[Any]): Table specs
        - `template_path` (Optional[str]): Layout template
        - `code_repository` (Optional[Any]): Code repository spec
        - `constraints` (Optional[GenerationConstraints]): Generation constraints
    """
    title: str = ""
    content_brief: Dict[str, str] = Field(default_factory=dict)
    references: List[str] = Field(default_factory=list)
    figures: List[Any] = Field(default_factory=list)
    tables: List[Any] = Field(default_factory=list)
    template_path: Optional[str] = None
    code_repository: Optional[Any] = None
    constraints: Optional[GenerationConstraints] = None
