"""
Models for Writer Agent
"""
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List, Set
import uuid


class ReviewResult(BaseModel):
    """
    Result from mini-review of generated content.
    """
    passed: bool = True
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    invalid_citations: List[str] = Field(default_factory=list)
    word_count: int = 0
    target_words: Optional[int] = None
    key_point_coverage: float = 1.0


class WriterPayload(BaseModel):
    """
    Payload for Writer Agent request
    - **Description**:
        - Contains the compiled prompt from Commander Agent
        - Supports iterative review with citation validation
    """
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    payload: Dict[str, Any]
    # Expected payload fields:
    # - system_prompt: str
    # - user_prompt: str
    # - section_type: str
    # - citation_format: str (default: "cite")
    # - constraints: List[str]
    
    # New fields for iterative review
    valid_citation_keys: List[str] = Field(default_factory=list)
    target_words: Optional[int] = None
    key_points: List[str] = Field(default_factory=list)
    revision_plan: Optional[Dict[str, Any]] = None
    max_iterations: int = 2
    enable_review: bool = True


class ExecutionReceipt(BaseModel):
    """
    Writer execution-only receipt for one revision task.
    - **Description**:
        - Captures what was executed without policy decisions
        - Standardized as target + instruction + constraints + disposition + evidence
    """
    target_id: str
    section_type: str
    disposition: str  # executed | no_change | failed
    instruction: str = ""
    constraints: Dict[str, Any] = Field(default_factory=dict)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    paragraph_index: Optional[int] = None


class GeneratedContent(BaseModel):
    """
    Generated LaTeX content from Writer Agent
    - **Description**:
        - Contains the raw LaTeX content and metadata
        - Includes review results from iterative refinement
    """
    latex_content: str
    section_type: str
    word_count: int = 0
    citation_ids: List[str] = Field(default_factory=list)
    figure_ids: List[str] = Field(default_factory=list)
    table_ids: List[str] = Field(default_factory=list)
    # Review results
    iterations_used: int = 1
    review_passed: bool = True
    invalid_citations_removed: List[str] = Field(default_factory=list)
    paragraph_units: List[Dict[str, Any]] = Field(default_factory=list)
    writer_response_section: List[ExecutionReceipt] = Field(default_factory=list)
    writer_response_paragraph: List[ExecutionReceipt] = Field(default_factory=list)


class ParagraphUnit(BaseModel):
    """
    Paragraph-level structured unit extracted from section LaTeX.
    - **Description**:
        - Provides stable paragraph IDs for targeted revision
        - Stores lightweight sentence splits for diagnostics
    """
    paragraph_id: str
    section_type: str
    paragraph_index: int
    text: str
    sentence_count: int = 0
    sentences: List[str] = Field(default_factory=list)


class SectionDraft(BaseModel):
    """
    Structured section draft with paragraph-addressable content.
    """
    section_type: str
    latex_content: str
    paragraphs: List[ParagraphUnit] = Field(default_factory=list)


class ParagraphResult(BaseModel):
    """
    Result from paragraph-level (claim-level) generation.
    - **Description**:
        - Returned by WriterAgent.generate_paragraph() in decomposed generation mode.
        - Tracks claim coverage and citation usage for immediate verification.
    """
    latex_content: str = ""
    paragraph_index: int = 0
    claim_id: str = ""
    used_citations: List[str] = Field(default_factory=list)
    word_count: int = 0
    claim_coverage: float = 1.0
    verification_passed: bool = False
    attempt: int = 1


class CoreContentResult(BaseModel):
    """
    Result from Stage 1 core content generation (no citations).
    """
    raw_latex: str = ""
    paragraph_index: int = 0
    word_count: int = 0
    attempt: int = 1


class CitationAction(BaseModel):
    """
    Single citation edit action from Stage 2 citation injection.
    """
    action: str  # "replace_marker", "insert_sentence", "rewrite_sentence"
    marker_or_location: str  # "[CITE:contrastive]" or "after_sentence:2"
    new_text: str  # replacement text with \cite{} included
    cite_keys: List[str] = Field(default_factory=list)


class CitationEditResult(BaseModel):
    """
    Result from Stage 2 citation injection LLM call.
    """
    actions: List[CitationAction] = Field(default_factory=list)
    raw_response: str = ""


class WriterResult(BaseModel):
    """
    Result from Writer Agent
    - **Description**:
        - Contains generated LaTeX content or error
    """
    request_id: str
    status: str  # 'ok' or 'error'
    result: Optional[GeneratedContent] = None
    error: Optional[str] = None
    # Detailed review info
    review_history: List[ReviewResult] = Field(default_factory=list)
    writer_response_section: List[ExecutionReceipt] = Field(default_factory=list)
    writer_response_paragraph: List[ExecutionReceipt] = Field(default_factory=list)
