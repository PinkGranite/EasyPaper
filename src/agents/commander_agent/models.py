"""
Models for Commander Agent
- **Description**:
    - Commander acts as adapter between FlowGram.ai and Writer Agent
    - Outputs unified SectionWritePayload format
    - Also supports canvas-level metadata extraction for full paper generation
"""
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List
import uuid

# Import unified models from Writer Agent
from ..writer_agent.section_models import SectionWritePayload


class CommanderPayload(BaseModel):
    """
    Payload for Commander Agent request
    - **Description**:
        - Input parameters for paper section generation orchestration
        - Accepts FlowGram.ai specific parameters (work_id, node_ids)
    """
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    payload: Dict[str, Any]


class CommanderResult(BaseModel):
    """
    Result from Commander Agent
    - **Description**:
        - Contains unified SectionWritePayload for Writer Agent
        - This is the interface between Commander and Writer
    """
    request_id: str
    status: str  # 'ok' or 'error'
    section_write_payload: Optional[SectionWritePayload] = None
    error: Optional[str] = None


# ============================================================================
# Canvas Metadata Extraction Models
# ============================================================================

class SectionHint(BaseModel):
    """
    Structural hint from a PaperSection node on the canvas.
    Guides the Planner during section planning.
    """
    section_type: str
    title: str = ""
    user_prompt: str = ""
    word_count_limit: Optional[int] = None


class CanvasMetadata(BaseModel):
    """
    Structured metadata extracted from a research canvas by LLM.
    Maps directly to MetadataGenerateRequest fields.
    """
    title: str = "Untitled Paper"
    idea_hypothesis: str = ""
    method: str = ""
    data: str = ""
    experiments: str = ""
    references: List[str] = Field(default_factory=list)
    section_hints: List[SectionHint] = Field(default_factory=list)
    style_guide: Optional[str] = None


class ExtractMetadataPayload(BaseModel):
    """Payload for the extract-metadata endpoint."""
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    canvas_data: Dict[str, Any]


class ExtractMetadataResult(BaseModel):
    """Result from the extract-metadata endpoint."""
    request_id: str
    status: str
    metadata: Optional[CanvasMetadata] = None
    error: Optional[str] = None
