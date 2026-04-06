"""
EasyPaper -- AI-powered academic paper generation SDK.

Public API::

    from easypaper import EasyPaper, PaperMetaData, EventType

    ep = EasyPaper(config_path="configs/my.yaml")

    # one-shot
    result = await ep.generate(metadata)

    # streaming
    async for event in ep.generate_stream(metadata):
        print(event["phase"], event["message"])

    # standalone Docling
    result = ep.parse_pdf("paper.pdf")
    result = await ep.download_and_parse("https://arxiv.org/pdf/2301.12345.pdf")
"""
from .client import EasyPaper
from src.agents.metadata_agent.models import (
    PaperMetaData,
    PaperGenerationResult,
    PaperGenerationRequest,
    SectionResult,
)
from src.agents.metadata_agent.progress import EventType
from src.agents.shared.docling_analyzer import DoclingPaperResult
from src.agents.shared.docling_service import DoclingService

__all__ = [
    "EasyPaper",
    "EventType",
    "PaperMetaData",
    "PaperGenerationResult",
    "PaperGenerationRequest",
    "SectionResult",
    "DoclingPaperResult",
    "DoclingService",
]
