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
"""
from .client import EasyPaper
from src.agents.metadata_agent.models import (
    PaperMetaData,
    PaperGenerationResult,
    PaperGenerationRequest,
    SectionResult,
)
from src.agents.metadata_agent.progress import EventType

__all__ = [
    "EasyPaper",
    "EventType",
    "PaperMetaData",
    "PaperGenerationResult",
    "PaperGenerationRequest",
    "SectionResult",
]
