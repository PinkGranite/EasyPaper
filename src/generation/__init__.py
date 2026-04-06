"""
Constrained generation layer — decomposed generation, template-slot filling,
and claim-level verification.
"""
from .template_slots import (
    SlotType,
    TemplateSlot,
    ParagraphTemplate,
    parse_template_slots,
    render_filled_template,
)
from .claim_verifier import (
    ClaimVerifier,
    VerificationResult,
    MAX_CLAIM_RETRIES,
    TEMPLATE_FALLBACK_ENABLED,
)

__all__ = [
    "SlotType",
    "TemplateSlot",
    "ParagraphTemplate",
    "parse_template_slots",
    "render_filled_template",
    "ClaimVerifier",
    "VerificationResult",
    "MAX_CLAIM_RETRIES",
    "TEMPLATE_FALLBACK_ENABLED",
]
