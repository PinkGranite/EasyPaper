from .writer_agent import WriterAgent
from .section_models import (
    # Enums
    SectionType,
    MaterialType,
    PointType,
    PointRelation,
    # Core argument models
    Material,
    Point,
    ArgumentStructure,
    # Resource models
    ReferenceInfo,
    FigureInfo,
    TableInfo,
    EquationInfo,
    TemplateRules,
    SectionResources,
    # Constraints
    SectionConstraints,
    # Payload and result
    SectionWritePayload,
    SectionWriteResult,
    SectionWriteRequest,
    SectionTaskStatus,
    # Validation
    SectionRequirement,
    ValidationResult,
    SECTION_REQUIREMENTS,
    validate_section_payload,
    get_section_requirements,
    # Paper assembly
    SectionChainItem,
    PaperChainConfig,
    PaperAssemblyResult,
    SectionGenerationStatus,
    # Backward compatibility (deprecated)
    NodeContent,
    SectionContext,
    ResourceRequirement,
    SECTION_REQUIRED_RESOURCES,
    ClaimType,       # Deprecated: use PointType
    ClaimRelation,   # Deprecated: use PointRelation
    Claim,           # Deprecated: use Point
)

__all__ = [
    "WriterAgent",
    # Enums
    "SectionType",
    "MaterialType",
    "PointType",
    "PointRelation",
    # Core argument models
    "Material",
    "Point",
    "ArgumentStructure",
    # Resource models
    "ReferenceInfo",
    "FigureInfo",
    "TableInfo",
    "EquationInfo",
    "TemplateRules",
    "SectionResources",
    # Constraints
    "SectionConstraints",
    # Payload and result
    "SectionWritePayload",
    "SectionWriteResult",
    "SectionWriteRequest",
    "SectionTaskStatus",
    # Validation
    "SectionRequirement",
    "ValidationResult",
    "SECTION_REQUIREMENTS",
    "validate_section_payload",
    "get_section_requirements",
    # Paper assembly
    "SectionChainItem",
    "PaperChainConfig",
    "PaperAssemblyResult",
    "SectionGenerationStatus",
    # Backward compatibility (deprecated)
    "NodeContent",
    "SectionContext",
    "ResourceRequirement",
    "SECTION_REQUIRED_RESOURCES",
    "ClaimType",       # Deprecated: use PointType
    "ClaimRelation",   # Deprecated: use PointRelation
    "Claim",           # Deprecated: use Point
]
