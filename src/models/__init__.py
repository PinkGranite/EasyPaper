"""
EasyPaper shared data models.
"""
from .evidence_graph import (
    EvidenceNodeType,
    ClaimNodeType,
    EdgeType,
    EvidenceNode,
    ClaimNode,
    DAGEdge,
    EvidenceDAG,
)
from .action_space import (
    TextAction,
    LayoutAction,
    ActionStatus,
    Action,
    ACTION_PRIORITY,
    from_legacy_action,
)
from .document_spec import (
    ContentSection,
    DocumentSpec,
    GenerationConstraints,
    DocumentInput,
)

__all__ = [
    "EvidenceNodeType",
    "ClaimNodeType",
    "EdgeType",
    "EvidenceNode",
    "ClaimNode",
    "DAGEdge",
    "EvidenceDAG",
    "TextAction",
    "LayoutAction",
    "ActionStatus",
    "Action",
    "ACTION_PRIORITY",
    "from_legacy_action",
    "ContentSection",
    "DocumentSpec",
    "GenerationConstraints",
    "DocumentInput",
]
