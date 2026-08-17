"""External expert-knowledge ports used by the planning graph."""

from underwater_tracking.knowledge.client import (
    KnowledgeReference,
    KnowledgeQueryResult,
    KnowledgeProvider,
    KnowledgeServiceError,
    OntologyKnowledgeClient,
)

__all__ = [
    "KnowledgeReference",
    "KnowledgeQueryResult",
    "KnowledgeProvider",
    "KnowledgeServiceError",
    "OntologyKnowledgeClient",
]
