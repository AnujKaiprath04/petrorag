"""
PetroRAG Verification Package
Exports citation extraction, source lineage auditing, atomic claim decomposition,
factual grounding verification, hallucination detection, multi-barrier abstention,
confidence calibration, contradiction detection, and document revision management.
"""

from src.verification.citation_engine import CitationEngine
from src.verification.lineage import (
    SourceTraceabilityAuditor,
    LineageRecord,
    LineageManifest,
)
from src.verification.claim_extractor import ClaimExtractor, AtomicClaim
from src.verification.grounding_evaluator import GroundingEvaluator
from src.verification.abstainer import (
    MultiBarrierAbstainer,
    AbstentionBarrier,
    AbstentionDecision,
)
from src.verification.confidence_estimator import (
    ConfidenceEstimator,
    ConfidenceLevel,
    ConfidenceAssessment,
)
from src.verification.contradiction_detector import (
    ContradictionDetector,
    ContradictionReport,
    ConflictPair,
)
from src.verification.revision_manager import (
    RevisionManager,
    RevisionMetadata,
)

__all__ = [
    "CitationEngine",
    "SourceTraceabilityAuditor",
    "LineageRecord",
    "LineageManifest",
    "ClaimExtractor",
    "AtomicClaim",
    "GroundingEvaluator",
    "MultiBarrierAbstainer",
    "AbstentionBarrier",
    "AbstentionDecision",
    "ConfidenceEstimator",
    "ConfidenceLevel",
    "ConfidenceAssessment",
    "ContradictionDetector",
    "ContradictionReport",
    "ConflictPair",
    "RevisionManager",
    "RevisionMetadata",
]
