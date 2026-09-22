"""
PetroRAG Confidence Estimation Engine (Module 2.25)
Synthesizes multi-factor operational confidence across retrieval quality,
NLI claim grounding, citation precision, and hallucination absence.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.core.interfaces import GroundingResult


class ConfidenceLevel(str, Enum):
    """Categorical confidence level for decision support triage."""
    HIGH = "HIGH"              # >= 0.85: Verified by SOP/standard with exact parameters
    MEDIUM = "MEDIUM"          # 0.70 - 0.84: Well grounded, acceptable for standard operations
    LOW = "LOW"                # 0.40 - 0.69: Partial evidence, require engineering review
    UNRELIABLE = "UNRELIABLE"  # < 0.40 or hallucinated: Must be blocked from field use


class ConfidenceAssessment(BaseModel):
    """Multi-dimensional confidence report."""
    composite_score: float = Field(..., ge=0.0, le=1.0, description="Weighted composite confidence score")
    level: ConfidenceLevel = Field(..., description="Categorical confidence grade")
    retrieval_confidence: float = Field(default=0.0)
    grounding_confidence: float = Field(default=0.0)
    citation_confidence: float = Field(default=0.0)
    is_actionable_for_field: bool = Field(default=False, description="Safe for direct operational execution")
    risk_factors: List[str] = Field(default_factory=list)


class ConfidenceEstimator:
    """
    Calibrates operational confidence using weighted multi-component evidence:
    - 40% Factual Grounding (claim-level NLI score)
    - 35% Retrieval Quality (cross-encoder relevance score)
    - 25% Citation Precision (valid supporting citations ratio)
    Penalizes heavily (drops to UNRELIABLE) if hallucination or contradiction is detected.
    """

    def __init__(
        self,
        weight_grounding: float = 0.45,
        weight_retrieval: float = 0.35,
        weight_citation: float = 0.20
    ):
        self.weight_grounding = weight_grounding
        self.weight_retrieval = weight_retrieval
        self.weight_citation = weight_citation

    def estimate_confidence(
        self,
        retrieval_score: float,
        grounding_result: GroundingResult,
        citation_precision: float = 1.0,
        is_abstained: bool = False
    ) -> ConfidenceAssessment:
        """
        Computes composite confidence and categorical field-readiness rating.
        """
        risks: List[str] = []

        if is_abstained:
            risks.append("System safety abstention active.")
            return ConfidenceAssessment(
                composite_score=0.0,
                level=ConfidenceLevel.UNRELIABLE,
                retrieval_confidence=retrieval_score,
                grounding_confidence=0.0,
                citation_confidence=0.0,
                is_actionable_for_field=False,
                risk_factors=risks
            )

        if grounding_result.hallucination_detected:
            risks.append("Hallucination or unsupported parameters detected.")
            return ConfidenceAssessment(
                composite_score=round(grounding_result.grounding_score * 0.3, 4),
                level=ConfidenceLevel.UNRELIABLE,
                retrieval_confidence=retrieval_score,
                grounding_confidence=grounding_result.grounding_score,
                citation_confidence=citation_precision,
                is_actionable_for_field=False,
                risk_factors=risks
            )

        # Calculate weighted composite
        composite = (
            self.weight_grounding * grounding_result.grounding_score
            + self.weight_retrieval * min(1.0, max(0.0, retrieval_score))
            + self.weight_citation * min(1.0, max(0.0, citation_precision))
        )
        composite = round(min(1.0, max(0.0, composite)), 4)

        if retrieval_score < 0.60:
            risks.append("Marginal retrieval relevance.")
        if grounding_result.partially_supported_claims > 0:
            risks.append(f"{grounding_result.partially_supported_claims} claims only partially supported.")
        if citation_precision < 0.80:
            risks.append("Sub-optimal citation precision.")

        if composite >= 0.85:
            level = ConfidenceLevel.HIGH
            is_actionable = True
        elif composite >= 0.70:
            level = ConfidenceLevel.MEDIUM
            is_actionable = True
        elif composite >= 0.40:
            level = ConfidenceLevel.LOW
            is_actionable = False
        else:
            level = ConfidenceLevel.UNRELIABLE
            is_actionable = False

        return ConfidenceAssessment(
            composite_score=composite,
            level=level,
            retrieval_confidence=round(retrieval_score, 4),
            grounding_confidence=round(grounding_result.grounding_score, 4),
            citation_confidence=round(citation_precision, 4),
            is_actionable_for_field=is_actionable,
            risk_factors=risks
        )
