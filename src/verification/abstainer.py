"""
PetroRAG Multi-Barrier Abstention Mechanism (Module 2.24)
Enforces multi-layer safety barriers preventing hallucinated or ungrounded answers
from being delivered to field engineers and operators.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.schemas.rag_response import AbstentionInfo
from src.core.config import settings
from src.core.interfaces import (
    BuiltContext,
    GroundingResult,
    RerankedChunk,
)
from src.core.logging import logger


class AbstentionBarrier(str, Enum):
    """The four defensive safety barriers in PetroRAG."""
    QUERY_SECURITY_OR_SCOPE = "QUERY_SECURITY_OR_SCOPE"
    RETRIEVAL_QUALITY = "RETRIEVAL_QUALITY"
    GROUNDING_FAILURE = "GROUNDING_FAILURE"
    HALLUCINATION_DETECTED = "HALLUCINATION_DETECTED"


class AbstentionDecision(BaseModel):
    """Outcome of multi-barrier safety verification."""
    is_abstained: bool = Field(..., description="True if any safety barrier was breached")
    barrier: Optional[AbstentionBarrier] = Field(None, description="The specific barrier that blocked output")
    reason: Optional[str] = Field(None, description="Detailed technical reason for abstention")
    abstention_message: Optional[str] = Field(None, description="Standardized, safe user-facing message")
    fallback_recommendation: Optional[str] = Field(None, description="Actionable escalation guidance")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)

    def to_schema(self) -> AbstentionInfo:
        """Converts to API response AbstentionInfo."""
        return AbstentionInfo(
            is_abstained=self.is_abstained,
            barrier=self.barrier.value if self.barrier else None,
            reason=self.reason,
            fallback_recommendation=self.fallback_recommendation
        )


class MultiBarrierAbstainer:
    """
    Evaluates retrieval quality, factual grounding, and technical hallucination
    barriers to decide whether to permit answer release or trigger safe abstention.
    """

    def __init__(
        self,
        retrieval_threshold: float = settings.ABSTENTION_RETRIEVAL_THRESHOLD,
        grounding_threshold: float = settings.GROUNDING_THRESHOLD
    ):
        self.retrieval_threshold = retrieval_threshold
        self.grounding_threshold = grounding_threshold

    def evaluate_retrieval_barrier(
        self,
        chunks: List[RerankedChunk]
    ) -> AbstentionDecision:
        """
        Barrier 2: Checks if candidate evidence pool meets minimum relevance standards.
        """
        if not chunks:
            return AbstentionDecision(
                is_abstained=True,
                barrier=AbstentionBarrier.RETRIEVAL_QUALITY,
                reason="No relevant documentation was retrieved from the knowledge base for this query.",
                abstention_message=(
                    "The available technical documentation does not contain information to answer this query. "
                    "No relevant procedures or equipment specifications were found."
                ),
                fallback_recommendation="Verify equipment tags or consult the physical vendor engineering repository.",
                confidence_score=0.0
            )

        best_score = max(c.reranker_score for c in chunks)
        if best_score < self.retrieval_threshold:
            return AbstentionDecision(
                is_abstained=True,
                barrier=AbstentionBarrier.RETRIEVAL_QUALITY,
                reason=f"Top retrieved candidate score ({best_score:.4f}) is below confidence threshold ({self.retrieval_threshold:.4f}).",
                abstention_message=(
                    "Available documents have low relevance to this specific question. "
                    "To prevent operational risk, an ungrounded response has been suppressed."
                ),
                fallback_recommendation="Refine search terms or check with the lead discipline engineer.",
                confidence_score=best_score
            )

        return AbstentionDecision(
            is_abstained=False,
            confidence_score=best_score
        )

    def evaluate_grounding_barrier(
        self,
        grounding_result: GroundingResult
    ) -> AbstentionDecision:
        """
        Barriers 3 & 4: Checks claim-level NLI grounding and hallucination flags.
        """
        if grounding_result.hallucination_detected:
            return AbstentionDecision(
                is_abstained=True,
                barrier=AbstentionBarrier.HALLUCINATION_DETECTED,
                reason="One or more unsupported numerical parameters or direct document contradictions were detected in the generated answer.",
                abstention_message=(
                    "The generated answer contains numerical parameters or operational claims "
                    "that cannot be verified against approved engineering documentation."
                ),
                fallback_recommendation="Refer to certified piping and instrumentation diagrams (P&IDs) or OEM datasheets.",
                confidence_score=grounding_result.grounding_score
            )

        if grounding_result.grounding_score < self.grounding_threshold:
            return AbstentionDecision(
                is_abstained=True,
                barrier=AbstentionBarrier.GROUNDING_FAILURE,
                reason=f"Grounding score ({grounding_result.grounding_score:.4f}) is below threshold ({self.grounding_threshold:.4f}).",
                abstention_message=(
                    "The system cannot verify all claims with sufficient factual certainty. "
                    "Portions of the required operational evidence are missing."
                ),
                fallback_recommendation="Escalate to operations supervisor for manual procedural confirmation.",
                confidence_score=grounding_result.grounding_score
            )

        return AbstentionDecision(
            is_abstained=False,
            confidence_score=grounding_result.grounding_score
        )

    def evaluate_all_barriers(
        self,
        chunks: List[RerankedChunk],
        grounding_result: Optional[GroundingResult] = None
    ) -> AbstentionDecision:
        """
        Sequential barrier evaluation across retrieval and post-generation grounding.
        """
        # 1. Retrieval barrier check
        retrieval_decision = self.evaluate_retrieval_barrier(chunks)
        if retrieval_decision.is_abstained:
            logger.warning(f"Abstention triggered at retrieval barrier: {retrieval_decision.reason}")
            return retrieval_decision

        # 2. Grounding barrier check
        if grounding_result is not None:
            grounding_decision = self.evaluate_grounding_barrier(grounding_result)
            if grounding_decision.is_abstained:
                logger.warning(f"Abstention triggered at grounding barrier: {grounding_decision.reason}")
                return grounding_decision

        return AbstentionDecision(
            is_abstained=False,
            confidence_score=grounding_result.grounding_score if grounding_result else retrieval_decision.confidence_score
        )
