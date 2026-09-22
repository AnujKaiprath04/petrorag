"""
PetroRAG Standardized API Response & Request Schemas (Module 2.18)
Comprehensive, type-safe Pydantic contracts representing end-to-end RAG answers,
citations, claim grounding verification, abstention barriers, and system telemetry.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SourceDocumentInfo(BaseModel):
    """Detailed metadata for a supporting document chunk."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Source document identifier")
    document_title: Optional[str] = Field(None, description="Document display name")
    page_number: Optional[int] = Field(None, description="Source page number")
    section_title: Optional[str] = Field(None, description="Section heading")
    relevance_score: float = Field(default=0.0, description="Calibrated relevance/reranking score")
    channel: str = Field(default="hybrid", description="Retrieval channel (dense, sparse, hybrid, metadata)")
    snippet: Optional[str] = Field(None, description="Excerpt from chunk text")


class CitationInfo(BaseModel):
    """Specific traceable citation mapping statement in answer to exact document page."""
    citation_id: str = Field(..., description="Citation marker (e.g. [1])")
    document_id: str = Field(..., description="Source document identifier")
    document_title: Optional[str] = Field(None, description="Document display name")
    page_number: Optional[int] = Field(None, description="1-indexed source page number")
    chunk_id: str = Field(..., description="Underlying chunk ID")
    text_snippet: str = Field(..., description="Verbatim supporting text quote")
    relevance_score: float = Field(default=0.0)


class ClaimVerificationInfo(BaseModel):
    """Atomic claim factual assessment."""
    claim_id: str = Field(..., description="Atomic claim ID")
    claim_text: str = Field(..., description="Extracted factual proposition")
    status: str = Field(..., description="Entailment status: SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, CONTRADICTED")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    supporting_citations: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None


class GroundingMetadata(BaseModel):
    """Factual grounding and NLI verification metrics."""
    grounding_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Ratio of supported claims")
    is_grounded: bool = Field(default=True, description="True if grounding_score >= threshold")
    total_claims: int = Field(default=0)
    supported_claims: int = Field(default=0)
    unsupported_claims: int = Field(default=0)
    claims: List[ClaimVerificationInfo] = Field(default_factory=list)


class AbstentionInfo(BaseModel):
    """Explicit safety barrier declaration when system cannot reliably answer."""
    is_abstained: bool = Field(default=False, description="True if safety barrier triggered abstention")
    barrier: Optional[str] = Field(None, description="Triggering barrier (RETRIEVAL_QUALITY, GROUNDING, INSUFFICIENT_EVIDENCE)")
    reason: Optional[str] = Field(None, description="Technical reason for abstention")
    fallback_recommendation: Optional[str] = Field(None, description="Recommended next action or manual escalation")


class TelemetryMetadata(BaseModel):
    """End-to-end performance and latency telemetry."""
    total_latency_ms: float = Field(default=0.0, description="Total request duration in ms")
    retrieval_latency_ms: float = Field(default=0.0, description="Dense/BM25 retrieval time in ms")
    rerank_latency_ms: float = Field(default=0.0, description="Cross-encoder reranking time in ms")
    generation_latency_ms: float = Field(default=0.0, description="LLM generation time in ms")
    verification_latency_ms: float = Field(default=0.0, description="Grounding verification time in ms")
    time_to_first_token_ms: Optional[float] = Field(None, description="Streaming TTFT")
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    tokens_per_second: float = Field(default=0.0)
    model_name: str = Field(default="petrorag-default")


class PetroRAGRequest(BaseModel):
    """Standard user query payload for decision support endpoint."""
    query: str = Field(..., description="User's operational or technical question")
    top_k: int = Field(default=5, ge=1, le=50)
    enable_reranking: bool = Field(default=True)
    enable_compression: bool = Field(default=True)
    enable_grounding: bool = Field(default=True)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    chat_history: Optional[List[Dict[str, str]]] = Field(default=None)
    metadata_filters: Optional[Dict[str, Any]] = Field(default=None)


class PetroRAGResponse(BaseModel):
    """
    Standardized, production-grade response schema for PetroRAG decision support.
    """
    query_id: str = Field(..., description="Unique request tracing ID")
    query: str = Field(..., description="Original user query")
    intent: str = Field(default="UNKNOWN", description="Classified O&G intent category")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted domain entities")
    answer: str = Field(..., description="Grounded technical answer text")
    sources: List[SourceDocumentInfo] = Field(default_factory=list, description="Supporting document chunks")
    citations: List[CitationInfo] = Field(default_factory=list, description="Traceable citations in answer")
    grounding: GroundingMetadata = Field(default_factory=GroundingMetadata)
    abstention: AbstentionInfo = Field(default_factory=AbstentionInfo)
    telemetry: TelemetryMetadata = Field(default_factory=TelemetryMetadata)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_grounded(self) -> bool:
        return self.grounding.is_grounded

    @property
    def is_abstained(self) -> bool:
        return self.abstention.is_abstained

    @property
    def confidence(self) -> float:
        return self.grounding.grounding_score
