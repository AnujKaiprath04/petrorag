"""
PetroRAG Core Architectural Interfaces & Schemas (Module 2.1)
Defines typed abstract interfaces and foundational data models for the
entire RAG pipeline, ensuring modularity, vendor independence, and testability.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional
from pydantic import BaseModel, Field


# ==============================================================================
# 1. CORE DATA TRANSFER MODELS
# ==============================================================================

class RetrievalChannel(str, Enum):
    """Retrieval channel provenance."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    METADATA = "metadata"


class RetrievedChunk(BaseModel):
    """
    Standardized payload for a single retrieved chunk of text
    from any retrieval channel (Dense vector, BM25 sparse, or metadata-filtered).
    """
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Source document identifier")
    document_title: Optional[str] = Field(None, description="Document display name or filename")
    text: str = Field(..., description="The textual chunk content")
    score: float = Field(default=0.0, description="Initial retrieval score (similarity or BM25)")
    page_number: Optional[int] = Field(None, description="Exact 1-based page number in source document")
    section_title: Optional[str] = Field(None, description="Heading/section title where chunk belongs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Domain metadata (field, well, equipment, etc.)")
    channel: RetrievalChannel = Field(default=RetrievalChannel.DENSE, description="Retrieval source channel")


class RerankedChunk(BaseModel):
    """
    Retrieved chunk enriched with second-stage reranker scoring and rank assignment.
    """
    chunk_id: str
    document_id: str
    document_title: Optional[str] = None
    text: str
    initial_score: float = Field(..., description="Score before reranking (e.g. RRF or cosine score)")
    reranker_score: float = Field(..., description="Calibrated cross-encoder relevance score")
    rank: int = Field(..., description="1-indexed position after reranking")
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CompressedChunk(BaseModel):
    """
    Extracted/compressed representation of a chunk with token reduction and parameter auditing.
    """
    chunk_id: str
    document_id: str
    document_title: Optional[str] = None
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float = Field(..., description="compressed_tokens / original_tokens")
    preserved_parameters: List[str] = Field(default_factory=list)
    retained_sentences_count: int = Field(default=0)
    total_sentences_count: int = Field(default=0)
    rank: int = Field(default=1)
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BuiltContext(BaseModel):
    """
    Compacted, ordered context assembled for LLM prompt injection.
    """
    context_text: str = Field(..., description="Formatted string ready to inject into prompt")
    chunks: List[RerankedChunk] = Field(default_factory=list, description="Ordered chunks included in context")
    token_count: int = Field(default=0, description="Estimated token count of the context block")
    total_candidates_evaluated: int = Field(default=0, description="Total candidates considered before compaction")
    pruned_chunks_count: int = Field(default=0, description="Chunks excluded due to budget or low relevance")
    preserved_parameters: List[str] = Field(default_factory=list, description="Critical technical values preserved")
    compression_ratio: Optional[float] = Field(default=None, description="Ratio of compressed tokens to raw tokens if compression applied")


class PromptBundle(BaseModel):
    """
    Complete prompt structure partitioned into system instruction and user content.
    Prevents indirect prompt injection by isolating untrusted retrieved context into data frames.
    """
    system_prompt: str = Field(..., description="Immutable system directives and safety constraints")
    user_prompt: str = Field(..., description="Sanitized user query with isolated data payload")
    raw_messages: Optional[List[Dict[str, str]]] = Field(None, description="Chat completion message format")


class GenerationConfig(BaseModel):
    """Configuration parameters for LLM generation."""
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    top_p: Optional[float] = Field(default=1.0)
    stop_sequences: Optional[List[str]] = Field(default=None)
    seed: Optional[int] = Field(default=42)
    stream: bool = Field(default=False)


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider."""
    content: str = Field(..., description="Generated text content")
    model_name: str = Field(..., description="Model identifier that produced the response")
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    finish_reason: Optional[str] = Field(default="stop")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Traceable citation pointing to an exact source chunk, page, and document."""
    citation_id: str = Field(..., description="Unique citation handle (e.g. [1], [DOC-42-p5])")
    document_id: str = Field(..., description="Unique source document ID")
    document_title: Optional[str] = Field(None, description="Human-readable document title")
    page_number: Optional[int] = Field(None, description="1-indexed source page number")
    section_title: Optional[str] = Field(None, description="Document section or heading")
    chunk_id: str = Field(..., description="Underlying chunk ID")
    text_snippet: str = Field(..., description="Verbatim text quote from the chunk supporting the claim")
    relevance_score: float = Field(default=0.0, description="Reranker or retrieval score of supporting source")


class ClaimSupportStatus(str, Enum):
    """NLI entailment status of an individual factual claim."""
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


class ClaimVerification(BaseModel):
    """Verification analysis of an individual atomic claim extracted from the answer."""
    claim_id: str
    claim_text: str
    status: ClaimSupportStatus
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_citation_ids: List[str] = Field(default_factory=list)
    supporting_chunk_ids: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None


class GroundingResult(BaseModel):
    """Complete post-generation grounding and verification assessment."""
    grounding_score: float = Field(..., ge=0.0, le=1.0, description="Ratio of supported claims to total claims")
    total_claims: int = Field(default=0)
    supported_claims: int = Field(default=0)
    partially_supported_claims: int = Field(default=0)
    unsupported_claims: int = Field(default=0)
    is_grounded: bool = Field(default=True, description="True if grounding_score >= system threshold")
    claims: List[ClaimVerification] = Field(default_factory=list)
    hallucination_detected: bool = Field(default=False)
    abstention_recommended: bool = Field(default=False)


# ==============================================================================
# 2. ABSTRACT BASE CLASSES (INTERFACES)
# ==============================================================================

class BaseRetriever(ABC):
    """
    Abstract interface for all retrieval components (Dense, BM25, Hybrid, Metadata).
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Synchronously retrieve relevant chunks matching query."""
        pass

    @abstractmethod
    async def aretrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Asynchronously retrieve relevant chunks matching query."""
        pass


class BaseReranker(ABC):
    """
    Abstract interface for second-stage candidate reranking (e.g. Cross-Encoder).
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = 5
    ) -> List[RerankedChunk]:
        """Synchronously score and rerank a candidate chunk pool."""
        pass

    @abstractmethod
    async def arerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = 5
    ) -> List[RerankedChunk]:
        """Asynchronously score and rerank a candidate chunk pool."""
        pass


class BaseContextBuilder(ABC):
    """
    Abstract interface for assembling, deduplicating, and token-budgeting context.
    """

    @abstractmethod
    def build_context(
        self,
        candidates: List[RerankedChunk],
        token_budget: int = 2048,
        preserve_parameters: bool = True
    ) -> BuiltContext:
        """Construct compact context payload within token budget."""
        pass


class BaseContextCompressor(ABC):
    """
    Abstract interface for extractive context compression engines.
    """

    @abstractmethod
    def compress_chunk(
        self,
        chunk: RerankedChunk,
        query: str,
        target_ratio: Optional[float] = None
    ) -> CompressedChunk:
        """Compress single chunk by selecting salient sentences against query."""
        pass

    @abstractmethod
    def compress_chunks(
        self,
        chunks: List[RerankedChunk],
        query: str,
        target_ratio: Optional[float] = None
    ) -> List[CompressedChunk]:
        """Compress a pool of candidate chunks."""
        pass


class BasePromptBuilder(ABC):
    """
    Abstract interface for composing injection-resistant RAG prompt structures.
    """

    @abstractmethod
    def build_prompt(
        self,
        query: str,
        context: BuiltContext,
        system_instruction: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> PromptBundle:
        """Assemble structured prompt isolating untrusted data from instructions."""
        pass


class BaseLLMProvider(ABC):
    """
    Abstract interface for model providers (OpenAI, Gemini, Anthropic, Ollama, Local).
    """

    @abstractmethod
    async def generate(
        self,
        prompt: PromptBundle,
        config: Optional[GenerationConfig] = None
    ) -> LLMResponse:
        """Asynchronously generate a completion."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: PromptBundle,
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        """Asynchronously stream generation tokens."""
        pass


class BaseCitationEngine(ABC):
    """
    Abstract interface for building traceable, deterministic citations from context.
    """

    @abstractmethod
    def extract_citations(
        self,
        answer_text: str,
        context: BuiltContext
    ) -> List[Citation]:
        """Extract and validate citations mapping answer statements to evidence chunks."""
        pass


class BaseGroundingEvaluator(ABC):
    """
    Abstract interface for claim extraction and NLI factual grounding verification.
    """

    @abstractmethod
    async def evaluate_grounding(
        self,
        answer_text: str,
        context: BuiltContext,
        citations: List[Citation]
    ) -> GroundingResult:
        """Deconstruct answer into claims and assess entailment against context."""
        pass
