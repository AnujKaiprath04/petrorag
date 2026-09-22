"""
PetroRAG Unified End-to-End Pipeline Orchestrator (Module 2.28)
Orchestrates query validation, intent classification, entity extraction,
hybrid retrieval, cross-encoder reranking, revision resolution,
context compression, lost-in-the-middle selection, LLM generation,
citation extraction, factual grounding verification, multi-barrier abstention,
and confidence estimation into a production-grade decision support engine.
"""

import asyncio
import concurrent.futures
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from backend.app.schemas.rag_response import (
    AbstentionInfo,
    CitationInfo,
    ClaimVerificationInfo,
    GroundingMetadata,
    PetroRAGRequest,
    PetroRAGResponse,
    SourceDocumentInfo,
    TelemetryMetadata,
)
from src.core.config import settings
from src.core.exceptions import (
    AbstentionTriggered,
    PromptInjectionError,
    QueryValidationError,
)
from src.core.interfaces import (
    BaseCitationEngine,
    BaseContextBuilder,
    BaseContextCompressor,
    BaseGroundingEvaluator,
    BaseLLMProvider,
    BasePromptBuilder,
    BaseReranker,
    BaseRetriever,
    BuiltContext,
    GenerationConfig,
    GroundingResult,
    RerankedChunk,
    RetrievedChunk,
)
from src.core.logging import logger
from src.generation.context_compressor import ContextCompressor
from src.generation.context_selector import ContextSelector
from src.generation.generator import LLMGenerationService
from src.generation.llm_provider import MockLLMProvider, get_llm_provider
from src.generation.prompt_engine import PromptEngine
from src.query.entities import RuleBasedEntityExtractor
from src.query.intent import HybridIntentClassifier, QueryIntent
from src.query.validator import QueryValidator
from src.retrieval.deduplication import RetrievalDeduplicator
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.metadata_filter import MetadataFilterEngine
from src.retrieval.reranker import MockCrossEncoderReranker
from src.verification.abstainer import (
    AbstentionBarrier,
    AbstentionDecision,
    MultiBarrierAbstainer,
)
from src.verification.citation_engine import CitationEngine
from src.verification.claim_extractor import ClaimExtractor
from src.verification.confidence_estimator import (
    ConfidenceAssessment,
    ConfidenceEstimator,
    ConfidenceLevel,
)
from src.verification.contradiction_detector import ContradictionDetector
from src.verification.grounding_evaluator import GroundingEvaluator
from src.verification.lineage import SourceTraceabilityAuditor
from src.verification.revision_manager import RevisionManager


class PetroRAGPipeline:
    """
    Production-grade end-to-end RAG intelligence engine for Oil & Gas fields.
    Implements all 5 architectural models and ablated sub-components.
    """

    def __init__(
        self,
        validator: Optional[QueryValidator] = None,
        intent_classifier: Optional[HybridIntentClassifier] = None,
        entity_extractor: Optional[RuleBasedEntityExtractor] = None,
        retriever: Optional[BaseRetriever] = None,
        deduplicator: Optional[RetrievalDeduplicator] = None,
        reranker: Optional[BaseReranker] = None,
        revision_manager: Optional[RevisionManager] = None,
        contradiction_detector: Optional[ContradictionDetector] = None,
        context_selector: Optional[BaseContextBuilder] = None,
        context_compressor: Optional[BaseContextCompressor] = None,
        prompt_builder: Optional[BasePromptBuilder] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        citation_engine: Optional[BaseCitationEngine] = None,
        traceability_auditor: Optional[SourceTraceabilityAuditor] = None,
        grounding_evaluator: Optional[BaseGroundingEvaluator] = None,
        abstainer: Optional[MultiBarrierAbstainer] = None,
        confidence_estimator: Optional[ConfidenceEstimator] = None
    ):
        self.validator = validator or QueryValidator()
        self.intent_classifier = intent_classifier or HybridIntentClassifier()
        self.entity_extractor = entity_extractor or RuleBasedEntityExtractor()
        self.retriever = retriever or HybridRetriever()
        self.deduplicator = deduplicator or RetrievalDeduplicator()
        self.reranker = reranker or MockCrossEncoderReranker()
        self.revision_manager = revision_manager or RevisionManager()
        self.contradiction_detector = contradiction_detector or ContradictionDetector()
        self.context_selector = context_selector or ContextSelector()
        self.context_compressor = context_compressor or ContextCompressor()
        self.prompt_builder = prompt_builder or PromptEngine()
        self.llm_provider = llm_provider or get_llm_provider()
        self.generator = LLMGenerationService(
            provider=self.llm_provider,
            prompt_builder=self.prompt_builder
        )
        self.citation_engine = citation_engine or CitationEngine()
        self.traceability_auditor = traceability_auditor or SourceTraceabilityAuditor()
        self.grounding_evaluator = grounding_evaluator or GroundingEvaluator()
        self.abstainer = abstainer or MultiBarrierAbstainer()
        self.confidence_estimator = confidence_estimator or ConfidenceEstimator()

    async def query(
        self,
        request: PetroRAGRequest,
        query_id: Optional[str] = None
    ) -> PetroRAGResponse:
        """
        Executes the full end-to-end RAG decision support workflow.
        """
        req_id = query_id or f"req-{uuid.uuid4().hex[:12]}"
        t_start = time.perf_counter()
        retrieval_ms = 0.0
        rerank_ms = 0.0
        gen_ms = 0.0
        verif_ms = 0.0

        # ======================================================================
        # STAGE 1: QUERY SECURITY VALIDATION (Barrier 1)
        # ======================================================================
        val_result = self.validator.validate(request.query)
        if not val_result.is_valid:
            logger.warning(f"Query validation failed for {req_id}: {val_result.error_message}")
            return PetroRAGResponse(
                query_id=req_id,
                query=request.query,
                intent="UNKNOWN",
                answer=(
                    val_result.error_message
                    or "The query failed security and structural validation. Prompt injection patterns or malformed inputs are strictly rejected."
                ),
                abstention=AbstentionInfo(
                    is_abstained=True,
                    barrier=AbstentionBarrier.QUERY_SECURITY_OR_SCOPE.value,
                    reason=val_result.error_message or "Query validation failure",
                    fallback_recommendation="Please reformulate query using standard technical terminology."
                ),
                telemetry=TelemetryMetadata(
                    total_latency_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                    model_name="security_guard"
                )
            )

        clean_query = val_result.sanitized_query

        # ======================================================================
        # STAGE 2: INTENT CLASSIFICATION & ENTITY EXTRACTION
        # ======================================================================
        intent_res = self.intent_classifier.classify(clean_query)
        intent_name = intent_res.primary_intent.value
        entities_res = self.entity_extractor.extract(clean_query)
        entities_dict = entities_res.model_dump()

        # Merge user filters with auto-extracted metadata filters if provided
        active_filters = dict(request.metadata_filters or {})

        # ======================================================================
        # STAGE 3: MULTI-CHANNEL RETRIEVAL & DEDUPLICATION
        # ======================================================================
        t_ret_start = time.perf_counter()
        raw_candidates: List[RetrievedChunk] = self.retriever.retrieve(
            query=clean_query,
            top_k=request.top_k * 4,  # Retrieve expanded candidate pool for reranker
            filters=active_filters if active_filters else None
        )
        dedup_res = self.deduplicator.deduplicate(raw_candidates)
        deduped_candidates: List[RetrievedChunk] = (
            dedup_res.deduplicated_chunks
            if hasattr(dedup_res, "deduplicated_chunks")
            else dedup_res
        )
        retrieval_ms = round((time.perf_counter() - t_ret_start) * 1000.0, 2)

        # ======================================================================
        # STAGE 4: CROSS-ENCODER RERANKING
        # ======================================================================
        t_rerank_start = time.perf_counter()
        if request.enable_reranking and deduped_candidates:
            reranked_chunks: List[RerankedChunk] = self.reranker.rerank(
                query=clean_query,
                candidates=deduped_candidates,
                top_k=request.top_k
            )
        else:
            # Fallback ranking from initial retrieval score
            reranked_chunks = [
                RerankedChunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    document_title=c.document_title,
                    text=c.text,
                    initial_score=c.score,
                    reranker_score=c.score,
                    rank=idx + 1,
                    page_number=c.page_number,
                    section_title=c.section_title,
                    metadata=c.metadata
                )
                for idx, c in enumerate(deduped_candidates[:request.top_k])
            ]

        # Apply Document Revision resolution & recency weighting
        reranked_chunks = self.revision_manager.resolve_revisions(reranked_chunks)
        rerank_ms = round((time.perf_counter() - t_rerank_start) * 1000.0, 2)

        # ======================================================================
        # STAGE 5: RETRIEVAL QUALITY BARRIER CHECK (Barrier 2)
        # ======================================================================
        retrieval_decision = self.abstainer.evaluate_retrieval_barrier(reranked_chunks)
        if retrieval_decision.is_abstained:
            logger.warning(f"Abstention at Retrieval Barrier for {req_id}: {retrieval_decision.reason}")
            total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            return PetroRAGResponse(
                query_id=req_id,
                query=clean_query,
                intent=intent_name,
                entities=entities_dict,
                answer=retrieval_decision.abstention_message or "No relevant documentation found.",
                abstention=retrieval_decision.to_schema(),
                telemetry=TelemetryMetadata(
                    total_latency_ms=total_ms,
                    retrieval_latency_ms=retrieval_ms,
                    rerank_latency_ms=rerank_ms
                )
            )

        # ======================================================================
        # STAGE 6: CONTEXT SELECTION & COMPRESSION
        # ======================================================================
        built_context: BuiltContext = self.context_selector.build_context(
            candidates=reranked_chunks,
            token_budget=settings.CONTEXT_TOKEN_BUDGET
        )

        if request.enable_compression:
            built_context = self.context_compressor.compress_built_context(
                context=built_context,
                query=clean_query,
                target_ratio=settings.COMPRESSION_TARGET_RATIO
            )

        # ======================================================================
        # STAGE 7: LLM GENERATION
        # ======================================================================
        t_gen_start = time.perf_counter()
        gen_config = GenerationConfig(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False
        )

        gen_result = await self.generator.generate_answer(
            query=clean_query,
            context=built_context,
            intent=intent_res.primary_intent,
            config=gen_config,
            chat_history=request.chat_history
        )
        gen_ms = round((time.perf_counter() - t_gen_start) * 1000.0, 2)

        # ======================================================================
        # STAGE 8: CITATION EXTRACTION & LINEAGE AUDITING
        # ======================================================================
        citations = self.citation_engine.extract_citations(gen_result.content, built_context)
        manifest = self.traceability_auditor.build_lineage(gen_result.content, citations, built_context)
        citation_metric = self.citation_engine.validate_citations(citations, gen_result.content, built_context)

        # ======================================================================
        # STAGE 9: FACTUAL GROUNDING VERIFICATION & MULTI-BARRIER ABSTENTION
        # ======================================================================
        t_verif_start = time.perf_counter()
        if request.enable_grounding:
            grounding_result: GroundingResult = await self.grounding_evaluator.evaluate_grounding(
                answer_text=gen_result.content,
                context=built_context,
                citations=citations
            )
        else:
            grounding_result = GroundingResult(
                grounding_score=1.0,
                total_claims=1,
                supported_claims=1,
                unsupported_claims=0,
                is_grounded=True,
                hallucination_detected=False
            )
        verif_ms = round((time.perf_counter() - t_verif_start) * 1000.0, 2)

        # Evaluate all remaining safety barriers (Barriers 3 & 4)
        final_barrier_decision = self.abstainer.evaluate_all_barriers(
            chunks=reranked_chunks,
            grounding_result=grounding_result
        )

        # Multi-factor Confidence Calibration
        top_retrieval_score = reranked_chunks[0].reranker_score if reranked_chunks else 0.0
        confidence_assessment: ConfidenceAssessment = self.confidence_estimator.estimate_confidence(
            retrieval_score=top_retrieval_score,
            grounding_result=grounding_result,
            citation_precision=citation_metric["citation_precision"],
            is_abstained=final_barrier_decision.is_abstained
        )

        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        # Format source chunk payloads
        sources_info: List[SourceDocumentInfo] = [
            SourceDocumentInfo(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_title=c.document_title,
                page_number=c.page_number,
                section_title=c.section_title,
                relevance_score=c.reranker_score,
                channel=str(c.metadata.get("channel", "hybrid")),
                snippet=c.text[:200]
            )
            for c in built_context.chunks
        ]

        # Format citations
        citations_info: List[CitationInfo] = [
            CitationInfo(
                citation_id=cit.citation_id,
                document_id=cit.document_id,
                document_title=cit.document_title,
                page_number=cit.page_number,
                chunk_id=cit.chunk_id,
                text_snippet=cit.text_snippet,
                relevance_score=cit.relevance_score
            )
            for cit in citations
        ]

        # Format claim verifications
        claims_info: List[ClaimVerificationInfo] = [
            ClaimVerificationInfo(
                claim_id=cv.claim_id,
                claim_text=cv.claim_text,
                status=cv.status.value,
                confidence=cv.confidence,
                supporting_citations=cv.supporting_citation_ids,
                reasoning=cv.reasoning
            )
            for cv in grounding_result.claims
        ]

        grounding_meta = GroundingMetadata(
            grounding_score=grounding_result.grounding_score,
            is_grounded=grounding_result.is_grounded,
            total_claims=grounding_result.total_claims,
            supported_claims=grounding_result.supported_claims,
            unsupported_claims=grounding_result.unsupported_claims,
            claims=claims_info
        )

        telemetry = TelemetryMetadata(
            total_latency_ms=total_ms,
            retrieval_latency_ms=retrieval_ms,
            rerank_latency_ms=rerank_ms,
            generation_latency_ms=gen_ms,
            verification_latency_ms=verif_ms,
            prompt_tokens=gen_result.prompt_tokens,
            completion_tokens=gen_result.completion_tokens,
            total_tokens=gen_result.total_tokens,
            tokens_per_second=gen_result.tokens_per_second,
            model_name=gen_result.model_name
        )

        # If abstained, override answer text with safe message
        final_answer = gen_result.content
        if final_barrier_decision.is_abstained:
            final_answer = final_barrier_decision.abstention_message or gen_result.content

        return PetroRAGResponse(
            query_id=req_id,
            query=clean_query,
            intent=intent_name,
            entities=entities_dict,
            answer=final_answer,
            sources=sources_info,
            citations=citations_info,
            grounding=grounding_meta,
            abstention=final_barrier_decision.to_schema(),
            telemetry=telemetry
        )

    def query_sync(
        self,
        request: PetroRAGRequest,
        query_id: Optional[str] = None
    ) -> PetroRAGResponse:
        """Synchronous wrapper for offline evaluation and batch benchmarks."""
        def _run():
            return asyncio.run(self.query(request, query_id=query_id))

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(_run).result()
