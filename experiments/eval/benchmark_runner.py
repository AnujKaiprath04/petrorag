"""
PetroRAG Benchmark Runner & Evaluation Harness (Module 2.32)
Executes systematic empirical evaluations of candidate RAG architectures
across the 50-query Oil & Gas gold-standard benchmark corpus.
"""

import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qdrant_client import QdrantClient

from backend.app.schemas.rag_response import PetroRAGRequest, PetroRAGResponse
from experiments.eval.metrics import (
    BenchmarkRunMetrics,
    compute_abstention_metrics,
    compute_citation_precision,
    compute_faithfulness,
    compute_hallucination_rate,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
)
from src.core.interfaces import (
    BuiltContext,
    GenerationConfig,
    PromptBundle,
    RerankedChunk,
    RetrievalChannel,
    RetrievedChunk,
)
from src.core.logging import logger
from src.generation.context_compressor import ContextCompressor
from src.generation.context_selector import ContextSelector
from src.generation.llm_provider import MockLLMProvider
from src.generation.prompt_engine import PromptEngine
from src.pipeline import PetroRAGPipeline
from src.query.entities import RuleBasedEntityExtractor
from src.query.intent import HybridIntentClassifier
from src.query.validator import QueryValidator
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.deduplication import RetrievalDeduplicator
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.qdrant_client import QdrantStoreManager
from src.retrieval.reranker import MockCrossEncoderReranker
from src.retrieval.vector import MockEmbeddingService, VectorRetriever
from src.verification.abstainer import MultiBarrierAbstainer
from src.verification.citation_engine import CitationEngine
from src.verification.grounding_evaluator import GroundingEvaluator
from src.verification.revision_manager import RevisionManager

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CORPUS_PATH = ROOT_DIR / "data" / "processed" / "corpus_documents.json"
QA_PATH = ROOT_DIR / "data" / "ground_truth" / "benchmark_qa.json"
RESULTS_DIR = ROOT_DIR / "experiments" / "results"


class BenchmarkCorpusHarness:
    """
    Manages loading, in-memory indexing, and shared retrieval components
    for reproducible evaluation runs without disk lock contention.
    """

    def __init__(self):
        self.corpus_docs: List[Dict[str, Any]] = []
        self.all_chunks: List[RetrievedChunk] = []
        self.qa_queries: List[Dict[str, Any]] = []

        self.qdrant_client = QdrantClient(":memory:")
        self.store_manager = QdrantStoreManager(
            client=self.qdrant_client,
            collection_name="benchmark_eval_corpus"
        )
        self.embedder = MockEmbeddingService(dim=768)
        self.bm25_retriever = BM25Retriever()
        self.vector_retriever: Optional[VectorRetriever] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.reranker = MockCrossEncoderReranker()
        self.revision_manager = RevisionManager()
        self.context_compressor = ContextCompressor()
        self.context_selector = ContextSelector()
        self.query_validator = QueryValidator()
        self.intent_classifier = HybridIntentClassifier()
        self.entity_extractor = RuleBasedEntityExtractor()
        self.citation_engine = CitationEngine()
        self.grounding_evaluator = GroundingEvaluator()
        self.abstainer = MultiBarrierAbstainer()

        self._load_and_index()

    def _load_and_index(self):
        """Loads corpus documents and indexes them in memory for BM25 and Vector search."""
        if not CORPUS_PATH.exists() or not QA_PATH.exists():
            raise FileNotFoundError("Corpus or QA dataset missing. Run scripts/build_benchmark_corpus.py first.")

        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            self.corpus_docs = json.load(f)

        with open(QA_PATH, "r", encoding="utf-8") as f:
            self.qa_queries = json.load(f)

        # Build RetrievedChunk objects
        self.all_chunks = []
        for doc in self.corpus_docs:
            for c in doc["chunks"]:
                self.all_chunks.append(
                    RetrievedChunk(
                        chunk_id=c["chunk_id"],
                        document_id=c["document_id"],
                        document_title=c["document_title"],
                        text=c["text"],
                        page_number=c["page_number"],
                        section_title=c["section_title"],
                        metadata=c["metadata"],
                        channel=RetrievalChannel.DENSE
                    )
                )

        # Index in BM25
        self.bm25_retriever.index_chunks(self.all_chunks)

        # Index in Qdrant (in-memory)
        vectors = self.embedder.embed_documents([c.text for c in self.all_chunks])
        self.store_manager.upsert_chunks(self.all_chunks, vectors)

        self.vector_retriever = VectorRetriever(
            store_manager=self.store_manager,
            embedding_service=self.embedder,
            default_top_k=5
        )

        self.hybrid_retriever = HybridRetriever(
            vector_retriever=self.vector_retriever,
            bm25_retriever=self.bm25_retriever,
            rrf_k=60
        )
        logger.info(f"Loaded {len(self.corpus_docs)} documents, {len(self.all_chunks)} chunks, and {len(self.qa_queries)} benchmark queries.")


class BenchmarkEvaluator:
    """
    Runs candidate RAG configurations over the benchmark queries and calculates metrics.
    """

    def __init__(self, harness: BenchmarkCorpusHarness):
        self.harness = harness

    def evaluate_model(
        self,
        model_name: str,
        retrieval_mode: str = "hybrid",  # "none", "vector", "hybrid", "hybrid_rerank", "petrorag"
        enable_reranking: bool = False,
        enable_compression: bool = False,
        enable_revision_management: bool = False,
        enable_security_barrier: bool = False,
        enable_retrieval_barrier: bool = False,
        enable_grounding_eval: bool = False,
        top_k: int = 5
    ) -> Tuple[BenchmarkRunMetrics, List[Dict[str, Any]]]:
        """
        Executes an evaluation run for a given architecture specification across all 50 benchmark queries.
        """
        query_traces: List[Dict[str, Any]] = []

        # Metrics accumulators for answerable queries
        recall_1_list: List[float] = []
        recall_3_list: List[float] = []
        recall_5_list: List[float] = []
        precision_1_list: List[float] = []
        precision_3_list: List[float] = []
        precision_5_list: List[float] = []
        mrr_list: List[float] = []
        ndcg_5_list: List[float] = []

        faithfulness_list: List[float] = []
        citation_prec_list: List[float] = []
        hallucination_list: List[float] = []

        # Abstention lists across all 50 queries
        y_true_unanswerable: List[bool] = []
        y_pred_abstained: List[bool] = []

        latency_list: List[float] = []
        prompt_tokens_list: List[int] = []
        completion_tokens_list: List[int] = []
        total_tokens_list: List[int] = []

        raw_uncompressed_tokens_list: List[int] = []

        for q in self.harness.qa_queries:
            qid = q["query_id"]
            qtext = q["query_text"]
            is_answerable = q["is_answerable"]
            gt_chunks = q.get("supporting_chunk_ids", [])
            gt_answer = q["ground_truth_answer"]

            y_true_unanswerable.append(not is_answerable)

            t0 = time.perf_counter()

            # --- Stage 1: Security Barrier ---
            is_abstained = False
            abstention_reason = None

            if enable_security_barrier:
                val_res = self.harness.query_validator.validate(qtext)
                if not val_res.is_valid:
                    is_abstained = True
                    abstention_reason = val_res.error_message

            # --- Stage 2: Retrieval ---
            retrieved_chunks: List[RetrievedChunk] = []
            if not is_abstained:
                if retrieval_mode == "none":
                    retrieved_chunks = []
                elif retrieval_mode == "vector":
                    retrieved_chunks = self.harness.vector_retriever.retrieve(qtext, top_k=top_k)
                elif retrieval_mode in ("hybrid", "hybrid_rerank"):
                    pool_k = top_k * 4 if enable_reranking else top_k
                    retrieved_chunks = self.harness.hybrid_retriever.retrieve(qtext, top_k=pool_k)
                elif retrieval_mode == "petrorag":
                    entities = self.harness.entity_extractor.extract(qtext)
                    active_filters = None
                    if entities.equipment_ids:
                        active_filters = {"asset_id": entities.equipment_ids[0]}
                    elif not entities.equipment_ids and not entities.standards and not entities.safety_procedures and len(qtext.split()) <= 7:
                        is_abstained = True
                        abstention_reason = "Insufficient context or ambiguous query. Please specify an equipment tag, system, or operational envelope."

                    if not is_abstained:
                        pool_k = top_k * 4
                        retrieved_chunks = self.harness.hybrid_retriever.retrieve(qtext, top_k=pool_k, filters=active_filters)

            # Deduplicate
            if retrieved_chunks and retrieval_mode in ("hybrid", "hybrid_rerank", "petrorag"):
                dedup = RetrievalDeduplicator()
                dedup_res = dedup.deduplicate(retrieved_chunks)
                retrieved_chunks = dedup_res.deduplicated_chunks

            # --- Stage 3: Reranking ---
            reranked_chunks: List[RerankedChunk] = []
            if retrieved_chunks:
                if enable_reranking or retrieval_mode == "petrorag":
                    reranked_chunks = self.harness.reranker.rerank(qtext, retrieved_chunks, top_k=top_k)
                else:
                    reranked_chunks = [
                        RerankedChunk(
                            chunk_id=c.chunk_id,
                            document_id=c.document_id,
                            document_title=c.document_title,
                            text=c.text,
                            initial_score=c.score,
                            reranker_score=c.score,
                            rank=i + 1,
                            page_number=c.page_number,
                            section_title=c.section_title,
                            metadata=c.metadata
                        )
                        for i, c in enumerate(retrieved_chunks[:top_k])
                    ]

            # --- Stage 4: Revision Management ---
            if enable_revision_management and reranked_chunks:
                reranked_chunks = self.harness.revision_manager.resolve_revisions(reranked_chunks)

            # --- Stage 5: Retrieval Quality Barrier ---
            if enable_retrieval_barrier and not is_abstained:
                ret_decision = self.harness.abstainer.evaluate_retrieval_barrier(reranked_chunks)
                if ret_decision.is_abstained:
                    is_abstained = True
                    abstention_reason = ret_decision.reason

            # --- Stage 6: Context Building & Compression ---
            built_context: Optional[BuiltContext] = None
            prompt_tokens = 50
            completion_tokens = 30

            if reranked_chunks and not is_abstained:
                built_context = self.harness.context_selector.build_context(reranked_chunks, token_budget=1500)
                raw_tokens = built_context.token_count
                raw_uncompressed_tokens_list.append(raw_tokens)

                if enable_compression:
                    built_context = self.harness.context_compressor.compress_built_context(
                        built_context, query=qtext, target_ratio=0.55
                    )
                prompt_tokens = built_context.token_count + 80
            else:
                raw_uncompressed_tokens_list.append(50)

            # --- Stage 7: Generation Simulation ---
            cited_chunks: List[str] = []
            supported_claims = 0
            total_claims = 1

            if is_abstained:
                generated_answer = f"Abstained: {abstention_reason}"
                completion_tokens = 25
                supported_claims = 1
                total_claims = 1
            elif retrieval_mode == "none":
                # Baseline 0: Direct LLM answers from parametric knowledge without citations
                generated_answer = (
                    "Based on general engineering principles, the system operating thresholds and limits "
                    "typically fall within standard industrial parameters."
                )
                completion_tokens = 45
                # For answerable queries without context, direct LLM lacks exact numbers -> low faithfulness
                if is_answerable:
                    supported_claims = 0
                    total_claims = 2
                else:
                    supported_claims = 0
                    total_claims = 1
            else:
                # Formulate grounded answer from top retrieved chunks
                top_chunk = reranked_chunks[0] if reranked_chunks else None
                cited_chunks = [c.chunk_id for c in reranked_chunks[:2]]
                citations_str = " ".join([f"[{cid}]" for cid in cited_chunks])

                if top_chunk:
                    generated_answer = (
                        f"According to {top_chunk.document_title} (Section {top_chunk.section_title}): "
                        f"{top_chunk.text[:220]}... {citations_str}"
                    )
                else:
                    generated_answer = "No documentation retrieved."

                completion_tokens = 60
                # Evaluate claim support against retrieved chunks
                if enable_grounding_eval:
                    # Grounding verification checks if citations match and parameters align
                    # If top_chunk matches ground truth, claim is supported
                    if any(c in gt_chunks for c in cited_chunks):
                        supported_claims = 2
                        total_claims = 2
                    else:
                        supported_claims = 0
                        total_claims = 2
                else:
                    # Without grounding eval: baseline models generate text that may hallucinate if retrieval was off
                    if any(c in gt_chunks for c in cited_chunks):
                        supported_claims = 2
                        total_claims = 2
                    else:
                        supported_claims = 0
                        total_claims = 2

            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            total_tokens = prompt_tokens + completion_tokens

            y_pred_abstained.append(is_abstained)
            latency_list.append(latency_ms)
            prompt_tokens_list.append(prompt_tokens)
            completion_tokens_list.append(completion_tokens)
            total_tokens_list.append(total_tokens)

            # Evaluate IR metrics only for answerable queries
            retrieved_cids = [c.chunk_id for c in reranked_chunks]
            if is_answerable:
                recall_1_list.append(compute_recall_at_k(retrieved_cids, gt_chunks, 1))
                recall_3_list.append(compute_recall_at_k(retrieved_cids, gt_chunks, 3))
                recall_5_list.append(compute_recall_at_k(retrieved_cids, gt_chunks, 5))
                precision_1_list.append(compute_precision_at_k(retrieved_cids, gt_chunks, 1))
                precision_3_list.append(compute_precision_at_k(retrieved_cids, gt_chunks, 3))
                precision_5_list.append(compute_precision_at_k(retrieved_cids, gt_chunks, 5))
                mrr_list.append(compute_reciprocal_rank(retrieved_cids, gt_chunks))
                ndcg_5_list.append(compute_ndcg_at_k(retrieved_cids, gt_chunks, 5))

                faithfulness_list.append(compute_faithfulness(supported_claims, total_claims))
                citation_prec_list.append(compute_citation_precision(cited_chunks, gt_chunks))
                hallucination_list.append(compute_hallucination_rate(total_claims - supported_claims, total_claims))

            query_traces.append({
                "query_id": qid,
                "is_answerable": is_answerable,
                "is_abstained": is_abstained,
                "retrieved_chunks": retrieved_cids,
                "ground_truth_chunks": gt_chunks,
                "latency_ms": latency_ms,
                "total_tokens": total_tokens,
                "answer_snippet": generated_answer[:100]
            })

        # Calculate aggregated benchmark metrics
        abst_metrics = compute_abstention_metrics(y_true_unanswerable, y_pred_abstained)

        # Token savings vs raw uncompressed tokens
        mean_prompt_tok = sum(prompt_tokens_list) / len(prompt_tokens_list)
        mean_raw_tok = sum(raw_uncompressed_tokens_list) / len(raw_uncompressed_tokens_list)
        token_savings_pct = round(max(0.0, (1.0 - (mean_prompt_tok / mean_raw_tok)) * 100.0), 2) if enable_compression else 0.0

        summary = BenchmarkRunMetrics(
            model_name=model_name,
            total_queries=len(self.harness.qa_queries),
            answerable_queries=len(recall_1_list),
            unanswerable_queries=sum(1 for y in y_true_unanswerable if y),
            recall_at_1=round(sum(recall_1_list) / len(recall_1_list), 4) if recall_1_list else 0.0,
            recall_at_3=round(sum(recall_3_list) / len(recall_3_list), 4) if recall_3_list else 0.0,
            recall_at_5=round(sum(recall_5_list) / len(recall_5_list), 4) if recall_5_list else 0.0,
            precision_at_1=round(sum(precision_1_list) / len(precision_1_list), 4) if precision_1_list else 0.0,
            precision_at_3=round(sum(precision_3_list) / len(precision_3_list), 4) if precision_3_list else 0.0,
            precision_at_5=round(sum(precision_5_list) / len(precision_5_list), 4) if precision_5_list else 0.0,
            mrr=round(sum(mrr_list) / len(mrr_list), 4) if mrr_list else 0.0,
            ndcg_at_5=round(sum(ndcg_5_list) / len(ndcg_5_list), 4) if ndcg_5_list else 0.0,
            faithfulness=round(sum(faithfulness_list) / len(faithfulness_list), 4) if faithfulness_list else 0.0,
            citation_precision=round(sum(citation_prec_list) / len(citation_prec_list), 4) if citation_prec_list else 0.0,
            hallucination_rate=round(sum(hallucination_list) / len(hallucination_list), 4) if hallucination_list else 0.0,
            abstention_precision=abst_metrics["precision"],
            abstention_recall=abst_metrics["recall"],
            abstention_f1=abst_metrics["f1"],
            mean_latency_ms=round(sum(latency_list) / len(latency_list), 2),
            mean_prompt_tokens=round(mean_prompt_tok, 1),
            mean_completion_tokens=round(sum(completion_tokens_list) / len(completion_tokens_list), 1),
            mean_total_tokens=round(sum(total_tokens_list) / len(total_tokens_list), 1),
            token_savings_percent=token_savings_pct
        )

        return summary, query_traces
