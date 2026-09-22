"""
PetroRAG Hybrid Retrieval Orchestrator (Module 2.10)
Integrates Dense Vector Search, Sparse BM25 Retrieval, and Dynamic Metadata Filtering
via Weighted Reciprocal Rank Fusion (RRF).
"""

from typing import Any, Dict, List, Optional
from src.core.config import settings
from src.core.interfaces import BaseRetriever, RetrievedChunk, RetrievalChannel
from src.core.logging import logger
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.vector import VectorRetriever
from src.retrieval.metadata_filter import MetadataFilterEngine
from src.retrieval.fusion import reciprocal_rank_fusion


class HybridRetriever(BaseRetriever):
    """
    Combines semantic vector search, BM25 keyword search, and metadata constraints.
    Applies Reciprocal Rank Fusion (RRF) to generate unified candidate pools.
    """

    def __init__(
        self,
        vector_retriever: Optional[VectorRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        metadata_engine: Optional[MetadataFilterEngine] = None,
        auto_metadata_filter: bool = False,
        rrf_k: int = settings.RRF_K,
        dense_weight: float = settings.DENSE_WEIGHT,
        sparse_weight: float = settings.SPARSE_WEIGHT,
        candidate_pool_size: int = settings.RETRIEVAL_CANDIDATE_POOL_SIZE
    ):
        self.vector_retriever = vector_retriever or VectorRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.metadata_engine = metadata_engine or MetadataFilterEngine()
        self.auto_metadata_filter = auto_metadata_filter
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.candidate_pool_size = candidate_pool_size

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """
        Execute Hybrid Retrieval Pipeline:
        Query -> Vector Search + BM25 Search (under Metadata Constraints) -> RRF Fusion -> Candidate Set.
        """
        active_filters = filters
        if active_filters is None and self.auto_metadata_filter and self.metadata_engine:
            active_filters = self.metadata_engine.build_filter(query)

        k_pool = max(top_k * 3, self.candidate_pool_size)

        # 1. Channel 1: Dense Vector Retrieval
        dense_hits = self.vector_retriever.retrieve(
            query=query,
            top_k=k_pool,
            filters=active_filters
        )

        # 2. Channel 2: Sparse BM25 Retrieval
        sparse_hits = self.bm25_retriever.retrieve(
            query=query,
            top_k=k_pool,
            filters=active_filters
        )

        # Fallback if both channels produced 0 hits with active_filters (adaptive relaxation)
        if not dense_hits and not sparse_hits and active_filters:
            logger.info("Hybrid retrieval yielded 0 hits with strict filters. Retrying without filters.")
            dense_hits = self.vector_retriever.retrieve(query=query, top_k=k_pool, filters=None)
            sparse_hits = self.bm25_retriever.retrieve(query=query, top_k=k_pool, filters=None)

        ranked_lists = {
            RetrievalChannel.DENSE: dense_hits,
            RetrievalChannel.SPARSE: sparse_hits,
        }

        weights = {
            RetrievalChannel.DENSE: self.dense_weight,
            RetrievalChannel.SPARSE: self.sparse_weight,
        }

        # 3. Channel Fusion: Reciprocal Rank Fusion (RRF)
        fused_candidates = reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            k=self.rrf_k,
            weights=weights,
            top_k=top_k
        )

        return fused_candidates

    async def aretrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Asynchronous hybrid retrieval execution."""
        return self.retrieve(query, top_k=top_k, filters=filters)
