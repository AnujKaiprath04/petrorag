"""
PetroRAG Retrieval Package
Exports BM25 lexical retrieval, Qdrant vector retrieval, hybrid fusion, and reranking.
"""

from src.retrieval.qdrant_client import QdrantStoreManager
from src.retrieval.bm25 import BM25Retriever, tokenize_og_text
from src.retrieval.vector import (
    VectorRetriever,
    BaseEmbeddingService,
    SentenceTransformerEmbeddingService,
    MockEmbeddingService,
)
from src.retrieval.metadata_filter import (
    MetadataFilterEngine,
    SUPPORTED_METADATA_KEYS,
)
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.deduplication import RetrievalDeduplicator, DeduplicationResult
from src.retrieval.reranker import CrossEncoderReranker, MockCrossEncoderReranker

__all__ = [
    "QdrantStoreManager",
    "BM25Retriever",
    "tokenize_og_text",
    "VectorRetriever",
    "BaseEmbeddingService",
    "SentenceTransformerEmbeddingService",
    "MockEmbeddingService",
    "MetadataFilterEngine",
    "SUPPORTED_METADATA_KEYS",
    "reciprocal_rank_fusion",
    "HybridRetriever",
    "RetrievalDeduplicator",
    "DeduplicationResult",
    "CrossEncoderReranker",
    "MockCrossEncoderReranker",
]
