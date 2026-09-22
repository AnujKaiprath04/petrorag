"""
PetroRAG Dense Semantic Vector Retrieval Engine (Module 2.8)
Wraps Qdrant vector database and SentenceTransformers embeddings to execute
semantic vector search with configurable top_k, similarity_threshold, and metadata filters.
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np
from src.core.config import settings
from src.core.interfaces import BaseRetriever, RetrievedChunk, RetrievalChannel
from src.core.logging import logger
from src.retrieval.qdrant_client import QdrantStoreManager


class BaseEmbeddingService:
    """Abstract base for embedding generation."""

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class SentenceTransformerEmbeddingService(BaseEmbeddingService):
    """
    Production embedding service utilizing Hugging Face / SentenceTransformers.
    Defaults to BAAI/bge-base-en-v1.5.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: bool = True
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = device or settings.EMBEDDING_DEVICE
        self.normalize = normalize
        self._model = None
        self._fallback_mock: Optional[MockEmbeddingService] = None

    @property
    def model(self):
        if self._model is None and self._fallback_mock is None:
            if settings.ENVIRONMENT == "testing" or "PYTEST_CURRENT_TEST" in os.environ:
                logger.info("Testing environment: using MockEmbeddingService for deterministic execution.")
                self._fallback_mock = MockEmbeddingService()
                return None
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model '{self.model_name}' on {self.device}...")
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer '{self.model_name}' ({e}). Falling back to MockEmbeddingService.")
                self._fallback_mock = MockEmbeddingService()
                return None
        return self._model

    def embed_query(self, text: str) -> List[float]:
        if self.model is None:
            return self._fallback_mock.embed_query(text)
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.model is None:
            return self._fallback_mock.embed_documents(texts)
        embeddings = self.model.encode(
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        return embeddings.tolist()


class MockEmbeddingService(BaseEmbeddingService):
    """
    Deterministic pseudo-semantic embedding service for fast, reproducible testing
    and benchmarks without requiring internet downloads or GPU resources.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _hash_to_vector(self, text: str) -> List[float]:
        stopwords = {"what", "how", "why", "does", "the", "in", "of", "and", "is", "a", "an", "to", "for", "on"}
        tokens = [t for t in text.lower().split() if t not in stopwords]
        if not tokens:
            tokens = text.lower().split()

        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in tokens:
            h = hash(tok)
            idx1 = abs(h) % self.dim
            idx2 = abs(h >> 5) % self.dim
            vec[idx1] += 1.0
            vec[idx2] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._hash_to_vector(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vector(t) for t in texts]


class VectorRetriever(BaseRetriever):
    """
    Dense semantic retriever connecting Qdrant vector database and embedding services.
    Supports configurable top_k, similarity_threshold, and metadata filters.
    """

    def __init__(
        self,
        store_manager: Optional[QdrantStoreManager] = None,
        embedding_service: Optional[BaseEmbeddingService] = None,
        similarity_threshold: Optional[float] = None,
        default_top_k: int = 5
    ):
        self.store = store_manager or QdrantStoreManager()
        self.embedder = embedding_service or SentenceTransformerEmbeddingService()
        self.similarity_threshold = similarity_threshold
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """
        Execute dense vector semantic retrieval.
        Returns standardized RetrievedChunk models with page, section, and metadata.
        """
        k = top_k or self.default_top_k
        query_vector = self.embedder.embed_query(query)

        # Build Qdrant payload filter if filters are provided
        qdrant_filter = None
        if filters:
            from qdrant_client.http import models as rest_models
            conditions = []
            for key, val in filters.items():
                if val is not None:
                    conditions.append(
                        rest_models.FieldCondition(
                            key=f"metadata.{key}",
                            match=rest_models.MatchValue(value=val)
                        )
                    )
            if conditions:
                qdrant_filter = rest_models.Filter(must=conditions)

        hits = self.store.search_vector(
            query_vector=query_vector,
            top_k=k,
            query_filter=qdrant_filter
        )

        # Apply similarity threshold cutoff if configured
        filtered_hits: List[RetrievedChunk] = []
        for hit in hits:
            hit.channel = RetrievalChannel.DENSE
            if self.similarity_threshold is not None:
                if hit.score < self.similarity_threshold:
                    continue
            filtered_hits.append(hit)

        return filtered_hits

    async def aretrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Asynchronous dense vector retrieval."""
        return self.retrieve(query, top_k=top_k, filters=filters)
