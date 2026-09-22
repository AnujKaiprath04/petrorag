"""
PetroRAG Qdrant Client Manager (Module 2.1)
Manages connection to Qdrant vector database (remote, embedded local disk, or in-memory),
ensures collection lifecycle, and maps payloads to standardized RetrievedChunk objects.
"""

import os
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from src.core.config import settings
from src.core.logging import logger
from src.core.interfaces import RetrievedChunk, RetrievalChannel


class QdrantStoreManager:
    """
    Manages Qdrant vector store connection, collection initialization,
    and chunk persistence/retrieval with rich domain metadata.
    """

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        collection_name: Optional[str] = None,
        vector_size: int = 768,
        distance: rest_models.Distance = rest_models.Distance.COSINE
    ):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.vector_size = vector_size
        self.distance = distance

        if client:
            self.client = client
        else:
            self.client = self._init_client()

        self.ensure_collection()

    def _init_client(self) -> QdrantClient:
        """Initialize client: remote server first, embedded disk fallback, or memory."""
        if settings.ENVIRONMENT == "testing" or "PYTEST_CURRENT_TEST" in os.environ:
            storage_path = str(settings.QDRANT_STORAGE_PATH)
            logger.info(f"Testing environment: using local disk Qdrant at {storage_path}")
            return QdrantClient(path=storage_path)

        try:
            if settings.QDRANT_URL and not settings.QDRANT_URL.startswith("local"):
                # Try connecting to Qdrant server with short timeout
                client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                    timeout=2.0
                )
                # Health check ping
                client.get_collections()
                logger.info(f"Connected to remote Qdrant at {settings.QDRANT_URL}")
                return client
        except Exception as e:
            logger.warning(f"Remote Qdrant unavailable ({e}). Falling back to local disk storage.")

        # Local embedded storage fallback
        storage_path = str(settings.QDRANT_STORAGE_PATH)
        logger.info(f"Initializing embedded Qdrant client at {storage_path}")
        return QdrantClient(path=storage_path)

    def ensure_collection(self) -> bool:
        """Verify collection exists; create with default vector params if not."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest_models.VectorParams(
                        size=self.vector_size,
                        distance=self.distance
                    )
                )
                logger.info(f"Created Qdrant collection '{self.collection_name}' (dim={self.vector_size})")
            return True
        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collection '{self.collection_name}': {e}")
            raise

    def count(self) -> int:
        """Return total points in collection."""
        try:
            res = self.client.count(collection_name=self.collection_name)
            return res.count
        except Exception as e:
            logger.error(f"Error counting points in collection '{self.collection_name}': {e}")
            return 0

    def upsert_chunks(
        self,
        chunks: List[RetrievedChunk],
        vectors: List[List[float]]
    ) -> bool:
        """
        Upsert a batch of chunks and their embedding vectors into Qdrant.
        Preserves all document, page, and domain metadata in payload.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"Mismatch between chunks ({len(chunks)}) and vectors ({len(vectors)})")

        import uuid
        points = []
        for chunk, vector in zip(chunks, vectors):
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_title": chunk.document_title,
                "text": chunk.text,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "metadata": chunk.metadata,
                "channel": chunk.channel.value
            }

            # Generate valid Qdrant ID (UUID or int)
            try:
                point_id = int(chunk.chunk_id)
            except (ValueError, TypeError):
                try:
                    point_id = str(uuid.UUID(chunk.chunk_id))
                except (ValueError, TypeError):
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk.chunk_id)))

            points.append(
                rest_models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )

        # Use batch upsert
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return True

    def search_vector(
        self,
        query_vector: List[float],
        top_k: int = 5,
        query_filter: Optional[rest_models.Filter] = None
    ) -> List[RetrievedChunk]:
        """
        Query Qdrant collection using dense vector and return standardized RetrievedChunk objects.
        Supports modern QdrantClient query_points API with fallback.
        """
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k
            )
            hits = response.points
        else:
            # Fallback for older qdrant-client versions
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k
            )

        retrieved: List[RetrievedChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            chunk = RetrievedChunk(
                chunk_id=str(payload.get("chunk_id", hit.id)),
                document_id=str(payload.get("document_id", "unknown")),
                document_title=payload.get("document_title"),
                text=str(payload.get("text", "")),
                score=float(hit.score),
                page_number=payload.get("page_number"),
                section_title=payload.get("section_title"),
                metadata=payload.get("metadata", {}),
                channel=RetrievalChannel.DENSE
            )
            retrieved.append(chunk)

        return retrieved
