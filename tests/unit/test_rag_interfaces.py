"""
Unit Test: Module 2.1 RAG Architecture Foundation & Interfaces Verification
Verifies:
1. Core interface ABC contracts and polymorphism.
2. Qdrant store connection, collection access, and lifecycle.
3. Ingestion payload mapping to RetrievedChunk.
4. Metadata integrity preservation (equipment, field, well).
5. Document and page reference preservation.
6. Pydantic schema validation and serialization.
"""

import pytest
from qdrant_client import QdrantClient
from src.core.interfaces import (
    BaseRetriever,
    BaseReranker,
    BaseContextBuilder,
    BaseContextCompressor,
    BasePromptBuilder,
    BaseLLMProvider,
    BaseCitationEngine,
    BaseGroundingEvaluator,
    RetrievedChunk,
    RerankedChunk,
    CompressedChunk,
    BuiltContext,
    PromptBundle,
    GenerationConfig,
    LLMResponse,
    Citation,
    ClaimSupportStatus,
    ClaimVerification,
    GroundingResult,
    RetrievalChannel,
)
from src.retrieval.qdrant_client import QdrantStoreManager


def test_interfaces_are_abstract():
    """Verify that all 8 core interfaces cannot be instantiated directly without implementing abstract methods."""
    interfaces = [
        BaseRetriever,
        BaseReranker,
        BaseContextBuilder,
        BaseContextCompressor,
        BasePromptBuilder,
        BaseLLMProvider,
        BaseCitationEngine,
        BaseGroundingEvaluator,
    ]
    for iface in interfaces:
        with pytest.raises(TypeError):
            iface()


def test_pydantic_schemas_serialization():
    """Verify serialization and validation across all foundational data transfer models."""
    chunk = RetrievedChunk(
        chunk_id="chk-001",
        document_id="DOC-SOP-77",
        document_title="Centrifugal Compressor Operating Manual Rev 2.pdf",
        text="Normal operating discharge pressure is 42.5 bar at 8500 RPM.",
        score=0.912,
        page_number=14,
        section_title="4.1 Operating Parameters",
        metadata={"equipment": "C-101", "field": "North Sea Alpha", "asset": "Platform A"},
        channel=RetrievalChannel.HYBRID
    )

    data = chunk.model_dump()
    assert data["chunk_id"] == "chk-001"
    assert data["page_number"] == 14
    assert data["metadata"]["equipment"] == "C-101"
    assert data["channel"] == "hybrid"

    # Reconstruct from dict
    restored = RetrievedChunk.model_validate(data)
    assert restored == chunk

    # Reranked chunk
    reranked = RerankedChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_title=chunk.document_title,
        text=chunk.text,
        initial_score=chunk.score,
        reranker_score=0.965,
        rank=1,
        page_number=chunk.page_number,
        section_title=chunk.section_title,
        metadata=chunk.metadata
    )
    assert reranked.reranker_score == 0.965
    assert reranked.rank == 1

    # Compressed chunk
    compressed = CompressedChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_title=chunk.document_title,
        original_text=chunk.text,
        compressed_text="Normal operating discharge pressure is 42.5 bar at 8500 RPM.",
        original_tokens=15,
        compressed_tokens=15,
        compression_ratio=1.0,
        preserved_parameters=["42.5 bar", "8500 RPM"],
        retained_sentences_count=1,
        total_sentences_count=1,
        rank=1
    )
    assert compressed.compression_ratio == 1.0
    assert "42.5 bar" in compressed.preserved_parameters

    # Citation
    citation = Citation(
        citation_id="[1]",
        document_id=chunk.document_id,
        document_title=chunk.document_title,
        page_number=14,
        section_title=chunk.section_title,
        chunk_id=chunk.chunk_id,
        text_snippet="Normal operating discharge pressure is 42.5 bar",
        relevance_score=0.965
    )
    assert citation.page_number == 14
    assert citation.citation_id == "[1]"

    # Grounding result
    claim = ClaimVerification(
        claim_id="clm-1",
        claim_text="Discharge pressure is 42.5 bar.",
        status=ClaimSupportStatus.SUPPORTED,
        confidence=0.98,
        supporting_citation_ids=["[1]"],
        supporting_chunk_ids=["chk-001"]
    )
    grounding = GroundingResult(
        grounding_score=1.0,
        total_claims=1,
        supported_claims=1,
        claims=[claim],
        is_grounded=True
    )
    assert grounding.is_grounded is True
    assert grounding.grounding_score == 1.0


def test_qdrant_collection_access_and_retrieval():
    """
    Verification Gate:
    1. Confirm Qdrant collection can be initialized/accessed.
    2. Confirm chunks can be upserted and retrieved.
    3. Confirm metadata remains intact (equipment, field, asset).
    4. Confirm document and page references remain intact.
    """
    client = QdrantClient(location=":memory:")
    manager = QdrantStoreManager(
        client=client,
        collection_name="test_petrorag_knowledge_base",
        vector_size=4
    )

    assert manager.ensure_collection() is True
    assert manager.count() == 0

    # Create test chunks with rich O&G domain metadata and page references
    test_chunks = [
        RetrievedChunk(
            chunk_id="chunk_sep_101",
            document_id="DOC-SEP-2024",
            document_title="Separator Manual.pdf",
            text="Separator S-101 high pressure trip is calibrated to 45.0 bar.",
            score=0.85,
            page_number=32,
            section_title="Emergency Shutdown Limits",
            metadata={"equipment_id": "S-101", "equipment_type": "separator", "field": "Gullfaks"},
            channel=RetrievalChannel.DENSE
        ),
        RetrievedChunk(
            chunk_id="chunk_comp_201",
            document_id="DOC-COMP-2023",
            document_title="Compressor Diagnostics.pdf",
            text="Compressor C-101 bearing vibration threshold is 4.5 mm/s RMS.",
            score=0.78,
            page_number=18,
            section_title="Vibration Diagnostics",
            metadata={"equipment_id": "C-101", "equipment_type": "compressor", "field": "Statfjord"},
            channel=RetrievalChannel.DENSE
        ),
    ]

    dummy_vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]

    # Upsert
    success = manager.upsert_chunks(test_chunks, dummy_vectors)
    assert success is True
    assert manager.count() == 2

    # Query matching first vector
    results = manager.search_vector(query_vector=[0.99, 0.01, 0.0, 0.0], top_k=2)
    assert len(results) == 2

    top_hit = results[0]
    assert top_hit.chunk_id == "chunk_sep_101"
    assert top_hit.document_id == "DOC-SEP-2024"
    assert top_hit.document_title == "Separator Manual.pdf"
    assert top_hit.page_number == 32
    assert top_hit.section_title == "Emergency Shutdown Limits"
    assert top_hit.metadata["equipment_id"] == "S-101"
    assert top_hit.metadata["equipment_type"] == "separator"
    assert top_hit.metadata["field"] == "Gullfaks"
    assert "Separator S-101" in top_hit.text
    assert top_hit.score > 0.90
