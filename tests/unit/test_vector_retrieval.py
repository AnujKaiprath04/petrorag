"""
Unit Test & Benchmark: Module 2.8 Dense Semantic Vector Retrieval Verification
Verifies:
1. Return structure: chunk_id, document_id, text, score, page_number, section_title, metadata.
2. Configurable parameters: top_k, similarity_threshold, metadata_filter, embedding_model.
3. Stable semantic retrieval baseline benchmark measuring Recall@1, Recall@3, Recall@5, and MRR.
"""

import pytest
from qdrant_client import QdrantClient
from src.core.interfaces import RetrievedChunk, RetrievalChannel
from src.retrieval.qdrant_client import QdrantStoreManager
from src.retrieval.vector import VectorRetriever, MockEmbeddingService


# Labeled semantic benchmark dataset for stable baseline measurement
SEMANTIC_BENCHMARK_CORPUS = [
    RetrievedChunk(
        chunk_id="chk_sem_01",
        document_id="DOC-COMP-01",
        document_title="Centrifugal Compressor Operation Manual.pdf",
        text="Rotor unbalance and misalignment trigger high radial vibration in centrifugal compressors.",
        page_number=14,
        section_title="Vibration Causes",
        metadata={"equipment_type": "compressor", "field": "Gullfaks"}
    ),
    RetrievedChunk(
        chunk_id="chk_sem_02",
        document_id="DOC-SEP-02",
        document_title="Three Phase Separator Guidelines.pdf",
        text="High liquid level in the production separator leads to carryover into the gas compression train.",
        page_number=28,
        section_title="Liquid Carryover",
        metadata={"equipment_type": "separator", "field": "Statfjord"}
    ),
    RetrievedChunk(
        chunk_id="chk_sem_03",
        document_id="DOC-PUMP-03",
        document_title="Centrifugal Pump Troubleshooting.pdf",
        text="Insufficient net positive suction head causes severe pump impeller cavitation and erosion.",
        page_number=52,
        section_title="Cavitation",
        metadata={"equipment_type": "pump", "field": "Gullfaks"}
    ),
    RetrievedChunk(
        chunk_id="chk_sem_04",
        document_id="DOC-SAFE-04",
        document_title="Platform Emergency Protocols.pdf",
        text="The emergency shutdown ESD system isolates hydrocarbon flow and initiates rapid depressurization.",
        page_number=9,
        section_title="Emergency Systems",
        metadata={"equipment_type": "safety_system", "field": "Ekofisk"}
    ),
]

SEMANTIC_BENCHMARK_QUERIES = [
    {
        "query": "What causes abnormal radial vibration in rotating compressors?",
        "expected_chunk": "chk_sem_01"
    },
    {
        "query": "How does excessive separator liquid level cause gas carryover?",
        "expected_chunk": "chk_sem_02"
    },
    {
        "query": "Why does low suction pressure trigger pump impeller cavitation?",
        "expected_chunk": "chk_sem_03"
    },
    {
        "query": "What happens during platform emergency depressurization?",
        "expected_chunk": "chk_sem_04"
    },
]


@pytest.fixture
def vector_retriever():
    """Sets up an in-memory Qdrant instance populated with the benchmark corpus."""
    client = QdrantClient(location=":memory:")
    manager = QdrantStoreManager(
        client=client,
        collection_name="benchmark_vector_retrieval",
        vector_size=128
    )
    embedder = MockEmbeddingService(dim=128)

    # Embed and upsert chunks
    texts = [c.text for c in SEMANTIC_BENCHMARK_CORPUS]
    vectors = embedder.embed_documents(texts)
    manager.upsert_chunks(SEMANTIC_BENCHMARK_CORPUS, vectors)

    return VectorRetriever(
        store_manager=manager,
        embedding_service=embedder,
        default_top_k=3
    )


def test_return_payload_structure(vector_retriever):
    """
    Verification Gate:
    Confirm returned chunks contain:
    - chunk_id
    - document_id
    - text
    - score
    - page_number
    - section_title
    - metadata
    """
    results = vector_retriever.retrieve("compressor vibration causes", top_k=2)
    assert len(results) > 0

    hit = results[0]
    assert hit.chunk_id is not None
    assert hit.document_id is not None
    assert isinstance(hit.text, str) and len(hit.text) > 0
    assert isinstance(hit.score, float)
    assert hit.page_number is not None
    assert hit.section_title is not None
    assert isinstance(hit.metadata, dict)
    assert hit.channel == RetrievalChannel.DENSE


def test_configurable_top_k_and_similarity_threshold(vector_retriever):
    """Verify top_k and similarity threshold configurations filter results as expected."""
    # Test top_k=1
    res_k1 = vector_retriever.retrieve("compressor vibration causes", top_k=1)
    assert len(res_k1) == 1

    # Test top_k=3
    res_k3 = vector_retriever.retrieve("compressor vibration causes", top_k=3)
    assert len(res_k3) == 3

    # Test high similarity threshold cutoff
    vector_retriever.similarity_threshold = 0.99
    res_strict = vector_retriever.retrieve("completely unrelated query with no match", top_k=5)
    assert len(res_strict) == 0

    # Reset threshold
    vector_retriever.similarity_threshold = None


def test_configurable_metadata_filtering(vector_retriever):
    """Verify metadata filtering restricts semantic search to target equipment or field."""
    # Query Gullfaks field only
    gullfaks_results = vector_retriever.retrieve(
        "impeller cavitation or compressor vibration",
        top_k=4,
        filters={"field": "Gullfaks"}
    )
    assert len(gullfaks_results) > 0
    for r in gullfaks_results:
        assert r.metadata.get("field") == "Gullfaks"

    # Query compressor equipment_type only
    compressor_results = vector_retriever.retrieve(
        "fluid dynamics and operating parameters",
        top_k=4,
        filters={"equipment_type": "compressor"}
    )
    for r in compressor_results:
        assert r.metadata.get("equipment_type") == "compressor"


def test_semantic_retrieval_baseline_benchmark(vector_retriever):
    """
    Empirical research benchmark:
    Runs semantic queries against corpus and computes baseline Recall@1, Recall@3, Recall@5, and MRR.
    """
    recalls_1 = []
    recalls_3 = []
    recalls_5 = []
    reciprocal_ranks = []

    for item in SEMANTIC_BENCHMARK_QUERIES:
        q = item["query"]
        expected = item["expected_chunk"]

        results = vector_retriever.retrieve(q, top_k=5)
        retrieved_ids = [r.chunk_id for r in results]

        # Recall@1, @3, @5
        recalls_1.append(1.0 if expected in retrieved_ids[:1] else 0.0)
        recalls_3.append(1.0 if expected in retrieved_ids[:3] else 0.0)
        recalls_5.append(1.0 if expected in retrieved_ids[:5] else 0.0)

        # MRR
        rr = 0.0
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid == expected:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    avg_r1 = sum(recalls_1) / len(recalls_1)
    avg_r3 = sum(recalls_3) / len(recalls_3)
    avg_r5 = sum(recalls_5) / len(recalls_5)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print("\n" + "=" * 60)
    print("STABLE SEMANTIC RETRIEVAL BASELINE (MODULE 2.8)")
    print(f"Recall@1: {avg_r1:.4f}")
    print(f"Recall@3: {avg_r3:.4f}")
    print(f"Recall@5: {avg_r5:.4f}")
    print(f"MRR:      {mrr:.4f}")
    print("=" * 60)

    assert avg_r3 >= 0.75
    assert mrr >= 0.75
