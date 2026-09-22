"""
Unit Test & Major Research Experiment: Module 2.10 Hybrid Retrieval & RRF Verification
Executes comparative empirical evaluation:
- Vector RAG vs. BM25 vs. Hybrid RAG (Dense + BM25 + RRF)
- Measures Recall@1, Recall@3, Recall@5, Precision@5, MRR, and NDCG@3.
"""

import math
from typing import Dict, List, Set
import pytest
from qdrant_client import QdrantClient
from src.core.interfaces import RetrievedChunk, RetrievalChannel
from src.retrieval.qdrant_client import QdrantStoreManager
from src.retrieval.vector import VectorRetriever, MockEmbeddingService
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.fusion import reciprocal_rank_fusion


# Curated representative Oil & Gas evaluation corpus (10 chunks)
HYBRID_RESEARCH_CORPUS = [
    RetrievedChunk(
        chunk_id="chunk_api610_pump",
        document_id="DOC-API610",
        document_title="Centrifugal Pumps for Petroleum Industries.pdf",
        text="API 610 Type BB2 between-bearing double suction centrifugal pump specifications and mechanical seal flush plans.",
        page_number=12,
        section_title="Pump Types",
        metadata={"equipment_type": "pump", "standard": "API 610"}
    ),
    RetrievedChunk(
        chunk_id="chunk_comp_vib_causes",
        document_id="DOC-COMP-DIAG",
        document_title="Centrifugal Compressor Vibration Diagnostics.pdf",
        text="Rotor mass unbalance, shaft misalignment, and bearing fluid-film instability cause elevated radial vibration in centrifugal compressors.",
        page_number=45,
        section_title="Vibration Diagnostics",
        metadata={"equipment_type": "compressor"}
    ),
    RetrievedChunk(
        chunk_id="chunk_psv_loto",
        document_id="DOC-SAFE-PSV",
        document_title="Pressure Safety Valve Isolation SOP.pdf",
        text="PSV pressure safety valve LOTO lockout tagout procedure requires car-seal open/closed verification and blind flange installation.",
        page_number=8,
        section_title="LOTO Isolation",
        metadata={"equipment_type": "valve", "safety": "LOTO"}
    ),
    RetrievedChunk(
        chunk_id="chunk_sep_overpressure",
        document_id="DOC-SEP-OPER",
        document_title="Three Phase Production Separator Manual.pdf",
        text="Production separator overpressure trip at 45.0 bar triggers emergency shutdown and automated blowdown valve opening.",
        page_number=33,
        section_title="Shutdown Logic",
        metadata={"equipment_type": "separator"}
    ),
    RetrievedChunk(
        chunk_id="chunk_esp_well",
        document_id="DOC-ESP-RUN",
        document_title="ESP Downhole Operation Guide.pdf",
        text="ESP electric submersible pump downhole frequency variation optimizes bpd production rate and prevents motor overheating.",
        page_number=19,
        section_title="Frequency Control",
        metadata={"equipment_type": "esp", "process": "artificial lift"}
    ),
    RetrievedChunk(
        chunk_id="chunk_gas_export_mmscfd",
        document_id="DOC-GAS-METER",
        document_title="Gas Compression and Export Logs.pdf",
        text="Natural gas export pipeline operational capacity exceeds 450 MMSCFD at 85 bar export header pressure.",
        page_number=4,
        section_title="Export Metering",
        metadata={"unit": "MMSCFD"}
    ),
    RetrievedChunk(
        chunk_id="chunk_comp_surge",
        document_id="DOC-COMP-SURGE",
        document_title="Compressor Antisurge Control Manual.pdf",
        text="Centrifugal compressor aerodynamic surge occurs when gas flow drops below the surge line, causing violent flow reversal.",
        page_number=22,
        section_title="Antisurge",
        metadata={"equipment_type": "compressor"}
    ),
    RetrievedChunk(
        chunk_id="chunk_pump_cavitation",
        document_id="DOC-PUMP-CAVIT",
        document_title="Pump Hydraulic Diagnostics.pdf",
        text="Centrifugal pump suction cavitation and vapor bubble collapse produce acoustic noise and impeller pitting.",
        page_number=15,
        section_title="Cavitation",
        metadata={"equipment_type": "pump"}
    ),
]

# Evaluation queries: spanning exact code queries, paraphrased conceptual queries, and combined queries
HYBRID_EVAL_QUERIES = [
    {
        "query": "API 610 Type BB2 pump specifications",
        "ground_truth": {"chunk_api610_pump"},
        "type": "exact_code"
    },
    {
        "query": "Why does rotating machinery experience elevated radial shaft oscillations?",
        "ground_truth": {"chunk_comp_vib_causes"},
        "type": "conceptual_paraphrase"
    },
    {
        "query": "What are the LOTO steps for pressure safety relief valves?",
        "ground_truth": {"chunk_psv_loto"},
        "type": "combined"
    },
    {
        "query": "What happens when flow drops below the compressor aerodynamic surge boundary?",
        "ground_truth": {"chunk_comp_surge"},
        "type": "conceptual_paraphrase"
    },
    {
        "query": "Electric submersible pump ESP frequency optimization",
        "ground_truth": {"chunk_esp_well"},
        "type": "combined"
    },
    {
        "query": "450 MMSCFD pipeline header operating pressure",
        "ground_truth": {"chunk_gas_export_mmscfd"},
        "type": "exact_code"
    },
]


@pytest.fixture
def hybrid_setup():
    # 1. Setup Qdrant & Vector Retriever
    client = QdrantClient(location=":memory:")
    manager = QdrantStoreManager(
        client=client,
        collection_name="hybrid_research_test",
        vector_size=128
    )
    embedder = MockEmbeddingService(dim=128)
    vectors = embedder.embed_documents([c.text for c in HYBRID_RESEARCH_CORPUS])
    manager.upsert_chunks(HYBRID_RESEARCH_CORPUS, vectors)

    vec_retriever = VectorRetriever(store_manager=manager, embedding_service=embedder)

    # 2. Setup BM25 Retriever
    bm25 = BM25Retriever()
    bm25.index_chunks(HYBRID_RESEARCH_CORPUS)

    # 3. Setup Hybrid Retriever
    hybrid = HybridRetriever(
        vector_retriever=vec_retriever,
        bm25_retriever=bm25,
        dense_weight=0.55,
        sparse_weight=0.45,
        rrf_k=60
    )

    return vec_retriever, bm25, hybrid


def _compute_eval_metrics(ranked_chunks: List[RetrievedChunk], ground_truth: Set[str]):
    ranked_ids = [c.chunk_id for c in ranked_chunks]

    recalls = {
        1: 1.0 if any(cid in ground_truth for cid in ranked_ids[:1]) else 0.0,
        3: 1.0 if any(cid in ground_truth for cid in ranked_ids[:3]) else 0.0,
        5: 1.0 if any(cid in ground_truth for cid in ranked_ids[:5]) else 0.0,
    }

    # Precision@5
    hits_5 = sum(1 for cid in ranked_ids[:5] if cid in ground_truth)
    prec_5 = hits_5 / 5.0

    # MRR
    mrr = 0.0
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in ground_truth:
            mrr = 1.0 / rank
            break

    # NDCG@3
    dcg = 0.0
    idcg = 1.0 / math.log2(2)  # Ideal DCG for 1 target relevant doc at rank 1
    for i, cid in enumerate(ranked_ids[:3]):
        if cid in ground_truth:
            dcg += 1.0 / math.log2(i + 2)
            break
    ndcg_3 = dcg / idcg if idcg > 0 else 0.0

    return recalls, prec_5, mrr, ndcg_3


def test_rrf_scoring_logic():
    """Verify mathematical correctness of Reciprocal Rank Fusion."""
    dense_list = [
        RetrievedChunk(chunk_id="c1", document_id="d1", text="text 1", score=0.9),
        RetrievedChunk(chunk_id="c2", document_id="d2", text="text 2", score=0.8),
    ]
    sparse_list = [
        RetrievedChunk(chunk_id="c2", document_id="d2", text="text 2", score=5.0),
        RetrievedChunk(chunk_id="c3", document_id="d3", text="text 3", score=4.0),
    ]

    fused = reciprocal_rank_fusion(
        ranked_lists={RetrievalChannel.DENSE: dense_list, RetrievalChannel.SPARSE: sparse_list},
        k=60,
        weights={RetrievalChannel.DENSE: 0.55, RetrievalChannel.SPARSE: 0.45},
        top_k=3
    )

    assert len(fused) == 3
    # c2 appears in both channels (rank 2 in dense, rank 1 in sparse)
    # c2 score = 0.55 / (60 + 2) + 0.45 / (60 + 1) = 0.55/62 + 0.45/61 = 0.00887 + 0.00737 = 0.01625
    # c1 score = 0.55 / (60 + 1) = 0.55/61 = 0.00901
    # c2 must rank #1 due to multi-channel consensus!
    assert fused[0].chunk_id == "c2"
    assert fused[0].channel == RetrievalChannel.HYBRID
    assert fused[0].score > fused[1].score


def test_major_research_experiment_comparison(hybrid_setup):
    """
    MAJOR RESEARCH EXPERIMENT (Module 2.10):
    Direct comparison across 3 configurations:
    1. Vector RAG
    2. BM25
    3. Hybrid RAG (Vector + BM25 + RRF)

    Measures Recall@1, Recall@3, Recall@5, Precision@5, MRR, and NDCG@3.
    """
    vec_retriever, bm25_retriever, hybrid_retriever = hybrid_setup

    models = {
        "Vector RAG": vec_retriever,
        "BM25": bm25_retriever,
        "Hybrid RAG": hybrid_retriever,
    }

    results_table: Dict[str, Dict[str, float]] = {}

    for model_name, retriever in models.items():
        r1_list = []
        r3_list = []
        r5_list = []
        p5_list = []
        mrr_list = []
        ndcg_list = []

        for item in HYBRID_EVAL_QUERIES:
            q = item["query"]
            gt = item["ground_truth"]

            hits = retriever.retrieve(q, top_k=5)
            recalls, p5, mrr, ndcg_3 = _compute_eval_metrics(hits, gt)

            r1_list.append(recalls[1])
            r3_list.append(recalls[3])
            r5_list.append(recalls[5])
            p5_list.append(p5)
            mrr_list.append(mrr)
            ndcg_list.append(ndcg_3)

        results_table[model_name] = {
            "Recall@1": sum(r1_list) / len(r1_list),
            "Recall@3": sum(r3_list) / len(r3_list),
            "Recall@5": sum(r5_list) / len(r5_list),
            "Precision@5": sum(p5_list) / len(p5_list),
            "MRR": sum(mrr_list) / len(mrr_list),
            "NDCG@3": sum(ndcg_list) / len(ndcg_list),
        }

    print("\n" + "=" * 78)
    print("MAJOR RESEARCH EXPERIMENT: VECTOR RAG vs. BM25 vs. HYBRID RAG (RRF)")
    print(f"{'Metric':<14} | {'Vector RAG':<16} | {'BM25':<16} | {'Hybrid RAG':<16}")
    print("-" * 78)
    for m in ["Recall@1", "Recall@3", "Recall@5", "Precision@5", "MRR", "NDCG@3"]:
        v_val = results_table["Vector RAG"][m]
        b_val = results_table["BM25"][m]
        h_val = results_table["Hybrid RAG"][m]
        print(f"{m:<14} | {v_val:<16.4f} | {b_val:<16.4f} | {h_val:<16.4f}")
    print("=" * 78)

    # Core Research Hypothesis Validation:
    # Hybrid RAG should equal or surpass both single channels across Recall and MRR
    hybrid_r5 = results_table["Hybrid RAG"]["Recall@5"]
    assert hybrid_r5 >= results_table["Vector RAG"]["Recall@5"]
    assert hybrid_r5 >= results_table["BM25"]["Recall@5"]
    assert hybrid_r5 == 1.0  # Captures 100% of targets across all queries
