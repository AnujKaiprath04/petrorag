"""
Unit Test & Research Experiment: Module 2.12 Cross-Encoder Reranking Verification
Tests:
1. Reranked chunk schema: initial_score, reranker_score, final_rank, and latency.
2. Comparative Research Experiment:
   Hybrid Retrieval vs. Hybrid Retrieval + Cross-Encoder Reranking.
   Measures: Recall@1, Recall@3, Recall@5, MRR, NDCG@3, and Latency (ms).
"""

import math
import time
from typing import Dict, List, Set
import pytest
from src.core.interfaces import RetrievedChunk, RerankedChunk
from src.retrieval.reranker import MockCrossEncoderReranker


# Benchmark candidate pool simulating top 20 hybrid retrieval candidates
RERANKER_TEST_POOL = [
    RetrievedChunk(
        chunk_id="chk_generic_compressor",
        document_id="DOC-COMP-GEN",
        text="Compressor auxiliary systems include cooling water loops and seal oil pumps.",
        score=0.88,
        page_number=3
    ),
    RetrievedChunk(
        chunk_id="chk_target_vibration_causes",
        document_id="DOC-COMP-DIAG",
        text="Centrifugal compressor high vibration causes include shaft mass unbalance, coupling misalignment, and fluid-film bearing whip.",
        score=0.82,  # Sub-optimal initial rank (Rank 2) due to lower term frequency in raw search
        page_number=14
    ),
    RetrievedChunk(
        chunk_id="chk_lube_oil",
        document_id="DOC-COMP-OIL",
        text="Lube oil pressure must be maintained between 2.5 and 3.0 bar during compressor operation.",
        score=0.80,
        page_number=22
    ),
    RetrievedChunk(
        chunk_id="chk_vibration_limits",
        document_id="DOC-COMP-LIMITS",
        text="Overall radial vibration trip setpoint is calibrated at 7.1 mm/s RMS for high-pressure casing.",
        score=0.79,
        page_number=45
    ),
]


@pytest.fixture
def reranker():
    return MockCrossEncoderReranker()


def test_reranker_payload_storage(reranker):
    """
    Verification Gate:
    Confirm reranker stores:
    - initial_score
    - reranker_score
    - final_rank
    - latency in metadata
    """
    query = "Why is centrifugal compressor vibration high?"
    reranked: List[RerankedChunk] = reranker.rerank(query, RERANKER_TEST_POOL, top_k=3)

    assert len(reranked) == 3

    for item in reranked:
        assert item.initial_score is not None
        assert item.reranker_score is not None
        assert item.rank in [1, 2, 3]
        assert "initial_rank" in item.metadata
        assert "final_rank" in item.metadata
        assert "rerank_latency_ms" in item.metadata
        assert item.metadata["rerank_latency_ms"] >= 0.0

    # Top hit must be the exact diagnostic cause chunk promoted by the cross-encoder
    top_hit = reranked[0]
    assert top_hit.chunk_id == "chk_target_vibration_causes"
    assert top_hit.rank == 1
    assert top_hit.metadata["rank_delta"] > 0  # Promoted from initial rank 2 to final rank 1


def _compute_metrics(ranked_ids: List[str], ground_truth: Set[str]):
    recalls = {
        1: 1.0 if any(cid in ground_truth for cid in ranked_ids[:1]) else 0.0,
        3: 1.0 if any(cid in ground_truth for cid in ranked_ids[:3]) else 0.0,
        5: 1.0 if any(cid in ground_truth for cid in ranked_ids[:5]) else 0.0,
    }

    mrr = 0.0
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in ground_truth:
            mrr = 1.0 / rank
            break

    dcg = 0.0
    idcg = 1.0 / math.log2(2)
    for i, cid in enumerate(ranked_ids[:3]):
        if cid in ground_truth:
            dcg += 1.0 / math.log2(i + 2)
            break
    ndcg_3 = dcg / idcg if idcg > 0 else 0.0

    return recalls, mrr, ndcg_3


def test_comparative_reranking_experiment(reranker):
    """
    MAJOR RESEARCH EXPERIMENT (Module 2.12):
    Compare:
    Hybrid Retrieval vs. Hybrid Retrieval + Cross-Encoder Reranking.
    Measures Recall@1, Recall@3, Recall@5, MRR, NDCG@3, and Latency (ms).
    """
    eval_suite = [
        {
            "query": "Centrifugal compressor high vibration causes",
            "candidates": RERANKER_TEST_POOL,
            "target": {"chk_target_vibration_causes"}
        },
        {
            "query": "What is the compressor lube oil operating pressure limit?",
            "candidates": RERANKER_TEST_POOL,
            "target": {"chk_lube_oil"}
        },
        {
            "query": "What is the radial vibration shutdown trip setpoint?",
            "candidates": RERANKER_TEST_POOL,
            "target": {"chk_vibration_limits"}
        },
    ]

    # 1. Baseline: Hybrid Retrieval (Ordered by initial_score)
    start_hybrid = time.perf_counter()
    hybrid_r1, hybrid_r3, hybrid_r5, hybrid_mrrs, hybrid_ndcgs = [], [], [], [], []
    for item in eval_suite:
        # Sort by initial score
        sorted_hybrid = sorted(item["candidates"], key=lambda c: c.score, reverse=True)
        h_ids = [c.chunk_id for c in sorted_hybrid[:3]]
        rec, mrr_val, ndcg_val = _compute_metrics(h_ids, item["target"])
        hybrid_r1.append(rec[1])
        hybrid_r3.append(rec[3])
        hybrid_r5.append(rec[5])
        hybrid_mrrs.append(mrr_val)
        hybrid_ndcgs.append(ndcg_val)
    hybrid_time_ms = (time.perf_counter() - start_hybrid) * 1000.0 / len(eval_suite)

    # 2. Proposed: Hybrid + Reranking
    start_rerank = time.perf_counter()
    rerank_r1, rerank_r3, rerank_r5, rerank_mrrs, rerank_ndcgs = [], [], [], [], []
    for item in eval_suite:
        reranked_chunks = reranker.rerank(item["query"], item["candidates"], top_k=3)
        r_ids = [c.chunk_id for c in reranked_chunks]
        rec, mrr_val, ndcg_val = _compute_metrics(r_ids, item["target"])
        rerank_r1.append(rec[1])
        rerank_r3.append(rec[3])
        rerank_r5.append(rec[5])
        rerank_mrrs.append(mrr_val)
        rerank_ndcgs.append(ndcg_val)
    rerank_time_ms = (time.perf_counter() - start_rerank) * 1000.0 / len(eval_suite)

    avg_hybrid = {
        "Recall@1": sum(hybrid_r1) / len(hybrid_r1),
        "Recall@3": sum(hybrid_r3) / len(hybrid_r3),
        "Recall@5": sum(hybrid_r5) / len(hybrid_r5),
        "MRR": sum(hybrid_mrrs) / len(hybrid_mrrs),
        "NDCG@3": sum(hybrid_ndcgs) / len(hybrid_ndcgs),
        "Latency (ms)": hybrid_time_ms,
    }

    avg_reranked = {
        "Recall@1": sum(rerank_r1) / len(rerank_r1),
        "Recall@3": sum(rerank_r3) / len(rerank_r3),
        "Recall@5": sum(rerank_r5) / len(rerank_r5),
        "MRR": sum(rerank_mrrs) / len(rerank_mrrs),
        "NDCG@3": sum(rerank_ndcgs) / len(rerank_ndcgs),
        "Latency (ms)": rerank_time_ms,
    }

    print("\n" + "=" * 76)
    print("EMPIRICAL RERANKING EXPERIMENT: HYBRID RETRIEVAL vs. HYBRID + RERANKING")
    print(f"{'Metric':<14} | {'Hybrid Retrieval':<18} | {'Hybrid + Reranking':<20} | {'Delta':<10}")
    print("-" * 76)
    for m in ["Recall@1", "Recall@3", "Recall@5", "MRR", "NDCG@3", "Latency (ms)"]:
        h_val = avg_hybrid[m]
        r_val = avg_reranked[m]
        delta = r_val - h_val
        sign = "+" if delta >= 0 else ""
        print(f"{m:<14} | {h_val:<18.4f} | {r_val:<20.4f} | {sign}{delta:.4f}")
    print("=" * 76)

    # Experimental finding verification: Reranking elevates high-relevance evidence to Rank 1
    assert avg_reranked["Recall@1"] >= avg_hybrid["Recall@1"]
    assert avg_reranked["MRR"] >= avg_hybrid["MRR"]
