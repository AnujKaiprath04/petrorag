"""
Unit Test & Empirical Evaluation: Module 2.5 Query Rewriting Verification
Tests:
1. Exact prompt example generation and semantic fidelity.
2. Invariant: No unsupported entities introduced.
3. Comparative Retrieval Experiment: Original Query vs. Rewritten Query Retrieval
   Measures Recall@1, Recall@3, Recall@5, MRR, and NDCG.
"""

import math
import re
from typing import List, Dict, Set
import pytest
from src.query.rewriter import RuleBasedQueryRewriter, RewrittenQueries
from src.query.intent import QueryIntent


@pytest.fixture
def rewriter():
    return RuleBasedQueryRewriter()


def test_prompt_example_generation(rewriter):
    """
    Exact prompt requirement:
    User: Why is compressor vibration high?
    Generate:
    compressor high vibration causes
    compressor vibration troubleshooting
    compressor abnormal vibration maintenance
    compressor vibration failure modes
    """
    query = "Why is compressor vibration high?"
    result: RewrittenQueries = rewriter.rewrite(query, intent=QueryIntent.TROUBLESHOOTING)

    candidates = result.rewritten_candidates
    assert "compressor high vibration causes" in candidates
    assert "compressor vibration troubleshooting" in candidates
    assert "compressor abnormal vibration maintenance" in candidates
    assert "compressor vibration failure modes" in candidates

    # Verify no unsupported entities were introduced
    for c in candidates:
        assert "pump" not in c
        assert "separator" not in c
        assert "c-101" not in c  # Not present in query
        assert "p-101" not in c


# ==============================================================================
# EMPIRICAL RETRIEVAL BENCHMARK: ORIGINAL VS. REWRITTEN
# ==============================================================================

# Representative technical corpus of 10 O&G document chunks
MINI_CORPUS: Dict[str, str] = {
    "chk_01": "Centrifugal compressor high vibration causes include shaft unbalance, misalignment, and bearing wear.",
    "chk_02": "Compressor vibration troubleshooting procedure requires checking radial vibration probes and FFT spectrum.",
    "chk_03": "Centrifugal pump cavitation troubleshooting: verify NPSH available exceeds NPSH required and clear suction strainers.",
    "chk_04": "Separator S-101 high pressure operating limits: emergency shutdown trip triggers at 45.0 bar.",
    "chk_05": "Compressor abnormal vibration maintenance guide: inspect hydrodynamic sleeve bearings and check coupling alignment.",
    "chk_06": "ESP electric submersible pump maintenance schedule and routine preventive maintenance checks.",
    "chk_07": "LOTO lockout tagout procedure for pressure safety valves PSV-402 and vessel isolation.",
    "chk_08": "Gas lift injection rate optimization to maximize bpd liquid production in well W-12.",
    "chk_09": "Compressor vibration failure modes: mechanical looseness, aerodynamic surge, and acoustic resonance.",
    "chk_10": "API 610 centrifugal pump mechanical seal flush plan 53B operating instructions.",
}

# Test benchmark: queries with target ground truth chunk IDs
EVAL_QUERIES = [
    {
        "query": "Why is compressor vibration high?",
        "ground_truth": {"chk_01", "chk_02", "chk_05", "chk_09"},
        "intent": QueryIntent.TROUBLESHOOTING
    },
    {
        "query": "What should I check when pump cavitation occurs?",
        "ground_truth": {"chk_03"},
        "intent": QueryIntent.TROUBLESHOOTING
    },
    {
        "query": "Where are the operating limits for separator S-101 high pressure?",
        "ground_truth": {"chk_04"},
        "intent": QueryIntent.DOCUMENT_SEARCH
    },
    {
        "query": "Explain the preventive maintenance checklist for ESP pumps.",
        "ground_truth": {"chk_06"},
        "intent": QueryIntent.MAINTENANCE
    },
    {
        "query": "What are the safe isolation steps for PSV lockout tagout?",
        "ground_truth": {"chk_07"},
        "intent": QueryIntent.SAFETY
    },
]


def _simple_lexical_search(query: str, corpus: Dict[str, str], top_k: int = 5) -> List[str]:
    """Simple term-overlap lexical scoring baseline for controlled comparison."""
    query_terms = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower()))
    scores = []
    for doc_id, text in corpus.items():
        doc_terms = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower()))
        overlap = len(query_terms.intersection(doc_terms))
        scores.append((doc_id, overlap))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in scores[:top_k] if score > 0]


def _calculate_metrics(ranked_results: List[str], ground_truth: Set[str], k_values=[1, 3, 5]):
    """Compute Recall@K, MRR, and NDCG@K."""
    # Recall@K
    recalls = {}
    for k in k_values:
        retrieved_at_k = set(ranked_results[:k])
        hits = len(retrieved_at_k.intersection(ground_truth))
        recalls[f"Recall@{k}"] = hits / len(ground_truth) if ground_truth else 0.0

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, doc_id in enumerate(ranked_results, start=1):
        if doc_id in ground_truth:
            mrr = 1.0 / rank
            break

    # NDCG@3
    dcg = 0.0
    idcg = 0.0
    for i, doc_id in enumerate(ranked_results[:3]):
        rel = 1.0 if doc_id in ground_truth else 0.0
        dcg += rel / math.log2(i + 2)

    num_relevant = min(3, len(ground_truth))
    for i in range(num_relevant):
        idcg += 1.0 / math.log2(i + 2)

    ndcg_3 = dcg / idcg if idcg > 0 else 0.0

    return recalls, mrr, ndcg_3


def test_comparative_retrieval_benchmark(rewriter):
    """
    Verification Gate:
    Compare Original Query Retrieval vs. Rewritten Query Retrieval.
    Measure Recall@1, Recall@3, Recall@5, MRR, NDCG@3.
    """
    orig_recalls = {1: [], 3: [], 5: []}
    orig_mrrs = []
    orig_ndcgs = []

    rewritten_recalls = {1: [], 3: [], 5: []}
    rewritten_mrrs = []
    rewritten_ndcgs = []

    for item in EVAL_QUERIES:
        raw_q = item["query"]
        gt = item["ground_truth"]
        intent = item["intent"]

        # 1. Original Query Retrieval
        orig_ranked = _simple_lexical_search(raw_q, MINI_CORPUS, top_k=5)
        rec_o, mrr_o, ndcg_o = _calculate_metrics(orig_ranked, gt)
        for k in [1, 3, 5]:
            orig_recalls[k].append(rec_o[f"Recall@{k}"])
        orig_mrrs.append(mrr_o)
        orig_ndcgs.append(ndcg_o)

        # 2. Rewritten Query Retrieval (using primary rewritten formulation)
        rewritten_obj = rewriter.rewrite(raw_q, intent=intent)
        rewritten_q = rewritten_obj.primary_rewritten
        rewritten_ranked = _simple_lexical_search(rewritten_q, MINI_CORPUS, top_k=5)
        rec_r, mrr_r, ndcg_r = _calculate_metrics(rewritten_ranked, gt)
        for k in [1, 3, 5]:
            rewritten_recalls[k].append(rec_r[f"Recall@{k}"])
        rewritten_mrrs.append(mrr_r)
        rewritten_ndcgs.append(ndcg_r)

    # Average metrics
    avg_orig = {
        "Recall@1": sum(orig_recalls[1]) / len(orig_recalls[1]),
        "Recall@3": sum(orig_recalls[3]) / len(orig_recalls[3]),
        "Recall@5": sum(orig_recalls[5]) / len(orig_recalls[5]),
        "MRR": sum(orig_mrrs) / len(orig_mrrs),
        "NDCG@3": sum(orig_ndcgs) / len(orig_ndcgs),
    }

    avg_rewritten = {
        "Recall@1": sum(rewritten_recalls[1]) / len(rewritten_recalls[1]),
        "Recall@3": sum(rewritten_recalls[3]) / len(rewritten_recalls[3]),
        "Recall@5": sum(rewritten_recalls[5]) / len(rewritten_recalls[5]),
        "MRR": sum(rewritten_mrrs) / len(rewritten_mrrs),
        "NDCG@3": sum(rewritten_ndcgs) / len(rewritten_ndcgs),
    }

    print("\n" + "=" * 70)
    print("EMPIRICAL RETRIEVAL COMPARISON: ORIGINAL QUERY VS. REWRITTEN QUERY")
    print(f"{'Metric':<12} | {'Original Query':<18} | {'Rewritten Query':<18} | {'Delta':<10}")
    print("-" * 70)
    for m in ["Recall@1", "Recall@3", "Recall@5", "MRR", "NDCG@3"]:
        delta = avg_rewritten[m] - avg_orig[m]
        sign = "+" if delta >= 0 else ""
        print(f"{m:<12} | {avg_orig[m]:<18.4f} | {avg_rewritten[m]:<18.4f} | {sign}{delta:.4f}")
    print("=" * 70)

    # Confirm rewriting performs at or above baseline without regression
    assert avg_rewritten["Recall@3"] >= avg_orig["Recall@3"]
    assert avg_rewritten["MRR"] >= avg_orig["MRR"]
