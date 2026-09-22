"""
PetroRAG Empirical Evaluation Metrics Engine (Module 2.32)
Strict mathematical implementations of standard Information Retrieval (IR)
and RAG generation fidelity metrics for oilfield benchmarking.
"""

import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


def compute_recall_at_k(
    retrieved_chunk_ids: List[str],
    ground_truth_chunk_ids: List[str],
    k: int
) -> float:
    """
    Recall@K = |Retrieved_K ∩ Relevant| / |Relevant|
    """
    if not ground_truth_chunk_ids:
        return 0.0
    top_k = retrieved_chunk_ids[:k]
    hits = len(set(top_k).intersection(ground_truth_chunk_ids))
    return round(hits / len(ground_truth_chunk_ids), 4)


def compute_precision_at_k(
    retrieved_chunk_ids: List[str],
    ground_truth_chunk_ids: List[str],
    k: int
) -> float:
    """
    Precision@K = |Retrieved_K ∩ Relevant| / K
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_chunk_ids[:k]
    hits = len(set(top_k).intersection(ground_truth_chunk_ids))
    return round(hits / k, 4)


def compute_reciprocal_rank(
    retrieved_chunk_ids: List[str],
    ground_truth_chunk_ids: List[str]
) -> float:
    """
    Reciprocal Rank (RR) = 1 / rank of the first relevant document retrieved.
    Returns 0.0 if no relevant document appears in the retrieved list.
    """
    gt_set = set(ground_truth_chunk_ids)
    for rank, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid in gt_set:
            return round(1.0 / rank, 4)
    return 0.0


def compute_ndcg_at_k(
    retrieved_chunk_ids: List[str],
    ground_truth_chunk_ids: List[str],
    k: int
) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance:
    DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
    IDCG@K = sum_{i=1}^{min(K, |Relevant|)} (1 / log2(i + 1))
    NDCG@K = DCG@K / IDCG@K
    """
    if not ground_truth_chunk_ids or k <= 0:
        return 0.0

    gt_set = set(ground_truth_chunk_ids)
    dcg = 0.0
    for rank, cid in enumerate(retrieved_chunk_ids[:k], start=1):
        rel = 1.0 if cid in gt_set else 0.0
        if rel > 0.0:
            dcg += rel / math.log2(rank + 1)

    max_possible_hits = min(k, len(gt_set))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, max_possible_hits + 1))

    if idcg <= 0.0:
        return 0.0
    return round(dcg / idcg, 4)


def compute_faithfulness(
    supported_claims: int,
    total_claims: int
) -> float:
    """
    Faithfulness = Supported Claims / Total Claims
    Returns 1.0 if zero claims detected (e.g. valid abstention).
    """
    if total_claims <= 0:
        return 1.0
    return round(supported_claims / total_claims, 4)


def compute_citation_precision(
    cited_chunk_ids: List[str],
    ground_truth_chunk_ids: List[str]
) -> float:
    """
    Citation Precision = |Cited_Chunks ∩ Ground_Truth_Chunks| / |Cited_Chunks|
    Returns 1.0 if ground truth is empty and citations empty; 0.0 if citations without relevance.
    """
    if not cited_chunk_ids:
        return 0.0
    hits = len(set(cited_chunk_ids).intersection(ground_truth_chunk_ids))
    return round(hits / len(cited_chunk_ids), 4)


def compute_hallucination_rate(
    unsupported_claims: int,
    total_claims: int
) -> float:
    """
    Hallucination Rate = Unsupported Claims / Total Claims
    """
    if total_claims <= 0:
        return 0.0
    return round(unsupported_claims / total_claims, 4)


def compute_abstention_metrics(
    y_true_unanswerable: List[bool],
    y_pred_abstained: List[bool]
) -> Dict[str, float]:
    """
    Computes confusion matrix and F1 for system abstention decisions:
    - Positive Class: Query is unanswerable / adversarial / out-of-scope.
    - True Positive (TP): Truly unanswerable AND model abstained.
    - False Positive (FP): Actually answerable BUT model erroneously abstained.
    - False Negative (FN): Truly unanswerable BUT model attempted an answer (hallucination hazard).
    - True Negative (TN): Actually answerable AND model answered.
    """
    if len(y_true_unanswerable) != len(y_pred_abstained):
        raise ValueError("Length mismatch between true labels and predictions")

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for yt, yp in zip(y_true_unanswerable, y_pred_abstained):
        if yt and yp:
            tp += 1
        elif not yt and yp:
            fp += 1
        elif yt and not yp:
            fn += 1
        else:
            tn += 1

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    total = len(y_true_unanswerable)
    accuracy = ((tp + tn) / total) if total > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "safety_rate": round(recall, 4)  # Safety compliance: % of hazards safely blocked
    }


class BenchmarkRunMetrics(BaseModel):
    """Aggregated empirical benchmark metrics across a query test suite."""
    model_name: str
    total_queries: int
    answerable_queries: int
    unanswerable_queries: int

    # Retrieval Metrics (evaluated over answerable queries)
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0

    # Generation & Grounding Metrics
    faithfulness: float = 0.0
    citation_precision: float = 0.0
    hallucination_rate: float = 0.0

    # Safety & Abstention Metrics
    abstention_precision: float = 0.0
    abstention_recall: float = 0.0
    abstention_f1: float = 0.0

    # Performance & Cost Telemetry
    mean_latency_ms: float = 0.0
    mean_prompt_tokens: float = 0.0
    mean_completion_tokens: float = 0.0
    mean_total_tokens: float = 0.0
    token_savings_percent: float = 0.0
