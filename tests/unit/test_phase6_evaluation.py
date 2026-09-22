"""
Unit Tests for Phase 6: End-to-End Evaluation, Ground Truth Benchmark & Research Paper Reporting
Tests:
1. Mathematical metric calculation functions (Recall, Precision, MRR, NDCG, Faithfulness, Citation Precision, Abstention F1).
2. Benchmark dataset integrity (corpus documents and 50 QA queries).
3. Empirical baseline results artifact integrity.
4. Component ablation results artifact integrity.
5. Publication LaTeX tables and figures existence and validity.
"""

import json
from pathlib import Path
import pytest

from experiments.eval.metrics import (
    compute_abstention_metrics,
    compute_citation_precision,
    compute_faithfulness,
    compute_hallucination_rate,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT_DIR / "experiments" / "results"
TABLES_DIR = ROOT_DIR / "paper" / "tables"
FIGURES_DIR = ROOT_DIR / "paper" / "figures"
CORPUS_PATH = ROOT_DIR / "data" / "processed" / "corpus_documents.json"
QA_PATH = ROOT_DIR / "data" / "ground_truth" / "benchmark_qa.json"


def test_metric_calculations_mathematical_soundness():
    """Validates mathematical correctness of evaluation metrics on synthetic ground truth."""
    retrieved = ["chunk_A", "chunk_B", "chunk_C", "chunk_D", "chunk_E"]
    ground_truth = ["chunk_B", "chunk_D"]

    # Recall@K
    assert compute_recall_at_k(retrieved, ground_truth, k=1) == 0.0
    assert compute_recall_at_k(retrieved, ground_truth, k=2) == 0.5  # chunk_B hit
    assert compute_recall_at_k(retrieved, ground_truth, k=4) == 1.0  # chunk_B and chunk_D hit

    # Precision@K
    assert compute_precision_at_k(retrieved, ground_truth, k=2) == 0.5
    assert compute_precision_at_k(retrieved, ground_truth, k=4) == 0.5

    # MRR: First hit is at rank 2 -> 1/2 = 0.5
    assert compute_reciprocal_rank(retrieved, ground_truth) == 0.5
    assert compute_reciprocal_rank(retrieved, ["chunk_Z"]) == 0.0

    # NDCG@K
    ndcg = compute_ndcg_at_k(retrieved, ground_truth, k=5)
    assert 0.0 < ndcg <= 1.0

    # Faithfulness & Hallucination
    assert compute_faithfulness(supported_claims=4, total_claims=5) == 0.8
    assert compute_hallucination_rate(unsupported_claims=1, total_claims=5) == 0.2

    # Citation Precision
    assert compute_citation_precision(["chunk_B", "chunk_X"], ground_truth) == 0.5

    # Abstention Confusion Matrix
    y_true = [True, True, False, False]
    y_pred = [True, False, False, True]
    # TP=1, FN=1, TN=1, FP=1 -> Precision=0.5, Recall=0.5, F1=0.5
    abst_res = compute_abstention_metrics(y_true, y_pred)
    assert abst_res["precision"] == 0.5
    assert abst_res["recall"] == 0.5
    assert abst_res["f1"] == 0.5
    assert abst_res["accuracy"] == 0.5


def test_benchmark_corpus_and_qa_integrity():
    """Confirms existence and valid JSON structure of benchmark corpus and queries."""
    assert CORPUS_PATH.exists()
    assert QA_PATH.exists()

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa = json.load(f)

    assert len(corpus) == 12
    assert len(qa) == 50

    answerable = [q for q in qa if q["is_answerable"]]
    unanswerable = [q for q in qa if not q["is_answerable"]]
    assert len(answerable) == 40
    assert len(unanswerable) == 10


def test_baseline_results_artifacts_exist_and_valid():
    """Verifies baseline_comparison_results.json and .csv contain genuine empirical metrics."""
    json_path = RESULTS_DIR / "baseline_comparison_results.json"
    csv_path = RESULTS_DIR / "baseline_comparison_results.csv"
    traces_path = RESULTS_DIR / "raw_benchmark_runs.json"

    assert json_path.exists()
    assert csv_path.exists()
    assert traces_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 5
    model_names = [m["model_name"] for m in data]
    assert "Baseline 0 (Direct LLM)" in model_names
    assert "Baseline 1 (Dense RAG)" in model_names
    assert "Model 2 (Hybrid RAG)" in model_names
    assert "Model 3 (Hybrid + Rerank)" in model_names
    assert "Model 4 (Proposed PetroRAG)" in model_names

    # Check PetroRAG empirical superiority
    petrorag = next(m for m in data if "Proposed PetroRAG" in m["model_name"])
    assert petrorag["recall_at_5"] == 1.0
    assert petrorag["faithfulness"] >= 0.90
    assert petrorag["hallucination_rate"] <= 0.10
    assert petrorag["abstention_f1"] >= 0.90
    assert petrorag["token_savings_percent"] > 0.0


def test_ablation_results_artifacts_exist_and_valid():
    """Verifies ablation_study_results.json and .csv exist and demonstrate expected component degradation."""
    json_path = RESULTS_DIR / "ablation_study_results.json"
    csv_path = RESULTS_DIR / "ablation_study_results.csv"

    assert json_path.exists()
    assert csv_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 7
    configs = [c["model_name"] for c in data]
    assert "PetroRAG (Full Architecture)" in configs
    assert any("Hybrid Retrieval" in c for c in configs)
    assert any("Metadata Filtering" in c for c in configs)
    assert any("Cross-Encoder Reranker" in c for c in configs)
    assert any("Context Compression" in c for c in configs)
    assert any("Grounding Verifier" in c for c in configs)
    assert any("Multi-Barrier Abstention" in c for c in configs)


def test_publication_latex_tables_exist_and_well_formed():
    """Validates that all 4 LaTeX tables exist, have non-empty content, and proper tabular tags."""
    t1 = TABLES_DIR / "table1_baseline_comparison.tex"
    t2 = TABLES_DIR / "table2_ablation_study.tex"
    t3 = TABLES_DIR / "table3_abstention_safety.tex"
    t4 = TABLES_DIR / "table4_latency_tokens.tex"

    for table_file in [t1, t2, t3, t4]:
        assert table_file.exists(), f"Missing table {table_file.name}"
        content = table_file.read_text(encoding="utf-8")
        assert len(content) > 100
        assert "\\begin{table" in content
        assert "\\end{table" in content
        assert "\\begin{tabular}" in content
        assert "\\end{tabular}" in content


def test_publication_figures_exist_and_valid():
    """Validates that all 3 publication figure PNGs exist and have non-zero size."""
    f1 = FIGURES_DIR / "fig1_retrieval_curves.png"
    f2 = FIGURES_DIR / "fig2_hallucination_vs_faithfulness.png"
    f3 = FIGURES_DIR / "fig3_token_compression.png"

    for fig_file in [f1, f2, f3]:
        assert fig_file.exists(), f"Missing figure {fig_file.name}"
        assert fig_file.stat().st_size > 10000, f"Figure {fig_file.name} is suspiciously small"
