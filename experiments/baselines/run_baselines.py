"""
Progressive Architectural Baseline Experiments (Module 2.32)
Runs all 5 progressive architectures against the gold-standard benchmark corpus:
1. Baseline 0: Direct LLM (Zero Retrieval)
2. Baseline 1: Dense Semantic Vector RAG
3. Model 2: Hybrid RAG (Dense Vector + BM25 Lexical via RRF)
4. Model 3: Hybrid RAG + Cross-Encoder Reranking
5. Model 4: Proposed PetroRAG (Full Architecture)

Saves results to experiments/results/baseline_comparison_results.json and .csv
"""

import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.eval.benchmark_runner import BenchmarkCorpusHarness, BenchmarkEvaluator
from experiments.eval.metrics import BenchmarkRunMetrics
from src.core.logging import logger

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_baseline_experiments() -> List[BenchmarkRunMetrics]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PetroRAG Phase 6: Executing Progressive Architectural Baselines")
    print("Authoritative Oil & Gas Benchmark Corpus (50 Gold Queries)")
    print("=" * 80)

    harness = BenchmarkCorpusHarness()
    evaluator = BenchmarkEvaluator(harness)

    architectures = [
        {
            "name": "Baseline 0 (Direct LLM)",
            "config": {
                "retrieval_mode": "none",
                "enable_reranking": False,
                "enable_compression": False,
                "enable_revision_management": False,
                "enable_security_barrier": False,
                "enable_retrieval_barrier": False,
                "enable_grounding_eval": False
            }
        },
        {
            "name": "Baseline 1 (Dense RAG)",
            "config": {
                "retrieval_mode": "vector",
                "enable_reranking": False,
                "enable_compression": False,
                "enable_revision_management": False,
                "enable_security_barrier": False,
                "enable_retrieval_barrier": False,
                "enable_grounding_eval": False
            }
        },
        {
            "name": "Model 2 (Hybrid RAG)",
            "config": {
                "retrieval_mode": "hybrid",
                "enable_reranking": False,
                "enable_compression": False,
                "enable_revision_management": False,
                "enable_security_barrier": False,
                "enable_retrieval_barrier": False,
                "enable_grounding_eval": False
            }
        },
        {
            "name": "Model 3 (Hybrid + Rerank)",
            "config": {
                "retrieval_mode": "hybrid_rerank",
                "enable_reranking": True,
                "enable_compression": False,
                "enable_revision_management": False,
                "enable_security_barrier": False,
                "enable_retrieval_barrier": False,
                "enable_grounding_eval": False
            }
        },
        {
            "name": "Model 4 (Proposed PetroRAG)",
            "config": {
                "retrieval_mode": "petrorag",
                "enable_reranking": True,
                "enable_compression": True,
                "enable_revision_management": True,
                "enable_security_barrier": True,
                "enable_retrieval_barrier": True,
                "enable_grounding_eval": True
            }
        }
    ]

    all_summaries: List[BenchmarkRunMetrics] = []
    all_raw_traces: Dict[str, Any] = {}

    for arch in architectures:
        name = arch["name"]
        cfg = arch["config"]
        print(f"\n--> Running Evaluation for: {name} ...")
        summary, traces = evaluator.evaluate_model(model_name=name, **cfg)
        all_summaries.append(summary)
        all_raw_traces[name] = traces

        print(
            f"    Recall@5: {summary.recall_at_5:.4f} | MRR: {summary.mrr:.4f} | "
            f"NDCG@5: {summary.ndcg_at_5:.4f} | Faithfulness: {summary.faithfulness:.4f} | "
            f"Hallucination: {summary.hallucination_rate:.4f} | Abstention F1: {summary.abstention_f1:.4f} | "
            f"Latency: {summary.mean_latency_ms:.1f}ms | Token Savings: {summary.token_savings_percent:.1f}%"
        )

    # Save to JSON
    json_path = RESULTS_DIR / "baseline_comparison_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([s.model_dump() for s in all_summaries], f, indent=2)
    print(f"\n[OK] Saved baseline summaries to {json_path}")

    # Save raw traces
    traces_path = RESULTS_DIR / "raw_benchmark_runs.json"
    with open(traces_path, "w", encoding="utf-8") as f:
        json.dump(all_raw_traces, f, indent=2)
    print(f"[OK] Saved raw query execution traces to {traces_path}")

    # Save to CSV
    csv_path = RESULTS_DIR / "baseline_comparison_results.csv"
    fieldnames = list(all_summaries[0].model_dump().keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_summaries:
            writer.writerow(s.model_dump())
    print(f"[OK] Saved baseline comparison CSV to {csv_path}")

    return all_summaries


if __name__ == "__main__":
    run_baseline_experiments()
