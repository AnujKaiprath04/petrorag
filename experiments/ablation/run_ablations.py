"""
Component Ablation Study Experiments (Module 2.33)
Evaluates PetroRAG performance with individual components systematically toggled off:
1. PetroRAG Full (Reference Complete Architecture)
2. Ablation 1 (\\ Hybrid Retrieval) -> Dense Vector Only
3. Ablation 2 (\\ Metadata Filtering) -> Cross-Asset Retrieval Noise
4. Ablation 3 (\\ Cross-Encoder Reranker) -> Raw RRF Order
5. Ablation 4 (\\ Context Compression) -> Full Uncompressed Chunks
6. Ablation 5 (\\ Grounding Verifier) -> No Claim-Level Entailment Check
7. Ablation 6 (\\ Multi-Barrier Abstention) -> No Abstention Safety Gates

Saves results to experiments/results/ablation_study_results.json and .csv
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


def run_ablation_experiments() -> List[BenchmarkRunMetrics]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PetroRAG Phase 6: Executing Systematic Component Ablation Study")
    print("Authoritative Oil & Gas Benchmark Corpus (50 Gold Queries)")
    print("=" * 80)

    harness = BenchmarkCorpusHarness()
    evaluator = BenchmarkEvaluator(harness)

    ablation_configurations = [
        {
            "name": "PetroRAG (Full Architecture)",
            "config": {
                "retrieval_mode": "petrorag",
                "enable_reranking": True,
                "enable_compression": True,
                "enable_revision_management": True,
                "enable_security_barrier": True,
                "enable_retrieval_barrier": True,
                "enable_grounding_eval": True
            }
        },
        {
            "name": "Ablation 1 (\\ Hybrid Retrieval)",
            "config": {
                "retrieval_mode": "vector",
                "enable_reranking": True,
                "enable_compression": True,
                "enable_revision_management": True,
                "enable_security_barrier": True,
                "enable_retrieval_barrier": True,
                "enable_grounding_eval": True
            }
        },
        {
            "name": "Ablation 2 (\\ Metadata Filtering)",
            "config": {
                "retrieval_mode": "hybrid_rerank",
                "enable_reranking": True,
                "enable_compression": True,
                "enable_revision_management": True,
                "enable_security_barrier": True,
                "enable_retrieval_barrier": True,
                "enable_grounding_eval": True
            }
        },
        {
            "name": "Ablation 3 (\\ Cross-Encoder Reranker)",
            "config": {
                "retrieval_mode": "petrorag",
                "enable_reranking": False,
                "enable_compression": True,
                "enable_revision_management": True,
                "enable_security_barrier": True,
                "enable_retrieval_barrier": True,
                "enable_grounding_eval": True
            }
        },
        {
            "name": "Ablation 4 (\\ Context Compression)",
            "config": {
                "retrieval_mode": "petrorag",
                "enable_reranking": True,
                "enable_compression": False,
                "enable_revision_management": True,
                "enable_security_barrier": True,
                "enable_retrieval_barrier": True,
                "enable_grounding_eval": True
            }
        },
        {
            "name": "Ablation 5 (\\ Grounding Verifier)",
            "config": {
                "retrieval_mode": "petrorag",
                "enable_reranking": True,
                "enable_compression": True,
                "enable_revision_management": True,
                "enable_security_barrier": True,
                "enable_retrieval_barrier": True,
                "enable_grounding_eval": False
            }
        },
        {
            "name": "Ablation 6 (\\ Multi-Barrier Abstention)",
            "config": {
                "retrieval_mode": "petrorag",
                "enable_reranking": True,
                "enable_compression": True,
                "enable_revision_management": True,
                "enable_security_barrier": False,
                "enable_retrieval_barrier": False,
                "enable_grounding_eval": True
            }
        }
    ]

    all_summaries: List[BenchmarkRunMetrics] = []
    all_raw_traces: Dict[str, Any] = {}

    for item in ablation_configurations:
        name = item["name"]
        cfg = item["config"]
        print(f"\n--> Running Ablation: {name} ...")
        summary, traces = evaluator.evaluate_model(model_name=name, **cfg)
        all_summaries.append(summary)
        all_raw_traces[name] = traces

        print(
            f"    Recall@5: {summary.recall_at_5:.4f} | MRR: {summary.mrr:.4f} | "
            f"NDCG@5: {summary.ndcg_at_5:.4f} | Faithfulness: {summary.faithfulness:.4f} | "
            f"Hallucination: {summary.hallucination_rate:.4f} | Abstention F1: {summary.abstention_f1:.4f} | "
            f"Prompt Tok: {summary.mean_prompt_tokens:.0f} | Token Savings: {summary.token_savings_percent:.1f}%"
        )

    # Save to JSON
    json_path = RESULTS_DIR / "ablation_study_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([s.model_dump() for s in all_summaries], f, indent=2)
    print(f"\n[OK] Saved ablation summaries to {json_path}")

    # Save to CSV
    csv_path = RESULTS_DIR / "ablation_study_results.csv"
    fieldnames = list(all_summaries[0].model_dump().keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_summaries:
            writer.writerow(s.model_dump())
    print(f"[OK] Saved ablation comparison CSV to {csv_path}")

    return all_summaries


if __name__ == "__main__":
    run_ablation_experiments()
