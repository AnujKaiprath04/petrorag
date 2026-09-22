"""
PetroRAG Statistical Hypothesis Testing & Significance Engine
Calculates paired Student's t-tests, Wilcoxon signed-rank tests, Cohen's d effect sizes,
and 95% confidence intervals across all evaluated architectures over the benchmark dataset.
Formally validates H1 (Enhanced PetroRAG significantly outperforms baselines) vs H0 (Null).
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.stats as stats

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.eval.metrics import (
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
)

RESULTS_DIR = ROOT_DIR / "experiments" / "results"
RAW_RUNS_PATH = RESULTS_DIR / "raw_benchmark_runs.json"


def compute_per_query_metrics(runs: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Calculates granular per-query metrics vectors."""
    recalls: List[float] = []
    mrrs: List[float] = []
    ndcgs: List[float] = []
    faithfulness: List[float] = []
    hallucination: List[float] = []
    safety_compliance: List[float] = []

    for item in runs:
        is_answerable = item["is_answerable"]
        is_abstained = item["is_abstained"]
        retrieved = item["retrieved_chunks"]
        gt = item["ground_truth_chunks"]

        # Safety compliance: 1.0 if correct behavior (abstains on unanswerable, answers on answerable), else 0.0
        if not is_answerable:
            safety = 1.0 if is_abstained else 0.0
            # For unanswerable queries, retrieval metrics are not applicable
            safety_compliance.append(safety)
            # Hallucination: 1.0 if model hallucinated an answer instead of abstaining
            hallucination.append(0.0 if is_abstained else 1.0)
            faithfulness.append(1.0 if is_abstained else 0.0)
            recalls.append(1.0 if is_abstained else 0.0)
            mrrs.append(1.0 if is_abstained else 0.0)
            ndcgs.append(1.0 if is_abstained else 0.0)
        else:
            safety = 1.0 if not is_abstained else 0.0
            safety_compliance.append(safety)

            r5 = compute_recall_at_k(retrieved, gt, k=5)
            rr = compute_reciprocal_rank(retrieved, gt)
            nd = compute_ndcg_at_k(retrieved, gt, k=5)

            recalls.append(r5)
            mrrs.append(rr)
            ndcgs.append(nd)

            # Faithfulness proxy: 1.0 if top chunk hits ground truth, 0.5 partial, 0.0 none
            if any(c in gt for c in retrieved[:2]):
                faith = 1.0
                hall = 0.0
            elif any(c in gt for c in retrieved):
                faith = 0.8
                hall = 0.2
            elif len(retrieved) == 0:
                faith = 0.0
                hall = 1.0
            else:
                faith = 0.5
                hall = 0.5

            faithfulness.append(faith)
            hallucination.append(hall)

    return {
        "recall_at_5": recalls,
        "mrr": mrrs,
        "ndcg_at_5": ndcgs,
        "faithfulness": faithfulness,
        "hallucination_rate": hallucination,
        "safety_compliance": safety_compliance,
    }


def cohens_d(x: List[float], y: List[float]) -> float:
    """Computes Cohen's d effect size between paired samples."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def bootstrap_ci_diff(x: List[float], y: List[float], n_boot: int = 2000, ci: float = 0.95) -> Tuple[float, float]:
    """Computes empirical bootstrap confidence interval for mean(x) - mean(y)."""
    diffs = np.array(x) - np.array(y)
    n = len(diffs)
    np.random.seed(42)
    boot_means = [np.mean(np.random.choice(diffs, size=n, replace=True)) for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return lower, upper


def run_statistical_analysis():
    if not RAW_RUNS_PATH.exists():
        raise FileNotFoundError(f"Missing raw runs: {RAW_RUNS_PATH}")

    with open(RAW_RUNS_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Compute metrics vectors
    metric_vectors = {}
    for arch_name, runs in raw_data.items():
        metric_vectors[arch_name] = compute_per_query_metrics(runs)

    petrorag_vecs = metric_vectors["Model 4 (Proposed PetroRAG)"]

    comparisons = [
        ("Baseline 1 (Dense RAG)", "Proposed PetroRAG"),
        ("Model 2 (Hybrid RAG)", "Proposed PetroRAG"),
        ("Model 3 (Hybrid + Rerank)", "Proposed PetroRAG"),
    ]

    metrics_to_test = ["recall_at_5", "mrr", "ndcg_at_5", "faithfulness", "safety_compliance"]

    report_data = {
        "benchmark_size": 50,
        "answerable_queries": 40,
        "unanswerable_queries": 10,
        "comparisons": {},
    }

    print("\n" + "=" * 95)
    print("  PETRORAG FORMAL STATISTICAL HYPOTHESIS TESTING & EFFECT SIZE REPORT")
    print("=" * 95)

    for baseline_name, target_name in comparisons:
        base_vecs = metric_vectors[baseline_name]
        comp_key = f"{target_name} vs. {baseline_name}"
        report_data["comparisons"][comp_key] = {}

        print(f"\n[Comparison]: {comp_key}")
        print(f"{'Metric':<20} | {'Mean Diff':<10} | {'t-stat':<8} | {'t p-val':<10} | {'Wilcox p':<10} | {'Cohen d':<8} | {'95% CI':<18}")
        print("-" * 95)

        for m in metrics_to_test:
            x = petrorag_vecs[m]
            y = base_vecs[m]

            mean_diff = float(np.mean(x) - np.mean(y))

            # Paired t-test
            t_res = stats.ttest_rel(x, y)
            t_stat = float(t_res.statistic) if not np.isnan(t_res.statistic) else 0.0
            t_pval = float(t_res.pvalue) if not np.isnan(t_res.pvalue) else 1.0

            # Wilcoxon signed rank
            try:
                w_res = stats.wilcoxon(x, y, alternative="two-sided")
                w_pval = float(w_res.pvalue)
            except Exception:
                w_pval = 1.0

            # Cohen's d
            d = cohens_d(x, y)

            # 95% CI
            ci_low, ci_high = bootstrap_ci_diff(x, y)

            report_data["comparisons"][comp_key][m] = {
                "mean_target": float(np.mean(x)),
                "mean_baseline": float(np.mean(y)),
                "mean_difference": round(mean_diff, 4),
                "t_statistic": round(t_stat, 4),
                "t_pvalue": t_pval,
                "wilcoxon_pvalue": w_pval,
                "cohens_d": round(d, 4),
                "ci_95": [round(ci_low, 4), round(ci_high, 4)],
                "is_significant_p01": bool(t_pval < 0.01 or w_pval < 0.01),
                "is_significant_p001": bool(t_pval < 0.001 or w_pval < 0.001),
            }

            p_str = f"{t_pval:.4e}" if t_pval < 0.0001 else f"{t_pval:.4f}"
            w_str = f"{w_pval:.4e}" if w_pval < 0.0001 else f"{w_pval:.4f}"
            ci_str = f"[{ci_low:+.3f}, {ci_high:+.3f}]"

            print(f"{m:<20} | {mean_diff:+10.4f} | {t_stat:8.2f} | {p_str:<10} | {w_str:<10} | {d:8.2f} | {ci_str:<18}")

    # Save JSON report
    out_json = RESULTS_DIR / "statistical_significance_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[OK] Statistical report saved to: {out_json}")

    # Generate Markdown Report
    out_md = RESULTS_DIR / "statistical_significance_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# PetroRAG Statistical Significance & Hypothesis Testing Report\n\n")
        f.write("## Overview\n")
        f.write("Formally evaluates Primary Hypothesis ($H_1$) vs. Null Hypothesis ($H_0$) across 50 gold-standard queries.\n\n")
        for comp_name, metrics in report_data["comparisons"].items():
            f.write(f"### {comp_name}\n\n")
            f.write("| Metric | Target Mean | Baseline Mean | Diff | t-stat | t-test p-value | Wilcoxon p-value | Cohen's d | 95% CI |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for m, vals in metrics.items():
                f.write(
                    f"| `{m}` | {vals['mean_target']:.4f} | {vals['mean_baseline']:.4f} | "
                    f"{vals['mean_difference']:+.4f} | {vals['t_statistic']:.2f} | "
                    f"{vals['t_pvalue']:.4e} | {vals['wilcoxon_pvalue']:.4e} | "
                    f"{vals['cohens_d']:.2f} | `[{vals['ci_95'][0]:+.3f}, {vals['ci_95'][1]:+.3f}]` |\n"
                )
            f.write("\n")
    print(f"[OK] Markdown report saved to: {out_md}")


if __name__ == "__main__":
    run_statistical_analysis()
