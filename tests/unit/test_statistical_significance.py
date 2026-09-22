"""
Unit Tests for Statistical Significance & Hypothesis Testing Module
Validates:
1. Cohen's d effect size calculation mathematical soundness.
2. Bootstrap confidence interval calculations.
3. Existence and valid schema of statistical significance report artifacts.
4. Formal rejection of Null Hypothesis (p < 0.01 across retrieval and safety metrics).
"""

import json
from pathlib import Path
import pytest
import numpy as np

from experiments.eval.statistical_significance import (
    cohens_d,
    bootstrap_ci_diff,
    compute_per_query_metrics,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT_DIR / "experiments" / "results"
REPORT_JSON = RESULTS_DIR / "statistical_significance_report.json"
REPORT_MD = RESULTS_DIR / "statistical_significance_report.md"


def test_cohens_d_mathematical_soundness():
    """Validates that Cohen's d matches standard formula."""
    x = [10.0, 11.0, 12.0, 10.5, 11.5]
    y = [8.0, 8.5, 9.0, 8.2, 8.8]
    d = cohens_d(x, y)
    assert d > 0.8  # Large effect size
    assert cohens_d(x, x) == 0.0


def test_bootstrap_ci_diff():
    """Validates that bootstrap CI captures true difference."""
    x = [5.2, 6.1, 7.4, 7.9, 9.3]
    y = [1.0, 2.3, 2.8, 4.1, 4.9]
    low, high = bootstrap_ci_diff(x, y, n_boot=1000, ci=0.95)
    mean_diff = float(np.mean(x) - np.mean(y))
    assert low <= mean_diff <= high
    assert low > 0.0  # Confidently greater than 0


def test_statistical_report_artifacts_exist_and_valid():
    """Ensures statistical report JSON and Markdown exist with complete comparisons."""
    assert REPORT_JSON.exists(), "statistical_significance_report.json missing"
    assert REPORT_MD.exists(), "statistical_significance_report.md missing"

    with open(REPORT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["benchmark_size"] == 50
    assert data["answerable_queries"] == 40
    assert data["unanswerable_queries"] == 10

    comparisons = data["comparisons"]
    assert "Proposed PetroRAG vs. Baseline 1 (Dense RAG)" in comparisons
    dense_comp = comparisons["Proposed PetroRAG vs. Baseline 1 (Dense RAG)"]

    # Check that recall, faithfulness, and safety improvements are statistically significant (p < 0.01)
    assert dense_comp["recall_at_5"]["is_significant_p01"] is True
    assert dense_comp["faithfulness"]["is_significant_p01"] is True
    assert dense_comp["safety_compliance"]["is_significant_p01"] is True
    assert dense_comp["recall_at_5"]["cohens_d"] >= 0.70
