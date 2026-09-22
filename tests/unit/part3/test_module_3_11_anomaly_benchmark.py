"""
Unit Tests for PetroRAG Module 3.11 - Anomaly Benchmark Verification (Injected Ground Truth)
Validates ground truth generation of 50 labeled operational anomalies across 5 failure categories,
confusion matrix mathematics (Precision, Recall, F1, FPR, Specificity, Latency),
empirical superiority of Isolation Forest on multivariate subspace shifts,
and publication-ready LaTeX table generation.
"""

from datetime import datetime
import json
import numpy as np
import pytest

from src.analytics.anomaly.benchmark import (
    AnomalyGroundTruthGenerator,
    AnomalyBenchmarkRunner,
    BenchmarkDataset,
    ModelBenchmarkMetrics,
    AnomalyBenchmarkReport,
    InjectedAnomalyRecord,
)


@pytest.fixture
def benchmark_dataset() -> BenchmarkDataset:
    generator = AnomalyGroundTruthGenerator()
    return generator.generate(n_samples=500, random_seed=42)


def test_ground_truth_generator_50_anomalies():
    generator = AnomalyGroundTruthGenerator()
    dataset = generator.generate(n_samples=1000, random_seed=42)

    assert dataset.telemetry_df.shape[0] == 1000
    assert dataset.telemetry_df.shape[1] == 7  # timestamp + 6 sensor channels
    assert len(dataset.injected_records) == 50
    assert sum(dataset.ground_truth_labels) == 50

    # Check distribution of the 5 failure modes
    categories = [rec.anomaly_type for rec in dataset.injected_records]
    assert categories.count("VIBRATION_SPIKE") == 15
    assert categories.count("PRESSURE_DROPOUT") == 10
    assert categories.count("THERMAL_RUNAWAY") == 10
    assert categories.count("MULTIVARIATE_CAVITATION") == 10
    assert categories.count("PROCESS_CHURN") == 5


def test_benchmark_confusion_matrix_mathematics(benchmark_dataset):
    runner = AnomalyBenchmarkRunner()
    report = runner.run_benchmark(benchmark_dataset)

    assert isinstance(report, AnomalyBenchmarkReport)
    assert report.total_points == 500
    assert report.total_injected_anomalies == 50
    assert len(report.model_results) == 4

    for name, m in report.model_results.items():
        assert m.total_samples == 500
        assert m.ground_truth_anomalies == 50
        # Fundamental identity: TP + FP + TN + FN == Total
        assert (m.tp + m.fp + m.tn + m.fn) == 500
        # Positive condition identity: TP + FN == Ground Truth Positives
        assert (m.tp + m.fn) == 50
        # Negative condition identity: TN + FP == Ground Truth Negatives
        assert (m.tn + m.fp) == 450

        # Mathematical metric definitions
        expected_prec = m.tp / max(1, m.tp + m.fp)
        expected_rec = m.tp / max(1, m.tp + m.fn)
        expected_fpr = m.fp / max(1, m.fp + m.tn)
        expected_spec = m.tn / max(1, m.tn + m.fp)

        assert abs(m.precision - expected_prec) < 1e-3
        assert abs(m.recall - expected_rec) < 1e-3
        assert abs(m.false_positive_rate - expected_fpr) < 1e-3
        assert abs(m.specificity - expected_spec) < 1e-3

        # Latency must be measured from real execution
        assert m.inference_latency_ms > 0.0


def test_isolation_forest_superior_multivariate_recall(benchmark_dataset):
    runner = AnomalyBenchmarkRunner()
    report = runner.run_benchmark(benchmark_dataset)

    iso_metrics = report.model_results["IsolationForest"]
    zscore_metrics = report.model_results["RollingZScore"]

    # Overall F1 score of Isolation Forest should be competitive or winning
    assert iso_metrics.f1_score >= 0.60
    assert iso_metrics.recall >= 0.70

    # Multivariate cavitation recall: coupled P-I-V signature
    iso_cav_rec = iso_metrics.category_recall["MULTIVARIATE_CAVITATION"]
    zscore_cav_rec = zscore_metrics.category_recall["MULTIVARIATE_CAVITATION"]

    # Multi-dimensional isolation trees should detect subspace shifts effectively
    assert iso_cav_rec >= zscore_cav_rec
    assert iso_metrics.specificity > 0.90


def test_latex_table_generation(benchmark_dataset):
    runner = AnomalyBenchmarkRunner()
    report = runner.run_benchmark(benchmark_dataset)

    latex_str = report.generate_latex_table()
    assert isinstance(latex_str, str)
    assert r"\begin{table}" in latex_str
    assert r"\end{table}" in latex_str
    assert "IsolationForest" in latex_str
    assert "RollingZScore" in latex_str
    assert "TukeyIQR" in latex_str
    assert "HampelMAD" in latex_str
    # Verify no unformatted placeholders
    assert "NOT YET MEASURED" not in latex_str
    assert "{m." not in latex_str


def test_json_serialization(benchmark_dataset):
    runner = AnomalyBenchmarkRunner()
    report = runner.run_benchmark(benchmark_dataset)

    rep_dict = report.to_dict()
    assert isinstance(rep_dict, dict)
    assert rep_dict["total_injected_anomalies"] == 50
    assert "IsolationForest" in rep_dict["model_results"]
    assert len(rep_dict["executive_findings"]) >= 3

    json_str = json.dumps(rep_dict)
    assert len(json_str) > 200
