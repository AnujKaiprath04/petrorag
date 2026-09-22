"""
Unit Tests for PetroRAG Module 3.6 - Exploratory Data Analysis (EDA) Pipeline
Validates data profiling, Pearson/Spearman correlation matrices, multicollinearity detection,
distribution & normality analysis, outlier bounds, production decline curve fits,
water cut / pressure trends, and end-to-end EDA reporting.
"""

import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.analytics.eda.profiler import DataProfiler, DatasetProfile, NumericColumnProfile
from src.analytics.eda.correlation import CorrelationAnalyzer, CorrelationReport
from src.analytics.eda.distribution import DistributionAnalyzer, DistributionProfile
from src.analytics.eda.trends import TrendAnalyzer, ProductionTrendReport
from src.analytics.eda.pipeline import EDAPipeline, EDAReport


@pytest.fixture
def synthetic_production_df() -> pd.DataFrame:
    """Generates 90 days of realistic production telemetry for a declining oil well."""
    np.random.seed(42)
    n_days = 90
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(n_days)]
    t = np.arange(n_days, dtype=float)

    # Exponential decline: q(t) = 1500 * exp(-0.003 * t)
    oil_rate = 1500.0 * np.exp(-0.003 * t) + np.random.normal(0, 15, n_days)
    # Gas rate proportional with small noise
    gas_rate = oil_rate * 0.9 + np.random.normal(0, 10, n_days)
    # Increasing water cut
    water_rate = 200.0 + 2.5 * t + np.random.normal(0, 5, n_days)
    total_liq = oil_rate + water_rate
    water_cut = (water_rate / total_liq) * 100.0
    # Depleting tubing pressure
    tubing_pressure = 2200.0 - 1.2 * t + np.random.normal(0, 4, n_days)
    # Choke size (stepped / correlated)
    choke_size = 32.0 - 0.05 * t

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "well_id": ["WELL-01"] * n_days,
            "oil_rate": oil_rate,
            "gas_rate": gas_rate,
            "water_rate": water_rate,
            "water_cut": water_cut,
            "tubing_pressure": tubing_pressure,
            "choke_size": choke_size,
        }
    )
    return df


@pytest.fixture
def synthetic_sensor_df() -> pd.DataFrame:
    """Generates 120 hours of equipment sensor readings with collinear channels and vibration surges."""
    np.random.seed(123)
    n_hours = 120
    timestamps = [datetime(2025, 3, 1, 0, 0) + timedelta(hours=i) for i in range(n_hours)]

    rpm = 3000.0 + np.random.normal(0, 5, n_hours)
    # Strong collinearity between motor_current and discharge_pressure
    motor_current = 45.0 + np.linspace(0, 10, n_hours) + np.random.normal(0, 0.5, n_hours)
    discharge_pressure = 12.0 + 0.8 * (motor_current - 45.0) + np.random.normal(0, 0.2, n_hours)
    bearing_temp = 65.0 + 0.5 * (motor_current - 45.0) + np.random.normal(0, 0.4, n_hours)

    # Vibration with injected outlier spikes
    vibration_rms = 2.2 + np.random.normal(0, 0.15, n_hours)
    vibration_rms[15] = 8.5  # Surge spike
    vibration_rms[45] = 9.2  # Surge spike
    vibration_rms[80] = 7.9  # Surge spike

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "equipment_id": ["PUMP-101"] * n_hours,
            "rpm": rpm,
            "motor_current": motor_current,
            "discharge_pressure": discharge_pressure,
            "bearing_temp": bearing_temp,
            "vibration_rms": vibration_rms,
        }
    )
    return df


def test_data_profiler_numerical_and_categorical(synthetic_production_df):
    profiler = DataProfiler()
    profile = profiler.profile_dataframe(synthetic_production_df)

    assert profile.row_count == 90
    assert profile.column_count == 8
    assert profile.overall_missing_pct == 0.0
    assert "oil_rate" in profile.numeric_profiles
    assert "well_id" in profile.categorical_profiles
    assert "timestamp" in profile.datetime_columns

    oil_prof = profile.numeric_profiles["oil_rate"]
    assert oil_prof.total_count == 90
    assert oil_prof.valid_count == 90
    assert oil_prof.mean > 1100.0
    assert oil_prof.min > 0.0
    assert oil_prof.max > oil_prof.min
    assert "p50" in oil_prof.percentiles
    assert oil_prof.iqr > 0.0

    well_prof = profile.categorical_profiles["well_id"]
    assert well_prof.unique_count == 1
    assert well_prof.top_categories["WELL-01"] == 90

    # Test empty dataframe behavior
    empty_prof = profiler.profile_dataframe(pd.DataFrame())
    assert empty_prof.row_count == 0
    assert empty_prof.column_count == 0


def test_correlation_analyzer_matrices_and_collinearity(synthetic_sensor_df):
    analyzer = CorrelationAnalyzer(collinearity_threshold=0.85)
    report = analyzer.analyze(
        df=synthetic_sensor_df,
        target_col="motor_current",
    )

    assert len(report.numeric_features) >= 4
    assert "motor_current" in report.pearson_matrix
    assert "discharge_pressure" in report.pearson_matrix["motor_current"]

    # motor_current and discharge_pressure should be highly correlated (r > 0.90)
    p_corr = report.pearson_matrix["motor_current"]["discharge_pressure"]
    assert p_corr > 0.90

    # Verify ranked pairs deduplication and sorting
    assert len(report.ranked_pairs) > 0
    for i in range(len(report.ranked_pairs) - 1):
        assert report.ranked_pairs[i].abs_pearson >= report.ranked_pairs[i + 1].abs_pearson

    # Verify collinear pairs flagged
    collinear_names = [(p.feature_a, p.feature_b) for p in report.collinear_pairs]
    assert any(
        ("motor_current" in pair and "discharge_pressure" in pair)
        for pair in collinear_names
    )

    # Verify target correlation ranking
    assert report.target_correlations is not None
    assert len(report.target_correlations) > 0
    assert report.target_correlations[0].target_feature == "motor_current"
    assert report.target_correlations[0].abs_pearson >= report.target_correlations[-1].abs_pearson


def test_distribution_analyzer_normality_and_outliers(synthetic_sensor_df):
    analyzer = DistributionAnalyzer(alpha=0.05, n_bins=10)
    dists = analyzer.analyze(synthetic_sensor_df)

    assert "vibration_rms" in dists
    vib_dist = dists["vibration_rms"]

    # Check outlier detection on vibration spikes
    assert vib_dist.outlier_count_iqr >= 3
    assert vib_dist.outlier_pct_iqr > 0.0
    assert vib_dist.outlier_count_zscore >= 3
    assert vib_dist.iqr_upper_bound < 8.0  # Spikes at 8.5, 9.2 exceed this upper fence

    # Check histogram
    assert len(vib_dist.histogram.counts) == 10
    assert len(vib_dist.histogram.bin_edges) == 11
    assert sum(vib_dist.histogram.counts) == len(synthetic_sensor_df)

    # Small sample test (N < 8)
    small_df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    small_dist = analyzer.analyze(small_df)
    assert small_dist["x"].normality_test_name == "insufficient_sample_size"
    assert small_dist["x"].is_normal is False


def test_trend_analyzer_production_decline(synthetic_production_df):
    analyzer = TrendAnalyzer()
    report = analyzer.analyze(synthetic_production_df)

    assert report.time_span_days == 89.0
    assert report.record_count == 90
    assert report.oil_decline is not None

    oil = report.oil_decline
    assert oil.fluid_type == "oil"
    assert oil.initial_rate > oil.final_rate
    assert oil.rate_change_pct < 0.0
    assert oil.linear_slope_per_day < 0.0
    assert oil.annual_nominal_decline_pct > 0.0
    assert oil.r_squared_exponential > 0.80
    assert oil.decline_regime in ["RAPID_DECLINE", "MODERATE_DECLINE", "LOW_DECLINE"]


def test_trend_analyzer_pressure_and_water_cut(synthetic_production_df):
    analyzer = TrendAnalyzer()
    report = analyzer.analyze(synthetic_production_df)

    # Tubing pressure depletion
    assert report.pressure_depletion is not None
    p_dep = report.pressure_depletion
    assert p_dep.cumulative_pressure_drop > 80.0
    assert p_dep.slope_per_day < -0.8
    assert p_dep.depletion_regime in ["RAPID_DEPLETION", "MODERATE_DEPLETION"]

    # Water cut progression
    assert report.water_evolution is not None
    wc = report.water_evolution
    assert wc.final_water_cut_pct > wc.initial_water_cut_pct
    assert wc.monthly_change_pct > 0.0
    assert wc.water_cut_regime in ["RAPID_BREAKTHROUGH", "GRADUAL_WATERING_OUT"]

    # Gas-Oil Ratio (GOR)
    assert report.gor_evolution is not None
    assert report.gor_evolution.mean_gor > 0.0

    # Summary observations
    assert len(report.summary_observations) > 0
    obs_text = " ".join(report.summary_observations).lower()
    assert "oil" in obs_text or "pressure" in obs_text or "water" in obs_text


def test_eda_pipeline_end_to_end(synthetic_production_df):
    pipeline = EDAPipeline(collinearity_threshold=0.85)
    eda_report = pipeline.run(
        df=synthetic_production_df,
        dataset_name="Well-01 Historical Production",
        dataset_type="PRODUCTION",
        target_col="oil_rate",
    )

    assert eda_report.dataset_name == "Well-01 Historical Production"
    assert eda_report.dataset_type == "PRODUCTION"
    assert eda_report.profile.row_count == 90
    assert len(eda_report.correlations.ranked_pairs) > 0
    assert len(eda_report.distributions) > 0
    assert eda_report.trends is not None

    # Key findings generated
    assert len(eda_report.key_findings) > 0

    # JSON serialization
    json_str = eda_report.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["dataset_name"] == "Well-01 Historical Production"
    assert "profile" in parsed
    assert "correlations" in parsed
    assert "trends" in parsed

    # RAG context generation
    rag_context = eda_report.generate_rag_context()
    assert isinstance(rag_context, str)
    assert "Exploratory Data Analysis" in rag_context
    assert "Key Empirical Observations" in rag_context


def test_eda_pipeline_sensor_dataset(synthetic_sensor_df):
    pipeline = EDAPipeline(collinearity_threshold=0.85)
    eda_report = pipeline.run(
        df=synthetic_sensor_df,
        dataset_name="Pump-101 Telemetry Stream",
        dataset_type="EQUIPMENT_SENSOR",
        target_col="discharge_pressure",
    )

    assert eda_report.profile.row_count == 120
    assert len(eda_report.correlations.collinear_pairs) >= 1
    # Check that high collinearity finding was included
    collinear_findings = [f for f in eda_report.key_findings if "collinear" in f.lower()]
    assert len(collinear_findings) > 0

    # Check outlier detection finding for vibration surges
    vib_findings = [f for f in eda_report.key_findings if "outlier" in f.lower() or "vibration_rms" in f]
    assert len(vib_findings) > 0
