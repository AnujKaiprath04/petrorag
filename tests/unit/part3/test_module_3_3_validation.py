"""
PetroRAG Module 3.3 Unit Test Suite — Data Quality Validation
Verifies missing values, duplicate timestamps, physical data errors vs
operational anomalies, sensor communication gaps, and quality score computation.
"""

from datetime import datetime, timedelta
import pandas as pd
import pytest

from src.analytics.validation.rules import IssueCategory
from src.analytics.validation.validator import DataQualityValidator


@pytest.fixture
def sample_clean_df() -> pd.DataFrame:
    """Fixture providing nominal, clean production time-series."""
    base = datetime(2026, 1, 1, 0, 0)
    data = []
    for i in range(20):
        data.append({
            "timestamp": base + timedelta(days=i),
            "well_id": "WELL-01",
            "oil_rate": 1200.0 - i * 2.0,
            "gas_rate": 1050.0 - i * 1.5,
            "water_rate": 310.0 + i * 1.0,
            "water_cut": 20.5 + i * 0.1,
            "pressure": 2450.0 - i * 3.0,
            "temperature": 65.0,
        })
    return pd.DataFrame(data)


def test_clean_dataset_validation(sample_clean_df):
    """Verify clean dataset returns high quality score with zero data errors."""
    validator = DataQualityValidator()
    report = validator.validate(sample_clean_df, expected_interval_hours=24.0)

    assert report.total_rows == 20
    assert report.clean_rows == 20
    assert report.data_error_count == 0
    assert report.duplicate_count == 0
    assert report.quality_score >= 95.0
    assert report.is_clean is True
    assert len(report.time_gaps) == 0


def test_physical_impossible_data_errors(sample_clean_df):
    """Verify physically impossible values are flagged strictly as DATA_ERROR."""
    df_corrupted = sample_clean_df.copy()
    # Inject 3 impossible values
    df_corrupted.loc[2, "pressure"] = -25.0       # Negative pressure is physically impossible
    df_corrupted.loc[5, "water_cut"] = 145.0      # Water cut cannot exceed 100%
    df_corrupted.loc[8, "temperature"] = -150.0   # Temperature below physical absolute

    validator = DataQualityValidator()
    report = validator.validate(df_corrupted)

    assert report.data_error_count == 3
    assert report.is_clean is False

    error_issues = [i for i in report.issues if i.category == IssueCategory.DATA_ERROR]
    assert len(error_issues) == 3
    params = {i.parameter for i in error_issues}
    assert params == {"pressure", "water_cut", "temperature"}

    for issue in error_issues:
        assert issue.severity == "CRITICAL"
        assert "Quarantine" in issue.suggested_action


def test_operational_anomalies_preserved(sample_clean_df):
    """
    CRITICAL TEST: Verify operational anomalies are NOT treated as bad data errors.
    They must be preserved for ML anomaly detection.
    """
    df_anomaly = sample_clean_df.copy()
    # Inject operational deviations (physically possible, but exceeding nominal envelopes)
    df_anomaly["vibration"] = 2.0
    df_anomaly.loc[4, "vibration"] = 8.8      # Severe vibration (>7.1 ISO limit, but <=50 mm/s sensor max)
    df_anomaly.loc[10, "pressure"] = 12500.0  # Pressure surge (>10,000 nominal, but <=25,000 psi physical)

    validator = DataQualityValidator()
    report = validator.validate(df_anomaly)

    # Must NOT count as data errors
    assert report.data_error_count == 0
    assert report.operational_anomaly_count == 2

    anom_issues = [i for i in report.issues if i.category == IssueCategory.OPERATIONAL_ANOMALY]
    assert len(anom_issues) == 2
    for issue in anom_issues:
        assert issue.severity == "HIGH"
        assert "Do NOT drop as bad data" in issue.suggested_action


def test_duplicate_timestamps_detection(sample_clean_df):
    """Verify repeated records for the same well at the same timestamp are detected."""
    df_dup = sample_clean_df.copy()
    # Duplicate row 3
    dup_row = df_dup.iloc[[3]].copy()
    df_dup = pd.concat([df_dup, dup_row], ignore_index=True)

    validator = DataQualityValidator()
    report = validator.validate(df_dup)

    assert report.duplicate_count == 2  # Both instances flagged
    assert report.duplicate_percentage > 0.0
    assert report.is_clean is False

    dup_issues = [i for i in report.issues if i.category == IssueCategory.DUPLICATE_RECORD]
    assert len(dup_issues) > 0
    assert dup_issues[0].severity == "HIGH"


def test_missing_values_audit(sample_clean_df):
    """Verify missing values are tracked with percentage per column."""
    df_missing = sample_clean_df.copy()
    df_missing.loc[2:5, "oil_rate"] = None
    df_missing.loc[10, "gas_rate"] = None

    validator = DataQualityValidator()
    report = validator.validate(df_missing)

    assert report.missing_percentage["oil_rate"] == 20.0  # 4 out of 20
    assert report.missing_percentage["gas_rate"] == 5.0   # 1 out of 20
    assert report.quality_score < 100.0

    missing_issues = [i for i in report.issues if i.category == IssueCategory.MISSING_VALUE]
    assert len(missing_issues) == 5


def test_sensor_blackout_time_gap_detection(sample_clean_df):
    """Verify time-series discontinuity / sensor blackout detection."""
    df_gap = sample_clean_df.copy()
    # Create a 7-day blackout between row 9 and row 10
    base = datetime(2026, 1, 1, 0, 0)
    for idx in range(10, 20):
        df_gap.loc[idx, "timestamp"] = base + timedelta(days=idx + 7)

    validator = DataQualityValidator()
    report = validator.validate(df_gap, expected_interval_hours=24.0)

    assert len(report.time_gaps) == 1
    gap = report.time_gaps[0]
    assert gap.entity_id == "WELL-01"
    assert gap.duration_hours >= 192.0  # 8 days (192 hours)

    gap_issues = [i for i in report.issues if i.category == IssueCategory.TIMESTAMP_GAP]
    assert len(gap_issues) == 1
    assert "blackout" in gap_issues[0].description.lower()


def test_empty_dataframe_handling():
    """Verify empty DataFrame returns zero metrics without exceptions."""
    validator = DataQualityValidator()
    report = validator.validate(pd.DataFrame())

    assert report.total_rows == 0
    assert report.quality_score == 0.0
    assert report.is_clean is False
    assert "empty" in report.summary_narrative.lower()
