"""
Unit Tests for PetroRAG Module 3.9 - Anomaly Baselines (Z-Score, Rolling, IQR, MAD)
Validates parametric Rolling Z-Score, non-parametric Tukey's IQR Fences,
and robust Median Absolute Deviation (MAD / Hampel) anomaly detection engines,
along with comparative benchmarking against Isolation Forest.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.analytics.anomaly.baselines import (
    RollingZScoreAnomalyDetector,
    IQRAnomalyDetector,
    MADAnomalyDetector,
)
from src.analytics.anomaly.isolation_forest import (
    IsolationForestAnomalyDetector,
    AnomalyDetectionResult,
)
from src.analytics.services import AnomalyDetectionService
from src.analytics.interfaces import AnomalyDetectionRequest


@pytest.fixture
def baseline_telemetry_df() -> pd.DataFrame:
    """Generates 150 normal telemetry points with 3 injected operational anomalies."""
    np.random.seed(42)
    n = 150
    timestamps = [datetime(2025, 5, 1) + timedelta(minutes=10 * i) for i in range(n)]

    rpm = 2950.0 + np.random.normal(0, 4.0, n)
    discharge_p = 15.0 + np.random.normal(0, 0.25, n)
    vibration = 2.0 + np.random.normal(0, 0.1, n)
    temp = 65.0 + np.random.normal(0, 0.5, n)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "rpm": rpm,
            "discharge_pressure": discharge_p,
            "vibration_rms": vibration,
            "bearing_temp": temp,
        }
    )

    # Injected anomalies:
    # 1. Extreme vibration spike at idx 30
    df.loc[30, "vibration_rms"] = 8.5
    # 2. Extreme pressure drop at idx 70
    df.loc[70, "discharge_pressure"] = 4.0
    # 3. Thermal runaway at idx 110
    df.loc[110, "bearing_temp"] = 110.0

    return df


def test_rolling_zscore_detector_global_mode(baseline_telemetry_df):
    detector = RollingZScoreAnomalyDetector(threshold=3.0)
    # Fit on clean segment (first 25 samples)
    detector.fit(baseline_telemetry_df.iloc[:25])
    assert detector.is_fitted is True
    assert "vibration_rms" in detector.baseline_means
    assert detector.baseline_stds["vibration_rms"] < 0.2

    # Predict across entire dataframe
    scores = detector.score_samples(baseline_telemetry_df)
    preds = detector.predict(baseline_telemetry_df)

    assert len(scores) == 150
    # Injected anomalies should be flagged
    assert preds[30] == True
    assert scores[30] > 0.50
    assert preds[70] == True
    assert preds[110] == True

    # Attribution for vibration spike
    contribs = detector.explain_point(baseline_telemetry_df.iloc[30][detector.feature_names].to_numpy())
    assert contribs[0].feature_name == "vibration_rms"
    assert contribs[0].contribution_weight > 0.50


def test_rolling_zscore_detector_rolling_mode(baseline_telemetry_df):
    detector = RollingZScoreAnomalyDetector(threshold=3.0, window_size=15, min_periods=5)
    detector.fit(baseline_telemetry_df)

    res = detector.detect(baseline_telemetry_df)
    assert isinstance(res, AnomalyDetectionResult)
    assert res.total_samples == 150
    assert res.anomaly_count >= 3

    # Point 30 should be an anomaly
    pt_30 = res.anomalies[30]
    assert pt_30.is_anomaly == True
    assert pt_30.top_affected_feature == "vibration_rms"


def test_iqr_anomaly_detector_global_mode(baseline_telemetry_df):
    detector = IQRAnomalyDetector(k=1.5)
    # Fit on clean baseline
    detector.fit(baseline_telemetry_df.iloc[:25])
    assert detector.is_fitted is True

    scores = detector.score_samples(baseline_telemetry_df)
    preds = detector.predict(baseline_telemetry_df)

    # Inliers should have scores < 0.50
    assert scores[0] < 0.50
    assert preds[0] == False

    # Outliers should be detected
    assert preds[30] == True
    assert scores[30] >= 0.50
    assert preds[70] == True
    assert preds[110] == True


def test_iqr_anomaly_detector_rolling_mode(baseline_telemetry_df):
    detector = IQRAnomalyDetector(k=1.5, window_size=20, min_periods=5)
    detector.fit(baseline_telemetry_df)

    res = detector.detect(baseline_telemetry_df)
    assert res.anomaly_count >= 3
    assert res.anomalies[70].is_anomaly == True
    assert res.anomalies[70].top_affected_feature == "discharge_pressure"


def test_mad_anomaly_detector(baseline_telemetry_df):
    # Fit MAD detector on data with some contamination
    detector = MADAnomalyDetector(threshold=3.5)
    detector.fit(baseline_telemetry_df)

    assert detector.is_fitted is True
    assert "vibration_rms" in detector.baseline_mads

    res = detector.detect(baseline_telemetry_df)
    assert res.anomaly_count >= 3

    # Check thermal runaway point 110
    pt_110 = res.anomalies[110]
    assert pt_110.is_anomaly == True
    assert pt_110.top_affected_feature == "bearing_temp"
    assert pt_110.feature_observed_value >= 100.0


def test_comparative_baseline_execution(baseline_telemetry_df):
    """Verifies that all 4 models (Isolation Forest, Z-score, IQR, MAD) execute on the same data."""
    models = {
        "IsolationForest": IsolationForestAnomalyDetector(contamination=0.03, random_state=42),
        "ZScore": RollingZScoreAnomalyDetector(threshold=3.0),
        "IQR": IQRAnomalyDetector(k=1.5),
        "MAD": MADAnomalyDetector(threshold=3.5),
    }

    results = {}
    for name, model in models.items():
        res = model.detect(baseline_telemetry_df)
        results[name] = res
        assert res.total_samples == 150
        assert res.anomaly_count >= 3
        # All models must catch the severe vibration surge at index 30
        assert res.anomalies[30].is_anomaly == True
        assert res.anomalies[30].top_affected_feature == "vibration_rms"

    # Verify that all results share consistent schemas
    for name, res in results.items():
        assert len(res.feature_names) == 4
        assert res.highest_severity in ["HIGH", "CRITICAL"]


def test_service_baseline_routing(baseline_telemetry_df):
    service = AnomalyDetectionService()
    data_records = baseline_telemetry_df.iloc[:40].to_dict(orient="records")

    # 1. Rolling Z-Score routing
    req_z = AnomalyDetectionRequest(
        entity_id="COMPRESSOR-01",
        entity_type="EQUIPMENT",
        data=data_records,
        detection_method="ROLLING_ZSCORE",
    )
    resp_z = service.detect_anomalies(req_z)
    assert resp_z.detection_method == "ROLLING_ZSCORE"
    assert resp_z.anomalies_detected >= 1
    assert resp_z.records[30].is_anomaly == True
    assert resp_z.records[30].affected_parameter == "vibration_rms"

    # 2. IQR routing
    req_iqr = AnomalyDetectionRequest(
        entity_id="COMPRESSOR-01",
        entity_type="EQUIPMENT",
        data=data_records,
        detection_method="IQR",
    )
    resp_iqr = service.detect_anomalies(req_iqr)
    assert resp_iqr.detection_method == "IQR"
    assert resp_iqr.anomalies_detected >= 1
    assert resp_iqr.records[30].is_anomaly == True
    assert resp_iqr.records[30].affected_parameter == "vibration_rms"
