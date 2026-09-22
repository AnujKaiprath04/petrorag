"""
Unit Tests for PetroRAG Module 3.8 - Multivariate Isolation Forest Anomaly Detection
Validates fitting, calibrated scoring [0, 1], binary prediction, robust scaling,
parameter attribution decomposition, model state serialization/deserialization,
and integration with AnomalyDetectionService.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.analytics.anomaly.isolation_forest import (
    IsolationForestAnomalyDetector,
    AnomalyDetectionResult,
    DetectedAnomalyPoint,
    FeatureContribution,
)
from src.analytics.services import AnomalyDetectionService
from src.analytics.interfaces import AnomalyDetectionRequest, AnomalyDetectionResponse


@pytest.fixture
def nominal_sensor_data() -> pd.DataFrame:
    """Generates 200 normal operational telemetry points for a centrifugal pump."""
    np.random.seed(42)
    n = 200
    timestamps = [datetime(2025, 4, 1) + timedelta(minutes=15 * i) for i in range(n)]

    rpm = 2980.0 + np.random.normal(0, 5.0, n)
    motor_current = 42.0 + np.random.normal(0, 0.8, n)
    discharge_pressure = 14.5 + np.random.normal(0, 0.3, n)
    suction_pressure = 2.2 + np.random.normal(0, 0.1, n)
    bearing_temp = 68.0 + np.random.normal(0, 0.5, n)
    vibration_rms = 2.1 + np.random.normal(0, 0.1, n)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "rpm": rpm,
            "motor_current": motor_current,
            "discharge_pressure": discharge_pressure,
            "suction_pressure": suction_pressure,
            "bearing_temp": bearing_temp,
            "vibration_rms": vibration_rms,
        }
    )


def test_isolation_forest_fit_and_predict(nominal_sensor_data):
    detector = IsolationForestAnomalyDetector(
        contamination=0.05,
        n_estimators=100,
        random_state=42,
    )
    detector.fit(nominal_sensor_data)
    assert detector.is_fitted is True
    assert len(detector.feature_names) == 6
    assert "vibration_rms" in detector.baseline_medians

    # Score nominal samples
    scores_nominal = detector.score_samples(nominal_sensor_data.iloc[:20])
    assert len(scores_nominal) == 20
    assert np.all(scores_nominal >= 0.0) and np.all(scores_nominal <= 1.0)
    # Most nominal samples should have scores < 0.50
    assert np.median(scores_nominal) < 0.50

    # Test distinct injected anomalies
    anomaly_df = pd.DataFrame(
        [
            # Extreme vibration surge (bearing failure)
            {
                "rpm": 2980.0,
                "motor_current": 42.0,
                "discharge_pressure": 14.5,
                "suction_pressure": 2.2,
                "bearing_temp": 68.0,
                "vibration_rms": 9.5,
            },
            # Severe discharge pressure drop (impeller failure / leak)
            {
                "rpm": 2980.0,
                "motor_current": 25.0,
                "discharge_pressure": 3.0,
                "suction_pressure": 2.2,
                "bearing_temp": 68.0,
                "vibration_rms": 2.1,
            },
        ]
    )

    scores_anom = detector.score_samples(anomaly_df)
    preds_anom = detector.predict(anomaly_df)

    assert scores_anom[0] > 0.50
    assert scores_anom[1] > 0.50
    assert preds_anom[0] == True
    assert preds_anom[1] == True


def test_parameter_attribution(nominal_sensor_data):
    detector = IsolationForestAnomalyDetector(random_state=42)
    detector.fit(nominal_sensor_data)

    # Injected vibration spike: 10.0 mm/s (normal is ~2.1)
    spike_sample = np.array([2980.0, 42.0, 14.5, 2.2, 68.0, 10.0])
    contribs = detector.explain_point(spike_sample)

    assert len(contribs) == 6
    # Top contributing feature should clearly be vibration_rms
    top = contribs[0]
    assert top.feature_name == "vibration_rms"
    assert top.observed_value == 10.0
    assert top.contribution_weight > 0.50

    # Injected bearing overheating: 120 C (normal is ~68)
    overheat_sample = np.array([2980.0, 42.0, 14.5, 2.2, 120.0, 2.1])
    contribs_heat = detector.explain_point(overheat_sample)
    assert contribs_heat[0].feature_name == "bearing_temp"
    assert contribs_heat[0].contribution_weight > 0.50


def test_robust_vs_standard_scaler(nominal_sensor_data):
    # RobustScaler
    det_robust = IsolationForestAnomalyDetector(scaler_type="robust", random_state=42)
    det_robust.fit(nominal_sensor_data)
    score_r = det_robust.score_samples(nominal_sensor_data.iloc[:5])
    assert len(score_r) == 5

    # StandardScaler
    det_std = IsolationForestAnomalyDetector(scaler_type="standard", random_state=42)
    det_std.fit(nominal_sensor_data)
    score_s = det_std.score_samples(nominal_sensor_data.iloc[:5])
    assert len(score_s) == 5

    # None scaler
    det_none = IsolationForestAnomalyDetector(scaler_type="none", random_state=42)
    det_none.fit(nominal_sensor_data)
    score_n = det_none.score_samples(nominal_sensor_data.iloc[:5])
    assert len(score_n) == 5


def test_model_save_and_load(tmp_path, nominal_sensor_data):
    detector = IsolationForestAnomalyDetector(n_estimators=50, random_state=42)
    detector.fit(nominal_sensor_data)

    save_path = str(tmp_path / "iso_forest.joblib")
    detector.save(save_path)

    loaded = IsolationForestAnomalyDetector.load(save_path)
    assert loaded.is_fitted is True
    assert loaded.feature_names == detector.feature_names
    assert loaded.baseline_medians == detector.baseline_medians

    # Verify identical predictions and scores
    test_sub = nominal_sensor_data.iloc[:10]
    orig_scores = detector.score_samples(test_sub)
    loaded_scores = loaded.score_samples(test_sub)
    np.testing.assert_allclose(orig_scores, loaded_scores, rtol=1e-5)


def test_dataframe_end_to_end_detection(nominal_sensor_data):
    # Inject 3 anomalies into the dataframe
    df = nominal_sensor_data.copy()
    df.loc[25, "vibration_rms"] = 11.2  # Spike
    df.loc[75, "bearing_temp"] = 115.0  # Overheat
    df.loc[140, "discharge_pressure"] = 2.0  # Pressure loss

    detector = IsolationForestAnomalyDetector(contamination=0.03, random_state=42)
    res = detector.detect(df, time_col="timestamp")

    assert isinstance(res, AnomalyDetectionResult)
    assert res.total_samples == 200
    assert res.anomaly_count >= 3
    assert res.highest_severity in ["HIGH", "CRITICAL"]

    # Check that injected point 25 was detected with top feature vibration_rms
    pt_25 = res.anomalies[25]
    assert pt_25.is_anomaly == True
    assert pt_25.top_affected_feature == "vibration_rms"
    assert pt_25.anomaly_score > 0.60

    # Check point 75 top feature bearing_temp
    pt_75 = res.anomalies[75]
    assert pt_75.is_anomaly == True
    assert pt_75.top_affected_feature == "bearing_temp"


def test_anomaly_detection_service_integration(nominal_sensor_data):
    service = AnomalyDetectionService()

    # Convert subset of rows into request dict list
    df_sample = nominal_sensor_data.iloc[:50].copy()
    df_sample.loc[10, "vibration_rms"] = 12.0  # Severe surge

    data_rows = df_sample.to_dict(orient="records")

    req = AnomalyDetectionRequest(
        entity_id="PUMP-CENTRIFUGAL-01",
        entity_type="EQUIPMENT",
        data=data_rows,
        detection_method="ISOLATION_FOREST",
        sensitivity=0.05,
    )

    resp = service.detect_anomalies(req)

    assert isinstance(resp, AnomalyDetectionResponse)
    assert resp.entity_id == "PUMP-CENTRIFUGAL-01"
    assert resp.total_points_analyzed == 50
    assert resp.anomalies_detected >= 1
    assert resp.highest_severity in ["HIGH", "CRITICAL"]

    # Injected point 10 should be an anomaly with affected_parameter vibration_rms
    anom_10 = resp.records[10]
    assert anom_10.is_anomaly == True
    assert anom_10.affected_parameter == "vibration_rms"
    assert anom_10.observed_value >= 11.0
