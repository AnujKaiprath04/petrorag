"""
Unit Tests for PetroRAG Module 3.10 - Anomaly Severity & Parameter Attribution
Validates directional deviation tagging (SURGE vs DROPOUT), normalized weight allocation (100%),
ISO 10816-3 / API 610 physical safety boundary checks, deterministic multi-tier severity triage,
multi-sensor co-occurrence matrices, and integration with AnomalyDetectionService.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.analytics.anomaly.attribution import (
    SeverityTriageEngine,
    SeverityTriageResult,
    AnomalyAttributionEngine,
    ParameterAttributionItem,
    DecomposedAnomalyReport,
    AttributionSummary,
    ISO_10816_VIBRATION_ALERT,
    ISO_10816_VIBRATION_DANGER,
    BEARING_TEMP_ALERT_C,
    BEARING_TEMP_DANGER_C,
)
from src.analytics.services import AnomalyDetectionService


@pytest.fixture
def baseline_stats():
    feature_names = ["rpm", "motor_current", "discharge_pressure", "vibration_rms", "bearing_temp"]
    medians = {
        "rpm": 2980.0,
        "motor_current": 42.0,
        "discharge_pressure": 14.5,
        "vibration_rms": 2.1,
        "bearing_temp": 68.0,
    }
    iqrs = {
        "rpm": 8.0,
        "motor_current": 1.2,
        "discharge_pressure": 0.4,
        "vibration_rms": 0.15,
        "bearing_temp": 0.8,
    }
    return feature_names, medians, iqrs


def test_parameter_attribution_weights_sum_to_100(baseline_stats):
    feature_names, medians, iqrs = baseline_stats
    engine = AnomalyAttributionEngine()

    # Record with severe vibration spike and moderate temperature increase
    row = np.array([2980.0, 42.0, 14.5, 6.5, 75.0])
    report = engine.decompose_point(
        row_values=row,
        feature_names=feature_names,
        baseline_medians=medians,
        baseline_iqrs=iqrs,
        anomaly_score=0.82,
        is_anomaly=True,
    )

    assert len(report.attributions) == 5
    total_weight = sum(a.contribution_weight_pct for a in report.attributions)
    assert abs(total_weight - 100.0) < 0.1

    # Primary contributor should be vibration_rms
    assert report.attributions[0].feature_name == "vibration_rms"
    assert report.attributions[0].contribution_weight_pct > 50.0


def test_directional_deviation_tagging(baseline_stats):
    feature_names, medians, iqrs = baseline_stats
    engine = AnomalyAttributionEngine()

    # row: vibration surged (+100%), pressure dropped (-50%), rpm nominal (+0.1%)
    row = np.array([2983.0, 42.0, 7.25, 4.2, 68.0])
    report = engine.decompose_point(
        row_values=row,
        feature_names=feature_names,
        baseline_medians=medians,
        baseline_iqrs=iqrs,
        anomaly_score=0.75,
        is_anomaly=True,
    )

    attr_dict = {a.feature_name: a for a in report.attributions}
    assert attr_dict["vibration_rms"].direction == "SURGE"
    assert attr_dict["discharge_pressure"].direction == "DROPOUT"
    assert attr_dict["rpm"].direction == "NOMINAL"


def test_iso_10816_vibration_safety_envelope(baseline_stats):
    feature_names, medians, iqrs = baseline_stats
    engine = AnomalyAttributionEngine()

    # Case 1: Danger / Trip limit (> 7.1 mm/s RMS)
    row_danger = np.array([2980.0, 42.0, 14.5, 8.5, 68.0])
    rep_danger = engine.decompose_point(
        row_values=row_danger,
        feature_names=feature_names,
        baseline_medians=medians,
        baseline_iqrs=iqrs,
        anomaly_score=0.70,
        is_anomaly=True,
    )
    vib_a = next(a for a in rep_danger.attributions if a.feature_name == "vibration_rms")
    assert vib_a.safety_status == "DANGER_TRIP"
    assert "ISO 10816-3 Zone D" in vib_a.safety_standard_reference
    assert rep_danger.triage.severity == "CRITICAL"
    assert rep_danger.triage.safety_breach_detected == True

    # Case 2: Alert limit (4.5 - 7.1 mm/s RMS)
    row_alert = np.array([2980.0, 42.0, 14.5, 5.2, 68.0])
    rep_alert = engine.decompose_point(
        row_values=row_alert,
        feature_names=feature_names,
        baseline_medians=medians,
        baseline_iqrs=iqrs,
        anomaly_score=0.65,
        is_anomaly=True,
    )
    vib_alert = next(a for a in rep_alert.attributions if a.feature_name == "vibration_rms")
    assert vib_alert.safety_status == "WARNING"
    assert "ISO 10816-3 Zone C" in vib_alert.safety_standard_reference
    assert rep_alert.triage.severity == "HIGH"


def test_thermal_safety_envelope_bearing_temp(baseline_stats):
    feature_names, medians, iqrs = baseline_stats
    engine = AnomalyAttributionEngine()

    # Critical thermal excursion (> 95 C)
    row_hot = np.array([2980.0, 42.0, 14.5, 2.1, 102.0])
    rep_hot = engine.decompose_point(
        row_values=row_hot,
        feature_names=feature_names,
        baseline_medians=medians,
        baseline_iqrs=iqrs,
        anomaly_score=0.68,
        is_anomaly=True,
    )
    temp_a = next(a for a in rep_hot.attributions if a.feature_name == "bearing_temp")
    assert temp_a.safety_status == "DANGER_TRIP"
    assert "API 610 Bearing Danger Trip" in temp_a.safety_standard_reference
    assert rep_hot.triage.severity == "CRITICAL"


def test_severity_triage_tiers():
    triage_engine = SeverityTriageEngine()

    # Normal point
    res_norm = triage_engine.triage(0.25, [], is_anomaly=False)
    assert res_norm.severity == "NORMAL"

    # Low severity: score 0.50, deviation 20%
    item_low = ParameterAttributionItem(
        feature_name="motor_current",
        observed_value=48.0,
        baseline_median=40.0,
        baseline_iqr=2.0,
        absolute_deviation=8.0,
        deviation_pct=20.0,
        relative_intensity_z=4.0,
        direction="SURGE",
        contribution_weight_pct=100.0,
        safety_status="SAFE",
    )
    res_low = triage_engine.triage(0.50, [item_low], is_anomaly=True)
    assert res_low.severity == "LOW"

    # Medium severity: score 0.65, deviation 40%
    item_med = ParameterAttributionItem(
        feature_name="motor_current",
        observed_value=56.0,
        baseline_median=40.0,
        baseline_iqr=2.0,
        absolute_deviation=16.0,
        deviation_pct=40.0,
        relative_intensity_z=8.0,
        direction="SURGE",
        contribution_weight_pct=100.0,
        safety_status="SAFE",
    )
    res_med = triage_engine.triage(0.65, [item_med], is_anomaly=True)
    assert res_med.severity == "MEDIUM"


def test_attribution_summary_and_co_occurrence(baseline_stats):
    feature_names, medians, iqrs = baseline_stats
    engine = AnomalyAttributionEngine()

    reports = []
    # 5 events with vibration + bearing temp co-occurring
    for i in range(5):
        row = np.array([2980.0, 42.0, 14.5, 7.5 + 0.2 * i, 96.0 + i])
        rep = engine.decompose_point(
            row_values=row,
            feature_names=feature_names,
            baseline_medians=medians,
            baseline_iqrs=iqrs,
            anomaly_score=0.90,
            is_anomaly=True,
            index=i,
        )
        reports.append(rep)

    summary = engine.analyze_attribution_matrix(reports)
    assert summary.total_anomalies_evaluated == 5
    assert summary.severity_distribution["CRITICAL"] == 5
    assert summary.safety_limit_breaches_count == 5

    # Check co-occurring sensor pairs
    assert len(summary.top_co_occurring_sensors) >= 1
    pair = summary.top_co_occurring_sensors[0].sensor_pair
    assert "vibration_rms" in pair and "bearing_temp" in pair
    assert summary.top_co_occurring_sensors[0].co_occurrence_count == 5


def test_anomaly_service_decompose_telemetry_integration():
    service = AnomalyDetectionService()
    np.random.seed(42)
    n = 60
    timestamps = [datetime(2025, 6, 1) + timedelta(minutes=10 * i) for i in range(n)]

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "rpm": 2980.0 + np.random.normal(0, 3, n),
            "motor_current": 42.0 + np.random.normal(0, 0.5, n),
            "discharge_pressure": 14.5 + np.random.normal(0, 0.2, n),
            "vibration_rms": 2.1 + np.random.normal(0, 0.1, n),
            "bearing_temp": 68.0 + np.random.normal(0, 0.4, n),
        }
    )

    # Injected ISO Zone D vibration trip event
    df.loc[20, "vibration_rms"] = 8.5
    df.loc[20, "bearing_temp"] = 98.0

    decomposed_list, summary = service.decompose_telemetry(df, time_col="timestamp")

    assert len(decomposed_list) >= 1
    assert isinstance(summary, AttributionSummary)
    assert summary.safety_limit_breaches_count >= 1

    # Point 20 should have been decomposed as CRITICAL
    pt_20 = next((r for r in decomposed_list if r.index == 20), None)
    assert pt_20 is not None
    assert pt_20.triage.severity == "CRITICAL"
    assert pt_20.triage.safety_breach_detected == True
