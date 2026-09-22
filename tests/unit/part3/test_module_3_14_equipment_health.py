"""
Unit Tests for PetroRAG Module 3.14 - Equipment Health Engine
Validates composite Equipment Health Index (EHI 0-100), 4-dimensional sub-indices,
Weibull age wear modeling, Remaining Useful Life (RUL) estimation with confidence bounds,
fleet-level prioritization queues, and RAG grounding context generation.
"""

import pytest
import numpy as np

from src.analytics.health.engine import (
    EquipmentHealthEngine,
    EquipmentProfile,
    ComprehensiveEquipmentHealthReport,
    FleetHealthSummary,
    SubIndexBreakdown,
    RULEstimate,
    DEFAULT_PROFILES,
)


@pytest.fixture
def health_engine():
    return EquipmentHealthEngine()


def test_pristine_healthy_asset_assessment(health_engine):
    # Brand new ESP with optimal telemetry and recent PM
    report = health_engine.evaluate_health(
        equipment_id="ESP-WELL-101",
        equipment_type="ESP",
        current_telemetry={
            "vibration_rms": 1.8,  # Zone A (New/Pristine)
            "bearing_temp": 64.0,  # Optimal (<75°C)
            "motor_current": 42.0,  # Below rated (45A)
            "discharge_pressure": 15.2,
        },
        historical_anomalies_30d=0,
        cumulative_run_hours=1200.0,  # 5% of 25,000h design life
        days_since_last_pm=20,        # Recent PM
    )

    assert isinstance(report, ComprehensiveEquipmentHealthReport)
    assert report.equipment_id == "ESP-WELL-101"
    assert report.equipment_type == "ESP"
    assert report.composite_health_index >= 90.0
    assert report.health_status == "HEALTHY"
    assert report.maintenance_status == "CURRENT"
    assert report.recommended_inspection_interval_days == 90

    # Verify sub-indices are all high
    sub = report.sub_indices
    assert sub.telemetry_stress_score >= 95.0
    assert sub.anomaly_history_score == 100.0
    assert sub.runtime_aging_score >= 90.0
    assert sub.maintenance_compliance_score >= 90.0

    # RUL should be healthy and long
    assert report.rul_forecast.rul_days_to_critical > 365.0
    assert report.rul_forecast.estimated_wear_rate_per_month < 3.0


def test_critical_asset_assessment_with_trip_proximity(health_engine):
    # Degraded asset with severe vibration and thermal runaway
    report = health_engine.evaluate_health(
        equipment_id="COMPRESSOR-C201",
        equipment_type="GAS_COMPRESSOR",
        current_telemetry={
            "vibration_rms": 8.4,    # Zone D trip (>7.1 mm/s)
            "bearing_temp": 98.5,    # API trip (>95°C)
            "motor_current": 145.0,  # Above rated (120A)
            "discharge_pressure": 6.8,  # Abnormal loss
        },
        historical_anomalies_30d=4,
        anomaly_severity_counts={"CRITICAL": 2, "HIGH": 2},
        cumulative_run_hours=42000.0,
        days_since_last_pm=110,
    )

    assert report.composite_health_index < 50.0
    assert report.health_status == "CRITICAL"
    assert report.recommended_inspection_interval_days == 1

    # Verify primary stress factors identified both trips
    stress_text = " ".join(report.primary_stress_factors)
    assert "vibration" in stress_text.lower()
    assert "bearing temp" in stress_text.lower()

    # Verify emergency actions prescribed
    assert any("IMMEDIATE" in act or "shutdown" in act.lower() for act in report.recommended_interventions)
    assert any("LOTO" in act or "lockout" in act.lower() for act in report.recommended_interventions)

    # RUL to critical should be 0 or near 0
    assert report.rul_forecast.rul_days_to_critical == 0.0


def test_weibull_aging_and_overdue_maintenance(health_engine):
    # Asset past its 25,000h design life and 120 days overdue for PM
    report = health_engine.evaluate_health(
        equipment_id="ESP-OLD-05",
        equipment_type="ESP",
        current_telemetry={
            "vibration_rms": 2.2,  # Nominal telemetry
            "bearing_temp": 68.0,
            "motor_current": 41.0,
            "discharge_pressure": 14.5,
        },
        historical_anomalies_30d=0,
        cumulative_run_hours=28500.0,  # 114% of design life
        days_since_last_pm=300,        # 120 days past 180d PM interval
    )

    assert report.maintenance_status == "OVERDUE"
    assert report.sub_indices.runtime_aging_score < 40.0
    assert report.sub_indices.maintenance_compliance_score < 40.0
    assert report.composite_health_index < 75.0

    # Stresses must capture overdue maintenance and design life
    stress_text = " ".join(report.primary_stress_factors)
    assert "overdue" in stress_text.lower()
    assert "design life" in stress_text.lower()


def test_rul_forecasting_with_empirical_trajectory(health_engine):
    # Asset with historical EHI decline trajectory: 88.0 -> 84.0 -> 79.0 -> 74.0 (14 points drop over 30d)
    trajectory = [88.0, 84.0, 79.0, 74.0]

    report = health_engine.evaluate_health(
        equipment_id="PUMP-P105",
        equipment_type="CENTRIFUGAL_PUMP",
        current_telemetry={
            "vibration_rms": 3.8,  # elevated
            "bearing_temp": 78.0,
            "motor_current": 46.0,
            "discharge_pressure": 13.5,
        },
        cumulative_run_hours=18000.0,
        days_since_last_pm=80,
        historical_ehi_trajectory=trajectory,
    )

    rul = report.rul_forecast
    assert isinstance(rul, RULEstimate)
    assert rul.estimated_wear_rate_per_month == 14.0
    assert rul.rul_days_to_critical > 0.0
    assert rul.confidence_interval_90_days[0] < rul.confidence_interval_90_days[1]
    assert rul.confidence_interval_90_days[0] > 0.0


def test_fleet_health_rollup_and_prioritization(health_engine):
    fleet = [
        {
            "equipment_id": "PUMP-01",
            "equipment_type": "CENTRIFUGAL_PUMP",
            "telemetry": {"vibration_rms": 1.5, "bearing_temp": 60.0},
            "cumulative_run_hours": 2000.0,
            "days_since_last_pm": 20,
        },
        {
            "equipment_id": "PUMP-02",
            "equipment_type": "CENTRIFUGAL_PUMP",
            "telemetry": {"vibration_rms": 4.8, "bearing_temp": 82.0},  # Observation
            "cumulative_run_hours": 15000.0,
            "days_since_last_pm": 70,
        },
        {
            "equipment_id": "ESP-03",
            "equipment_type": "ESP",
            "telemetry": {"vibration_rms": 8.6, "bearing_temp": 98.0},  # Critical
            "historical_anomalies_30d": 3,
            "cumulative_run_hours": 24000.0,
            "days_since_last_pm": 190,
        },
        {
            "equipment_id": "COMP-04",
            "equipment_type": "GAS_COMPRESSOR",
            "telemetry": {"vibration_rms": 6.2, "bearing_temp": 88.0},  # Degraded
            "historical_anomalies_30d": 2,
            "cumulative_run_hours": 32000.0,
            "days_since_last_pm": 95,
        },
    ]

    summary = health_engine.evaluate_fleet(fleet)

    assert isinstance(summary, FleetHealthSummary)
    assert summary.fleet_size == 4
    assert 50.0 < summary.mean_fleet_health < 85.0
    assert summary.critical_assets_count >= 1

    # Priority dispatch queue must contain degraded/critical assets, with worst asset first
    assert "ESP-03" in summary.priority_dispatch_queue
    assert summary.priority_dispatch_queue[0] == "ESP-03"


def test_rag_context_and_markdown_generation(health_engine):
    report = health_engine.evaluate_health(
        equipment_id="WELL-09-ESP",
        equipment_type="ESP",
        current_telemetry={
            "vibration_rms": 4.9,
            "bearing_temp": 86.0,
            "motor_current": 47.0,
            "discharge_pressure": 11.0,
        },
        historical_anomalies_30d=1,
        cumulative_run_hours=16500.0,
        days_since_last_pm=120,
    )

    # 1. RAG context block
    rag_ctx = report.to_rag_context()
    assert "<EQUIPMENT_HEALTH_GROUNDING>" in rag_ctx
    assert "</EQUIPMENT_HEALTH_GROUNDING>" in rag_ctx
    assert "WELL-09-ESP" in rag_ctx
    assert "Composite Health Index:" in rag_ctx
    assert "Telemetry Stress Score:" in rag_ctx
    assert "Remaining Useful Life (RUL):" in rag_ctx

    # 2. Markdown card
    md = report.to_markdown()
    assert "## Asset Health Assessment: `WELL-09-ESP`" in md
    assert "Composite Health Index (EHI):" in md
    assert "Sub-Index Breakdown" in md
    assert "Prescribed Interventions" in md
