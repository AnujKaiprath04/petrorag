"""
Unit Tests for PetroRAG Module 3.12 - Anomaly Factual Explanation Formatter
Validates 4-part operational explanation schemas (Observations, Confidence, Safety, Remediation),
ISO 10816-3 & API 610 safety envelope compliance, multi-parameter physical failure mode diagnosis,
multi-model consensus evaluation, and LLM RAG prompt context generation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.analytics.anomaly.explainer import (
    AnomalyExplanationFormatter,
    EnsembleAnomalyExplainerService,
    FactualAnomalyExplanation,
    TelemetryObservationSection,
    StatisticalConfidenceSection,
    SafetyEnvelopeSection,
    OperationalRemediationSection,
    ENGINEERING_UNITS,
)
from src.analytics.anomaly.attribution import (
    AnomalyAttributionEngine,
    DecomposedAnomalyReport,
    ParameterAttributionItem,
    SeverityTriageResult,
)


@pytest.fixture
def baseline_stats():
    return {
        "medians": {
            "rpm": 2980.0,
            "motor_current": 42.0,
            "discharge_pressure": 14.8,
            "suction_pressure": 2.4,
            "bearing_temp": 68.0,
            "vibration_rms": 2.1,
        },
        "iqrs": {
            "rpm": 5.0,
            "motor_current": 0.8,
            "discharge_pressure": 0.35,
            "suction_pressure": 0.12,
            "bearing_temp": 0.6,
            "vibration_rms": 0.15,
        },
    }


def test_four_part_structure_completeness_and_units(baseline_stats):
    formatter = AnomalyExplanationFormatter(default_equipment_id="WELL-07-ESP")

    # Construct point with vibration surge
    telemetry = {
        "rpm": 2982.0,
        "motor_current": 42.1,
        "discharge_pressure": 14.7,
        "suction_pressure": 2.39,
        "bearing_temp": 68.4,
        "vibration_rms": 8.6,  # severe spike
    }

    explanation = formatter.explain_point(
        features_dict=telemetry,
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.88,
        is_anomaly=True,
        equipment_id="WELL-07-ESP",
    )

    assert isinstance(explanation, FactualAnomalyExplanation)
    assert explanation.equipment_id == "WELL-07-ESP"
    assert explanation.is_anomaly is True
    assert explanation.overall_severity in ["HIGH", "CRITICAL"]

    # Part 1: Telemetry observations
    p1 = explanation.telemetry_observations
    assert isinstance(p1, TelemetryObservationSection)
    assert p1.primary_driver == "vibration_rms"
    assert len(p1.observations) == 6
    vib_obs = next(o for o in p1.observations if o.channel == "vibration_rms")
    assert vib_obs.unit == "mm/s RMS"
    assert vib_obs.observed_value == 8.6
    assert vib_obs.direction == "SURGE"
    assert vib_obs.is_primary_driver is True

    # Part 2: Statistical confidence
    p2 = explanation.statistical_confidence
    assert isinstance(p2, StatisticalConfidenceSection)
    assert p2.primary_anomaly_score == 0.88
    assert p2.confidence_tier in ["HIGH", "VERY_HIGH"]

    # Part 3: Safety envelope
    p3 = explanation.safety_envelope
    assert isinstance(p3, SafetyEnvelopeSection)
    assert p3.safety_breach_detected is True
    assert p3.iso_vibration_zone == "ZONE_D_TRIP"
    assert "ISO 10816-3 (Mechanical Vibration)" in p3.governing_standards

    # Part 4: Operational remediation
    p4 = explanation.operational_remediation
    assert isinstance(p4, OperationalRemediationSection)
    assert p4.urgency_level == "IMMEDIATE_SHUTDOWN"
    assert len(p4.recommended_sop_steps) >= 3


def test_iso_and_api_safety_boundary_triage(baseline_stats):
    formatter = AnomalyExplanationFormatter()

    # 1. Thermal runaway breaching API 610 trip boundary (>95°C)
    heat_telemetry = {
        "rpm": 2980.0,
        "motor_current": 42.0,
        "discharge_pressure": 14.8,
        "suction_pressure": 2.4,
        "bearing_temp": 99.5,  # API critical trip
        "vibration_rms": 2.1,  # Safe vibration
    }
    heat_expl = formatter.explain_point(
        features_dict=heat_telemetry,
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.92,
        is_anomaly=True,
    )
    assert heat_expl.safety_envelope.api_bearing_temp_status == "CRITICAL_TRIP"
    assert heat_expl.safety_envelope.highest_safety_tier == "CRITICAL_TRIP"
    assert heat_expl.operational_remediation.urgency_level == "IMMEDIATE_SHUTDOWN"
    assert heat_expl.operational_remediation.diagnosed_failure_mode == "THERMAL_OVERHEAT_LUBRICATION_DEGRADATION"

    # 2. Zone C Alert vibration (between 4.5 and 7.1 mm/s)
    alert_telemetry = {
        "rpm": 2980.0,
        "motor_current": 42.0,
        "discharge_pressure": 14.8,
        "suction_pressure": 2.4,
        "bearing_temp": 70.0,
        "vibration_rms": 5.4,  # Zone C
    }
    alert_expl = formatter.explain_point(
        features_dict=alert_telemetry,
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.68,
        is_anomaly=True,
    )
    assert alert_expl.safety_envelope.iso_vibration_zone == "ZONE_C_ALERT"
    assert alert_expl.operational_remediation.urgency_level == "URGENT_INSPECTION_24H"

    # 3. Fully nominal telemetry
    nominal_telemetry = {
        "rpm": 2980.0,
        "motor_current": 42.0,
        "discharge_pressure": 14.8,
        "suction_pressure": 2.4,
        "bearing_temp": 68.0,
        "vibration_rms": 2.1,
    }
    nominal_expl = formatter.explain_point(
        features_dict=nominal_telemetry,
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.10,
        is_anomaly=False,
    )
    assert nominal_expl.safety_envelope.safety_breach_detected is False
    assert nominal_expl.safety_envelope.highest_safety_tier == "SAFE"
    assert nominal_expl.operational_remediation.urgency_level == "ROUTINE_SURVEILLANCE"


def test_domain_diagnostic_failure_mode_matrix(baseline_stats):
    formatter = AnomalyExplanationFormatter()

    # Case A: High vibration + High temperature -> Bearing Degradation / Lubrication Failure
    case_a = formatter.explain_point(
        features_dict={
            "rpm": 2980.0,
            "motor_current": 44.0,
            "discharge_pressure": 14.5,
            "suction_pressure": 2.3,
            "bearing_temp": 92.0,  # Alert/High
            "vibration_rms": 6.8,  # Alert/High
        },
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.85,
        is_anomaly=True,
    )
    assert case_a.operational_remediation.diagnosed_failure_mode == "BEARING_DEGRADATION_OR_LUBRICATION_FAILURE"

    # Case B: High vibration + Normal temperature -> Mechanical Unbalance / Looseness
    case_b = formatter.explain_point(
        features_dict={
            "rpm": 2980.0,
            "motor_current": 42.0,
            "discharge_pressure": 14.7,
            "suction_pressure": 2.4,
            "bearing_temp": 68.0,  # Normal
            "vibration_rms": 6.2,  # Alert
        },
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.72,
        is_anomaly=True,
    )
    assert case_b.operational_remediation.diagnosed_failure_mode == "MECHANICAL_UNBALANCE_OR_LOOSENESS"

    # Case C: Pressure drop + Current surge -> Pump Cavitation / Vapor Lock
    case_c = formatter.explain_point(
        features_dict={
            "rpm": 2980.0,
            "motor_current": 49.5,  # Surge (+17.8%)
            "discharge_pressure": 9.2,  # Drop (-37.8%)
            "suction_pressure": 1.4,  # Drop
            "bearing_temp": 69.0,
            "vibration_rms": 3.8,
        },
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.81,
        is_anomaly=True,
    )
    assert case_c.operational_remediation.diagnosed_failure_mode == "PUMP_CAVITATION_OR_VAPOR_LOCK"

    # Case D: Pressure drop + Current drop -> Fluid Starvation / Decoupling
    case_d = formatter.explain_point(
        features_dict={
            "rpm": 2980.0,
            "motor_current": 28.0,  # Drop (-33%)
            "discharge_pressure": 6.5,  # Drop (-56%)
            "suction_pressure": 2.4,
            "bearing_temp": 68.0,
            "vibration_rms": 2.0,
        },
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.79,
        is_anomaly=True,
    )
    assert case_d.operational_remediation.diagnosed_failure_mode == "FLUID_STARVATION_OR_DRIVE_DECOUPLING"


def test_multi_model_consensus_evaluation(baseline_stats):
    formatter = AnomalyExplanationFormatter()

    features = {
        "rpm": 2980.0,
        "motor_current": 42.0,
        "discharge_pressure": 14.8,
        "suction_pressure": 2.4,
        "bearing_temp": 68.0,
        "vibration_rms": 8.5,
    }

    # Unanimous 4/4 agreement
    expl_unanimous = formatter.explain_point(
        features_dict=features,
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.90,
        is_anomaly=True,
        ensemble_votes={"IsolationForest": True, "RollingZScore": True, "TukeyIQR": True, "HampelMAD": True},
    )
    assert expl_unanimous.statistical_confidence.models_flagging_count == 4
    assert expl_unanimous.statistical_confidence.total_models_evaluated == 4
    assert expl_unanimous.statistical_confidence.consensus_ratio == 1.0
    assert expl_unanimous.statistical_confidence.confidence_tier == "VERY_HIGH"

    # Split 2/4 agreement
    expl_split = formatter.explain_point(
        features_dict=features,
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.55,
        is_anomaly=True,
        ensemble_votes={"IsolationForest": True, "RollingZScore": False, "TukeyIQR": True, "HampelMAD": False},
    )
    assert expl_split.statistical_confidence.models_flagging_count == 2
    assert expl_split.statistical_confidence.consensus_ratio == 0.50
    assert expl_split.statistical_confidence.confidence_tier in ["MEDIUM", "HIGH"]


def test_rag_context_and_text_report_generation(baseline_stats):
    formatter = AnomalyExplanationFormatter(default_equipment_id="OFFSHORE_ESP_PLATFORM_BRAVO")

    telemetry = {
        "rpm": 2980.0,
        "motor_current": 48.0,
        "discharge_pressure": 10.5,
        "suction_pressure": 1.6,
        "bearing_temp": 71.0,
        "vibration_rms": 4.8,
    }

    explanation = formatter.explain_point(
        features_dict=telemetry,
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.76,
        is_anomaly=True,
        ensemble_votes={"IsolationForest": True, "RollingZScore": True, "TukeyIQR": True, "HampelMAD": False},
        equipment_id="OFFSHORE_ESP_PLATFORM_BRAVO",
    )

    # 1. Text bulletin
    text_rep = explanation.to_text_report()
    assert "PETRORAG OPERATIONAL ANOMALY BULLETIN" in text_rep
    assert "OFFSHORE_ESP_PLATFORM_BRAVO" in text_rep
    assert "1. TELEMETRY OBSERVATIONS:" in text_rep
    assert "2. STATISTICAL & MODEL CONFIDENCE:" in text_rep
    assert "3. SAFETY ENVELOPE STATUS:" in text_rep
    assert "4. OPERATIONAL RISK & RECOMMENDED ACTIONS:" in text_rep
    assert "Recommended SOP Checklist:" in text_rep

    # 2. RAG Prompt grounding
    rag_ctx = explanation.to_rag_context()
    assert "<FACTUAL_ANOMALY_GROUNDING>" in rag_ctx
    assert "</FACTUAL_ANOMALY_GROUNDING>" in rag_ctx
    assert "[SECTION 1: TELEMETRY READINGS]" in rag_ctx
    assert "[SECTION 2: MODEL CONFIDENCE]" in rag_ctx
    assert "[SECTION 3: SAFETY ENVELOPE]" in rag_ctx
    assert "[SECTION 4: DIAGNOSTIC REMEDIATION]" in rag_ctx
    assert "Primary Contributing Sensor:" in rag_ctx


def test_ensemble_explainer_service_end_to_end():
    # Build baseline training telemetry
    np.random.seed(42)
    n = 200
    df_train = pd.DataFrame(
        {
            "rpm": 2980.0 + np.random.normal(0, 3.0, n),
            "motor_current": 42.0 + np.random.normal(0, 0.5, n),
            "discharge_pressure": 14.8 + np.random.normal(0, 0.2, n),
            "suction_pressure": 2.4 + np.random.normal(0, 0.08, n),
            "bearing_temp": 68.0 + np.random.normal(0, 0.4, n),
            "vibration_rms": 2.1 + np.random.normal(0, 0.1, n),
        }
    )

    service = EnsembleAnomalyExplainerService(default_equipment_id="ESP_WELL_03")
    service.fit(df_train)

    assert service.is_fitted is True
    assert len(service.feature_names) == 6

    # Test anomalous row with severe vibration
    anom_row = pd.Series(
        {
            "rpm": 2981.0,
            "motor_current": 42.2,
            "discharge_pressure": 14.7,
            "suction_pressure": 2.38,
            "bearing_temp": 68.2,
            "vibration_rms": 9.1,  # severe spike
        }
    )

    expl = service.explain_dataframe_row(
        row=anom_row,
        timestamp="2025-06-15T14:30:00Z",
        index=105,
    )

    assert isinstance(expl, FactualAnomalyExplanation)
    assert expl.is_anomaly is True
    assert expl.telemetry_observations.primary_driver == "vibration_rms"
    assert expl.safety_envelope.iso_vibration_zone == "ZONE_D_TRIP"
    assert expl.operational_remediation.urgency_level == "IMMEDIATE_SHUTDOWN"
    assert "IsolationForest" in expl.statistical_confidence.ensemble_votes
