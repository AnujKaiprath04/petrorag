"""
Unit Tests for PetroRAG Module 3.13 - RAG-Enhanced Anomaly Analysis Service
Validates integration of Module 3.12 Factual Anomaly Explanations with Part 2 RAG hybrid retrieval,
targeted query formulation, safety clearance & LOTO isolation rules, and multi-source action plans.
"""

import pytest
import asyncio
from datetime import datetime

from src.pipeline import PetroRAGPipeline
from src.generation.llm_provider import MockLLMProvider
from src.analytics.anomaly.rag_enhancer import (
    RAGEnhancedAnomalyService,
    RAGEnhancedAnomalyReport,
    MockAnomalyRetriever,
)
from src.analytics.anomaly.explainer import (
    AnomalyExplanationFormatter,
    FactualAnomalyExplanation,
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


@pytest.fixture
def test_rag_pipeline():
    """In-memory deterministic test pipeline avoiding Qdrant local disk lock."""
    return PetroRAGPipeline(
        retriever=MockAnomalyRetriever(),
        llm_provider=MockLLMProvider(
            default_response="Centrifugal compressor and ESP operating limit: Vibration shutdown threshold is 7.1 mm/s RMS (ISO 10816-3 Zone D). Verify NPSHa against cavitation."
        ),
    )


@pytest.fixture
def rag_service(test_rag_pipeline):
    return RAGEnhancedAnomalyService(rag_pipeline=test_rag_pipeline)


@pytest.mark.asyncio
async def test_rag_enhanced_anomaly_service_execution(baseline_stats, rag_service):
    formatter = AnomalyExplanationFormatter(default_equipment_id="WELL-07-ESP")

    # Cavitation telemetry signature
    telemetry = {
        "rpm": 2980.0,
        "motor_current": 48.5,  # Surge
        "discharge_pressure": 9.5,  # Dropout
        "suction_pressure": 1.5,
        "bearing_temp": 69.0,
        "vibration_rms": 3.9,
    }

    explanation = formatter.explain_point(
        features_dict=telemetry,
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.82,
        is_anomaly=True,
        equipment_id="WELL-07-ESP",
    )

    report = await rag_service.enhance_explanation(explanation, top_k_chunks=2)

    assert isinstance(report, RAGEnhancedAnomalyReport)
    assert report.equipment_id == "WELL-07-ESP"
    assert report.report_id.startswith("RAG-ANOM-")
    assert report.factual_explanation.operational_remediation.diagnosed_failure_mode == "PUMP_CAVITATION_OR_VAPOR_LOCK"

    # Verify RAG retrieved evidence
    assert isinstance(report.retrieved_evidence, list)
    assert len(report.retrieved_evidence) >= 1
    assert hasattr(report.retrieved_evidence[0], "document_title")

    # Verify action plan contains both physical SOPs and lessons learned
    assert len(report.integrated_action_plan) >= 3
    assert any("NPSH" in step or "strainer" in step for step in report.integrated_action_plan)

    # Verify grounded diagnostic statement
    assert "PUMP_CAVITATION_OR_VAPOR_LOCK" in report.grounded_diagnostic_statement
    assert report.overall_confidence > 0.0


def test_rag_enhanced_anomaly_service_sync_wrapper(baseline_stats, rag_service):
    formatter = AnomalyExplanationFormatter(default_equipment_id="COMPRESSOR-C02")

    # Severe vibration trip telemetry
    telemetry = {
        "rpm": 2980.0,
        "motor_current": 42.0,
        "discharge_pressure": 14.8,
        "suction_pressure": 2.4,
        "bearing_temp": 68.5,
        "vibration_rms": 8.9,  # Zone D Trip
    }

    explanation = formatter.explain_point(
        features_dict=telemetry,
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.91,
        is_anomaly=True,
        equipment_id="COMPRESSOR-C02",
    )

    report = rag_service.enhance_explanation_sync(explanation, top_k_chunks=2)

    assert isinstance(report, RAGEnhancedAnomalyReport)
    assert report.equipment_id == "COMPRESSOR-C02"
    assert report.loto_required is True
    assert report.permit_to_work_required is True
    assert "lockout/tagout" in report.integrated_action_plan[0].lower()


def test_targeted_query_formulation(baseline_stats, rag_service):
    formatter = AnomalyExplanationFormatter(default_equipment_id="PUMP-P101")

    explanation = formatter.explain_point(
        features_dict={
            "rpm": 2980.0,
            "motor_current": 42.0,
            "discharge_pressure": 14.8,
            "suction_pressure": 2.4,
            "bearing_temp": 97.0,  # Critical API trip
            "vibration_rms": 2.1,
        },
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.89,
        is_anomaly=True,
        equipment_id="PUMP-P101",
    )

    query = rag_service._formulate_targeted_query(explanation)
    assert "PUMP-P101" in query
    assert "THERMAL OVERHEAT" in query or "LUBRICATION" in query
    assert "bearing temp" in query
    assert "API 610" in query or "Machinery" in query


def test_safety_clearance_and_loto_logic(baseline_stats, rag_service):
    formatter = AnomalyExplanationFormatter()

    # 1. Critical Trip Condition -> LOTO Required
    expl_trip = formatter.explain_point(
        features_dict={
            "rpm": 2980.0,
            "motor_current": 42.0,
            "discharge_pressure": 14.8,
            "suction_pressure": 2.4,
            "bearing_temp": 98.0,  # Critical trip
            "vibration_rms": 2.1,
        },
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.90,
        is_anomaly=True,
    )
    rep_trip = rag_service.enhance_explanation_sync(expl_trip)
    assert rep_trip.loto_required is True
    assert rep_trip.permit_to_work_required is True
    assert "LOTO" in rep_trip.integrated_action_plan[0]

    # 2. Zone C Warning Alert -> PTW Required, but NOT LOTO
    expl_alert = formatter.explain_point(
        features_dict={
            "rpm": 2980.0,
            "motor_current": 42.0,
            "discharge_pressure": 14.8,
            "suction_pressure": 2.4,
            "bearing_temp": 68.0,
            "vibration_rms": 5.2,  # Zone C Alert
        },
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.65,
        is_anomaly=True,
    )
    rep_alert = rag_service.enhance_explanation_sync(expl_alert)
    assert rep_alert.loto_required is False
    assert rep_alert.permit_to_work_required is True
    assert "Permit to Work" in rep_alert.integrated_action_plan[0]

    # 3. Fully nominal -> Neither required
    expl_nom = formatter.explain_point(
        features_dict={
            "rpm": 2980.0,
            "motor_current": 42.0,
            "discharge_pressure": 14.8,
            "suction_pressure": 2.4,
            "bearing_temp": 68.0,
            "vibration_rms": 2.1,
        },
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.10,
        is_anomaly=False,
    )
    rep_nom = rag_service.enhance_explanation_sync(expl_nom)
    assert rep_nom.loto_required is False
    assert rep_nom.permit_to_work_required is False


@pytest.mark.asyncio
async def test_end_to_end_telemetry_point_analysis(baseline_stats, rag_service):
    report = await rag_service.analyze_telemetry_point(
        features_dict={
            "rpm": 2980.0,
            "motor_current": 25.0,  # Decoupling drop
            "discharge_pressure": 5.0,  # Severe drop
            "suction_pressure": 2.4,
            "bearing_temp": 68.0,
            "vibration_rms": 2.1,
        },
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.83,
        is_anomaly=True,
        equipment_id="WELL-04-PUMP",
    )

    assert isinstance(report, RAGEnhancedAnomalyReport)
    assert report.equipment_id == "WELL-04-PUMP"
    assert report.factual_explanation.operational_remediation.diagnosed_failure_mode == "FLUID_STARVATION_OR_DRIVE_DECOUPLING"
    assert len(report.retrieved_evidence) >= 1


def test_report_formatting_markdown_and_executive_summary(baseline_stats, rag_service):
    formatter = AnomalyExplanationFormatter(default_equipment_id="FPSO-BOOSTER-B1")

    explanation = formatter.explain_point(
        features_dict={
            "rpm": 2980.0,
            "motor_current": 42.0,
            "discharge_pressure": 14.8,
            "suction_pressure": 2.4,
            "bearing_temp": 68.0,
            "vibration_rms": 7.8,  # Zone D Trip
        },
        baseline_medians=baseline_stats["medians"],
        baseline_iqrs=baseline_stats["iqrs"],
        anomaly_score=0.87,
        is_anomaly=True,
        equipment_id="FPSO-BOOSTER-B1",
    )

    report = rag_service.enhance_explanation_sync(explanation)

    # 1. Executive Summary
    summary = report.to_executive_summary()
    assert "FPSO-BOOSTER-B1" in summary
    assert "OPERATIONAL ALERT" in summary
    assert "vibration_rms" in summary
    assert "MANDATORY LOTO ISOLATION" in summary

    # 2. Markdown Report
    md = report.to_markdown()
    assert "# PetroRAG Operational Intelligence Directive" in md
    assert "### 1. Telemetry Observations & Physical Baselines" in md
    assert "### 2. Safety Standards Compliance" in md
    assert "### 3. Retrieved Technical Documentation & OEM Guidance" in md
    assert "### 4. Integrated Action & Remediation Plan" in md
    assert "vibration_rms" in md
    assert "ZONE_D_TRIP" in md
