"""
Unit Tests for PetroRAG Module 3.16 - Operational REST API Router & Unified Endpoints
Validates FastAPI endpoints for operational diagnosis, anomaly explanation,
equipment health assessment, fleet rollups, and interactive troubleshooting state machine.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from src.pipeline import PetroRAGPipeline
from src.generation.llm_provider import MockLLMProvider
from src.analytics.anomaly.rag_enhancer import MockAnomalyRetriever
from backend.app.api.v1.endpoints import operational


@pytest.fixture(autouse=True)
def inject_test_pipeline(monkeypatch):
    """Ensure in-memory RAG pipeline is used for API tests to avoid Qdrant local disk locking."""
    test_pipeline = PetroRAGPipeline(
        retriever=MockAnomalyRetriever(),
        llm_provider=MockLLMProvider(
            default_response="OEM Operating Guidance: Centrifugal compressor C-101 vibration trip threshold is 7.1 mm/s RMS."
        ),
    )
    op_service = operational.OperationalRAGService(rag_pipeline=test_pipeline)
    monkeypatch.setattr(operational, "get_operational_rag_service", lambda: op_service)


@pytest.fixture
def client():
    return TestClient(app)


def test_operational_diagnose_endpoint(client):
    payload = {
        "entity_id": "C-101",
        "entity_type": "COMPRESSOR",
        "symptom_description": "Vibration surge and discharge pressure loss",
        "observed_telemetry": {
            "vibration_rms": 7.9,
            "bearing_temp": 88.0,
            "discharge_pressure": 41.5,
        },
        "anomaly_score": 0.89,
        "anomaly_severity": "CRITICAL",
        "condition_duration_hours": 3.0,
    }

    response = client.post("/api/v1/operational/diagnose", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["entity_id"] == "C-101"
    assert "observed_condition" in data
    assert "ml_signals" in data
    assert data["ml_signals"]["anomaly_severity"] == "CRITICAL"
    assert len(data["retrieved_technical_evidence"]) >= 1
    assert len(data["recommended_diagnostic_actions"]) >= 2
    assert "explanation_statement" in data


def test_anomaly_explain_endpoint(client):
    payload = {
        "equipment_id": "ESP-WELL-02",
        "telemetry_readings": {
            "rpm": 2980.0,
            "motor_current": 49.2,
            "discharge_pressure": 8.4,
            "suction_pressure": 1.5,
            "bearing_temp": 68.0,
            "vibration_rms": 3.9,
        },
        "anomaly_score": 0.81,
        "is_anomaly": True,
    }

    response = client.post("/api/v1/operational/anomalies/explain", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["equipment_id"] == "ESP-WELL-02"
    assert data["is_anomaly"] is True
    assert "telemetry_observations" in data
    assert "statistical_confidence" in data
    assert "safety_envelope" in data
    assert "operational_remediation" in data
    assert data["operational_remediation"]["diagnosed_failure_mode"] == "PUMP_CAVITATION_OR_VAPOR_LOCK"


def test_equipment_health_evaluate_endpoint(client):
    payload = {
        "equipment_id": "PUMP-101",
        "equipment_type": "CENTRIFUGAL_PUMP",
        "current_telemetry": {
            "vibration_rms": 1.9,
            "bearing_temp": 65.0,
            "motor_current": 42.0,
        },
        "historical_anomalies_30d": 0,
        "cumulative_run_hours": 4500.0,
        "days_since_last_pm": 25,
    }

    response = client.post("/api/v1/operational/health/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["equipment_id"] == "PUMP-101"
    assert data["composite_health_index"] >= 85.0
    assert data["health_status"] == "HEALTHY"
    assert data["sub_indices"]["telemetry_stress_score"] >= 90.0
    assert data["rul_forecast"]["rul_days_to_critical"] > 100.0
    assert len(data["recommended_interventions"]) >= 1


def test_fleet_health_rollup_endpoint(client):
    payload = {
        "fleet_records": [
            {
                "equipment_id": "PUMP-A",
                "equipment_type": "CENTRIFUGAL_PUMP",
                "current_telemetry": {"vibration_rms": 1.8},
                "cumulative_run_hours": 3000.0,
                "days_since_last_pm": 20,
            },
            {
                "equipment_id": "PUMP-B",
                "equipment_type": "CENTRIFUGAL_PUMP",
                "current_telemetry": {"vibration_rms": 8.5, "bearing_temp": 98.0},
                "cumulative_run_hours": 38000.0,
                "days_since_last_pm": 180,
            },
        ]
    }

    response = client.post("/api/v1/operational/health/fleet", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["fleet_size"] == 2
    assert "mean_fleet_health" in data
    assert data["critical_assets_count"] >= 1
    assert "PUMP-B" in data["priority_dispatch_queue"]


def test_troubleshoot_session_lifecycle_endpoints(client):
    # 1. Start session
    start_res = client.post(
        "/api/v1/operational/troubleshoot/start",
        json={"equipment_id": "ESP-WELL-09", "equipment_type": "ESP"},
    )
    assert start_res.status_code == 200
    start_data = start_res.json()

    session_id = start_data["session_id"]
    assert start_data["is_completed"] is False
    assert start_data["current_active_node"]["node_id"] == "ESP_ROOT"
    assert start_data["steps_executed_count"] == 0

    # 2. Step 1: Select High Vibration
    step1_res = client.post(
        "/api/v1/operational/troubleshoot/step",
        json={
            "session_id": session_id,
            "option_key": "VIBRATION_HIGH",
            "operator_notes": "Operator confirmed vibration 6.2 mm/s on field gauge",
            "safety_confirmed": True,
        },
    )
    assert step1_res.status_code == 200
    step1_data = step1_res.json()
    assert step1_data["is_completed"] is False
    assert step1_data["current_active_node"]["node_id"] == "ESP_VIB_CHECK"
    assert step1_data["steps_executed_count"] == 1

    # 3. Step 2: High Temp observed -> Terminal root cause
    step2_res = client.post(
        "/api/v1/operational/troubleshoot/step",
        json={
            "session_id": session_id,
            "option_key": "YES_HIGH_TEMP",
            "operator_notes": "Thermal RTD confirms 92C bearing temperature",
            "safety_confirmed": True,
        },
    )
    assert step2_res.status_code == 200
    step2_data = step2_res.json()
    assert step2_data["is_completed"] is True
    assert step2_data["current_active_node"] is None
    assert step2_data["resolution"] is not None
    assert step2_data["resolution"]["confirmed_root_cause"] == "JOURNAL_BEARING_DEGRADATION_OR_LUBE_FAILURE"
    assert step2_data["resolution"]["resolution_status"] == "RESOLVED"

    # 4. Error testing: Invalid session ID
    err_res = client.post(
        "/api/v1/operational/troubleshoot/step",
        json={
            "session_id": "NON-EXISTENT-SESSION",
            "option_key": "YES_HIGH_TEMP",
            "safety_confirmed": True,
        },
    )
    assert err_res.status_code == 404
