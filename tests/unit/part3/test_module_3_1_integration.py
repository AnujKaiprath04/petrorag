"""
PetroRAG Module 3.1 Unit Test Suite — Architecture Integration
Verifies database schema creation, abstract service interfaces, concrete service implementations,
and the OperationalRAGService bridge connecting structured ML signals to Part 2 RAG retrieval.
"""

import pytest
from datetime import datetime, date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.analytics.db import (
    Base,
    WellModel,
    EquipmentModel,
    SensorDataModel,
    init_db,
)
from src.analytics.interfaces import (
    ProductionMetricsRequest,
    AnomalyDetectionRequest,
    ForecastingRequest,
    EquipmentHealthRequest,
    IncidentSearchRequest,
    RiskAssessmentRequest,
)
from src.analytics.services import (
    ProductionAnalyticsService,
    AnomalyDetectionService,
    ForecastingService,
    EquipmentHealthService,
    IncidentIntelligenceService,
    RiskAssessmentService,
)
from src.operational.operational_rag import (
    OperationalRAGService,
    OperationalDiagnosticRequest,
)
from src.pipeline import PetroRAGPipeline
from src.retrieval.vector import MockEmbeddingService
from src.generation.llm_provider import MockLLMProvider


@pytest.fixture
def in_memory_db():
    """Create a temporary in-memory SQLite engine for testing."""
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_database_models_and_session(in_memory_db):
    """Verify ORM models, relations, and persistence in relational store."""
    well = WellModel(
        well_id="WELL-A1",
        well_name="North Field Alpha-1",
        field_name="North Field",
        reservoir="Khuff Gas",
        status="ACTIVE"
    )
    in_memory_db.add(well)
    in_memory_db.commit()

    equipment = EquipmentModel(
        equipment_id="EQ-C101",
        well_id="WELL-A1",
        tag_name="C-101",
        equipment_type="COMPRESSOR",
        vibration_alarm_threshold_mms=4.5,
        status="OPERATIONAL"
    )
    in_memory_db.add(equipment)
    in_memory_db.commit()

    sensor = SensorDataModel(
        timestamp=datetime.utcnow(),
        equipment_id="EQ-C101",
        suction_pressure_bar=15.2,
        discharge_pressure_bar=42.5,
        temperature_c=65.0,
        vibration_rms_mms=2.1,
    )
    in_memory_db.add(sensor)
    in_memory_db.commit()

    # Query back
    retrieved_eq = in_memory_db.query(EquipmentModel).filter_by(tag_name="C-101").first()
    assert retrieved_eq is not None
    assert retrieved_eq.well.field_name == "North Field"
    assert len(retrieved_eq.sensor_records) == 1
    assert retrieved_eq.sensor_records[0].discharge_pressure_bar == 42.5


def test_production_analytics_service():
    """Verify production analytics computes decline, water cut, and volumes."""
    service = ProductionAnalyticsService()
    req = ProductionMetricsRequest(well_id="WELL-01", window_days=30)
    res = service.calculate_production_metrics(req)

    assert res.well_id == "WELL-01"
    assert res.total_oil_bbl > 0.0
    assert res.total_gas_mscf > 0.0
    assert res.average_oil_rate_bopd > 0.0
    assert res.water_cut_trend in ("INCREASING", "STABLE", "DECREASING")
    assert "WELL-01" in res.observation_summary


def test_anomaly_detection_service_multivariate():
    """Verify Isolation Forest flags injected vibration/pressure anomalies."""
    service = AnomalyDetectionService()

    # Normal baseline readings
    readings = []
    base_time = datetime(2026, 3, 1, 10, 0)
    for i in range(25):
        readings.append({
            "timestamp": base_time.isoformat(),
            "pressure": 40.0 + (i % 3),
            "temperature": 60.0 + (i % 2),
            "vibration": 2.0 + (i % 2) * 0.2,
            "flow_rate": 100.0,
        })

    # Injected severe vibration & pressure spike anomaly
    readings.append({
        "timestamp": datetime(2026, 3, 1, 12, 0).isoformat(),
        "pressure": 75.0,     # severe spike
        "temperature": 115.0, # high temp
        "vibration": 8.5,     # critical vibration
        "flow_rate": 40.0,
    })

    req = AnomalyDetectionRequest(
        entity_id="C-101",
        entity_type="EQUIPMENT",
        data=readings,
        detection_method="ISOLATION_FOREST",
        sensitivity=0.08,
    )
    res = service.detect_anomalies(req)

    assert res.entity_id == "C-101"
    assert res.total_points_analyzed == 26
    assert res.anomalies_detected >= 1
    assert res.highest_severity in ("HIGH", "CRITICAL")

    # Injected spike should have high anomaly score
    last_item = res.records[-1]
    assert last_item.is_anomaly is True
    assert last_item.anomaly_score > 0.50
    assert last_item.severity in ("HIGH", "CRITICAL")


def test_forecasting_service():
    """Verify chronological validation and multi-day forecasting output."""
    service = ForecastingService()

    history = []
    base_val = 1500.0
    for i in range(40):
        history.append({"oil_rate_bopd": base_val - i * 5.0})

    req = ForecastingRequest(
        entity_id="WELL-02",
        target_metric="oil_rate_bopd",
        historical_data=history,
        horizon_days=14,
        model_name="ARIMA",
    )
    res = service.generate_forecast(req)

    assert res.entity_id == "WELL-02"
    assert res.horizon_days == 14
    assert len(res.forecast_dates) == 14
    assert len(res.predicted_values) == 14
    assert res.mae is not None
    assert res.rmse is not None
    assert res.mae >= 0.0


def test_equipment_health_service():
    """Verify equipment health index scoring and status triage."""
    service = EquipmentHealthService()

    # Case 1: Healthy equipment
    req_healthy = EquipmentHealthRequest(
        equipment_id="C-101",
        current_sensor_telemetry={"vibration_rms_mms": 1.8, "temperature_c": 55.0, "discharge_pressure_bar": 42.0},
        historical_anomalies_count_30d=0,
        days_since_last_maintenance=45,
    )
    res_healthy = service.evaluate_health(req_healthy)
    assert res_healthy.health_index >= 80.0
    assert res_healthy.status == "HEALTHY"

    # Case 2: Degraded/Critical equipment
    req_bad = EquipmentHealthRequest(
        equipment_id="C-101",
        current_sensor_telemetry={"vibration_rms_mms": 7.9, "temperature_c": 125.0, "discharge_pressure_bar": 42.0},
        historical_anomalies_count_30d=4,
        days_since_last_maintenance=210,
    )
    res_bad = service.evaluate_health(req_bad)
    assert res_bad.health_index < 60.0
    assert res_bad.status in ("DEGRADED", "CRITICAL")
    assert len(res_bad.primary_stress_factors) >= 2


def test_incident_intelligence_service():
    """Verify semantic incident search returns relevant failure modes."""
    mock_embedder = MockEmbeddingService(dim=768)
    service = IncidentIntelligenceService(embedding_service=mock_embedder)

    req = IncidentSearchRequest(
        query_description="compressor high vibration bearing wear",
        top_k=2,
    )
    res = service.find_similar_incidents(req)

    assert res.total_matches_found == 2
    # Ensure returned incident has structured root causes and actions
    top_inc = res.matches[0]
    assert top_inc.incident_id is not None
    assert top_inc.root_cause is not None
    assert top_inc.action_taken is not None


def test_risk_assessment_service():
    """Verify reproducible P * S * E risk score calculation."""
    service = RiskAssessmentService()

    # Normal low-risk condition
    low_req = RiskAssessmentRequest(
        entity_id="C-101",
        active_anomaly_score=0.10,
        consequence_severity=2,
        condition_duration_hours=1.0,
    )
    low_res = service.assess_risk(low_req)
    assert low_res.risk_level == "LOW"
    assert low_res.risk_score <= 18
    assert low_res.probability == 1

    # High-risk persistent condition
    high_req = RiskAssessmentRequest(
        entity_id="C-101",
        active_anomaly_score=0.92,
        consequence_severity=5,
        condition_duration_hours=72.0,
        system_redundancy=False,
    )
    high_res = service.assess_risk(high_req)
    assert high_res.risk_level == "CRITICAL"
    assert high_res.risk_score >= 80
    assert high_res.probability == 5
    assert high_res.severity == 5
    assert high_res.exposure == 5


class MockTestRetriever:
    """Deterministic in-memory retriever for operational integration testing."""

    def __init__(self):
        from src.core.interfaces import RetrievedChunk, RetrievalChannel
        self.sample_chunks = [
            RetrievedChunk(
                chunk_id="chk-c101-manual",
                document_id="DOC-C101-SOP",
                document_title="Centrifugal Compressor C-101 OEM Operating Manual.pdf",
                text="Compressor C-101 vibration shutdown setpoint is 7.1 mm/s RMS per ISO 10816-3 Zone D. If vibration exceeds 4.5 mm/s, inspect bearings.",
                score=0.95,
                page_number=28,
                section_title="5.3 Vibration Limits",
                channel=RetrievalChannel.HYBRID,
            )
        ]

    def retrieve(self, query: str, top_k: int = 10, filters=None):
        return self.sample_chunks

    async def aretrieve(self, query: str, top_k: int = 10, filters=None):
        return self.sample_chunks


@pytest.mark.asyncio
async def test_operational_rag_service_bridge():
    """
    Verify the operational RAG service bridges telemetry + ML signals
    with Part 2 RAG hybrid retrieval and evidence grounding.
    """
    mock_llm = MockLLMProvider(
        default_response="Centrifugal compressor C-101 vibration shutdown limit is 7.1 mm/s RMS according to ISO 10816-3 Zone D [1].",
        model_name="mock-petrogpt-op-rag"
    )
    test_pipeline = PetroRAGPipeline(
        retriever=MockTestRetriever(),
        llm_provider=mock_llm
    )
    op_rag = OperationalRAGService(rag_pipeline=test_pipeline)

    req = OperationalDiagnosticRequest(
        entity_id="C-101",
        entity_type="EQUIPMENT",
        symptom_description="High vibration at 7.8 mm/s exceeding ISO 10816-3 Zone D shutdown limits",
        observed_telemetry={"vibration_rms_mms": 7.8, "temperature_c": 85.0, "discharge_pressure_bar": 42.5},
        anomaly_score=0.88,
        anomaly_severity="HIGH",
        condition_duration_hours=4.0,
    )

    response = await op_rag.diagnose_condition(req)

    # Validate 4-part structured response
    assert response.entity_id == "C-101"
    assert "7.8" in response.observed_condition
    assert response.ml_signals["anomaly_severity"] == "HIGH"
    assert response.ml_signals["risk_score"] > 0
    assert len(response.retrieved_technical_evidence) > 0
    assert len(response.recommended_diagnostic_actions) > 0
    assert len(response.historical_similar_incidents) > 0
    assert "[OBSERVED]" in response.explanation_statement
    assert "[ML SIGNALS]" in response.explanation_statement
    assert "[EVIDENCE]" in response.explanation_statement
    assert "[RECOMMENDED CHECKS]" in response.explanation_statement
