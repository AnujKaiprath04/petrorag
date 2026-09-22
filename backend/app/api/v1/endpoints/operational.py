"""
PetroRAG Operational Intelligence REST API Router (Module 3.16)
Exposes operational diagnostic, anomaly explanation, equipment health,
fleet rollup, and interactive troubleshooting endpoints.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, List, Optional

from src.core.logging import logger
from src.operational.operational_rag import (
    OperationalRAGService,
    OperationalDiagnosticRequest,
    OperationalDiagnosticResponse,
)
from src.analytics.anomaly.explainer import (
    AnomalyExplanationFormatter,
    FactualAnomalyExplanation,
)
from src.analytics.health.engine import (
    EquipmentHealthEngine,
    ComprehensiveEquipmentHealthReport,
    FleetHealthSummary,
)
from src.analytics.troubleshooting.workflow import (
    TroubleshootingWorkflowService,
    TroubleshootingSession,
)
from backend.app.schemas.operational import (
    OperationalDiagnoseApiRequest,
    AnomalyExplainApiRequest,
    EquipmentHealthApiRequest,
    FleetHealthApiRequest,
    TroubleshootStartApiRequest,
    TroubleshootStepApiRequest,
    TroubleshootSessionApiResponse,
)

router = APIRouter(prefix="/operational", tags=["Operational Intelligence"])

# Shared service singletons (lazy loaded)
_op_rag_service: Optional[OperationalRAGService] = None
_explainer_formatter: Optional[AnomalyExplanationFormatter] = None
_health_engine: Optional[EquipmentHealthEngine] = None
_troubleshooting_service: Optional[TroubleshootingWorkflowService] = None


def get_operational_rag_service() -> OperationalRAGService:
    global _op_rag_service
    if _op_rag_service is None:
        from backend.app.api.v1.endpoints.rag import get_pipeline
        _op_rag_service = OperationalRAGService(rag_pipeline=get_pipeline())
    return _op_rag_service


def get_explainer_formatter() -> AnomalyExplanationFormatter:
    global _explainer_formatter
    if _explainer_formatter is None:
        _explainer_formatter = AnomalyExplanationFormatter()
    return _explainer_formatter


def get_health_engine() -> EquipmentHealthEngine:
    global _health_engine
    if _health_engine is None:
        _health_engine = EquipmentHealthEngine()
    return _health_engine


def get_troubleshooting_service() -> TroubleshootingWorkflowService:
    global _troubleshooting_service
    if _troubleshooting_service is None:
        _troubleshooting_service = TroubleshootingWorkflowService()
    return _troubleshooting_service


@router.post(
    "/diagnose",
    response_model=OperationalDiagnosticResponse,
    summary="Execute RAG-Enhanced Operational Diagnostic",
)
async def diagnose_telemetry(request: OperationalDiagnoseApiRequest) -> OperationalDiagnosticResponse:
    """
    Diagnoses active operational condition combining live telemetry,
    ML anomaly signals, technical RAG retrieval, and historical incident intelligence.
    """
    try:
        service = get_operational_rag_service()
        diag_req = OperationalDiagnosticRequest(
            entity_id=request.entity_id,
            entity_type=request.entity_type,
            symptom_description=request.symptom_description,
            observed_telemetry=request.observed_telemetry,
            anomaly_score=request.anomaly_score,
            anomaly_severity=request.anomaly_severity,
            condition_duration_hours=request.condition_duration_hours,
        )
        return await service.diagnose_condition(diag_req)
    except Exception as e:
        logger.error(f"Failed to execute operational diagnosis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Operational diagnosis execution failed: {str(e)}",
        )


@router.post(
    "/anomalies/explain",
    response_model=FactualAnomalyExplanation,
    summary="Generate 4-Part Factual Anomaly Explanation",
)
def explain_anomaly(request: AnomalyExplainApiRequest) -> FactualAnomalyExplanation:
    """
    Constructs a deterministic, auditable 4-part explanation (Observations,
    Statistical Confidence, ISO/API Safety Envelopes, Remedial Actions) for an anomalous telemetry reading.
    """
    try:
        # Default baselines if not provided
        medians = request.baseline_medians or {
            "vibration_rms": 2.1,
            "bearing_temp": 68.0,
            "motor_current": 42.0,
            "discharge_pressure": 14.8,
            "suction_pressure": 2.4,
            "rpm": 2980.0,
        }
        iqrs = request.baseline_iqrs or {
            "vibration_rms": 0.15,
            "bearing_temp": 0.6,
            "motor_current": 0.8,
            "discharge_pressure": 0.35,
            "suction_pressure": 0.12,
            "rpm": 5.0,
        }

        formatter = get_explainer_formatter()
        return formatter.explain_point(
            features_dict=request.telemetry_readings,
            baseline_medians=medians,
            baseline_iqrs=iqrs,
            anomaly_score=request.anomaly_score,
            is_anomaly=request.is_anomaly,
            ensemble_votes=request.ensemble_votes,
            equipment_id=request.equipment_id,
        )
    except Exception as e:
        logger.error(f"Failed to generate anomaly explanation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly explanation failed: {str(e)}",
        )


@router.post(
    "/health/evaluate",
    response_model=ComprehensiveEquipmentHealthReport,
    summary="Evaluate Multi-Factor Equipment Health Index",
)
def evaluate_equipment_health(request: EquipmentHealthApiRequest) -> ComprehensiveEquipmentHealthReport:
    """
    Computes composite 0-100 Equipment Health Index (EHI) integrating active telemetry stress,
    historical anomaly frequency, Weibull cumulative age wear, and PM compliance.
    """
    try:
        engine = get_health_engine()
        return engine.evaluate_health(
            equipment_id=request.equipment_id,
            equipment_type=request.equipment_type,
            current_telemetry=request.current_telemetry,
            historical_anomalies_30d=request.historical_anomalies_30d,
            anomaly_severity_counts=request.anomaly_severity_counts,
            cumulative_run_hours=request.cumulative_run_hours,
            days_since_last_pm=request.days_since_last_pm,
            historical_ehi_trajectory=request.historical_ehi_trajectory,
        )
    except Exception as e:
        logger.error(f"Failed to evaluate equipment health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health evaluation failed: {str(e)}",
        )


@router.post(
    "/health/fleet",
    response_model=FleetHealthSummary,
    summary="Rollup Fleet-Wide Health and Priority Dispatch Queue",
)
def evaluate_fleet_health(request: FleetHealthApiRequest) -> FleetHealthSummary:
    """
    Aggregates asset health across an entire facility/field and generates
    a priority maintenance dispatch queue.
    """
    try:
        engine = get_health_engine()
        fleet_dicts = [
            {
                "equipment_id": rec.equipment_id,
                "equipment_type": rec.equipment_type,
                "telemetry": rec.current_telemetry,
                "historical_anomalies_30d": rec.historical_anomalies_30d,
                "anomaly_severity_counts": rec.anomaly_severity_counts,
                "cumulative_run_hours": rec.cumulative_run_hours,
                "days_since_last_pm": rec.days_since_last_pm,
            }
            for rec in request.fleet_records
        ]
        return engine.evaluate_fleet(fleet_dicts)
    except Exception as e:
        logger.error(f"Failed to evaluate fleet health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fleet health rollup failed: {str(e)}",
        )


@router.post(
    "/troubleshoot/start",
    response_model=TroubleshootSessionApiResponse,
    summary="Initiate Interactive Equipment Troubleshooting Session",
)
def start_troubleshooting(request: TroubleshootStartApiRequest) -> TroubleshootSessionApiResponse:
    """
    Initializes a new stateful equipment troubleshooting session and returns
    the initial root decision node.
    """
    try:
        service = get_troubleshooting_service()
        session = service.start_session(
            equipment_id=request.equipment_id,
            equipment_type=request.equipment_type,
        )
        return TroubleshootSessionApiResponse(
            session_id=session.session_id,
            equipment_id=session.equipment_id,
            equipment_type=session.equipment_type,
            is_completed=session.is_completed,
            current_active_node=session.get_current_node(),
            resolution=session.resolution,
            steps_executed_count=len(session.step_history),
            rag_context_block=session.to_rag_context(),
        )
    except Exception as e:
        logger.error(f"Failed to start troubleshooting session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start troubleshooting session: {str(e)}",
        )


@router.post(
    "/troubleshoot/step",
    response_model=TroubleshootSessionApiResponse,
    summary="Advance Troubleshooting Gate with Operator Selection",
)
def advance_troubleshooting_step(request: TroubleshootStepApiRequest) -> TroubleshootSessionApiResponse:
    """
    Executes an operator decision step, verifies safety clearance holds (PTW/LOTO),
    and transitions to the next diagnostic gate or terminal root cause resolution.
    """
    service = get_troubleshooting_service()
    session = service.get_session(request.session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Troubleshooting session '{request.session_id}' not found.",
        )

    ok, msg = session.select_option(
        option_key=request.option_key,
        operator_notes=request.operator_notes,
        safety_confirmed=request.safety_confirmed,
    )

    if not ok and not session.is_completed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg,
        )

    return TroubleshootSessionApiResponse(
        session_id=session.session_id,
        equipment_id=session.equipment_id,
        equipment_type=session.equipment_type,
        is_completed=session.is_completed,
        current_active_node=session.get_current_node(),
        resolution=session.resolution,
        steps_executed_count=len(session.step_history),
        rag_context_block=session.to_rag_context(),
    )
