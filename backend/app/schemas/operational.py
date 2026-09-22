"""
PetroRAG Operational Intelligence API Schemas (Module 3.16)
Pydantic input and output payloads for operational REST endpoints.
"""

from typing import Dict, List, Any, Optional, Tuple, Literal
from datetime import datetime
from pydantic import BaseModel, Field

from src.analytics.anomaly.explainer import FactualAnomalyExplanation
from src.analytics.anomaly.rag_enhancer import RAGEnhancedAnomalyReport
from src.analytics.health.engine import (
    ComprehensiveEquipmentHealthReport,
    FleetHealthSummary,
)
from src.analytics.troubleshooting.workflow import (
    DecisionNode,
    TroubleshootingResolution,
)


class OperationalDiagnoseApiRequest(BaseModel):
    """Payload for RAG-enhanced operational telemetry diagnosis."""
    entity_id: str = Field(...)
    entity_type: str = Field("EQUIPMENT")
    symptom_description: str = Field(...)
    observed_telemetry: Dict[str, float] = Field(default_factory=dict)
    anomaly_score: Optional[float] = Field(None)
    anomaly_severity: Optional[str] = Field(None)
    condition_duration_hours: float = Field(1.0)


class AnomalyExplainApiRequest(BaseModel):
    """Payload for deterministic 4-part anomaly explanation."""
    equipment_id: str = Field("ESP_WELL_01")
    telemetry_readings: Dict[str, float] = Field(...)
    baseline_medians: Optional[Dict[str, float]] = None
    baseline_iqrs: Optional[Dict[str, float]] = None
    anomaly_score: float = Field(0.85, ge=0.0, le=1.0)
    is_anomaly: bool = True
    ensemble_votes: Optional[Dict[str, bool]] = None


class EquipmentHealthApiRequest(BaseModel):
    """Payload for multi-factor equipment health index evaluation."""
    equipment_id: str = Field(...)
    equipment_type: str = Field("ESP")
    current_telemetry: Dict[str, float] = Field(default_factory=dict)
    historical_anomalies_30d: int = Field(0)
    anomaly_severity_counts: Optional[Dict[str, int]] = None
    cumulative_run_hours: float = Field(5000.0)
    days_since_last_pm: int = Field(45)
    historical_ehi_trajectory: Optional[List[float]] = None


class FleetHealthApiRequest(BaseModel):
    """Payload for fleet-wide asset health rollup."""
    fleet_records: List[EquipmentHealthApiRequest]


class TroubleshootStartApiRequest(BaseModel):
    """Payload to initiate a stateful troubleshooting session."""
    equipment_id: str = Field(...)
    equipment_type: str = Field("ESP")


class TroubleshootStepApiRequest(BaseModel):
    """Payload to advance an interactive troubleshooting gate."""
    session_id: str = Field(...)
    option_key: str = Field(...)
    operator_notes: Optional[str] = None
    safety_confirmed: bool = True


class TroubleshootSessionApiResponse(BaseModel):
    """Response payload representing the current troubleshooting state."""
    session_id: str
    equipment_id: str
    equipment_type: str
    is_completed: bool
    current_active_node: Optional[DecisionNode] = None
    resolution: Optional[TroubleshootingResolution] = None
    steps_executed_count: int
    rag_context_block: str
