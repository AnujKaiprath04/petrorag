"""
PetroRAG Operational Intelligence Service Interfaces (Module 3.1)
Defines abstract base classes and Pydantic schemas for Part 3 operational services:
Production Analytics, Anomaly Detection, Forecasting, Equipment Health,
Incident Intelligence, and Risk Assessment.
"""

from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Schemas: Production Analytics
# ---------------------------------------------------------------------------

class ProductionMetricsRequest(BaseModel):
    well_id: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    window_days: int = Field(default=30, ge=1, le=365)


class ProductionMetricsResponse(BaseModel):
    well_id: str
    total_oil_bbl: float
    total_gas_mscf: float
    total_water_bbl: float
    average_oil_rate_bopd: float
    average_gas_rate_mscfd: float
    average_water_cut_pct: float
    oil_decline_rate_pct: float
    water_cut_trend: Literal["INCREASING", "STABLE", "DECREASING"]
    pressure_trend_psi_per_day: float
    observation_summary: str


# ---------------------------------------------------------------------------
# Schemas: Anomaly Detection
# ---------------------------------------------------------------------------

class AnomalyDetectionRequest(BaseModel):
    entity_id: str
    entity_type: Literal["EQUIPMENT", "WELL"] = "EQUIPMENT"
    data: List[Dict[str, Any]]
    detection_method: Literal["ISOLATION_FOREST", "ROLLING_ZSCORE", "IQR"] = "ISOLATION_FOREST"
    sensitivity: float = Field(default=0.05, ge=0.01, le=0.5)


class AnomalyItem(BaseModel):
    timestamp: datetime
    entity_id: str
    is_anomaly: bool
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    severity: Literal["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    affected_parameter: str
    observed_value: float
    baseline_value: float
    deviation_pct: float
    contributing_signals: Dict[str, float] = Field(default_factory=dict)
    explanation: str


class AnomalyDetectionResponse(BaseModel):
    entity_id: str
    entity_type: str
    detection_method: str
    total_points_analyzed: int
    anomalies_detected: int
    anomaly_rate_pct: float
    highest_severity: Literal["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    records: List[AnomalyItem]


# ---------------------------------------------------------------------------
# Schemas: Time-Series Forecasting
# ---------------------------------------------------------------------------

class ForecastingRequest(BaseModel):
    entity_id: str
    target_metric: str = "oil_rate_bopd"
    historical_data: List[Dict[str, Any]]
    horizon_days: int = Field(default=30, ge=1, le=90)
    model_name: Literal["NAIVE", "ARIMA", "XGBOOST"] = "ARIMA"


class ForecastingResponse(BaseModel):
    entity_id: str
    target_metric: str
    model_name: str
    horizon_days: int
    forecast_dates: List[date]
    predicted_values: List[float]
    lower_bounds: Optional[List[float]] = None
    upper_bounds: Optional[List[float]] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    summary_statement: str


# ---------------------------------------------------------------------------
# Schemas: Equipment Health
# ---------------------------------------------------------------------------

class EquipmentHealthRequest(BaseModel):
    equipment_id: str
    current_sensor_telemetry: Dict[str, float]
    historical_anomalies_count_30d: int = 0
    days_since_last_maintenance: int = 0


class EquipmentHealthResponse(BaseModel):
    equipment_id: str
    tag_name: str
    equipment_type: str
    health_index: float = Field(..., ge=0.0, le=100.0, description="100=optimal, 0=failed")
    status: Literal["HEALTHY", "WATCH", "DEGRADED", "CRITICAL"]
    primary_stress_factors: List[str]
    recent_anomalies: int
    maintenance_status: str
    recommended_inspection_interval_days: int
    evidence_statement: str


# ---------------------------------------------------------------------------
# Schemas: Incident Intelligence
# ---------------------------------------------------------------------------

class IncidentSearchRequest(BaseModel):
    query_description: str
    equipment_type: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=10)


class IncidentMatch(BaseModel):
    incident_id: str
    date: date
    equipment_id: Optional[str] = None
    similarity_score: float
    incident_type: str
    severity: str
    description: str
    root_cause: Optional[str] = None
    action_taken: Optional[str] = None
    document_ref: Optional[str] = None


class IncidentSearchResponse(BaseModel):
    query: str
    total_matches_found: int
    matches: List[IncidentMatch]


# ---------------------------------------------------------------------------
# Schemas: Risk Assessment
# ---------------------------------------------------------------------------

class RiskAssessmentRequest(BaseModel):
    entity_id: str
    active_anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0)
    consequence_severity: int = Field(default=3, ge=1, le=5, description="1=minor to 5=catastrophic")
    condition_duration_hours: float = Field(default=1.0, ge=0.0)
    system_redundancy: bool = True


class RiskAssessmentResponse(BaseModel):
    entity_id: str
    probability: int = Field(..., ge=1, le=5)
    severity: int = Field(..., ge=1, le=5)
    exposure: int = Field(..., ge=1, le=5)
    risk_score: int = Field(..., ge=1, le=125, description="P * S * E")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    contributing_factors: List[str]
    suggested_mitigations: List[str]
    reproducible_calculation_formula: str


# ---------------------------------------------------------------------------
# Abstract Base Classes
# ---------------------------------------------------------------------------

class BaseProductionAnalyticsService(ABC):
    """Interface for field and well-level production analytics."""

    @abstractmethod
    def calculate_production_metrics(self, request: ProductionMetricsRequest) -> ProductionMetricsResponse:
        """Compute cumulative production, decline rates, and water cut trends."""
        pass


class BaseAnomalyDetectionService(ABC):
    """Interface for multivariate and baseline time-series anomaly detection."""

    @abstractmethod
    def detect_anomalies(self, request: AnomalyDetectionRequest) -> AnomalyDetectionResponse:
        """Execute anomaly detection pipeline with severity and signal attribution."""
        pass


class BaseForecastingService(ABC):
    """Interface for time-series production and pressure forecasting."""

    @abstractmethod
    def generate_forecast(self, request: ForecastingRequest) -> ForecastingResponse:
        """Train model on chronological slice and predict future values with metrics."""
        pass


class BaseEquipmentHealthService(ABC):
    """Interface for composite equipment health scoring."""

    @abstractmethod
    def evaluate_health(self, request: EquipmentHealthRequest) -> EquipmentHealthResponse:
        """Combine active telemetry, anomalies, and maintenance history into health index."""
        pass


class BaseIncidentIntelligenceService(ABC):
    """Interface for historical incident semantic retrieval and failure analysis."""

    @abstractmethod
    def find_similar_incidents(self, request: IncidentSearchRequest) -> IncidentSearchResponse:
        """Search incident catalog for semantic similarity to current symptoms."""
        pass


class BaseRiskAssessmentService(ABC):
    """Interface for quantitative industrial risk assessment."""

    @abstractmethod
    def assess_risk(self, request: RiskAssessmentRequest) -> RiskAssessmentResponse:
        """Calculate reproducible P * S * E risk score and safety triage."""
        pass
