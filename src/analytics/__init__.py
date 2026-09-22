"""
PetroRAG Operational Analytics Package (Part 3)
Provides structured data ingestion, quality validation, unit normalization,
time-series processing, anomaly detection, forecasting, equipment health,
incident intelligence, risk assessment, and explainable AI services.
"""

from src.analytics.interfaces import (
    BaseProductionAnalyticsService,
    BaseAnomalyDetectionService,
    BaseForecastingService,
    BaseEquipmentHealthService,
    BaseIncidentIntelligenceService,
    BaseRiskAssessmentService,
)
from src.analytics.services import (
    ProductionAnalyticsService,
    AnomalyDetectionService,
    ForecastingService,
    EquipmentHealthService,
    IncidentIntelligenceService,
    RiskAssessmentService,
)
from src.analytics.db import (
    init_db,
    get_db_session,
    Base,
    WellModel,
    EquipmentModel,
    ProductionDataModel,
    SensorDataModel,
    MaintenanceModel,
    IncidentModel,
    AnomalyModel,
    ForecastModel,
    RiskModel,
)

from src.analytics.eda import EDAPipeline, EDAReport
from src.analytics.production import (
    ProductionAnalyzer,
    ComprehensiveProductionReport,
    ArpsDeclineCurve,
    ArpsFitResult,
    FieldKPI,
    WellKPI,
    IPRResult,
    ChanDiagnosticResult,
)
from src.analytics.anomaly import (
    IsolationForestAnomalyDetector,
    AnomalyDetectionResult,
    DetectedAnomalyPoint,
    FeatureContribution,
)

from src.analytics.health import (
    EquipmentProfile,
    DEFAULT_PROFILES,
    SubIndexBreakdown,
    RULEstimate,
    ComprehensiveEquipmentHealthReport,
    FleetHealthSummary,
    EquipmentHealthEngine,
)
from src.analytics.troubleshooting import (
    DecisionOption,
    DecisionNode,
    SessionStepRecord,
    TroubleshootingResolution,
    TroubleshootingSession,
    TroubleshootingWorkflowService,
    build_esp_troubleshooting_tree,
    build_compressor_troubleshooting_tree,
)

__all__ = [
    "BaseProductionAnalyticsService",
    "BaseAnomalyDetectionService",
    "BaseForecastingService",
    "BaseEquipmentHealthService",
    "BaseIncidentIntelligenceService",
    "BaseRiskAssessmentService",
    "ProductionAnalyticsService",
    "AnomalyDetectionService",
    "ForecastingService",
    "EquipmentHealthService",
    "IncidentIntelligenceService",
    "RiskAssessmentService",
    "init_db",
    "get_db_session",
    "Base",
    "WellModel",
    "EquipmentModel",
    "ProductionDataModel",
    "SensorDataModel",
    "MaintenanceModel",
    "IncidentModel",
    "AnomalyModel",
    "ForecastModel",
    "RiskModel",
    "EDAPipeline",
    "EDAReport",
    "ProductionAnalyzer",
    "ComprehensiveProductionReport",
    "ArpsDeclineCurve",
    "ArpsFitResult",
    "FieldKPI",
    "WellKPI",
    "IPRResult",
    "ChanDiagnosticResult",
    "IsolationForestAnomalyDetector",
    "RollingZScoreAnomalyDetector",
    "IQRAnomalyDetector",
    "MADAnomalyDetector",
    "AnomalyDetectionResult",
    "DetectedAnomalyPoint",
    "FeatureContribution",
    "EquipmentProfile",
    "DEFAULT_PROFILES",
    "SubIndexBreakdown",
    "RULEstimate",
    "ComprehensiveEquipmentHealthReport",
    "FleetHealthSummary",
    "EquipmentHealthEngine",
    "DecisionOption",
    "DecisionNode",
    "SessionStepRecord",
    "TroubleshootingResolution",
    "TroubleshootingSession",
    "TroubleshootingWorkflowService",
    "build_esp_troubleshooting_tree",
    "build_compressor_troubleshooting_tree",
]



