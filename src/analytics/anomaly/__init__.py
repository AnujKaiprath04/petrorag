"""
PetroRAG Anomaly Detection Package (Module 3.8)
Provides multivariate Isolation Forest detection, robust scaling,
calibrated anomaly scoring, and parameter attribution.
"""

from src.analytics.anomaly.isolation_forest import (
    IsolationForestAnomalyDetector,
    AnomalyDetectionResult,
    DetectedAnomalyPoint,
    FeatureContribution,
)
from src.analytics.anomaly.baselines import (
    BaseStatisticalAnomalyDetector,
    RollingZScoreAnomalyDetector,
    IQRAnomalyDetector,
    MADAnomalyDetector,
)
from src.analytics.anomaly.attribution import (
    SeverityTriageEngine,
    SeverityTriageResult,
    AnomalyAttributionEngine,
    ParameterAttributionItem,
    DecomposedAnomalyReport,
    AttributionSummary,
    SensorCoOccurrence,
)
from src.analytics.anomaly.benchmark import (
    AnomalyGroundTruthGenerator,
    AnomalyBenchmarkRunner,
    BenchmarkDataset,
    ModelBenchmarkMetrics,
    AnomalyBenchmarkReport,
    InjectedAnomalyRecord,
)
from src.analytics.anomaly.explainer import (
    ChannelObservation,
    TelemetryObservationSection,
    StatisticalConfidenceSection,
    SafetyEnvelopeSection,
    OperationalRemediationSection,
    FactualAnomalyExplanation,
    AnomalyExplanationFormatter,
    EnsembleAnomalyExplainerService,
    ENGINEERING_UNITS,
)
from src.analytics.anomaly.rag_enhancer import (
    RAGEnhancedAnomalyReport,
    RAGEnhancedAnomalyService,
)

__all__ = [
    "IsolationForestAnomalyDetector",
    "AnomalyDetectionResult",
    "DetectedAnomalyPoint",
    "FeatureContribution",
    "BaseStatisticalAnomalyDetector",
    "RollingZScoreAnomalyDetector",
    "IQRAnomalyDetector",
    "MADAnomalyDetector",
    "SeverityTriageEngine",
    "SeverityTriageResult",
    "AnomalyAttributionEngine",
    "ParameterAttributionItem",
    "DecomposedAnomalyReport",
    "AttributionSummary",
    "SensorCoOccurrence",
    "AnomalyGroundTruthGenerator",
    "AnomalyBenchmarkRunner",
    "BenchmarkDataset",
    "ModelBenchmarkMetrics",
    "AnomalyBenchmarkReport",
    "InjectedAnomalyRecord",
    "ChannelObservation",
    "TelemetryObservationSection",
    "StatisticalConfidenceSection",
    "SafetyEnvelopeSection",
    "OperationalRemediationSection",
    "FactualAnomalyExplanation",
    "AnomalyExplanationFormatter",
    "EnsembleAnomalyExplainerService",
    "ENGINEERING_UNITS",
    "RAGEnhancedAnomalyReport",
    "RAGEnhancedAnomalyService",
]



