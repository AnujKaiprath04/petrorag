"""
PetroRAG Exploratory Data Analysis (EDA) Module (Module 3.6)
Provides automated statistical profiling, correlation analysis, distribution testing,
production decline analysis, and LLM-ready RAG operational context generation.
"""

from src.analytics.eda.profiler import (
    DataProfiler,
    DatasetProfile,
    NumericColumnProfile,
    CategoricalColumnProfile,
)
from src.analytics.eda.correlation import (
    CorrelationAnalyzer,
    CorrelationReport,
    CorrelationPair,
    TargetCorrelation,
)
from src.analytics.eda.distribution import (
    DistributionAnalyzer,
    DistributionProfile,
    HistogramData,
)
from src.analytics.eda.trends import (
    TrendAnalyzer,
    ProductionTrendReport,
    ProductionDeclineFit,
    PressureDepletionTrend,
    WaterCutEvolution,
    GOREvolution,
)
from src.analytics.eda.pipeline import (
    EDAPipeline,
    EDAReport,
)

__all__ = [
    "DataProfiler",
    "DatasetProfile",
    "NumericColumnProfile",
    "CategoricalColumnProfile",
    "CorrelationAnalyzer",
    "CorrelationReport",
    "CorrelationPair",
    "TargetCorrelation",
    "DistributionAnalyzer",
    "DistributionProfile",
    "HistogramData",
    "TrendAnalyzer",
    "ProductionTrendReport",
    "ProductionDeclineFit",
    "PressureDepletionTrend",
    "WaterCutEvolution",
    "GOREvolution",
    "EDAPipeline",
    "EDAReport",
]
