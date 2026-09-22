"""
PetroRAG Production Analytics Package (Module 3.7)
Decline Curve Analysis (Arps), Well/Field KPIs, Inflow Performance (IPR),
and Chan Water Coning/Breakthrough Diagnostics.
"""

from src.analytics.production.arps import (
    ArpsDeclineCurve,
    ArpsFitResult,
    ArpsForecastPoint,
)
from src.analytics.production.kpi import (
    KPICalculator,
    WellKPI,
    FieldKPI,
    WellRankingItem,
)
from src.analytics.production.ipr import (
    InflowPerformanceCalculator,
    IPRResult,
    IPRCurvePoint,
)
from src.analytics.production.diagnostics import (
    ChanWaterDiagnosticEngine,
    ChanDiagnosticResult,
    WORPoint,
)
from src.analytics.production.analytics import (
    ProductionAnalyzer,
    ComprehensiveProductionReport,
)

__all__ = [
    "ArpsDeclineCurve",
    "ArpsFitResult",
    "ArpsForecastPoint",
    "KPICalculator",
    "WellKPI",
    "FieldKPI",
    "WellRankingItem",
    "InflowPerformanceCalculator",
    "IPRResult",
    "IPRCurvePoint",
    "ChanWaterDiagnosticEngine",
    "ChanDiagnosticResult",
    "WORPoint",
    "ProductionAnalyzer",
    "ComprehensiveProductionReport",
]
