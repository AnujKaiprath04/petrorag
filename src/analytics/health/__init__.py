"""
PetroRAG Equipment Health Package (Module 3.14)
Provides multi-factor health index computation, Weibull degradation modeling,
Remaining Useful Life (RUL) estimation, and fleet-wide health rollups.
"""

from src.analytics.health.engine import (
    EquipmentProfile,
    DEFAULT_PROFILES,
    SubIndexBreakdown,
    RULEstimate,
    ComprehensiveEquipmentHealthReport,
    FleetHealthSummary,
    EquipmentHealthEngine,
)

__all__ = [
    "EquipmentProfile",
    "DEFAULT_PROFILES",
    "SubIndexBreakdown",
    "RULEstimate",
    "ComprehensiveEquipmentHealthReport",
    "FleetHealthSummary",
    "EquipmentHealthEngine",
]
