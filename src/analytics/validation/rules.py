"""
PetroRAG Physical Boundary Rules & Issue Categorization (Module 3.3)
Defines physical limits for Oil & Gas engineering parameters and
categorizes deviations into Data Errors vs Operational Anomalies.
"""

from enum import Enum
from typing import Dict, Any, Optional, NamedTuple
from pydantic import BaseModel


class IssueCategory(str, Enum):
    DATA_ERROR = "DATA_ERROR"                # Physically impossible measurement (e.g. P < 0, WC > 100%)
    OPERATIONAL_ANOMALY = "OPERATIONAL_ANOMALY"  # Physically possible but exceeds operating envelope/alarm
    TIMESTAMP_GAP = "TIMESTAMP_GAP"          # Missing data interval in time-series
    DUPLICATE_RECORD = "DUPLICATE_RECORD"    # Duplicate timestamp for same entity
    MISSING_VALUE = "MISSING_VALUE"          # Null / NaN field


class ParameterBoundary(NamedTuple):
    min_physical: float        # Values outside [min_physical, max_physical] are DATA_ERRORS
    max_physical: float
    min_nominal: float         # Values outside [min_nominal, max_nominal] but within physical are OPERATIONAL_ANOMALIES
    max_nominal: float
    unit_symbol: str


# Physical and nominal boundaries for Oil & Gas engineering parameters
PHYSICAL_BOUNDARIES: Dict[str, ParameterBoundary] = {
    "pressure": ParameterBoundary(
        min_physical=0.0,
        max_physical=25000.0,  # 25,000 psi (~1,720 bar) max HP/HT wellhead rating
        min_nominal=1.0,
        max_nominal=10000.0,   # Nominal range up to 10,000 psi (or 700 bar)
        unit_symbol="psi/bar"
    ),
    "temperature": ParameterBoundary(
        min_physical=-50.0,    # Arctic / cryogenic refrigeration limit
        max_physical=550.0,    # Exhaust / combustor max
        min_nominal=10.0,
        max_nominal=150.0,
        unit_symbol="°C"
    ),
    "oil_rate": ParameterBoundary(
        min_physical=0.0,
        max_physical=100000.0, # 100k bbl/d max super-giant well
        min_nominal=5.0,
        max_nominal=15000.0,
        unit_symbol="bopd"
    ),
    "gas_rate": ParameterBoundary(
        min_physical=0.0,
        max_physical=250000.0, # 250 MMSCFD
        min_nominal=0.0,
        max_nominal=50000.0,
        unit_symbol="mscfd"
    ),
    "water_rate": ParameterBoundary(
        min_physical=0.0,
        max_physical=100000.0,
        min_nominal=0.0,
        max_nominal=30000.0,
        unit_symbol="bwpd"
    ),
    "water_cut": ParameterBoundary(
        min_physical=0.0,
        max_physical=100.0,    # Physical percentage limit
        min_nominal=0.0,
        max_nominal=98.0,
        unit_symbol="%"
    ),
    "vibration": ParameterBoundary(
        min_physical=0.0,
        max_physical=50.0,     # Max sensor dynamic range (mm/s RMS)
        min_nominal=0.1,
        max_nominal=7.1,       # ISO 10816-3 Zone D shutdown limit
        unit_symbol="mm/s"
    ),
    "rpm": ParameterBoundary(
        min_physical=0.0,
        max_physical=35000.0,  # Micro-turbine / high speed compressor
        min_nominal=500.0,
        max_nominal=18000.0,
        unit_symbol="RPM"
    ),
    "flow_rate": ParameterBoundary(
        min_physical=0.0,
        max_physical=10000.0,
        min_nominal=0.0,
        max_nominal=2500.0,
        unit_symbol="m3/h"
    ),
    "power": ParameterBoundary(
        min_physical=0.0,
        max_physical=50000.0,  # 50 MW
        min_nominal=1.0,
        max_nominal=15000.0,
        unit_symbol="kW"
    ),
}
