"""
PetroRAG Data Quality Validation Package (Module 3.3)
Provides physical boundary rules, distinction between Data Errors and
Operational Anomalies, and comprehensive data quality reporting.
"""

from src.analytics.validation.rules import (
    IssueCategory,
    ParameterBoundary,
    PHYSICAL_BOUNDARIES,
)
from src.analytics.validation.validator import (
    DataQualityValidator,
    DataQualityReport,
    DataQualityIssue,
    TimeGapInfo,
)

__all__ = [
    "IssueCategory",
    "ParameterBoundary",
    "PHYSICAL_BOUNDARIES",
    "DataQualityValidator",
    "DataQualityReport",
    "DataQualityIssue",
    "TimeGapInfo",
]
