"""
PetroRAG Structured Ingestion Package (Module 3.2)
Provides parsing, column normalization, and database loading for
CSV, XLSX, JSON, and SQL operational datasets.
"""

from src.analytics.ingestion.schemas import DatasetType, IngestionSummary, COLUMN_ALIASES
from src.analytics.ingestion.parsers import (
    CSVParser,
    ExcelParser,
    JSONParser,
    SQLParser,
    ColumnNormalizer,
)
from src.analytics.ingestion.loader import StructuredDataIngestionService

__all__ = [
    "DatasetType",
    "IngestionSummary",
    "COLUMN_ALIASES",
    "CSVParser",
    "ExcelParser",
    "JSONParser",
    "SQLParser",
    "ColumnNormalizer",
    "StructuredDataIngestionService",
]
