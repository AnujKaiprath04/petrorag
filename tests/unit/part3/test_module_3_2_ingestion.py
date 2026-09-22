"""
PetroRAG Module 3.2 Unit Test Suite — Structured Data Ingestion
Verifies CSV, Excel, JSON, and SQL table parsers, column alias mapping,
dataset type detection, and relational database persistence.
"""

import io
import pytest
from pathlib import Path
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.analytics.db import Base, ProductionDataModel, init_db
from src.analytics.ingestion.schemas import DatasetType
from src.analytics.ingestion.parsers import (
    CSVParser,
    ExcelParser,
    JSONParser,
    SQLParser,
    ColumnNormalizer,
)
from src.analytics.ingestion.loader import StructuredDataIngestionService


@pytest.fixture
def sample_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent / "data" / "sample"


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionMaker = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionMaker()
    try:
        yield session
    finally:
        session.close()


def test_column_normalizer_and_type_detection():
    """Verify alias mapping identifies dataset types and normalizes columns."""
    prod_cols = ["date", "well_name", "oil_bpd", "gas_mscf", "bsw_pct", "tubing_pressure"]
    detected = ColumnNormalizer.detect_dataset_type(prod_cols)
    assert detected == DatasetType.PRODUCTION

    df_raw = pd.DataFrame({
        "date": ["2026-01-01"],
        "well_name": ["WELL-01"],
        "oil_bpd": [1200.0],
        "bsw_pct": [22.5],
        "tubing_pressure": [2400.0],
    })
    df_norm = ColumnNormalizer.map_columns(df_raw, detected)

    assert "timestamp" in df_norm.columns or "date" in df_norm.columns
    assert "well_id" in df_norm.columns
    assert "oil_rate" in df_norm.columns
    assert "water_cut" in df_norm.columns
    assert "pressure" in df_norm.columns


def test_csv_production_ingestion(sample_dir):
    """Verify ingestion of CSV production dataset with alias resolution."""
    csv_file = sample_dir / "production_sample.csv"
    assert csv_file.exists(), f"Missing test file {csv_file}"

    service = StructuredDataIngestionService()
    df, summary = service.ingest_file(csv_file)

    assert summary.source_format == "CSV"
    assert summary.dataset_type == DatasetType.PRODUCTION
    assert summary.total_rows_read == 10
    assert summary.valid_rows == 10
    assert set(summary.unique_entities) == {"WELL-A1", "WELL-B2"}
    assert "oil_rate" in df.columns
    assert "water_cut" in df.columns
    assert "pressure" in df.columns


def test_csv_sensor_ingestion(sample_dir):
    """Verify ingestion of CSV equipment sensor telemetry with ISO timestamps."""
    sensor_file = sample_dir / "sensor_sample.csv"
    assert sensor_file.exists(), f"Missing test file {sensor_file}"

    service = StructuredDataIngestionService()
    df, summary = service.ingest_file(sensor_file)

    assert summary.dataset_type == DatasetType.EQUIPMENT_SENSOR
    assert summary.total_rows_read == 10
    assert "C-101" in summary.unique_entities
    assert "P-201" in summary.unique_entities
    assert "vibration" in df.columns
    assert "pressure" in df.columns
    assert "rpm" in df.columns
    assert summary.date_range_start is not None


def test_json_catalog_ingestion(sample_dir):
    """Verify ingestion of JSON equipment catalog."""
    json_file = sample_dir / "equipment_catalog.json"
    assert json_file.exists(), f"Missing test file {json_file}"

    service = StructuredDataIngestionService()
    df, summary = service.ingest_file(json_file)

    assert summary.source_format == "JSON"
    assert summary.total_rows_read == 3
    assert "equipment_id" in df.columns
    assert "tag_name" in df.columns
    assert "equipment_type" in df.columns


def test_raw_csv_string_ingestion():
    """Verify in-memory stream ingestion without file I/O."""
    raw_csv = """record_date;well;oil_prod;water_cut_pct
2026-02-01;W-99;1450.0;12.5
2026-02-02;W-99;1442.0;12.8
"""
    service = StructuredDataIngestionService()
    df, summary = service.ingest_raw_data(raw_csv, source_format="CSV")

    assert summary.total_rows_read == 2
    assert "well_id" in df.columns
    assert "oil_rate" in df.columns
    assert "water_cut" in df.columns
    assert summary.unique_entities == ["W-99"]


def test_excel_parser_roundtrip():
    """Verify ExcelParser reads sheets correctly from an in-memory buffer."""
    df_orig = pd.DataFrame({
        "equipment_tag": ["C-101", "C-102"],
        "maintenance_type": ["PM", "CM"],
        "date": ["2026-01-10", "2026-01-15"],
        "description": ["Quarterly filter change", "Replaced dry gas seal"],
        "resolution": ["Complete", "Restored to service"],
    })

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_orig.to_excel(writer, sheet_name="MaintenanceLogs", index=False)
    buffer.seek(0)

    df_read = ExcelParser.parse(buffer, sheet_name="MaintenanceLogs")
    assert len(df_read) == 2
    assert "equipment_tag" in df_read.columns

    # Pass to normalizer
    norm = ColumnNormalizer.map_columns(df_read, DatasetType.MAINTENANCE_RECORD)
    assert "equipment_id" in norm.columns
    assert "maintenance_type" in norm.columns


def test_database_persistence_and_sql_parser(sample_dir, in_memory_db):
    """Verify records are written to relational DB and can be queried back via SQLParser."""
    csv_file = sample_dir / "production_sample.csv"
    service = StructuredDataIngestionService(db_session=in_memory_db)

    # Ingest and persist
    df, summary = service.ingest_file(csv_file, persist_to_db=True)
    assert summary.persisted_to_database is True

    # Read back through SQLParser
    df_sql = SQLParser.parse_table(in_memory_db, "production_data")
    assert len(df_sql) == 10
    assert "oil_rate_bopd" in df_sql.columns
    assert "well_id" in df_sql.columns
