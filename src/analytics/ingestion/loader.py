"""
PetroRAG Structured Ingestion Loader & Orchestrator (Module 3.2)
Orchestrates ingestion of multi-format operational datasets, validates fields,
persists records to the relational database, and returns execution telemetry.
"""

from __future__ import annotations
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
from sqlalchemy.orm import Session

from src.core.logging import logger
from src.analytics.db import (
    SessionLocal,
    WellModel,
    EquipmentModel,
    ProductionDataModel,
    SensorDataModel,
    MaintenanceModel,
    IncidentModel,
)
from src.analytics.ingestion.schemas import DatasetType, IngestionSummary
from src.analytics.ingestion.parsers import (
    CSVParser,
    ExcelParser,
    JSONParser,
    SQLParser,
    ColumnNormalizer,
)

Tuple_Result = Tuple[pd.DataFrame, IngestionSummary]


class StructuredDataIngestionService:
    """
    Unified ingestion service for multi-format Oil & Gas structured datasets.
    Supports CSV, Excel, JSON, and SQL table queries with automatic column normalization.
    """

    def __init__(self, db_session: Optional[Session] = None):
        self._external_session = db_session

    def ingest_file(
        self,
        file_path: Union[str, Path],
        dataset_type: Optional[DatasetType] = None,
        persist_to_db: bool = False,
    ) -> Tuple_Result:
        """Parse structured file, normalize schema, and optionally persist to relational DB."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix in [".csv", ".txt"]:
            df = CSVParser.parse(path)
            fmt = "CSV"
        elif suffix in [".xlsx", ".xls"]:
            df = ExcelParser.parse(path)
            fmt = "EXCEL"
        elif suffix == ".json":
            df = JSONParser.parse(path)
            fmt = "JSON"
        else:
            raise ValueError(f"Unsupported file format '{suffix}'. Supported: CSV, XLSX, JSON.")

        return self.process_dataframe(
            df=df,
            source_name=path.name,
            source_format=fmt,
            dataset_type=dataset_type,
            persist_to_db=persist_to_db,
        )

    def ingest_raw_data(
        self,
        raw_content: Union[str, Dict, List],
        source_format: str = "CSV",
        dataset_type: Optional[DatasetType] = None,
        persist_to_db: bool = False,
    ) -> Tuple_Result:
        """Parse in-memory raw string or JSON structure."""
        fmt = source_format.upper()
        if fmt == "CSV":
            df = CSVParser.parse(raw_content)
        elif fmt == "JSON":
            df = JSONParser.parse(raw_content)
        else:
            raise ValueError(f"Direct raw parsing only supports CSV and JSON. Provided: {fmt}")

        return self.process_dataframe(
            df=df,
            source_name="in_memory_stream",
            source_format=fmt,
            dataset_type=dataset_type,
            persist_to_db=persist_to_db,
        )

    def process_dataframe(
        self,
        df: pd.DataFrame,
        source_name: str,
        source_format: str,
        dataset_type: Optional[DatasetType] = None,
        persist_to_db: bool = False,
    ) -> Tuple_Result:
        """Normalize, parse timestamps, validate rows, and record summary."""
        start_time = time.perf_counter()
        errors: List[str] = []

        if df.empty:
            return pd.DataFrame(), IngestionSummary(
                source_name=source_name,
                source_format=source_format,
                dataset_type=DatasetType.UNKNOWN,
                total_rows_read=0,
                valid_rows=0,
                rejected_rows=0,
                errors=["Input DataFrame is empty."],
            )

        # Detect dataset type if not specified
        if not dataset_type or dataset_type == DatasetType.UNKNOWN:
            detected_type = ColumnNormalizer.detect_dataset_type(list(df.columns))
            logger.info(f"Auto-detected dataset type: {detected_type}")
        else:
            detected_type = dataset_type

        # Normalize column names
        df_norm = ColumnNormalizer.map_columns(df, detected_type)
        if df_norm.columns.duplicated().any():
            df_norm = df_norm.loc[:, ~df_norm.columns.duplicated()]

        # Standardize timestamp column if present
        date_start = None
        date_end = None
        if "timestamp" in df_norm.columns:
            try:
                df_norm["timestamp"] = pd.to_datetime(df_norm["timestamp"], errors="coerce")
                valid_dates = df_norm["timestamp"].dropna()
                if not valid_dates.empty:
                    date_start = valid_dates.min().to_pydatetime()
                    date_end = valid_dates.max().to_pydatetime()
            except Exception as e:
                errors.append(f"Timestamp parsing error: {e}")

        elif "date" in df_norm.columns:
            try:
                df_norm["date"] = pd.to_datetime(df_norm["date"], errors="coerce").dt.date
                valid_dates = df_norm["date"].dropna()
                if not valid_dates.empty:
                    date_start = datetime.combine(valid_dates.min(), datetime.min.time())
                    date_end = datetime.combine(valid_dates.max(), datetime.min.time())
            except Exception as e:
                errors.append(f"Date parsing error: {e}")

        # Collect unique entities (well_id, equipment_id, or tag_name)
        entities: List[str] = []
        for id_col in ["well_id", "equipment_id", "tag_name"]:
            if id_col in df_norm.columns:
                col_data = df_norm[id_col]
                if isinstance(col_data, pd.DataFrame):
                    col_data = col_data.iloc[:, 0]
                extracted = [str(x) for x in col_data.dropna().unique().tolist()]
                if extracted:
                    entities = extracted
                    break

        total_rows = len(df_norm)
        valid_rows = total_rows

        # Persist to database if requested
        persisted = False
        if persist_to_db and detected_type != DatasetType.UNKNOWN:
            try:
                self._persist_records(df_norm, detected_type)
                persisted = True
            except Exception as e:
                errors.append(f"Database persistence failure: {e}")
                logger.error(f"Error persisting {detected_type} records: {e}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        summary = IngestionSummary(
            source_name=source_name,
            source_format=source_format,
            dataset_type=detected_type,
            total_rows_read=total_rows,
            valid_rows=valid_rows,
            rejected_rows=0,
            date_range_start=date_start,
            date_range_end=date_end,
            unique_entities=entities,
            persisted_to_database=persisted,
            processing_time_ms=round(elapsed_ms, 2),
            errors=errors,
        )

        return df_norm, summary

    def _persist_records(self, df: pd.DataFrame, dataset_type: DatasetType) -> None:
        """Helper to write standardized records into relational DB tables."""
        session = self._external_session or SessionLocal()
        should_close = self._external_session is None

        try:
            records = df.to_dict(orient="records")

            if dataset_type == DatasetType.PRODUCTION:
                for row in records:
                    ts = row.get("timestamp")
                    if pd.isna(ts):
                        continue
                    rec = ProductionDataModel(
                        timestamp=ts if isinstance(ts, datetime) else pd.to_datetime(ts).to_pydatetime(),
                        well_id=str(row.get("well_id", "UNKNOWN")),
                        oil_rate_bopd=float(row.get("oil_rate", 0.0)) if pd.notna(row.get("oil_rate")) else None,
                        gas_rate_mscfd=float(row.get("gas_rate", 0.0)) if pd.notna(row.get("gas_rate")) else None,
                        water_rate_bwpd=float(row.get("water_rate", 0.0)) if pd.notna(row.get("water_rate")) else None,
                        water_cut_pct=float(row.get("water_cut", 0.0)) if pd.notna(row.get("water_cut")) else None,
                        tubing_head_pressure_psi=float(row.get("pressure", 0.0)) if pd.notna(row.get("pressure")) else None,
                    )
                    session.merge(rec)

            elif dataset_type == DatasetType.EQUIPMENT_SENSOR:
                for row in records:
                    ts = row.get("timestamp")
                    if pd.isna(ts):
                        continue
                    rec = SensorDataModel(
                        timestamp=ts if isinstance(ts, datetime) else pd.to_datetime(ts).to_pydatetime(),
                        equipment_id=str(row.get("equipment_id", "UNKNOWN")),
                        discharge_pressure_bar=float(row.get("pressure", 0.0)) if pd.notna(row.get("pressure")) else None,
                        temperature_c=float(row.get("temperature", 0.0)) if pd.notna(row.get("temperature")) else None,
                        flow_rate_m3h=float(row.get("flow_rate", 0.0)) if pd.notna(row.get("flow_rate")) else None,
                        vibration_rms_mms=float(row.get("vibration", 0.0)) if pd.notna(row.get("vibration")) else None,
                        rpm=float(row.get("rpm", 0.0)) if pd.notna(row.get("rpm")) else None,
                        power_kw=float(row.get("power", 0.0)) if pd.notna(row.get("power")) else None,
                    )
                    session.merge(rec)

            session.commit()
            logger.info(f"Successfully persisted {len(records)} records for {dataset_type}.")
        except Exception as e:
            session.rollback()
            raise
        finally:
            if should_close:
                session.close()
