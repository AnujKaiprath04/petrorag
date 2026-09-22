"""
PetroRAG Structured File & Data Parsers (Module 3.2)
Parses CSV, Excel (XLSX), JSON, and SQL sources, detects dataset types,
and maps heterogeneous field names to standardized canonical schemas.
"""

import json
import io
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.core.logging import logger
from src.analytics.ingestion.schemas import DatasetType, COLUMN_ALIASES


class ColumnNormalizer:
    """Normalizes arbitrary input column names to canonical schema keys."""

    @staticmethod
    def detect_dataset_type(columns: List[str]) -> DatasetType:
        """Heuristically detect dataset type by counting alias matches."""
        col_lower = {str(c).lower().strip().replace(" ", "_"): c for c in columns}
        scores: Dict[DatasetType, int] = {t: 0 for t in DatasetType if t != DatasetType.UNKNOWN}

        for dtype, mapping in COLUMN_ALIASES.items():
            for canonical, aliases in mapping.items():
                if any(alias in col_lower for alias in aliases):
                    scores[dtype] += 1

        best_type, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score >= 3:
            return best_type
        return DatasetType.UNKNOWN

    @staticmethod
    def map_columns(df: pd.DataFrame, dataset_type: DatasetType) -> pd.DataFrame:
        """Rename DataFrame columns to canonical names according to dataset type."""
        if dataset_type not in COLUMN_ALIASES:
            return df

        alias_map = COLUMN_ALIASES[dataset_type]
        rename_dict: Dict[str, str] = {}
        col_clean = {c: str(c).lower().strip().replace(" ", "_") for c in df.columns}

        for orig_col, clean_name in col_clean.items():
            for canonical, aliases in alias_map.items():
                if clean_name in aliases:
                    rename_dict[orig_col] = canonical
                    break

        df_renamed = df.rename(columns=rename_dict)
        return df_renamed


class CSVParser:
    """Parses delimited text files with automatic separator detection."""

    @staticmethod
    def parse(source: Union[str, Path, io.BytesIO, io.StringIO]) -> pd.DataFrame:
        try:
            if isinstance(source, (str, Path)) and Path(str(source)).is_file():
                df = pd.read_csv(source, sep=None, engine="python")
            elif isinstance(source, str):
                df = pd.read_csv(io.StringIO(source), sep=None, engine="python")
            else:
                df = pd.read_csv(source, sep=None, engine="python")
            return df
        except Exception as e:
            logger.error(f"Error parsing CSV data: {e}")
            raise ValueError(f"Failed to parse CSV: {e}") from e


class ExcelParser:
    """Parses Excel spreadsheets with sheet selection."""

    @staticmethod
    def parse(source: Union[str, Path, io.BytesIO], sheet_name: Optional[Union[str, int]] = 0) -> pd.DataFrame:
        try:
            df = pd.read_excel(source, sheet_name=sheet_name)
            return df
        except Exception as e:
            logger.error(f"Error parsing Excel spreadsheet: {e}")
            raise ValueError(f"Failed to parse Excel: {e}") from e


class JSONParser:
    """Parses JSON records into a tabular DataFrame."""

    @staticmethod
    def parse(source: Union[str, Path, io.BytesIO, io.StringIO, Dict, List]) -> pd.DataFrame:
        try:
            if isinstance(source, (dict, list)):
                raw_data = source
            elif isinstance(source, (str, Path)) and Path(str(source)).is_file():
                with open(source, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            elif isinstance(source, str):
                raw_data = json.loads(source)
            else:
                raw_data = json.load(source)

            if isinstance(raw_data, list):
                df = pd.DataFrame(raw_data)
            elif isinstance(raw_data, dict):
                # Check if records are nested under 'records', 'data', or 'items'
                for key in ["records", "data", "items"]:
                    if key in raw_data and isinstance(raw_data[key], list):
                        return pd.DataFrame(raw_data[key])
                # Direct dictionary (orient='index' or scalar)
                df = pd.DataFrame([raw_data])
            else:
                raise ValueError("JSON must be an object or an array of objects.")
            return df
        except Exception as e:
            logger.error(f"Error parsing JSON data: {e}")
            raise ValueError(f"Failed to parse JSON: {e}") from e


class SQLParser:
    """Extracts operational records directly from database tables."""

    @staticmethod
    def parse_table(session: Session, table_name: str, limit: int = 10000) -> pd.DataFrame:
        try:
            # Query table safely
            query = text(f"SELECT * FROM {table_name} LIMIT :limit")
            result = session.execute(query, {"limit": limit})
            rows = result.fetchall()
            cols = list(result.keys())
            return pd.DataFrame(rows, columns=cols)
        except Exception as e:
            logger.error(f"Error reading SQL table '{table_name}': {e}")
            raise ValueError(f"Failed to read SQL table: {e}") from e
