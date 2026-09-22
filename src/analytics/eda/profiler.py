"""
PetroRAG EDA Module - Data Profiler (Module 3.6)
Computes summary statistics, missingness audits, percentiles, skewness,
kurtosis, and value distribution profiles for structured oil & gas datasets.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
from pydantic import BaseModel, Field


class NumericColumnProfile(BaseModel):
    column_name: str
    data_type: str
    total_count: int
    valid_count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    iqr: float
    skewness: float
    kurtosis: float
    percentiles: Dict[str, float]
    zero_count: int
    zero_pct: float
    negative_count: int


class CategoricalColumnProfile(BaseModel):
    column_name: str
    data_type: str
    total_count: int
    valid_count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    top_categories: Dict[str, int]


class DatasetProfile(BaseModel):
    row_count: int
    column_count: int
    memory_usage_bytes: int
    missing_cell_count: int
    overall_missing_pct: float
    duplicate_rows: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    numeric_profiles: Dict[str, NumericColumnProfile]
    categorical_profiles: Dict[str, CategoricalColumnProfile]


class DataProfiler:
    """
    Profiles pandas DataFrames to extract rigorous statistical measures
    tailored for oil & gas production, sensor, and catalog datasets.
    """

    def __init__(self, exclude_cols: Optional[List[str]] = None):
        self.exclude_cols = set(exclude_cols or [])

    def profile_dataframe(self, df: pd.DataFrame) -> DatasetProfile:
        """
        Generates a comprehensive dataset profile.
        """
        if df.empty:
            return DatasetProfile(
                row_count=0,
                column_count=0,
                memory_usage_bytes=0,
                missing_cell_count=0,
                overall_missing_pct=0.0,
                duplicate_rows=0,
                numeric_columns=[],
                categorical_columns=[],
                datetime_columns=[],
                numeric_profiles={},
                categorical_profiles={},
            )

        row_count = len(df)
        column_count = len(df.columns)
        memory_usage = int(df.memory_usage(deep=True).sum())
        total_cells = row_count * column_count
        missing_cells = int(df.isna().sum().sum())
        overall_missing_pct = float((missing_cells / max(1, total_cells)) * 100.0)
        duplicate_rows = int(df.duplicated().sum())

        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        datetime_cols: List[str] = []

        numeric_profiles: Dict[str, NumericColumnProfile] = {}
        categorical_profiles: Dict[str, CategoricalColumnProfile] = {}

        for col in df.columns:
            if col in self.exclude_cols:
                continue

            series = df[col]

            if pd.api.types.is_datetime64_any_dtype(series):
                datetime_cols.append(col)
            elif pd.api.types.is_numeric_dtype(series):
                numeric_cols.append(col)
                profile = self._profile_numeric_column(series, col, row_count)
                if profile is not None:
                    numeric_profiles[col] = profile
            else:
                categorical_cols.append(col)
                profile = self._profile_categorical_column(series, col, row_count)
                categorical_profiles[col] = profile

        return DatasetProfile(
            row_count=row_count,
            column_count=column_count,
            memory_usage_bytes=memory_usage,
            missing_cell_count=missing_cells,
            overall_missing_pct=round(overall_missing_pct, 2),
            duplicate_rows=duplicate_rows,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            numeric_profiles=numeric_profiles,
            categorical_profiles=categorical_profiles,
        )

    def _profile_numeric_column(
        self, series: pd.Series, col_name: str, total_rows: int
    ) -> Optional[NumericColumnProfile]:
        valid_series = series.dropna()
        valid_count = len(valid_series)
        missing_count = total_rows - valid_count
        missing_pct = float((missing_count / max(1, total_rows)) * 100.0)
        unique_count = int(valid_series.nunique())

        if valid_count == 0:
            return NumericColumnProfile(
                column_name=col_name,
                data_type=str(series.dtype),
                total_count=total_rows,
                valid_count=0,
                missing_count=missing_count,
                missing_pct=round(missing_pct, 2),
                unique_count=0,
                mean=0.0,
                std=0.0,
                median=0.0,
                min=0.0,
                max=0.0,
                iqr=0.0,
                skewness=0.0,
                kurtosis=0.0,
                percentiles={
                    f"p{p}": 0.0 for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
                },
                zero_count=0,
                zero_pct=0.0,
                negative_count=0,
            )

        vals = valid_series.to_numpy(dtype=float)
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if valid_count > 1 else 0.0
        median_val = float(np.median(vals))
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))

        p25 = float(np.percentile(vals, 25))
        p75 = float(np.percentile(vals, 75))
        iqr_val = float(p75 - p25)

        # Skewness and kurtosis
        if valid_count >= 3 and std_val > 1e-12:
            skew_val = float(stats.skew(vals, bias=False))
            kurt_val = float(stats.kurtosis(vals, bias=False))
        else:
            skew_val = 0.0
            kurt_val = 0.0

        percentiles_dict = {
            f"p{p}": round(float(np.percentile(vals, p)), 4)
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }

        zero_count = int(np.sum(vals == 0.0))
        zero_pct = float((zero_count / max(1, valid_count)) * 100.0)
        negative_count = int(np.sum(vals < 0.0))

        return NumericColumnProfile(
            column_name=col_name,
            data_type=str(series.dtype),
            total_count=total_rows,
            valid_count=valid_count,
            missing_count=missing_count,
            missing_pct=round(missing_pct, 2),
            unique_count=unique_count,
            mean=round(mean_val, 4),
            std=round(std_val, 4),
            median=round(median_val, 4),
            min=round(min_val, 4),
            max=round(max_val, 4),
            iqr=round(iqr_val, 4),
            skewness=round(skew_val, 4),
            kurtosis=round(kurt_val, 4),
            percentiles=percentiles_dict,
            zero_count=zero_count,
            zero_pct=round(zero_pct, 2),
            negative_count=negative_count,
        )

    def _profile_categorical_column(
        self, series: pd.Series, col_name: str, total_rows: int
    ) -> CategoricalColumnProfile:
        valid_series = series.dropna()
        valid_count = len(valid_series)
        missing_count = total_rows - valid_count
        missing_pct = float((missing_count / max(1, total_rows)) * 100.0)
        unique_count = int(valid_series.nunique())

        top_counts = valid_series.value_counts().head(5).to_dict()
        top_categories = {str(k): int(v) for k, v in top_counts.items()}

        return CategoricalColumnProfile(
            column_name=col_name,
            data_type=str(series.dtype),
            total_count=total_rows,
            valid_count=valid_count,
            missing_count=missing_count,
            missing_pct=round(missing_pct, 2),
            unique_count=unique_count,
            top_categories=top_categories,
        )
