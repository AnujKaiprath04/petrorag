"""
PetroRAG Time-Series Resampling & Gap Handling (Module 3.5)
Chronologically sorts, resamples to regular sampling frequencies, and performs
safe short-gap interpolation while preserving and marking long sensor blackouts.
"""

from typing import Optional, List, Union
import numpy as np
import pandas as pd
from src.core.logging import logger


class TimeSeriesResampler:
    """
    Standardizes unevenly sampled SCADA and production time-series into uniform grids.
    Enforces the industrial safety rule: never interpolate across large sensor dropouts.
    """

    @staticmethod
    def prepare_time_index(
        df: pd.DataFrame,
        time_column: str = "timestamp",
        entity_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Parse datetime, sort chronologically, and drop duplicate timestamps."""
        df_out = df.copy()
        if time_column not in df_out.columns:
            raise KeyError(f"Time column '{time_column}' not found in DataFrame.")

        df_out[time_column] = pd.to_datetime(df_out[time_column], errors="coerce", utc=True)
        df_out = df_out.dropna(subset=[time_column])

        sort_cols = [entity_column, time_column] if entity_column and entity_column in df_out.columns else [time_column]
        df_out = df_out.sort_values(by=sort_cols).reset_index(drop=True)

        # Drop exact duplicates if any
        subset_cols = [entity_column, time_column] if entity_column and entity_column in df_out.columns else [time_column]
        df_out = df_out.drop_duplicates(subset=subset_cols, keep="last").reset_index(drop=True)
        return df_out

    @classmethod
    def resample_entity_series(
        cls,
        df: pd.DataFrame,
        freq: str = "1h",
        time_column: str = "timestamp",
        numeric_agg: str = "mean",
        max_interpolation_gap_steps: int = 3,
    ) -> pd.DataFrame:
        """
        Resample single-entity time-series to regular frequency.
        Interpolates short gaps (<= max_interpolation_gap_steps) and flags them.
        Leaves long gaps as NaN and marks them in 'is_sensor_blackout'.
        """
        if df.empty:
            return df

        df_indexed = df.set_index(time_column)
        numeric_cols = df_indexed.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in df_indexed.columns if c not in numeric_cols]

        # 1. Resample numeric columns
        if numeric_agg == "median":
            resampled_num = df_indexed[numeric_cols].resample(freq).median()
        elif numeric_agg == "last":
            resampled_num = df_indexed[numeric_cols].resample(freq).last()
        else:
            resampled_num = df_indexed[numeric_cols].resample(freq).mean()

        # 2. Resample non-numeric (e.g. well_id, status) via forward-fill
        resampled_non_num = df_indexed[non_numeric_cols].resample(freq).ffill()
        resampled = pd.concat([resampled_non_num, resampled_num], axis=1)

        # 3. Track gap sizes and interpolate ONLY short gaps
        # Record which values were originally present
        is_interpolated = pd.DataFrame(False, index=resampled.index, columns=numeric_cols)
        is_sensor_blackout = pd.Series(False, index=resampled.index)

        for col in numeric_cols:
            series = resampled[col]
            is_na = series.isna()

            long_gap_mask = pd.Series(False, index=series.index)
            short_gap_mask = pd.Series(False, index=series.index)

            if is_na.any():
                blocks = (is_na != is_na.shift()).cumsum()[is_na]
                gap_sizes = is_na.groupby(blocks).transform("size")

                long_indices = gap_sizes[gap_sizes > max_interpolation_gap_steps].index
                long_gap_mask.loc[long_indices] = True

                short_indices = gap_sizes[gap_sizes <= max_interpolation_gap_steps].index
                short_gap_mask.loc[short_indices] = True

            # Linear interpolation bounded by inside area
            interpolated_series = series.interpolate(method="time", limit_area="inside")

            # CRITICAL: Enforce that values inside long sensor blackouts remain NaN!
            interpolated_series.loc[long_gap_mask] = np.nan

            # Flag newly populated short-gap values
            is_interpolated[col] = short_gap_mask & interpolated_series.notna()
            resampled[col] = interpolated_series

            is_sensor_blackout = is_sensor_blackout | long_gap_mask

        resampled["is_interpolated"] = is_interpolated.any(axis=1)
        resampled["is_sensor_blackout"] = is_sensor_blackout

        return resampled.reset_index()

    @classmethod
    def resample_dataset(
        cls,
        df: pd.DataFrame,
        freq: str = "1h",
        time_column: str = "timestamp",
        entity_column: Optional[str] = None,
        max_interpolation_gap_steps: int = 3,
    ) -> pd.DataFrame:
        """Multi-entity resampler ensuring well/equipment boundaries are preserved."""
        df_clean = cls.prepare_time_index(df, time_column=time_column, entity_column=entity_column)

        if entity_column and entity_column in df_clean.columns:
            results = []
            for _, group in df_clean.groupby(entity_column, as_index=False):
                res = cls.resample_entity_series(
                    group,
                    freq=freq,
                    time_column=time_column,
                    max_interpolation_gap_steps=max_interpolation_gap_steps
                )
                results.append(res)
            return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        else:
            return cls.resample_entity_series(
                df_clean,
                freq=freq,
                time_column=time_column,
                max_interpolation_gap_steps=max_interpolation_gap_steps
            )
