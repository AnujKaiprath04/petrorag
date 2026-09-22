"""
PetroRAG Time-Series Preprocessing Pipeline (Module 3.5)
Coordinates time alignment, uniform grid resampling, short-gap interpolation,
blackout preservation, and multi-entity feature engineering.
"""

from typing import List, Optional, Dict, Any
import pandas as pd

from src.core.logging import logger
from src.analytics.preprocessing.resampler import TimeSeriesResampler
from src.analytics.preprocessing.feature_engineer import TimeSeriesFeatureEngineer


class TimeSeriesPreprocessingPipeline:
    """
    End-to-end preprocessing coordinator transforming raw irregular sensor/production
    readings into ready-to-train ML feature matrices.
    """

    def __init__(
        self,
        time_column: str = "timestamp",
        entity_column: Optional[str] = None,
        resample_freq: Optional[str] = "1h",
        max_gap_steps: int = 3,
        rolling_windows: List[int] = [3, 7, 24],
        lags: List[int] = [1, 2, 7],
    ):
        self.time_column = time_column
        self.entity_column = entity_column
        self.resample_freq = resample_freq
        self.max_gap_steps = max_gap_steps
        self.rolling_windows = rolling_windows
        self.lags = lags

    def process(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Execute chronological alignment -> resampling -> gap handling -> feature engineering."""
        if df.empty:
            return df

        logger.info(f"Preprocessing time-series with {len(df)} rows across entity: '{self.entity_column}'")

        # 1. Resample and handle short vs long sensor gaps
        if self.resample_freq:
            df_resampled = TimeSeriesResampler.resample_dataset(
                df=df,
                freq=self.resample_freq,
                time_column=self.time_column,
                entity_column=self.entity_column,
                max_interpolation_gap_steps=self.max_gap_steps,
            )
        else:
            df_resampled = TimeSeriesResampler.prepare_time_index(
                df=df,
                time_column=self.time_column,
                entity_column=self.entity_column,
            )

        # 2. Select numeric feature columns if not explicitly provided
        if not feature_columns:
            ignore_cols = [self.time_column, "is_interpolated", "is_sensor_blackout"]
            if self.entity_column:
                ignore_cols.append(self.entity_column)
            numeric_cols = df_resampled.select_dtypes(include=["number"]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c not in ignore_cols]
        else:
            feature_cols = [c for c in feature_columns if c in df_resampled.columns]

        # 3. Engineer causal features
        df_featured = TimeSeriesFeatureEngineer.engineer_features(
            df=df_resampled,
            feature_columns=feature_cols,
            entity_column=self.entity_column,
            rolling_windows=self.rolling_windows,
            lags=self.lags,
            periods=[1],
            ema_spans=[7],
        )

        logger.info(f"Feature engineering complete. Result shape: {df_featured.shape}")
        return df_featured
