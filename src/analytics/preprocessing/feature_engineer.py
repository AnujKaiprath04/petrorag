"""
PetroRAG Time-Series Feature Engineering Engine (Module 3.5)
Generates rolling statistics, autoregressive lags, rate-of-change, deltas,
and exponential moving averages with mathematical zero-lookahead bias guarantee.
"""

from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd
from src.core.logging import logger


class TimeSeriesFeatureEngineer:
    """
    Computes time-series features for industrial ML and forecasting.
    Enforces strict temporal causality: features at time t depend strictly on observations <= t.
    """

    @classmethod
    def _apply_grouped_or_single(
        cls,
        df: pd.DataFrame,
        entity_column: Optional[str],
        calc_fn,
    ) -> pd.DataFrame:
        if entity_column and entity_column in df.columns:
            pieces = [calc_fn(group) for _, group in df.groupby(entity_column, as_index=False, sort=False)]
            return pd.concat(pieces, ignore_index=True) if pieces else df
        return calc_fn(df)

    @classmethod
    def add_rolling_features(
        cls,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [3, 7, 14],
        entity_column: Optional[str] = None,
        min_periods: int = 1,
    ) -> pd.DataFrame:
        """
        Compute rolling mean, std, min, and max per window.
        Prevents lookahead bias by computing over historical trailing window.
        """
        df_out = df.copy()

        def _calc(sub_df: pd.DataFrame) -> pd.DataFrame:
            res = sub_df.copy()
            for col in columns:
                if col not in sub_df.columns:
                    continue
                series = pd.to_numeric(sub_df[col], errors="coerce")
                for w in windows:
                    roll = series.rolling(window=w, min_periods=min_periods)
                    res[f"{col}_rolling_mean_{w}"] = roll.mean()
                    res[f"{col}_rolling_std_{w}"] = roll.std().fillna(0.0)
                    res[f"{col}_rolling_min_{w}"] = roll.min()
                    res[f"{col}_rolling_max_{w}"] = roll.max()
            return res

        return cls._apply_grouped_or_single(df_out, entity_column, _calc)

    @classmethod
    def add_lag_features(
        cls,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 2, 7],
        entity_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Add lagged values x(t - k) for autoregressive forecasting."""
        df_out = df.copy()

        def _calc(sub_df: pd.DataFrame) -> pd.DataFrame:
            res = sub_df.copy()
            for col in columns:
                if col not in sub_df.columns:
                    continue
                series = pd.to_numeric(sub_df[col], errors="coerce")
                for lag in lags:
                    res[f"{col}_lag_{lag}"] = series.shift(lag)
            return res

        return cls._apply_grouped_or_single(df_out, entity_column, _calc)

    @classmethod
    def add_delta_and_change_features(
        cls,
        df: pd.DataFrame,
        columns: List[str],
        periods: List[int] = [1, 7],
        entity_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute differences (delta) and percentage changes over k periods.
        delta = x(t) - x(t - k)
        pct_change = (x(t) - x(t - k)) / max(eps, |x(t - k)|) * 100
        """
        df_out = df.copy()

        def _calc(sub_df: pd.DataFrame) -> pd.DataFrame:
            res = sub_df.copy()
            for col in columns:
                if col not in sub_df.columns:
                    continue
                series = pd.to_numeric(sub_df[col], errors="coerce")
                for p in periods:
                    prev = series.shift(p)
                    diff = series - prev
                    denom = np.maximum(1e-5, np.abs(prev))
                    res[f"{col}_delta_{p}"] = diff
                    res[f"{col}_pct_change_{p}"] = (diff / denom) * 100.0
                    res[f"{col}_rate_of_change_{p}"] = diff / float(p)
            return res

        return cls._apply_grouped_or_single(df_out, entity_column, _calc)

    @classmethod
    def add_exponential_smoothing(
        cls,
        df: pd.DataFrame,
        columns: List[str],
        spans: List[int] = [7, 30],
        entity_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compute exponential moving averages (EMA) for trend smoothing."""
        df_out = df.copy()

        def _calc(sub_df: pd.DataFrame) -> pd.DataFrame:
            res = sub_df.copy()
            for col in columns:
                if col not in sub_df.columns:
                    continue
                series = pd.to_numeric(sub_df[col], errors="coerce")
                for s in spans:
                    res[f"{col}_ema_{s}"] = series.ewm(span=s, adjust=False).mean()
            return res

        return cls._apply_grouped_or_single(df_out, entity_column, _calc)

    @classmethod
    def engineer_features(
        cls,
        df: pd.DataFrame,
        feature_columns: List[str],
        entity_column: Optional[str] = None,
        rolling_windows: List[int] = [3, 7],
        lags: List[int] = [1, 2, 7],
        periods: List[int] = [1],
        ema_spans: List[int] = [7],
    ) -> pd.DataFrame:
        """Apply full feature engineering pipeline in a single step."""
        res = cls.add_rolling_features(df, columns=feature_columns, windows=rolling_windows, entity_column=entity_column)
        res = cls.add_lag_features(res, columns=feature_columns, lags=lags, entity_column=entity_column)
        res = cls.add_delta_and_change_features(res, columns=feature_columns, periods=periods, entity_column=entity_column)
        res = cls.add_exponential_smoothing(res, columns=feature_columns, spans=ema_spans, entity_column=entity_column)
        return res
