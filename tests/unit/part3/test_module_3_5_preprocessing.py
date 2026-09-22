"""
PetroRAG Module 3.5 Unit Test Suite — Time-Series Preprocessing & Feature Engineering
Verifies chronological sorting, uniform resampling, short vs long gap handling,
rolling statistics, lag generation, delta/rate-of-change, and multi-entity isolation.
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytest

from src.analytics.preprocessing.resampler import TimeSeriesResampler
from src.analytics.preprocessing.feature_engineer import TimeSeriesFeatureEngineer
from src.analytics.preprocessing.pipeline import TimeSeriesPreprocessingPipeline


def test_chronological_sorting_and_indexing():
    """Verify time index sorting and duplicate timestamp removal."""
    data = [
        {"timestamp": "2026-01-03T10:00:00", "well_id": "W1", "pressure": 42.0},
        {"timestamp": "2026-01-01T10:00:00", "well_id": "W1", "pressure": 40.0},
        {"timestamp": "2026-01-02T10:00:00", "well_id": "W1", "pressure": 41.0},
        {"timestamp": "2026-01-02T10:00:00", "well_id": "W1", "pressure": 41.5},  # duplicate, should keep last
    ]
    df = pd.DataFrame(data)
    df_prepared = TimeSeriesResampler.prepare_time_index(df, entity_column="well_id")

    assert len(df_prepared) == 3
    assert df_prepared.loc[0, "timestamp"] < df_prepared.loc[1, "timestamp"] < df_prepared.loc[2, "timestamp"]
    assert df_prepared.loc[1, "pressure"] == 41.5  # kept last duplicate


def test_short_gap_vs_long_gap_interpolation():
    """
    CRITICAL INDUSTRIAL RULE:
    Interpolate short transient gaps (<= 3 steps), but do NOT interpolate long sensor blackouts.
    """
    # 10 hourly points with:
    # - a 2-hour gap (hours 3 and 4 missing) -> SHORT GAP: should interpolate
    # - a 5-hour gap (hours 8 to 12 missing) -> LONG GAP: should NOT interpolate
    base = datetime(2026, 1, 1, 0, 0)
    timestamps = [
        base + timedelta(hours=0),
        base + timedelta(hours=1),
        base + timedelta(hours=2),
        # hours 3, 4 missing (2 steps gap)
        base + timedelta(hours=5),
        base + timedelta(hours=6),
        base + timedelta(hours=7),
        # hours 8, 9, 10, 11, 12 missing (5 steps gap)
        base + timedelta(hours=13),
        base + timedelta(hours=14),
    ]
    pressures = [40.0, 41.0, 42.0, 45.0, 46.0, 47.0, 53.0, 54.0]
    df = pd.DataFrame({"timestamp": timestamps, "pressure": pressures})

    df_resampled = TimeSeriesResampler.resample_dataset(
        df,
        freq="1h",
        time_column="timestamp",
        max_interpolation_gap_steps=3,
    )

    # Hourly grid should span 0 to 14 = 15 points
    assert len(df_resampled) == 15

    # Short gap at hour 3 and 4 should be linearly interpolated
    p_h3 = df_resampled.loc[df_resampled["timestamp"] == pd.to_datetime(base + timedelta(hours=3), utc=True), "pressure"].values[0]
    assert not np.isnan(p_h3)
    assert pytest.approx(p_h3, abs=0.1) == 43.0  # between 42.0 and 45.0

    # Short gap should be flagged as is_interpolated
    flag_h3 = df_resampled.loc[df_resampled["timestamp"] == pd.to_datetime(base + timedelta(hours=3), utc=True), "is_interpolated"].values[0]
    assert bool(flag_h3) is True

    # Long gap at hour 9, 10, 11 should NOT be interpolated
    p_h9 = df_resampled.loc[df_resampled["timestamp"] == pd.to_datetime(base + timedelta(hours=9), utc=True), "pressure"].values[0]
    assert np.isnan(p_h9)

    # Long gap should be flagged as blackout
    blackout_h9 = df_resampled.loc[df_resampled["timestamp"] == pd.to_datetime(base + timedelta(hours=9), utc=True), "is_sensor_blackout"].values[0]
    assert bool(blackout_h9) is True


def test_rolling_statistics():
    """Verify rolling mean, std, min, and max match exact formulas."""
    df = pd.DataFrame({"val": [10.0, 20.0, 30.0, 40.0, 50.0]})
    df_feat = TimeSeriesFeatureEngineer.add_rolling_features(df, columns=["val"], windows=[3])

    # Trailing window 3:
    # idx 0: [10] -> mean 10
    # idx 1: [10, 20] -> mean 15
    # idx 2: [10, 20, 30] -> mean 20
    # idx 3: [20, 30, 40] -> mean 30
    # idx 4: [30, 40, 50] -> mean 40
    assert df_feat.loc[0, "val_rolling_mean_3"] == 10.0
    assert df_feat.loc[1, "val_rolling_mean_3"] == 15.0
    assert df_feat.loc[2, "val_rolling_mean_3"] == 20.0
    assert df_feat.loc[3, "val_rolling_mean_3"] == 30.0
    assert df_feat.loc[4, "val_rolling_mean_3"] == 40.0

    assert df_feat.loc[2, "val_rolling_min_3"] == 10.0
    assert df_feat.loc[2, "val_rolling_max_3"] == 30.0


def test_lag_features_and_causality():
    """Verify lag features enforce zero lookahead bias."""
    df = pd.DataFrame({"oil": [100.0, 110.0, 120.0, 130.0]})
    df_lags = TimeSeriesFeatureEngineer.add_lag_features(df, columns=["oil"], lags=[1, 2])

    assert np.isnan(df_lags.loc[0, "oil_lag_1"])
    assert df_lags.loc[1, "oil_lag_1"] == 100.0
    assert df_lags.loc[2, "oil_lag_1"] == 110.0

    assert np.isnan(df_lags.loc[0, "oil_lag_2"])
    assert np.isnan(df_lags.loc[1, "oil_lag_2"])
    assert df_lags.loc[2, "oil_lag_2"] == 100.0


def test_delta_and_percentage_change():
    """Verify delta and percentage change calculations."""
    df = pd.DataFrame({"pressure": [100.0, 120.0, 90.0]})
    df_delta = TimeSeriesFeatureEngineer.add_delta_and_change_features(df, columns=["pressure"], periods=[1])

    # row 1: 120 - 100 = 20, +20%
    assert df_delta.loc[1, "pressure_delta_1"] == 20.0
    assert df_delta.loc[1, "pressure_pct_change_1"] == 20.0

    # row 2: 90 - 120 = -30, -25%
    assert df_delta.loc[2, "pressure_delta_1"] == -30.0
    assert df_delta.loc[2, "pressure_pct_change_1"] == -25.0


def test_multi_entity_isolation():
    """
    CRITICAL: Verify feature engineering respects entity boundaries.
    Lags and rolling statistics of Well B must never be calculated using Well A data.
    """
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=6, freq="1D"),
        "well_id": ["W1", "W1", "W1", "W2", "W2", "W2"],
        "rate": [500.0, 510.0, 520.0, 900.0, 910.0, 920.0],
    })

    df_feat = TimeSeriesFeatureEngineer.engineer_features(
        df,
        feature_columns=["rate"],
        entity_column="well_id",
        rolling_windows=[2],
        lags=[1],
    )

    # First row of W2 (idx 3) must have NaN for lag_1, NOT 520.0 from W1
    assert np.isnan(df_feat.loc[3, "rate_lag_1"])

    # Rolling mean at first row of W2 must be 900.0, NOT average of W1 and W2
    assert df_feat.loc[3, "rate_rolling_mean_2"] == 900.0


def test_pipeline_end_to_end():
    """Verify full preprocessing pipeline executes smoothly."""
    base = datetime(2026, 2, 1, 0, 0)
    data = []
    for i in range(15):
        data.append({
            "timestamp": base + timedelta(hours=i),
            "equipment_id": "C-101",
            "vibration": 2.1 + i * 0.1,
            "pressure": 42.0 + i * 0.2,
        })
    df = pd.DataFrame(data)

    pipeline = TimeSeriesPreprocessingPipeline(
        time_column="timestamp",
        entity_column="equipment_id",
        resample_freq="1h",
        rolling_windows=[3],
        lags=[1],
    )
    df_out = pipeline.process(df)

    assert "vibration_rolling_mean_3" in df_out.columns
    assert "pressure_lag_1" in df_out.columns
    assert "vibration_delta_1" in df_out.columns
    assert len(df_out) == 15
