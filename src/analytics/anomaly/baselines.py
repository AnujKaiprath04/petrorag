"""
PetroRAG Anomaly Detection - Baseline Detectors (Module 3.9)
Implements statistical baseline anomaly detection algorithms for comparative
empirical benchmarking against Isolation Forest:
1. Global & Rolling Z-Score Detector (Gaussian parametric baseline)
2. Interquartile Range (IQR) / Tukey's Fences Detector (Non-parametric quartile baseline)
3. Median Absolute Deviation (MAD) / Hampel Filter Detector (Robust scale baseline)
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Literal, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.core.logging import logger
from src.analytics.anomaly.isolation_forest import (
    FeatureContribution,
    DetectedAnomalyPoint,
    AnomalyDetectionResult,
)


class BaseStatisticalAnomalyDetector:
    """Base class for univariate and multivariate statistical anomaly detectors."""

    def __init__(self, threshold: float, window_size: Optional[int] = None, min_periods: int = 5):
        self.threshold = threshold
        self.window_size = window_size
        self.min_periods = min_periods
        self.feature_names: List[str] = []
        self.baseline_medians: Dict[str, float] = {}
        self.baseline_iqrs: Dict[str, float] = {}
        self.is_fitted: bool = False

    def _prepare_data(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        if isinstance(data, pd.DataFrame):
            if feature_names is not None:
                selected_cols = [c for c in feature_names if c in data.columns]
            else:
                selected_cols = [
                    c
                    for c in data.columns
                    if pd.api.types.is_numeric_dtype(data[c])
                    and c not in ["id", "index"]
                ]
            X = data[selected_cols].to_numpy(dtype=float)
            f_names = selected_cols
        else:
            X = np.array(data, dtype=float)
            if feature_names is not None and len(feature_names) == X.shape[1]:
                f_names = feature_names
            else:
                f_names = [f"feature_{i}" for i in range(X.shape[1])]

        return X, f_names

    def explain_point(self, row_values: np.ndarray) -> List[FeatureContribution]:
        """Calculates feature-level attribution weights based on deviation from baseline."""
        deviations = []
        for i, name in enumerate(self.feature_names):
            val = float(row_values[i])
            med = self.baseline_medians.get(name, 0.0)
            iqr = self.baseline_iqrs.get(name, 1.0)
            norm_dev = abs(val - med) / max(1e-4, iqr)
            deviations.append((name, val, med, iqr, norm_dev))

        total_dev = sum(d[4] for d in deviations)
        contributions = []
        for name, val, med, iqr, norm_dev in deviations:
            weight = float(norm_dev / max(1e-4, total_dev))
            contributions.append(
                FeatureContribution(
                    feature_name=name,
                    observed_value=round(val, 4),
                    baseline_median=round(med, 4),
                    baseline_iqr=round(iqr, 4),
                    normalized_deviation=round(norm_dev, 4),
                    contribution_weight=round(weight, 4),
                )
            )

        contributions.sort(key=lambda x: x.contribution_weight, reverse=True)
        return contributions

    def detect(
        self,
        df: pd.DataFrame,
        time_col: str = "timestamp",
        entity_id: Optional[str] = None,
    ) -> AnomalyDetectionResult:
        """End-to-end detection producing structured AnomalyDetectionResult."""
        if not self.is_fitted:
            self.fit(df)  # type: ignore

        X, _ = self._prepare_data(df, self.feature_names)
        scores = self.score_samples(df)  # type: ignore
        is_anomaly_flags = self.predict(df)  # type: ignore

        timestamps = None
        if time_col in df.columns:
            timestamps = pd.to_datetime(df[time_col]).tolist()

        detected_points: List[DetectedAnomalyPoint] = []
        severities: List[str] = []

        for idx in range(len(X)):
            score = float(scores[idx])
            is_anom = bool(is_anomaly_flags[idx])

            # Severity triage
            if score >= 0.85:
                sev = "CRITICAL"
            elif score >= 0.70:
                sev = "HIGH"
            elif score >= 0.55:
                sev = "MEDIUM"
            elif score >= 0.45 or is_anom:
                sev = "LOW"
            else:
                sev = "NORMAL"

            severities.append(sev)

            contribs = self.explain_point(X[idx])
            top = contribs[0] if contribs else None

            if top:
                top_name = top.feature_name
                obs_val = top.observed_value
                base_val = top.baseline_median
                dev_pct = (
                    float(((obs_val - base_val) / max(1e-4, abs(base_val))) * 100.0)
                    if base_val != 0.0
                    else 0.0
                )
            else:
                top_name = "unknown"
                obs_val, base_val, dev_pct = 0.0, 0.0, 0.0

            ts = timestamps[idx] if timestamps is not None and idx < len(timestamps) else None

            detected_points.append(
                DetectedAnomalyPoint(
                    timestamp=ts,
                    index=idx,
                    is_anomaly=is_anom,
                    anomaly_score=round(score, 4),
                    severity=sev,  # type: ignore
                    top_affected_feature=top_name,
                    feature_observed_value=round(obs_val, 2),
                    feature_baseline_value=round(base_val, 2),
                    feature_deviation_pct=round(dev_pct, 2),
                    contributions=contribs,
                )
            )

        anomaly_count = sum(1 for p in detected_points if p.is_anomaly)
        total_samples = len(detected_points)
        rate_pct = float((anomaly_count / max(1, total_samples)) * 100.0)

        sev_order = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        highest_sev = "NORMAL"
        for s in severities:
            if sev_order.index(s) > sev_order.index(highest_sev):
                highest_sev = s

        return AnomalyDetectionResult(
            total_samples=total_samples,
            anomaly_count=anomaly_count,
            anomaly_rate_pct=round(rate_pct, 2),
            feature_names=self.feature_names,
            highest_severity=highest_sev,  # type: ignore
            anomalies=detected_points,
        )


# ===========================================================================
# 1. Rolling & Global Z-Score Detector
# ===========================================================================

class RollingZScoreAnomalyDetector(BaseStatisticalAnomalyDetector):
    """
    Parametric Gaussian anomaly detector based on standardized Z-scores.
    Supports both static global baseline statistics and rolling dynamic windows.
    """

    def __init__(
        self,
        threshold: float = 3.0,
        window_size: Optional[int] = None,
        min_periods: int = 5,
    ):
        super().__init__(threshold=threshold, window_size=window_size, min_periods=min_periods)
        self.baseline_means: Dict[str, float] = {}
        self.baseline_stds: Dict[str, float] = {}

    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> "RollingZScoreAnomalyDetector":
        X, f_names = self._prepare_data(data, feature_names)
        self.feature_names = f_names

        for i, name in enumerate(self.feature_names):
            vals = X[:, i]
            self.baseline_means[name] = float(np.mean(vals))
            self.baseline_stds[name] = float(max(1e-5, np.std(vals)))
            self.baseline_medians[name] = float(np.median(vals))
            q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
            self.baseline_iqrs[name] = float(max(1e-4, q75 - q25))

        self.is_fitted = True
        return self

    def score_samples(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("RollingZScoreAnomalyDetector must be fitted before scoring.")

        X, _ = self._prepare_data(data, self.feature_names)
        n_samples = len(X)

        if self.window_size and self.window_size > 0 and isinstance(data, pd.DataFrame):
            # Rolling window mode
            df_sub = data[self.feature_names]
            rolling_mean = df_sub.rolling(self.window_size, min_periods=self.min_periods).mean().bfill().to_numpy()
            rolling_std = df_sub.rolling(self.window_size, min_periods=self.min_periods).std().bfill().fillna(1.0).to_numpy()
            rolling_std = np.maximum(1e-5, rolling_std)
            z_matrix = np.abs((X - rolling_mean) / rolling_std)
        else:
            # Global baseline mode
            means = np.array([self.baseline_means[name] for name in self.feature_names])
            stds = np.array([self.baseline_stds[name] for name in self.feature_names])
            z_matrix = np.abs((X - means) / stds)

        # Maximum Z-score across features
        max_z = np.max(z_matrix, axis=1) if z_matrix.shape[1] > 0 else np.zeros(n_samples)

        # Normalized score: threshold maps to 0.50, 2*threshold maps to 1.0
        scores = max_z / (2.0 * max(1e-4, self.threshold))
        return np.clip(scores, 0.0, 1.0)

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        scores = self.score_samples(data)
        return scores >= 0.50


# ===========================================================================
# 2. Interquartile Range (IQR) / Tukey's Fences Detector
# ===========================================================================

class IQRAnomalyDetector(BaseStatisticalAnomalyDetector):
    """
    Non-parametric anomaly detector using Tukey's fences:
    [Q1 - k*IQR, Q3 + k*IQR]. Resilient to skewed and non-Gaussian sensor distributions.
    """

    def __init__(
        self,
        k: float = 1.5,
        window_size: Optional[int] = None,
        min_periods: int = 5,
    ):
        super().__init__(threshold=k, window_size=window_size, min_periods=min_periods)
        self.k = k
        self.baseline_q25: Dict[str, float] = {}
        self.baseline_q75: Dict[str, float] = {}

    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> "IQRAnomalyDetector":
        X, f_names = self._prepare_data(data, feature_names)
        self.feature_names = f_names

        for i, name in enumerate(self.feature_names):
            vals = X[:, i]
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            iqr = float(max(1e-4, q75 - q25))
            med = float(np.median(vals))

            self.baseline_q25[name] = q25
            self.baseline_q75[name] = q75
            self.baseline_iqrs[name] = iqr
            self.baseline_medians[name] = med

        self.is_fitted = True
        return self

    def score_samples(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("IQRAnomalyDetector must be fitted before scoring.")

        X, _ = self._prepare_data(data, self.feature_names)
        n_samples = len(X)

        if self.window_size and self.window_size > 0 and isinstance(data, pd.DataFrame):
            df_sub = data[self.feature_names]
            r_q25 = df_sub.rolling(self.window_size, min_periods=self.min_periods).quantile(0.25).bfill().to_numpy()
            r_q75 = df_sub.rolling(self.window_size, min_periods=self.min_periods).quantile(0.75).bfill().to_numpy()
            r_iqr = np.maximum(1e-4, r_q75 - r_q25)
            lower_fence = r_q25 - self.k * r_iqr
            upper_fence = r_q75 + self.k * r_iqr
            excess_d = np.maximum(0.0, lower_fence - X) + np.maximum(0.0, X - upper_fence)
            norm_excess = excess_d / r_iqr
        else:
            q25_vec = np.array([self.baseline_q25[name] for name in self.feature_names])
            q75_vec = np.array([self.baseline_q75[name] for name in self.feature_names])
            iqr_vec = np.array([self.baseline_iqrs[name] for name in self.feature_names])
            lower_fence = q25_vec - self.k * iqr_vec
            upper_fence = q75_vec + self.k * iqr_vec
            excess_d = np.maximum(0.0, lower_fence - X) + np.maximum(0.0, X - upper_fence)
            norm_excess = excess_d / iqr_vec

        max_norm_excess = np.max(norm_excess, axis=1) if norm_excess.shape[1] > 0 else np.zeros(n_samples)

        # When within fence (excess=0), score is in [0.0, 0.45]
        # When beyond fence (excess>0), score is in [0.50, 1.0]
        scores = np.where(
            max_norm_excess > 0.0,
            0.50 + 0.50 * (1.0 - np.exp(-max_norm_excess / 2.0)),
            0.25,
        )
        return np.clip(scores, 0.0, 1.0)

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        scores = self.score_samples(data)
        return scores >= 0.50


# ===========================================================================
# 3. Median Absolute Deviation (MAD) / Hampel Filter Detector
# ===========================================================================

class MADAnomalyDetector(BaseStatisticalAnomalyDetector):
    """
    Robust scale anomaly detector using Median Absolute Deviation (MAD) and
    Hampel identifiers. Extremely resistant to heavy contamination in historical baselines.
    """

    def __init__(
        self,
        threshold: float = 3.5,
        window_size: Optional[int] = None,
        min_periods: int = 5,
    ):
        super().__init__(threshold=threshold, window_size=window_size, min_periods=min_periods)
        self.baseline_mads: Dict[str, float] = {}

    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> "MADAnomalyDetector":
        X, f_names = self._prepare_data(data, feature_names)
        self.feature_names = f_names

        for i, name in enumerate(self.feature_names):
            vals = X[:, i]
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            consistent_scale = float(max(1e-5, 1.4826 * mad))

            self.baseline_medians[name] = med
            self.baseline_mads[name] = consistent_scale
            q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
            self.baseline_iqrs[name] = float(max(1e-4, q75 - q25))

        self.is_fitted = True
        return self

    def score_samples(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("MADAnomalyDetector must be fitted before scoring.")

        X, _ = self._prepare_data(data, self.feature_names)
        n_samples = len(X)

        if self.window_size and self.window_size > 0 and isinstance(data, pd.DataFrame):
            df_sub = data[self.feature_names]
            r_med = df_sub.rolling(self.window_size, min_periods=self.min_periods).median().bfill().to_numpy()
            diff = np.abs(X - r_med)
            r_mad = (
                pd.DataFrame(diff, columns=self.feature_names)
                .rolling(self.window_size, min_periods=self.min_periods)
                .median()
                .bfill()
                .to_numpy()
            )
            r_scale = np.maximum(1e-5, 1.4826 * r_mad)
            mod_z = np.abs((X - r_med) / r_scale)
        else:
            meds = np.array([self.baseline_medians[name] for name in self.feature_names])
            scales = np.array([self.baseline_mads[name] for name in self.feature_names])
            mod_z = np.abs((X - meds) / scales)

        max_mod_z = np.max(mod_z, axis=1) if mod_z.shape[1] > 0 else np.zeros(n_samples)
        scores = max_mod_z / (2.0 * max(1e-4, self.threshold))
        return np.clip(scores, 0.0, 1.0)

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        scores = self.score_samples(data)
        return scores >= 0.50
