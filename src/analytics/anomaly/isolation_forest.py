"""
PetroRAG Anomaly Detection - Multivariate Isolation Forest (Module 3.8)
Implements industrial-grade multivariate Isolation Forest anomaly detection
with RobustScaler preprocessing, calibrated anomaly scoring [0, 1], parameter
attribution decomposition, and model state persistence.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Literal, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
from pydantic import BaseModel, Field

from src.core.logging import logger


class FeatureContribution(BaseModel):
    feature_name: str
    observed_value: float
    baseline_median: float
    baseline_iqr: float
    normalized_deviation: float
    contribution_weight: float = Field(..., ge=0.0, le=1.0)


class DetectedAnomalyPoint(BaseModel):
    timestamp: Optional[datetime] = None
    index: int
    is_anomaly: bool
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    severity: Literal["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    top_affected_feature: str
    feature_observed_value: float
    feature_baseline_value: float
    feature_deviation_pct: float
    contributions: List[FeatureContribution]


class AnomalyDetectionResult(BaseModel):
    total_samples: int
    anomaly_count: int
    anomaly_rate_pct: float
    feature_names: List[str]
    highest_severity: Literal["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    anomalies: List[DetectedAnomalyPoint]

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class IsolationForestAnomalyDetector:
    """
    Multivariate Isolation Forest detector for sensor streams and wellhead telemetry.
    Combines robust scaling, isolated tree partition path length evaluation,
    calibrated anomaly scoring, and parameter attribution.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        contamination: float = 0.05,
        max_samples: Union[str, int, float] = "auto",
        scaler_type: Literal["robust", "standard", "none"] = "robust",
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.scaler_type = scaler_type
        self.random_state = random_state

        self.feature_names: List[str] = []
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[Union[RobustScaler, StandardScaler]] = None
        self.baseline_medians: Dict[str, float] = {}
        self.baseline_iqrs: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> "IsolationForestAnomalyDetector":
        """
        Fits the scaler and Isolation Forest model on nominal historical telemetry.
        """
        X, f_names = self._prepare_data(data, feature_names)
        self.feature_names = f_names

        if len(X) == 0:
            raise ValueError("Cannot fit Isolation Forest on empty dataset.")

        # Compute baseline statistics for parameter attribution
        for i, name in enumerate(self.feature_names):
            vals = X[:, i]
            med = float(np.median(vals))
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            iqr = max(1e-4, q75 - q25)
            self.baseline_medians[name] = med
            self.baseline_iqrs[name] = iqr

        # Scale features
        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            self.scaler = None
            X_scaled = X

        # Fit Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self.is_fitted = True

        logger.info(
            f"Fitted IsolationForestAnomalyDetector on {len(X)} samples with {len(self.feature_names)} features: {self.feature_names}"
        )
        return self

    def score_samples(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Computes calibrated anomaly scores in range [0.0, 1.0].
        Higher values represent stronger anomalousness.
        Combines isolation forest decision boundary with robust deviation margins.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("IsolationForestAnomalyDetector must be fitted before scoring samples.")

        X, _ = self._prepare_data(data, self.feature_names)
        X_scaled = self.scaler.transform(X) if self.scaler is not None else X

        decision_scores = self.model.decision_function(X_scaled)

        # Sigmoid calibration: S(x) = 1 / (1 + exp(12.0 * decision_function))
        slope = 12.0
        iso_scores = 1.0 / (1.0 + np.exp(slope * decision_scores))

        # Robust multi-feature margin check against baseline medians & IQRs
        med_vec = np.array([self.baseline_medians.get(name, 0.0) for name in self.feature_names])
        iqr_vec = np.array([self.baseline_iqrs.get(name, 1.0) for name in self.feature_names])
        robust_devs = np.abs(X - med_vec) / np.maximum(1e-4, iqr_vec)
        max_dev = np.max(robust_devs, axis=1) if len(robust_devs) > 0 else np.zeros(len(X))

        # Points with max_dev > 2.5 (beyond Tukey 2.5*IQR fence) receive scaled score
        robust_scores = np.where(max_dev > 2.5, 1.0 - np.exp(-(max_dev - 2.5) / 2.0), 0.0)

        combined_scores = np.maximum(iso_scores, robust_scores)
        return np.clip(combined_scores, 0.0, 1.0)

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Returns boolean array: True for anomalies, False for normal points.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("IsolationForestAnomalyDetector must be fitted before predicting.")

        X, _ = self._prepare_data(data, self.feature_names)
        X_scaled = self.scaler.transform(X) if self.scaler is not None else X
        raw_preds = self.model.predict(X_scaled) == -1
        scores = self.score_samples(data)
        return raw_preds | (scores >= 0.50)

    def explain_point(self, row_values: np.ndarray) -> List[FeatureContribution]:
        """
        Decomposes anomaly score into individual feature contributions using
        robust median/IQR deviations.
        """
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
        """
        End-to-end anomaly detection on input DataFrame.
        """
        if not self.is_fitted:
            self.fit(df)

        X, _ = self._prepare_data(df, self.feature_names)
        scores = self.score_samples(df)
        is_anomaly_flags = self.predict(df)

        timestamps = None
        if time_col in df.columns:
            timestamps = pd.to_datetime(df[time_col]).tolist()

        detected_points: List[DetectedAnomalyPoint] = []
        severities: List[str] = []

        for idx in range(len(X)):
            score = float(scores[idx])
            is_anom = bool(is_anomaly_flags[idx])

            # Severity triage
            if score >= 0.90:
                sev = "CRITICAL"
            elif score >= 0.75:
                sev = "HIGH"
            elif score >= 0.60:
                sev = "MEDIUM"
            elif score >= 0.50 or is_anom:
                sev = "LOW"
            else:
                sev = "NORMAL"

            severities.append(sev)

            # Feature contributions
            contribs = self.explain_point(X[idx])
            top_feature = contribs[0] if contribs else None

            if top_feature:
                top_name = top_feature.feature_name
                obs_val = top_feature.observed_value
                base_val = top_feature.baseline_median
                dev_pct = (
                    float(((obs_val - base_val) / max(1e-4, abs(base_val))) * 100.0)
                    if base_val != 0.0
                    else 0.0
                )
            else:
                top_name = "unknown"
                obs_val = 0.0
                base_val = 0.0
                dev_pct = 0.0

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

        # Highest severity ranking
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

    def save(self, filepath: str) -> None:
        """Saves fitted detector state to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted IsolationForestAnomalyDetector.")
        state = {
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "max_samples": self.max_samples,
            "scaler_type": self.scaler_type,
            "random_state": self.random_state,
            "feature_names": self.feature_names,
            "baseline_medians": self.baseline_medians,
            "baseline_iqrs": self.baseline_iqrs,
            "scaler": self.scaler,
            "model": self.model,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(state, filepath)
        logger.info(f"Saved IsolationForestAnomalyDetector state to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "IsolationForestAnomalyDetector":
        """Loads fitted detector state from disk."""
        state = joblib.load(filepath)
        detector = cls(
            n_estimators=state["n_estimators"],
            contamination=state["contamination"],
            max_samples=state["max_samples"],
            scaler_type=state["scaler_type"],
            random_state=state["random_state"],
        )
        detector.feature_names = state["feature_names"]
        detector.baseline_medians = state["baseline_medians"]
        detector.baseline_iqrs = state["baseline_iqrs"]
        detector.scaler = state["scaler"]
        detector.model = state["model"]
        detector.is_fitted = state["is_fitted"]
        logger.info(f"Loaded IsolationForestAnomalyDetector state from {filepath}")
        return detector

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
