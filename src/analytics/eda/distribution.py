"""
PetroRAG EDA Module - Distribution Analyzer (Module 3.6)
Evaluates normality, skewness/kurtosis, outlier rates (IQR & Z-score),
and histogram binning across operational telemetry and production variables.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
from pydantic import BaseModel, Field


class HistogramData(BaseModel):
    counts: List[int]
    bin_edges: List[float]
    bin_centers: List[float]


class DistributionProfile(BaseModel):
    column_name: str
    sample_size: int
    is_normal: bool
    normality_test_name: str
    normality_stat: float
    normality_pvalue: float
    outlier_count_iqr: int
    outlier_pct_iqr: float
    iqr_lower_bound: float
    iqr_upper_bound: float
    outlier_count_zscore: int
    outlier_pct_zscore: float
    histogram: HistogramData


class DistributionAnalyzer:
    """
    Analyzes probability distributions, normality criteria, and outlier
    proportions for sensor readings and production rates.
    """

    def __init__(self, alpha: float = 0.05, n_bins: int = 10):
        self.alpha = alpha
        self.n_bins = n_bins

    def analyze(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
    ) -> Dict[str, DistributionProfile]:
        """
        Runs distribution analysis across all specified numeric columns.
        """
        if numeric_cols is None:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        results: Dict[str, DistributionProfile] = {}

        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            vals = series.to_numpy(dtype=float)
            n = len(vals)

            # Normality testing
            is_normal, test_name, test_stat, p_val = self._test_normality(vals)

            # Outliers via IQR
            p25 = float(np.percentile(vals, 25))
            p75 = float(np.percentile(vals, 75))
            iqr = p75 - p25
            lower_fence = p25 - 1.5 * iqr
            upper_fence = p75 + 1.5 * iqr
            outliers_iqr = int(np.sum((vals < lower_fence) | (vals > upper_fence)))
            pct_iqr = float((outliers_iqr / max(1, n)) * 100.0)

            # Outliers via Z-score (|z| > 3.0)
            std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            mean_val = float(np.mean(vals))
            if std_val > 1e-12:
                z_scores = np.abs((vals - mean_val) / std_val)
                outliers_z = int(np.sum(z_scores > 3.0))
            else:
                outliers_z = 0
            pct_z = float((outliers_z / max(1, n)) * 100.0)

            # Histogram
            counts, bin_edges = np.histogram(vals, bins=self.n_bins)
            bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2.0 for i in range(len(counts))]

            results[col] = DistributionProfile(
                column_name=col,
                sample_size=n,
                is_normal=is_normal,
                normality_test_name=test_name,
                normality_stat=round(float(test_stat), 4),
                normality_pvalue=round(float(p_val), 4),
                outlier_count_iqr=outliers_iqr,
                outlier_pct_iqr=round(pct_iqr, 2),
                iqr_lower_bound=round(lower_fence, 4),
                iqr_upper_bound=round(upper_fence, 4),
                outlier_count_zscore=outliers_z,
                outlier_pct_zscore=round(pct_z, 2),
                histogram=HistogramData(
                    counts=[int(c) for c in counts],
                    bin_edges=[round(float(e), 4) for e in bin_edges],
                    bin_centers=[round(float(c), 4) for c in bin_centers],
                ),
            )

        return results

    def _test_normality(self, vals: np.ndarray) -> tuple[bool, str, float, float]:
        n = len(vals)
        if n < 8:
            return False, "insufficient_sample_size", 0.0, 0.0

        std = np.std(vals)
        if std < 1e-12:
            # Constant value
            return False, "constant_series", 0.0, 0.0

        try:
            if n >= 20:
                stat_val, p_val = stats.normaltest(vals)
                test_name = "d_agostino_pearson"
            else:
                stat_val, p_val = stats.shapiro(vals)
                test_name = "shapiro_wilk"

            if np.isnan(p_val) or np.isnan(stat_val):
                return False, test_name, 0.0, 0.0

            is_normal = bool(p_val > self.alpha)
            return is_normal, test_name, float(stat_val), float(p_val)
        except Exception:
            return False, "error", 0.0, 0.0
