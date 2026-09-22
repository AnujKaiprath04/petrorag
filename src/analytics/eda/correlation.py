"""
PetroRAG EDA Module - Correlation Analyzer (Module 3.6)
Computes Pearson and Spearman correlation matrices, detects multicollinearity,
and identifies key feature-target relationships in oil & gas operational datasets.
"""

from typing import Dict, List, Any, Optional, Literal
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class CorrelationPair(BaseModel):
    feature_a: str
    feature_b: str
    pearson: float
    spearman: float
    abs_pearson: float
    relationship: Literal[
        "STRONG_POSITIVE",
        "MODERATE_POSITIVE",
        "WEAK_POSITIVE",
        "UNCORRELATED",
        "WEAK_NEGATIVE",
        "MODERATE_NEGATIVE",
        "STRONG_NEGATIVE",
    ]


class TargetCorrelation(BaseModel):
    target_feature: str
    feature: str
    pearson: float
    spearman: float
    abs_pearson: float
    relationship: str


class CorrelationReport(BaseModel):
    numeric_features: List[str]
    pearson_matrix: Dict[str, Dict[str, float]]
    spearman_matrix: Dict[str, Dict[str, float]]
    ranked_pairs: List[CorrelationPair]
    collinear_pairs: List[CorrelationPair]
    target_correlations: Optional[List[TargetCorrelation]] = None


class CorrelationAnalyzer:
    """
    Computes pairwise linear (Pearson) and non-linear monotonic (Spearman)
    correlations, identifying multi-collinear sensor channels and production drivers.
    """

    def __init__(self, collinearity_threshold: float = 0.85):
        self.collinearity_threshold = collinearity_threshold

    def analyze(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        min_periods: int = 5,
    ) -> CorrelationReport:
        """
        Calculates correlation matrices, ranked pairs, collinearity flags,
        and optional target correlations.
        """
        if numeric_cols is None:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        # Filter out columns with zero variance or all NaNs
        valid_cols = []
        for c in numeric_cols:
            s = df[c].dropna()
            if len(s) >= min_periods and s.std() > 1e-12:
                valid_cols.append(c)

        if len(valid_cols) < 2:
            return CorrelationReport(
                numeric_features=valid_cols,
                pearson_matrix={},
                spearman_matrix={},
                ranked_pairs=[],
                collinear_pairs=[],
                target_correlations=[],
            )

        sub_df = df[valid_cols]

        # Compute matrices
        pearson_df = sub_df.corr(method="pearson", min_periods=min_periods).fillna(0.0)
        spearman_df = sub_df.corr(method="spearman", min_periods=min_periods).fillna(0.0)

        pearson_matrix = {
            col: {k: round(float(v), 4) for k, v in row.items()}
            for col, row in pearson_df.to_dict(orient="index").items()
        }
        spearman_matrix = {
            col: {k: round(float(v), 4) for k, v in row.items()}
            for col, row in spearman_df.to_dict(orient="index").items()
        }

        # Ranked pairs
        pairs: List[CorrelationPair] = []
        for i in range(len(valid_cols)):
            for j in range(i + 1, len(valid_cols)):
                fa = valid_cols[i]
                fb = valid_cols[j]
                r_p = float(pearson_df.loc[fa, fb])
                r_s = float(spearman_df.loc[fa, fb])
                abs_p = abs(r_p)

                rel = self._classify_relationship(r_p)

                pairs.append(
                    CorrelationPair(
                        feature_a=fa,
                        feature_b=fb,
                        pearson=round(r_p, 4),
                        spearman=round(r_s, 4),
                        abs_pearson=round(abs_p, 4),
                        relationship=rel,
                    )
                )

        pairs.sort(key=lambda x: x.abs_pearson, reverse=True)

        # Collinear pairs
        collinear_pairs = [
            p for p in pairs if p.abs_pearson >= self.collinearity_threshold
        ]

        # Target correlations if requested
        target_corrs: Optional[List[TargetCorrelation]] = None
        if target_col and target_col in valid_cols:
            target_corrs = []
            for col in valid_cols:
                if col == target_col:
                    continue
                r_p = float(pearson_df.loc[target_col, col])
                r_s = float(spearman_df.loc[target_col, col])
                abs_p = abs(r_p)
                target_corrs.append(
                    TargetCorrelation(
                        target_feature=target_col,
                        feature=col,
                        pearson=round(r_p, 4),
                        spearman=round(r_s, 4),
                        abs_pearson=round(abs_p, 4),
                        relationship=self._classify_relationship(r_p),
                    )
                )
            target_corrs.sort(key=lambda x: x.abs_pearson, reverse=True)

        return CorrelationReport(
            numeric_features=valid_cols,
            pearson_matrix=pearson_matrix,
            spearman_matrix=spearman_matrix,
            ranked_pairs=pairs,
            collinear_pairs=collinear_pairs,
            target_correlations=target_corrs,
        )

    @staticmethod
    def _classify_relationship(
        r: float,
    ) -> Literal[
        "STRONG_POSITIVE",
        "MODERATE_POSITIVE",
        "WEAK_POSITIVE",
        "UNCORRELATED",
        "WEAK_NEGATIVE",
        "MODERATE_NEGATIVE",
        "STRONG_NEGATIVE",
    ]:
        if r >= 0.7:
            return "STRONG_POSITIVE"
        if r >= 0.3:
            return "MODERATE_POSITIVE"
        if r >= 0.1:
            return "WEAK_POSITIVE"
        if r > -0.1:
            return "UNCORRELATED"
        if r > -0.3:
            return "WEAK_NEGATIVE"
        if r > -0.7:
            return "MODERATE_NEGATIVE"
        return "STRONG_NEGATIVE"
