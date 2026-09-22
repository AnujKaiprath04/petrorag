"""
PetroRAG EDA Module - Unified EDA Pipeline (Module 3.6)
Coordinates dataset profiling, multi-correlation analysis, distribution testing,
and petroleum trend characterization into an integrated analytical pipeline.
"""

import json
from typing import Dict, List, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field

from src.analytics.eda.profiler import DataProfiler, DatasetProfile
from src.analytics.eda.correlation import CorrelationAnalyzer, CorrelationReport
from src.analytics.eda.distribution import DistributionAnalyzer, DistributionProfile
from src.analytics.eda.trends import TrendAnalyzer, ProductionTrendReport


class EDAReport(BaseModel):
    dataset_name: Optional[str] = None
    dataset_type: Optional[str] = None
    profile: DatasetProfile
    correlations: CorrelationReport
    distributions: Dict[str, DistributionProfile]
    trends: Optional[ProductionTrendReport] = None
    key_findings: List[str]

    def to_json(self, indent: int = 2) -> str:
        """Serializes report to JSON string."""
        return self.model_dump_json(indent=indent)

    def generate_rag_context(self) -> str:
        """
        Generates an LLM-ready markdown summary of exploratory data analysis
        findings for contextual grounding in PetroRAG queries.
        """
        lines = [
            "### Exploratory Data Analysis & Statistical Telemetry Summary",
            f"- **Records Analysed**: {self.profile.row_count} rows across {self.profile.column_count} features.",
            f"- **Missing Data**: {self.profile.overall_missing_pct}% overall missingness.",
        ]

        # Key Findings
        if self.key_findings:
            lines.append("\n**Key Empirical Observations**:")
            for finding in self.key_findings:
                lines.append(f"- {finding}")

        # Highly Collinear Variables
        if self.correlations.collinear_pairs:
            lines.append("\n**High Multicollinearity (|r| >= 0.85)**:")
            for pair in self.correlations.collinear_pairs[:5]:
                lines.append(
                    f"- `{pair.feature_a}` & `{pair.feature_b}`: Pearson r = {pair.pearson:.2f}, Spearman = {pair.spearman:.2f}"
                )

        # Trends
        if self.trends and self.trends.summary_observations:
            lines.append("\n**Production & Reservoir Depletion Trends**:")
            for obs in self.trends.summary_observations:
                lines.append(f"- {obs}")

        # Outlier highlights
        high_outliers = [
            f"`{col}` ({dist.outlier_pct_iqr:.1f}%)"
            for col, dist in self.distributions.items()
            if dist.outlier_pct_iqr > 2.0
        ]
        if high_outliers:
            lines.append(f"\n**Elevated Outlier Rates (>2% IQR)**: {', '.join(high_outliers)}")

        return "\n".join(lines)


class EDAPipeline:
    """
    Unified exploratory data analysis pipeline for upstream oil & gas datasets.
    """

    def __init__(
        self,
        collinearity_threshold: float = 0.85,
        alpha: float = 0.05,
        n_bins: int = 10,
    ):
        self.profiler = DataProfiler()
        self.correlation_analyzer = CorrelationAnalyzer(collinearity_threshold=collinearity_threshold)
        self.distribution_analyzer = DistributionAnalyzer(alpha=alpha, n_bins=n_bins)
        self.trend_analyzer = TrendAnalyzer()

    def run(
        self,
        df: pd.DataFrame,
        dataset_name: Optional[str] = None,
        dataset_type: Optional[str] = None,
        target_col: Optional[str] = None,
        time_col: str = "timestamp",
        oil_col: Optional[str] = "oil_rate",
        gas_col: Optional[str] = "gas_rate",
        water_col: Optional[str] = "water_rate",
        water_cut_col: Optional[str] = "water_cut",
        pressure_col: Optional[str] = "tubing_pressure",
    ) -> EDAReport:
        """
        Executes end-to-end exploratory data analysis across input DataFrame.
        """
        # 1. Dataset Profiling
        profile = self.profiler.profile_dataframe(df)

        # 2. Correlation Analysis
        correlations = self.correlation_analyzer.analyze(
            df=df,
            numeric_cols=profile.numeric_columns,
            target_col=target_col,
        )

        # 3. Distribution & Outlier Analysis
        distributions = self.distribution_analyzer.analyze(
            df=df,
            numeric_cols=profile.numeric_columns,
        )

        # 4. Petroleum Trend Analysis (if time and production/pressure columns are present)
        trends = None
        has_time = time_col in df.columns or isinstance(df.index, pd.DatetimeIndex)
        has_rates = any(col in df.columns for col in [oil_col, gas_col, water_col, water_cut_col, pressure_col] if col)

        if has_time and has_rates:
            trends = self.trend_analyzer.analyze(
                df=df,
                time_col=time_col,
                oil_col=oil_col,
                gas_col=gas_col,
                water_col=water_col,
                water_cut_col=water_cut_col,
                pressure_col=pressure_col,
            )

        # 5. Synthesize Key Findings
        key_findings = self._synthesize_findings(profile, correlations, distributions, trends)

        return EDAReport(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            profile=profile,
            correlations=correlations,
            distributions=distributions,
            trends=trends,
            key_findings=key_findings,
        )

    def _synthesize_findings(
        self,
        profile: DatasetProfile,
        correlations: CorrelationReport,
        distributions: Dict[str, DistributionProfile],
        trends: Optional[ProductionTrendReport],
    ) -> List[str]:
        findings: List[str] = []

        # Data completeness
        if profile.overall_missing_pct > 15.0:
            findings.append(
                f"Elevated dataset missingness: {profile.overall_missing_pct}% of total cells are missing across "
                f"{len(profile.numeric_columns)} numeric features."
            )
        elif profile.overall_missing_pct == 0.0:
            findings.append("Complete dataset: zero missing cells across all features.")

        # Multicollinearity
        if correlations.collinear_pairs:
            findings.append(
                f"Detected {len(correlations.collinear_pairs)} highly collinear variable pair(s) (|r| >= "
                f"{self.correlation_analyzer.collinearity_threshold}), suggesting redundant telemetry channels."
            )

        # Top target correlation
        if correlations.target_correlations:
            top_t = correlations.target_correlations[0]
            findings.append(
                f"Strongest predictor for '{top_t.target_feature}' is '{top_t.feature}' "
                f"(Pearson r={top_t.pearson:.2f}, Spearman r={top_t.spearman:.2f})."
            )

        # Outliers & Normality
        non_normal = [c for c, d in distributions.items() if not d.is_normal and d.sample_size >= 20]
        if non_normal:
            findings.append(
                f"{len(non_normal)} numeric feature(s) deviate significantly from Gaussian distribution (p < 0.05), "
                f"requiring robust non-parametric estimators or tree-based models."
            )

        high_outliers = [c for c, d in distributions.items() if d.outlier_pct_iqr > 2.0]
        if high_outliers:
            findings.append(
                f"Features {high_outliers} display elevated outlier rates (>2% IQR Tukey fences), indicative of transient "
                f"surges or process disturbances."
            )

        # Trends
        if trends:
            findings.extend(trends.summary_observations)

        return findings
