"""
PetroRAG Data Quality Validation Engine (Module 3.3)
Audits structured Oil & Gas datasets for missing fields, duplicate timestamps,
physical impossibilities, sensor dropouts, and statistical outliers.
Strictly distinguishes Data Errors from real Operational Anomalies.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Literal
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.core.logging import logger
from src.analytics.validation.rules import (
    IssueCategory,
    PHYSICAL_BOUNDARIES,
    ParameterBoundary,
)


class DataQualityIssue(BaseModel):
    """Detailed record of a flagged data quality condition."""
    row_index: int
    entity_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    parameter: str
    observed_value: Any
    category: IssueCategory
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    description: str
    suggested_action: str


class TimeGapInfo(BaseModel):
    """Identified sensor communication loss or sampling blackout."""
    entity_id: str
    gap_start: datetime
    gap_end: datetime
    duration_hours: float


class DataQualityReport(BaseModel):
    """Comprehensive data quality audit report."""
    total_rows: int
    clean_rows: int
    missing_percentage: Dict[str, float]
    duplicate_count: int
    duplicate_percentage: float
    invalid_rows_count: int
    data_error_count: int
    operational_anomaly_count: int
    time_gaps: List[TimeGapInfo]
    outlier_counts: Dict[str, int]
    quality_score: float = Field(..., ge=0.0, le=100.0)
    is_clean: bool
    issues: List[DataQualityIssue]
    summary_narrative: str


class DataQualityValidator:
    """
    Validates industrial O&G tabular datasets against first-principles physical laws,
    sampling frequency consistency, and statistical distribution parameters.
    """

    def __init__(self, physical_rules: Optional[Dict[str, ParameterBoundary]] = None):
        self.rules = physical_rules or PHYSICAL_BOUNDARIES

    def validate(
        self,
        df: pd.DataFrame,
        expected_interval_hours: Optional[float] = None,
        max_issues_to_record: int = 500,
    ) -> DataQualityReport:
        """
        Execute full data quality verification pipeline over a DataFrame.
        Does NOT blindly delete or modify values; generates diagnostic telemetry.
        """
        logger.info(f"Initiating data quality audit over DataFrame with {len(df)} rows and {len(df.columns)} columns.")

        if df.empty:
            return DataQualityReport(
                total_rows=0,
                clean_rows=0,
                missing_percentage={},
                duplicate_count=0,
                duplicate_percentage=0.0,
                invalid_rows_count=0,
                data_error_count=0,
                operational_anomaly_count=0,
                time_gaps=[],
                outlier_counts={},
                quality_score=0.0,
                is_clean=False,
                issues=[],
                summary_narrative="Dataset is empty.",
            )

        df_work = df.copy()
        total_rows = len(df_work)
        issues: List[DataQualityIssue] = []

        # 1. Missing Values Audit
        missing_pct: Dict[str, float] = {}
        for col in df_work.columns:
            null_count = int(df_work[col].isna().sum())
            pct = round((null_count / total_rows) * 100.0, 2)
            missing_pct[col] = pct
            if pct > 0:
                null_indices = df_work[df_work[col].isna()].index.tolist()
                for idx in null_indices[:20]:  # Cap sample records
                    issues.append(
                        DataQualityIssue(
                            row_index=int(idx),
                            parameter=col,
                            observed_value=None,
                            category=IssueCategory.MISSING_VALUE,
                            severity="LOW" if pct < 5.0 else "MEDIUM",
                            description=f"Missing value (null/NaN) in column '{col}'.",
                            suggested_action="Impute via forward-fill if transient, or flag sensor connection.",
                        )
                    )

        # 2. Duplicate Timestamps Audit
        duplicate_count = 0
        entity_col = "well_id" if "well_id" in df_work.columns else ("equipment_id" if "equipment_id" in df_work.columns else None)
        time_col = "timestamp" if "timestamp" in df_work.columns else ("date" if "date" in df_work.columns else None)

        if entity_col and time_col:
            dup_mask = df_work.duplicated(subset=[entity_col, time_col], keep=False)
            duplicate_count = int(dup_mask.sum())
            dup_indices = df_work[dup_mask].index.tolist()
            for idx in dup_indices[:30]:
                entity_val = str(df_work.loc[idx, entity_col])
                ts_val = df_work.loc[idx, time_col]
                issues.append(
                    DataQualityIssue(
                        row_index=int(idx),
                        entity_id=entity_val,
                        timestamp=pd.to_datetime(ts_val).to_pydatetime() if pd.notna(ts_val) else None,
                        parameter=time_col,
                        observed_value=str(ts_val),
                        category=IssueCategory.DUPLICATE_RECORD,
                        severity="HIGH",
                        description=f"Duplicate record for entity '{entity_val}' at timestamp '{ts_val}'.",
                        suggested_action="Deduplicate by retaining the latest logged record or averaging readings.",
                    )
                )

        duplicate_pct = round((duplicate_count / total_rows) * 100.0, 2)

        # 3. Physical Boundaries vs Operational Anomalies Audit
        data_error_count = 0
        operational_anomaly_count = 0
        outlier_counts: Dict[str, int] = {}

        for col, boundary in self.rules.items():
            if col not in df_work.columns:
                continue

            # Ensure numeric conversion
            numeric_series = pd.to_numeric(df_work[col], errors="coerce")
            outlier_counts[col] = 0

            # Calculate statistical IQR bounds
            clean_num = numeric_series.dropna()
            iqr_low, iqr_high = -np.inf, np.inf
            if len(clean_num) >= 10:
                q25, q75 = np.percentile(clean_num, 25), np.percentile(clean_num, 75)
                iqr = max(1e-4, q75 - q25)
                iqr_low = q25 - 2.5 * iqr
                iqr_high = q75 + 2.5 * iqr

            for idx, val in numeric_series.items():
                if pd.isna(val):
                    continue

                entity_val = str(df_work.loc[idx, entity_col]) if entity_col else None
                ts_val = df_work.loc[idx, time_col] if time_col else None
                ts_dt = pd.to_datetime(ts_val).to_pydatetime() if pd.notna(ts_val) else None

                # Check 3A: Physical Impossibility -> DATA_ERROR
                if val < boundary.min_physical or val > boundary.max_physical:
                    data_error_count += 1
                    issues.append(
                        DataQualityIssue(
                            row_index=int(idx),
                            entity_id=entity_val,
                            timestamp=ts_dt,
                            parameter=col,
                            observed_value=float(val),
                            category=IssueCategory.DATA_ERROR,
                            severity="CRITICAL",
                            description=(
                                f"Physically impossible measurement {val:.2f} {boundary.unit_symbol} "
                                f"outside valid physical envelope [{boundary.min_physical}, {boundary.max_physical}]."
                            ),
                            suggested_action="Quarantine row from ML training. Inspect sensor calibration or wire fault.",
                        )
                    )

                # Check 3B: Nominal Envelope Violation -> OPERATIONAL_ANOMALY
                elif val < boundary.min_nominal or val > boundary.max_nominal:
                    operational_anomaly_count += 1
                    issues.append(
                        DataQualityIssue(
                            row_index=int(idx),
                            entity_id=entity_val,
                            timestamp=ts_dt,
                            parameter=col,
                            observed_value=float(val),
                            category=IssueCategory.OPERATIONAL_ANOMALY,
                            severity="HIGH",
                            description=(
                                f"Operational deviation: {val:.2f} {boundary.unit_symbol} violates nominal "
                                f"limits [{boundary.min_nominal}, {boundary.max_nominal}]. Physically valid."
                            ),
                            suggested_action="Retain for anomaly detection pipeline. Do NOT drop as bad data.",
                        )
                    )

                # Check 3C: Statistical Outlier (within physical/nominal but extreme IQR)
                elif val < iqr_low or val > iqr_high:
                    outlier_counts[col] += 1

        # 4. Timestamp Continuity & Sensor Gaps
        time_gaps: List[TimeGapInfo] = []
        if entity_col and time_col:
            for entity, group in df_work.groupby(entity_col):
                grp_sorted = group.dropna(subset=[time_col]).sort_values(by=time_col)
                if len(grp_sorted) < 2:
                    continue

                ts_series = pd.to_datetime(grp_sorted[time_col])
                deltas = ts_series.diff().dt.total_seconds() / 3600.0  # hours

                # Establish baseline threshold
                if expected_interval_hours:
                    threshold_hours = expected_interval_hours * 2.5
                else:
                    median_delta = deltas.median()
                    threshold_hours = max(2.0, median_delta * 3.0) if pd.notna(median_delta) else 24.0

                gap_mask = deltas > threshold_hours
                for gap_idx in grp_sorted[gap_mask].index:
                    end_time = ts_series.loc[gap_idx].to_pydatetime()
                    # Previous timestamp
                    prev_pos = grp_sorted.index.get_loc(gap_idx) - 1
                    prev_idx = grp_sorted.index[prev_pos]
                    start_time = ts_series.loc[prev_idx].to_pydatetime()
                    dur_hours = round(float(deltas.loc[gap_idx]), 2)

                    time_gaps.append(
                        TimeGapInfo(
                            entity_id=str(entity),
                            gap_start=start_time,
                            gap_end=end_time,
                            duration_hours=dur_hours,
                        )
                    )
                    issues.append(
                        DataQualityIssue(
                            row_index=int(gap_idx),
                            entity_id=str(entity),
                            timestamp=end_time,
                            parameter=time_col,
                            observed_value=f"{dur_hours} hours",
                            category=IssueCategory.TIMESTAMP_GAP,
                            severity="MEDIUM" if dur_hours < 24.0 else "HIGH",
                            description=f"Sensor blackout of {dur_hours}h between {start_time} and {end_time}.",
                            suggested_action="Mark gap boundary. Do not linearly interpolate across large blackouts.",
                        )
                    )

        # 5. Quality Score Calculation (0 to 100)
        # Deductions:
        # - Missing values: up to -20 pts
        avg_missing = float(np.mean(list(missing_pct.values()))) if missing_pct else 0.0
        p_missing = min(20.0, avg_missing * 2.0)

        # - Duplicates: up to -20 pts
        p_duplicates = min(20.0, duplicate_pct * 4.0)

        # - Data errors (physically impossible): up to -40 pts
        err_pct = (data_error_count / max(1, total_rows)) * 100.0
        p_errors = min(40.0, err_pct * 10.0)

        # - Sensor gaps: up to -20 pts
        p_gaps = min(20.0, len(time_gaps) * 4.0)

        quality_score = max(0.0, round(100.0 - (p_missing + p_duplicates + p_errors + p_gaps), 1))
        invalid_rows = data_error_count + duplicate_count
        clean_rows = max(0, total_rows - invalid_rows)
        is_clean = (data_error_count == 0 and duplicate_count == 0 and avg_missing < 1.0)

        # Limit issues to user budget
        reported_issues = issues[:max_issues_to_record]

        narrative = (
            f"Data Quality Score: {quality_score}/100 ({'CLEAN' if is_clean else 'FLAGGED'}). "
            f"Processed {total_rows} rows: {clean_rows} clean, {data_error_count} physical data errors, "
            f"{duplicate_count} duplicate timestamps, and {operational_anomaly_count} genuine operational anomalies. "
            f"Identified {len(time_gaps)} sensor communication gaps."
        )

        return DataQualityReport(
            total_rows=total_rows,
            clean_rows=clean_rows,
            missing_percentage=missing_pct,
            duplicate_count=duplicate_count,
            duplicate_percentage=duplicate_pct,
            invalid_rows_count=invalid_rows,
            data_error_count=data_error_count,
            operational_anomaly_count=operational_anomaly_count,
            time_gaps=time_gaps,
            outlier_counts=outlier_counts,
            quality_score=quality_score,
            is_clean=is_clean,
            issues=reported_issues,
            summary_narrative=narrative,
        )
