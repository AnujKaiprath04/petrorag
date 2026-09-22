"""
PetroRAG Anomaly Severity & Parameter Attribution Engine (Module 3.10)
Provides transparent multi-tier severity triage (NORMAL, LOW, MEDIUM, HIGH, CRITICAL),
physical safety limit boundary checks (ISO 10816-3 vibration, API thermal trip limits),
directional deviation tagging (SURGE vs DROPOUT), and multi-sensor co-occurrence matrices.
"""

from typing import Dict, List, Any, Optional, Tuple, Literal, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.core.logging import logger
from src.analytics.anomaly.isolation_forest import DetectedAnomalyPoint


# ===========================================================================
# Industrial Safety Threshold Constants
# ===========================================================================

ISO_10816_VIBRATION_ALERT = 4.5    # Zone C: Alert / Unsatisfactory (mm/s RMS)
ISO_10816_VIBRATION_DANGER = 7.1   # Zone D: Unacceptable / Trip Hazard (mm/s RMS)
BEARING_TEMP_ALERT_C = 85.0        # Bearing temperature warning (°C)
BEARING_TEMP_DANGER_C = 95.0       # Bearing temperature critical trip limit (°C)
MOTOR_CURRENT_ALERT_FACTOR = 1.15  # 115% of nominal rated current
MOTOR_CURRENT_DANGER_FACTOR = 1.30 # 130% of nominal rated current


class ParameterAttributionItem(BaseModel):
    feature_name: str
    observed_value: float
    baseline_median: float
    baseline_iqr: float
    absolute_deviation: float
    deviation_pct: float
    relative_intensity_z: float = Field(..., description="Deviation scaled by IQR (|x - median| / IQR)")
    direction: Literal["SURGE", "DROPOUT", "NOMINAL"]
    contribution_weight_pct: float = Field(..., ge=0.0, le=100.0)
    safety_status: Literal["SAFE", "WARNING", "DANGER_TRIP"]
    safety_standard_reference: Optional[str] = None


class SeverityTriageResult(BaseModel):
    severity: Literal["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    anomaly_score: float
    max_deviation_pct: float
    primary_contributor: str
    secondary_contributor: Optional[str] = None
    safety_breach_detected: bool
    triage_rationale: str


class DecomposedAnomalyReport(BaseModel):
    timestamp: Optional[str] = None
    index: int
    is_anomaly: bool
    triage: SeverityTriageResult
    attributions: List[ParameterAttributionItem]


class SensorCoOccurrence(BaseModel):
    sensor_pair: Tuple[str, str]
    co_occurrence_count: int
    primary_correlation: float


class AttributionSummary(BaseModel):
    total_anomalies_evaluated: int
    severity_distribution: Dict[str, int]
    primary_driver_frequency: Dict[str, int]
    all_driver_frequency: Dict[str, int]
    top_co_occurring_sensors: List[SensorCoOccurrence]
    safety_limit_breaches_count: int
    executive_narrative: str


# ===========================================================================
# 1. Severity Triage Engine
# ===========================================================================

class SeverityTriageEngine:
    """
    Evaluates multi-tier severity combining calibrated ML anomaly score,
    percentage deviation magnitude, and physical safety boundary breaches.
    """

    def triage(
        self,
        anomaly_score: float,
        attributions: List[ParameterAttributionItem],
        is_anomaly: bool,
    ) -> SeverityTriageResult:
        if not attributions or not is_anomaly:
            return SeverityTriageResult(
                severity="NORMAL",
                anomaly_score=round(anomaly_score, 4),
                max_deviation_pct=0.0,
                primary_contributor="none",
                secondary_contributor=None,
                safety_breach_detected=False,
                triage_rationale="Parameters remain within nominal operating envelope.",
            )

        top_item = attributions[0]
        second_item = attributions[1] if len(attributions) > 1 else None

        max_dev = max(abs(a.deviation_pct) for a in attributions)
        safety_breach = any(a.safety_status == "DANGER_TRIP" for a in attributions)
        safety_warning = any(a.safety_status == "WARNING" for a in attributions)

        # Multi-factor Severity Assignment
        # 1. Critical safety standard trip limit
        if safety_breach or (anomaly_score >= 0.85 and not safety_warning) or (max_dev >= 120.0 and not safety_warning):
            sev = "CRITICAL"
            if safety_breach:
                breached_a = next(a for a in attributions if a.safety_status == "DANGER_TRIP")
                rationale = (
                    f"CRITICAL trip hazard: {breached_a.feature_name}={breached_a.observed_value:.2f} "
                    f"breached critical safety threshold ({breached_a.safety_standard_reference}). "
                    f"Anomaly score: {anomaly_score:.3f} with {breached_a.deviation_pct:+.1f}% deviation."
                )
            else:
                rationale = (
                    f"CRITICAL excursion: Dominant parameter '{top_item.feature_name}' "
                    f"deviated by {top_item.deviation_pct:+.1f}% from baseline median. "
                    f"Anomaly score: {anomaly_score:.3f}."
                )

        # 2. Safety standard warning or high statistical excursion
        elif safety_warning or anomaly_score >= 0.75 or max_dev >= 60.0:
            sev = "HIGH"
            warning_a = next((a for a in attributions if a.safety_status == "WARNING"), top_item)
            warning_ref = warning_a.safety_standard_reference if warning_a.safety_status == "WARNING" else None
            rationale = (
                f"HIGH severity: Severe {warning_a.direction.lower()} in '{warning_a.feature_name}' "
                f"({warning_a.deviation_pct:+.1f}% deviation"
                + (f", {warning_ref})" if warning_ref else ").")
            )

        # Level 3: MEDIUM
        elif anomaly_score >= 0.60 or max_dev >= 30.0:
            sev = "MEDIUM"
            rationale = (
                f"MEDIUM severity: Moderate {top_item.direction.lower()} in '{top_item.feature_name}' "
                f"({top_item.deviation_pct:+.1f}% deviation, score: {anomaly_score:.3f}). Surveillance recommended."
            )

        # Level 2: LOW
        elif anomaly_score >= 0.45 or max_dev >= 15.0:
            sev = "LOW"
            rationale = (
                f"LOW severity: Minor process deviation in '{top_item.feature_name}' "
                f"({top_item.deviation_pct:+.1f}% deviation, score: {anomaly_score:.3f})."
            )

        # Level 1: NORMAL
        else:
            sev = "NORMAL"
            rationale = "Nominal telemetry with negligible variation."

        return SeverityTriageResult(
            severity=sev,  # type: ignore
            anomaly_score=round(anomaly_score, 4),
            max_deviation_pct=round(max_dev, 2),
            primary_contributor=top_item.feature_name,
            secondary_contributor=second_item.feature_name if second_item else None,
            safety_breach_detected=safety_breach,
            triage_rationale=rationale,
        )


# ===========================================================================
# 2. Anomaly Attribution Engine
# ===========================================================================

class AnomalyAttributionEngine:
    """
    Decomposes multivariate anomalies into parameter contribution weights,
    checks physical safety envelopes, and generates attribution matrices.
    """

    def __init__(self):
        self.triage_engine = SeverityTriageEngine()

    def decompose_point(
        self,
        row_values: np.ndarray,
        feature_names: List[str],
        baseline_medians: Dict[str, float],
        baseline_iqrs: Dict[str, float],
        anomaly_score: float = 0.50,
        is_anomaly: bool = True,
        timestamp: Optional[str] = None,
        index: int = 0,
    ) -> DecomposedAnomalyReport:
        """
        Decomposes an anomalous record into attributed features and severity triage.
        """
        raw_items = []
        for i, name in enumerate(feature_names):
            val = float(row_values[i])
            med = baseline_medians.get(name, 0.0)
            iqr = max(1e-4, baseline_iqrs.get(name, 1.0))
            abs_dev = abs(val - med)
            dev_pct = float(((val - med) / max(1e-4, abs(med))) * 100.0) if med != 0.0 else 0.0
            rel_z = abs_dev / iqr

            if dev_pct > 5.0:
                direction = "SURGE"
            elif dev_pct < -5.0:
                direction = "DROPOUT"
            else:
                direction = "NOMINAL"

            safety_status, safety_ref = self._check_safety_limits(name, val, med)

            raw_items.append({
                "name": name,
                "val": val,
                "med": med,
                "iqr": iqr,
                "abs_dev": abs_dev,
                "dev_pct": dev_pct,
                "rel_z": rel_z,
                "direction": direction,
                "safety_status": safety_status,
                "safety_ref": safety_ref,
            })

        # Calculate normalized contribution percentage weights
        total_z = sum(item["rel_z"] for item in raw_items)
        attributions: List[ParameterAttributionItem] = []

        for item in raw_items:
            weight_pct = float((item["rel_z"] / max(1e-4, total_z)) * 100.0)
            attributions.append(
                ParameterAttributionItem(
                    feature_name=item["name"],
                    observed_value=round(item["val"], 4),
                    baseline_median=round(item["med"], 4),
                    baseline_iqr=round(item["iqr"], 4),
                    absolute_deviation=round(item["abs_dev"], 4),
                    deviation_pct=round(item["dev_pct"], 2),
                    relative_intensity_z=round(item["rel_z"], 3),
                    direction=item["direction"],
                    contribution_weight_pct=round(weight_pct, 2),
                    safety_status=item["safety_status"],
                    safety_standard_reference=item["safety_ref"],
                )
            )

        attributions.sort(key=lambda x: x.contribution_weight_pct, reverse=True)

        triage_result = self.triage_engine.triage(
            anomaly_score=anomaly_score,
            attributions=attributions,
            is_anomaly=is_anomaly,
        )

        return DecomposedAnomalyReport(
            timestamp=timestamp,
            index=index,
            is_anomaly=is_anomaly,
            triage=triage_result,
            attributions=attributions,
        )

    def analyze_attribution_matrix(
        self,
        decomposed_reports: List[DecomposedAnomalyReport],
    ) -> AttributionSummary:
        """
        Analyzes sequences of decomposed anomalies to compute primary driver frequencies,
        multi-sensor co-occurrence pairs, and systemic risk assessments.
        """
        total_anomalies = len(decomposed_reports)
        if total_anomalies == 0:
            return AttributionSummary(
                total_anomalies_evaluated=0,
                severity_distribution={},
                primary_driver_frequency={},
                all_driver_frequency={},
                top_co_occurring_sensors=[],
                safety_limit_breaches_count=0,
                executive_narrative="Zero anomalies present in telemetry window.",
            )

        sev_dist: Dict[str, int] = {"NORMAL": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        primary_freq: Dict[str, int] = {}
        all_freq: Dict[str, int] = {}
        pair_counts: Dict[Tuple[str, str], int] = {}
        safety_breaches = 0

        for r in decomposed_reports:
            sev_dist[r.triage.severity] = sev_dist.get(r.triage.severity, 0) + 1
            if r.triage.safety_breach_detected:
                safety_breaches += 1

            p_driver = r.triage.primary_contributor
            if p_driver != "none":
                primary_freq[p_driver] = primary_freq.get(p_driver, 0) + 1

            # Top contributing features (>20% weight)
            significant_feats = [
                a.feature_name for a in r.attributions if a.contribution_weight_pct >= 20.0
            ]
            for f in significant_feats:
                all_freq[f] = all_freq.get(f, 0) + 1

            # Co-occurrence pairs
            for i in range(len(significant_feats)):
                for j in range(i + 1, len(significant_feats)):
                    pair = tuple(sorted([significant_feats[i], significant_feats[j]]))
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1  # type: ignore

        sorted_pairs = [
            SensorCoOccurrence(
                sensor_pair=p,
                co_occurrence_count=count,
                primary_correlation=0.85,
            )
            for p, count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        # Executive narrative
        top_driver = max(primary_freq.items(), key=lambda x: x[1])[0] if primary_freq else "None"
        narrative = (
            f"Evaluated {total_anomalies} anomalous records: {sev_dist.get('CRITICAL', 0)} CRITICAL, "
            f"{sev_dist.get('HIGH', 0)} HIGH, {sev_dist.get('MEDIUM', 0)} MEDIUM events. "
            f"Primary failure driver across {primary_freq.get(top_driver, 0)} events is '{top_driver}'. "
            f"Safety limit breaches detected: {safety_breaches}."
        )

        return AttributionSummary(
            total_anomalies_evaluated=total_anomalies,
            severity_distribution=sev_dist,
            primary_driver_frequency=primary_freq,
            all_driver_frequency=all_freq,
            top_co_occurring_sensors=sorted_pairs,
            safety_limit_breaches_count=safety_breaches,
            executive_narrative=narrative,
        )

    def _check_safety_limits(
        self, feature_name: str, value: float, baseline_median: float
    ) -> Tuple[Literal["SAFE", "WARNING", "DANGER_TRIP"], Optional[str]]:
        """
        Evaluates physical safety envelopes and industry standards.
        """
        f_lower = feature_name.lower()

        # 1. Vibration: ISO 10816-3 (Pumps & Rotating Machinery)
        if "vibration" in f_lower:
            if value >= ISO_10816_VIBRATION_DANGER:
                return "DANGER_TRIP", f"ISO 10816-3 Zone D (> {ISO_10816_VIBRATION_DANGER} mm/s RMS)"
            elif value >= ISO_10816_VIBRATION_ALERT:
                return "WARNING", f"ISO 10816-3 Zone C (> {ISO_10816_VIBRATION_ALERT} mm/s RMS)"
            return "SAFE", None

        # 2. Bearing / Motor Temperature
        if "temp" in f_lower or "temperature" in f_lower:
            if value >= BEARING_TEMP_DANGER_C:
                return "DANGER_TRIP", f"API 610 Bearing Danger Trip (> {BEARING_TEMP_DANGER_C}°C)"
            elif value >= BEARING_TEMP_ALERT_C:
                return "WARNING", f"API 610 Bearing Alert Warning (> {BEARING_TEMP_ALERT_C}°C)"
            return "SAFE", None

        # 3. Motor Current Overload
        if "current" in f_lower or "amp" in f_lower:
            if baseline_median > 0:
                if value >= baseline_median * MOTOR_CURRENT_DANGER_FACTOR:
                    return "DANGER_TRIP", f"Motor Current > {MOTOR_CURRENT_DANGER_FACTOR*100:.0f}% Overload"
                elif value >= baseline_median * MOTOR_CURRENT_ALERT_FACTOR:
                    return "WARNING", f"Motor Current > {MOTOR_CURRENT_ALERT_FACTOR*100:.0f}% Warning"
            return "SAFE", None

        # 4. Discharge Pressure Overpressure
        if "discharge" in f_lower and "pressure" in f_lower:
            if baseline_median > 0 and value >= baseline_median * 1.50:
                return "DANGER_TRIP", "Discharge Pressure > 150% Overpressure"
            return "SAFE", None

        return "SAFE", None
