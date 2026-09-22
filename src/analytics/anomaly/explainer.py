"""
PetroRAG Anomaly Factual Explanation Formatter (Module 3.12)
Transforms raw numerical anomaly detections into structured, auditable,
and factual 4-part operational diagnostic explanations for petroleum engineers:
  Part 1: Telemetry Observations (exact observed vs nominal baseline, deviation %, direction)
  Part 2: Statistical & Model Confidence (calibrated score, multi-model consensus)
  Part 3: Safety Envelope Status (ISO 10816-3 vibration, API 610 thermal limits, electrical)
  Part 4: Operational Risk & Recommended Remedial Actions (failure hypotheses, SOP checklist, urgency)
"""

from typing import Dict, List, Any, Optional, Tuple, Literal, Union
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.core.logging import logger
from src.analytics.anomaly.attribution import (
    ParameterAttributionItem,
    SeverityTriageResult,
    DecomposedAnomalyReport,
    AnomalyAttributionEngine,
    ISO_10816_VIBRATION_ALERT,
    ISO_10816_VIBRATION_DANGER,
    BEARING_TEMP_ALERT_C,
    BEARING_TEMP_DANGER_C,
    MOTOR_CURRENT_ALERT_FACTOR,
    MOTOR_CURRENT_DANGER_FACTOR,
)
from src.analytics.anomaly.isolation_forest import (
    IsolationForestAnomalyDetector,
    DetectedAnomalyPoint,
)
from src.analytics.anomaly.baselines import (
    RollingZScoreAnomalyDetector,
    IQRAnomalyDetector,
    MADAnomalyDetector,
)


# ===========================================================================
# Engineering Units & Reference Constants
# ===========================================================================

ENGINEERING_UNITS: Dict[str, str] = {
    "vibration_rms": "mm/s RMS",
    "bearing_temp": "°C",
    "discharge_pressure": "bar",
    "suction_pressure": "bar",
    "motor_current": "A",
    "rpm": "RPM",
    "flow_rate": "m³/h",
    "water_cut": "%",
    "gor": "scf/bbl",
    "thp": "bar",
    "chp": "bar",
}

# ISO 10816-3 Vibration Severity Zones (Class II / Class III Rotating Machinery)
ISO_ZONE_A_MAX = 2.3  # mm/s RMS (Newly commissioned / Good)
ISO_ZONE_B_MAX = 4.5  # mm/s RMS (Acceptable for unrestricted long-term operation)
ISO_ZONE_C_MAX = 7.1  # mm/s RMS (Unsatisfactory / Alert / Restricted operation)
# > 7.1 mm/s RMS is Zone D (Danger / Unacceptable / Immediate Trip Hazard)


# ===========================================================================
# 4-Part Structured Data Schemas
# ===========================================================================

class ChannelObservation(BaseModel):
    """Part 1: Detailed telemetry observation for a single sensor channel."""
    channel: str
    unit: str
    observed_value: float
    baseline_median: float
    baseline_iqr: float
    absolute_deviation: float
    deviation_pct: float
    relative_intensity_z: float
    direction: Literal["SURGE", "DROPOUT", "NOMINAL"]
    contribution_weight_pct: float
    is_primary_driver: bool = False
    is_secondary_driver: bool = False


class TelemetryObservationSection(BaseModel):
    """Part 1: Full Telemetry Observations Section."""
    observations: List[ChannelObservation]
    primary_driver: str
    secondary_driver: Optional[str] = None
    max_deviation_pct: float
    summary_statement: str


class StatisticalConfidenceSection(BaseModel):
    """Part 2: Statistical & Model Confidence Section."""
    primary_model_name: str = "IsolationForest"
    primary_anomaly_score: float = Field(..., ge=0.0, le=1.0)
    ensemble_models_evaluated: List[str] = Field(default_factory=list)
    ensemble_votes: Dict[str, bool] = Field(default_factory=dict)
    models_flagging_count: int = 1
    total_models_evaluated: int = 1
    consensus_ratio: float = Field(..., ge=0.0, le=1.0)
    confidence_tier: Literal["VERY_HIGH", "HIGH", "MEDIUM", "LOW"]
    statistical_rationale: str


class SafetyEnvelopeSection(BaseModel):
    """Part 3: Safety Envelope & Physical Standards Compliance Section."""
    iso_vibration_zone: Optional[Literal["ZONE_A", "ZONE_B", "ZONE_C_ALERT", "ZONE_D_TRIP"]] = None
    iso_vibration_value_mms: Optional[float] = None
    api_bearing_temp_status: Optional[Literal["SAFE", "WARNING_ALERT", "CRITICAL_TRIP"]] = None
    api_bearing_temp_c: Optional[float] = None
    motor_current_status: Optional[Literal["SAFE", "OVERLOAD_ALERT", "OVERLOAD_TRIP"]] = None
    motor_current_observed_a: Optional[float] = None
    hydraulic_pressure_status: Optional[Literal["NOMINAL", "PRESSURE_DROPOUT", "OVERPRESSURE"]] = None
    safety_breach_detected: bool = False
    highest_safety_tier: str = "SAFE"
    governing_standards: List[str] = Field(default_factory=list)
    safety_summary: str


class OperationalRemediationSection(BaseModel):
    """Part 4: Operational Risk & Recommended Remedial Actions Section."""
    diagnosed_failure_mode: str
    failure_mode_description: str
    urgency_level: Literal["IMMEDIATE_SHUTDOWN", "URGENT_INSPECTION_24H", "MONITOR_NEXT_SHIFT", "ROUTINE_SURVEILLANCE"]
    recommended_sop_steps: List[str]
    operator_action_summary: str
    preventive_maintenance_id: Optional[str] = None


class FactualAnomalyExplanation(BaseModel):
    """
    Complete, auditable, 4-part factual operational anomaly explanation.
    Guarantees zero hallucinated metrics by deriving every figure directly
    from verified sensor readings and international engineering standards.
    """
    equipment_id: str = "UNKNOWN_EQUIPMENT"
    timestamp: str
    index: int = 0
    is_anomaly: bool
    overall_severity: Literal["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    telemetry_observations: TelemetryObservationSection
    statistical_confidence: StatisticalConfidenceSection
    safety_envelope: SafetyEnvelopeSection
    operational_remediation: OperationalRemediationSection

    def to_text_report(self) -> str:
        """Generates a clean, human-readable operational incident bulletin."""
        lines = [
            f"================================================================================",
            f"PETRORAG OPERATIONAL ANOMALY BULLETIN - {self.equipment_id.upper()}",
            f"Timestamp: {self.timestamp} | Severity: [{self.overall_severity}] | Urgency: [{self.operational_remediation.urgency_level}]",
            f"================================================================================",
            f"",
            f"1. TELEMETRY OBSERVATIONS:",
            f"   - {self.telemetry_observations.summary_statement}",
        ]
        for obs in self.telemetry_observations.observations:
            star = " (PRIMARY DRIVER)" if obs.is_primary_driver else (" (SECONDARY)" if obs.is_secondary_driver else "")
            lines.append(
                f"   * {obs.channel.upper()}: Observed={obs.observed_value:.2f} {obs.unit} "
                f"(Baseline Median={obs.baseline_median:.2f} {obs.unit}, Dev={obs.deviation_pct:+.1f}%, "
                f"Dir={obs.direction}, Rel-Z={obs.relative_intensity_z:.2f}){star}"
            )

        lines.extend([
            f"",
            f"2. STATISTICAL & MODEL CONFIDENCE:",
            f"   - Primary ML Score: {self.statistical_confidence.primary_anomaly_score:.3f} ({self.statistical_confidence.primary_model_name})",
            f"   - Ensemble Agreement: {self.statistical_confidence.models_flagging_count}/{self.statistical_confidence.total_models_evaluated} models "
            f"({self.statistical_confidence.consensus_ratio * 100:.0f}% consensus) -> Confidence: [{self.statistical_confidence.confidence_tier}]",
            f"   - Rationale: {self.statistical_confidence.statistical_rationale}",
            f"",
            f"3. SAFETY ENVELOPE STATUS:",
            f"   - Safety Breach Detected: {self.safety_envelope.safety_breach_detected} (Highest Tier: {self.safety_envelope.highest_safety_tier})",
            f"   - Standards Evaluated: {', '.join(self.safety_envelope.governing_standards) if self.safety_envelope.governing_standards else 'None'}",
            f"   - Assessment: {self.safety_envelope.safety_summary}",
            f"",
            f"4. OPERATIONAL RISK & RECOMMENDED ACTIONS:",
            f"   - Diagnosed Failure Mode: {self.operational_remediation.diagnosed_failure_mode}",
            f"   - Description: {self.operational_remediation.failure_mode_description}",
            f"   - Action Required: {self.operational_remediation.operator_action_summary}",
            f"   - Recommended SOP Checklist:",
        ])
        for i, step in enumerate(self.operational_remediation.recommended_sop_steps, 1):
            lines.append(f"     [{i}] {step}")

        lines.append(f"================================================================================")
        return "\n".join(lines)

    def to_rag_context(self) -> str:
        """
        Formats factual structured grounding context for LLM prompt injection.
        Employs unambiguous delimiters and pre-formatted facts to prevent hallucination.
        """
        obs_lines = []
        for obs in self.telemetry_observations.observations:
            obs_lines.append(
                f"- Channel: {obs.channel}, Observed: {obs.observed_value} {obs.unit}, "
                f"Baseline: {obs.baseline_median} {obs.unit}, Deviation: {obs.deviation_pct}%, "
                f"Direction: {obs.direction}, Weight: {obs.contribution_weight_pct}%"
            )

        rag_block = f"""<FACTUAL_ANOMALY_GROUNDING>
Equipment: {self.equipment_id}
Timestamp: {self.timestamp}
Overall Severity: {self.overall_severity}
Operational Urgency: {self.operational_remediation.urgency_level}
Primary Contributing Sensor: {self.telemetry_observations.primary_driver}

[SECTION 1: TELEMETRY READINGS]
{chr(10).join(obs_lines)}

[SECTION 2: MODEL CONFIDENCE]
Primary Score: {self.statistical_confidence.primary_anomaly_score}
Model Consensus: {self.statistical_confidence.models_flagging_count} of {self.statistical_confidence.total_models_evaluated} detectors flagging
Confidence Level: {self.statistical_confidence.confidence_tier}

[SECTION 3: SAFETY ENVELOPE]
ISO 10816-3 Vibration Zone: {self.safety_envelope.iso_vibration_zone}
API 610 Bearing Temp Status: {self.safety_envelope.api_bearing_temp_status}
Safety Breach Detected: {self.safety_envelope.safety_breach_detected}
Applicable Standards: {', '.join(self.safety_envelope.governing_standards)}

[SECTION 4: DIAGNOSTIC REMEDIATION]
Hypothesized Failure Mode: {self.operational_remediation.diagnosed_failure_mode}
Recommended Remedial Steps:
{chr(10).join(f"- {step}" for step in self.operational_remediation.recommended_sop_steps)}
</FACTUAL_ANOMALY_GROUNDING>"""
        return rag_block


# ===========================================================================
# Core Anomaly Explanation Formatter Engine
# ===========================================================================

class AnomalyExplanationFormatter:
    """
    Deterministic rule-based formatter that converts decomposed anomaly reports
    into factual, auditable, multi-part engineering explanations.
    """

    def __init__(self, default_equipment_id: str = "ESP_WELL_PROD"):
        self.default_equipment_id = default_equipment_id
        self.attribution_engine = AnomalyAttributionEngine()

    def format_explanation(
        self,
        decomposed_report: DecomposedAnomalyReport,
        ensemble_votes: Optional[Dict[str, bool]] = None,
        equipment_id: Optional[str] = None,
    ) -> FactualAnomalyExplanation:
        """
        Constructs a complete FactualAnomalyExplanation from a DecomposedAnomalyReport.
        """
        eq_id = equipment_id or self.default_equipment_id
        ts = decomposed_report.timestamp or datetime.now(timezone.utc).isoformat()
        attributions = decomposed_report.attributions
        triage = decomposed_report.triage

        # -------------------------------------------------------------------
        # Part 1: Telemetry Observations
        # -------------------------------------------------------------------
        channel_obs: List[ChannelObservation] = []
        for item in attributions:
            unit = ENGINEERING_UNITS.get(item.feature_name.lower(), "units")
            is_prim = (item.feature_name == triage.primary_contributor)
            is_sec = (item.feature_name == triage.secondary_contributor)

            channel_obs.append(
                ChannelObservation(
                    channel=item.feature_name,
                    unit=unit,
                    observed_value=item.observed_value,
                    baseline_median=item.baseline_median,
                    baseline_iqr=item.baseline_iqr,
                    absolute_deviation=item.absolute_deviation,
                    deviation_pct=item.deviation_pct,
                    relative_intensity_z=item.relative_intensity_z,
                    direction=item.direction,
                    contribution_weight_pct=item.contribution_weight_pct,
                    is_primary_driver=is_prim,
                    is_secondary_driver=is_sec,
                )
            )

        if channel_obs:
            prim_obs = next((o for o in channel_obs if o.is_primary_driver), channel_obs[0])
            obs_summary = (
                f"Primary telemetry excursion driven by {prim_obs.channel} ({prim_obs.direction} of "
                f"{prim_obs.deviation_pct:+.1f}% to {prim_obs.observed_value:.2f} {prim_obs.unit}, "
                f"nominal baseline: {prim_obs.baseline_median:.2f} {prim_obs.unit})."
            )
        else:
            obs_summary = "Nominal telemetry state across all monitored channels."

        telemetry_section = TelemetryObservationSection(
            observations=channel_obs,
            primary_driver=triage.primary_contributor,
            secondary_driver=triage.secondary_contributor,
            max_deviation_pct=triage.max_deviation_pct,
            summary_statement=obs_summary,
        )

        # -------------------------------------------------------------------
        # Part 2: Statistical & Model Confidence
        # -------------------------------------------------------------------
        stat_section = self._build_statistical_section(
            anomaly_score=triage.anomaly_score,
            is_anomaly=decomposed_report.is_anomaly,
            ensemble_votes=ensemble_votes,
            max_rel_z=max([o.relative_intensity_z for o in channel_obs], default=0.0),
        )

        # -------------------------------------------------------------------
        # Part 3: Safety Envelope & Physical Standards
        # -------------------------------------------------------------------
        safety_section = self._build_safety_section(channel_obs)

        # -------------------------------------------------------------------
        # Part 4: Operational Remediation & Diagnostic SOPs
        # -------------------------------------------------------------------
        remediation_section = self._build_remediation_section(
            triage=triage,
            channel_obs=channel_obs,
            safety_section=safety_section,
        )

        return FactualAnomalyExplanation(
            equipment_id=eq_id,
            timestamp=ts,
            index=decomposed_report.index,
            is_anomaly=decomposed_report.is_anomaly,
            overall_severity=triage.severity,
            telemetry_observations=telemetry_section,
            statistical_confidence=stat_section,
            safety_envelope=safety_section,
            operational_remediation=remediation_section,
        )

    def explain_point(
        self,
        features_dict: Dict[str, float],
        baseline_medians: Dict[str, float],
        baseline_iqrs: Dict[str, float],
        anomaly_score: float = 0.5,
        is_anomaly: bool = True,
        timestamp: Optional[str] = None,
        index: int = 0,
        ensemble_votes: Optional[Dict[str, bool]] = None,
        equipment_id: Optional[str] = None,
    ) -> FactualAnomalyExplanation:
        """
        Convenience method to explain an arbitrary point from dictionaries.
        """
        feature_names = list(features_dict.keys())
        row_values = np.array([features_dict[k] for k in feature_names])

        decomposed = self.attribution_engine.decompose_point(
            row_values=row_values,
            feature_names=feature_names,
            baseline_medians=baseline_medians,
            baseline_iqrs=baseline_iqrs,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            timestamp=timestamp,
            index=index,
        )

        return self.format_explanation(
            decomposed_report=decomposed,
            ensemble_votes=ensemble_votes,
            equipment_id=equipment_id,
        )

    # -----------------------------------------------------------------------
    # Helper Section Builders
    # -----------------------------------------------------------------------

    def _build_statistical_section(
        self,
        anomaly_score: float,
        is_anomaly: bool,
        ensemble_votes: Optional[Dict[str, bool]],
        max_rel_z: float,
    ) -> StatisticalConfidenceSection:
        if ensemble_votes:
            models_evaluated = list(ensemble_votes.keys())
            flagging_count = sum(1 for v in ensemble_votes.values() if v)
            total_eval = len(ensemble_votes)
            consensus_ratio = float(flagging_count / max(1, total_eval))
        else:
            models_evaluated = ["IsolationForest"]
            flagging_count = 1 if is_anomaly else 0
            total_eval = 1
            consensus_ratio = 1.0 if is_anomaly else 0.0

        # Confidence Tier Determination
        if not is_anomaly:
            tier = "LOW"
            stat_rationale = f"Telemetry sits within statistical norms (Score={anomaly_score:.3f})."
        elif consensus_ratio >= 0.75 and (anomaly_score >= 0.75 or max_rel_z >= 4.0):
            tier = "VERY_HIGH"
            stat_rationale = (
                f"Multi-detector consensus ({flagging_count}/{total_eval}) coupled with high "
                f"calibrated ML score ({anomaly_score:.3f}) and significant deviation intensity ({max_rel_z:.1f}σ)."
            )
        elif consensus_ratio >= 0.50 or anomaly_score >= 0.60 or max_rel_z >= 3.0:
            tier = "HIGH"
            stat_rationale = (
                f"Elevated ML anomaly score ({anomaly_score:.3f}) supported by detector agreement "
                f"({flagging_count}/{total_eval})."
            )
        elif anomaly_score >= 0.40 or max_rel_z >= 2.0:
            tier = "MEDIUM"
            stat_rationale = (
                f"Moderate ML score ({anomaly_score:.3f}) indicating early-stage or subtle subspace shift."
            )
        else:
            tier = "LOW"
            stat_rationale = f"Weak statistical divergence (Score={anomaly_score:.3f})."

        return StatisticalConfidenceSection(
            primary_model_name="IsolationForest",
            primary_anomaly_score=round(anomaly_score, 4),
            ensemble_models_evaluated=models_evaluated,
            ensemble_votes=ensemble_votes or {"IsolationForest": is_anomaly},
            models_flagging_count=flagging_count,
            total_models_evaluated=total_eval,
            consensus_ratio=round(consensus_ratio, 2),
            confidence_tier=tier,
            statistical_rationale=stat_rationale,
        )

    def _build_safety_section(
        self,
        channel_obs: List[ChannelObservation],
    ) -> SafetyEnvelopeSection:
        obs_map = {o.channel.lower(): o for o in channel_obs}

        # 1. ISO 10816-3 Vibration Zone
        iso_zone = None
        vib_val = None
        if "vibration_rms" in obs_map:
            vib_val = obs_map["vibration_rms"].observed_value
            if vib_val < ISO_ZONE_A_MAX:
                iso_zone = "ZONE_A"
            elif vib_val < ISO_ZONE_B_MAX:
                iso_zone = "ZONE_B"
            elif vib_val <= ISO_ZONE_C_MAX:
                iso_zone = "ZONE_C_ALERT"
            else:
                iso_zone = "ZONE_D_TRIP"

        # 2. API 610 Bearing Temperature
        api_temp = None
        temp_val = None
        if "bearing_temp" in obs_map:
            temp_val = obs_map["bearing_temp"].observed_value
            if temp_val >= BEARING_TEMP_DANGER_C:
                api_temp = "CRITICAL_TRIP"
            elif temp_val >= BEARING_TEMP_ALERT_C:
                api_temp = "WARNING_ALERT"
            else:
                api_temp = "SAFE"

        # 3. Motor Current Overload
        motor_status = None
        motor_val = None
        if "motor_current" in obs_map:
            m_obs = obs_map["motor_current"]
            motor_val = m_obs.observed_value
            med = m_obs.baseline_median
            if med > 0 and motor_val >= med * MOTOR_CURRENT_DANGER_FACTOR:
                motor_status = "OVERLOAD_TRIP"
            elif med > 0 and motor_val >= med * MOTOR_CURRENT_ALERT_FACTOR:
                motor_status = "OVERLOAD_ALERT"
            else:
                motor_status = "SAFE"

        # 4. Hydraulic Pressure Status
        hyd_status = "NOMINAL"
        if "discharge_pressure" in obs_map:
            p_obs = obs_map["discharge_pressure"]
            if p_obs.deviation_pct <= -35.0:
                hyd_status = "PRESSURE_DROPOUT"
            elif p_obs.deviation_pct >= 35.0:
                hyd_status = "OVERPRESSURE"

        # Tiers, Standards, and Breaches
        standards = []
        breaches = []
        highest_tier = "SAFE"

        if iso_zone in ["ZONE_C_ALERT", "ZONE_D_TRIP"]:
            standards.append("ISO 10816-3 (Mechanical Vibration)")
            breaches.append(f"Vibration {vib_val:.2f} mm/s in {iso_zone}")
            if iso_zone == "ZONE_D_TRIP":
                highest_tier = "ZONE_D_TRIP"
            elif highest_tier != "ZONE_D_TRIP":
                highest_tier = "ZONE_C_ALERT"

        if api_temp in ["WARNING_ALERT", "CRITICAL_TRIP"]:
            standards.append("API 610 / API 670 (Machinery Protection)")
            breaches.append(f"Bearing temperature {temp_val:.1f}°C in {api_temp}")
            if api_temp == "CRITICAL_TRIP":
                highest_tier = "CRITICAL_TRIP"
            elif highest_tier == "SAFE":
                highest_tier = "WARNING_ALERT"

        if motor_status in ["OVERLOAD_ALERT", "OVERLOAD_TRIP"]:
            standards.append("IEC 60034-1 (Rotating Electrical Machines)")
            breaches.append(f"Motor current {motor_val:.1f}A in {motor_status}")
            if motor_status == "OVERLOAD_TRIP" and highest_tier not in ["ZONE_D_TRIP", "CRITICAL_TRIP"]:
                highest_tier = "OVERLOAD_TRIP"
            elif highest_tier == "SAFE":
                highest_tier = "OVERLOAD_ALERT"

        if hyd_status != "NOMINAL":
            standards.append("API RP 14C (Offshore Surface Safety Systems)")
            breaches.append(f"Hydraulic pressure status: {hyd_status}")

        breach_detected = len(breaches) > 0
        if breach_detected:
            summary = "SAFETY ENVELOPE BREACH DETECTED: " + "; ".join(breaches) + "."
        else:
            summary = "All monitored physical parameters reside within certified safe operating envelopes."

        return SafetyEnvelopeSection(
            iso_vibration_zone=iso_zone,
            iso_vibration_value_mms=vib_val,
            api_bearing_temp_status=api_temp,
            api_bearing_temp_c=temp_val,
            motor_current_status=motor_status,
            motor_current_observed_a=motor_val,
            hydraulic_pressure_status=hyd_status,
            safety_breach_detected=breach_detected,
            highest_safety_tier=highest_tier,
            governing_standards=standards,
            safety_summary=summary,
        )

    def _build_remediation_section(
        self,
        triage: SeverityTriageResult,
        channel_obs: List[ChannelObservation],
        safety_section: SafetyEnvelopeSection,
    ) -> OperationalRemediationSection:
        obs_map = {o.channel.lower(): o for o in channel_obs}

        vib = obs_map.get("vibration_rms")
        temp = obs_map.get("bearing_temp")
        dp = obs_map.get("discharge_pressure")
        curr = obs_map.get("motor_current")

        vib_severe = vib and vib.observed_value >= ISO_10816_VIBRATION_ALERT
        temp_severe = temp and temp.observed_value >= BEARING_TEMP_ALERT_C
        dp_dropped = dp and dp.deviation_pct <= -20.0
        curr_surged = curr and curr.deviation_pct >= 5.0
        curr_dropped = curr and curr.deviation_pct <= -10.0

        if vib_severe and temp_severe:
            failure_mode = "BEARING_DEGRADATION_OR_LUBRICATION_FAILURE"
            description = (
                "Simultaneous elevated vibration and high bearing temperature indicate severe mechanical friction, "
                "bearing race spalling, or acute lubrication failure."
            )
            sops = [
                "Verify lube oil pressure, sump temperature, and visual oil level immediately.",
                "Draw lube oil sample to test for metal particulate wear, varnish, or water contamination.",
                "Execute narrowband FFT vibration spectrum analysis to inspect bearing defect pass frequencies (BPFO, BPFI).",
                "Prepare standby pump/compressor unit and schedule hot-standby changeover if vibration exceeds 7.1 mm/s.",
            ]
        elif vib_severe and not temp_severe:
            failure_mode = "MECHANICAL_UNBALANCE_OR_LOOSENESS"
            description = (
                "Vibration spike without elevated bearing temperature typically points to dynamic rotor unbalance, "
                "coupling misalignment, impeller vane fouling, or foundation bolt loosening."
            )
            sops = [
                "Inspect foundation hold-down bolts and structural skid mounts for mechanical looseness.",
                "Check flexible drive coupling alignment, spider condition, and shaft runout.",
                "Review vibration phase analysis (1X vs 2X harmonic components) to distinguish unbalance from misalignment.",
                "Verify steady process flow to ensure acoustic resonance or line pulsation is not driving skid vibration.",
            ]
        elif dp_dropped and curr_surged:
            failure_mode = "PUMP_CAVITATION_OR_VAPOR_LOCK"
            description = (
                "Discharge pressure drop coupled with motor current surge and elevated suction disturbance is "
                "the classical signature of impeller cavitation, vapor locking, or high gas-volume fraction (GVF)."
            )
            sops = [
                "Inspect suction line strainer for differential pressure clog or debris build-up.",
                "Verify Net Positive Suction Head Available (NPSHa) exceeds pump manufacturer NPSHr.",
                "Check well fluid Gas-Oil Ratio (GOR) and casing-head backpressure for free gas breakout.",
                "Throttle discharge control valve slightly to increase backpressure and suppress cavitation bubble collapse.",
            ]
        elif dp_dropped and curr_dropped:
            failure_mode = "FLUID_STARVATION_OR_DRIVE_DECOUPLING"
            description = (
                "Concomitant loss of discharge pressure and electrical underload indicates fluid supply starvation, "
                "closed suction block valve, or potential shaft shearing/decoupling."
            )
            sops = [
                "Verify upstream separator liquid level and suction ESD valve open position.",
                "Inspect pump drive coupling and motor shaft rotation to confirm mechanical torque transmission.",
                "Examine flowline check valve for reverse flow or stuck flapper.",
                "Verify wellhead casing pressure and tubing head pressure to rule out wellbore fluid depletion.",
            ]
        elif temp_severe:
            failure_mode = "THERMAL_OVERHEAT_LUBRICATION_DEGRADATION"
            description = (
                "Isolated bearing temperature runaway without vibration indicates lube oil starvation, cooling water "
                "loop restriction, or abnormal axial thrust load."
            )
            sops = [
                "Inspect lube oil heat exchanger / cooling water circuit flow rate and delta-T.",
                "Check oil filter differential pressure indicator for bypass or fouling.",
                "Verify thrust bearing balance line pressure and axial shaft clearance.",
                "Ensure oil viscosity matches equipment ambient operating temperature specification.",
            ]
        elif curr_surged and not dp_dropped:
            failure_mode = "ELECTRICAL_OVERLOAD_OR_MOTOR_STALL"
            description = (
                "Elevated motor current under steady hydraulic load indicates motor winding insulation breakdown, "
                "supply voltage imbalance, or internal mechanical drag."
            )
            sops = [
                "Perform 3-phase current balance and supply bus voltage check at motor control center (MCC).",
                "Measure stator winding temperature sensors (RTDs) across all three phases.",
                "Inspect Variable Frequency Drive (VFD) output harmonic distortion and frequency setpoint.",
                "Schedule offline insulation resistance (Megger) and surge test at earliest planned downtime.",
            ]
        else:
            failure_mode = "PROCESS_UPSET_OR_TRANSIENT_INSTABILITY"
            description = (
                "Multivariate parameter perturbation consistent with upstream slugging, choke manifold adjustment, "
                "or transient process cycling."
            )
            sops = [
                "Correlate event timestamp with upstream well choke adjustments and pigging operations.",
                "Review automated control valve actuator hunting or pressure transmitter calibration.",
                "Monitor trending over subsequent 2 operational cycles to confirm return to steady-state envelope.",
            ]

        # Urgency triage
        if safety_section.highest_safety_tier in ["ZONE_D_TRIP", "CRITICAL_TRIP", "OVERLOAD_TRIP"] or triage.severity == "CRITICAL":
            urgency = "IMMEDIATE_SHUTDOWN"
            action_summary = "CRITICAL SAFETY INTERVENTION: Initiate controlled emergency shutdown or immediate switch to standby train."
        elif safety_section.highest_safety_tier in ["ZONE_C_ALERT", "WARNING_ALERT", "OVERLOAD_ALERT"] or triage.severity == "HIGH":
            urgency = "URGENT_INSPECTION_24H"
            action_summary = "PRIORITY FIELD DISPATCH: Dispatch mechanical/production specialist within 24h to inspect equipment and execute SOP checklist."
        elif triage.severity == "MEDIUM":
            urgency = "MONITOR_NEXT_SHIFT"
            action_summary = "SURVEILLANCE WATCH: Handover anomalous signature to incoming shift supervisor with heightened logging frequency."
        else:
            urgency = "ROUTINE_SURVEILLANCE"
            action_summary = "ROUTINE MONITORING: Continue standard supervisory control and data acquisition (SCADA) monitoring."

        return OperationalRemediationSection(
            diagnosed_failure_mode=failure_mode,
            failure_mode_description=description,
            urgency_level=urgency,
            recommended_sop_steps=sops,
            operator_action_summary=action_summary,
            preventive_maintenance_id=f"PM-{abs(hash(failure_mode)) % 10000:04d}",
        )


# ===========================================================================
# Ensemble Anomaly Explainer Service
# ===========================================================================

class EnsembleAnomalyExplainerService:
    """
    High-level operational service that orchestrates all 4 anomaly detection models
    (Isolation Forest, Rolling Z-Score, Tukey IQR, Hampel MAD), runs multi-model consensus,
    and formats complete factual explanations for operational intelligence.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        default_equipment_id: str = "OFFSHORE_ESP_01",
    ):
        self.default_equipment_id = default_equipment_id
        self.formatter = AnomalyExplanationFormatter(default_equipment_id=default_equipment_id)
        self.iso_model = IsolationForestAnomalyDetector(contamination=contamination, scaler_type="robust")
        self.zscore_model = RollingZScoreAnomalyDetector(threshold=3.0)
        self.iqr_model = IQRAnomalyDetector(k=1.5)
        self.mad_model = MADAnomalyDetector(threshold=3.5)
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.baseline_medians: Dict[str, float] = {}
        self.baseline_iqrs: Dict[str, float] = {}

    def fit(self, telemetry_df: pd.DataFrame) -> "EnsembleAnomalyExplainerService":
        """Fits all underlying models on nominal historical telemetry."""
        self.iso_model.fit(telemetry_df)
        self.zscore_model.fit(telemetry_df)
        self.iqr_model.fit(telemetry_df)
        self.mad_model.fit(telemetry_df)

        self.feature_names = self.iso_model.feature_names
        self.baseline_medians = self.iso_model.baseline_medians
        self.baseline_iqrs = self.iso_model.baseline_iqrs
        self.is_fitted = True
        logger.info(f"Fitted EnsembleAnomalyExplainerService with {len(self.feature_names)} features.")
        return self

    def explain_dataframe_row(
        self,
        row: Union[pd.Series, Dict[str, float]],
        timestamp: Optional[str] = None,
        index: int = 0,
        equipment_id: Optional[str] = None,
    ) -> FactualAnomalyExplanation:
        """
        Runs all 4 models on a single telemetry point, evaluates ensemble votes,
        and returns a complete FactualAnomalyExplanation.
        """
        if not self.is_fitted:
            raise ValueError("Service must be fitted on baseline telemetry before explaining rows.")

        if isinstance(row, pd.Series):
            features_dict = {f: float(row[f]) for f in self.feature_names if f in row}
        else:
            features_dict = {f: float(row[f]) for f in self.feature_names if f in row}

        single_df = pd.DataFrame([features_dict])

        # Model predictions
        iso_pred = bool(self.iso_model.predict(single_df)[0] == 1)
        zscore_pred = bool(self.zscore_model.predict(single_df)[0] == 1)
        iqr_pred = bool(self.iqr_model.predict(single_df)[0] == 1)
        mad_pred = bool(self.mad_model.predict(single_df)[0] == 1)

        ensemble_votes = {
            "IsolationForest": iso_pred,
            "RollingZScore": zscore_pred,
            "TukeyIQR": iqr_pred,
            "HampelMAD": mad_pred,
        }

        # Calibrated Isolation Forest anomaly score
        iso_score = float(self.iso_model.score_samples(single_df)[0])
        is_anomaly = iso_pred or (sum(ensemble_votes.values()) >= 2)

        return self.formatter.explain_point(
            features_dict=features_dict,
            baseline_medians=self.baseline_medians,
            baseline_iqrs=self.baseline_iqrs,
            anomaly_score=iso_score,
            is_anomaly=is_anomaly,
            timestamp=timestamp,
            index=index,
            ensemble_votes=ensemble_votes,
            equipment_id=equipment_id or self.default_equipment_id,
        )
