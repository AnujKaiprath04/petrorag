"""
PetroRAG Equipment Health Engine (Module 3.14)
Provides multi-factor asset health index calculation (0-100), Weibull age wear modeling,
telemetry stress evaluation, historical anomaly frequency penalties, maintenance compliance tracking,
and Remaining Useful Life (RUL) estimation with confidence bounds.
"""

from typing import Dict, List, Any, Optional, Tuple, Literal
from datetime import datetime, timezone
import math
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.core.logging import logger
from src.analytics.anomaly.attribution import (
    ISO_10816_VIBRATION_ALERT,
    ISO_10816_VIBRATION_DANGER,
    BEARING_TEMP_ALERT_C,
    BEARING_TEMP_DANGER_C,
    MOTOR_CURRENT_ALERT_FACTOR,
    MOTOR_CURRENT_DANGER_FACTOR,
)


# ===========================================================================
# Equipment Profile Specifications
# ===========================================================================

class EquipmentProfile(BaseModel):
    """OEM design specifications and operating limits for equipment classes."""
    equipment_type: str
    design_life_hours: float
    pm_interval_days: int
    weibull_shape_beta: float = 2.0
    vibration_warning_mms: float = ISO_10816_VIBRATION_ALERT
    vibration_trip_mms: float = ISO_10816_VIBRATION_DANGER
    temp_warning_c: float = BEARING_TEMP_ALERT_C
    temp_trip_c: float = BEARING_TEMP_DANGER_C
    rated_current_amps: Optional[float] = None


DEFAULT_PROFILES: Dict[str, EquipmentProfile] = {
    "ESP": EquipmentProfile(
        equipment_type="ESP",
        design_life_hours=25000.0,
        pm_interval_days=180,
        weibull_shape_beta=1.8,
        vibration_warning_mms=4.5,
        vibration_trip_mms=7.1,
        temp_warning_c=85.0,
        temp_trip_c=95.0,
        rated_current_amps=45.0,
    ),
    "CENTRIFUGAL_PUMP": EquipmentProfile(
        equipment_type="CENTRIFUGAL_PUMP",
        design_life_hours=40000.0,
        pm_interval_days=120,
        weibull_shape_beta=2.1,
        vibration_warning_mms=4.5,
        vibration_trip_mms=7.1,
        temp_warning_c=85.0,
        temp_trip_c=95.0,
        rated_current_amps=50.0,
    ),
    "GAS_COMPRESSOR": EquipmentProfile(
        equipment_type="GAS_COMPRESSOR",
        design_life_hours=50000.0,
        pm_interval_days=90,
        weibull_shape_beta=2.4,
        vibration_warning_mms=4.5,
        vibration_trip_mms=7.1,
        temp_warning_c=85.0,
        temp_trip_c=95.0,
        rated_current_amps=120.0,
    ),
    "RECIPROCATING_PUMP": EquipmentProfile(
        equipment_type="RECIPROCATING_PUMP",
        design_life_hours=30000.0,
        pm_interval_days=90,
        weibull_shape_beta=2.0,
        vibration_warning_mms=5.0,
        vibration_trip_mms=8.0,
        temp_warning_c=85.0,
        temp_trip_c=95.0,
        rated_current_amps=40.0,
    ),
    "DEFAULT": EquipmentProfile(
        equipment_type="DEFAULT",
        design_life_hours=35000.0,
        pm_interval_days=180,
        weibull_shape_beta=2.0,
        vibration_warning_mms=4.5,
        vibration_trip_mms=7.1,
        temp_warning_c=85.0,
        temp_trip_c=95.0,
        rated_current_amps=50.0,
    ),
}


# ===========================================================================
# Structured Data Schemas
# ===========================================================================

class SubIndexBreakdown(BaseModel):
    """Decomposition of the composite 0-100 health index."""
    telemetry_stress_score: float = Field(..., ge=0.0, le=100.0, description="Active sensor compliance score (35% weight)")
    anomaly_history_score: float = Field(..., ge=0.0, le=100.0, description="Historical 30-day incident penalty score (25% weight)")
    runtime_aging_score: float = Field(..., ge=0.0, le=100.0, description="Weibull cumulative operating hours wear score (20% weight)")
    maintenance_compliance_score: float = Field(..., ge=0.0, le=100.0, description="PM interval compliance score (20% weight)")


class RULEstimate(BaseModel):
    """Remaining Useful Life projection with uncertainty intervals."""
    rul_days_to_critical: float = Field(..., description="Estimated days until health index reaches CRITICAL (<50)")
    rul_days_to_degraded: float = Field(..., description="Estimated days until health index reaches DEGRADED (<70)")
    confidence_interval_90_days: Tuple[float, float]
    estimated_wear_rate_per_month: float = Field(..., description="Projected EHI decline points per 30 operational days")
    projected_failure_mode: str


class ComprehensiveEquipmentHealthReport(BaseModel):
    """Complete, auditable equipment health assessment report."""
    equipment_id: str
    equipment_type: str
    assessment_timestamp: str
    composite_health_index: float = Field(..., ge=0.0, le=100.0)
    health_status: Literal["HEALTHY", "OBSERVATION", "DEGRADED", "CRITICAL"]
    sub_indices: SubIndexBreakdown
    rul_forecast: RULEstimate
    primary_stress_factors: List[str]
    maintenance_status: Literal["CURRENT", "DUE_SOON", "OVERDUE"]
    days_since_last_pm: int
    cumulative_run_hours: float
    recommended_inspection_interval_days: int
    recommended_interventions: List[str]

    def to_rag_context(self) -> str:
        """Grounding context block for LLM prompt injection."""
        stresses = "\n".join(f"- {s}" for s in self.primary_stress_factors)
        actions = "\n".join(f"- {a}" for a in self.recommended_interventions)
        return f"""<EQUIPMENT_HEALTH_GROUNDING>
Asset: {self.equipment_id} ({self.equipment_type})
Timestamp: {self.assessment_timestamp}
Composite Health Index: {self.composite_health_index:.1f}/100 [{self.health_status}]
Remaining Useful Life (RUL): {self.rul_forecast.rul_days_to_critical:.0f} days to critical failure

[SUB-INDICES BREAKDOWN]
Telemetry Stress Score: {self.sub_indices.telemetry_stress_score:.1f}/100
Anomaly History Score: {self.sub_indices.anomaly_history_score:.1f}/100
Runtime Wear Score: {self.sub_indices.runtime_aging_score:.1f}/100
Maintenance Compliance: {self.sub_indices.maintenance_compliance_score:.1f}/100

[PRIMARY STRESS FACTORS]
{stresses}

[RUL & WEAR PROJECTION]
Wear Rate: {self.rul_forecast.estimated_wear_rate_per_month:.2f} EHI points/month
Projected Failure Mode: {self.rul_forecast.projected_failure_mode}
RUL 90% Confidence Interval: [{self.rul_forecast.confidence_interval_90_days[0]:.0f}, {self.rul_forecast.confidence_interval_90_days[1]:.0f}] days

[RECOMMENDED OPERATIONAL ACTIONS]
{actions}
</EQUIPMENT_HEALTH_GROUNDING>"""

    def to_markdown(self) -> str:
        """Generates a Markdown dashboard card."""
        status_colors = {
            "HEALTHY": "🟢",
            "OBSERVATION": "🟡",
            "DEGRADED": "🟠",
            "CRITICAL": "🔴",
        }
        icon = status_colors.get(self.health_status, "⚪")
        return f"""## Asset Health Assessment: `{self.equipment_id}` {icon}

- **Equipment Class:** `{self.equipment_type}` | **Assessed:** `{self.assessment_timestamp}`
- **Composite Health Index (EHI):** `{self.composite_health_index:.1f} / 100` (`{self.health_status}`)
- **Remaining Useful Life (RUL):** `{self.rul_forecast.rul_days_to_critical:.0f} days` (90% CI: `{self.rul_forecast.confidence_interval_90_days[0]:.0f} - {self.rul_forecast.confidence_interval_90_days[1]:.0f}` days)
- **Maintenance Compliance:** `{self.maintenance_status}` (`{self.days_since_last_pm}` days since PM)

### Sub-Index Breakdown
| Metric Dimension | Score | Weight | Status |
| :--- | :--- | :--- | :--- |
| Telemetry Physical Envelope | `{self.sub_indices.telemetry_stress_score:.1f}/100` | 35% | {"PASS" if self.sub_indices.telemetry_stress_score >= 70 else "ALERT"} |
| Historical Anomaly Density | `{self.sub_indices.anomaly_history_score:.1f}/100` | 25% | {"PASS" if self.sub_indices.anomaly_history_score >= 70 else "ALERT"} |
| Runtime Hours & Aging | `{self.sub_indices.runtime_aging_score:.1f}/100` | 20% | {"PASS" if self.sub_indices.runtime_aging_score >= 70 else "ALERT"} |
| Maintenance PM Timeliness | `{self.sub_indices.maintenance_compliance_score:.1f}/100` | 20% | {"PASS" if self.sub_indices.maintenance_compliance_score >= 70 else "ALERT"} |

### Primary Stress Drivers
{chr(10).join(f"- {s}" for s in self.primary_stress_factors)}

### Prescribed Interventions
{chr(10).join(f"1. {a}" for a in self.recommended_interventions)}
"""


class FleetHealthSummary(BaseModel):
    """Aggregate fleet-wide health statistics and critical asset prioritization."""
    fleet_size: int
    mean_fleet_health: float
    health_tier_distribution: Dict[str, int]
    critical_assets_count: int
    overdue_pm_count: int
    priority_dispatch_queue: List[str]
    assessment_timestamp: str


# ===========================================================================
# Core Equipment Health Engine
# ===========================================================================

class EquipmentHealthEngine:
    """
    Computes rigorous multi-factor Equipment Health Indices (EHI 0-100),
    Weibull aging wear curves, and Remaining Useful Life (RUL) projections.
    """

    # Sub-index weights summing to 1.0
    WEIGHT_TELEMETRY = 0.35
    WEIGHT_ANOMALY = 0.25
    WEIGHT_RUNTIME = 0.20
    WEIGHT_MAINTENANCE = 0.20

    def __init__(self, profiles: Optional[Dict[str, EquipmentProfile]] = None):
        self.profiles = profiles or DEFAULT_PROFILES

    def get_profile(self, equipment_type: str) -> EquipmentProfile:
        """Retrieves equipment OEM profile or returns default."""
        eq_key = equipment_type.upper()
        return self.profiles.get(eq_key, self.profiles["DEFAULT"])

    def evaluate_health(
        self,
        equipment_id: str,
        equipment_type: str = "ESP",
        current_telemetry: Optional[Dict[str, float]] = None,
        historical_anomalies_30d: int = 0,
        anomaly_severity_counts: Optional[Dict[str, int]] = None,
        cumulative_run_hours: float = 5000.0,
        days_since_last_pm: int = 45,
        historical_ehi_trajectory: Optional[List[float]] = None,
    ) -> ComprehensiveEquipmentHealthReport:
        """
        Executes comprehensive multi-factor equipment health assessment.
        """
        profile = self.get_profile(equipment_type)
        telemetry = current_telemetry or {}
        stress_factors: List[str] = []

        # -------------------------------------------------------------------
        # 1. Telemetry Stress Sub-Index (35% weight)
        # -------------------------------------------------------------------
        tel_score, tel_stresses = self._compute_telemetry_sub_index(telemetry, profile)
        stress_factors.extend(tel_stresses)

        # -------------------------------------------------------------------
        # 2. Historical Anomaly Density Sub-Index (25% weight)
        # -------------------------------------------------------------------
        anom_score, anom_stresses = self._compute_anomaly_sub_index(
            historical_anomalies_30d, anomaly_severity_counts
        )
        stress_factors.extend(anom_stresses)

        # -------------------------------------------------------------------
        # 3. Runtime Hours & Weibull Aging Sub-Index (20% weight)
        # -------------------------------------------------------------------
        run_score, run_stresses = self._compute_runtime_sub_index(
            cumulative_run_hours, profile
        )
        stress_factors.extend(run_stresses)

        # -------------------------------------------------------------------
        # 4. Maintenance Compliance Sub-Index (20% weight)
        # -------------------------------------------------------------------
        mnt_score, mnt_stresses, mnt_status = self._compute_maintenance_sub_index(
            days_since_last_pm, profile
        )
        stress_factors.extend(mnt_stresses)

        # -------------------------------------------------------------------
        # Composite Health Index Calculation
        # -------------------------------------------------------------------
        composite_ehi = (
            self.WEIGHT_TELEMETRY * tel_score
            + self.WEIGHT_ANOMALY * anom_score
            + self.WEIGHT_RUNTIME * run_score
            + self.WEIGHT_MAINTENANCE * mnt_score
        )
        composite_ehi = max(0.0, min(100.0, round(composite_ehi, 1)))

        # Status classification
        if composite_ehi >= 85.0:
            status = "HEALTHY"
            insp_interval = 90
        elif composite_ehi >= 70.0:
            status = "OBSERVATION"
            insp_interval = 30
        elif composite_ehi >= 50.0:
            status = "DEGRADED"
            insp_interval = 7
        else:
            status = "CRITICAL"
            insp_interval = 1

        if not stress_factors:
            stress_factors.append("All mechanical and operational parameters within certified OEM bounds.")

        # -------------------------------------------------------------------
        # 5. Remaining Useful Life (RUL) & Wear Trajectory
        # -------------------------------------------------------------------
        rul = self._compute_rul(
            composite_ehi=composite_ehi,
            status=status,
            cumulative_run_hours=cumulative_run_hours,
            profile=profile,
            stress_factors=stress_factors,
            historical_ehi_trajectory=historical_ehi_trajectory,
        )

        # -------------------------------------------------------------------
        # 6. Prescribed Maintenance Actions
        # -------------------------------------------------------------------
        interventions = self._prescribe_interventions(status, mnt_status, stress_factors)

        ts = datetime.now(timezone.utc).isoformat()

        return ComprehensiveEquipmentHealthReport(
            equipment_id=equipment_id,
            equipment_type=profile.equipment_type,
            assessment_timestamp=ts,
            composite_health_index=composite_ehi,
            health_status=status,
            sub_indices=SubIndexBreakdown(
                telemetry_stress_score=round(tel_score, 1),
                anomaly_history_score=round(anom_score, 1),
                runtime_aging_score=round(run_score, 1),
                maintenance_compliance_score=round(mnt_score, 1),
            ),
            rul_forecast=rul,
            primary_stress_factors=stress_factors,
            maintenance_status=mnt_status,
            days_since_last_pm=days_since_last_pm,
            cumulative_run_hours=round(cumulative_run_hours, 1),
            recommended_inspection_interval_days=insp_interval,
            recommended_interventions=interventions,
        )

    # -----------------------------------------------------------------------
    # Sub-Index Calculators
    # -----------------------------------------------------------------------

    def _compute_telemetry_sub_index(
        self,
        telemetry: Dict[str, float],
        profile: EquipmentProfile,
    ) -> Tuple[float, List[str]]:
        penalties = 0.0
        stresses: List[str] = []

        # Vibration analysis
        vib = telemetry.get("vibration_rms") or telemetry.get("vibration_rms_mms")
        if vib is not None:
            if vib >= profile.vibration_trip_mms:
                penalties += 80.0
                stresses.append(f"Severe vibration ({vib:.2f} mm/s) breaching ISO trip limit ({profile.vibration_trip_mms} mm/s)")
            elif vib >= profile.vibration_warning_mms:
                penalties += 40.0
                stresses.append(f"Elevated vibration ({vib:.2f} mm/s) in ISO Alert Zone C")
            elif vib > 2.3:
                # Moderate long-term baseline drift
                penalties += 10.0

        # Temperature analysis
        temp = telemetry.get("bearing_temp") or telemetry.get("temperature_c")
        if temp is not None:
            if temp >= profile.temp_trip_c:
                penalties += 85.0
                stresses.append(f"Critical bearing temp ({temp:.1f}°C) breaching API trip boundary ({profile.temp_trip_c}°C)")
            elif temp >= profile.temp_warning_c:
                penalties += 45.0
                stresses.append(f"Bearing temperature warning ({temp:.1f}°C) exceeding API alert limit ({profile.temp_warning_c}°C)")
            elif temp > 75.0:
                penalties += 12.0

        # Motor current overload
        curr = telemetry.get("motor_current") or telemetry.get("motor_current_a")
        if curr is not None and profile.rated_current_amps:
            ratio = curr / profile.rated_current_amps
            if ratio >= MOTOR_CURRENT_DANGER_FACTOR:
                penalties += 60.0
                stresses.append(f"Motor current overload ({curr:.1f}A, {ratio*100:.0f}% rated)")
            elif ratio >= MOTOR_CURRENT_ALERT_FACTOR:
                penalties += 25.0
                stresses.append(f"Motor operating near rated capacity ({curr:.1f}A, {ratio*100:.0f}% rated)")

        # Hydraulic discharge drop
        dp = telemetry.get("discharge_pressure") or telemetry.get("discharge_pressure_bar")
        if dp is not None and dp < 8.0:
            penalties += 25.0
            stresses.append(f"Abnormal discharge pressure loss ({dp:.1f} bar)")

        score = max(0.0, 100.0 - penalties)
        return score, stresses

    def _compute_anomaly_sub_index(
        self,
        historical_anomalies_30d: int,
        severity_counts: Optional[Dict[str, int]],
    ) -> Tuple[float, List[str]]:
        stresses: List[str] = []

        if severity_counts:
            n_crit = severity_counts.get("CRITICAL", 0)
            n_high = severity_counts.get("HIGH", 0)
            n_med = severity_counts.get("MEDIUM", 0)
            n_low = severity_counts.get("LOW", 0)
            penalty = (n_crit * 30.0) + (n_high * 15.0) + (n_med * 6.0) + (n_low * 2.0)
            if n_crit > 0 or n_high > 0:
                stresses.append(f"High anomaly recurrence: {n_crit} critical and {n_high} high alerts in 30 days")
        else:
            # Fallback based on raw count
            penalty = float(historical_anomalies_30d) * 12.0
            if historical_anomalies_30d > 0:
                stresses.append(f"Recurring operational anomalies ({historical_anomalies_30d} incidents in past 30 days)")

        score = max(0.0, 100.0 - penalty)
        return score, stresses

    def _compute_runtime_sub_index(
        self,
        cumulative_run_hours: float,
        profile: EquipmentProfile,
    ) -> Tuple[float, List[str]]:
        stresses: List[str] = []
        design_life = max(1000.0, profile.design_life_hours)
        age_ratio = cumulative_run_hours / design_life

        # Weibull cumulative wear model: (t / eta)^beta
        beta = profile.weibull_shape_beta
        wear_fraction = math.pow(min(1.5, age_ratio), beta)
        penalty = min(100.0, wear_fraction * 75.0)

        if age_ratio >= 1.0:
            stresses.append(
                f"Cumulative operating runtime ({cumulative_run_hours:,.0f}h) has reached or exceeded design life ({design_life:,.0f}h)"
            )
        elif age_ratio >= 0.75:
            stresses.append(
                f"Advanced operating age ({cumulative_run_hours:,.0f}h, {age_ratio*100:.0f}% of design life)"
            )

        score = max(0.0, 100.0 - penalty)
        return score, stresses

    def _compute_maintenance_sub_index(
        self,
        days_since_last_pm: int,
        profile: EquipmentProfile,
    ) -> Tuple[float, List[str], Literal["CURRENT", "DUE_SOON", "OVERDUE"]]:
        stresses: List[str] = []
        pm_interval = profile.pm_interval_days

        if days_since_last_pm <= int(pm_interval * 0.8):
            status: Literal["CURRENT", "DUE_SOON", "OVERDUE"] = "CURRENT"
            score = 100.0 - (15.0 * (days_since_last_pm / pm_interval))
        elif days_since_last_pm <= pm_interval:
            status = "DUE_SOON"
            score = 85.0 - (15.0 * (days_since_last_pm - pm_interval * 0.8) / (pm_interval * 0.2))
            stresses.append(f"Preventive maintenance due soon ({days_since_last_pm}/{pm_interval} days elapsed)")
        else:
            status = "OVERDUE"
            overdue_days = days_since_last_pm - pm_interval
            score = max(0.0, 70.0 - (overdue_days * 0.8))
            stresses.append(f"Overdue maintenance: {overdue_days} days past scheduled PM interval ({pm_interval}d)")

        score = max(0.0, min(100.0, score))
        return score, stresses, status

    # -----------------------------------------------------------------------
    # RUL & Intervention Predictors
    # -----------------------------------------------------------------------

    def _compute_rul(
        self,
        composite_ehi: float,
        status: str,
        cumulative_run_hours: float,
        profile: EquipmentProfile,
        stress_factors: List[str],
        historical_ehi_trajectory: Optional[List[float]],
    ) -> RULEstimate:
        # Determine empirical degradation rate per 30 days
        if historical_ehi_trajectory and len(historical_ehi_trajectory) >= 2:
            delta = historical_ehi_trajectory[0] - historical_ehi_trajectory[-1]
            wear_rate_per_month = max(0.5, delta)
        else:
            # Synthetic model based on current stress
            if status == "CRITICAL":
                wear_rate_per_month = 18.0
            elif status == "DEGRADED":
                wear_rate_per_month = 8.5
            elif status == "OBSERVATION":
                wear_rate_per_month = 3.5
            else:
                wear_rate_per_month = 1.2

        daily_decay = wear_rate_per_month / 30.0

        # Days to degraded (70.0)
        if composite_ehi > 70.0:
            days_to_degraded = (composite_ehi - 70.0) / daily_decay
        else:
            days_to_degraded = 0.0

        # Days to critical (50.0)
        if composite_ehi > 50.0:
            days_to_critical = (composite_ehi - 50.0) / daily_decay
        else:
            days_to_critical = 0.0

        # 90% confidence interval: [0.75 * RUL, 1.25 * RUL]
        ci_lower = max(1.0, days_to_critical * 0.75)
        ci_upper = max(2.0, days_to_critical * 1.25)

        # Hypothesized failure mode
        s_text = " ".join(stress_factors).lower()
        if "vibration" in s_text and "temp" in s_text:
            fail_mode = "Journal Bearing Wear & Lubrication Breakdown"
        elif "vibration" in s_text:
            fail_mode = "Rotor Unbalance & Mechanical Coupling Looseness"
        elif "temp" in s_text:
            fail_mode = "Thermal Overheat & Lube Oil Carbonization"
        elif "runtime" in s_text or "design life" in s_text:
            fail_mode = "Fatigue Wear & Impeller Cavitation Erosion"
        else:
            fail_mode = "Progressive Mechanical Aging"

        return RULEstimate(
            rul_days_to_critical=round(days_to_critical, 1),
            rul_days_to_degraded=round(days_to_degraded, 1),
            confidence_interval_90_days=(round(ci_lower, 1), round(ci_upper, 1)),
            estimated_wear_rate_per_month=round(wear_rate_per_month, 2),
            projected_failure_mode=fail_mode,
        )

    def _prescribe_interventions(
        self,
        status: str,
        mnt_status: str,
        stress_factors: List[str],
    ) -> List[str]:
        actions: List[str] = []

        if status == "CRITICAL":
            actions.append("IMMEDIATE: Issue emergency maintenance work order and schedule controlled equipment shutdown within 24 hours.")
            actions.append("ISOLATION: Apply Lockout/Tagout (LOTO) and verify mechanical line blinds prior to casing disassembly.")
        elif status == "DEGRADED":
            actions.append("HIGH PRIORITY: Expedite preventive maintenance overhaul within 7 calendar days.")
            actions.append("SURVEILLANCE: Increase vibration and thermographic inspection frequency to daily shift patrols.")
        elif status == "OBSERVATION":
            actions.append("PLANNED MAINTENANCE: Schedule detailed mechanical inspection and oil sampling at next routine turnaround.")
            actions.append("DATA LOGGING: Trend daily peak vibration RMS and bearing temperature in SCADA.")
        else:
            actions.append("ROUTINE: Continue standard supervisory monitoring per baseline OEM schedule.")

        if mnt_status == "OVERDUE":
            actions.append("COMPLIANCE: Execute overdue preventive maintenance procedure immediately to preserve asset integrity certification.")
        elif mnt_status == "DUE_SOON":
            actions.append("LOGISTICS: Pre-stage spare mechanical seals, bearing sets, and lube oil drums for upcoming PM.")

        return actions

    # -----------------------------------------------------------------------
    # Fleet-Level Rollup
    # -----------------------------------------------------------------------

    def evaluate_fleet(
        self,
        fleet_records: List[Dict[str, Any]],
    ) -> FleetHealthSummary:
        """
        Rolls up health assessments across an entire offshore platform or field fleet.
        """
        reports: List[ComprehensiveEquipmentHealthReport] = []

        for rec in fleet_records:
            rep = self.evaluate_health(
                equipment_id=rec["equipment_id"],
                equipment_type=rec.get("equipment_type", "ESP"),
                current_telemetry=rec.get("telemetry"),
                historical_anomalies_30d=rec.get("historical_anomalies_30d", 0),
                anomaly_severity_counts=rec.get("anomaly_severity_counts"),
                cumulative_run_hours=rec.get("cumulative_run_hours", 5000.0),
                days_since_last_pm=rec.get("days_since_last_pm", 30),
            )
            reports.append(rep)

        if not reports:
            return FleetHealthSummary(
                fleet_size=0,
                mean_fleet_health=100.0,
                health_tier_distribution={"HEALTHY": 0, "OBSERVATION": 0, "DEGRADED": 0, "CRITICAL": 0},
                critical_assets_count=0,
                overdue_pm_count=0,
                priority_dispatch_queue=[],
                assessment_timestamp=datetime.now(timezone.utc).isoformat(),
            )

        mean_h = float(np.mean([r.composite_health_index for r in reports]))
        tiers = {"HEALTHY": 0, "OBSERVATION": 0, "DEGRADED": 0, "CRITICAL": 0}
        for r in reports:
            tiers[r.health_status] = tiers.get(r.health_status, 0) + 1

        crit_count = sum(1 for r in reports if r.health_status == "CRITICAL")
        overdue_count = sum(1 for r in reports if r.maintenance_status == "OVERDUE")

        # Sort fleet ascending by health index for priority dispatch queue
        sorted_reports = sorted(reports, key=lambda x: x.composite_health_index)
        dispatch_queue = [r.equipment_id for r in sorted_reports if r.health_status in ["CRITICAL", "DEGRADED"]]

        return FleetHealthSummary(
            fleet_size=len(reports),
            mean_fleet_health=round(mean_h, 1),
            health_tier_distribution=tiers,
            critical_assets_count=crit_count,
            overdue_pm_count=overdue_count,
            priority_dispatch_queue=dispatch_queue,
            assessment_timestamp=datetime.now(timezone.utc).isoformat(),
        )
