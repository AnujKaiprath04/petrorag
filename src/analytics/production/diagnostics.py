"""
PetroRAG Production Analytics - Chan Water Coning & Breakthrough Diagnostics (Module 3.7)
Implements K.S. Chan (1995) diagnostic plots using Water-Oil Ratio (WOR) and WOR derivative
curves to diagnose the mechanism of excess water production (coning vs channeling vs normal depletion).
"""

from typing import Dict, List, Any, Optional, Tuple, Literal
import numpy as np
import pandas as pd
from scipy import stats
from pydantic import BaseModel, Field


class WORPoint(BaseModel):
    time_days: float
    oil_rate: float
    water_rate: float
    wor: float
    wor_derivative: float


class ChanDiagnosticResult(BaseModel):
    well_id: str
    record_count: int
    mean_wor: float
    current_wor: float
    wor_slope_log_log: float
    wor_deriv_slope_log_log: float
    water_breakthrough_mechanism: Literal[
        "BOTTOM_WATER_CONING",
        "CHANNELING_OR_FRACTURE",
        "NORMAL_DEPLETION",
        "STABLE_LOW_WATER",
        "INSUFFICIENT_DATA",
    ]
    diagnostic_confidence: float = Field(..., ge=0.0, le=1.0)
    diagnostic_narrative: str
    recommended_remedial_action: str
    wor_curve: List[WORPoint]


class ChanWaterDiagnosticEngine:
    """
    Diagnoses excessive water production mechanisms using log-log plots of
    Water-Oil Ratio (WOR) and its time derivative d(WOR)/dt.
    """

    def analyze_well(
        self,
        df: pd.DataFrame,
        well_id: Optional[str] = None,
        oil_col: str = "oil_rate",
        water_col: str = "water_rate",
        time_col: str = "timestamp",
    ) -> ChanDiagnosticResult:
        """
        Executes Chan diagnostic evaluation on well production history.
        """
        w_id = well_id or (str(df["well_id"].iloc[0]) if "well_id" in df.columns and len(df) > 0 else "WELL-01")

        if df.empty or len(df) < 5:
            return ChanDiagnosticResult(
                well_id=w_id,
                record_count=len(df),
                mean_wor=0.0,
                current_wor=0.0,
                wor_slope_log_log=0.0,
                wor_deriv_slope_log_log=0.0,
                water_breakthrough_mechanism="INSUFFICIENT_DATA",
                diagnostic_confidence=0.0,
                diagnostic_narrative="Insufficient time-series data points (<5 records) to perform Chan WOR derivative diagnostics.",
                recommended_remedial_action="Collect additional production logs before initiating water shutoff diagnostics.",
                wor_curve=[],
            )

        working_df = df.copy()
        if time_col in working_df.columns:
            working_df[time_col] = pd.to_datetime(working_df[time_col])
            working_df = working_df.sort_values(by=time_col).reset_index(drop=True)
            t_seconds = (working_df[time_col] - working_df[time_col].iloc[0]).dt.total_seconds()
            t_days = np.maximum(1.0, t_seconds.to_numpy(dtype=float) / 86400.0)
        else:
            t_days = np.arange(1.0, len(working_df) + 1.0, dtype=float)

        oil_rates = np.nan_to_num(working_df[oil_col].to_numpy(dtype=float), nan=0.0)
        water_rates = np.nan_to_num(working_df[water_col].to_numpy(dtype=float), nan=0.0)

        # Calculate WOR = Qw / Qo (avoid division by zero)
        wor_vals = np.where(oil_rates > 1e-3, water_rates / np.maximum(1e-3, oil_rates), 0.0)

        # Compute numerical derivative d(WOR)/dt using central differences
        dt = np.gradient(t_days)
        dt = np.where(dt <= 0, 1.0, dt)
        dwor = np.gradient(wor_vals)
        wor_deriv = np.maximum(1e-6, np.abs(dwor / dt))

        wor_points: List[WORPoint] = []
        for i in range(len(t_days)):
            wor_points.append(
                WORPoint(
                    time_days=round(float(t_days[i]), 1),
                    oil_rate=round(float(oil_rates[i]), 2),
                    water_rate=round(float(water_rates[i]), 2),
                    wor=round(float(wor_vals[i]), 4),
                    wor_derivative=round(float(wor_deriv[i]), 6),
                )
            )

        mean_wor = float(np.mean(wor_vals))
        current_wor = float(wor_vals[-1])

        # Evaluate slopes on log-log scale
        # log(WOR) vs log(t) and log(WOR') vs log(t)
        valid_log = (t_days > 0) & (wor_vals > 1e-4) & (wor_deriv > 1e-6)
        if np.sum(valid_log) >= 5:
            log_t = np.log10(t_days[valid_log])
            log_wor = np.log10(wor_vals[valid_log])
            log_deriv = np.log10(wor_deriv[valid_log])

            slope_wor = float(stats.linregress(log_t, log_wor).slope)

            # Look at derivative trajectory: early vs late slope
            n_pts = len(log_t)
            mid = n_pts // 2
            slope_early_deriv = float(stats.linregress(log_t[:mid], log_deriv[:mid]).slope) if mid >= 3 else 0.0
            slope_late_deriv = float(stats.linregress(log_t[mid:], log_deriv[mid:]).slope) if (n_pts - mid) >= 3 else 0.0
            overall_deriv_slope = float(stats.linregress(log_t, log_deriv).slope)
        else:
            slope_wor = 0.0
            slope_early_deriv = 0.0
            slope_late_deriv = 0.0
            overall_deriv_slope = 0.0

        # Classification based on Chan (1995) criteria
        if current_wor < 0.1 and mean_wor < 0.1:
            mechanism = "STABLE_LOW_WATER"
            confidence = 0.90
            narrative = f"Well {w_id} exhibits very low water production (current WOR={current_wor:.3f}). No water control intervention required."
            action = "Continue standard surveillance. Monitor water cut during routine production testing."

        elif slope_wor > 0.8 and overall_deriv_slope > 0.5:
            # Steep slope in both WOR and derivative -> Channeling or Fault
            mechanism = "CHANNELING_OR_FRACTURE"
            confidence = 0.88
            narrative = (
                f"Severe near-wellbore channeling, fracture breakthrough, or casing leak detected on {w_id}. "
                f"Log-log WOR slope is steep ({slope_wor:.2f}) with upward-trending derivative ({overall_deriv_slope:.2f})."
            )
            action = (
                "Run production logging tool (PLT) or temperature log to identify breakthrough intervals. "
                "Evaluate mechanical bridge plug, straddle packers, or polymer squeeze treatment for zonal isolation."
            )

        elif slope_early_deriv > 0.3 and (slope_late_deriv <= 0.1 or overall_deriv_slope < 0.3):
            # Derivative climbs then flattens/drops -> Bottom Water Coning
            mechanism = "BOTTOM_WATER_CONING"
            confidence = 0.85
            narrative = (
                f"Bottom water coning signature identified on well {w_id}. "
                f"WOR derivative initially surged and is now leveling off or plateauing."
            )
            action = (
                "Reduce choke size to decrease wellbore drawdown below critical coning rate (Q_crit). "
                "Consider horizontal recompletion, downhole water sink (DWS), or gelled water shutoff polymer barrier."
            )

        else:
            # Gradual multilayer depletion
            mechanism = "NORMAL_DEPLETION"
            confidence = 0.82
            narrative = (
                f"Normal multilayer reservoir water breakthrough and gradual displacement observed on {w_id}. "
                f"WOR exhibits steady moderate progression ({slope_wor:.2f} log-log slope)."
            )
            action = (
                "Optimize artificial lift (gas lift or ESP) to handle liquid loading. "
                "Ensure surface separation and water treatment facilities have adequate capacity."
            )

        return ChanDiagnosticResult(
            well_id=w_id,
            record_count=len(df),
            mean_wor=round(mean_wor, 4),
            current_wor=round(current_wor, 4),
            wor_slope_log_log=round(slope_wor, 3),
            wor_deriv_slope_log_log=round(overall_deriv_slope, 3),
            water_breakthrough_mechanism=mechanism,
            diagnostic_confidence=confidence,
            diagnostic_narrative=narrative,
            recommended_remedial_action=action,
            wor_curve=wor_points,
        )
