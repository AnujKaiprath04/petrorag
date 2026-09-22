"""
PetroRAG EDA Module - Trend Analyzer (Module 3.6)
Analyzes time-series production decline curves, reservoir/wellhead pressure depletion,
water cut progression, and Gas-Oil Ratio (GOR) dynamics.
"""

from typing import Dict, List, Any, Optional, Literal
import numpy as np
import pandas as pd
from scipy import stats
from pydantic import BaseModel, Field


class ProductionDeclineFit(BaseModel):
    fluid_type: str  # "oil", "gas", "water", or custom
    initial_rate: float
    final_rate: float
    rate_change_pct: float
    daily_decline_factor_d: float
    annual_nominal_decline_pct: float
    linear_slope_per_day: float
    r_squared_exponential: float
    r_squared_linear: float
    decline_regime: Literal[
        "RAPID_DECLINE",
        "MODERATE_DECLINE",
        "LOW_DECLINE",
        "STABLE",
        "INCREASING",
        "INSUFFICIENT_DATA",
    ]


class PressureDepletionTrend(BaseModel):
    pressure_col: str
    initial_pressure: float
    final_pressure: float
    cumulative_pressure_drop: float
    slope_per_day: float
    depletion_regime: Literal[
        "RAPID_DEPLETION",
        "MODERATE_DEPLETION",
        "STABLE",
        "PRESSURE_BUILDUP",
        "INSUFFICIENT_DATA",
    ]


class WaterCutEvolution(BaseModel):
    initial_water_cut_pct: float
    final_water_cut_pct: float
    mean_water_cut_pct: float
    max_water_cut_pct: float
    slope_pct_per_day: float
    monthly_change_pct: float
    water_cut_regime: Literal[
        "RAPID_BREAKTHROUGH",
        "GRADUAL_WATERING_OUT",
        "STABLE",
        "DECREASING",
        "INSUFFICIENT_DATA",
    ]


class GOREvolution(BaseModel):
    initial_gor: float
    final_gor: float
    mean_gor: float
    slope_per_day: float
    gor_regime: Literal[
        "INCREASING_GAS_CONING",
        "STABLE",
        "DECREASING",
        "INSUFFICIENT_DATA",
    ]


class ProductionTrendReport(BaseModel):
    time_span_days: float
    record_count: int
    oil_decline: Optional[ProductionDeclineFit] = None
    gas_decline: Optional[ProductionDeclineFit] = None
    water_evolution: Optional[WaterCutEvolution] = None
    pressure_depletion: Optional[PressureDepletionTrend] = None
    gor_evolution: Optional[GOREvolution] = None
    summary_observations: List[str]


class TrendAnalyzer:
    """
    Evaluates petroleum engineering trends across time-series production records.
    Fits decline curve analysis (DCA) and tracks reservoir depletion signatures.
    """

    def analyze(
        self,
        df: pd.DataFrame,
        time_col: str = "timestamp",
        oil_col: Optional[str] = "oil_rate",
        gas_col: Optional[str] = "gas_rate",
        water_col: Optional[str] = "water_rate",
        water_cut_col: Optional[str] = "water_cut",
        pressure_col: Optional[str] = "tubing_pressure",
    ) -> ProductionTrendReport:
        """
        Executes full trend analysis over production time-series.
        """
        if df.empty:
            return ProductionTrendReport(
                time_span_days=0.0,
                record_count=0,
                summary_observations=["Dataset is empty; no trends computed."],
            )

        working_df = df.copy()

        # Ensure datetime indexing or sorting
        if time_col in working_df.columns:
            working_df[time_col] = pd.to_datetime(working_df[time_col])
            working_df = working_df.sort_values(by=time_col).reset_index(drop=True)
            t_deltas = (working_df[time_col] - working_df[time_col].iloc[0]).dt.total_seconds() / 86400.0
            time_days = t_deltas.to_numpy(dtype=float)
            time_span_days = float(time_days[-1] - time_days[0]) if len(time_days) > 1 else 0.0
        elif isinstance(working_df.index, pd.DatetimeIndex):
            t_deltas = (working_df.index - working_df.index[0]).total_seconds() / 86400.0
            time_days = t_deltas.to_numpy(dtype=float)
            time_span_days = float(time_days[-1] - time_days[0]) if len(time_days) > 1 else 0.0
        else:
            time_days = np.arange(len(working_df), dtype=float)
            time_span_days = float(len(working_df) - 1)

        observations: List[str] = []

        # 1. Oil Decline
        oil_decline = None
        if oil_col and oil_col in working_df.columns:
            oil_decline = self._fit_decline(working_df[oil_col].to_numpy(dtype=float), time_days, "oil")
            if oil_decline.decline_regime in ["RAPID_DECLINE", "MODERATE_DECLINE"]:
                observations.append(
                    f"Oil rate exhibits {oil_decline.decline_regime.replace('_', ' ').lower()} "
                    f"of {oil_decline.annual_nominal_decline_pct:.1f}%/yr (R²={oil_decline.r_squared_exponential:.2f})."
                )
            elif oil_decline.decline_regime == "INCREASING":
                observations.append(f"Oil production is increasing (+{oil_decline.rate_change_pct:.1f}% overall).")

        # 2. Gas Decline
        gas_decline = None
        if gas_col and gas_col in working_df.columns:
            gas_decline = self._fit_decline(working_df[gas_col].to_numpy(dtype=float), time_days, "gas")

        # 3. Water Cut Evolution
        water_evolution = None
        wc_series = None
        if water_cut_col and water_cut_col in working_df.columns:
            wc_series = working_df[water_cut_col].to_numpy(dtype=float)
        elif (
            oil_col
            and oil_col in working_df.columns
            and water_col
            and water_col in working_df.columns
        ):
            oil_vals = np.nan_to_num(working_df[oil_col].to_numpy(dtype=float), nan=0.0)
            wat_vals = np.nan_to_num(working_df[water_col].to_numpy(dtype=float), nan=0.0)
            total_liq = oil_vals + wat_vals
            wc_series = np.where(total_liq > 1e-6, (wat_vals / total_liq) * 100.0, 0.0)

        if wc_series is not None and len(wc_series) >= 2:
            water_evolution = self._fit_water_cut(wc_series, time_days)
            if water_evolution.water_cut_regime == "RAPID_BREAKTHROUGH":
                observations.append(
                    f"CRITICAL: Rapid water breakthrough detected: water cut rising at "
                    f"{water_evolution.monthly_change_pct:.1f}% per month (currently {water_evolution.final_water_cut_pct:.1f}%)."
                )
            elif water_evolution.water_cut_regime == "GRADUAL_WATERING_OUT":
                observations.append(
                    f"Water cut is gradually watering out (+{water_evolution.monthly_change_pct:.1f}%/mo, "
                    f"averaging {water_evolution.mean_water_cut_pct:.1f}%)."
                )

        # 4. Pressure Depletion
        pressure_depletion = None
        if pressure_col and pressure_col in working_df.columns:
            pressure_depletion = self._fit_pressure(
                working_df[pressure_col].to_numpy(dtype=float), time_days, pressure_col
            )
            if pressure_depletion.depletion_regime in ["RAPID_DEPLETION", "MODERATE_DEPLETION"]:
                observations.append(
                    f"Pressure depletion observed: {pressure_depletion.cumulative_pressure_drop:.1f} unit drop "
                    f"at {pressure_depletion.slope_per_day:.2f} units/day."
                )

        # 5. GOR Evolution
        gor_evolution = None
        if (
            oil_col
            and oil_col in working_df.columns
            and gas_col
            and gas_col in working_df.columns
        ):
            oil_v = working_df[oil_col].to_numpy(dtype=float)
            gas_v = working_df[gas_col].to_numpy(dtype=float)
            valid_mask = (oil_v > 1e-4) & (gas_v >= 0.0) & ~np.isnan(oil_v) & ~np.isnan(gas_v)
            if np.sum(valid_mask) >= 3:
                gor_vals = gas_v[valid_mask] / oil_v[valid_mask]
                gor_t = time_days[valid_mask]
                gor_evolution = self._fit_gor(gor_vals, gor_t)
                if gor_evolution.gor_regime == "INCREASING_GAS_CONING":
                    observations.append(
                        f"Rising Gas-Oil Ratio (+{gor_evolution.slope_per_day * 30:.1f}/mo) "
                        f"indicates potential gas coning or reservoir bubble point crossing."
                    )

        if not observations:
            observations.append("Production rates, pressures, and fluid cuts remain stable within nominal ranges.")

        return ProductionTrendReport(
            time_span_days=round(time_span_days, 2),
            record_count=len(working_df),
            oil_decline=oil_decline,
            gas_decline=gas_decline,
            water_evolution=water_evolution,
            pressure_depletion=pressure_depletion,
            gor_evolution=gor_evolution,
            summary_observations=observations,
        )

    def _fit_decline(
        self, rates: np.ndarray, time_days: np.ndarray, fluid_type: str
    ) -> ProductionDeclineFit:
        valid_mask = ~np.isnan(rates) & ~np.isnan(time_days)
        y = rates[valid_mask]
        t = time_days[valid_mask]

        if len(y) < 3:
            return ProductionDeclineFit(
                fluid_type=fluid_type,
                initial_rate=0.0,
                final_rate=0.0,
                rate_change_pct=0.0,
                daily_decline_factor_d=0.0,
                annual_nominal_decline_pct=0.0,
                linear_slope_per_day=0.0,
                r_squared_exponential=0.0,
                r_squared_linear=0.0,
                decline_regime="INSUFFICIENT_DATA",
            )

        q_i = float(np.mean(y[: max(1, min(5, len(y)))]))
        q_f = float(np.mean(y[-max(1, min(5, len(y))):]))
        pct_change = float(((q_f - q_i) / max(1e-6, q_i)) * 100.0)

        # Linear regression: q = a + b*t
        lin_res = stats.linregress(t, y)
        b_lin = float(lin_res.slope)
        r2_lin = float(lin_res.rvalue ** 2) if not np.isnan(lin_res.rvalue) else 0.0

        # Exponential decline: ln(q) = ln(q_i) - D*t
        pos_mask = y > 1e-4
        if np.sum(pos_mask) >= 3:
            log_y = np.log(y[pos_mask])
            log_t = t[pos_mask]
            exp_res = stats.linregress(log_t, log_y)
            D_daily = float(-exp_res.slope)
            r2_exp = float(exp_res.rvalue ** 2) if not np.isnan(exp_res.rvalue) else 0.0
        else:
            D_daily = 0.0
            r2_exp = 0.0

        if D_daily > 0:
            ann_decline = float((1.0 - np.exp(-365.25 * D_daily)) * 100.0)
        else:
            ann_decline = 0.0

        # Classification
        if ann_decline > 25.0:
            regime = "RAPID_DECLINE"
        elif ann_decline >= 10.0:
            regime = "MODERATE_DECLINE"
        elif ann_decline > 2.0:
            regime = "LOW_DECLINE"
        elif pct_change > 5.0 and b_lin > 0.01:
            regime = "INCREASING"
        else:
            regime = "STABLE"

        return ProductionDeclineFit(
            fluid_type=fluid_type,
            initial_rate=round(q_i, 2),
            final_rate=round(q_f, 2),
            rate_change_pct=round(pct_change, 2),
            daily_decline_factor_d=round(D_daily, 6),
            annual_nominal_decline_pct=round(ann_decline, 2),
            linear_slope_per_day=round(b_lin, 4),
            r_squared_exponential=round(r2_exp, 4),
            r_squared_linear=round(r2_lin, 4),
            decline_regime=regime,
        )

    def _fit_pressure(
        self, pressures: np.ndarray, time_days: np.ndarray, col_name: str
    ) -> PressureDepletionTrend:
        valid_mask = ~np.isnan(pressures) & ~np.isnan(time_days)
        y = pressures[valid_mask]
        t = time_days[valid_mask]

        if len(y) < 2:
            return PressureDepletionTrend(
                pressure_col=col_name,
                initial_pressure=0.0,
                final_pressure=0.0,
                cumulative_pressure_drop=0.0,
                slope_per_day=0.0,
                depletion_regime="INSUFFICIENT_DATA",
            )

        p_i = float(y[0])
        p_f = float(y[-1])
        drop = float(p_i - p_f)

        lin_res = stats.linregress(t, y)
        slope = float(lin_res.slope)

        if slope < -2.0:
            regime = "RAPID_DEPLETION"
        elif slope < -0.2:
            regime = "MODERATE_DEPLETION"
        elif slope > 0.5:
            regime = "PRESSURE_BUILDUP"
        else:
            regime = "STABLE"

        return PressureDepletionTrend(
            pressure_col=col_name,
            initial_pressure=round(p_i, 2),
            final_pressure=round(p_f, 2),
            cumulative_pressure_drop=round(drop, 2),
            slope_per_day=round(slope, 4),
            depletion_regime=regime,
        )

    def _fit_water_cut(
        self, wc: np.ndarray, time_days: np.ndarray
    ) -> WaterCutEvolution:
        valid_mask = ~np.isnan(wc) & ~np.isnan(time_days)
        y = wc[valid_mask]
        t = time_days[valid_mask]

        if len(y) < 2:
            return WaterCutEvolution(
                initial_water_cut_pct=0.0,
                final_water_cut_pct=0.0,
                mean_water_cut_pct=0.0,
                max_water_cut_pct=0.0,
                slope_pct_per_day=0.0,
                monthly_change_pct=0.0,
                water_cut_regime="INSUFFICIENT_DATA",
            )

        wc_i = float(y[0])
        wc_f = float(y[-1])
        wc_mean = float(np.mean(y))
        wc_max = float(np.max(y))

        lin_res = stats.linregress(t, y)
        slope = float(lin_res.slope)
        monthly_change = slope * 30.0

        if monthly_change > 10.0 or slope > 0.33:
            regime = "RAPID_BREAKTHROUGH"
        elif monthly_change >= 2.0:
            regime = "GRADUAL_WATERING_OUT"
        elif monthly_change < -2.0:
            regime = "DECREASING"
        else:
            regime = "STABLE"

        return WaterCutEvolution(
            initial_water_cut_pct=round(wc_i, 2),
            final_water_cut_pct=round(wc_f, 2),
            mean_water_cut_pct=round(wc_mean, 2),
            max_water_cut_pct=round(wc_max, 2),
            slope_pct_per_day=round(slope, 4),
            monthly_change_pct=round(monthly_change, 2),
            water_cut_regime=regime,
        )

    def _fit_gor(
        self, gor: np.ndarray, time_days: np.ndarray
    ) -> GOREvolution:
        valid_mask = ~np.isnan(gor) & ~np.isnan(time_days)
        y = gor[valid_mask]
        t = time_days[valid_mask]

        if len(y) < 2:
            return GOREvolution(
                initial_gor=0.0,
                final_gor=0.0,
                mean_gor=0.0,
                slope_per_day=0.0,
                gor_regime="INSUFFICIENT_DATA",
            )

        gor_i = float(y[0])
        gor_f = float(y[-1])
        gor_mean = float(np.mean(y))

        lin_res = stats.linregress(t, y)
        slope = float(lin_res.slope)

        # 10% relative increase over month
        rel_change_mo = (slope * 30.0) / max(1e-3, gor_mean)

        if rel_change_mo > 0.10:
            regime = "INCREASING_GAS_CONING"
        elif rel_change_mo < -0.10:
            regime = "DECREASING"
        else:
            regime = "STABLE"

        return GOREvolution(
            initial_gor=round(gor_i, 2),
            final_gor=round(gor_f, 2),
            mean_gor=round(gor_mean, 2),
            slope_per_day=round(slope, 4),
            gor_regime=regime,
        )
