"""
PetroRAG Production Analytics - Key Performance Indicators (Module 3.7)
Calculates well-level and field-level operational KPIs, operating uptime,
cumulative production, rate averages, multi-well rankings, and status triage.
"""

from typing import Dict, List, Any, Optional, Literal
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class WellKPI(BaseModel):
    well_id: str
    calendar_days: int
    operating_days: int
    uptime_pct: float
    cum_oil_bbl: float
    cum_gas_mscf: float
    cum_water_bbl: float
    cum_liquid_bbl: float
    avg_oil_rate_operating_bopd: float
    avg_oil_rate_calendar_bopd: float
    avg_gas_rate_operating_mscfd: float
    avg_water_cut_pct: float
    current_water_cut_pct: float
    avg_gor_scf_bbl: float
    current_gor_scf_bbl: float
    current_oil_rate_bopd: float
    status: Literal["PRODUCING", "SHUT_IN", "WATERED_OUT"]


class WellRankingItem(BaseModel):
    well_id: str
    rate_bopd: float
    cum_oil_bbl: float
    water_cut_pct: float


class FieldKPI(BaseModel):
    field_name: str
    total_wells: int
    active_wells: int
    shut_in_wells: int
    field_uptime_pct: float
    total_daily_oil_rate_bopd: float
    total_daily_gas_rate_mscfd: float
    total_cum_oil_bbl: float
    total_cum_gas_mscf: float
    total_cum_water_bbl: float
    field_avg_water_cut_pct: float
    top_oil_producers: List[WellRankingItem]
    top_water_producers: List[WellRankingItem]
    well_kpis: Dict[str, WellKPI]


class KPICalculator:
    """
    Computes petroleum production metrics across single well histories
    and multi-well field assets.
    """

    def __init__(self, min_operating_oil_rate: float = 1.0, watered_out_wc_threshold: float = 95.0):
        self.min_operating_oil_rate = min_operating_oil_rate
        self.watered_out_wc_threshold = watered_out_wc_threshold

    def compute_well_kpi(
        self,
        df: pd.DataFrame,
        well_id: Optional[str] = None,
        oil_col: str = "oil_rate",
        gas_col: str = "gas_rate",
        water_col: str = "water_rate",
        water_cut_col: str = "water_cut",
        time_col: str = "timestamp",
    ) -> WellKPI:
        """
        Computes performance indicators for a single well history.
        """
        if df.empty:
            w_id = well_id or "UNKNOWN_WELL"
            return WellKPI(
                well_id=w_id,
                calendar_days=0,
                operating_days=0,
                uptime_pct=0.0,
                cum_oil_bbl=0.0,
                cum_gas_mscf=0.0,
                cum_water_bbl=0.0,
                cum_liquid_bbl=0.0,
                avg_oil_rate_operating_bopd=0.0,
                avg_oil_rate_calendar_bopd=0.0,
                avg_gas_rate_operating_mscfd=0.0,
                avg_water_cut_pct=0.0,
                current_water_cut_pct=0.0,
                avg_gor_scf_bbl=0.0,
                current_gor_scf_bbl=0.0,
                current_oil_rate_bopd=0.0,
                status="SHUT_IN",
            )

        working_df = df.copy()
        if "well_id" in working_df.columns and well_id:
            working_df = working_df[working_df["well_id"] == well_id].reset_index(drop=True)
            if working_df.empty:
                return self.compute_well_kpi(pd.DataFrame(), well_id=well_id)

        w_id = well_id or str(working_df["well_id"].iloc[0] if "well_id" in working_df.columns else "WELL-01")

        if time_col in working_df.columns:
            working_df[time_col] = pd.to_datetime(working_df[time_col])
            working_df = working_df.sort_values(by=time_col).reset_index(drop=True)

        calendar_days = len(working_df)
        oil_series = working_df[oil_col].to_numpy(dtype=float) if oil_col in working_df.columns else np.zeros(calendar_days)
        gas_series = working_df[gas_col].to_numpy(dtype=float) if gas_col in working_df.columns else np.zeros(calendar_days)
        water_series = working_df[water_col].to_numpy(dtype=float) if water_col in working_df.columns else np.zeros(calendar_days)

        oil_clean = np.nan_to_num(oil_series, nan=0.0)
        gas_clean = np.nan_to_num(gas_series, nan=0.0)
        water_clean = np.nan_to_num(water_series, nan=0.0)

        operating_mask = oil_clean >= self.min_operating_oil_rate
        operating_days = int(np.sum(operating_mask))
        uptime_pct = float((operating_days / max(1, calendar_days)) * 100.0)

        cum_oil = float(np.sum(oil_clean))
        cum_gas = float(np.sum(gas_clean))
        cum_water = float(np.sum(water_clean))
        cum_liquid = cum_oil + cum_water

        avg_oil_op = float(np.mean(oil_clean[operating_mask])) if operating_days > 0 else 0.0
        avg_oil_cal = float(cum_oil / max(1, calendar_days))
        avg_gas_op = float(np.mean(gas_clean[operating_mask])) if operating_days > 0 else 0.0

        # Water cut
        if water_cut_col in working_df.columns:
            wc_vals = np.nan_to_num(working_df[water_cut_col].to_numpy(dtype=float), nan=0.0)
        else:
            total_liq = oil_clean + water_clean
            wc_vals = np.where(total_liq > 1e-4, (water_clean / total_liq) * 100.0, 0.0)

        avg_wc = float(np.mean(wc_vals[operating_mask])) if operating_days > 0 else float(np.mean(wc_vals))
        current_wc = float(wc_vals[-1]) if len(wc_vals) > 0 else 0.0
        current_oil = float(oil_clean[-1]) if len(oil_clean) > 0 else 0.0

        # GOR
        pos_oil = oil_clean > 1e-3
        if np.sum(pos_oil) > 0:
            gor_series = gas_clean[pos_oil] / oil_clean[pos_oil]
            avg_gor = float(np.mean(gor_series))
            current_gor = float(gas_clean[-1] / max(1e-3, oil_clean[-1]))
        else:
            avg_gor = 0.0
            current_gor = 0.0

        # Status
        if current_oil < self.min_operating_oil_rate and current_wc >= self.watered_out_wc_threshold:
            status = "WATERED_OUT"
        elif current_oil < self.min_operating_oil_rate:
            status = "SHUT_IN"
        else:
            status = "PRODUCING"

        return WellKPI(
            well_id=w_id,
            calendar_days=calendar_days,
            operating_days=operating_days,
            uptime_pct=round(uptime_pct, 2),
            cum_oil_bbl=round(cum_oil, 2),
            cum_gas_mscf=round(cum_gas, 2),
            cum_water_bbl=round(cum_water, 2),
            cum_liquid_bbl=round(cum_liquid, 2),
            avg_oil_rate_operating_bopd=round(avg_oil_op, 2),
            avg_oil_rate_calendar_bopd=round(avg_oil_cal, 2),
            avg_gas_rate_operating_mscfd=round(avg_gas_op, 2),
            avg_water_cut_pct=round(avg_wc, 2),
            current_water_cut_pct=round(current_wc, 2),
            avg_gor_scf_bbl=round(avg_gor, 2),
            current_gor_scf_bbl=round(current_gor, 2),
            current_oil_rate_bopd=round(current_oil, 2),
            status=status,
        )

    def compute_field_kpi(
        self,
        df: pd.DataFrame,
        field_name: str = "Asset-Alpha",
        well_id_col: str = "well_id",
        oil_col: str = "oil_rate",
        gas_col: str = "gas_rate",
        water_col: str = "water_rate",
    ) -> FieldKPI:
        """
        Rolls up multi-well production data to provide field-level KPIs and well rankings.
        """
        if df.empty:
            return FieldKPI(
                field_name=field_name,
                total_wells=0,
                active_wells=0,
                shut_in_wells=0,
                field_uptime_pct=0.0,
                total_daily_oil_rate_bopd=0.0,
                total_daily_gas_rate_mscfd=0.0,
                total_cum_oil_bbl=0.0,
                total_cum_gas_mscf=0.0,
                total_cum_water_bbl=0.0,
                field_avg_water_cut_pct=0.0,
                top_oil_producers=[],
                top_water_producers=[],
                well_kpis={},
            )

        if well_id_col not in df.columns:
            # Single well dataset
            kpi = self.compute_well_kpi(df, well_id="WELL-01", oil_col=oil_col, gas_col=gas_col, water_col=water_col)
            well_kpis = {kpi.well_id: kpi}
        else:
            well_kpis = {}
            for w_id, group in df.groupby(well_id_col):
                well_kpis[str(w_id)] = self.compute_well_kpi(
                    group, well_id=str(w_id), oil_col=oil_col, gas_col=gas_col, water_col=water_col
                )

        total_wells = len(well_kpis)
        active_wells = sum(1 for k in well_kpis.values() if k.status == "PRODUCING")
        shut_in = total_wells - active_wells

        total_cum_oil = sum(k.cum_oil_bbl for k in well_kpis.values())
        total_cum_gas = sum(k.cum_gas_mscf for k in well_kpis.values())
        total_cum_water = sum(k.cum_water_bbl for k in well_kpis.values())
        current_daily_oil = sum(k.current_oil_rate_bopd for k in well_kpis.values())
        current_daily_gas = sum(k.avg_gas_rate_operating_mscfd for k in well_kpis.values() if k.status == "PRODUCING")

        avg_uptime = float(np.mean([k.uptime_pct for k in well_kpis.values()])) if well_kpis else 0.0
        field_avg_wc = float(
            (total_cum_water / max(1e-4, total_cum_oil + total_cum_water)) * 100.0
        ) if (total_cum_oil + total_cum_water) > 0 else 0.0

        # Rankings
        ranked_by_rate = sorted(
            [
                WellRankingItem(
                    well_id=k.well_id,
                    rate_bopd=k.current_oil_rate_bopd,
                    cum_oil_bbl=k.cum_oil_bbl,
                    water_cut_pct=k.current_water_cut_pct,
                )
                for k in well_kpis.values()
            ],
            key=lambda x: x.rate_bopd,
            reverse=True,
        )

        ranked_by_water = sorted(
            ranked_by_rate,
            key=lambda x: x.water_cut_pct,
            reverse=True,
        )

        return FieldKPI(
            field_name=field_name,
            total_wells=total_wells,
            active_wells=active_wells,
            shut_in_wells=shut_in,
            field_uptime_pct=round(avg_uptime, 2),
            total_daily_oil_rate_bopd=round(current_daily_oil, 2),
            total_daily_gas_rate_mscfd=round(current_daily_gas, 2),
            total_cum_oil_bbl=round(total_cum_oil, 2),
            total_cum_gas_mscf=round(total_cum_gas, 2),
            total_cum_water_bbl=round(total_cum_water, 2),
            field_avg_water_cut_pct=round(field_avg_wc, 2),
            top_oil_producers=ranked_by_rate[:5],
            top_water_producers=ranked_by_water[:5],
            well_kpis=well_kpis,
        )
