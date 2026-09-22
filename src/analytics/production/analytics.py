"""
PetroRAG Production Analytics - Unified Production Intelligence Service (Module 3.7)
Orchestrates Arps decline curve analysis, well & field KPIs, Inflow Performance Relationships (IPR),
and Chan water coning/breakthrough diagnostics into comprehensive production evaluations.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field

from src.analytics.production.arps import ArpsDeclineCurve, ArpsFitResult
from src.analytics.production.kpi import KPICalculator, WellKPI, FieldKPI
from src.analytics.production.ipr import InflowPerformanceCalculator, IPRResult
from src.analytics.production.diagnostics import ChanWaterDiagnosticEngine, ChanDiagnosticResult


class ComprehensiveProductionReport(BaseModel):
    well_id: str
    well_kpi: WellKPI
    arps_dca: ArpsFitResult
    chan_diagnostic: ChanDiagnosticResult
    ipr: Optional[IPRResult] = None
    executive_summary: str
    recommended_interventions: List[str]

    def to_json(self, indent: int = 2) -> str:
        """Serializes report to JSON string."""
        return self.model_dump_json(indent=indent)

    def generate_rag_context(self) -> str:
        """
        Generates technical markdown context for PetroRAG LLM grounding.
        """
        lines = [
            f"### Production Engineering Diagnostic Assessment: Well {self.well_id}",
            f"- **Status**: {self.well_kpi.status} | **Uptime**: {self.well_kpi.uptime_pct}% over {self.well_kpi.calendar_days} days.",
            f"- **Current Oil Rate**: {self.well_kpi.current_oil_rate_bopd} BOPD | **Cumulative Oil**: {self.well_kpi.cum_oil_bbl:,.0f} bbl.",
            f"- **Water Cut**: {self.well_kpi.current_water_cut_pct:.1f}% (Current) | {self.well_kpi.avg_water_cut_pct:.1f}% (Average).",
            f"- **Gas-Oil Ratio**: {self.well_kpi.current_gor_scf_bbl:.1f} scf/bbl.",
            "",
            "#### Decline Curve Analysis (Arps)",
            f"- **Best-Fit Model**: {self.arps_dca.model_type} (R² = {self.arps_dca.r_squared:.3f}, b = {self.arps_dca.b:.2f}).",
            f"- **Decline Rate**: {self.arps_dca.Di_annual_pct:.1f}% per year (nominal Di = {self.arps_dca.Di:.6f}/day).",
            f"- **Estimated Ultimate Recovery (EUR)**: {self.arps_dca.eur_bbl:,.0f} bbl (at {self.arps_dca.economic_limit_rate} BOPD economic limit).",
            f"- **Remaining Recoverable Reserves (RRR)**: {self.arps_dca.remaining_reserves_bbl:,.0f} bbl ({self.arps_dca.time_to_economic_limit_days:.0f} days to limit).",
            "",
            "#### Water Production Diagnostics (Chan 1995)",
            f"- **Diagnosis**: {self.chan_diagnostic.water_breakthrough_mechanism.replace('_', ' ')} (Confidence: {self.chan_diagnostic.diagnostic_confidence * 100:.0f}%).",
            f"- **Observation**: {self.chan_diagnostic.diagnostic_narrative}",
        ]

        if self.ipr:
            lines.extend(
                [
                    "",
                    "#### Inflow Performance & Deliverability (IPR)",
                    f"- **Model**: {self.ipr.model_applied} | **Productivity Index (J)**: {self.ipr.productivity_index_j:.3f} BOPD/psi.",
                    f"- **AOF Potential**: {self.ipr.aof_potential_bopd:.1f} BOPD | **Current Operating Ratio**: {self.ipr.operating_flow_ratio_pct:.1f}% of AOF.",
                    f"- **Reservoir Pressure**: {self.ipr.reservoir_pressure_pr_psi} psi | **Drawdown**: {self.ipr.drawdown_psi} psi ({self.ipr.drawdown_pct:.1f}%).",
                ]
            )

        if self.recommended_interventions:
            lines.extend(["", "#### Recommended Interventions & Remedial Actions:"])
            for rec in self.recommended_interventions:
                lines.append(f"- {rec}")

        return "\n".join(lines)


class ProductionAnalyzer:
    """
    Integrates Decline Curve Analysis, Production KPIs, Inflow Performance,
    and Chan Diagnostics for comprehensive well evaluations.
    """

    def __init__(self, economic_limit_rate: float = 15.0):
        self.dca_engine = ArpsDeclineCurve(economic_limit_rate=economic_limit_rate)
        self.kpi_engine = KPICalculator()
        self.ipr_engine = InflowPerformanceCalculator()
        self.chan_engine = ChanWaterDiagnosticEngine()

    def evaluate_well(
        self,
        df: pd.DataFrame,
        well_id: Optional[str] = None,
        reservoir_pressure_psi: Optional[float] = None,
        flowing_pressure_psi: Optional[float] = None,
        bubble_point_psi: Optional[float] = None,
        oil_col: str = "oil_rate",
        gas_col: str = "gas_rate",
        water_col: str = "water_rate",
        time_col: str = "timestamp",
    ) -> ComprehensiveProductionReport:
        """
        Executes full production analytics workflow for a target well.
        """
        w_id = well_id or (str(df["well_id"].iloc[0]) if "well_id" in df.columns and len(df) > 0 else "WELL-01")

        # 1. Compute KPIs
        kpi = self.kpi_engine.compute_well_kpi(
            df=df,
            well_id=w_id,
            oil_col=oil_col,
            gas_col=gas_col,
            water_col=water_col,
            time_col=time_col,
        )

        # 2. Fit Arps DCA
        if time_col in df.columns:
            sorted_df = df.sort_values(by=time_col)
            t_deltas = (pd.to_datetime(sorted_df[time_col]) - pd.to_datetime(sorted_df[time_col]).iloc[0]).dt.total_seconds() / 86400.0
            time_days = t_deltas.to_numpy(dtype=float)
            rates = sorted_df[oil_col].to_numpy(dtype=float)
        else:
            time_days = np.arange(len(df), dtype=float)
            rates = df[oil_col].to_numpy(dtype=float) if oil_col in df.columns else np.zeros(len(df))

        dca_fit = self.dca_engine.fit(time_days, rates)

        # 3. Chan Water Diagnostics
        chan_res = self.chan_engine.analyze_well(
            df=df,
            well_id=w_id,
            oil_col=oil_col,
            water_col=water_col,
            time_col=time_col,
        )

        # 4. Optional IPR Deliverability
        ipr_res = None
        if reservoir_pressure_psi is not None and flowing_pressure_psi is not None:
            tested_rate = kpi.current_oil_rate_bopd if kpi.current_oil_rate_bopd > 0 else 100.0
            ipr_res = self.ipr_engine.calculate_ipr(
                well_id=w_id,
                tested_rate_bopd=tested_rate,
                tested_pwf_psi=flowing_pressure_psi,
                reservoir_pressure_pr_psi=reservoir_pressure_psi,
                bubble_point_pb_psi=bubble_point_psi,
            )

        # 5. Synthesize Recommendations & Summary
        interventions = []
        if chan_res.recommended_remedial_action:
            interventions.append(chan_res.recommended_remedial_action)

        if dca_fit.Di_annual_pct > 25.0:
            interventions.append(
                f"High reservoir decline rate ({dca_fit.Di_annual_pct:.1f}%/yr). Evaluate pressure maintenance, "
                f"infill drilling, or stimulation workover."
            )

        if ipr_res and ipr_res.operating_flow_ratio_pct < 40.0:
            interventions.append(
                f"Well is producing at only {ipr_res.operating_flow_ratio_pct:.1f}% of AOF potential. "
                f"Significant wellbore drawdown potential exists; consider optimizing artificial lift."
            )

        exec_summary = (
            f"Well {w_id} has produced {kpi.cum_oil_bbl:,.0f} bbl oil with {kpi.uptime_pct:.1f}% operating uptime. "
            f"Current rate is {kpi.current_oil_rate_bopd:.1f} BOPD with {kpi.current_water_cut_pct:.1f}% water cut. "
            f"Arps DCA ({dca_fit.model_type}) estimates {dca_fit.remaining_reserves_bbl:,.0f} bbl remaining recoverable reserves. "
            f"Water diagnostic identifies: {chan_res.water_breakthrough_mechanism.replace('_', ' ').lower()}."
        )

        return ComprehensiveProductionReport(
            well_id=w_id,
            well_kpi=kpi,
            arps_dca=dca_fit,
            chan_diagnostic=chan_res,
            ipr=ipr_res,
            executive_summary=exec_summary,
            recommended_interventions=interventions,
        )
