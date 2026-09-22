"""
PetroRAG Production Analytics - Inflow Performance Relationship (IPR) (Module 3.7)
Computes reservoir deliverability, Productivity Index (J), Vogel's two-phase IPR,
Composite IPR, wellbore drawdown, and Absolute Open Flow (AOF) potential.
"""

from typing import Dict, List, Any, Optional, Tuple, Literal
import numpy as np
from pydantic import BaseModel, Field


class IPRCurvePoint(BaseModel):
    pwf_psi: float
    oil_rate_bopd: float
    drawdown_psi: float


class IPRResult(BaseModel):
    well_id: str
    model_applied: Literal["LINEAR", "VOGEL", "COMPOSITE"]
    reservoir_pressure_pr_psi: float
    bubble_point_pb_psi: float
    tested_flowing_pressure_pwf_psi: float
    tested_oil_rate_bopd: float
    drawdown_psi: float
    drawdown_pct: float
    productivity_index_j: float = Field(..., description="Productivity Index J in BOPD/psi")
    aof_potential_bopd: float = Field(..., description="Absolute Open Flow potential at Pwf=0")
    operating_flow_ratio_pct: float = Field(..., description="Tested rate as percentage of AOF")
    ipr_curve: List[IPRCurvePoint]


class InflowPerformanceCalculator:
    """
    Computes well deliverability and inflow performance relationships
    using Darcy linear flow, Vogel's equation, or Composite Darcy-Vogel.
    """

    def calculate_ipr(
        self,
        well_id: str,
        tested_rate_bopd: float,
        tested_pwf_psi: float,
        reservoir_pressure_pr_psi: float,
        bubble_point_pb_psi: Optional[float] = None,
        curve_points: int = 15,
    ) -> IPRResult:
        """
        Calculates Productivity Index and generates full IPR deliverability curve.
        """
        pr = max(1.0, float(reservoir_pressure_pr_psi))
        pwf = float(np.clip(tested_pwf_psi, 0.0, pr))
        qt = max(0.1, float(tested_rate_bopd))
        drawdown = pr - pwf
        drawdown_pct = float((drawdown / pr) * 100.0)

        pb = float(bubble_point_pb_psi) if bubble_point_pb_psi is not None else pr

        # Determine appropriate model
        if pb <= 0.0 or pwf >= pb:
            # Entirely single-phase liquid above bubble point -> Linear Darcy
            model = "LINEAR"
            j = float(qt / max(1e-4, drawdown))
            aof = float(j * pr)
            curve_fn = lambda p: j * (pr - p)

        elif pb >= pr:
            # Entirely two-phase flow -> Pure Vogel (1968)
            model = "VOGEL"
            ratio = pwf / pr
            vogel_factor = 1.0 - 0.2 * ratio - 0.8 * (ratio ** 2)
            aof = float(qt / max(1e-4, vogel_factor))
            j = float(aof / (1.8 * pr))  # Tangent PI at Pwf=Pr
            curve_fn = lambda p: aof * (1.0 - 0.2 * (p / pr) - 0.8 * ((p / pr) ** 2))

        else:
            # Composite IPR: Linear for Pwf >= Pb, Vogel for Pwf < Pb
            model = "COMPOSITE"
            # Solve for J given tested point below Pb
            # q_t = q_b + (J * pb / 1.8) * (1 - 0.2*(pwf/pb) - 0.8*(pwf/pb)^2)
            # where q_b = J * (pr - pb)
            # => q_t = J * [ (pr - pb) + (pb / 1.8) * (1 - 0.2*(pwf/pb) - 0.8*(pwf/pb)^2) ]
            ratio_b = pwf / pb
            vogel_term = (pb / 1.8) * (1.0 - 0.2 * ratio_b - 0.8 * (ratio_b ** 2))
            composite_denom = (pr - pb) + vogel_term
            j = float(qt / max(1e-4, composite_denom))
            qb = j * (pr - pb)
            aof = float(qb + (j * pb / 1.8))

            def composite_curve(p: float) -> float:
                if p >= pb:
                    return float(j * (pr - p))
                else:
                    r = p / pb
                    return float(qb + (j * pb / 1.8) * (1.0 - 0.2 * r - 0.8 * (r ** 2)))

            curve_fn = composite_curve

        operating_pct = float((qt / max(1e-4, aof)) * 100.0)

        # Generate discrete curve
        pwf_steps = np.linspace(pr, 0.0, max(5, curve_points))
        curve_pts: List[IPRCurvePoint] = []
        for p_val in pwf_steps:
            q_val = float(max(0.0, curve_fn(p_val)))
            dd = float(pr - p_val)
            curve_pts.append(
                IPRCurvePoint(
                    pwf_psi=round(float(p_val), 1),
                    oil_rate_bopd=round(q_val, 1),
                    drawdown_psi=round(dd, 1),
                )
            )

        return IPRResult(
            well_id=well_id,
            model_applied=model,
            reservoir_pressure_pr_psi=round(pr, 1),
            bubble_point_pb_psi=round(pb, 1),
            tested_flowing_pressure_pwf_psi=round(pwf, 1),
            tested_oil_rate_bopd=round(qt, 1),
            drawdown_psi=round(drawdown, 1),
            drawdown_pct=round(drawdown_pct, 1),
            productivity_index_j=round(j, 4),
            aof_potential_bopd=round(aof, 1),
            operating_flow_ratio_pct=round(operating_pct, 1),
            ipr_curve=curve_pts,
        )
