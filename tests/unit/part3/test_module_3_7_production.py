"""
Unit Tests for PetroRAG Module 3.7 - Production Performance Analytics
Validates Arps decline curve analysis (exponential, hyperbolic, harmonic),
well & field KPIs, Inflow Performance Relationships (IPR / Vogel),
Chan water coning & breakthrough diagnostics, and end-to-end production evaluation.
"""

from datetime import datetime, timedelta
from typing import Tuple
import numpy as np
import pandas as pd
import pytest

from src.analytics.production.arps import ArpsDeclineCurve, ArpsFitResult
from src.analytics.production.kpi import KPICalculator, WellKPI, FieldKPI
from src.analytics.production.ipr import InflowPerformanceCalculator, IPRResult
from src.analytics.production.diagnostics import ChanWaterDiagnosticEngine, ChanDiagnosticResult
from src.analytics.production.analytics import ProductionAnalyzer, ComprehensiveProductionReport
from src.analytics.services import ProductionAnalyticsService


@pytest.fixture
def synthetic_exponential_production() -> Tuple[np.ndarray, np.ndarray]:
    """Generates 180 days of known exponential decline data."""
    t = np.arange(180, dtype=float)
    qi_true = 1200.0
    di_true = 0.003
    q = qi_true * np.exp(-di_true * t) + np.random.normal(0, 10, 180)
    return t, np.maximum(50.0, q)


@pytest.fixture
def multi_well_production_df() -> pd.DataFrame:
    """Generates 60 days of production for 3 distinct wells in an asset."""
    np.random.seed(42)
    records = []
    base_date = datetime(2025, 1, 1)

    for day in range(60):
        ts = base_date + timedelta(days=day)
        # Well A: High oil, low water
        records.append({
            "timestamp": ts,
            "well_id": "WELL-A",
            "oil_rate": max(0.0, 1500.0 - 2.0 * day + np.random.normal(0, 15)),
            "gas_rate": max(0.0, 1200.0 - 1.5 * day + np.random.normal(0, 10)),
            "water_rate": 150.0 + 0.5 * day + np.random.normal(0, 5),
            "tubing_pressure": 2400.0 - 1.0 * day,
        })
        # Well B: Moderate oil, high water (channeling)
        records.append({
            "timestamp": ts,
            "well_id": "WELL-B",
            "oil_rate": max(0.0, 600.0 - 4.0 * day + np.random.normal(0, 10)),
            "gas_rate": max(0.0, 400.0 - 2.0 * day + np.random.normal(0, 8)),
            "water_rate": 200.0 + 8.0 * day + np.random.normal(0, 8),
            "tubing_pressure": 2100.0 - 2.5 * day,
        })
        # Well C: Shut-in intermittently
        oil_c = 0.0 if day % 5 == 0 else max(0.0, 300.0 + np.random.normal(0, 10))
        records.append({
            "timestamp": ts,
            "well_id": "WELL-C",
            "oil_rate": oil_c,
            "gas_rate": oil_c * 0.8,
            "water_rate": 50.0,
            "tubing_pressure": 1800.0,
        })

    return pd.DataFrame(records)


def test_arps_exponential_fit_and_eur():
    t = np.arange(120, dtype=float)
    q = 1000.0 * np.exp(-0.0025 * t)
    dca = ArpsDeclineCurve(economic_limit_rate=15.0)
    fit = dca.fit(t, q, preferred_model="EXPONENTIAL")

    assert fit.model_type == "EXPONENTIAL"
    assert fit.r_squared > 0.98
    assert abs(fit.qi - 1000.0) < 30.0
    assert abs(fit.Di - 0.0025) < 0.0005
    assert fit.eur_bbl > fit.cum_historical_production_bbl
    assert fit.time_to_economic_limit_days > 120.0
    assert len(fit.forecast) == 12


def test_arps_hyperbolic_and_harmonic_fit():
    t = np.arange(150, dtype=float)
    qi = 800.0
    di = 0.005
    b_true = 0.4
    q_hyp = qi / ((1.0 + b_true * di * t) ** (1.0 / b_true))

    dca = ArpsDeclineCurve(economic_limit_rate=20.0)
    fit = dca.fit(t, q_hyp, preferred_model="HYPERBOLIC")

    assert fit.model_type == "HYPERBOLIC"
    assert fit.r_squared > 0.95
    assert 0.05 <= fit.b <= 0.95
    assert fit.eur_bbl > 0.0
    assert fit.remaining_reserves_bbl > 0.0


def test_well_kpi_computation(multi_well_production_df):
    kpi_calc = KPICalculator()
    df_a = multi_well_production_df[multi_well_production_df["well_id"] == "WELL-A"]
    kpi_a = kpi_calc.compute_well_kpi(df_a, well_id="WELL-A")

    assert kpi_a.well_id == "WELL-A"
    assert kpi_a.calendar_days == 60
    assert kpi_a.operating_days == 60
    assert kpi_a.uptime_pct == 100.0
    assert kpi_a.cum_oil_bbl > 70000.0
    assert kpi_a.current_oil_rate_bopd > 1200.0
    assert kpi_a.status == "PRODUCING"

    # Well C with shut-in days
    df_c = multi_well_production_df[multi_well_production_df["well_id"] == "WELL-C"]
    kpi_c = kpi_calc.compute_well_kpi(df_c, well_id="WELL-C")
    assert kpi_c.calendar_days == 60
    assert kpi_c.operating_days < 60
    assert kpi_c.uptime_pct < 100.0


def test_field_kpi_rollup_and_rankings(multi_well_production_df):
    kpi_calc = KPICalculator()
    field_kpi = kpi_calc.compute_field_kpi(multi_well_production_df, field_name="Offshore-Block-A")

    assert field_kpi.field_name == "Offshore-Block-A"
    assert field_kpi.total_wells == 3
    assert field_kpi.active_wells >= 2
    assert field_kpi.total_cum_oil_bbl > 80000.0
    assert field_kpi.total_daily_oil_rate_bopd > 1000.0

    # Top oil producer should be WELL-A
    assert len(field_kpi.top_oil_producers) == 3
    assert field_kpi.top_oil_producers[0].well_id == "WELL-A"

    # Top water producer should be WELL-B
    assert field_kpi.top_water_producers[0].well_id == "WELL-B"


def test_inflow_performance_relationship_darcy_and_vogel():
    ipr_calc = InflowPerformanceCalculator()

    # 1. Linear Darcy test (Single phase above Pb)
    linear_ipr = ipr_calc.calculate_ipr(
        well_id="WELL-01",
        tested_rate_bopd=500.0,
        tested_pwf_psi=2000.0,
        reservoir_pressure_pr_psi=3000.0,
        bubble_point_pb_psi=1500.0,
    )
    assert linear_ipr.model_applied == "LINEAR"
    assert linear_ipr.drawdown_psi == 1000.0
    assert abs(linear_ipr.productivity_index_j - 0.5) < 1e-3
    assert abs(linear_ipr.aof_potential_bopd - 1500.0) < 1.0
    assert len(linear_ipr.ipr_curve) == 15

    # 2. Pure Vogel test (Two phase: Pb >= Pr)
    vogel_ipr = ipr_calc.calculate_ipr(
        well_id="WELL-02",
        tested_rate_bopd=600.0,
        tested_pwf_psi=1800.0,
        reservoir_pressure_pr_psi=3000.0,
        bubble_point_pb_psi=3200.0,
    )
    assert vogel_ipr.model_applied == "VOGEL"
    assert vogel_ipr.aof_potential_bopd > 600.0
    assert vogel_ipr.productivity_index_j > 0.0


def test_chan_water_coning_and_channeling_diagnostics():
    chan_engine = ChanWaterDiagnosticEngine()

    # Case 1: Steep Channeling dataset
    t = np.arange(1, 40, dtype=float)
    oil = np.maximum(50.0, 1000.0 - 15.0 * t)
    # Rapid water surge (exponential climb)
    water_channeling = 100.0 * np.exp(0.08 * t)
    df_channel = pd.DataFrame({
        "timestamp": [datetime(2025, 1, 1) + timedelta(days=int(i)) for i in t],
        "well_id": "WELL-CHAN",
        "oil_rate": oil,
        "water_rate": water_channeling,
    })

    res_chan = chan_engine.analyze_well(df_channel)
    assert res_chan.water_breakthrough_mechanism == "CHANNELING_OR_FRACTURE"
    assert "channeling" in res_chan.diagnostic_narrative.lower() or "fracture" in res_chan.diagnostic_narrative.lower()
    assert "isolation" in res_chan.recommended_remedial_action.lower() or "packers" in res_chan.recommended_remedial_action.lower()

    # Case 2: Stable low water
    df_low = pd.DataFrame({
        "timestamp": [datetime(2025, 1, 1) + timedelta(days=int(i)) for i in t],
        "well_id": "WELL-LOW",
        "oil_rate": [1200.0] * len(t),
        "water_rate": [10.0] * len(t),
    })
    res_low = chan_engine.analyze_well(df_low)
    assert res_low.water_breakthrough_mechanism == "STABLE_LOW_WATER"


def test_production_analyzer_end_to_end(multi_well_production_df):
    analyzer = ProductionAnalyzer(economic_limit_rate=15.0)
    df_a = multi_well_production_df[multi_well_production_df["well_id"] == "WELL-A"]

    report = analyzer.evaluate_well(
        df=df_a,
        well_id="WELL-A",
        reservoir_pressure_psi=2800.0,
        flowing_pressure_psi=2100.0,
        bubble_point_psi=1900.0,
    )

    assert report.well_id == "WELL-A"
    assert report.well_kpi.status == "PRODUCING"
    assert report.arps_dca.eur_bbl > 0.0
    assert report.ipr is not None
    assert report.ipr.productivity_index_j > 0.0
    assert len(report.recommended_interventions) >= 1

    # RAG context generation
    rag_context = report.generate_rag_context()
    assert "Production Engineering Diagnostic Assessment" in rag_context
    assert "Decline Curve Analysis" in rag_context
    assert "Water Production Diagnostics" in rag_context
    assert "Inflow Performance" in rag_context

    # JSON serialization
    json_str = report.to_json()
    assert "WELL-A" in json_str
    assert "arps_dca" in json_str


def test_production_analytics_service_integration(multi_well_production_df):
    service = ProductionAnalyticsService()
    df_a = multi_well_production_df[multi_well_production_df["well_id"] == "WELL-A"]

    eval_report = service.evaluate_production(
        df=df_a,
        well_id="WELL-A",
        reservoir_pressure_psi=2900.0,
        flowing_pressure_psi=2200.0,
    )
    assert isinstance(eval_report, ComprehensiveProductionReport)
    assert eval_report.well_id == "WELL-A"

    field_kpi = service.compute_field_overview(
        df=multi_well_production_df,
        field_name="North-Sea-Bravo",
    )
    assert isinstance(field_kpi, FieldKPI)
    assert field_kpi.total_wells == 3
