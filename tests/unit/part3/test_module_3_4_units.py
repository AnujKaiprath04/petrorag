"""
PetroRAG Module 3.4 Unit Test Suite — Unit Normalization Engine
Verifies high-precision conversions across Pressure, Temperature, Liquid Rates,
Gas Rates, Speed, Vibration, and ensures strict preservation of original values.
"""

import pytest
import pandas as pd
import numpy as np

from src.analytics.units.converter import (
    UnitDimension,
    NormalizedMeasurement,
    UnitConverter,
)


def test_pressure_conversions():
    """Verify pressure conversions against NIST and API reference values."""
    # 1 bar to psi
    m1 = UnitConverter.convert(1.0, from_unit="bar", to_unit="psi")
    assert pytest.approx(m1.normalized_value, rel=1e-4) == 14.5038
    assert m1.original_value == 1.0
    assert m1.dimension == UnitDimension.PRESSURE

    # 14.50377 psi to bar
    m2 = UnitConverter.convert(14.50377, from_unit="psi", to_unit="bar")
    assert pytest.approx(m2.normalized_value, rel=1e-4) == 1.0

    # 100 kPa to bar
    m3 = UnitConverter.convert(100.0, from_unit="kPa", to_unit="bar")
    assert pytest.approx(m3.normalized_value, rel=1e-5) == 1.0

    # 1 MPa to bar
    m4 = UnitConverter.convert(1.0, from_unit="MPa", to_unit="bar")
    assert pytest.approx(m4.normalized_value, rel=1e-5) == 10.0


def test_temperature_conversions():
    """Verify temperature conversions (Celsius, Fahrenheit, Kelvin)."""
    # 0 degC to degF
    m1 = UnitConverter.convert(0.0, from_unit="degC", to_unit="degF")
    assert pytest.approx(m1.normalized_value, abs=1e-2) == 32.0

    # 100 degC to degF
    m2 = UnitConverter.convert(100.0, from_unit="degC", to_unit="degF")
    assert pytest.approx(m2.normalized_value, abs=1e-2) == 212.0

    # 212 degF to degC
    m3 = UnitConverter.convert(212.0, from_unit="degF", to_unit="degC")
    assert pytest.approx(m3.normalized_value, abs=1e-2) == 100.0

    # 0 degC to Kelvin
    m4 = UnitConverter.convert(0.0, from_unit="degC", to_unit="K")
    assert pytest.approx(m4.normalized_value, abs=1e-2) == 273.15

    # 300 K to degC
    m5 = UnitConverter.convert(300.0, from_unit="K", to_unit="degC")
    assert pytest.approx(m5.normalized_value, abs=1e-2) == 26.85


def test_liquid_rate_conversions():
    """Verify volumetric liquid rate conversions (m3/day <-> bbl/day)."""
    # 1 m3/day to bbl/day
    m1 = UnitConverter.convert(1.0, from_unit="m3/day", to_unit="bopd")
    assert pytest.approx(m1.normalized_value, rel=1e-4) == 6.2898

    # 100 bopd to m3/day
    m2 = UnitConverter.convert(100.0, from_unit="bopd", to_unit="m3/day")
    assert pytest.approx(m2.normalized_value, rel=1e-4) == 15.8987


def test_gas_rate_conversions():
    """Verify volumetric gas rate conversions (MMSCFD <-> MSCFD <-> m3/day)."""
    # 1 MMSCFD = 1000 MSCFD
    m1 = UnitConverter.convert(1.0, from_unit="MMSCFD", to_unit="MSCFD")
    assert m1.normalized_value == 1000.0

    # 5000 MSCFD to MMSCFD
    m2 = UnitConverter.convert(5000.0, from_unit="MSCFD", to_unit="MMSCFD")
    assert m2.normalized_value == 5.0


def test_speed_and_vibration_conversions():
    """Verify rotational speed and vibration conversions."""
    # 50 Hz to RPM
    m1 = UnitConverter.convert(50.0, from_unit="Hz", to_unit="RPM")
    assert m1.normalized_value == 3000.0

    # 60 Hz to RPM
    m2 = UnitConverter.convert(60.0, from_unit="Hz", to_unit="RPM")
    assert m2.normalized_value == 3600.0

    # 1.0 in/s to mm/s
    m3 = UnitConverter.convert(1.0, from_unit="in/s", to_unit="mm/s")
    assert m3.normalized_value == 25.4


def test_original_value_preservation():
    """CRITICAL: Ensure original value, unit, and dimension are never mutated or lost."""
    meas = UnitConverter.convert(2450.0, from_unit="psi", to_unit="bar")

    assert meas.original_value == 2450.0
    assert meas.original_unit == "psi"
    assert pytest.approx(meas.normalized_value, rel=1e-4) == 168.9216
    assert meas.normalized_unit == "bar"
    assert meas.dimension == UnitDimension.PRESSURE


def test_dataframe_unit_normalization():
    """Verify batch DataFrame normalization adds normalized columns while preserving originals."""
    df_raw = pd.DataFrame({
        "timestamp": ["2026-01-01", "2026-01-02"],
        "well_id": ["W-101", "W-101"],
        "pressure_psi": [2450.0, 2435.0],
        "temperature_f": [149.0, 150.8],
        "oil_rate_bpd": [1200.0, 1180.0],
    })

    df_norm = UnitConverter.normalize_dataframe(df_raw, unit_system="FIELD")

    # Original columns MUST still exist and remain untouched
    assert "pressure_psi" in df_norm.columns
    assert df_norm.loc[0, "pressure_psi"] == 2450.0
    assert df_norm.loc[0, "temperature_f"] == 149.0
    assert df_norm.loc[0, "oil_rate_bpd"] == 1200.0

    # Normalized columns MUST be added
    assert "pressure_psi_normalized" in df_norm.columns
    assert "temperature_f_normalized" in df_norm.columns
    assert "oil_rate_bpd_normalized" in df_norm.columns

    # Check normalized values (converted to canonical units: bar, degC, bopd)
    assert pytest.approx(df_norm.loc[0, "pressure_psi_normalized"], rel=1e-3) == 168.92
    assert pytest.approx(df_norm.loc[0, "temperature_f_normalized"], abs=1e-1) == 65.0
    assert df_norm.loc[0, "oil_rate_bpd_normalized"] == 1200.0  # bpd to bopd is 1:1

    # Audit columns
    assert df_norm.loc[0, "pressure_psi_orig_unit"] == "psi"
    assert df_norm.loc[0, "pressure_psi_norm_unit"] == "bar"
