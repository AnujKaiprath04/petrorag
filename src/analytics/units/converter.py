"""
PetroRAG Unit Normalization Engine (Module 3.4)
Provides high-precision unit conversions across Oil & Gas engineering dimensions:
Pressure (bar, psi, kPa, MPa), Temperature (°C, °F, K), Liquid Rate (bbl/d, m3/d, m3/h),
Gas Rate (MMSCFD, MSCFD, m3/d), Rotational Speed (RPM, Hz), and Vibration (mm/s, in/s).
Preserves original measurements at all times.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.core.logging import logger


class UnitDimension(str, Enum):
    PRESSURE = "PRESSURE"
    TEMPERATURE = "TEMPERATURE"
    LIQUID_RATE = "LIQUID_RATE"
    GAS_RATE = "GAS_RATE"
    ROTATIONAL_SPEED = "ROTATIONAL_SPEED"
    VIBRATION = "VIBRATION"


class NormalizedMeasurement(BaseModel):
    """Immutable audit record maintaining original vs normalized values."""
    original_value: float
    original_unit: str
    normalized_value: float
    normalized_unit: str
    dimension: UnitDimension

    def __repr__(self) -> str:
        return f"{self.original_value} {self.original_unit} -> {self.normalized_value} {self.normalized_unit}"


class UnitConverter:
    """
    Standardizes industrial measurements against reference SI / Metric / Field standards.
    Conversions conform to NIST and Society of Petroleum Engineers (SPE) standards.
    """

    # Reference canonical target units
    CANONICAL_TARGETS: Dict[UnitDimension, str] = {
        UnitDimension.PRESSURE: "bar",
        UnitDimension.TEMPERATURE: "degC",
        UnitDimension.LIQUID_RATE: "bopd",      # barrels of oil per day (standard oilfield)
        UnitDimension.GAS_RATE: "mscfd",        # thousand standard cubic feet per day
        UnitDimension.ROTATIONAL_SPEED: "rpm",
        UnitDimension.VIBRATION: "mm/s",        # ISO 10816-3 standard (mm/s RMS)
    }

    # Factor-to-canonical conversion multipliers
    # e.g., value_in_unit * factor = value_in_canonical
    _PRESSURE_FACTORS_TO_BAR = {
        "bar": 1.0,
        "psi": 0.06894757293,       # 1 psi = 0.06894757 bar (1 bar = 14.50377 psi)
        "kpa": 0.01,                # 1 bar = 100 kPa
        "mpa": 10.0,                # 1 MPa = 10 bar
        "atm": 1.01325,
    }

    _LIQUID_FACTORS_TO_BOPD = {
        "bopd": 1.0,
        "bpd": 1.0,
        "bbl/day": 1.0,
        "bwpd": 1.0,
        "m3/day": 6.28981077,       # 1 m3 = 6.28981077 bbl (API standard)
        "m3/d": 6.28981077,
        "m3/h": 150.9554585,        # 6.28981077 * 24
        "l/min": 9.0573275,         # (1e-3 * 6.28981077) * 1440
    }

    _GAS_FACTORS_TO_MSCFD = {
        "mscfd": 1.0,
        "mmscfd": 1000.0,           # 1 MMSCFD = 1000 MSCFD
        "scfd": 0.001,
        "m3/day": 0.0353146667,     # 1 m3 = 35.3146667 scf = 0.0353146667 MSCF
        "m3/d": 0.0353146667,
        "m3/h": 0.847552,           # 0.0353146667 * 24
    }

    _SPEED_FACTORS_TO_RPM = {
        "rpm": 1.0,
        "hz": 60.0,                 # 1 Hz = 60 RPM
        "rad/s": 9.5492965855,      # 60 / (2 * pi)
    }

    _VIBRATION_FACTORS_TO_MMS = {
        "mm/s": 1.0,
        "mms": 1.0,
        "in/s": 25.4,               # 1 in = 25.4 mm
        "ips": 25.4,
    }

    @classmethod
    def convert(
        cls,
        value: float,
        from_unit: str,
        to_unit: Optional[str] = None,
        dimension: Optional[UnitDimension] = None,
    ) -> NormalizedMeasurement:
        """
        Convert scalar measurement between engineering units.
        If to_unit is None, converts to canonical reference unit for that dimension.
        """
        u_from = from_unit.lower().strip().replace("°", "").replace("deg", "")
        if u_from in ["c", "degc"]:
            dim = UnitDimension.TEMPERATURE
        elif u_from in ["f", "degf"]:
            dim = UnitDimension.TEMPERATURE
        elif u_from in ["k", "kelvin"]:
            dim = UnitDimension.TEMPERATURE
        elif u_from in cls._PRESSURE_FACTORS_TO_BAR:
            dim = UnitDimension.PRESSURE
        elif u_from in cls._LIQUID_FACTORS_TO_BOPD:
            dim = UnitDimension.LIQUID_RATE
        elif u_from in cls._GAS_FACTORS_TO_MSCFD:
            dim = UnitDimension.GAS_RATE
        elif u_from in cls._SPEED_FACTORS_TO_RPM:
            dim = UnitDimension.ROTATIONAL_SPEED
        elif u_from in cls._VIBRATION_FACTORS_TO_MMS:
            dim = UnitDimension.VIBRATION
        elif dimension:
            dim = dimension
        else:
            raise ValueError(f"Unrecognized unit '{from_unit}'.")

        target_unit = to_unit or cls.CANONICAL_TARGETS[dim]
        u_to = target_unit.lower().strip().replace("°", "").replace("deg", "")

        # 1. Temperature Conversion (Affine)
        if dim == UnitDimension.TEMPERATURE:
            # Convert to Celsius first
            if u_from == "c":
                c_val = value
            elif u_from == "f":
                c_val = (value - 32.0) * (5.0 / 9.0)
            elif u_from in ["k", "kelvin"]:
                c_val = value - 273.15
            else:
                raise ValueError(f"Unsupported temperature unit '{from_unit}'")

            # Convert Celsius to target
            if u_to in ["c", "degc"]:
                out_val = c_val
            elif u_to in ["f", "degf"]:
                out_val = c_val * (9.0 / 5.0) + 32.0
            elif u_to in ["k", "kelvin"]:
                out_val = c_val + 273.15
            else:
                raise ValueError(f"Unsupported target temperature unit '{to_unit}'")

        # 2. Pressure Conversion
        elif dim == UnitDimension.PRESSURE:
            if u_from not in cls._PRESSURE_FACTORS_TO_BAR:
                raise ValueError(f"Unsupported pressure unit '{from_unit}'")
            if u_to not in cls._PRESSURE_FACTORS_TO_BAR:
                raise ValueError(f"Unsupported target pressure unit '{to_unit}'")
            bar_val = value * cls._PRESSURE_FACTORS_TO_BAR[u_from]
            out_val = bar_val / cls._PRESSURE_FACTORS_TO_BAR[u_to]

        # 3. Liquid Rate Conversion
        elif dim == UnitDimension.LIQUID_RATE:
            if u_from not in cls._LIQUID_FACTORS_TO_BOPD:
                raise ValueError(f"Unsupported liquid rate unit '{from_unit}'")
            if u_to not in cls._LIQUID_FACTORS_TO_BOPD:
                raise ValueError(f"Unsupported target liquid rate unit '{to_unit}'")
            bopd_val = value * cls._LIQUID_FACTORS_TO_BOPD[u_from]
            out_val = bopd_val / cls._LIQUID_FACTORS_TO_BOPD[u_to]

        # 4. Gas Rate Conversion
        elif dim == UnitDimension.GAS_RATE:
            if u_from not in cls._GAS_FACTORS_TO_MSCFD:
                raise ValueError(f"Unsupported gas rate unit '{from_unit}'")
            if u_to not in cls._GAS_FACTORS_TO_MSCFD:
                raise ValueError(f"Unsupported target gas rate unit '{to_unit}'")
            mscfd_val = value * cls._GAS_FACTORS_TO_MSCFD[u_from]
            out_val = mscfd_val / cls._GAS_FACTORS_TO_MSCFD[u_to]

        # 5. Speed Conversion
        elif dim == UnitDimension.ROTATIONAL_SPEED:
            if u_from not in cls._SPEED_FACTORS_TO_RPM:
                raise ValueError(f"Unsupported speed unit '{from_unit}'")
            if u_to not in cls._SPEED_FACTORS_TO_RPM:
                raise ValueError(f"Unsupported target speed unit '{to_unit}'")
            rpm_val = value * cls._SPEED_FACTORS_TO_RPM[u_from]
            out_val = rpm_val / cls._SPEED_FACTORS_TO_RPM[u_to]

        # 6. Vibration Conversion
        elif dim == UnitDimension.VIBRATION:
            if u_from not in cls._VIBRATION_FACTORS_TO_MMS:
                raise ValueError(f"Unsupported vibration unit '{from_unit}'")
            if u_to not in cls._VIBRATION_FACTORS_TO_MMS:
                raise ValueError(f"Unsupported target vibration unit '{to_unit}'")
            mms_val = value * cls._VIBRATION_FACTORS_TO_MMS[u_from]
            out_val = mms_val / cls._VIBRATION_FACTORS_TO_MMS[u_to]

        return NormalizedMeasurement(
            original_value=round(value, 4),
            original_unit=from_unit,
            normalized_value=round(out_val, 4),
            normalized_unit=target_unit,
            dimension=dim,
        )

    @classmethod
    def normalize_dataframe(
        cls,
        df: pd.DataFrame,
        unit_system: str = "FIELD",
        column_units: Optional[Dict[str, str]] = None,
        preserve_originals: bool = True,
    ) -> pd.DataFrame:
        """
        Normalize DataFrame columns while preserving original readings.
        Adds `{col}_norm` and `{col}_unit` columns without destroying `{col}`.
        """
        df_out = df.copy()
        user_units = column_units or {}

        # Default standard unit mapping by parameter name and unit_system
        if unit_system.upper() == "FIELD":
            default_map = {
                "pressure": "psi",
                "temperature": "degF",
                "oil_rate": "bopd",
                "gas_rate": "mscfd",
                "water_rate": "bwpd",
                "vibration": "mm/s",
                "rpm": "rpm",
            }
        else:  # METRIC / SI
            default_map = {
                "pressure": "bar",
                "temperature": "degC",
                "oil_rate": "m3/day",
                "gas_rate": "m3/day",
                "water_rate": "m3/day",
                "vibration": "mm/s",
                "rpm": "rpm",
            }

        for col in df_out.columns:
            # Check if column matches an engineering parameter
            col_lower = col.lower()
            matched_key = None
            for key in default_map:
                if key in col_lower:
                    matched_key = key
                    break
            if not matched_key:
                continue

            base_col = matched_key

            input_unit = user_units.get(col, None)
            if not input_unit:
                # Infer unit from column name suffix
                if "_psi" in col.lower():
                    input_unit = "psi"
                elif "_bar" in col.lower():
                    input_unit = "bar"
                elif "_c" in col.lower() or "temp_c" in col.lower():
                    input_unit = "degC"
                elif "_f" in col.lower():
                    input_unit = "degF"
                elif "_bopd" in col.lower() or "_bpd" in col.lower():
                    input_unit = "bopd"
                elif "_mscfd" in col.lower():
                    input_unit = "mscfd"
                elif "_mmscfd" in col.lower():
                    input_unit = "mmscfd"
                elif "_m3h" in col.lower():
                    input_unit = "m3/h"
                elif "_mms" in col.lower() or "vib" in col.lower():
                    input_unit = "mm/s"
                else:
                    input_unit = default_map[base_col]

            target_unit = cls.CANONICAL_TARGETS.get(
                cls._get_dimension_for_param(base_col),
                input_unit
            )

            # Apply conversion
            series = pd.to_numeric(df_out[col], errors="coerce")
            if series.dropna().empty:
                continue

            norm_vals = []
            for val in series:
                if pd.isna(val):
                    norm_vals.append(np.nan)
                else:
                    meas = cls.convert(float(val), from_unit=input_unit, to_unit=target_unit)
                    norm_vals.append(meas.normalized_value)

            norm_col_name = f"{col}_normalized"
            df_out[norm_col_name] = norm_vals
            if preserve_originals:
                df_out[f"{col}_orig_unit"] = input_unit
                df_out[f"{col}_norm_unit"] = target_unit

        return df_out

    @classmethod
    def _get_dimension_for_param(cls, param: str) -> UnitDimension:
        if "pressure" in param:
            return UnitDimension.PRESSURE
        elif "temp" in param:
            return UnitDimension.TEMPERATURE
        elif "oil" in param or "water" in param or "liquid" in param:
            return UnitDimension.LIQUID_RATE
        elif "gas" in param:
            return UnitDimension.GAS_RATE
        elif "speed" in param or "rpm" in param:
            return UnitDimension.ROTATIONAL_SPEED
        elif "vib" in param:
            return UnitDimension.VIBRATION
        return UnitDimension.PRESSURE
