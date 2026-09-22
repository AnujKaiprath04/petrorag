"""
PetroRAG Production Analytics - Arps Decline Curve Analysis (Module 3.7)
Implements classical petroleum engineering decline curve analysis (DCA):
Exponential (b=0), Hyperbolic (0 < b < 1), and Harmonic (b=1) decline models,
non-linear curve fitting, model selection (AIC/BIC/R²), and Estimated Ultimate
Recovery (EUR) calculation at economic limit rates.
"""

from typing import Dict, List, Any, Optional, Tuple, Literal
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from pydantic import BaseModel, Field


class ArpsForecastPoint(BaseModel):
    time_days: float
    rate_bopd: float
    cum_production_bbl: float


class ArpsFitResult(BaseModel):
    model_type: Literal["EXPONENTIAL", "HYPERBOLIC", "HARMONIC"]
    qi: float = Field(..., description="Initial production rate at t=0 (bopd or m3/d)")
    Di: float = Field(..., description="Nominal decline rate per day")
    Di_annual_pct: float = Field(..., description="Effective annual decline rate in percent")
    b: float = Field(..., description="Hyperbolic decline exponent (0=exp, 1=harmonic)")
    r_squared: float
    rmse: float
    aic: float
    bic: float
    economic_limit_rate: float
    time_to_economic_limit_days: float
    eur_bbl: float = Field(..., description="Estimated Ultimate Recovery at economic limit")
    remaining_reserves_bbl: float = Field(..., description="Remaining recoverable reserves from current time")
    cum_historical_production_bbl: float
    forecast: List[ArpsForecastPoint]


class ArpsDeclineCurve:
    """
    Decline Curve Analysis engine implementing Arps equations (1945).
    Fits exponential, hyperbolic, and harmonic models to production data,
    evaluating the optimum reservoir decline trajectory.
    """

    def __init__(self, economic_limit_rate: float = 15.0, forecast_horizon_days: int = 365):
        self.economic_limit_rate = max(0.1, economic_limit_rate)
        self.forecast_horizon_days = forecast_horizon_days

    # -----------------------------------------------------------------------
    # Rate Formulations
    # -----------------------------------------------------------------------
    @staticmethod
    def rate_exponential(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
        return qi * np.exp(-Di * t)

    @staticmethod
    def rate_harmonic(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
        return qi / (1.0 + Di * t)

    @staticmethod
    def rate_hyperbolic(t: np.ndarray, qi: float, Di: float, b: float) -> np.ndarray:
        # Avoid zero or negative bases
        base = np.maximum(1e-9, 1.0 + b * Di * t)
        return qi / (base ** (1.0 / max(1e-4, b)))

    # -----------------------------------------------------------------------
    # Cumulative Production Formulations
    # -----------------------------------------------------------------------
    @staticmethod
    def cum_exponential(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
        return (qi - qi * np.exp(-Di * t)) / max(1e-9, Di)

    @staticmethod
    def cum_harmonic(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
        return (qi / max(1e-9, Di)) * np.log(np.maximum(1e-9, 1.0 + Di * t))

    @staticmethod
    def cum_hyperbolic(t: np.ndarray, qi: float, Di: float, b: float) -> np.ndarray:
        if abs(b - 1.0) < 1e-4:
            return ArpsDeclineCurve.cum_harmonic(t, qi, Di)
        if b < 1e-4:
            return ArpsDeclineCurve.cum_exponential(t, qi, Di)
        q_t = ArpsDeclineCurve.rate_hyperbolic(t, qi, Di, b)
        return (qi ** b / ((1.0 - b) * max(1e-9, Di))) * (qi ** (1.0 - b) - q_t ** (1.0 - b))

    # -----------------------------------------------------------------------
    # EUR & Time to Economic Limit
    # -----------------------------------------------------------------------
    def calculate_eur_and_time(
        self, model_type: str, qi: float, Di: float, b: float
    ) -> Tuple[float, float]:
        q_econ = self.economic_limit_rate
        if qi <= q_econ or Di <= 1e-9:
            return 0.0, 0.0

        if model_type == "EXPONENTIAL" or b < 1e-4:
            t_econ = np.log(qi / q_econ) / Di
            eur = (qi - q_econ) / Di
        elif model_type == "HARMONIC" or abs(b - 1.0) < 1e-4:
            t_econ = (qi / q_econ - 1.0) / Di
            eur = (qi / Di) * np.log(qi / q_econ)
        else:
            # Hyperbolic
            t_econ = (((qi / q_econ) ** b) - 1.0) / (b * Di)
            eur = (qi ** b / ((1.0 - b) * Di)) * (qi ** (1.0 - b) - q_econ ** (1.0 - b))

        return float(max(0.0, eur)), float(max(0.0, t_econ))

    # -----------------------------------------------------------------------
    # Fitting Methods
    # -----------------------------------------------------------------------
    def fit(
        self,
        time_days: np.ndarray,
        rates: np.ndarray,
        preferred_model: Optional[Literal["AUTO", "EXPONENTIAL", "HYPERBOLIC", "HARMONIC"]] = "AUTO",
    ) -> ArpsFitResult:
        """
        Fits decline curve models to (time, rate) series and selects optimal fit.
        """
        valid_mask = (~np.isnan(time_days)) & (~np.isnan(rates)) & (rates > 1e-3)
        t = np.array(time_days[valid_mask], dtype=float)
        q = np.array(rates[valid_mask], dtype=float)

        # Normalize time starting at t=0
        t_start = t[0] if len(t) > 0 else 0.0
        t_norm = t - t_start
        n = len(t_norm)

        if n < 5:
            # Fallback for sparse points
            q_mean = float(np.mean(q)) if n > 0 else 100.0
            return self._create_fallback_result(q_mean)

        # Approximate initial guesses
        qi_guess = float(np.mean(q[: min(5, n)]))
        q_end = float(np.mean(q[-min(5, n):]))
        t_max = float(t_norm[-1]) if t_norm[-1] > 0 else 1.0
        di_guess = max(1e-5, (qi_guess - q_end) / max(1.0, qi_guess * t_max))

        models_to_fit = ["EXPONENTIAL", "HARMONIC", "HYPERBOLIC"]
        if preferred_model and preferred_model != "AUTO":
            models_to_fit = [preferred_model]

        best_fit = None
        best_aic = float("inf")

        for m_type in models_to_fit:
            try:
                res = self._fit_single_model(t_norm, q, m_type, qi_guess, di_guess)
                if res is not None and res.aic < best_aic:
                    best_aic = res.aic
                    best_fit = res
            except Exception:
                continue

        if best_fit is None:
            # Linear log regression fallback
            best_fit = self._fit_exponential_log_linear(t_norm, q)

        return best_fit

    def _fit_single_model(
        self, t: np.ndarray, q: np.ndarray, model_type: str, qi_init: float, di_init: float
    ) -> Optional[ArpsFitResult]:
        n = len(t)

        if model_type == "EXPONENTIAL":
            popt, _ = curve_fit(
                self.rate_exponential,
                t,
                q,
                p0=[qi_init, di_init],
                bounds=([1e-3, 1e-6], [np.max(q) * 2.0, 0.1]),
                maxfev=3000,
            )
            qi, Di = float(popt[0]), float(popt[1])
            b = 0.0
            k_params = 2
            pred = self.rate_exponential(t, qi, Di)

        elif model_type == "HARMONIC":
            popt, _ = curve_fit(
                self.rate_harmonic,
                t,
                q,
                p0=[qi_init, di_init],
                bounds=([1e-3, 1e-6], [np.max(q) * 2.0, 0.1]),
                maxfev=3000,
            )
            qi, Di = float(popt[0]), float(popt[1])
            b = 1.0
            k_params = 2
            pred = self.rate_harmonic(t, qi, Di)

        else:  # HYPERBOLIC
            popt, _ = curve_fit(
                self.rate_hyperbolic,
                t,
                q,
                p0=[qi_init, di_init, 0.5],
                bounds=([1e-3, 1e-6, 0.01], [np.max(q) * 2.0, 0.1, 0.99]),
                maxfev=4000,
            )
            qi, Di, b = float(popt[0]), float(popt[1]), float(popt[2])
            k_params = 3
            pred = self.rate_hyperbolic(t, qi, Di, b)

        # Statistical goodness-of-fit
        residuals = q - pred
        rss = float(np.sum(residuals ** 2))
        tss = float(np.sum((q - np.mean(q)) ** 2))
        r2 = float(max(0.0, 1.0 - (rss / max(1e-9, tss))))
        rmse = float(np.sqrt(rss / max(1, n)))

        # Information Criteria
        sigma_sq = max(1e-9, rss / n)
        aic = float(2 * k_params + n * np.log(sigma_sq))
        bic = float(k_params * np.log(n) + n * np.log(sigma_sq))

        # EUR and reserves
        eur, t_econ = self.calculate_eur_and_time(model_type, qi, Di, b)
        cum_hist = float(np.trapezoid(q, t)) if hasattr(np, "trapezoid") else float(np.trapz(q, t))
        remaining_reserves = max(0.0, eur - cum_hist)
        ann_decline = float((1.0 - np.exp(-365.25 * Di)) * 100.0)

        # Forecast generation
        forecast_pts = self._generate_forecast(t[-1], qi, Di, b, model_type, cum_hist)

        return ArpsFitResult(
            model_type=model_type,  # type: ignore
            qi=round(qi, 2),
            Di=round(Di, 6),
            Di_annual_pct=round(ann_decline, 2),
            b=round(b, 4),
            r_squared=round(r2, 4),
            rmse=round(rmse, 2),
            aic=round(aic, 2),
            bic=round(bic, 2),
            economic_limit_rate=self.economic_limit_rate,
            time_to_economic_limit_days=round(t_econ, 1),
            eur_bbl=round(eur, 2),
            remaining_reserves_bbl=round(remaining_reserves, 2),
            cum_historical_production_bbl=round(cum_hist, 2),
            forecast=forecast_pts,
        )

    def _fit_exponential_log_linear(self, t: np.ndarray, q: np.ndarray) -> ArpsFitResult:
        """Robust linear regression on log rates as deterministic fallback."""
        log_q = np.log(np.maximum(1e-3, q))
        lin = stats.linregress(t, log_q)
        qi = float(np.exp(lin.intercept))
        Di = float(max(1e-6, -lin.slope))
        b = 0.0
        r2 = float(max(0.0, lin.rvalue ** 2)) if not np.isnan(lin.rvalue) else 0.0

        pred = self.rate_exponential(t, qi, Di)
        rss = float(np.sum((q - pred) ** 2))
        rmse = float(np.sqrt(rss / max(1, len(t))))
        aic = float(4.0 + len(t) * np.log(max(1e-9, rss / len(t))))
        bic = float(2.0 * np.log(len(t)) + len(t) * np.log(max(1e-9, rss / len(t))))

        eur, t_econ = self.calculate_eur_and_time("EXPONENTIAL", qi, Di, b)
        cum_hist = float(np.trapezoid(q, t)) if hasattr(np, "trapezoid") else float(np.trapz(q, t))
        remaining = max(0.0, eur - cum_hist)
        ann_decline = float((1.0 - np.exp(-365.25 * Di)) * 100.0)

        forecast_pts = self._generate_forecast(t[-1], qi, Di, b, "EXPONENTIAL", cum_hist)

        return ArpsFitResult(
            model_type="EXPONENTIAL",
            qi=round(qi, 2),
            Di=round(Di, 6),
            Di_annual_pct=round(ann_decline, 2),
            b=0.0,
            r_squared=round(r2, 4),
            rmse=round(rmse, 2),
            aic=round(aic, 2),
            bic=round(bic, 2),
            economic_limit_rate=self.economic_limit_rate,
            time_to_economic_limit_days=round(t_econ, 1),
            eur_bbl=round(eur, 2),
            remaining_reserves_bbl=round(remaining, 2),
            cum_historical_production_bbl=round(cum_hist, 2),
            forecast=forecast_pts,
        )

    def _generate_forecast(
        self,
        last_t: float,
        qi: float,
        Di: float,
        b: float,
        model_type: str,
        initial_cum: float,
    ) -> List[ArpsForecastPoint]:
        forecast_days = np.linspace(last_t, last_t + self.forecast_horizon_days, 13)[1:]
        pts: List[ArpsForecastPoint] = []
        for t_f in forecast_days:
            if model_type == "EXPONENTIAL":
                r_f = float(self.rate_exponential(np.array([t_f]), qi, Di)[0])
                c_f = float(self.cum_exponential(np.array([t_f]), qi, Di)[0])
            elif model_type == "HARMONIC":
                r_f = float(self.rate_harmonic(np.array([t_f]), qi, Di)[0])
                c_f = float(self.cum_harmonic(np.array([t_f]), qi, Di)[0])
            else:
                r_f = float(self.rate_hyperbolic(np.array([t_f]), qi, Di, b)[0])
                c_f = float(self.cum_hyperbolic(np.array([t_f]), qi, Di, b)[0])

            pts.append(
                ArpsForecastPoint(
                    time_days=round(float(t_f), 1),
                    rate_bopd=round(max(0.0, r_f), 2),
                    cum_production_bbl=round(max(0.0, c_f), 2),
                )
            )
        return pts

    def _create_fallback_result(self, default_rate: float) -> ArpsFitResult:
        return ArpsFitResult(
            model_type="EXPONENTIAL",
            qi=round(default_rate, 2),
            Di=0.0005,
            Di_annual_pct=16.7,
            b=0.0,
            r_squared=0.0,
            rmse=0.0,
            aic=0.0,
            bic=0.0,
            economic_limit_rate=self.economic_limit_rate,
            time_to_economic_limit_days=365.0,
            eur_bbl=round(default_rate * 365.0, 2),
            remaining_reserves_bbl=round(default_rate * 365.0, 2),
            cum_historical_production_bbl=0.0,
            forecast=[],
        )
