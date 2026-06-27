"""
Model Calibration Engine.

Calibrates Heston and Merton model parameters to market implied
volatilities using global optimisation (differential evolution)
followed by local refinement (Nelder-Mead / L-BFGS-B).

The objective function minimises the root-mean-square error (RMSE)
between model-implied and market-implied volatilities:

    min_θ  √( (1/N) Σ (σ_model(K_i,T_i; θ) - σ_market(K_i,T_i))² )

For Heston, we use the semi-analytical pricer + Newton IV inversion.
For Merton, we use the analytical series + Newton IV inversion.

References:
    Cont, R. & Tankov, P. (2004). "Financial Modelling With Jump
    Processes." Chapter 13: Model Calibration.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize

from pricing.heston import heston_price
from pricing.jump_diffusion import merton_price
from volatility.implied_vol import implied_vol


@dataclass
class CalibrationResult:
    """Container for calibration output.

    Attributes
    ----------
    params : dict[str, float]
        Calibrated model parameters.
    rmse : float
        Root-mean-square implied vol error.
    model_vols : np.ndarray
        Model-implied volatilities at calibration points.
    success : bool
        Whether optimisation converged.
    """

    params: dict[str, float]
    rmse: float
    model_vols: np.ndarray
    success: bool


def _price_to_iv(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
) -> float:
    """Convert an option price to implied vol, returning NaN on failure."""
    try:
        return implied_vol(price, S, K, T, r, option_type)
    except Exception:
        return np.nan


def calibrate_heston(
    S: float,
    r: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    market_vols: np.ndarray,
    option_types: np.ndarray | None = None,
) -> CalibrationResult:
    """
    Calibrate the Heston model to market implied volatilities.

    Parameters
    ----------
    S : float
        Spot price.
    r : float
        Risk-free rate.
    strikes : np.ndarray
        1D array of strikes.
    maturities : np.ndarray
        1D array of maturities (same length as strikes).
    market_vols : np.ndarray
        1D array of market implied vols.
    option_types : np.ndarray or None
        Array of ``'call'``/``'put'`` strings. Defaults to all calls.

    Returns
    -------
    CalibrationResult
        Calibrated Heston parameters and fit quality.
    """
    n = len(strikes)
    if option_types is None:
        option_types = np.array(["call"] * n)

    # Parameter bounds: [v0, kappa, theta, xi, rho]
    bounds = [
        (0.001, 1.0),    # v0
        (0.01, 10.0),    # kappa
        (0.001, 1.0),    # theta
        (0.01, 2.0),     # xi
        (-0.999, 0.999), # rho
    ]

    def objective(params):
        v0, kappa, theta, xi, rho = params
        total_sq = 0.0
        count = 0
        for i in range(n):
            try:
                price = heston_price(
                    S, strikes[i], maturities[i], r,
                    v0, kappa, theta, xi, rho, option_types[i],
                )
                model_iv = _price_to_iv(price, S, strikes[i], maturities[i], r, option_types[i])
                if not np.isnan(model_iv):
                    total_sq += (model_iv - market_vols[i]) ** 2
                    count += 1
                else:
                    total_sq += 1.0
                    count += 1
            except Exception:
                total_sq += 1.0
                count += 1
        return total_sq / max(count, 1)

    # Global search
    result = differential_evolution(
        objective, bounds, seed=42, maxiter=30, popsize=10, tol=1e-6, polish=False,
    )

    # Local refinement
    refined = minimize(
        objective, result.x, method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-8},
    )

    best = refined.x if refined.fun < result.fun else result.x
    v0, kappa, theta, xi, rho = best

    # Compute final model vols
    model_vols = np.full(n, np.nan)
    for i in range(n):
        try:
            price = heston_price(S, strikes[i], maturities[i], r, v0, kappa, theta, xi, rho, option_types[i])
            model_vols[i] = _price_to_iv(price, S, strikes[i], maturities[i], r, option_types[i])
        except Exception:
            pass

    valid = ~np.isnan(model_vols)
    rmse = float(np.sqrt(np.mean((model_vols[valid] - market_vols[valid]) ** 2))) if valid.any() else 1.0

    return CalibrationResult(
        params={"v0": float(v0), "kappa": float(kappa), "theta": float(theta), "xi": float(xi), "rho": float(rho)},
        rmse=rmse,
        model_vols=model_vols,
        success=rmse < 0.05,
    )


def calibrate_merton(
    S: float,
    r: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    market_vols: np.ndarray,
    option_types: np.ndarray | None = None,
) -> CalibrationResult:
    """
    Calibrate the Merton jump-diffusion model to market implied vols.

    Parameters
    ----------
    S : float
        Spot price.
    r : float
        Risk-free rate.
    strikes : np.ndarray
        1D array of strikes.
    maturities : np.ndarray
        1D array of maturities.
    market_vols : np.ndarray
        1D array of market implied vols.
    option_types : np.ndarray or None
        Array of option types. Defaults to all calls.

    Returns
    -------
    CalibrationResult
        Calibrated Merton parameters and fit quality.
    """
    n = len(strikes)
    if option_types is None:
        option_types = np.array(["call"] * n)

    # Parameter bounds: [sigma, lam, m, delta]
    bounds = [
        (0.01, 1.0),     # sigma (diffusion vol)
        (0.0, 5.0),      # lam (jump intensity)
        (-0.5, 0.1),     # m (mean log-jump)
        (0.01, 0.5),     # delta (std log-jump)
    ]

    def objective(params):
        sigma, lam, m, delta = params
        total_sq = 0.0
        count = 0
        for i in range(n):
            try:
                price = merton_price(
                    S, strikes[i], maturities[i], r,
                    sigma, lam, m, delta, option_types[i],
                )
                model_iv = _price_to_iv(price, S, strikes[i], maturities[i], r, option_types[i])
                if not np.isnan(model_iv):
                    total_sq += (model_iv - market_vols[i]) ** 2
                    count += 1
                else:
                    total_sq += 1.0
                    count += 1
            except Exception:
                total_sq += 1.0
                count += 1
        return total_sq / max(count, 1)

    result = differential_evolution(
        objective, bounds, seed=42, maxiter=30, popsize=10, tol=1e-6, polish=False,
    )

    refined = minimize(
        objective, result.x, method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-8},
    )

    best = refined.x if refined.fun < result.fun else result.x
    sigma, lam, m, delta = best

    model_vols = np.full(n, np.nan)
    for i in range(n):
        try:
            price = merton_price(S, strikes[i], maturities[i], r, sigma, lam, m, delta, option_types[i])
            model_vols[i] = _price_to_iv(price, S, strikes[i], maturities[i], r, option_types[i])
        except Exception:
            pass

    valid = ~np.isnan(model_vols)
    rmse = float(np.sqrt(np.mean((model_vols[valid] - market_vols[valid]) ** 2))) if valid.any() else 1.0

    return CalibrationResult(
        params={"sigma": float(sigma), "lam": float(lam), "m": float(m), "delta": float(delta)},
        rmse=rmse,
        model_vols=model_vols,
        success=rmse < 0.05,
    )
