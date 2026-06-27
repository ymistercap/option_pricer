"""
SABR Stochastic Alpha-Beta-Rho Model.

The SABR model (Hagan et al., 2002) describes the forward price F and
its stochastic volatility α:

    dF = α·F^β dW_1
    dα = ν·α dW_2
    dW_1·dW_2 = ρ dt

Parameters:
    α (alpha) — initial volatility level
    β (beta)  — elasticity (0 ≤ β ≤ 1); β=1 is log-normal, β=0 is normal
    ρ (rho)   — correlation between forward and vol (-1 < ρ < 1)
    ν (nu)    — vol-of-vol

Hagan's (2002) asymptotic formula gives an implied BS volatility σ_B(K, F)
that is accurate for short maturities and away from β=0.

Calibration: given market implied vols at several strikes, fit (α, ρ, ν)
for a fixed β (often β=0.5 or β=1) by least squares.

References:
    Hagan, P. et al. (2002). "Managing Smile Risk."
    Wilmott Magazine, July, 84-108.
"""

import numpy as np
from scipy.optimize import minimize


def sabr_implied_vol(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """
    Compute the SABR implied Black volatility using Hagan's formula.

    Parameters
    ----------
    F : float
        Forward price.
    K : float
        Strike price.
    T : float
        Time to maturity.
    alpha : float
        Initial vol level (α > 0).
    beta : float
        Elasticity parameter (0 ≤ β ≤ 1).
    rho : float
        Correlation (-1 < ρ < 1).
    nu : float
        Vol-of-vol (ν ≥ 0).

    Returns
    -------
    float
        Implied Black volatility.
    """
    if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:
        return alpha

    # ATM case
    if abs(F - K) < 1e-12:
        FK_mid = F
        logFK = 0.0
    else:
        FK_mid = (F * K) ** ((1.0 - beta) / 2.0)
        logFK = np.log(F / K)

    # ATM formula
    if abs(F - K) < 1e-12:
        term1 = alpha / (F ** (1.0 - beta))
        correction = 1.0 + (
            ((1.0 - beta) ** 2 / 24.0) * alpha**2 / (F ** (2.0 - 2.0 * beta))
            + 0.25 * rho * beta * nu * alpha / (F ** (1.0 - beta))
            + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
        ) * T
        return float(term1 * correction)

    # General case
    z = (nu / alpha) * FK_mid * logFK
    x_z = np.log((np.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))

    if abs(x_z) < 1e-12:
        zeta_over_x = 1.0
    else:
        zeta_over_x = z / x_z

    prefix = alpha / (
        FK_mid * (1.0 + (1.0 - beta) ** 2 / 24.0 * logFK**2
                  + (1.0 - beta) ** 4 / 1920.0 * logFK**4)
    )

    correction = 1.0 + (
        ((1.0 - beta) ** 2 / 24.0) * alpha**2 / (FK_mid**2)
        + 0.25 * rho * beta * nu * alpha / FK_mid
        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    ) * T

    return float(prefix * zeta_over_x * correction)


def sabr_smile(
    F: float,
    strikes: np.ndarray,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    """
    Compute the SABR implied vol smile across multiple strikes.

    Parameters
    ----------
    F : float
        Forward price.
    strikes : np.ndarray
        Array of strike prices.
    T : float
        Time to maturity.
    alpha, beta, rho, nu : float
        SABR parameters.

    Returns
    -------
    np.ndarray
        Array of implied volatilities.
    """
    return np.array([
        sabr_implied_vol(F, K, T, alpha, beta, rho, nu) for K in strikes
    ])


def sabr_calibrate(
    F: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    T: float,
    beta: float = 0.5,
) -> dict[str, float]:
    """
    Calibrate SABR parameters (α, ρ, ν) to market implied vols.

    Minimises the sum of squared differences between Hagan's formula
    and market vols for a fixed β.

    Parameters
    ----------
    F : float
        Forward price.
    strikes : np.ndarray
        Array of strike prices.
    market_vols : np.ndarray
        Observed implied volatilities at each strike.
    T : float
        Time to maturity.
    beta : float
        Fixed elasticity parameter.

    Returns
    -------
    dict
        Calibrated parameters: ``{'alpha', 'beta', 'rho', 'nu', 'rmse'}``.
    """
    def objective(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu < 0 or abs(rho) >= 1:
            return 1e10
        model_vols = sabr_smile(F, strikes, T, alpha, beta, rho, nu)
        return float(np.sum((model_vols - market_vols) ** 2))

    # Initial guess: alpha from ATM, rho=0, nu=0.3
    atm_idx = np.argmin(np.abs(strikes - F))
    alpha0 = market_vols[atm_idx] * F ** (1.0 - beta)

    result = minimize(
        objective,
        x0=[alpha0, -0.1, 0.3],
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10},
    )

    alpha_cal, rho_cal, nu_cal = result.x
    model_vols = sabr_smile(F, strikes, T, alpha_cal, beta, rho_cal, nu_cal)
    rmse = float(np.sqrt(np.mean((model_vols - market_vols) ** 2)))

    return {
        "alpha": float(alpha_cal),
        "beta": beta,
        "rho": float(np.clip(rho_cal, -0.9999, 0.9999)),
        "nu": float(max(nu_cal, 0.0)),
        "rmse": rmse,
    }
