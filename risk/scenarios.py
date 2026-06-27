"""
Stress Testing and Scenario Analysis Module.

Stress testing evaluates how option prices and Greeks change under
extreme market conditions. This is crucial for risk management as
VaR and normal-conditions analysis may underestimate tail risks.

Scenarios include:
    - Spot shocks: ±10%, ±20% moves in the underlying
    - Volatility shocks: ±5, ±10 vol points
    - Rate shocks: ±50bps, ±100bps
    - Combined scenarios: simultaneous spot and vol moves (crash scenario)

References:
    Hull, J. (2018). "Options, Futures, and Other Derivatives." Ch. 24.
"""

import numpy as np
import pandas as pd

from pricing.black_scholes import bs_price
from greeks.analytical import delta, gamma, vega, theta, rho


def spot_stress(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    shocks: list[float] | None = None,
) -> pd.DataFrame:
    """
    Stress test: impact of spot price shocks on option price and Greeks.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Base-case parameters.
    option_type : str
        'call' or 'put'.
    shocks : list of float
        Relative spot shocks (e.g., [-0.20, -0.10, 0, 0.10, 0.20]).

    Returns
    -------
    pd.DataFrame
        Impact table.
    """
    if shocks is None:
        shocks = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]

    results = []
    for shock in shocks:
        S_new = S * (1 + shock)
        results.append({
            "shock_pct": shock * 100,
            "spot": S_new,
            "price": bs_price(S_new, K, T, r, sigma, option_type),
            "delta": delta(S_new, K, T, r, sigma, option_type),
            "gamma": gamma(S_new, K, T, r, sigma),
            "vega": vega(S_new, K, T, r, sigma),
            "theta": theta(S_new, K, T, r, sigma, option_type),
        })

    return pd.DataFrame(results)


def vol_stress(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    shocks_pts: list[float] | None = None,
) -> pd.DataFrame:
    """
    Stress test: impact of volatility shocks.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Base-case parameters.
    option_type : str
        'call' or 'put'.
    shocks_pts : list of float
        Absolute vol shocks in points (e.g., [-0.10, -0.05, 0, 0.05, 0.10]).

    Returns
    -------
    pd.DataFrame
        Impact table.
    """
    if shocks_pts is None:
        shocks_pts = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]

    results = []
    for shock in shocks_pts:
        sigma_new = max(sigma + shock, 0.001)
        results.append({
            "vol_shock_pts": shock * 100,
            "sigma": sigma_new,
            "price": bs_price(S, K, T, r, sigma_new, option_type),
            "delta": delta(S, K, T, r, sigma_new, option_type),
            "vega": vega(S, K, T, r, sigma_new),
        })

    return pd.DataFrame(results)


def rate_stress(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    shocks_bps: list[float] | None = None,
) -> pd.DataFrame:
    """
    Stress test: impact of interest rate shocks.

    Parameters
    ----------
    shocks_bps : list of float
        Rate shocks in basis points (e.g., [-100, -50, 0, 50, 100]).
    """
    if shocks_bps is None:
        shocks_bps = [-100, -50, 0, 50, 100]

    results = []
    for shock in shocks_bps:
        r_new = r + shock / 10000.0
        results.append({
            "rate_shock_bps": shock,
            "rate": r_new,
            "price": bs_price(S, K, T, r_new, sigma, option_type),
            "rho": rho(S, K, T, r_new, sigma, option_type),
        })

    return pd.DataFrame(results)


def scenario_matrix(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    spot_shocks: list[float] | None = None,
    vol_shocks: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 2D scenario matrix: spot × vol → option price.

    Useful for heatmap visualization.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Base-case parameters.
    option_type : str
        'call' or 'put'.
    spot_shocks : list of float
        Relative spot shocks.
    vol_shocks : list of float
        Absolute vol shocks.

    Returns
    -------
    tuple of (spot_levels, vol_levels, price_matrix)
        spot_levels : np.ndarray, shape (n_spot,)
        vol_levels : np.ndarray, shape (n_vol,)
        price_matrix : np.ndarray, shape (n_vol, n_spot)
    """
    if spot_shocks is None:
        spot_shocks = [-0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20]
    if vol_shocks is None:
        vol_shocks = [-0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10]

    spot_levels = np.array([S * (1 + sh) for sh in spot_shocks])
    vol_levels = np.array([max(sigma + sh, 0.001) for sh in vol_shocks])

    price_matrix = np.zeros((len(vol_levels), len(spot_levels)))
    for i, v in enumerate(vol_levels):
        for j, s in enumerate(spot_levels):
            price_matrix[i, j] = bs_price(s, K, T, r, v, option_type)

    return spot_levels, vol_levels, price_matrix
