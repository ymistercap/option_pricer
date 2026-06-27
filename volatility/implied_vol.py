"""
Implied Volatility Computation Module.

The implied volatility σ_imp is the unique value of σ such that:
    BS(S, K, T, r, σ_imp) = V_market

Uniqueness is guaranteed because the BS price is strictly monotonically
increasing in σ (since Vega = ∂V/∂σ > 0 for T > 0).

Methods:
    1. Newton-Raphson: Uses Vega as the derivative for quadratic convergence:
       σ_{n+1} = σ_n - [BS(σ_n) - V_market] / Vega(σ_n)

       Convergence is quadratic near the root, but may diverge if the initial
       guess is poor or Vega is near zero (deep ITM/OTM, near expiry).

    2. Brent's method (fallback): Bisection-based root finder from scipy.
       Guaranteed to converge on a bracketed interval. Slower but robust.

Initial Guess:
    Brenner-Subrahmanyam (1988) approximation for ATM options:
        σ₀ ≈ √(2π/T) · C/S

    This provides a good starting point for Newton-Raphson.

References:
    Brenner, M. & Subrahmanyam, M. (1988). "A Simple Formula to Compute
    the Implied Standard Deviation."
"""

import numpy as np
from scipy.optimize import brentq

from config.settings import IV_MAX_ITER, IV_TOL, IV_VOL_LOWER, IV_VOL_UPPER
from greeks.analytical import vega as bs_vega
from pricing.black_scholes import bs_price


def _brenner_subrahmanyam(market_price: float, S: float, T: float) -> float:
    """
    Brenner-Subrahmanyam initial guess for implied volatility.

    σ₀ ≈ √(2π/T) · C/S

    This is exact for ATM options and provides a reasonable starting
    point for near-the-money options.
    """
    if T <= 0 or S <= 0:
        return 0.2  # default fallback
    return np.sqrt(2.0 * np.pi / T) * market_price / S


def implied_vol_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    max_iter: int = IV_MAX_ITER,
    tol: float = IV_TOL,
    sigma_init: float | None = None,
) -> float | None:
    """
    Compute implied volatility using Newton-Raphson iteration.

    σ_{n+1} = σ_n - [BS(σ_n) - market_price] / Vega(σ_n)

    Convergence is quadratic when the initial guess is reasonable.

    Parameters
    ----------
    market_price : float
        Observed market price of the option.
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate.
    option_type : str
        'call' or 'put'.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on price difference.
    sigma_init : float, optional
        Initial volatility guess. Defaults to Brenner-Subrahmanyam.

    Returns
    -------
    float or None
        The implied volatility, or None if convergence fails.
    """
    if T <= 0:
        return None

    sigma = sigma_init if sigma_init is not None else _brenner_subrahmanyam(market_price, S, T)
    sigma = max(sigma, IV_VOL_LOWER)

    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        v = bs_vega(S, K, T, r, sigma)
        if v < 1e-15:
            return None  # Vega too small, NR will diverge

        sigma -= diff / v
        sigma = max(sigma, IV_VOL_LOWER)

        if sigma > IV_VOL_UPPER:
            return None

    return None  # Did not converge


def implied_vol_brent(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    vol_lower: float = IV_VOL_LOWER,
    vol_upper: float = IV_VOL_UPPER,
    tol: float = IV_TOL,
) -> float | None:
    """
    Compute implied volatility using Brent's method (scipy.optimize.brentq).

    Brent's method combines bisection, secant, and inverse quadratic interpolation.
    It is guaranteed to converge if the root is bracketed, which is always the
    case for valid market prices (since BS is monotonic in σ).

    Parameters
    ----------
    market_price : float
        Observed market price.
    S, K, T, r : float
        Standard BS parameters.
    option_type : str
        'call' or 'put'.
    vol_lower, vol_upper : float
        Bracketing interval for volatility.
    tol : float
        Convergence tolerance.

    Returns
    -------
    float or None
        The implied volatility, or None if not bracketed.
    """
    if T <= 0:
        return None

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    # Check bracketing
    f_lower = objective(vol_lower)
    f_upper = objective(vol_upper)

    if f_lower * f_upper > 0:
        return None  # Root not bracketed

    try:
        return brentq(objective, vol_lower, vol_upper, xtol=tol)
    except ValueError:
        return None


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
) -> float | None:
    """
    Compute implied volatility with Newton-Raphson (primary) and Brent (fallback).

    Parameters
    ----------
    market_price : float
        Observed market price.
    S, K, T, r : float
        Standard BS parameters.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float or None
        The implied volatility, or None if computation fails.

    Examples
    --------
    >>> from pricing.black_scholes import bs_call_price
    >>> price = bs_call_price(100, 100, 1.0, 0.05, 0.30)
    >>> iv = implied_vol(price, 100, 100, 1.0, 0.05, 'call')
    >>> abs(iv - 0.30) < 1e-6
    True
    """
    # Validate: price must be above intrinsic value
    if option_type == "call":
        intrinsic = max(S - K * np.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0.0)

    if market_price < intrinsic - 1e-10:
        return None  # Price below intrinsic — no valid IV

    if market_price < 1e-12:
        return None

    # Try Newton-Raphson first (fast, quadratic convergence)
    result = implied_vol_newton(market_price, S, K, T, r, option_type)
    if result is not None:
        return result

    # Fallback to Brent (robust, guaranteed convergence)
    return implied_vol_brent(market_price, S, K, T, r, option_type)
