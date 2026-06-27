"""
Numerical Greeks via Bump-and-Reprice (Finite Differences).

This module computes option Greeks by numerically differentiating the pricing
function — i.e., bumping each parameter by a small amount and observing the
change in price. This approach:

    1. Works with ANY pricing function (BS, MC, FD, exotics).
    2. Does not require closed-form derivatives.
    3. Is the standard industry approach for complex products.

We use central differences for improved accuracy:
    ∂V/∂x ≈ [V(x+h) - V(x-h)] / (2h)        — O(h²) error
    ∂²V/∂x² ≈ [V(x+h) - 2V(x) + V(x-h)] / h² — O(h²) error

The bump size h should be small enough for accuracy but large enough to
avoid floating-point cancellation. Typical values:
    - Spot (delta/gamma): h = 0.01 * S (1% of spot)
    - Volatility (vega): h = 0.01 (1 vol point)
    - Time (theta): h = 1/252 (one trading day)
    - Rate (rho): h = 0.0001 (1 basis point)
"""

from typing import Callable

from pricing.black_scholes import bs_price


def delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    pricer: Callable[..., float] = bs_price,
    h: float | None = None,
) -> float:
    """
    Numerical Delta via central differences: ∂V/∂S ≈ [V(S+h) - V(S-h)] / (2h).

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard option parameters.
    option_type : str
        'call' or 'put'.
    pricer : callable
        Pricing function with signature pricer(S, K, T, r, sigma, option_type).
    h : float, optional
        Bump size. Defaults to 0.01 * S.

    Returns
    -------
    float
        Numerical delta estimate.
    """
    if h is None:
        h = max(S * 0.01, 0.01)
    v_up = pricer(S + h, K, T, r, sigma, option_type)
    v_down = pricer(S - h, K, T, r, sigma, option_type)
    return (v_up - v_down) / (2.0 * h)


def gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    pricer: Callable[..., float] = bs_price,
    h: float | None = None,
) -> float:
    """
    Numerical Gamma via central differences: ∂²V/∂S² ≈ [V(S+h) - 2V(S) + V(S-h)] / h².

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard option parameters.
    option_type : str
        'call' or 'put'.
    pricer : callable
        Pricing function with signature pricer(S, K, T, r, sigma, option_type).
    h : float, optional
        Bump size. Defaults to 0.01 * S.

    Returns
    -------
    float
        Numerical gamma estimate.
    """
    if h is None:
        h = max(S * 0.01, 0.01)
    v_up = pricer(S + h, K, T, r, sigma, option_type)
    v_mid = pricer(S, K, T, r, sigma, option_type)
    v_down = pricer(S - h, K, T, r, sigma, option_type)
    return (v_up - 2.0 * v_mid + v_down) / (h * h)


def vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    pricer: Callable[..., float] = bs_price,
    h: float = 0.01,
) -> float:
    """
    Numerical Vega via central differences: ∂V/∂σ ≈ [V(σ+h) - V(σ-h)] / (2h).

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard option parameters.
    option_type : str
        'call' or 'put'.
    pricer : callable
        Pricing function.
    h : float
        Volatility bump size (default 0.01 = 1 vol point).

    Returns
    -------
    float
        Numerical vega estimate.
    """
    v_up = pricer(S, K, T, r, sigma + h, option_type)
    v_down = pricer(S, K, T, r, sigma - h, option_type)
    return (v_up - v_down) / (2.0 * h)


def theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    pricer: Callable[..., float] = bs_price,
    h: float = 1.0 / 252.0,
) -> float:
    """
    Numerical Theta via forward difference: ∂V/∂t ≈ [V(T-h) - V(T)] / h.

    Note: Theta = ∂V/∂t, and since T decreases as time passes, we compute
    the price at T-h and compare to the current price.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard option parameters.
    option_type : str
        'call' or 'put'.
    pricer : callable
        Pricing function.
    h : float
        Time bump in years (default 1/252 ≈ 1 trading day).

    Returns
    -------
    float
        Numerical theta estimate (per year). For daily, this already uses 1/252.
    """
    if T <= h:
        v_now = pricer(S, K, T, r, sigma, option_type)
        v_exp = pricer(S, K, 1e-10, r, sigma, option_type)
        return (v_exp - v_now) / T if T > 0 else 0.0

    v_now = pricer(S, K, T, r, sigma, option_type)
    v_later = pricer(S, K, T - h, r, sigma, option_type)
    return (v_later - v_now) / h


def rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    pricer: Callable[..., float] = bs_price,
    h: float = 0.0001,
) -> float:
    """
    Numerical Rho via central differences: ∂V/∂r ≈ [V(r+h) - V(r-h)] / (2h).

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard option parameters.
    option_type : str
        'call' or 'put'.
    pricer : callable
        Pricing function.
    h : float
        Rate bump (default 0.0001 = 1 basis point).

    Returns
    -------
    float
        Numerical rho estimate.
    """
    v_up = pricer(S, K, T, r + h, sigma, option_type)
    v_down = pricer(S, K, T, r - h, sigma, option_type)
    return (v_up - v_down) / (2.0 * h)


def vanna(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    pricer: Callable[..., float] = bs_price,
    h_s: float | None = None,
    h_sigma: float = 0.01,
) -> float:
    """
    Numerical Vanna: ∂²V/∂S∂σ via cross finite differences.

    Vanna ≈ [V(S+h,σ+k) - V(S+h,σ-k) - V(S-h,σ+k) + V(S-h,σ-k)] / (4·h·k)

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard option parameters.
    option_type : str
        'call' or 'put'.
    pricer : callable
        Pricing function.
    h_s : float, optional
        Spot bump. Defaults to 0.01 * S.
    h_sigma : float
        Vol bump (default 0.01).

    Returns
    -------
    float
        Numerical vanna estimate.
    """
    if h_s is None:
        h_s = max(S * 0.01, 0.01)
    v_pp = pricer(S + h_s, K, T, r, sigma + h_sigma, option_type)
    v_pm = pricer(S + h_s, K, T, r, sigma - h_sigma, option_type)
    v_mp = pricer(S - h_s, K, T, r, sigma + h_sigma, option_type)
    v_mm = pricer(S - h_s, K, T, r, sigma - h_sigma, option_type)
    return (v_pp - v_pm - v_mp + v_mm) / (4.0 * h_s * h_sigma)


def volga(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    pricer: Callable[..., float] = bs_price,
    h: float = 0.01,
) -> float:
    """
    Numerical Volga (Vomma): ∂²V/∂σ² ≈ [V(σ+h) - 2V(σ) + V(σ-h)] / h².

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard option parameters.
    option_type : str
        'call' or 'put'.
    pricer : callable
        Pricing function.
    h : float
        Vol bump (default 0.01).

    Returns
    -------
    float
        Numerical volga estimate.
    """
    v_up = pricer(S, K, T, r, sigma + h, option_type)
    v_mid = pricer(S, K, T, r, sigma, option_type)
    v_down = pricer(S, K, T, r, sigma - h, option_type)
    return (v_up - 2.0 * v_mid + v_down) / (h * h)
