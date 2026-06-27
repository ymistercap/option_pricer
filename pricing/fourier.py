"""
Carr-Madan FFT Option Pricing.

The Carr-Madan (1999) method prices European options across a range
of strikes simultaneously using the Fast Fourier Transform.

The call price for log-strike k = ln(K) is:

    C(k) = (e^{-αk}/π) · ∫₀^∞ e^{-ivk} ψ(v) dv

where:
    ψ(v) = e^{-rT} φ(v-(α+1)i) / (α² + α - v² + i(2α+1)v)

and φ(u) is the characteristic function of ln(S_T).

The FFT discretises this integral on a grid of N points with spacing η,
producing N call prices at log-strikes k_j = -b + λ·j where λ·η = 2π/N.

The damping factor α > 0 ensures integrability; α ≈ 1.5 is standard.

This method prices O(N) strikes in O(N·log N) time, making it ideal for
calibration where many strikes must be priced per evaluation.

References:
    Carr, P. & Madan, D. (1999). "Option Valuation Using the Fast
    Fourier Transform." J. Computational Finance, 2(4), 61-73.
"""

from typing import Callable

import numpy as np


def _bs_char_func(u: complex, T: float, r: float, log_S: float, sigma: float) -> complex:
    """Characteristic function of ln(S_T) under BS (GBM)."""
    iu = 1j * u
    return np.exp(iu * (log_S + (r - 0.5 * sigma**2) * T) - 0.5 * sigma**2 * u**2 * T)


def carr_madan_fft(
    S: float,
    T: float,
    r: float,
    char_func: Callable[[complex], complex],
    N: int = 4096,
    alpha: float = 1.5,
    eta: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute call prices across a range of strikes via the Carr-Madan FFT.

    Parameters
    ----------
    S : float
        Spot price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    char_func : Callable
        Characteristic function φ(u) of ln(S_T). Must accept a complex
        argument and return a complex value.
    N : int
        FFT grid size (must be a power of 2).
    alpha : float
        Damping factor (α > 0, typically 1.5).
    eta : float
        Spacing in the frequency domain.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (strikes, call_prices) — arrays of length N.
    """
    lam = 2.0 * np.pi / (N * eta)
    b = N * lam / 2.0

    # Frequency grid
    v = np.arange(N) * eta

    # Build the integrand ψ(v)
    # ψ(v) = exp(-rT) * φ(v - (α+1)i) / (α² + α - v² + i(2α+1)v)
    u_shifted = v - (alpha + 1.0) * 1j

    cf_vals = np.array([char_func(u_shifted[j]) for j in range(N)])

    denom = alpha**2 + alpha - v**2 + 1j * (2.0 * alpha + 1.0) * v
    psi = np.exp(-r * T) * cf_vals / denom

    # Simpson's rule weights
    simpson = 3.0 + (-1.0) ** (np.arange(N) + 1)
    simpson[0] = 1.0
    simpson *= eta / 3.0

    # FFT input
    x = np.exp(1j * v * b) * psi * simpson

    fft_result = np.fft.fft(x)

    # Log-strikes and call prices
    k = -b + lam * np.arange(N)
    call_prices = np.real(np.exp(-alpha * k) / np.pi * fft_result)

    strikes = np.exp(k)
    return strikes, call_prices


def carr_madan_bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    N: int = 4096,
    alpha: float = 1.5,
    eta: float = 0.25,
) -> float:
    """
    Price a single European option using Carr-Madan FFT with BS char func.

    Interpolates the FFT grid to find the price at the requested strike.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    option_type : str
        ``'call'`` or ``'put'``.
    N : int
        FFT grid size.
    alpha : float
        Damping factor.
    eta : float
        Frequency spacing.

    Returns
    -------
    float
        Option price.
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    log_S = np.log(S)

    def char_func(u):
        return _bs_char_func(u, T, r, log_S, sigma)

    strikes, call_prices = carr_madan_fft(S, T, r, char_func, N, alpha, eta)

    # Interpolate to find price at K
    call_price = float(np.interp(K, strikes, call_prices))
    call_price = max(call_price, 0.0)

    if option_type == "call":
        return call_price
    else:
        return max(call_price - S + K * np.exp(-r * T), 0.0)


def carr_madan_heston_price(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    option_type: str = "call",
    N: int = 4096,
    alpha: float = 1.5,
    eta: float = 0.25,
) -> float:
    """
    Price a European option using Carr-Madan FFT with Heston char func.

    Parameters
    ----------
    S, K, T, r : float
        Standard option parameters.
    v0, kappa, theta, xi, rho : float
        Heston model parameters.
    option_type : str
        ``'call'`` or ``'put'``.
    N, alpha, eta : float
        FFT parameters.

    Returns
    -------
    float
        Option price.
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    from pricing.heston import heston_char_func_external

    log_S = np.log(S)

    def char_func(u):
        return heston_char_func_external(u, T, r, log_S, v0, kappa, theta, xi, rho)

    strikes, call_prices = carr_madan_fft(S, T, r, char_func, N, alpha, eta)

    call_price = float(np.interp(K, strikes, call_prices))
    call_price = max(call_price, 0.0)

    if option_type == "call":
        return call_price
    else:
        return max(call_price - S + K * np.exp(-r * T), 0.0)
