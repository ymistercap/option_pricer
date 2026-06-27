"""
Heston (1993) Stochastic Volatility Option Pricing.

The Heston model replaces constant volatility with a mean-reverting
CIR process:

    dS  = r·S dt + √v·S dW_S
    dv  = κ(θ - v) dt + ξ√v dW_v
    dW_S·dW_v = ρ dt

Parameters:
    v0  — initial variance
    κ   — mean-reversion speed
    θ   — long-run variance
    ξ   — vol-of-vol (volatility of variance)
    ρ   — correlation between spot and variance Brownians

The semi-analytical price uses the Gil-Pelaez inversion:

    C = S·P₁ - K·e^{-rT}·P₂

where P_j = ½ + (1/π) ∫₀^∞ Re[e^{-iu·ln K} f_j(u; ln S, v0, T) / (iu)] du

Uses the Albrecher et al. formulation to avoid branch-cut
discontinuities ("the little Heston trap").

References:
    Heston, S.L. (1993). "A Closed-Form Solution for Options with
    Stochastic Volatility." Review of Financial Studies, 6(2), 327-343.
    Albrecher, H. et al. (2007). "The little Heston trap."
"""

import numpy as np
from scipy.integrate import quad

from pricing.monte_carlo import MCResult


def _heston_cf_j(
    u: complex,
    T: float,
    r: float,
    log_S: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    j: int,
) -> complex:
    """
    Heston characteristic function f_j for measure j (j=1 or j=2).

    Uses the Albrecher et al. formulation (little Heston trap fix).

    j=1: stock-price numeraire measure  (b₁ = κ - ρξ,  u₁ = +½)
    j=2: risk-neutral measure           (b₂ = κ,       u₂ = -½)
    """
    if j == 1:
        b = kappa - rho * xi
        uj = 0.5
    else:
        b = kappa
        uj = -0.5

    iu = 1j * u
    a = kappa * theta

    d = np.sqrt((rho * xi * iu - b) ** 2 - xi**2 * (2.0 * uj * iu - u**2))

    # Albrecher formulation: g uses (b - ρξiu - d) in numerator
    g = (b - rho * xi * iu - d) / (b - rho * xi * iu + d)

    exp_dT = np.exp(-d * T)

    C = (r * iu * T
         + (a / xi**2) * ((b - rho * xi * iu - d) * T
                          - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))))

    D = ((b - rho * xi * iu - d) / xi**2) * ((1.0 - exp_dT) / (1.0 - g * exp_dT))

    return np.exp(C + D * v0 + iu * log_S)


def _heston_P(
    j: int,
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> float:
    """Compute P_j = ½ + (1/π) ∫₀^∞ Re[e^{-iu·ln K} f_j(u)/(iu)] du."""
    log_S = np.log(S)
    log_K = np.log(K)

    def integrand(u):
        cf = _heston_cf_j(u, T, r, log_S, v0, kappa, theta, xi, rho, j)
        return np.real(np.exp(-1j * u * log_K) * cf / (1j * u))

    integral, _ = quad(integrand, 1e-8, 200, limit=500)
    return 0.5 + integral / np.pi


def heston_price(
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
) -> float:
    """
    Price a European option under the Heston model (semi-analytical).

    Uses Gil-Pelaez inversion of the Heston characteristic function
    with the Albrecher formulation.

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
    v0 : float
        Initial variance (σ₀²).
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run variance.
    xi : float
        Vol-of-vol.
    rho : float
        Spot-vol correlation (-1 < ρ < 1).
    option_type : str
        ``'call'`` or ``'put'``.

    Returns
    -------
    float
        Heston option price.
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    P1 = _heston_P(1, S, K, T, r, v0, kappa, theta, xi, rho)
    P2 = _heston_P(2, S, K, T, r, v0, kappa, theta, xi, rho)

    call_price = S * P1 - K * np.exp(-r * T) * P2

    if option_type == "call":
        return max(float(call_price), 0.0)
    else:
        put_price = call_price - S + K * np.exp(-r * T)
        return max(float(put_price), 0.0)


def heston_mc(
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
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int = 42,
) -> MCResult:
    """
    Price a European option under Heston via Monte Carlo.

    Uses the Euler scheme with full truncation to keep variance
    non-negative.

    Parameters
    ----------
    S, K, T, r : float
        Standard option parameters.
    v0, kappa, theta, xi, rho : float
        Heston model parameters.
    option_type : str
        ``'call'`` or ``'put'``.
    n_paths : int
        Number of simulation paths.
    n_steps : int
        Time steps per path.
    seed : int
        Random seed.

    Returns
    -------
    MCResult
        Monte Carlo result.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    log_S = np.full(n_paths, np.log(S))
    v = np.full(n_paths, v0)

    for _ in range(n_steps):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)
        W_S = Z1
        W_v = rho * Z1 + np.sqrt(1.0 - rho**2) * Z2

        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)

        log_S += (r - 0.5 * v_pos) * dt + sqrt_v * np.sqrt(dt) * W_S
        v += kappa * (theta - v_pos) * dt + xi * sqrt_v * np.sqrt(dt) * W_v
        v = np.maximum(v, 0.0)

    S_T = np.exp(log_S)
    disc = np.exp(-r * T)

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    discounted = disc * payoffs
    price = float(np.mean(discounted))
    std_err = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))

    return MCResult(
        price=price,
        std_error=std_err,
        ci_lower=price - 1.96 * std_err,
        ci_upper=price + 1.96 * std_err,
    )


def heston_char_func_external(
    u: complex,
    T: float,
    r: float,
    log_S: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> complex:
    """Public wrapper exposing the Heston characteristic function (j=2, risk-neutral) for FFT pricing."""
    return _heston_cf_j(u, T, r, log_S, v0, kappa, theta, xi, rho, j=2)
