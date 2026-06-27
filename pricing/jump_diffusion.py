"""
Merton (1976) Jump-Diffusion Option Pricing.

The Merton model adds a compound Poisson jump process to the GBM
diffusion, capturing sudden large moves (earnings, crashes):

    dS/S = (r - λ·k) dt + σ dW + J dN

where:
    N(t) ~ Poisson(λ·t)       — jump arrival process
    J = exp(m + δ·Z) - 1      — relative jump size (log-normal)
    k = E[J] = exp(m + δ²/2) - 1
    λ — jump intensity (expected jumps per year)
    m — mean of log-jump size
    δ — std dev of log-jump size

The price is an infinite series (Merton's formula):

    V = Σ_{n=0}^{∞}  e^{-λ'T} (λ'T)^n / n!  · BS(S, K, T, r_n, σ_n)

where:
    λ' = λ(1+k)
    r_n = r - λk + n·ln(1+k)/T
    σ_n² = σ² + n·δ²/T

References:
    Merton, R.C. (1976). "Option pricing when underlying stock returns
    are discontinuous." J. Financial Economics, 3, 125-144.
"""

import numpy as np
from scipy.special import factorial

from pricing.black_scholes import bs_price
from pricing.monte_carlo import MCResult


def merton_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    m: float,
    delta: float,
    option_type: str = "call",
    n_terms: int = 50,
) -> float:
    """
    Price a European option under Merton's jump-diffusion (analytical series).

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate.
    sigma : float
        Diffusion volatility.
    lam : float
        Jump intensity (expected number of jumps per year).
    m : float
        Mean of log-jump size.
    delta : float
        Std deviation of log-jump size.
    option_type : str
        ``'call'`` or ``'put'``.
    n_terms : int
        Number of terms in the series expansion.

    Returns
    -------
    float
        Merton jump-diffusion price.
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    k = np.exp(m + 0.5 * delta**2) - 1.0
    lam_prime = lam * (1.0 + k)

    price = 0.0
    for n in range(n_terms):
        sigma_n = np.sqrt(sigma**2 + n * delta**2 / T)
        r_n = r - lam * k + n * np.log(1.0 + k) / T
        weight = np.exp(-lam_prime * T) * (lam_prime * T) ** n / factorial(n, exact=True)
        price += weight * bs_price(S, K, T, r_n, sigma_n, option_type)

    return float(price)


def merton_mc(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    m: float,
    delta: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int = 42,
) -> MCResult:
    """
    Price a European option under Merton's jump-diffusion via Monte Carlo.

    Simulates paths with both diffusion and Poisson-driven log-normal jumps.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate.
    sigma : float
        Diffusion volatility.
    lam : float
        Jump intensity.
    m : float
        Mean of log-jump size.
    delta : float
        Std deviation of log-jump size.
    option_type : str
        ``'call'`` or ``'put'``.
    n_paths : int
        Number of simulation paths.
    n_steps : int
        Number of time steps per path.
    seed : int
        Random seed.

    Returns
    -------
    MCResult
        Monte Carlo result with price, std_error, and confidence interval.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    k = np.exp(m + 0.5 * delta**2) - 1.0

    # Drift compensated for jumps
    drift = (r - 0.5 * sigma**2 - lam * k) * dt

    log_S = np.full(n_paths, np.log(S))

    for _ in range(n_steps):
        # Diffusion
        dW = sigma * np.sqrt(dt) * rng.standard_normal(n_paths)
        # Jumps: number of jumps in this dt
        n_jumps = rng.poisson(lam * dt, n_paths)
        # Aggregate jump sizes (sum of n_jumps log-normal jumps)
        jump_component = np.zeros(n_paths)
        max_jumps = n_jumps.max() if n_jumps.max() > 0 else 0
        for j in range(1, max_jumps + 1):
            mask = n_jumps >= j
            jump_component[mask] += m + delta * rng.standard_normal(mask.sum())

        log_S += drift + dW + jump_component

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
