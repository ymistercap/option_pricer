"""
Lookback Option Pricing Module.

Lookback options have payoffs that depend on the extremum (maximum or minimum)
of the underlying price over the life of the option.

Floating Strike Lookback:
    - Call: payoff = S_T - min(S)  (buy at the lowest price)
    - Put:  payoff = max(S) - S_T  (sell at the highest price)

These options are always in the money (payoff ≥ 0), so they are more
expensive than vanilla options.

Analytical Formula (Goldman-Sosin-Gatto, 1979):
    For a floating strike lookback call under continuous monitoring:
        C = S·N(a₁) - S·σ²/(2r)·N(-a₁) - M·e^{-rT}·[N(a₂) - σ²/(2r)·e^{rT}·N(-a₃)]
    where M = min(S) at inception (= S for a new option).

Monte Carlo:
    Simulate paths and track the running min/max. Discrete monitoring
    leads to a bias (the true continuous min/max is more extreme).

References:
    Goldman, B., Sosin, H. & Gatto, M. (1979). "Path Dependent Options:
    Buy at the Low, Sell at the High."
"""

import numpy as np
from scipy.stats import norm

from pricing.monte_carlo import MCResult


def lookback_floating_analytical(
    S: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Price a floating-strike lookback option analytically (Goldman-Sosin-Gatto).

    Assumes continuous monitoring and that the option is newly issued
    (so min = max = S at inception).

    Parameters
    ----------
    S : float
        Current spot price (also the initial min/max).
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    option_type : str
        'call' (buy at min) or 'put' (sell at max).

    Returns
    -------
    float
        Analytical lookback option price.
    """
    if T <= 0:
        return 0.0

    sqrtT = np.sqrt(T)
    a1 = (r + 0.5 * sigma**2) * T / (sigma * sqrtT)
    a2 = a1 - sigma * sqrtT  # = (r - 0.5*sigma^2)*T / (sigma*sqrt(T))

    if r == 0:
        # Special case: r = 0 — use limit formula
        if option_type == "call":
            price = S * (norm.cdf(a1) - norm.cdf(-a1)) + S * sigma * sqrtT * (
                2.0 * norm.pdf(a1)
            )
            return max(float(price), 0.0)
        else:
            price = S * (norm.cdf(-a1) - norm.cdf(a1)) + S * sigma * sqrtT * (
                2.0 * norm.pdf(a1)
            )
            return max(float(price), 0.0)

    eta = sigma**2 / (2.0 * r)
    disc = np.exp(-r * T)

    # Goldman-Sosin-Gatto formulas for floating strike lookback (m=M=S at inception)
    # Note: -a1 + 2r√T/σ = a2, and a1 - 2r√T/σ = -a2
    if option_type == "call":
        # C = S·N(a1) - S·e^{-rT}·N(a2) - S·η·[N(-a1) - e^{-rT}·N(a2)]
        price = (
            S * norm.cdf(a1)
            - S * disc * norm.cdf(a2)
            - S * eta * (norm.cdf(-a1) - disc * norm.cdf(a2))
        )
    elif option_type == "put":
        # P = -S·N(-a1) + S·e^{-rT}·N(-a2) + S·η·[N(a1) - e^{-rT}·N(-a2)]
        price = (
            -S * norm.cdf(-a1)
            + S * disc * norm.cdf(-a2)
            + S * eta * (norm.cdf(a1) - disc * norm.cdf(-a2))
        )
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    return max(float(price), 0.0)


def lookback_mc(
    S: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int | None = 42,
) -> MCResult:
    """
    Price a floating-strike lookback option via Monte Carlo.

    Parameters
    ----------
    S : float
        Current spot price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    option_type : str
        'call' or 'put'.
    n_paths : int
        Number of simulation paths.
    n_steps : int
        Number of time steps.
    seed : int or None
        Random seed.

    Returns
    -------
    MCResult
        Price estimate with standard error and 95% CI.
    """
    from pricing.monte_carlo import simulate_paths

    paths = simulate_paths(S, T, r, sigma, n_paths, n_steps, seed)
    S_T = paths[:, -1]

    if option_type == "call":
        S_min = np.min(paths, axis=1)
        payoff = S_T - S_min
    elif option_type == "put":
        S_max = np.max(paths, axis=1)
        payoff = S_max - S_T
    else:
        raise ValueError(f"option_type must be 'call' or 'put'")

    discount = np.exp(-r * T)
    discounted = discount * payoff
    price = float(np.mean(discounted))
    std_error = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))

    return MCResult(
        price=price,
        std_error=std_error,
        ci_lower=price - 1.96 * std_error,
        ci_upper=price + 1.96 * std_error,
    )
