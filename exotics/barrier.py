"""
Barrier Option Pricing Module.

Barrier options are path-dependent options that are either activated ("in")
or deactivated ("out") when the underlying asset hits a barrier level H.

Types:
    - Up-and-Out: option ceases to exist if S hits H from below (H > S₀)
    - Up-and-In: option activates only if S hits H from below
    - Down-and-Out: option ceases to exist if S hits H from above (H < S₀)
    - Down-and-In: option activates only if S hits H from above

In-Out Parity:
    V_in + V_out = V_vanilla
    This is because one of them always activates.

Analytical Formulas (Merton 1973, Reiner & Rubinstein 1991):
    Closed-form solutions exist for European barrier options under GBM.
    These are used for validation of our Monte Carlo implementation.

Monte Carlo:
    Simulate paths and check at each step if the barrier is breached.
    Note: discrete monitoring may miss barrier crossings between steps,
    leading to a bias. Finer time steps reduce this bias.

References:
    Reiner, E. & Rubinstein, M. (1991). "Breaking Down the Barriers."
    Merton, R.C. (1973). "Theory of Rational Option Pricing."
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pricing.monte_carlo import MCResult


@dataclass
class BarrierSpec:
    """Specification for a barrier option."""
    barrier_type: str  # 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
    barrier: float     # Barrier level H
    option_type: str   # 'call' or 'put'


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def barrier_analytical(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    H: float,
    barrier_type: str = "down-and-out",
    option_type: str = "call",
) -> float:
    """
    Price a European barrier option using the Reiner-Rubinstein analytical formulas.

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    H : float
        Barrier level.
    barrier_type : str
        One of 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        Analytical barrier option price.
    """
    from pricing.black_scholes import bs_call_price, bs_put_price

    # Vanilla price for in-out parity
    if option_type == "call":
        vanilla = bs_call_price(S, K, T, r, sigma)
    else:
        vanilla = bs_put_price(S, K, T, r, sigma)

    lam = (r + 0.5 * sigma**2) / (sigma**2)
    y = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
    x1 = np.log(S / H) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
    y1 = np.log(H / S) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)

    if barrier_type == "down-and-out" and option_type == "call":
        if H >= K:
            # Case: H >= K
            price = (
                S * norm.cdf(x1)
                - K * np.exp(-r * T) * norm.cdf(x1 - sigma * np.sqrt(T))
                - S * (H / S) ** (2 * lam) * norm.cdf(y1)
                + K * np.exp(-r * T) * (H / S) ** (2 * lam - 2) * norm.cdf(y1 - sigma * np.sqrt(T))
            )
        else:
            # Case: H < K
            d1_val = _d1(S, K, T, r, sigma)
            d2_val = _d2(S, K, T, r, sigma)
            price = (
                S * norm.cdf(d1_val)
                - K * np.exp(-r * T) * norm.cdf(d2_val)
                - S * (H / S) ** (2 * lam) * norm.cdf(y)
                + K * np.exp(-r * T) * (H / S) ** (2 * lam - 2) * norm.cdf(y - sigma * np.sqrt(T))
            )
        return max(price, 0.0)

    elif barrier_type == "down-and-in" and option_type == "call":
        out_price = barrier_analytical(S, K, T, r, sigma, H, "down-and-out", "call")
        return max(vanilla - out_price, 0.0)

    elif barrier_type == "up-and-out" and option_type == "call":
        if H <= K:
            return 0.0
        d1_val = _d1(S, K, T, r, sigma)
        d2_val = _d2(S, K, T, r, sigma)
        price = (
            S * norm.cdf(d1_val)
            - K * np.exp(-r * T) * norm.cdf(d2_val)
            - S * norm.cdf(x1)
            + K * np.exp(-r * T) * norm.cdf(x1 - sigma * np.sqrt(T))
            + S * (H / S) ** (2 * lam) * (norm.cdf(-y) - norm.cdf(-y1))
            - K * np.exp(-r * T) * (H / S) ** (2 * lam - 2)
            * (norm.cdf(-y + sigma * np.sqrt(T)) - norm.cdf(-y1 + sigma * np.sqrt(T)))
        )
        return max(price, 0.0)

    elif barrier_type == "up-and-in" and option_type == "call":
        out_price = barrier_analytical(S, K, T, r, sigma, H, "up-and-out", "call")
        return max(vanilla - out_price, 0.0)

    elif barrier_type == "down-and-out" and option_type == "put":
        if H >= K:
            return 0.0
        d1_val = _d1(S, K, T, r, sigma)
        d2_val = _d2(S, K, T, r, sigma)
        price = (
            -S * norm.cdf(-d1_val)
            + K * np.exp(-r * T) * norm.cdf(-d2_val)
            + S * norm.cdf(-x1)
            - K * np.exp(-r * T) * norm.cdf(-x1 + sigma * np.sqrt(T))
            - S * (H / S) ** (2 * lam) * (norm.cdf(y) - norm.cdf(y1))
            + K * np.exp(-r * T) * (H / S) ** (2 * lam - 2)
            * (norm.cdf(y - sigma * np.sqrt(T)) - norm.cdf(y1 - sigma * np.sqrt(T)))
        )
        return max(price, 0.0)

    elif barrier_type == "down-and-in" and option_type == "put":
        out_price = barrier_analytical(S, K, T, r, sigma, H, "down-and-out", "put")
        return max(vanilla - out_price, 0.0)

    elif barrier_type == "up-and-out" and option_type == "put":
        if H <= S:
            return vanilla  # Barrier never hit
        x1_put = np.log(S / H) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
        y1_put = np.log(H / S) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
        if H >= K:
            price = (
                -S * norm.cdf(-x1_put)
                + K * np.exp(-r * T) * norm.cdf(-x1_put + sigma * np.sqrt(T))
                + S * (H / S) ** (2 * lam) * norm.cdf(-y1_put)
                - K * np.exp(-r * T) * (H / S) ** (2 * lam - 2) * norm.cdf(-y1_put + sigma * np.sqrt(T))
            )
        else:
            d1_val = _d1(S, K, T, r, sigma)
            d2_val = _d2(S, K, T, r, sigma)
            y_put = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
            price = (
                -S * norm.cdf(-d1_val)
                + K * np.exp(-r * T) * norm.cdf(-d2_val)
                + S * (H / S) ** (2 * lam) * norm.cdf(-y_put)
                - K * np.exp(-r * T) * (H / S) ** (2 * lam - 2) * norm.cdf(-y_put + sigma * np.sqrt(T))
            )
        return max(price, 0.0)

    elif barrier_type == "up-and-in" and option_type == "put":
        out_price = barrier_analytical(S, K, T, r, sigma, H, "up-and-out", "put")
        return max(vanilla - out_price, 0.0)

    else:
        raise ValueError(f"Invalid barrier_type '{barrier_type}' or option_type '{option_type}'")


def barrier_mc(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    H: float,
    barrier_type: str = "down-and-out",
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int | None = 42,
) -> MCResult:
    """
    Price a European barrier option via Monte Carlo simulation.

    Simulates full paths and checks barrier condition at each step.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard BS parameters.
    H : float
        Barrier level.
    barrier_type : str
        One of 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'.
    option_type : str
        'call' or 'put'.
    n_paths : int
        Number of simulation paths.
    n_steps : int
        Number of time steps per path.
    seed : int or None
        Random seed.

    Returns
    -------
    MCResult
        Price estimate with standard error and 95% CI.
    """
    from pricing.monte_carlo import simulate_paths

    paths = simulate_paths(S, T, r, sigma, n_paths, n_steps, seed)

    # Determine if barrier was hit for each path
    if "down" in barrier_type:
        hit = np.any(paths <= H, axis=1)
    else:  # "up"
        hit = np.any(paths >= H, axis=1)

    # Terminal payoff
    S_T = paths[:, -1]
    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    # Apply barrier condition
    if "out" in barrier_type:
        payoff = np.where(hit, 0.0, payoff)
    else:  # "in"
        payoff = np.where(hit, payoff, 0.0)

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
