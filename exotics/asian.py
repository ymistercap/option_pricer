"""
Asian Option Pricing Module.

Asian options have payoffs that depend on the average price of the underlying
over a period, rather than just the terminal price.

Types:
    - Arithmetic average: payoff = max(Ā - K, 0) where Ā = (1/n)ΣS(tᵢ)
      No closed-form solution → must use Monte Carlo.

    - Geometric average: payoff = max(G̃ - K, 0) where G̃ = (ΠS(tᵢ))^{1/n}
      Has a closed-form solution under GBM because the geometric average
      of log-normal variables is log-normal.

The geometric average price serves as an excellent control variate for
the arithmetic average (they are highly correlated).

Geometric Average Closed-Form:
    Under GBM, the geometric average G̃ = exp((1/n)Σln(S(tᵢ))) is log-normal
    with adjusted drift and volatility:
        σ_G = σ · √((2n+1)/(6(n+1)))
        μ_G = (r - σ²/2)·(n+1)/(2n) + σ_G²/2
    (using the continuous-time approximation for large n)

References:
    Kemna, A. & Vorst, A. (1990). "A Pricing Method for Options Based on
    Average Asset Values."
"""

import numpy as np
from scipy.stats import norm

from pricing.monte_carlo import MCResult


def asian_geometric_analytical(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int = 252,
    option_type: str = "call",
) -> float:
    """
    Price a geometric average Asian option analytically.

    The geometric average of a GBM is itself log-normal, allowing a
    closed-form solution analogous to Black-Scholes with adjusted parameters.

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
    n_steps : int
        Number of averaging points.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        Analytical price of the geometric average Asian option.
    """
    n = n_steps
    sigma_g = sigma * np.sqrt((2 * n + 1) / (6 * (n + 1)))
    mu_g = 0.5 * (r - 0.5 * sigma**2) * (n + 1) / n + 0.5 * sigma_g**2

    d1 = (np.log(S / K) + (mu_g + 0.5 * sigma_g**2) * T) / (sigma_g * np.sqrt(T))
    d2 = d1 - sigma_g * np.sqrt(T)

    if option_type == "call":
        price = np.exp(-r * T) * (S * np.exp(mu_g * T) * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == "put":
        price = np.exp(-r * T) * (K * norm.cdf(-d2) - S * np.exp(mu_g * T) * norm.cdf(-d1))
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    return max(float(price), 0.0)


def asian_mc(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 100_000,
    n_steps: int = 252,
    average_type: str = "arithmetic",
    option_type: str = "call",
    seed: int | None = 42,
) -> MCResult:
    """
    Price an Asian option via Monte Carlo.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard BS parameters.
    n_paths : int
        Number of simulation paths.
    n_steps : int
        Number of time steps / averaging points.
    average_type : str
        'arithmetic' or 'geometric'.
    option_type : str
        'call' or 'put'.
    seed : int or None
        Random seed.

    Returns
    -------
    MCResult
        Price estimate with standard error and 95% CI.
    """
    from pricing.monte_carlo import simulate_paths

    paths = simulate_paths(S, T, r, sigma, n_paths, n_steps, seed)

    # Compute average (excluding initial price — use monitoring points only)
    monitoring = paths[:, 1:]

    if average_type == "arithmetic":
        avg = np.mean(monitoring, axis=1)
    elif average_type == "geometric":
        avg = np.exp(np.mean(np.log(monitoring), axis=1))
    else:
        raise ValueError(f"average_type must be 'arithmetic' or 'geometric'")

    if option_type == "call":
        payoff = np.maximum(avg - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - avg, 0.0)
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


def asian_arithmetic_cv(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 100_000,
    n_steps: int = 252,
    option_type: str = "call",
    seed: int | None = 42,
) -> MCResult:
    """
    Price an arithmetic Asian option using geometric average as control variate.

    The geometric average Asian has a known price, and is highly correlated
    with the arithmetic average, making it an excellent control variate.

    V̂_arith_CV = V̂_arith - β·(V̂_geom_MC - V_geom_analytical)

    Parameters
    ----------
    S, K, T, r, sigma, n_paths, n_steps, option_type, seed
        Same as asian_mc.

    Returns
    -------
    MCResult
        Price estimate with standard error and 95% CI.
    """
    from pricing.monte_carlo import simulate_paths

    paths = simulate_paths(S, T, r, sigma, n_paths, n_steps, seed)
    monitoring = paths[:, 1:]

    arith_avg = np.mean(monitoring, axis=1)
    geom_avg = np.exp(np.mean(np.log(monitoring), axis=1))

    if option_type == "call":
        arith_payoff = np.maximum(arith_avg - K, 0.0)
        geom_payoff = np.maximum(geom_avg - K, 0.0)
    else:
        arith_payoff = np.maximum(K - arith_avg, 0.0)
        geom_payoff = np.maximum(K - geom_avg, 0.0)

    discount = np.exp(-r * T)
    arith_disc = discount * arith_payoff
    geom_disc = discount * geom_payoff

    # Known geometric price
    geom_analytical = asian_geometric_analytical(S, K, T, r, sigma, n_steps, option_type)

    # Optimal beta
    cov_matrix = np.cov(arith_disc, geom_disc, ddof=1)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0.0

    adjusted = arith_disc - beta * (geom_disc - geom_analytical)
    price = float(np.mean(adjusted))
    std_error = float(np.std(adjusted, ddof=1) / np.sqrt(n_paths))

    return MCResult(
        price=price,
        std_error=std_error,
        ci_lower=price - 1.96 * std_error,
        ci_upper=price + 1.96 * std_error,
    )
