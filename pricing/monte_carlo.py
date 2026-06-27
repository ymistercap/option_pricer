"""
Monte Carlo Option Pricing Module.

This module prices European options by simulating the underlying asset's
risk-neutral dynamics and computing the discounted expected payoff.

Theoretical Foundation:
    Under the risk-neutral measure Q (obtained via Girsanov's theorem), the
    asset price follows:
        dS = rS dt + σS dW^Q

    The exact solution of this GBM is:
        S(t+dt) = S(t) · exp((r - σ²/2)·dt + σ·√dt·Z),  Z ~ N(0,1)

    The option price is then:
        V = e^{-rT} · E^Q[payoff(S_T)]

    We estimate this expectation by averaging over N simulated paths:
        V̂ = e^{-rT} · (1/N) · Σ payoff(S_T^i)

    By the Central Limit Theorem, the standard error decreases as O(1/√N),
    so quadrupling the number of paths halves the error.

Variance Reduction Techniques:
    1. Antithetic Variates: For each Z, also simulate -Z. The payoffs are
       negatively correlated, reducing variance. Cost: negligible overhead.

    2. Control Variates: Use a correlated variable with known expectation
       (e.g., the geometric average or the asset itself) to reduce variance.
       V̂_CV = V̂ - β·(Ĉ - E[C]), where β = Cov(V,C)/Var(C).

References:
    Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering."
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    """Container for Monte Carlo pricing results.

    Attributes
    ----------
    price : float
        Estimated option price.
    std_error : float
        Standard error of the estimate.
    ci_lower : float
        Lower bound of 95% confidence interval.
    ci_upper : float
        Upper bound of 95% confidence interval.
    """

    price: float
    std_error: float
    ci_lower: float
    ci_upper: float


def _simulate_gbm_terminal(
    S: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate terminal asset prices under GBM (exact, single step).

    S_T = S · exp((r - σ²/2)T + σ√T · Z)

    Parameters
    ----------
    S : float
        Initial spot price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    n_paths : int
        Number of paths to simulate.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of terminal prices, shape (n_paths,).
    """
    Z = rng.standard_normal(n_paths)
    return S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def _simulate_gbm_paths(
    S: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate full GBM paths using the exact log-normal scheme.

    At each step: S(t+dt) = S(t) · exp((r - σ²/2)·dt + σ·√dt·Z)

    Parameters
    ----------
    S : float
        Initial spot price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    n_paths : int
        Number of paths.
    n_steps : int
        Number of time steps.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of shape (n_paths, n_steps + 1) with full paths.
        Column 0 is the initial price S.
    """
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    log_paths = np.zeros((n_paths, n_steps + 1))
    log_paths[:, 0] = np.log(S)
    log_paths[:, 1:] = np.cumsum(log_returns, axis=1) + np.log(S)

    return np.exp(log_paths)


def _payoff(S_T: np.ndarray, K: float, option_type: str) -> np.ndarray:
    """Compute option payoffs at maturity."""
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        return np.maximum(K - S_T, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def _build_result(payoffs: np.ndarray, r: float, T: float) -> MCResult:
    """Compute discounted MC estimate with confidence interval."""
    discount = np.exp(-r * T)
    discounted = discount * payoffs
    price = float(np.mean(discounted))
    std_error = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
    return MCResult(
        price=price,
        std_error=std_error,
        ci_lower=price - 1.96 * std_error,
        ci_upper=price + 1.96 * std_error,
    )


def mc_european(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 100_000,
    option_type: str = "call",
    seed: int | None = 42,
) -> MCResult:
    """
    Price a European option via standard Monte Carlo.

    Simulates N independent paths of the terminal asset price under GBM
    and averages the discounted payoffs.

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    n_paths : int
        Number of simulation paths.
    option_type : str
        'call' or 'put'.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    MCResult
        Price estimate with standard error and 95% CI.

    Examples
    --------
    >>> result = mc_european(100, 100, 1.0, 0.05, 0.2, n_paths=500_000)
    >>> abs(result.price - 10.4506) < 0.15
    True
    """
    rng = np.random.default_rng(seed)
    S_T = _simulate_gbm_terminal(S, T, r, sigma, n_paths, rng)
    payoffs = _payoff(S_T, K, option_type)
    return _build_result(payoffs, r, T)


def mc_antithetic(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 100_000,
    option_type: str = "call",
    seed: int | None = 42,
) -> MCResult:
    """
    Price a European option using antithetic variates.

    For each random draw Z, we also use -Z, creating a negatively correlated
    pair of payoffs. The average of each pair has lower variance than using
    independent draws.

    If Var(payoff(Z)) = σ², then:
        Var((payoff(Z) + payoff(-Z))/2) = σ²/2 · (1 + ρ)
    where ρ = Corr(payoff(Z), payoff(-Z)) < 0 for most payoff functions,
    yielding variance reduction.

    Parameters
    ----------
    S, K, T, r, sigma, n_paths, option_type, seed
        Same as mc_european.

    Returns
    -------
    MCResult
        Price estimate with standard error and 95% CI.
    """
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    Z = rng.standard_normal(half)

    drift = (r - 0.5 * sigma**2) * T
    vol_term = sigma * np.sqrt(T)

    S_T_pos = S * np.exp(drift + vol_term * Z)
    S_T_neg = S * np.exp(drift + vol_term * (-Z))

    payoff_pos = _payoff(S_T_pos, K, option_type)
    payoff_neg = _payoff(S_T_neg, K, option_type)

    # Average each antithetic pair
    payoffs = 0.5 * (payoff_pos + payoff_neg)
    return _build_result(payoffs, r, T)


def mc_control_variate(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 100_000,
    option_type: str = "call",
    seed: int | None = 42,
) -> MCResult:
    """
    Price a European option using control variates.

    We use the terminal asset price S_T as the control variate, since
    E^Q[S_T] = S·e^{rT} is known exactly.

    The adjusted estimator is:
        V̂_CV = V̂ - β·(S̄_T - E[S_T])
    where β = Cov(payoff, S_T) / Var(S_T), estimated from the sample.

    This reduces variance when the payoff is correlated with S_T, which is
    strongly the case for calls and puts.

    Parameters
    ----------
    S, K, T, r, sigma, n_paths, option_type, seed
        Same as mc_european.

    Returns
    -------
    MCResult
        Price estimate with standard error and 95% CI.
    """
    rng = np.random.default_rng(seed)
    S_T = _simulate_gbm_terminal(S, T, r, sigma, n_paths, rng)
    payoffs = _payoff(S_T, K, option_type)

    # Control variate: S_T with known expectation
    E_ST = S * np.exp(r * T)

    # Estimate optimal beta
    cov = np.cov(payoffs, S_T, ddof=1)
    beta = cov[0, 1] / cov[1, 1]

    # Adjusted payoffs
    adjusted_payoffs = payoffs - beta * (S_T - E_ST)
    return _build_result(adjusted_payoffs, r, T)


def simulate_paths(
    S: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 10_000,
    n_steps: int = 252,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Simulate full GBM paths (exposed for use by hedging, exotics, etc.).

    Parameters
    ----------
    S : float
        Initial spot price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    n_paths : int
        Number of paths.
    n_steps : int
        Number of time steps.
    seed : int or None
        Random seed.

    Returns
    -------
    np.ndarray
        Array of shape (n_paths, n_steps + 1).
    """
    rng = np.random.default_rng(seed)
    return _simulate_gbm_paths(S, T, r, sigma, n_paths, n_steps, rng)
