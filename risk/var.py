"""
Value at Risk (VaR) and Expected Shortfall Module.

VaR answers: "What is the maximum loss at a given confidence level over a given horizon?"

Methods:
    1. Historical VaR: Uses the empirical distribution of past returns.
       Non-parametric, captures fat tails and skewness naturally.
       VaR_α = -Quantile(returns, α)

    2. Parametric (Variance-Covariance) VaR: Assumes returns are normally distributed.
       VaR_α = -(μ + z_α · σ) · portfolio_value
       where z_α is the normal quantile (e.g., z_0.05 = -1.645).

    3. Monte Carlo VaR: Simulates future returns from a fitted distribution
       and computes the quantile of the simulated P&L.

Expected Shortfall (CVaR):
    ES = E[Loss | Loss > VaR]
    The average loss given that we are in the tail. More coherent than VaR
    as a risk measure (subadditive).

Scaling:
    Under i.i.d. returns, VaR scales as √t:
    VaR_10d = VaR_1d · √10

Backtesting:
    Compare actual violations (days where loss > VaR) against expected
    violations (α · n_days). Too many violations → model underestimates risk.

References:
    Jorion, P. (2006). "Value at Risk: The New Benchmark for Managing Financial Risk."
"""

import numpy as np
import pandas as pd


def historical_var(
    returns: np.ndarray | pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    """
    Compute historical Value at Risk.

    Parameters
    ----------
    returns : array-like
        Historical daily returns (as decimals, e.g., 0.01 = 1%).
    confidence : float
        Confidence level (e.g., 0.95 or 0.99).
    horizon : int
        Holding period in days.

    Returns
    -------
    float
        VaR as a positive number (loss).
    """
    returns = np.asarray(returns)
    alpha = 1.0 - confidence
    var_1d = -np.percentile(returns, alpha * 100)
    return float(var_1d * np.sqrt(horizon))


def parametric_var(
    returns: np.ndarray | pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    """
    Compute parametric (Gaussian) Value at Risk.

    Assumes returns ~ N(μ, σ²). VaR = -(μ + z_α · σ) scaled by √horizon.

    Parameters
    ----------
    returns : array-like
        Historical daily returns.
    confidence : float
        Confidence level.
    horizon : int
        Holding period in days.

    Returns
    -------
    float
        VaR as a positive number (loss).
    """
    from scipy.stats import norm

    returns = np.asarray(returns)
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    alpha = 1.0 - confidence
    z = norm.ppf(alpha)
    var_1d = -(mu + z * sigma)
    return float(var_1d * np.sqrt(horizon))


def monte_carlo_var(
    returns: np.ndarray | pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
    n_sims: int = 100_000,
    seed: int | None = 42,
) -> float:
    """
    Compute Monte Carlo Value at Risk.

    Fits a normal distribution to historical returns and simulates
    future returns to estimate the VaR.

    Parameters
    ----------
    returns : array-like
        Historical daily returns.
    confidence : float
        Confidence level.
    horizon : int
        Holding period in days.
    n_sims : int
        Number of MC simulations.
    seed : int or None
        Random seed.

    Returns
    -------
    float
        VaR as a positive number (loss).
    """
    returns = np.asarray(returns)
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)

    rng = np.random.default_rng(seed)
    sim_returns = rng.normal(mu, sigma, (n_sims, horizon))
    cum_returns = np.sum(sim_returns, axis=1)

    alpha = 1.0 - confidence
    return float(-np.percentile(cum_returns, alpha * 100))


def expected_shortfall(
    returns: np.ndarray | pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    """
    Compute Expected Shortfall (Conditional VaR).

    ES = E[Loss | Loss > VaR] = average of losses beyond the VaR threshold.

    ES is a coherent risk measure (subadditive), unlike VaR.

    Parameters
    ----------
    returns : array-like
        Historical daily returns.
    confidence : float
        Confidence level.
    horizon : int
        Holding period in days.

    Returns
    -------
    float
        Expected shortfall as a positive number.
    """
    returns = np.asarray(returns)
    alpha = 1.0 - confidence
    var_threshold = np.percentile(returns, alpha * 100)
    tail_losses = returns[returns <= var_threshold]
    es_1d = -np.mean(tail_losses) if len(tail_losses) > 0 else 0.0
    return float(es_1d * np.sqrt(horizon))


def backtest_var(
    returns: np.ndarray | pd.Series,
    confidence: float = 0.95,
    window: int = 252,
) -> pd.DataFrame:
    """
    Backtest VaR by computing rolling VaR and counting violations.

    A violation occurs when the actual loss exceeds the predicted VaR.
    Expected violation rate = 1 - confidence.

    Parameters
    ----------
    returns : array-like
        Full history of daily returns.
    confidence : float
        Confidence level.
    window : int
        Rolling window size for VaR estimation.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'return', 'var', 'violation'.
    """
    returns = np.asarray(returns)
    n = len(returns)

    results = []
    for i in range(window, n):
        hist = returns[i - window:i]
        var = historical_var(hist, confidence)
        actual = returns[i]
        violation = actual < -var

        results.append({
            "day": i,
            "return": actual,
            "var": var,
            "violation": violation,
        })

    df = pd.DataFrame(results)
    return df
