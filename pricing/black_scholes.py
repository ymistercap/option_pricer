"""
Black-Scholes Analytical Option Pricing Module.

This module implements closed-form solutions for European option pricing under
the Black-Scholes-Merton (BSM) framework.

Model Assumptions:
    - The underlying asset follows a Geometric Brownian Motion (GBM):
      dS = μS dt + σS dW, where W is a standard Brownian motion.
    - Volatility σ is constant over the life of the option.
    - The risk-free rate r is constant and identical for all maturities.
    - No dividends are paid during the option's life.
    - No transaction costs or taxes.
    - Continuous trading is possible.
    - Markets are complete (every contingent claim can be replicated).

Under the risk-neutral measure Q (via Girsanov's theorem), the dynamics become:
    dS = rS dt + σS dW^Q

The price of a European call is the discounted expected payoff under Q:
    C = e^{-rT} E^Q[max(S_T - K, 0)]
      = S·N(d₁) - K·e^{-rT}·N(d₂)

where:
    d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d₂ = d₁ - σ√T
    N(·) is the standard normal CDF.

The put price follows from put-call parity:
    P = K·e^{-rT}·N(-d₂) - S·N(-d₁)

Limitations:
    - The constant volatility assumption is violated in practice (volatility smile/skew).
    - Real asset returns exhibit fat tails and skewness not captured by log-normality.
    - Does not account for stochastic volatility, jumps, or discrete dividends.
    - Assumes continuous hedging, which is impossible in practice.

References:
    Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities."
    Merton, R. C. (1973). "Theory of Rational Option Pricing."
"""

import numpy as np
from scipy.stats import norm


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute d₁ in the Black-Scholes formula.

    d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)

    Parameters
    ----------
    S : float
        Current price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate (annualized, continuous compounding).
    sigma : float
        Volatility of the underlying asset (annualized).

    Returns
    -------
    float
        The d₁ value.
    """
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute d₂ in the Black-Scholes formula.

    d₂ = d₁ - σ√T

    Parameters
    ----------
    S : float
        Current price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate (annualized, continuous compounding).
    sigma : float
        Volatility of the underlying asset (annualized).

    Returns
    -------
    float
        The d₂ value.
    """
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Price a European call option using the Black-Scholes formula.

    C = S·N(d₁) - K·e^{-rT}·N(d₂)

    This is derived by computing the discounted expected payoff under the
    risk-neutral measure: C = e^{-rT} E^Q[max(S_T - K, 0)], where S_T
    is log-normally distributed under Q.

    Parameters
    ----------
    S : float
        Current price of the underlying asset (S > 0).
    K : float
        Strike price (K > 0).
    T : float
        Time to maturity in years (T >= 0).
    r : float
        Risk-free interest rate (annualized, continuous compounding).
    sigma : float
        Volatility of the underlying asset (annualized, sigma >= 0).

    Returns
    -------
    float
        The Black-Scholes call price.

    Examples
    --------
    >>> round(bs_call_price(100, 100, 1.0, 0.05, 0.2), 4)
    10.4506
    """
    if T == 0:
        return max(S - K, 0.0)
    if S == 0:
        return 0.0
    if sigma == 0:
        return max(S - K * np.exp(-r * T), 0.0)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Price a European put option using the Black-Scholes formula.

    P = K·e^{-rT}·N(-d₂) - S·N(-d₁)

    Equivalently, from put-call parity: P = C - S + K·e^{-rT}.

    Parameters
    ----------
    S : float
        Current price of the underlying asset (S > 0).
    K : float
        Strike price (K > 0).
    T : float
        Time to maturity in years (T >= 0).
    r : float
        Risk-free interest rate (annualized, continuous compounding).
    sigma : float
        Volatility of the underlying asset (annualized, sigma >= 0).

    Returns
    -------
    float
        The Black-Scholes put price.

    Examples
    --------
    >>> round(bs_put_price(100, 100, 1.0, 0.05, 0.2), 4)
    5.5735
    """
    if T == 0:
        return max(K - S, 0.0)
    if S == 0:
        return K * np.exp(-r * T)
    if sigma == 0:
        return max(K * np.exp(-r * T) - S, 0.0)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_digital_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Price a European digital (binary/cash-or-nothing) call option.

    A digital call pays $1 if S_T > K, and $0 otherwise.

    Price = e^{-rT} · N(d₂)

    This is the risk-neutral probability (discounted) that the option
    finishes in the money.

    Parameters
    ----------
    S : float
        Current price of the underlying asset (S > 0).
    K : float
        Strike price (K > 0).
    T : float
        Time to maturity in years (T >= 0).
    r : float
        Risk-free interest rate (annualized, continuous compounding).
    sigma : float
        Volatility of the underlying asset (annualized, sigma >= 0).

    Returns
    -------
    float
        The digital call price (between 0 and e^{-rT}).

    Examples
    --------
    >>> round(bs_digital_call(100, 100, 1.0, 0.05, 0.2), 4)
    0.5323
    """
    if T == 0:
        return 1.0 if S > K else 0.0
    if S == 0:
        return 0.0
    if sigma == 0:
        return np.exp(-r * T) if S * np.exp(r * T) > K else 0.0

    d2 = _d2(S, K, T, r, sigma)
    return np.exp(-r * T) * norm.cdf(d2)


def bs_digital_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Price a European digital (binary/cash-or-nothing) put option.

    A digital put pays $1 if S_T < K, and $0 otherwise.

    Price = e^{-rT} · N(-d₂)

    Note: digital_call + digital_put = e^{-rT} (they cover all outcomes).

    Parameters
    ----------
    S : float
        Current price of the underlying asset (S > 0).
    K : float
        Strike price (K > 0).
    T : float
        Time to maturity in years (T >= 0).
    r : float
        Risk-free interest rate (annualized, continuous compounding).
    sigma : float
        Volatility of the underlying asset (annualized, sigma >= 0).

    Returns
    -------
    float
        The digital put price (between 0 and e^{-rT}).

    Examples
    --------
    >>> round(bs_digital_put(100, 100, 1.0, 0.05, 0.2), 4)
    0.4189
    """
    if T == 0:
        return 1.0 if S < K else 0.0
    if S == 0:
        return np.exp(-r * T)
    if sigma == 0:
        return np.exp(-r * T) if S * np.exp(r * T) < K else 0.0

    d2 = _d2(S, K, T, r, sigma)
    return np.exp(-r * T) * norm.cdf(-d2)


def bs_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """
    Price a European option using the Black-Scholes formula.

    Convenience wrapper that dispatches to bs_call_price or bs_put_price.

    Parameters
    ----------
    S : float
        Current price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate (annualized, continuous compounding).
    sigma : float
        Volatility of the underlying asset (annualized).
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        The Black-Scholes option price.
    """
    if option_type == "call":
        return bs_call_price(S, K, T, r, sigma)
    elif option_type == "put":
        return bs_put_price(S, K, T, r, sigma)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
