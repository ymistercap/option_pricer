"""
Analytical Greeks for European Options under Black-Scholes.

Greeks measure the sensitivity of an option's price to changes in underlying
parameters. They are essential for risk management and hedging.

All formulas are derived by differentiating the Black-Scholes pricing formula:
    C = S·N(d₁) - K·e^{-rT}·N(d₂)
    P = K·e^{-rT}·N(-d₂) - S·N(-d₁)

where:
    d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d₂ = d₁ - σ√T
    n(x) = N'(x) = (1/√(2π))·e^{-x²/2}  (standard normal PDF)

Key relationships:
    - Gamma = ∂Delta/∂S  (curvature of the price w.r.t. spot)
    - The BS PDE links Theta, Delta, and Gamma:
      Θ + ½σ²S²Γ + rSΔ - rV = 0

References:
    Hull, J. (2018). "Options, Futures, and Other Derivatives."
"""

import numpy as np
from scipy.stats import norm


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)."""
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d₂ = d₁ - σ√T."""
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def delta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """
    Compute Delta: ∂V/∂S — sensitivity of option price to spot price.

    Call Delta = N(d₁) ∈ (0, 1)
    Put  Delta = N(d₁) - 1 = -N(-d₁) ∈ (-1, 0)

    Interpretation: Delta approximates the probability (under the share measure)
    that the option finishes in-the-money. It also represents the hedge ratio —
    the number of shares needed to delta-hedge one option.

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate (continuous).
    sigma : float
        Volatility (annualized).
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        The option's delta.
    """
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = _d1(S, K, T, r, sigma)
    if option_type == "call":
        return float(norm.cdf(d1))
    elif option_type == "put":
        return float(norm.cdf(d1) - 1.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute Gamma: ∂²V/∂S² — rate of change of delta w.r.t. spot.

    Γ = n(d₁) / (S·σ·√T)

    Gamma is identical for calls and puts (from put-call parity, their deltas
    differ by a constant, so the second derivative is the same).

    Interpretation: Gamma measures the curvature of the option price as a
    function of the underlying. High Gamma means delta changes rapidly —
    the option is harder to hedge.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard BS parameters.

    Returns
    -------
    float
        The option's gamma.
    """
    if T <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma)
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute Vega: ∂V/∂σ — sensitivity of option price to volatility.

    ν = S·√T·n(d₁)

    Vega is identical for calls and puts. It is always positive: higher
    volatility increases the value of both calls and puts (more optionality).

    Note: Vega is not a Greek letter — it is sometimes called "kappa" in
    academic literature.

    Returns the vega per 1 unit change in sigma (not per 1% change).

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard BS parameters.

    Returns
    -------
    float
        The option's vega.
    """
    if T <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma)
    return float(S * np.sqrt(T) * norm.pdf(d1))


def theta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """
    Compute Theta: ∂V/∂t — sensitivity of option price to the passage of time.

    Convention: returns the rate of change per year. For daily theta, divide by 252.

    Call Θ = -[S·n(d₁)·σ / (2√T)] - r·K·e^{-rT}·N(d₂)
    Put  Θ = -[S·n(d₁)·σ / (2√T)] + r·K·e^{-rT}·N(-d₂)

    Theta is typically negative for long option positions ("time decay"):
    options lose value as time passes, all else equal.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard BS parameters.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        The option's theta (per year).
    """
    if T <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)

    term1 = -S * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))

    if option_type == "call":
        return float(term1 - r * K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == "put":
        return float(term1 + r * K * np.exp(-r * T) * norm.cdf(-d2))
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def rho(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """
    Compute Rho: ∂V/∂r — sensitivity of option price to the risk-free rate.

    Call ρ = K·T·e^{-rT}·N(d₂)
    Put  ρ = -K·T·e^{-rT}·N(-d₂)

    Interpretation: increasing rates raises the forward price of the underlying,
    benefiting calls and hurting puts.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard BS parameters.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        The option's rho.
    """
    if T <= 0:
        return 0.0

    d2 = _d2(S, K, T, r, sigma)

    if option_type == "call":
        return float(K * T * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == "put":
        return float(-K * T * np.exp(-r * T) * norm.cdf(-d2))
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def vanna(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute Vanna: ∂²V/∂S∂σ = ∂Delta/∂σ = ∂Vega/∂S.

    Vanna = -n(d₁)·d₂/σ = Vega/S · (1 - d₁/(σ√T))

    This is a second-order cross-Greek measuring how delta changes with
    volatility (or equivalently how vega changes with spot).

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard BS parameters.

    Returns
    -------
    float
        The option's vanna.
    """
    if T <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return float(-norm.pdf(d1) * d2 / sigma)


def volga(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute Volga (Vomma): ∂²V/∂σ² — sensitivity of vega to volatility.

    Volga = S·√T·n(d₁)·d₁·d₂/σ = Vega · d₁·d₂/σ

    Volga measures the convexity of the option price w.r.t. volatility.
    It is important for volatility trading and for the vega-hedging of
    exotic options.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard BS parameters.

    Returns
    -------
    float
        The option's volga.
    """
    if T <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    v = vega(S, K, T, r, sigma)
    return float(v * d1 * d2 / sigma)
