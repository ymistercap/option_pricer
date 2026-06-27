"""
Cox-Ross-Rubinstein (CRR) Binomial Tree Option Pricing.

The CRR model discretises the continuous GBM into a recombining binomial
lattice.  At each time step dt = T/N the spot can move:

    u = exp(σ√dt)       (up factor)
    d = 1/u = exp(-σ√dt) (down factor)

The risk-neutral probability of an up move is:

    p = (exp(r·dt) - d) / (u - d)

European prices are computed by backward induction from the terminal
payoff.  American prices additionally check for early exercise at every
node.

Convergence is O(1/N) with oscillatory behaviour that can be dampened by
Richardson extrapolation or by averaging results for N and N+1 steps.

References:
    Cox, J., Ross, S. & Rubinstein, M. (1979).
    "Option Pricing: A Simplified Approach."
"""

import numpy as np


def crr_european(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_steps: int = 200,
) -> float:
    """
    Price a European option on a CRR binomial tree.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate (continuous compounding).
    sigma : float
        Annualised volatility.
    option_type : str
        ``'call'`` or ``'put'``.
    n_steps : int
        Number of time steps in the tree.

    Returns
    -------
    float
        European option price.
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)

    # Terminal asset prices at step n_steps
    js = np.arange(n_steps + 1)
    S_T = S * u ** (n_steps - js) * d ** js

    if option_type == "call":
        values = np.maximum(S_T - K, 0.0)
    else:
        values = np.maximum(K - S_T, 0.0)

    # Backward induction
    for _ in range(n_steps):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])

    return float(values[0])


def crr_american(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_steps: int = 200,
) -> float:
    """
    Price an American option on a CRR binomial tree.

    At each node the holder may exercise early, so the value is the
    maximum of continuation value and intrinsic value.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate (continuous compounding).
    sigma : float
        Annualised volatility.
    option_type : str
        ``'call'`` or ``'put'``.
    n_steps : int
        Number of time steps in the tree.

    Returns
    -------
    float
        American option price.
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)

    # Terminal payoff
    js = np.arange(n_steps + 1)
    S_T = S * u ** (n_steps - js) * d ** js

    if option_type == "call":
        values = np.maximum(S_T - K, 0.0)
    else:
        values = np.maximum(K - S_T, 0.0)

    # Backward induction with early-exercise check
    for i in range(n_steps - 1, -1, -1):
        js_i = np.arange(i + 1)
        S_i = S * u ** (i - js_i) * d ** js_i
        continuation = disc * (p * values[:-1] + (1 - p) * values[1:])
        if option_type == "call":
            exercise = np.maximum(S_i - K, 0.0)
        else:
            exercise = np.maximum(K - S_i, 0.0)
        values = np.maximum(continuation, exercise)

    return float(values[0])
