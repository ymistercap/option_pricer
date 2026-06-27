"""
Dynamic Delta-Hedging Simulation Module.

Delta-hedging is the cornerstone of Black-Scholes theory. A trader who sells
a call and continuously adjusts a position in the underlying (holding Δ shares)
can replicate the option payoff, eliminating directional risk.

In practice, hedging is discrete (e.g., daily), leading to hedging error.
The residual P&L comes from the gamma exposure:

    P&L ≈ ½ · Γ · S² · (σ_realized² - σ_implied²) · dt

This means:
    - If realized vol > implied vol → long gamma profits (hedger loses if short gamma)
    - If realized vol < implied vol → short gamma profits
    - The P&L is path-dependent through gamma and spot

The total P&L distribution narrows as rebalancing frequency increases,
converging to zero (BS world) or to a deterministic value (when σ_real ≠ σ_impl).

References:
    Taleb, N. (1997). "Dynamic Hedging: Managing Vanilla and Exotic Options."
    Hull, J. (2018). "Options, Futures, and Other Derivatives." Ch. 19.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from greeks.analytical import delta as bs_delta
from pricing.black_scholes import bs_call_price


@dataclass
class HedgeResult:
    """Results of a delta-hedging simulation.

    Attributes
    ----------
    pnl : np.ndarray
        P&L for each simulation path.
    mean_pnl : float
        Mean P&L across all paths.
    std_pnl : float
        Standard deviation of P&L.
    paths : np.ndarray
        Simulated spot paths, shape (n_paths, n_steps+1).
    deltas : np.ndarray
        Delta at each step for the first path, shape (n_steps+1,).
    hedge_pnl_path : np.ndarray
        Cumulative P&L for the first path, shape (n_steps+1,).
    """

    pnl: np.ndarray
    mean_pnl: float
    std_pnl: float
    paths: np.ndarray
    deltas: np.ndarray
    hedge_pnl_path: np.ndarray


def simulate_delta_hedge(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma_impl: float,
    sigma_real: float | None = None,
    n_paths: int = 10_000,
    n_steps: int = 252,
    transaction_cost: float = 0.0,
    seed: int | None = 42,
) -> HedgeResult:
    """
    Simulate dynamic delta-hedging of a short European call position.

    The trader:
    1. Sells one European call at the BS price (using sigma_impl)
    2. At each time step, computes delta and adjusts the hedge
    3. At maturity, settles the option payoff

    The underlying evolves with sigma_real (which may differ from sigma_impl).

    Parameters
    ----------
    S : float
        Initial spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate.
    sigma_impl : float
        Implied volatility used for hedging (delta computation).
    sigma_real : float, optional
        Realized volatility used to simulate the underlying.
        Defaults to sigma_impl (perfect BS world).
    n_paths : int
        Number of simulation paths.
    n_steps : int
        Number of hedging intervals.
    transaction_cost : float
        Proportional transaction cost per dollar traded (e.g., 0.001 = 10bps).
    seed : int or None
        Random seed.

    Returns
    -------
    HedgeResult
        Simulation results including P&L distribution.
    """
    if sigma_real is None:
        sigma_real = sigma_impl

    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # Simulate paths under real-world dynamics (using risk-neutral for simplicity)
    Z = rng.standard_normal((n_paths, n_steps))
    log_returns = (r - 0.5 * sigma_real**2) * dt + sigma_real * np.sqrt(dt) * Z

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S
    for i in range(n_steps):
        paths[:, i + 1] = paths[:, i] * np.exp(log_returns[:, i])

    # Option premium received at t=0
    premium = bs_call_price(S, K, T, r, sigma_impl)

    # Track P&L for all paths
    # Cash account starts with premium, earns risk-free rate
    cash = np.full(n_paths, premium)
    shares = np.zeros(n_paths)

    # Store first path details
    deltas_first = np.zeros(n_steps + 1)
    pnl_first = np.zeros(n_steps + 1)

    for i in range(n_steps):
        t_remaining = T - i * dt
        if t_remaining < 1e-10:
            t_remaining = 1e-10

        # Current spot for all paths
        S_curr = paths[:, i]

        # Compute delta using implied vol
        new_delta = np.array([
            bs_delta(s, K, t_remaining, r, sigma_impl, "call")
            for s in S_curr
        ])

        # Trade: buy/sell shares to match new delta
        trade = new_delta - shares
        cost = np.abs(trade) * S_curr * transaction_cost

        # Update cash: sell shares at current price, pay transaction costs
        cash = cash * np.exp(r * dt)  # cash earns risk-free rate
        cash -= trade * S_curr  # buy shares costs money
        cash -= cost  # transaction costs

        shares = new_delta

        # Store first path details
        deltas_first[i] = new_delta[0]
        pnl_first[i] = cash[0] + shares[0] * S_curr[0]

    # At maturity: close position
    S_T = paths[:, -1]
    cash = cash * np.exp(r * dt) if n_steps > 0 else cash

    # Liquidate shares
    cash += shares * S_T
    cost_final = np.abs(shares) * S_T * transaction_cost
    cash -= cost_final

    # Pay option payoff
    payoff = np.maximum(S_T - K, 0.0)
    cash -= payoff

    pnl = cash  # Final P&L

    deltas_first[-1] = shares[0] if n_paths > 0 else 0.0
    pnl_first[-1] = pnl[0]

    return HedgeResult(
        pnl=pnl,
        mean_pnl=float(np.mean(pnl)),
        std_pnl=float(np.std(pnl)),
        paths=paths,
        deltas=deltas_first,
        hedge_pnl_path=pnl_first,
    )


def hedge_frequency_analysis(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    frequencies: list[int] | None = None,
    n_paths: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Analyze how hedging frequency affects P&L standard deviation.

    As rebalancing frequency increases, the hedge becomes more precise
    and P&L std deviation decreases (converges to 0 in the BS limit).

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard BS parameters.
    frequencies : list of int
        Number of hedging steps to test.
    n_paths : int
        Number of simulation paths per frequency.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'n_steps', 'mean_pnl', 'std_pnl'.
    """
    if frequencies is None:
        frequencies = [12, 52, 126, 252, 504]

    results = []
    for n_steps in frequencies:
        result = simulate_delta_hedge(S, K, T, r, sigma, sigma, n_paths, n_steps, seed=seed)
        results.append({
            "n_steps": n_steps,
            "mean_pnl": result.mean_pnl,
            "std_pnl": result.std_pnl,
        })

    return pd.DataFrame(results)
