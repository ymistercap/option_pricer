"""
Volatility Surface Construction Module.

A volatility surface maps (strike K, maturity T) → implied volatility σ(K,T).

In a pure Black-Scholes world, σ would be constant. In reality, the surface
exhibits:
    - Smile: vol is higher for OTM puts and OTM calls (wings) vs ATM
    - Skew: OTM puts tend to have higher vol than OTM calls (equity markets)
    - Term structure: ATM vol varies with maturity

This module:
    1. Fetches real option chain data via yfinance
    2. Computes implied vol for each (K, T) pair
    3. Filters outliers (low volume, wide spreads)
    4. Interpolates to build a smooth surface
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from volatility.implied_vol import implied_vol


def compute_iv_from_chain(
    chain: pd.DataFrame,
    S: float,
    T: float,
    r: float,
    option_type: str = "call",
    min_volume: int = 10,
    max_spread_ratio: float = 0.5,
    moneyness_range: tuple[float, float] = (0.8, 1.2),
) -> pd.DataFrame:
    """
    Compute implied volatilities from an option chain DataFrame.

    Filters are applied to remove unreliable data points:
    - Low volume options (illiquid, unreliable prices)
    - Wide bid-ask spreads (uncertain mid-price)
    - Extreme moneyness (numerical difficulties)

    Parameters
    ----------
    chain : pd.DataFrame
        Option chain with columns: 'strike', 'bid', 'ask', 'volume'.
    S : float
        Current spot price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate.
    option_type : str
        'call' or 'put'.
    min_volume : int
        Minimum volume filter.
    max_spread_ratio : float
        Maximum (ask-bid)/mid ratio.
    moneyness_range : tuple
        (min, max) K/S range to include.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'strike', 'mid_price', 'implied_vol', 'moneyness'.
    """
    df = chain.copy()

    # Compute mid price
    df["mid_price"] = (df["bid"] + df["ask"]) / 2.0
    df["moneyness"] = df["strike"] / S
    df["spread_ratio"] = (df["ask"] - df["bid"]) / df["mid_price"].clip(lower=0.01)

    # Apply filters
    mask = (
        (df["volume"] >= min_volume)
        & (df["spread_ratio"] <= max_spread_ratio)
        & (df["moneyness"] >= moneyness_range[0])
        & (df["moneyness"] <= moneyness_range[1])
        & (df["mid_price"] > 0.01)
    )
    df = df[mask].copy()

    # Compute implied vol for each row
    ivs = []
    for _, row in df.iterrows():
        iv = implied_vol(row["mid_price"], S, row["strike"], T, r, option_type)
        ivs.append(iv)
    df["implied_vol"] = ivs

    # Remove failed computations
    df = df.dropna(subset=["implied_vol"])
    df = df[df["implied_vol"] > 0.001]

    return df[["strike", "mid_price", "implied_vol", "moneyness"]].reset_index(drop=True)


def build_surface(
    strikes: np.ndarray,
    maturities: np.ndarray,
    ivs: np.ndarray,
    n_strike_grid: int = 50,
    n_maturity_grid: int = 50,
    method: str = "cubic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate a volatility surface from scattered (K, T, IV) data.

    Uses scipy.interpolate.griddata to create a regular grid.

    Parameters
    ----------
    strikes : np.ndarray
        Strike prices (or moneyness).
    maturities : np.ndarray
        Times to maturity.
    ivs : np.ndarray
        Implied volatilities.
    n_strike_grid : int
        Number of grid points along the strike axis.
    n_maturity_grid : int
        Number of grid points along the maturity axis.
    method : str
        Interpolation method: 'linear', 'nearest', 'cubic'.

    Returns
    -------
    tuple of (K_grid, T_grid, IV_grid)
        K_grid : np.ndarray, shape (n_strike_grid,)
        T_grid : np.ndarray, shape (n_maturity_grid,)
        IV_grid : np.ndarray, shape (n_maturity_grid, n_strike_grid)
    """
    K_grid = np.linspace(strikes.min(), strikes.max(), n_strike_grid)
    T_grid = np.linspace(maturities.min(), maturities.max(), n_maturity_grid)

    K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)
    points = np.column_stack([strikes, maturities])

    IV_grid = griddata(points, ivs, (K_mesh, T_mesh), method=method)

    # Fill NaN with nearest neighbor
    if np.any(np.isnan(IV_grid)):
        IV_nearest = griddata(points, ivs, (K_mesh, T_mesh), method="nearest")
        IV_grid = np.where(np.isnan(IV_grid), IV_nearest, IV_grid)

    return K_grid, T_grid, IV_grid


def atm_term_structure(
    maturities: np.ndarray,
    atm_ivs: np.ndarray,
) -> pd.DataFrame:
    """
    Build the ATM volatility term structure.

    Parameters
    ----------
    maturities : np.ndarray
        Times to maturity in years.
    atm_ivs : np.ndarray
        ATM implied volatilities for each maturity.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'maturity', 'atm_iv'.
    """
    return pd.DataFrame({"maturity": maturities, "atm_iv": atm_ivs}).sort_values("maturity")
