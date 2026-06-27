"""
Volatility Smile Analysis Module.

The volatility smile is the pattern where implied volatility varies with
strike price for a fixed maturity. Key observations:

    - Equity markets typically show a skew: OTM puts have higher IV than
      OTM calls, reflecting demand for downside protection and the
      leverage effect (vol rises when stock falls).

    - FX and commodity markets often show a more symmetric smile.

    - The smile contradicts Black-Scholes (constant vol assumption) and
      motivates stochastic volatility models (Heston, SABR) and local
      volatility models (Dupire).

This module extracts and analyzes smile curves from option chain data.
"""

import numpy as np
import pandas as pd

from volatility.implied_vol import implied_vol


def extract_smile(
    chain: pd.DataFrame,
    S: float,
    T: float,
    r: float,
    option_type: str = "call",
    min_volume: int = 10,
    moneyness_range: tuple[float, float] = (0.8, 1.2),
) -> pd.DataFrame:
    """
    Extract the volatility smile for a single maturity.

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
    moneyness_range : tuple
        (min, max) K/S range.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'strike', 'moneyness', 'implied_vol'.
    """
    df = chain.copy()
    df["mid_price"] = (df["bid"] + df["ask"]) / 2.0
    df["moneyness"] = df["strike"] / S

    mask = (
        (df["volume"] >= min_volume)
        & (df["moneyness"] >= moneyness_range[0])
        & (df["moneyness"] <= moneyness_range[1])
        & (df["mid_price"] > 0.01)
    )
    df = df[mask].copy()

    ivs = []
    for _, row in df.iterrows():
        iv = implied_vol(row["mid_price"], S, row["strike"], T, r, option_type)
        ivs.append(iv)
    df["implied_vol"] = ivs

    df = df.dropna(subset=["implied_vol"])
    return df[["strike", "moneyness", "implied_vol"]].sort_values("strike").reset_index(drop=True)


def smile_metrics(smile_df: pd.DataFrame) -> dict[str, float]:
    """
    Compute summary metrics for a volatility smile.

    Parameters
    ----------
    smile_df : pd.DataFrame
        DataFrame with columns: 'moneyness', 'implied_vol'.

    Returns
    -------
    dict
        Metrics: 'atm_vol', 'skew_25d' (approx), 'min_vol', 'max_vol', 'curvature'.
    """
    if smile_df.empty:
        return {}

    moneyness = smile_df["moneyness"].values
    iv = smile_df["implied_vol"].values

    # ATM vol (closest to moneyness = 1.0)
    atm_idx = np.argmin(np.abs(moneyness - 1.0))
    atm_vol = iv[atm_idx]

    # Skew: difference between OTM put vol (low moneyness) and OTM call vol (high moneyness)
    low_idx = np.argmin(np.abs(moneyness - 0.9))
    high_idx = np.argmin(np.abs(moneyness - 1.1))
    skew = iv[low_idx] - iv[high_idx] if len(iv) > 1 else 0.0

    # Curvature: average of wings minus ATM
    curvature = 0.5 * (iv[low_idx] + iv[high_idx]) - atm_vol if len(iv) > 2 else 0.0

    return {
        "atm_vol": float(atm_vol),
        "skew_90_110": float(skew),
        "min_vol": float(iv.min()),
        "max_vol": float(iv.max()),
        "curvature": float(curvature),
    }
