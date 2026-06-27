"""
Market Data Fetcher Module.

Retrieves real market data from free sources:
    - yfinance: stock prices, option chains (Yahoo Finance, no API key needed)
    - Risk-free rate: configurable default or fetched from Treasury data

All data is cached locally to avoid redundant API calls.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import CACHE_DIR, DEFAULT_RISK_FREE_RATE, DEFAULT_TICKER


def _ensure_cache_dir() -> Path:
    """Create cache directory if it doesn't exist."""
    cache = Path(CACHE_DIR)
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def get_stock_data(
    ticker: str = DEFAULT_TICKER,
    period: str = "1y",
) -> pd.DataFrame:
    """
    Fetch historical stock price data.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'SPY').
    period : str
        Data period ('1mo', '3mo', '6mo', '1y', '2y', '5y').

    Returns
    -------
    pd.DataFrame
        DataFrame with OHLCV data.
    """
    import yfinance as yf

    cache_dir = _ensure_cache_dir()
    cache_file = cache_dir / f"{ticker}_{period}_prices.csv"

    # Use cache if fresh (< 1 hour old)
    if cache_file.exists():
        age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if age < 3600:
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if not df.empty:
        df.to_csv(cache_file)

    return df


def get_option_chain(
    ticker: str = DEFAULT_TICKER,
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """
    Fetch all available option chains for a ticker.

    Returns
    -------
    tuple
        (expiration_dates, chains_dict)
        chains_dict maps each expiration to a DataFrame with call and put data.
    """
    import yfinance as yf

    stock = yf.Ticker(ticker)
    expirations = list(stock.options)

    chains = {}
    for exp in expirations:
        try:
            chain = stock.option_chain(exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls["option_type"] = "call"
            puts["option_type"] = "put"
            chains[exp] = pd.concat([calls, puts], ignore_index=True)
        except Exception:
            continue

    return expirations, chains


def get_spot_price(ticker: str = DEFAULT_TICKER) -> float:
    """Get the current spot price for a ticker."""
    import yfinance as yf

    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    if hist.empty:
        raise ValueError(f"Could not fetch price for {ticker}")
    return float(hist["Close"].iloc[-1])


def get_returns(
    ticker: str = DEFAULT_TICKER,
    period: str = "1y",
) -> pd.Series:
    """
    Compute daily log-returns from historical data.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    period : str
        Data period.

    Returns
    -------
    pd.Series
        Daily log-returns.
    """
    df = get_stock_data(ticker, period)
    prices = df["Close"]
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def get_risk_free_rate() -> float:
    """
    Return the risk-free rate.

    Uses the configured default rate. For production use, this could
    fetch the current 3-month Treasury yield.

    Returns
    -------
    float
        Annualized risk-free rate.
    """
    return DEFAULT_RISK_FREE_RATE
