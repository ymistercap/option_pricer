"""
Global configuration parameters for the Option Pricer.

These defaults can be overridden at the function level or via the Streamlit UI.
"""

# Default risk-free rate (annualized, continuous compounding)
DEFAULT_RISK_FREE_RATE: float = 0.05

# Monte Carlo defaults
MC_DEFAULT_PATHS: int = 100_000
MC_DEFAULT_STEPS: int = 252  # Trading days in a year
MC_SEED: int = 42

# Finite difference defaults
FD_S_STEPS: int = 200
FD_T_STEPS: int = 200
FD_S_MAX_MULT: float = 3.0  # S_max = S_MAX_MULT * K

# Implied volatility solver
IV_MAX_ITER: int = 100
IV_TOL: float = 1e-10
IV_VOL_LOWER: float = 1e-6
IV_VOL_UPPER: float = 10.0

# Data fetching
CACHE_DIR: str = "data/cache"
DEFAULT_TICKER: str = "AAPL"
