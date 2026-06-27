"""
Dupire Local Volatility Model.

The Dupire (1994) local volatility surface σ_loc(K, T) is the unique
deterministic volatility function consistent with the observed implied
volatility surface.

Dupire's formula (in terms of total implied variance w = σ_imp²·T):

    σ²_loc(K, T) = (∂w/∂T) / (1 - (y/w)·∂w/∂y + ¼(-¼ - 1/w + y²/w²)·(∂w/∂y)²
                                   + ½·∂²w/∂y²)

where y = ln(K / F(T)),  F(T) = S·e^{rT} is the forward.

In practice we compute local vol numerically from the implied vol surface
using finite differences.

References:
    Dupire, B. (1994). "Pricing with a Smile."
    Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide."
"""

import numpy as np

from pricing.monte_carlo import MCResult


def dupire_local_vol(
    strikes: np.ndarray,
    maturities: np.ndarray,
    iv_surface: np.ndarray,
    S: float,
    r: float,
) -> np.ndarray:
    """
    Compute the Dupire local volatility surface from an implied vol surface.

    Uses central finite differences for ∂σ/∂T, ∂σ/∂K, ∂²σ/∂K².

    Parameters
    ----------
    strikes : np.ndarray
        1D array of strikes (n_K,).
    maturities : np.ndarray
        1D array of maturities (n_T,).
    iv_surface : np.ndarray
        2D array of implied volatilities, shape (n_T, n_K).
    S : float
        Spot price.
    r : float
        Risk-free rate.

    Returns
    -------
    np.ndarray
        Local volatility surface, shape (n_T, n_K).
    """
    n_T, n_K = iv_surface.shape
    local_vol = np.full_like(iv_surface, np.nan)

    for i in range(n_T):
        T = maturities[i]
        if T <= 0:
            continue

        for j in range(n_K):
            K = strikes[j]
            sigma = iv_surface[i, j]
            if np.isnan(sigma) or sigma <= 0:
                continue

            # ∂σ/∂T by finite differences
            if i == 0:
                if n_T > 1:
                    dT = maturities[1] - maturities[0]
                    dsigma_dT = (iv_surface[1, j] - iv_surface[0, j]) / dT
                else:
                    dsigma_dT = 0.0
            elif i == n_T - 1:
                dT = maturities[-1] - maturities[-2]
                dsigma_dT = (iv_surface[-1, j] - iv_surface[-2, j]) / dT
            else:
                dT = maturities[i + 1] - maturities[i - 1]
                dsigma_dT = (iv_surface[i + 1, j] - iv_surface[i - 1, j]) / dT

            # ∂σ/∂K, ∂²σ/∂K² by finite differences
            if j == 0:
                if n_K > 1:
                    dK = strikes[1] - strikes[0]
                    dsigma_dK = (iv_surface[i, 1] - iv_surface[i, 0]) / dK
                    if n_K > 2:
                        d2sigma_dK2 = (iv_surface[i, 2] - 2 * iv_surface[i, 1] + iv_surface[i, 0]) / dK**2
                    else:
                        d2sigma_dK2 = 0.0
                else:
                    dsigma_dK = 0.0
                    d2sigma_dK2 = 0.0
            elif j == n_K - 1:
                dK = strikes[-1] - strikes[-2]
                dsigma_dK = (iv_surface[i, -1] - iv_surface[i, -2]) / dK
                d2sigma_dK2 = (iv_surface[i, -1] - 2 * iv_surface[i, -2] + iv_surface[i, -3]) / dK**2 if n_K > 2 else 0.0
            else:
                dK_fwd = strikes[j + 1] - strikes[j]
                dK_bwd = strikes[j] - strikes[j - 1]
                dsigma_dK = (iv_surface[i, j + 1] - iv_surface[i, j - 1]) / (dK_fwd + dK_bwd)
                d2sigma_dK2 = (iv_surface[i, j + 1] - 2 * iv_surface[i, j] + iv_surface[i, j - 1]) / (0.5 * (dK_fwd + dK_bwd))**2

            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

            numerator = sigma**2 + 2.0 * sigma * T * (dsigma_dT + r * K * dsigma_dK)
            denominator = (1.0 + K * d1 * np.sqrt(T) * dsigma_dK) ** 2 + K**2 * T * sigma * (d2sigma_dK2 - d1 * np.sqrt(T) * dsigma_dK**2)

            if denominator > 1e-12 and numerator > 0:
                local_vol[i, j] = np.sqrt(numerator / denominator)

    return local_vol


def local_vol_mc(
    S: float,
    K: float,
    T: float,
    r: float,
    strikes_grid: np.ndarray,
    maturities_grid: np.ndarray,
    local_vol_surface: np.ndarray,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int = 42,
) -> MCResult:
    """
    Price a European option by MC simulation under local volatility.

    At each time step, the local vol is looked up from the pre-computed
    surface by bilinear interpolation.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    strikes_grid : np.ndarray
        1D array of strikes for the local vol surface.
    maturities_grid : np.ndarray
        1D array of maturities for the local vol surface.
    local_vol_surface : np.ndarray
        2D array of local vols, shape (n_T, n_K).
    option_type : str
        ``'call'`` or ``'put'``.
    n_paths : int
        Number of paths.
    n_steps : int
        Time steps per path.
    seed : int
        Random seed.

    Returns
    -------
    MCResult
        Monte Carlo result.
    """
    from scipy.interpolate import RegularGridInterpolator

    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # Build interpolator for local vol
    valid_mask = ~np.isnan(local_vol_surface)
    lv_clean = np.where(valid_mask, local_vol_surface, np.nanmean(local_vol_surface))

    interp = RegularGridInterpolator(
        (maturities_grid, strikes_grid), lv_clean,
        method="linear", bounds_error=False, fill_value=None,
    )

    spot = np.full(n_paths, S)

    for step in range(n_steps):
        t = step * dt
        points = np.column_stack([np.full(n_paths, t), spot])
        sigma_local = interp(points)
        sigma_local = np.maximum(sigma_local, 1e-6)

        Z = rng.standard_normal(n_paths)
        spot = spot * np.exp((r - 0.5 * sigma_local**2) * dt + sigma_local * np.sqrt(dt) * Z)

    disc = np.exp(-r * T)
    if option_type == "call":
        payoffs = np.maximum(spot - K, 0.0)
    else:
        payoffs = np.maximum(K - spot, 0.0)

    discounted = disc * payoffs
    price = float(np.mean(discounted))
    std_err = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))

    return MCResult(
        price=price,
        std_error=std_err,
        ci_lower=price - 1.96 * std_err,
        ci_upper=price + 1.96 * std_err,
    )
