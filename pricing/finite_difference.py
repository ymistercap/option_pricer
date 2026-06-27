"""
Finite Difference Option Pricing Module.

Solves the Black-Scholes PDE numerically on a discrete grid (S, t):

    ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0

We transform to a uniform grid by working in log-space (x = ln(S)), which
simplifies the discretization and avoids non-uniform spacing issues.

Under x = ln(S), the PDE becomes:
    ∂V/∂t + ½σ²∂²V/∂x² + (r - ½σ²)∂V/∂x - rV = 0

Schemes Implemented:
    1. Crank-Nicolson (CN): Averages implicit and explicit schemes.
       Second-order accurate in both time O(dt²) and space O(dx²).
       Unconditionally stable. This is the industry standard.

    2. Explicit: Forward-in-time, centered-in-space. First-order in time.
       Conditionally stable: requires dt ≤ dx²/(σ²) (CFL condition).
       Included for pedagogical comparison.

The resulting tridiagonal systems are solved via Thomas algorithm (scipy's
solve_banded or direct tridiagonal solver).

Boundary Conditions (European options):
    - Call: V(S_max, t) ≈ S_max - K·e^{-r(T-t)},  V(0, t) = 0
    - Put:  V(0, t) = K·e^{-r(T-t)},  V(S_max, t) = 0

References:
    Wilmott, P. (2006). "Paul Wilmott on Quantitative Finance."
    Tavella, D. & Randall, C. (2000). "Pricing Financial Instruments: The FD Method."
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded


@dataclass
class FDResult:
    """Container for finite difference pricing results.

    Attributes
    ----------
    price : float
        Option price at the current spot.
    grid_S : np.ndarray
        Spot price grid.
    grid_t : np.ndarray
        Time grid.
    grid_V : np.ndarray
        Full price surface V(S, t), shape (n_S, n_t).
    """

    price: float
    grid_S: np.ndarray
    grid_t: np.ndarray
    grid_V: np.ndarray


def _solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system using scipy's banded solver (Thomas algorithm)."""
    n = len(diag)
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]  # superdiagonal
    ab[1, :] = diag          # main diagonal
    ab[2, :-1] = lower[1:]   # subdiagonal
    return solve_banded((1, 1), ab, rhs)


def crank_nicolson(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_S: int = 200,
    n_t: int = 200,
    S_max_mult: float = 3.0,
) -> FDResult:
    """
    Price a European option using the Crank-Nicolson finite difference scheme.

    The CN scheme averages the explicit and implicit discretizations:
        (V^{n+1} - V^n)/dt = ½[L·V^{n+1} + L·V^n]

    where L is the spatial differential operator. This yields a tridiagonal
    system at each time step:
        A · V^{n+1} = B · V^n + boundary terms

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    option_type : str
        'call' or 'put'.
    n_S : int
        Number of spatial grid points.
    n_t : int
        Number of time steps.
    S_max_mult : float
        S_max = S_max_mult * K.

    Returns
    -------
    FDResult
        Price and full grid data.
    """
    S_max = S_max_mult * K
    dS = S_max / n_S
    dt = T / n_t

    # Spatial grid
    S_grid = np.linspace(0, S_max, n_S + 1)
    t_grid = np.linspace(0, T, n_t + 1)

    # Initialize with payoff at maturity
    if option_type == "call":
        V = np.maximum(S_grid - K, 0.0)
    elif option_type == "put":
        V = np.maximum(K - S_grid, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    # Store full grid (columns = time steps, from T back to 0)
    V_grid = np.zeros((n_S + 1, n_t + 1))
    V_grid[:, -1] = V.copy()

    # Interior indices: 1 to n_S-1
    j = np.arange(1, n_S)
    Sj = j * dS

    # Coefficients for the tridiagonal system
    alpha = 0.5 * dt * (sigma**2 * j**2 - r * j)
    beta_val = -dt * (sigma**2 * j**2 + r)
    gamma_coeff = 0.5 * dt * (sigma**2 * j**2 + r * j)

    # LHS matrix (implicit part): (I - ½·L·dt)
    a_lower = -0.5 * alpha      # sub-diagonal
    a_diag = 1.0 - 0.5 * beta_val  # main diagonal
    a_upper = -0.5 * gamma_coeff    # super-diagonal

    # RHS matrix coefficients: (I + ½·L·dt)
    b_lower = 0.5 * alpha
    b_diag = 1.0 + 0.5 * beta_val
    b_upper = 0.5 * gamma_coeff

    # Time-stepping (backward from T to 0)
    for i in range(n_t - 1, -1, -1):
        tau = T - t_grid[i]  # time remaining

        # Build RHS: B · V_interior + boundary corrections
        V_int = V[1:-1]
        rhs = b_diag * V_int
        rhs[:-1] += b_upper[:-1] * V[2:-1]
        rhs[1:] += b_lower[1:] * V[1:-2]

        # Boundary conditions
        if option_type == "call":
            V_low = 0.0
            V_high = S_max - K * np.exp(-r * (T - t_grid[i]))
        else:
            V_low = K * np.exp(-r * (T - t_grid[i]))
            V_high = 0.0

        # Add boundary contributions
        rhs[0] += (0.5 * alpha[0]) * V_low + (0.5 * alpha[0]) * V_low
        rhs[-1] += (0.5 * gamma_coeff[-1]) * V_high + (0.5 * gamma_coeff[-1]) * V_high

        # Solve tridiagonal system
        V_new = _solve_tridiagonal(a_lower, a_diag, a_upper, rhs)

        V[0] = V_low
        V[-1] = V_high
        V[1:-1] = V_new
        V_grid[:, i] = V.copy()

    # Interpolate to get price at the exact spot S
    price = float(np.interp(S, S_grid, V))

    return FDResult(price=price, grid_S=S_grid, grid_t=t_grid, grid_V=V_grid)


def explicit_fd(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_S: int = 200,
    n_t: int = 5000,
    S_max_mult: float = 3.0,
) -> FDResult:
    """
    Price a European option using the explicit finite difference scheme.

    The explicit scheme is forward-in-time: V^n is computed directly from V^{n+1}
    without solving a linear system. However, it is only conditionally stable:
    the CFL condition dt ≤ 1/(σ²·j_max² + r) must hold.

    Included for pedagogical comparison with Crank-Nicolson.

    Parameters
    ----------
    S, K, T, r, sigma, option_type : standard BS params
    n_S : int
        Number of spatial grid points.
    n_t : int
        Number of time steps (must be large enough for stability).
    S_max_mult : float
        S_max = S_max_mult * K.

    Returns
    -------
    FDResult
        Price and full grid data.
    """
    S_max = S_max_mult * K
    dS = S_max / n_S
    dt = T / n_t

    S_grid = np.linspace(0, S_max, n_S + 1)
    t_grid = np.linspace(0, T, n_t + 1)

    if option_type == "call":
        V = np.maximum(S_grid - K, 0.0)
    elif option_type == "put":
        V = np.maximum(K - S_grid, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    V_grid = np.zeros((n_S + 1, n_t + 1))
    V_grid[:, -1] = V.copy()

    j = np.arange(1, n_S)

    alpha = 0.5 * dt * (sigma**2 * j**2 - r * j)
    beta_val = 1.0 - dt * (sigma**2 * j**2 + r)
    gamma_coeff = 0.5 * dt * (sigma**2 * j**2 + r * j)

    for i in range(n_t - 1, -1, -1):
        V_new = alpha * V[:-2] + beta_val * V[1:-1] + gamma_coeff * V[2:]

        if option_type == "call":
            V_low = 0.0
            V_high = S_max - K * np.exp(-r * (T - t_grid[i]))
        else:
            V_low = K * np.exp(-r * (T - t_grid[i]))
            V_high = 0.0

        V[0] = V_low
        V[-1] = V_high
        V[1:-1] = V_new
        V_grid[:, i] = V.copy()

    price = float(np.interp(S, S_grid, V))
    return FDResult(price=price, grid_S=S_grid, grid_t=t_grid, grid_V=V_grid)


def fd_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    method: str = "crank-nicolson",
    n_S: int = 200,
    n_t: int = 200,
) -> float:
    """
    Convenience wrapper: price a European option via finite differences.

    Parameters
    ----------
    S, K, T, r, sigma, option_type : standard BS params
    method : str
        'crank-nicolson' or 'explicit'.
    n_S, n_t : int
        Grid resolution.

    Returns
    -------
    float
        Option price.
    """
    if method == "crank-nicolson":
        result = crank_nicolson(S, K, T, r, sigma, option_type, n_S, n_t)
    elif method == "explicit":
        result = explicit_fd(S, K, T, r, sigma, option_type, n_S, n_t)
    else:
        raise ValueError(f"method must be 'crank-nicolson' or 'explicit', got '{method}'")
    return result.price
