"""
QuantLib Benchmark Module.

Uses QuantLib-Python as a reference implementation to validate our custom
pricing engines. QuantLib is an industry-standard open-source library for
quantitative finance.

This module wraps QuantLib's pricing engines with a clean interface matching
our own, allowing direct comparison in tests.

We benchmark against:
    - QuantLib's AnalyticEuropeanEngine (BS closed-form)
    - QuantLib's FdBlackScholesVanillaEngine (finite differences)
    - QuantLib's MCEuropeanEngine (Monte Carlo)
"""

import QuantLib as ql


def _setup_ql_option(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> tuple[ql.VanillaOption, ql.BlackScholesMertonProcess]:
    """
    Create a QuantLib vanilla option and BS process.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard BS parameters.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    tuple
        (VanillaOption, BlackScholesMertonProcess)
    """
    # Option type
    ql_type = ql.Option.Call if option_type == "call" else ql.Option.Put

    # Use a reference date and compute maturity from T
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    maturity_date = today + ql.Period(int(round(T * 365)), ql.Days)

    # Payoff and exercise
    payoff = ql.PlainVanillaPayoff(ql_type, K)
    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.VanillaOption(payoff, exercise)

    # Market data handles
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(r)), ql.Actual365Fixed())
    )
    div_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(0.0)), ql.Actual365Fixed())
    )
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(sigma)), ql.Actual365Fixed())
    )

    process = ql.BlackScholesMertonProcess(spot_handle, div_handle, rate_handle, vol_handle)
    return option, process


def ql_bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Price a European option using QuantLib's analytical BS engine.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard BS parameters.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        QuantLib BS price.
    """
    option, process = _setup_ql_option(S, K, T, r, sigma, option_type)
    engine = ql.AnalyticEuropeanEngine(process)
    option.setPricingEngine(engine)
    return option.NPV()


def ql_fd_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_S: int = 200,
    n_t: int = 200,
) -> float:
    """
    Price a European option using QuantLib's FD engine.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard BS parameters.
    option_type : str
        'call' or 'put'.
    n_S, n_t : int
        Grid resolution.

    Returns
    -------
    float
        QuantLib FD price.
    """
    option, process = _setup_ql_option(S, K, T, r, sigma, option_type)
    engine = ql.FdBlackScholesVanillaEngine(process, n_t, n_S)
    option.setPricingEngine(engine)
    return option.NPV()


def ql_mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    seed: int = 42,
) -> float:
    """
    Price a European option using QuantLib's MC engine.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard BS parameters.
    option_type : str
        'call' or 'put'.
    n_paths : int
        Number of MC paths.
    seed : int
        Random seed.

    Returns
    -------
    float
        QuantLib MC price.
    """
    option, process = _setup_ql_option(S, K, T, r, sigma, option_type)
    engine = ql.MCEuropeanEngine(
        process, "pseudorandom", timeSteps=1, requiredSamples=n_paths, seed=seed
    )
    option.setPricingEngine(engine)
    return option.NPV()


def ql_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict[str, float]:
    """
    Compute Greeks using QuantLib's analytical engine.

    Returns
    -------
    dict
        Dictionary with keys: 'delta', 'gamma', 'vega', 'theta', 'rho'.
        Vega is per unit sigma. Theta is per year.
    """
    option, process = _setup_ql_option(S, K, T, r, sigma, option_type)
    engine = ql.AnalyticEuropeanEngine(process)
    option.setPricingEngine(engine)

    return {
        "delta": option.delta(),
        "gamma": option.gamma(),
        "vega": option.vega(),  # QuantLib vega is ∂V/∂σ (per 100% vol change)
        "theta": option.thetaPerDay() * 365.0,  # Convert to per-year
        "rho": option.rho(),  # QuantLib rho is ∂V/∂r (per 100% rate change)
    }
