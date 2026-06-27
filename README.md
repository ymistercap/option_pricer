# Option Pricer & Risk Analyzer

A comprehensive Python application for pricing options (vanilla, exotic, and advanced models), computing Greeks, building implied volatility surfaces from real market data, simulating delta-hedging strategies, calibrating stochastic volatility models, and performing risk analysis — all exposed through an interactive Streamlit dashboard.

Built as a quantitative finance portfolio project demonstrating proficiency in stochastic calculus, numerical methods, and financial engineering.

## Features

| Module | Description |
|--------|-------------|
| **Black-Scholes** | Closed-form pricing for European calls, puts, and digitals |
| **Monte Carlo** | Standard MC + antithetic variates + control variates |
| **Finite Differences** | Crank-Nicolson and explicit schemes for the BS PDE |
| **Binomial Trees** | Cox-Ross-Rubinstein (CRR) for European and American options |
| **Heston Model** | Semi-analytical pricing + MC for stochastic volatility |
| **Merton Jump-Diffusion** | Analytical series + MC for jump processes |
| **Carr-Madan FFT** | Fast Fourier Transform pricing across strike grids |
| **SABR Model** | Hagan approximation + calibration for smile modelling |
| **Dupire Local Vol** | Local volatility extraction + MC simulation |
| **Calibration Engine** | Differential evolution + Nelder-Mead for Heston/Merton fitting |
| **Greeks** | Analytical (closed-form) and numerical (bump-and-reprice) |
| **Implied Volatility** | Newton-Raphson + Brent root-finding with Brenner-Subrahmanyam initial guess |
| **Volatility Surface** | 3D implied vol surface from real option chains (yfinance) |
| **Exotic Options** | Barrier (Reiner-Rubinstein), Asian (geometric CV), Lookback (Goldman-Sosin-Gatto) |
| **Delta-Hedging** | Dynamic hedging simulation with P&L analysis, frequency study, transaction costs |
| **Risk (VaR)** | Historical, parametric, MC VaR + Expected Shortfall + backtesting |
| **Stress Testing** | Spot/vol/rate shocks, scenario matrices |
| **QuantLib Benchmark** | Cross-validation of all custom implementations against QuantLib |

## Quick Start

```bash
# Clone
git clone https://github.com/ymistercap/option_pricer.git
cd option_pricer

# Install
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Launch dashboard
streamlit run app/app.py
```

## Project Architecture

```
option-pricer/
├── config/settings.py              # Global parameters (rates, MC defaults)
├── data/fetcher.py                 # Market data via yfinance
├── pricing/
│   ├── black_scholes.py            # BS closed-form (call, put, digital)
│   ├── monte_carlo.py              # MC + variance reduction techniques
│   ├── finite_difference.py        # Crank-Nicolson & explicit FD
│   ├── binomial_tree.py            # CRR binomial tree (European + American)
│   ├── heston.py                   # Heston stochastic vol (semi-analytical + MC)
│   ├── jump_diffusion.py           # Merton jump-diffusion (analytical + MC)
│   ├── fourier.py                  # Carr-Madan FFT pricing
│   └── quantlib_bench.py           # QuantLib cross-validation
├── greeks/
│   ├── analytical.py               # Closed-form Greeks (Δ, Γ, ν, Θ, ρ, vanna, volga)
│   └── numerical.py                # Bump-and-reprice (works with any pricer)
├── volatility/
│   ├── implied_vol.py              # IV solver (Newton-Raphson + Brent)
│   ├── surface.py                  # Vol surface construction & interpolation
│   └── smile.py                    # Smile extraction & metrics
├── models/
│   ├── local_vol.py                # Dupire local volatility + MC
│   └── sabr.py                     # SABR model (Hagan approx + calibration)
├── calibration/
│   └── calibrate.py                # Heston/Merton calibration engine
├── exotics/
│   ├── barrier.py                  # Barrier options (analytical + MC)
│   ├── asian.py                    # Asian options (geometric analytical + CV MC)
│   └── lookback.py                 # Lookback options (analytical + MC)
├── hedging/
│   └── delta_hedge.py              # Dynamic delta-hedging simulation
├── risk/
│   ├── var.py                      # VaR (historical, parametric, MC) + ES
│   └── scenarios.py                # Stress testing & scenario matrices
├── app/
│   ├── app.py                      # Streamlit entry point
│   └── pages/                      # Dashboard pages (9 pages)
│       ├── pricer.py               # BS / MC / FD comparison
│       ├── greeks_viz.py           # Greeks visualization
│       ├── vol_surface.py          # Implied vol surface
│       ├── exotics.py              # Exotic options
│       ├── hedging_sim.py          # Delta-hedging simulation
│       ├── risk.py                 # VaR & stress testing
│       ├── models.py               # Advanced models (Heston, Merton, SABR, CRR, FFT)
│       ├── calibration_page.py     # Model calibration dashboard
│       └── model_risk.py           # Model risk analysis & comparison
├── tests/                          # 256+ tests with full coverage
│   ├── test_black_scholes.py       # 44 tests: known values, parity, edge cases
│   ├── test_greeks.py              # 33 tests: analytical vs numerical, BS PDE
│   ├── test_monte_carlo.py         # 16 tests: convergence, variance reduction
│   ├── test_finite_diff.py         # 13 tests: CN vs BS, convergence
│   ├── test_quantlib_bench.py      # 15 tests: cross-validation vs QuantLib
│   ├── test_implied_vol.py         # 24 tests: round-trip, edge cases, surface
│   ├── test_exotics.py             # 18 tests: in-out parity, MC vs analytical
│   ├── test_hedging.py             # 7 tests: P&L properties, frequency analysis
│   ├── test_var.py                 # 14 tests: VaR ordering, backtesting, stress
│   ├── test_binomial_tree.py       # 15 tests: convergence, American vs European
│   ├── test_jump_diffusion.py      # 13 tests: series vs MC, zero-jump = BS
│   ├── test_heston.py              # 12 tests: semi-analytical vs MC, skew
│   ├── test_fourier.py             # 10 tests: FFT vs BS, FFT vs Heston
│   ├── test_local_vol.py           # 6 tests: flat surface, MC pricing
│   ├── test_sabr.py                # 11 tests: smile properties, calibration
│   └── test_calibration.py         # 6 tests: Heston/Merton parameter recovery
└── docs/math_notes.md              # Detailed mathematical derivations
```

## Pricing Methods Comparison

For an ATM European call (S=100, K=100, T=1y, r=5%, σ=20%):

| Method | Price | Diff vs BS | Time |
|--------|-------|-----------|------|
| Black-Scholes (analytical) | $10.4506 | — | <1 ms |
| Monte Carlo (100K paths, CV) | $10.4507 | +$0.0001 | ~50 ms |
| Finite Differences (CN 300×300) | $10.4505 | -$0.0001 | ~5 ms |
| CRR Binomial (500 steps) | $10.4506 | <$0.0001 | ~2 ms |
| Carr-Madan FFT | $10.4506 | <$0.0001 | ~5 ms |
| QuantLib (analytical) | $10.4506 | $0.0000 | <1 ms |

## Test Strategy

All tests validate:
- **Put-call parity**: C - P = S - Ke^{-rT} (model-free arbitrage)
- **Cross-method convergence**: BS ≈ MC ≈ FD ≈ CRR ≈ FFT ≈ QuantLib
- **Greeks consistency**: Analytical vs numerical, BS PDE relationship
- **IV round-trip**: BS(σ) → price → IV(price) ≈ σ
- **Exotic parity**: V_in + V_out = V_vanilla (barrier options)
- **Risk properties**: VaR₉₉ > VaR₉₅, ES > VaR
- **Model limits**: Heston(ξ→0) = BS, Merton(λ=0) = BS
- **American options**: V_american ≥ V_european (early exercise premium)
- **Calibration**: Parameter recovery from synthetic data, low RMSE

```bash
pytest tests/ -v     # Run all tests
pytest tests/ -k bs  # Run Black-Scholes tests only
```

## Stack

- **Python 3.11+** — core language
- **NumPy / SciPy** — numerical computation, optimization, FFT
- **pandas** — data manipulation
- **plotly** — interactive 3D visualizations
- **QuantLib** — industry benchmark for validation
- **Streamlit** — interactive dashboard
- **yfinance** — free market data (no API key required)
- **pytest** — test framework

## Mathematical Notes

Detailed derivations are in [`docs/math_notes.md`](docs/math_notes.md), covering:
- Black-Scholes derivation (PDE and risk-neutral approaches)
- Greeks formulas with proofs
- Crank-Nicolson scheme and stability analysis
- Monte Carlo variance reduction techniques
- Goldman-Sosin-Gatto lookback formula
- Gamma P&L formula for delta-hedging
- Cox-Ross-Rubinstein binomial tree convergence
- Merton jump-diffusion series formula
- Heston characteristic function and semi-analytical pricing
- Carr-Madan FFT framework
- Dupire local volatility formula
- SABR Hagan approximation
- Calibration methodology (differential evolution + local refinement)

## License

MIT
