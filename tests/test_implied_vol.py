"""
Tests for implied volatility computation.

Test strategy:
    - Round-trip: compute BS price with known σ, extract IV, verify σ ≈ IV
    - Newton-Raphson vs Brent consistency
    - Edge cases: deep ITM/OTM, near expiry
    - Monotonicity: higher price → higher IV
    - Surface building with synthetic data
"""

import numpy as np
import pandas as pd
import pytest

from pricing.black_scholes import bs_call_price, bs_put_price, bs_price
from volatility.implied_vol import (
    implied_vol,
    implied_vol_brent,
    implied_vol_newton,
)
from volatility.surface import build_surface, compute_iv_from_chain
from volatility.smile import extract_smile, smile_metrics

S, K, T, r = 100.0, 100.0, 1.0, 0.05


class TestImpliedVolRoundTrip:
    """Round-trip tests: BS(σ) → price → IV(price) ≈ σ."""

    @pytest.mark.parametrize("sigma", [0.10, 0.20, 0.30, 0.50, 0.80])
    def test_call_round_trip(self, sigma):
        """IV of a BS call price should recover the original volatility."""
        price = bs_call_price(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r, "call")
        assert iv is not None
        assert abs(iv - sigma) < 1e-6, f"Expected σ={sigma}, got IV={iv}"

    @pytest.mark.parametrize("sigma", [0.10, 0.20, 0.30, 0.50, 0.80])
    def test_put_round_trip(self, sigma):
        """IV of a BS put price should recover the original volatility."""
        price = bs_put_price(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r, "put")
        assert iv is not None
        assert abs(iv - sigma) < 1e-6, f"Expected σ={sigma}, got IV={iv}"

    @pytest.mark.parametrize(
        "S,K",
        [
            (100, 100),  # ATM
            (120, 100),  # ITM call
            (80, 100),   # OTM call
            (100, 80),   # Deep ITM
            (100, 130),  # OTM
        ],
    )
    def test_round_trip_various_moneyness(self, S, K):
        """Round-trip works across moneyness levels."""
        sigma = 0.25
        price = bs_call_price(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r, "call")
        assert iv is not None
        assert abs(iv - sigma) < 1e-5


class TestNewtonVsBrent:
    """Verify Newton-Raphson and Brent give the same answer."""

    def test_newton_brent_consistency(self):
        """Both methods should return the same IV."""
        sigma = 0.25
        price = bs_call_price(S, K, T, r, sigma)
        iv_nr = implied_vol_newton(price, S, K, T, r, "call")
        iv_br = implied_vol_brent(price, S, K, T, r, "call")
        assert iv_nr is not None
        assert iv_br is not None
        assert abs(iv_nr - iv_br) < 1e-6


class TestImpliedVolEdgeCases:
    """Edge cases and robustness."""

    def test_price_below_intrinsic_returns_none(self):
        """Price below intrinsic value should return None."""
        iv = implied_vol(0.01, 110, 100, T, r, "call")  # Intrinsic ≈ 15
        assert iv is None

    def test_zero_price_returns_none(self):
        """Zero price should return None."""
        iv = implied_vol(0.0, S, K, T, r, "call")
        assert iv is None

    def test_very_short_maturity(self):
        """Short maturity options should still work."""
        sigma = 0.20
        price = bs_call_price(100, 100, 0.01, r, sigma)
        iv = implied_vol(price, 100, 100, 0.01, r, "call")
        if iv is not None:
            assert abs(iv - sigma) < 0.01

    def test_high_volatility(self):
        """High vol round-trip should work."""
        sigma = 1.5
        price = bs_call_price(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r, "call")
        assert iv is not None
        assert abs(iv - sigma) < 1e-4


class TestImpliedVolMonotonicity:
    """IV should be monotonic in price."""

    def test_higher_call_price_implies_higher_iv(self):
        """For calls, higher price → higher implied vol."""
        prices = [bs_call_price(S, K, T, r, s) for s in [0.1, 0.2, 0.3, 0.4]]
        ivs = [implied_vol(p, S, K, T, r, "call") for p in prices]
        for i in range(len(ivs) - 1):
            assert ivs[i] < ivs[i + 1]


class TestVolSurface:
    """Tests for surface building with synthetic data."""

    def test_build_surface_from_synthetic_data(self):
        """Build a surface from synthetically generated IV data."""
        # Generate synthetic data: IV = 0.2 + 0.1*(K/S - 1)^2 + 0.01/T
        strikes = np.array([80, 90, 95, 100, 105, 110, 120.0])
        maturities = np.array([0.25, 0.5, 1.0, 2.0])
        K_all, T_all, IV_all = [], [], []

        for k in strikes:
            for t in maturities:
                iv = 0.2 + 0.1 * (k / S - 1) ** 2 + 0.01 / t
                K_all.append(k)
                T_all.append(t)
                IV_all.append(iv)

        K_arr = np.array(K_all)
        T_arr = np.array(T_all)
        IV_arr = np.array(IV_all)

        K_grid, T_grid, IV_grid = build_surface(K_arr, T_arr, IV_arr)

        assert K_grid.shape[0] == 50
        assert T_grid.shape[0] == 50
        assert IV_grid.shape == (50, 50)
        assert not np.any(np.isnan(IV_grid))
        assert np.all(IV_grid > 0)


class TestComputeIVFromChain:
    """Tests for computing IV from a synthetic option chain."""

    def test_synthetic_chain(self):
        """Compute IV from a synthetic chain and verify round-trip."""
        sigma_true = 0.25
        strikes = np.arange(85, 116, 5.0)
        bids, asks, volumes = [], [], []

        for k in strikes:
            price = bs_call_price(S, k, T, r, sigma_true)
            bids.append(price * 0.98)  # 2% spread
            asks.append(price * 1.02)
            volumes.append(100)

        chain = pd.DataFrame({
            "strike": strikes,
            "bid": bids,
            "ask": asks,
            "volume": volumes,
        })

        result = compute_iv_from_chain(chain, S, T, r, "call")
        assert len(result) > 0
        # All IVs should be close to sigma_true (within spread noise)
        for iv in result["implied_vol"]:
            assert abs(iv - sigma_true) < 0.02


class TestSmileMetrics:
    """Tests for smile metrics computation."""

    def test_smile_metrics_from_synthetic(self):
        """Compute smile metrics from synthetic data."""
        smile_df = pd.DataFrame({
            "moneyness": [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15],
            "implied_vol": [0.30, 0.27, 0.24, 0.22, 0.23, 0.25, 0.28],
        })
        metrics = smile_metrics(smile_df)
        assert "atm_vol" in metrics
        assert abs(metrics["atm_vol"] - 0.22) < 0.01
        assert metrics["skew_90_110"] > 0  # Put wing higher than call wing
        assert metrics["curvature"] > 0  # Smile shape
