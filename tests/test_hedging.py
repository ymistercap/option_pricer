"""
Tests for delta-hedging simulation.

Test strategy:
    - In a perfect BS world (sigma_real = sigma_impl), mean P&L ≈ 0
    - P&L std decreases with more frequent rebalancing
    - When sigma_real > sigma_impl, short gamma position loses on average
    - When sigma_real < sigma_impl, short gamma position profits on average
"""

import numpy as np
import pytest

from hedging.delta_hedge import simulate_delta_hedge, hedge_frequency_analysis

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


class TestDeltaHedge:
    """Tests for delta-hedging simulation."""

    def test_perfect_hedge_mean_pnl_near_zero(self):
        """In BS world, mean P&L should be close to zero."""
        result = simulate_delta_hedge(
            S, K, T, r, sigma, sigma, n_paths=5000, n_steps=252
        )
        assert abs(result.mean_pnl) < 1.0, f"Mean P&L = {result.mean_pnl}"

    def test_pnl_shape(self):
        """P&L array should have correct shape."""
        result = simulate_delta_hedge(S, K, T, r, sigma, n_paths=100, n_steps=50)
        assert result.pnl.shape == (100,)
        assert result.paths.shape == (100, 51)
        assert result.deltas.shape == (51,)

    def test_higher_real_vol_causes_loss(self):
        """Short gamma + realized vol > implied vol → negative mean P&L."""
        result = simulate_delta_hedge(
            S, K, T, r, sigma_impl=0.20, sigma_real=0.30,
            n_paths=5000, n_steps=252
        )
        # Short gamma loses when real vol is higher
        assert result.mean_pnl < 0.0

    def test_lower_real_vol_causes_profit(self):
        """Short gamma + realized vol < implied vol → positive mean P&L."""
        result = simulate_delta_hedge(
            S, K, T, r, sigma_impl=0.30, sigma_real=0.20,
            n_paths=5000, n_steps=252
        )
        assert result.mean_pnl > 0.0

    def test_transaction_costs_reduce_pnl(self):
        """Transaction costs should reduce P&L."""
        no_cost = simulate_delta_hedge(
            S, K, T, r, sigma, n_paths=1000, n_steps=252,
            transaction_cost=0.0, seed=42
        )
        with_cost = simulate_delta_hedge(
            S, K, T, r, sigma, n_paths=1000, n_steps=252,
            transaction_cost=0.005, seed=42
        )
        assert with_cost.mean_pnl < no_cost.mean_pnl

    def test_reproducibility(self):
        """Same seed produces same result."""
        r1 = simulate_delta_hedge(S, K, T, r, sigma, n_paths=100, seed=42)
        r2 = simulate_delta_hedge(S, K, T, r, sigma, n_paths=100, seed=42)
        np.testing.assert_array_equal(r1.pnl, r2.pnl)


class TestHedgeFrequencyAnalysis:
    """Tests for hedging frequency analysis."""

    def test_std_decreases_with_frequency(self):
        """P&L std should generally decrease with more rebalancing."""
        df = hedge_frequency_analysis(
            S, K, T, r, sigma,
            frequencies=[12, 52, 252],
            n_paths=3000
        )
        assert len(df) == 3
        # Allow some noise, but trend should be clear
        assert df.iloc[-1]["std_pnl"] < df.iloc[0]["std_pnl"]
