"""
Tests for Value at Risk and stress testing.

Test strategy:
    - VaR is positive (represents a loss)
    - VaR 99% > VaR 95% (higher confidence = larger VaR)
    - ES > VaR (expected shortfall always exceeds VaR)
    - Parametric VaR close to historical for normal-like data
    - Scenario matrix has correct shape
    - Backtesting: violation rate ≈ 1 - confidence
"""

import numpy as np
import pandas as pd
import pytest

from risk.var import (
    historical_var,
    parametric_var,
    monte_carlo_var,
    expected_shortfall,
    backtest_var,
)
from risk.scenarios import spot_stress, vol_stress, rate_stress, scenario_matrix


class TestHistoricalVaR:
    """Tests for historical VaR."""

    @pytest.fixture
    def returns(self):
        rng = np.random.default_rng(42)
        return rng.normal(0.0005, 0.02, 1000)

    def test_var_positive(self, returns):
        """VaR should be a positive number."""
        var = historical_var(returns, confidence=0.95)
        assert var > 0

    def test_var_99_greater_than_95(self, returns):
        """VaR at 99% > VaR at 95%."""
        var_95 = historical_var(returns, confidence=0.95)
        var_99 = historical_var(returns, confidence=0.99)
        assert var_99 > var_95

    def test_var_10d_greater_than_1d(self, returns):
        """10-day VaR > 1-day VaR (√10 scaling)."""
        var_1d = historical_var(returns, confidence=0.95, horizon=1)
        var_10d = historical_var(returns, confidence=0.95, horizon=10)
        assert var_10d > var_1d
        assert abs(var_10d / var_1d - np.sqrt(10)) < 0.01


class TestParametricVaR:
    """Tests for parametric (Gaussian) VaR."""

    @pytest.fixture
    def returns(self):
        rng = np.random.default_rng(42)
        return rng.normal(0.0, 0.02, 10000)

    def test_parametric_close_to_historical(self, returns):
        """For normal data, parametric and historical VaR should be close."""
        h_var = historical_var(returns, confidence=0.95)
        p_var = parametric_var(returns, confidence=0.95)
        assert abs(h_var - p_var) / h_var < 0.1  # Within 10%


class TestMonteCarloVaR:
    """Tests for Monte Carlo VaR."""

    @pytest.fixture
    def returns(self):
        rng = np.random.default_rng(42)
        return rng.normal(0.0, 0.02, 1000)

    def test_mc_var_close_to_parametric(self, returns):
        """MC VaR should be close to parametric for normal data."""
        mc_var = monte_carlo_var(returns, confidence=0.95)
        p_var = parametric_var(returns, confidence=0.95)
        assert abs(mc_var - p_var) / p_var < 0.15


class TestExpectedShortfall:
    """Tests for Expected Shortfall."""

    @pytest.fixture
    def returns(self):
        rng = np.random.default_rng(42)
        return rng.normal(0.0, 0.02, 1000)

    def test_es_greater_than_var(self, returns):
        """ES should always be >= VaR (it averages the tail)."""
        var = historical_var(returns, confidence=0.95)
        es = expected_shortfall(returns, confidence=0.95)
        assert es >= var

    def test_es_positive(self, returns):
        """ES should be positive."""
        es = expected_shortfall(returns, confidence=0.95)
        assert es > 0


class TestBacktest:
    """Tests for VaR backtesting."""

    def test_backtest_violation_rate(self):
        """Violation rate should be approximately 1 - confidence."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, 2000)
        df = backtest_var(returns, confidence=0.95, window=252)
        violation_rate = df["violation"].mean()
        # Should be near 5% (allow wide range due to randomness)
        assert 0.01 < violation_rate < 0.15

    def test_backtest_output_shape(self):
        """Backtest output should have correct columns."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, 500)
        df = backtest_var(returns, confidence=0.95, window=252)
        assert "return" in df.columns
        assert "var" in df.columns
        assert "violation" in df.columns
        assert len(df) == 500 - 252


class TestStressTests:
    """Tests for scenario analysis."""

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    def test_spot_stress_shape(self):
        """Spot stress test returns correct shape."""
        df = spot_stress(self.S, self.K, self.T, self.r, self.sigma)
        assert len(df) == 7
        assert "price" in df.columns
        assert "delta" in df.columns

    def test_vol_stress_shape(self):
        """Vol stress test returns correct shape."""
        df = vol_stress(self.S, self.K, self.T, self.r, self.sigma)
        assert len(df) == 7

    def test_rate_stress_shape(self):
        """Rate stress test returns correct shape."""
        df = rate_stress(self.S, self.K, self.T, self.r, self.sigma)
        assert len(df) == 5

    def test_scenario_matrix_shape(self):
        """Scenario matrix has correct dimensions."""
        spots, vols, matrix = scenario_matrix(
            self.S, self.K, self.T, self.r, self.sigma
        )
        assert len(spots) == 9
        assert len(vols) == 7
        assert matrix.shape == (7, 9)

    def test_scenario_matrix_positive_prices(self):
        """All prices in scenario matrix should be non-negative."""
        _, _, matrix = scenario_matrix(
            self.S, self.K, self.T, self.r, self.sigma
        )
        assert np.all(matrix >= 0)
