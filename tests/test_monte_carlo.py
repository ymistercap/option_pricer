"""
Tests for Monte Carlo option pricing.

Test strategy:
    - Convergence to BS analytical price with enough paths
    - Standard error decreases as O(1/√n)
    - Antithetic and control variate methods have lower variance than standard MC
    - Put-call parity holds for MC prices
    - Path simulation produces valid GBM paths
"""

import numpy as np
import pytest

from pricing.black_scholes import bs_call_price, bs_put_price
from pricing.monte_carlo import (
    mc_antithetic,
    mc_control_variate,
    mc_european,
    simulate_paths,
)

# Standard parameters
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
N_PATHS = 500_000  # Enough for convergence


class TestMCEuropean:
    """Tests for standard Monte Carlo pricing."""

    def test_mc_call_converges_to_bs(self):
        """MC call price should converge to BS analytical within 3 SE."""
        bs = bs_call_price(S, K, T, r, sigma)
        mc = mc_european(S, K, T, r, sigma, n_paths=N_PATHS, option_type="call")
        assert abs(mc.price - bs) < 3 * mc.std_error
        assert abs(mc.price - bs) < 0.15  # absolute tolerance

    def test_mc_put_converges_to_bs(self):
        """MC put price should converge to BS analytical within 3 SE."""
        bs = bs_put_price(S, K, T, r, sigma)
        mc = mc_european(S, K, T, r, sigma, n_paths=N_PATHS, option_type="put")
        assert abs(mc.price - bs) < 3 * mc.std_error
        assert abs(mc.price - bs) < 0.15

    def test_mc_confidence_interval_contains_bs(self):
        """95% CI should contain the BS price (with high probability)."""
        bs = bs_call_price(S, K, T, r, sigma)
        mc = mc_european(S, K, T, r, sigma, n_paths=N_PATHS)
        assert mc.ci_lower <= bs <= mc.ci_upper

    def test_mc_std_error_decreases_with_paths(self):
        """Standard error should decrease roughly as O(1/√n)."""
        mc1 = mc_european(S, K, T, r, sigma, n_paths=10_000, seed=1)
        mc2 = mc_european(S, K, T, r, sigma, n_paths=100_000, seed=1)
        # 10x more paths → ~√10 ≈ 3.16x smaller SE
        ratio = mc1.std_error / mc2.std_error
        assert 2.0 < ratio < 5.0  # Allow some slack

    def test_mc_put_call_parity(self):
        """C_mc - P_mc ≈ S - K·e^{-rT} (put-call parity)."""
        call = mc_european(S, K, T, r, sigma, n_paths=N_PATHS, option_type="call", seed=42)
        put = mc_european(S, K, T, r, sigma, n_paths=N_PATHS, option_type="put", seed=42)
        parity = S - K * np.exp(-r * T)
        assert abs((call.price - put.price) - parity) < 0.3

    def test_mc_reproducibility(self):
        """Same seed produces same result."""
        mc1 = mc_european(S, K, T, r, sigma, n_paths=10_000, seed=123)
        mc2 = mc_european(S, K, T, r, sigma, n_paths=10_000, seed=123)
        assert mc1.price == mc2.price

    def test_mc_invalid_option_type(self):
        """Invalid option type raises ValueError."""
        with pytest.raises(ValueError):
            mc_european(S, K, T, r, sigma, option_type="straddle")


class TestMCAntithetic:
    """Tests for antithetic variate Monte Carlo."""

    def test_antithetic_converges_to_bs(self):
        """Antithetic MC should converge to BS price."""
        bs = bs_call_price(S, K, T, r, sigma)
        mc = mc_antithetic(S, K, T, r, sigma, n_paths=N_PATHS)
        assert abs(mc.price - bs) < 0.15

    def test_antithetic_lower_variance_than_standard(self):
        """Antithetic should have lower standard error than standard MC."""
        std = mc_european(S, K, T, r, sigma, n_paths=N_PATHS, seed=1)
        anti = mc_antithetic(S, K, T, r, sigma, n_paths=N_PATHS, seed=1)
        assert anti.std_error < std.std_error


class TestMCControlVariate:
    """Tests for control variate Monte Carlo."""

    def test_control_variate_converges_to_bs(self):
        """Control variate MC should converge to BS price."""
        bs = bs_call_price(S, K, T, r, sigma)
        mc = mc_control_variate(S, K, T, r, sigma, n_paths=N_PATHS)
        assert abs(mc.price - bs) < 0.15

    def test_control_variate_lower_variance_than_standard(self):
        """Control variate should have lower standard error than standard MC."""
        std = mc_european(S, K, T, r, sigma, n_paths=N_PATHS, seed=1)
        cv = mc_control_variate(S, K, T, r, sigma, n_paths=N_PATHS, seed=1)
        assert cv.std_error < std.std_error


class TestSimulatePaths:
    """Tests for full path simulation."""

    def test_path_shape(self):
        """Paths should have correct shape (n_paths, n_steps+1)."""
        paths = simulate_paths(S, T, r, sigma, n_paths=100, n_steps=50, seed=42)
        assert paths.shape == (100, 51)

    def test_path_initial_value(self):
        """All paths should start at S."""
        paths = simulate_paths(S, T, r, sigma, n_paths=100, n_steps=50, seed=42)
        np.testing.assert_allclose(paths[:, 0], S)

    def test_path_positive(self):
        """All prices on all paths should be positive (GBM property)."""
        paths = simulate_paths(S, T, r, sigma, n_paths=1000, n_steps=252, seed=42)
        assert np.all(paths > 0)

    def test_path_terminal_mean(self):
        """Mean terminal price ≈ S·e^{rT} (risk-neutral drift)."""
        n = 200_000
        paths = simulate_paths(S, T, r, sigma, n_paths=n, n_steps=1, seed=42)
        mean_ST = np.mean(paths[:, -1])
        expected = S * np.exp(r * T)
        assert abs(mean_ST - expected) / expected < 0.01  # 1% tolerance

    def test_path_terminal_distribution(self):
        """Terminal log-returns should be approximately normal."""
        paths = simulate_paths(S, T, r, sigma, n_paths=100_000, n_steps=252, seed=42)
        log_returns = np.log(paths[:, -1] / S)
        expected_mean = (r - 0.5 * sigma**2) * T
        expected_std = sigma * np.sqrt(T)
        assert abs(np.mean(log_returns) - expected_mean) < 0.01
        assert abs(np.std(log_returns) - expected_std) < 0.01
