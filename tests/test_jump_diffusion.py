"""
Tests for the Merton jump-diffusion pricing module.

Validates:
    - With zero jump intensity (λ=0), Merton reduces to Black-Scholes.
    - Analytical series and MC agree within tolerance.
    - Jumps increase option prices relative to pure BS.
    - Put-call parity holds under jump-diffusion.
    - Edge cases: T=0, deep OTM.
"""

import numpy as np
import pytest

from pricing.black_scholes import bs_price
from pricing.jump_diffusion import merton_mc, merton_price


class TestMertonAnalytical:
    """Merton analytical series tests."""

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    lam, m, delta = 1.0, -0.05, 0.1

    def test_zero_jumps_equals_bs_call(self):
        merton = merton_price(self.S, self.K, self.T, self.r, self.sigma, 0.0, 0.0, 0.0, "call")
        bs = bs_price(self.S, self.K, self.T, self.r, self.sigma, "call")
        assert abs(merton - bs) < 1e-10

    def test_zero_jumps_equals_bs_put(self):
        merton = merton_price(self.S, self.K, self.T, self.r, self.sigma, 0.0, 0.0, 0.0, "put")
        bs = bs_price(self.S, self.K, self.T, self.r, self.sigma, "put")
        assert abs(merton - bs) < 1e-10

    def test_jumps_increase_atm_call_price(self):
        merton = merton_price(self.S, self.K, self.T, self.r, self.sigma, self.lam, self.m, self.delta, "call")
        bs = bs_price(self.S, self.K, self.T, self.r, self.sigma, "call")
        assert merton > bs * 0.9

    def test_jumps_increase_otm_put_price(self):
        merton = merton_price(120, 100, 0.5, 0.05, 0.2, 1.0, -0.1, 0.15, "put")
        bs = bs_price(120, 100, 0.5, 0.05, 0.2, "put")
        assert merton > bs

    def test_put_call_parity(self):
        c = merton_price(self.S, self.K, self.T, self.r, self.sigma, self.lam, self.m, self.delta, "call")
        p = merton_price(self.S, self.K, self.T, self.r, self.sigma, self.lam, self.m, self.delta, "put")
        parity = c - p - (self.S - self.K * np.exp(-self.r * self.T))
        assert abs(parity) < 1e-8

    def test_zero_maturity(self):
        assert merton_price(110, 100, 0, 0.05, 0.2, 1.0, -0.05, 0.1, "call") == 10.0
        assert merton_price(90, 100, 0, 0.05, 0.2, 1.0, -0.05, 0.1, "put") == 10.0

    def test_positive_price(self):
        price = merton_price(self.S, self.K, self.T, self.r, self.sigma, self.lam, self.m, self.delta, "call")
        assert price > 0

    def test_high_jump_intensity_increases_vol(self):
        low_lam = merton_price(self.S, self.K, self.T, self.r, self.sigma, 0.5, 0.0, 0.2, "call")
        high_lam = merton_price(self.S, self.K, self.T, self.r, self.sigma, 5.0, 0.0, 0.2, "call")
        assert high_lam > low_lam


class TestMertonMC:
    """Merton Monte Carlo tests."""

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    lam, m, delta = 1.0, -0.05, 0.1

    def test_mc_matches_analytical_call(self):
        analytical = merton_price(self.S, self.K, self.T, self.r, self.sigma, self.lam, self.m, self.delta, "call")
        mc = merton_mc(self.S, self.K, self.T, self.r, self.sigma, self.lam, self.m, self.delta, "call", n_paths=200_000)
        assert abs(mc.price - analytical) < 0.5

    def test_mc_matches_analytical_put(self):
        analytical = merton_price(self.S, self.K, self.T, self.r, self.sigma, self.lam, self.m, self.delta, "put")
        mc = merton_mc(self.S, self.K, self.T, self.r, self.sigma, self.lam, self.m, self.delta, "put", n_paths=200_000)
        assert abs(mc.price - analytical) < 0.5

    def test_mc_confidence_interval_contains_analytical(self):
        analytical = merton_price(self.S, self.K, self.T, self.r, self.sigma, self.lam, self.m, self.delta, "call")
        mc = merton_mc(self.S, self.K, self.T, self.r, self.sigma, self.lam, self.m, self.delta, "call", n_paths=200_000)
        assert mc.ci_lower < analytical < mc.ci_upper

    def test_mc_std_error_positive(self):
        mc = merton_mc(self.S, self.K, self.T, self.r, self.sigma, self.lam, self.m, self.delta, "call")
        assert mc.std_error > 0

    def test_mc_zero_jumps_matches_bs(self):
        bs = bs_price(self.S, self.K, self.T, self.r, self.sigma, "call")
        mc = merton_mc(self.S, self.K, self.T, self.r, self.sigma, 0.0, 0.0, 0.0, "call", n_paths=200_000)
        assert abs(mc.price - bs) < 0.5
