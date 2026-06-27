"""
Tests for the Heston stochastic volatility pricing module.

Validates:
    - When ξ→0 (no vol-of-vol), Heston reduces to Black-Scholes.
    - Semi-analytical and MC agree within tolerance.
    - Put-call parity holds.
    - Prices are positive and sensible.
    - Negative correlation produces a skew (OTM puts more expensive).
"""

import numpy as np
import pytest

from pricing.black_scholes import bs_price
from pricing.heston import heston_mc, heston_price


class TestHestonSemiAnalytical:
    """Heston semi-analytical tests."""

    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7

    def test_low_volvol_matches_bs_call(self):
        price = heston_price(self.S, self.K, self.T, self.r, 0.04, 2.0, 0.04, 1e-6, 0.0, "call")
        bs = bs_price(self.S, self.K, self.T, self.r, 0.2, "call")
        assert abs(price - bs) < 0.15

    def test_low_volvol_matches_bs_put(self):
        price = heston_price(self.S, self.K, self.T, self.r, 0.04, 2.0, 0.04, 1e-6, 0.0, "put")
        bs = bs_price(self.S, self.K, self.T, self.r, 0.2, "put")
        assert abs(price - bs) < 0.15

    def test_put_call_parity(self):
        c = heston_price(self.S, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "call")
        p = heston_price(self.S, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "put")
        parity = c - p - (self.S - self.K * np.exp(-self.r * self.T))
        assert abs(parity) < 0.01

    def test_positive_call_price(self):
        price = heston_price(self.S, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "call")
        assert price > 0

    def test_positive_put_price(self):
        price = heston_price(self.S, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "put")
        assert price > 0

    def test_zero_maturity(self):
        assert heston_price(110, 100, 0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, "call") == 10.0
        assert heston_price(90, 100, 0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, "put") == 10.0

    def test_negative_rho_skew(self):
        otm_put = heston_price(100, 80, 0.5, 0.05, 0.04, 2.0, 0.04, 0.5, -0.8, "put")
        bs_put = bs_price(100, 80, 0.5, 0.05, 0.2, "put")
        assert otm_put > bs_put

    def test_higher_volvol_increases_otm_prices(self):
        p_low = heston_price(100, 80, 0.5, 0.05, 0.04, 2.0, 0.04, 0.1, -0.7, "put")
        p_high = heston_price(100, 80, 0.5, 0.05, 0.04, 2.0, 0.04, 0.8, -0.7, "put")
        assert p_high > p_low


class TestHestonMC:
    """Heston Monte Carlo tests."""

    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7

    def test_mc_matches_analytical_call(self):
        analytical = heston_price(self.S, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "call")
        mc = heston_mc(self.S, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "call", n_paths=200_000)
        assert abs(mc.price - analytical) < 0.5

    def test_mc_matches_analytical_put(self):
        analytical = heston_price(self.S, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "put")
        mc = heston_mc(self.S, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "put", n_paths=200_000)
        assert abs(mc.price - analytical) < 0.5

    def test_mc_std_error_positive(self):
        mc = heston_mc(self.S, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "call")
        assert mc.std_error > 0

    def test_mc_confidence_interval(self):
        mc = heston_mc(self.S, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "call")
        assert mc.ci_lower < mc.price < mc.ci_upper
