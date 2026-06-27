"""
Tests for the CRR binomial tree pricing module.

Validates:
    - European prices converge to Black-Scholes analytical values.
    - American put >= European put (early exercise premium).
    - American call on non-dividend stock == European call.
    - Put-call parity holds for European prices.
    - Edge cases: T=0, deep ITM/OTM.
    - Convergence improves with more steps.
"""

import numpy as np
import pytest

from pricing.binomial_tree import crr_american, crr_european
from pricing.black_scholes import bs_price


class TestCRREuropean:
    """CRR European pricing tests."""

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    def test_call_atm_matches_bs(self):
        crr = crr_european(self.S, self.K, self.T, self.r, self.sigma, "call", 500)
        bs = bs_price(self.S, self.K, self.T, self.r, self.sigma, "call")
        assert abs(crr - bs) < 0.05

    def test_put_atm_matches_bs(self):
        crr = crr_european(self.S, self.K, self.T, self.r, self.sigma, "put", 500)
        bs = bs_price(self.S, self.K, self.T, self.r, self.sigma, "put")
        assert abs(crr - bs) < 0.05

    def test_call_itm_matches_bs(self):
        crr = crr_european(120, 100, 0.5, 0.05, 0.25, "call", 400)
        bs = bs_price(120, 100, 0.5, 0.05, 0.25, "call")
        assert abs(crr - bs) < 0.05

    def test_put_otm_matches_bs(self):
        crr = crr_european(120, 100, 0.5, 0.05, 0.25, "put", 400)
        bs = bs_price(120, 100, 0.5, 0.05, 0.25, "put")
        assert abs(crr - bs) < 0.05

    def test_put_call_parity(self):
        c = crr_european(self.S, self.K, self.T, self.r, self.sigma, "call", 300)
        p = crr_european(self.S, self.K, self.T, self.r, self.sigma, "put", 300)
        parity = c - p - (self.S - self.K * np.exp(-self.r * self.T))
        assert abs(parity) < 0.05

    def test_zero_maturity_call(self):
        assert crr_european(110, 100, 0, 0.05, 0.2, "call") == 10.0

    def test_zero_maturity_put(self):
        assert crr_european(90, 100, 0, 0.05, 0.2, "put") == 10.0

    def test_deep_otm_call_near_zero(self):
        price = crr_european(50, 100, 0.25, 0.05, 0.2, "call", 200)
        assert price < 0.01

    def test_convergence_improves_with_steps(self):
        bs = bs_price(self.S, self.K, self.T, self.r, self.sigma, "call")
        err_50 = abs(crr_european(self.S, self.K, self.T, self.r, self.sigma, "call", 50) - bs)
        err_500 = abs(crr_european(self.S, self.K, self.T, self.r, self.sigma, "call", 500) - bs)
        assert err_500 < err_50


class TestCRRAmerican:
    """CRR American pricing tests."""

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    def test_american_put_geq_european_put(self):
        am = crr_american(self.S, self.K, self.T, self.r, self.sigma, "put", 300)
        eu = crr_european(self.S, self.K, self.T, self.r, self.sigma, "put", 300)
        assert am >= eu - 1e-10

    def test_american_call_eq_european_call(self):
        am = crr_american(self.S, self.K, self.T, self.r, self.sigma, "call", 300)
        eu = crr_european(self.S, self.K, self.T, self.r, self.sigma, "call", 300)
        assert abs(am - eu) < 0.01

    def test_american_put_itm_early_exercise_premium(self):
        am = crr_american(80, 100, 1.0, 0.05, 0.2, "put", 300)
        eu = crr_european(80, 100, 1.0, 0.05, 0.2, "put", 300)
        assert am > eu + 0.01

    def test_american_put_geq_intrinsic(self):
        am = crr_american(90, 100, 0.5, 0.05, 0.2, "put", 200)
        assert am >= 10.0 - 1e-10

    def test_zero_maturity(self):
        assert crr_american(110, 100, 0, 0.05, 0.2, "call") == 10.0
        assert crr_american(90, 100, 0, 0.05, 0.2, "put") == 10.0

    def test_deep_otm_american_put_near_zero(self):
        price = crr_american(150, 100, 0.5, 0.05, 0.2, "put", 200)
        assert price < 0.01
