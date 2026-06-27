"""
Cross-validation tests: our implementations vs QuantLib.

These tests prove our pricing engines and Greeks are correct by comparing
against the industry-standard QuantLib library.
"""

import numpy as np
import pytest

from pricing.black_scholes import bs_call_price, bs_put_price
from pricing.finite_difference import fd_price
from pricing.monte_carlo import mc_european
from pricing.quantlib_bench import ql_bs_price, ql_fd_price, ql_greeks, ql_mc_price
from greeks.analytical import delta, gamma, vega, theta, rho

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


class TestPricingVsQuantLib:
    """Cross-validate pricing engines against QuantLib."""

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_bs_analytical_matches_quantlib(self, option_type):
        """Our BS analytical should match QuantLib's analytical engine exactly."""
        if option_type == "call":
            my_price = bs_call_price(S, K, T, r, sigma)
        else:
            my_price = bs_put_price(S, K, T, r, sigma)
        ql_price = ql_bs_price(S, K, T, r, sigma, option_type)
        assert abs(my_price - ql_price) < 0.01, f"my={my_price}, ql={ql_price}"

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_fd_matches_quantlib(self, option_type):
        """Our CN FD should match QuantLib's FD engine."""
        my_price = fd_price(S, K, T, r, sigma, option_type, n_S=300, n_t=300)
        ql_price = ql_fd_price(S, K, T, r, sigma, option_type)
        assert abs(my_price - ql_price) < 0.1, f"my={my_price}, ql={ql_price}"

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_mc_matches_quantlib_order_of_magnitude(self, option_type):
        """Our MC should be in the right ballpark vs QuantLib MC."""
        my_mc = mc_european(S, K, T, r, sigma, n_paths=200_000, option_type=option_type)
        ql_price = ql_bs_price(S, K, T, r, sigma, option_type)  # Use analytical as reference
        assert abs(my_mc.price - ql_price) < 0.2

    @pytest.mark.parametrize(
        "S,K,T,r,sigma",
        [
            (100, 100, 1.0, 0.05, 0.20),
            (120, 80, 0.5, 0.03, 0.30),
            (80, 120, 2.0, 0.08, 0.15),
            (100, 100, 0.1, 0.01, 0.50),
        ],
    )
    def test_bs_call_matches_quantlib_various_params(self, S, K, T, r, sigma):
        """BS call matches QuantLib across various parameter sets."""
        my_price = bs_call_price(S, K, T, r, sigma)
        ql_price = ql_bs_price(S, K, T, r, sigma, "call")
        assert abs(my_price - ql_price) < 0.05  # Small daycount differences with QL


class TestGreeksVsQuantLib:
    """Cross-validate Greeks against QuantLib."""

    def test_delta_matches_quantlib(self):
        """Our analytical delta should match QuantLib."""
        ql_g = ql_greeks(S, K, T, r, sigma, "call")
        my_d = delta(S, K, T, r, sigma, "call")
        assert abs(my_d - ql_g["delta"]) < 0.001

    def test_gamma_matches_quantlib(self):
        """Our analytical gamma should match QuantLib."""
        ql_g = ql_greeks(S, K, T, r, sigma, "call")
        my_g = gamma(S, K, T, r, sigma)
        assert abs(my_g - ql_g["gamma"]) < 0.001

    def test_vega_matches_quantlib(self):
        """Our analytical vega should match QuantLib."""
        ql_g = ql_greeks(S, K, T, r, sigma, "call")
        my_v = vega(S, K, T, r, sigma)
        assert abs(my_v - ql_g["vega"]) < 0.1

    def test_theta_matches_quantlib(self):
        """Our analytical theta should match QuantLib."""
        ql_g = ql_greeks(S, K, T, r, sigma, "call")
        my_t = theta(S, K, T, r, sigma, "call")
        assert abs(my_t - ql_g["theta"]) < 0.1

    def test_rho_matches_quantlib(self):
        """Our analytical rho should match QuantLib."""
        ql_g = ql_greeks(S, K, T, r, sigma, "call")
        my_r = rho(S, K, T, r, sigma, "call")
        assert abs(my_r - ql_g["rho"]) < 0.1
