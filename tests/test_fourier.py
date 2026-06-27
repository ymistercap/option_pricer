"""
Tests for the Carr-Madan FFT pricing module.

Validates:
    - FFT with BS char func matches BS analytical for calls and puts.
    - FFT with Heston char func matches Heston semi-analytical.
    - FFT returns a full strike grid (batch pricing).
    - Put-call parity for FFT prices.
"""

import numpy as np
import pytest

from pricing.black_scholes import bs_price
from pricing.fourier import carr_madan_bs_price, carr_madan_fft, carr_madan_heston_price
from pricing.heston import heston_price


class TestCarrMadanBS:
    """FFT pricing with BS characteristic function."""

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    def test_atm_call_matches_bs(self):
        fft_price = carr_madan_bs_price(self.S, self.K, self.T, self.r, self.sigma, "call")
        bs = bs_price(self.S, self.K, self.T, self.r, self.sigma, "call")
        assert abs(fft_price - bs) < 0.05

    def test_atm_put_matches_bs(self):
        fft_price = carr_madan_bs_price(self.S, self.K, self.T, self.r, self.sigma, "put")
        bs = bs_price(self.S, self.K, self.T, self.r, self.sigma, "put")
        assert abs(fft_price - bs) < 0.05

    def test_itm_call_matches_bs(self):
        fft_price = carr_madan_bs_price(120, 100, 0.5, 0.05, 0.25, "call")
        bs = bs_price(120, 100, 0.5, 0.05, 0.25, "call")
        assert abs(fft_price - bs) < 0.1

    def test_otm_put_matches_bs(self):
        fft_price = carr_madan_bs_price(120, 100, 0.5, 0.05, 0.25, "put")
        bs = bs_price(120, 100, 0.5, 0.05, 0.25, "put")
        assert abs(fft_price - bs) < 0.1

    def test_put_call_parity(self):
        c = carr_madan_bs_price(self.S, self.K, self.T, self.r, self.sigma, "call")
        p = carr_madan_bs_price(self.S, self.K, self.T, self.r, self.sigma, "put")
        parity = c - p - (self.S - self.K * np.exp(-self.r * self.T))
        assert abs(parity) < 0.1

    def test_batch_pricing_returns_arrays(self):
        log_S = np.log(self.S)

        def char_func(u):
            iu = 1j * u
            return np.exp(iu * (log_S + (self.r - 0.5 * self.sigma**2) * self.T)
                          - 0.5 * self.sigma**2 * u**2 * self.T)

        strikes, prices = carr_madan_fft(self.S, self.T, self.r, char_func)
        assert len(strikes) == 4096
        assert len(prices) == 4096

    def test_zero_maturity(self):
        assert carr_madan_bs_price(110, 100, 0, 0.05, 0.2, "call") == 10.0


class TestCarrMadanHeston:
    """FFT pricing with Heston characteristic function."""

    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7

    def test_call_matches_heston_semi(self):
        fft_price = carr_madan_heston_price(
            self.S, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "call"
        )
        semi = heston_price(
            self.S, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "call"
        )
        assert abs(fft_price - semi) < 0.1

    def test_put_matches_heston_semi(self):
        fft_price = carr_madan_heston_price(
            self.S, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "put"
        )
        semi = heston_price(
            self.S, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "put"
        )
        assert abs(fft_price - semi) < 0.1

    def test_positive_prices(self):
        c = carr_madan_heston_price(
            self.S, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "call"
        )
        p = carr_madan_heston_price(
            self.S, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "put"
        )
        assert c > 0
        assert p > 0
