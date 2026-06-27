"""
Tests for finite difference option pricing.

Test strategy:
    - Crank-Nicolson vs BS analytical (should match within tolerance)
    - Explicit FD vs BS analytical
    - Put-call parity
    - Grid output validation
    - Convergence with grid refinement
"""

import numpy as np
import pytest

from pricing.black_scholes import bs_call_price, bs_put_price
from pricing.finite_difference import crank_nicolson, explicit_fd, fd_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


class TestCrankNicolson:
    """Tests for the Crank-Nicolson scheme."""

    def test_cn_call_matches_bs(self):
        """CN call price should match BS analytical."""
        bs = bs_call_price(S, K, T, r, sigma)
        fd = crank_nicolson(S, K, T, r, sigma, "call", n_S=400, n_t=400)
        assert abs(fd.price - bs) < 0.05, f"CN={fd.price}, BS={bs}"

    def test_cn_put_matches_bs(self):
        """CN put price should match BS analytical."""
        bs = bs_put_price(S, K, T, r, sigma)
        fd = crank_nicolson(S, K, T, r, sigma, "put", n_S=400, n_t=400)
        assert abs(fd.price - bs) < 0.05, f"CN={fd.price}, BS={bs}"

    @pytest.mark.parametrize(
        "S,K,T,r,sigma",
        [
            (100, 100, 1.0, 0.05, 0.20),
            (120, 100, 0.5, 0.03, 0.30),
            (80, 100, 2.0, 0.08, 0.15),
        ],
    )
    def test_cn_put_call_parity(self, S, K, T, r, sigma):
        """Put-call parity should hold for CN prices."""
        call = crank_nicolson(S, K, T, r, sigma, "call", n_S=300, n_t=300).price
        put = crank_nicolson(S, K, T, r, sigma, "put", n_S=300, n_t=300).price
        parity = S - K * np.exp(-r * T)
        assert abs((call - put) - parity) < 0.2, f"C-P={call-put}, parity={parity}"

    def test_cn_grid_shape(self):
        """Grid output should have correct shape."""
        fd = crank_nicolson(S, K, T, r, sigma, n_S=100, n_t=50)
        assert fd.grid_S.shape == (101,)
        assert fd.grid_t.shape == (51,)
        assert fd.grid_V.shape == (101, 51)

    def test_cn_payoff_at_maturity(self):
        """At maturity (last column), grid should match the payoff."""
        fd = crank_nicolson(S, K, T, r, sigma, "call", n_S=100, n_t=50)
        expected_payoff = np.maximum(fd.grid_S - K, 0.0)
        np.testing.assert_allclose(fd.grid_V[:, -1], expected_payoff, atol=1e-10)

    def test_cn_convergence_with_refinement(self):
        """Error should decrease with grid refinement."""
        bs = bs_call_price(S, K, T, r, sigma)
        err_coarse = abs(crank_nicolson(S, K, T, r, sigma, n_S=50, n_t=50).price - bs)
        err_fine = abs(crank_nicolson(S, K, T, r, sigma, n_S=200, n_t=200).price - bs)
        assert err_fine < err_coarse


class TestExplicitFD:
    """Tests for the explicit finite difference scheme."""

    def test_explicit_call_matches_bs(self):
        """Explicit FD call should match BS (with fine enough grid)."""
        bs = bs_call_price(S, K, T, r, sigma)
        fd = explicit_fd(S, K, T, r, sigma, "call", n_S=200, n_t=10000)
        assert abs(fd.price - bs) < 0.1, f"Explicit={fd.price}, BS={bs}"

    def test_explicit_put_matches_bs(self):
        """Explicit FD put should match BS."""
        bs = bs_put_price(S, K, T, r, sigma)
        fd = explicit_fd(S, K, T, r, sigma, "put", n_S=200, n_t=10000)
        assert abs(fd.price - bs) < 0.1, f"Explicit={fd.price}, BS={bs}"


class TestFDPriceWrapper:
    """Tests for the fd_price convenience wrapper."""

    def test_wrapper_crank_nicolson(self):
        """Wrapper dispatches to CN correctly."""
        bs = bs_call_price(S, K, T, r, sigma)
        price = fd_price(S, K, T, r, sigma, "call", method="crank-nicolson", n_S=300, n_t=300)
        assert abs(price - bs) < 0.1

    def test_wrapper_explicit(self):
        """Wrapper dispatches to explicit correctly."""
        bs = bs_call_price(S, K, T, r, sigma)
        price = fd_price(S, K, T, r, sigma, "call", method="explicit", n_S=200, n_t=10000)
        assert abs(price - bs) < 0.1

    def test_wrapper_invalid_method(self):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError):
            fd_price(S, K, T, r, sigma, method="invalid")
