"""
Tests for analytical and numerical Greeks.

Test strategy:
    - Verify analytical Greeks against known values
    - Compare numerical Greeks to analytical Greeks (should match within tolerance)
    - Check Greek properties (signs, symmetries, bounds)
    - Verify the BS PDE relationship: Θ + ½σ²S²Γ + rSΔ - rV = 0
"""

import numpy as np
import pytest

from greeks import analytical as ag
from greeks import numerical as ng
from pricing.black_scholes import bs_price

# Standard test parameters
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


class TestAnalyticalDelta:
    """Tests for analytical delta."""

    def test_call_delta_atm(self):
        """ATM call delta should be slightly above 0.5 (due to drift)."""
        d = ag.delta(S, K, T, r, sigma, "call")
        assert 0.5 < d < 0.7

    def test_put_delta_atm(self):
        """ATM put delta should be negative and close to delta_call - 1."""
        d = ag.delta(S, K, T, r, sigma, "put")
        assert -0.7 < d < 0.0

    def test_call_put_delta_relationship(self):
        """Call delta - Put delta = 1 (from put-call parity)."""
        dc = ag.delta(S, K, T, r, sigma, "call")
        dp = ag.delta(S, K, T, r, sigma, "put")
        assert abs((dc - dp) - 1.0) < 1e-10

    def test_call_delta_bounds(self):
        """Call delta ∈ [0, 1]."""
        for s in [50, 80, 100, 120, 200]:
            d = ag.delta(s, K, T, r, sigma, "call")
            assert 0.0 <= d <= 1.0

    def test_put_delta_bounds(self):
        """Put delta ∈ [-1, 0]."""
        for s in [50, 80, 100, 120, 200]:
            d = ag.delta(s, K, T, r, sigma, "put")
            assert -1.0 <= d <= 0.0

    def test_deep_itm_call_delta(self):
        """Deep ITM call delta → 1."""
        d = ag.delta(200.0, K, T, r, sigma, "call")
        assert d > 0.99

    def test_deep_otm_call_delta(self):
        """Deep OTM call delta → 0."""
        d = ag.delta(30.0, K, T, r, sigma, "call")
        assert d < 0.01


class TestAnalyticalGamma:
    """Tests for analytical gamma."""

    def test_gamma_positive(self):
        """Gamma is always positive for vanilla options."""
        g = ag.gamma(S, K, T, r, sigma)
        assert g > 0

    def test_gamma_max_near_atm(self):
        """Gamma is highest near ATM."""
        g_atm = ag.gamma(100.0, K, T, r, sigma)
        g_itm = ag.gamma(130.0, K, T, r, sigma)
        g_otm = ag.gamma(70.0, K, T, r, sigma)
        assert g_atm > g_itm
        assert g_atm > g_otm

    def test_gamma_same_call_put(self):
        """Gamma is identical for call and put (from put-call parity)."""
        # Gamma doesn't take option_type — it's the same by definition
        g = ag.gamma(S, K, T, r, sigma)
        assert g > 0  # Just verify it computes


class TestAnalyticalVega:
    """Tests for analytical vega."""

    def test_vega_positive(self):
        """Vega is always positive (higher vol → higher option value)."""
        v = ag.vega(S, K, T, r, sigma)
        assert v > 0

    def test_vega_max_near_atm(self):
        """Vega is highest near ATM."""
        v_atm = ag.vega(100.0, K, T, r, sigma)
        v_itm = ag.vega(130.0, K, T, r, sigma)
        v_otm = ag.vega(70.0, K, T, r, sigma)
        assert v_atm > v_itm
        assert v_atm > v_otm


class TestAnalyticalTheta:
    """Tests for analytical theta."""

    def test_call_theta_negative(self):
        """Long call theta is typically negative (time decay)."""
        t = ag.theta(S, K, T, r, sigma, "call")
        assert t < 0

    def test_put_theta_typically_negative(self):
        """Long ATM put theta is typically negative."""
        t = ag.theta(S, K, T, r, sigma, "put")
        assert t < 0


class TestAnalyticalRho:
    """Tests for analytical rho."""

    def test_call_rho_positive(self):
        """Call rho is positive (higher rates → higher call value)."""
        r_val = ag.rho(S, K, T, r, sigma, "call")
        assert r_val > 0

    def test_put_rho_negative(self):
        """Put rho is negative (higher rates → lower put value)."""
        r_val = ag.rho(S, K, T, r, sigma, "put")
        assert r_val < 0


class TestBSPDERelationship:
    """
    Verify the Black-Scholes PDE: Θ + ½σ²S²Γ + rSΔ - rV = 0.

    This fundamental relationship must hold for any BS price.
    """

    @pytest.mark.parametrize(
        "S,K,T,r,sigma,option_type",
        [
            (100, 100, 1.0, 0.05, 0.20, "call"),
            (100, 100, 1.0, 0.05, 0.20, "put"),
            (120, 100, 0.5, 0.03, 0.30, "call"),
            (80, 100, 2.0, 0.08, 0.15, "put"),
        ],
    )
    def test_bs_pde_holds(self, S, K, T, r, sigma, option_type):
        """The BS PDE should hold to numerical precision."""
        th = ag.theta(S, K, T, r, sigma, option_type)
        g = ag.gamma(S, K, T, r, sigma)
        d = ag.delta(S, K, T, r, sigma, option_type)
        v = bs_price(S, K, T, r, sigma, option_type)

        pde_residual = th + 0.5 * sigma**2 * S**2 * g + r * S * d - r * v
        assert abs(pde_residual) < 1e-8, f"PDE residual = {pde_residual}"


class TestNumericalVsAnalytical:
    """
    Compare numerical Greeks (bump-and-reprice) against analytical values.

    This validates both implementations simultaneously.
    """

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_delta_matches(self, option_type):
        """Numerical delta ≈ analytical delta."""
        a = ag.delta(S, K, T, r, sigma, option_type)
        n = ng.delta(S, K, T, r, sigma, option_type)
        assert abs(a - n) < 0.001, f"analytical={a}, numerical={n}"

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_gamma_matches(self, option_type):
        """Numerical gamma ≈ analytical gamma."""
        a = ag.gamma(S, K, T, r, sigma)
        n = ng.gamma(S, K, T, r, sigma, option_type)
        assert abs(a - n) < 0.001, f"analytical={a}, numerical={n}"

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_vega_matches(self, option_type):
        """Numerical vega ≈ analytical vega."""
        a = ag.vega(S, K, T, r, sigma)
        n = ng.vega(S, K, T, r, sigma, option_type)
        assert abs(a - n) < 0.01, f"analytical={a}, numerical={n}"

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_theta_matches(self, option_type):
        """Numerical theta ≈ analytical theta."""
        a = ag.theta(S, K, T, r, sigma, option_type)
        n = ng.theta(S, K, T, r, sigma, option_type)
        assert abs(a - n) < 0.5, f"analytical={a}, numerical={n}"

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_rho_matches(self, option_type):
        """Numerical rho ≈ analytical rho."""
        a = ag.rho(S, K, T, r, sigma, option_type)
        n = ng.rho(S, K, T, r, sigma, option_type)
        assert abs(a - n) < 0.01, f"analytical={a}, numerical={n}"

    def test_vanna_matches(self):
        """Numerical vanna ≈ analytical vanna."""
        a = ag.vanna(S, K, T, r, sigma)
        n = ng.vanna(S, K, T, r, sigma)
        assert abs(a - n) < 0.1, f"analytical={a}, numerical={n}"

    def test_volga_matches(self):
        """Numerical volga ≈ analytical volga."""
        a = ag.volga(S, K, T, r, sigma)
        n = ng.volga(S, K, T, r, sigma)
        assert abs(a - n) < 0.5, f"analytical={a}, numerical={n}"


class TestNumericalGreeksWithDifferentPricers:
    """Verify that numerical Greeks work with any pricer function."""

    def test_numerical_delta_with_custom_pricer(self):
        """Numerical Greeks should work with a custom pricing function."""
        def custom_pricer(S, K, T, r, sigma, option_type):
            return bs_price(S, K, T, r, sigma, option_type)

        d = ng.delta(S, K, T, r, sigma, "call", pricer=custom_pricer)
        a = ag.delta(S, K, T, r, sigma, "call")
        assert abs(d - a) < 0.001
