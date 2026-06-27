"""
Tests for exotic options: barrier, Asian, and lookback.

Test strategy:
    - Barrier: analytical vs MC, in-out parity
    - Asian: geometric analytical vs MC, arithmetic CV vs standard MC
    - Lookback: analytical vs MC
    - All exotics should be more expensive than corresponding vanillas (lookback)
      or cheaper (barrier out options)
"""

import numpy as np
import pytest

from pricing.black_scholes import bs_call_price, bs_put_price
from exotics.barrier import barrier_analytical, barrier_mc
from exotics.asian import asian_geometric_analytical, asian_mc, asian_arithmetic_cv
from exotics.lookback import lookback_floating_analytical, lookback_mc

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


class TestBarrierOptions:
    """Tests for barrier options."""

    def test_down_and_out_call_in_out_parity(self):
        """down-and-out + down-and-in = vanilla call."""
        H = 80.0
        out = barrier_analytical(S, K, T, r, sigma, H, "down-and-out", "call")
        in_ = barrier_analytical(S, K, T, r, sigma, H, "down-and-in", "call")
        vanilla = bs_call_price(S, K, T, r, sigma)
        assert abs((out + in_) - vanilla) < 0.01, f"out={out}, in={in_}, vanilla={vanilla}"

    def test_up_and_out_call_in_out_parity(self):
        """up-and-out + up-and-in = vanilla call."""
        H = 130.0
        out = barrier_analytical(S, K, T, r, sigma, H, "up-and-out", "call")
        in_ = barrier_analytical(S, K, T, r, sigma, H, "up-and-in", "call")
        vanilla = bs_call_price(S, K, T, r, sigma)
        assert abs((out + in_) - vanilla) < 0.01, f"out={out}, in={in_}, vanilla={vanilla}"

    def test_down_and_out_put_in_out_parity(self):
        """down-and-out + down-and-in = vanilla put."""
        H = 80.0
        out = barrier_analytical(S, K, T, r, sigma, H, "down-and-out", "put")
        in_ = barrier_analytical(S, K, T, r, sigma, H, "down-and-in", "put")
        vanilla = bs_put_price(S, K, T, r, sigma)
        assert abs((out + in_) - vanilla) < 0.01

    def test_up_and_out_put_in_out_parity(self):
        """up-and-out + up-and-in = vanilla put."""
        H = 130.0
        out = barrier_analytical(S, K, T, r, sigma, H, "up-and-out", "put")
        in_ = barrier_analytical(S, K, T, r, sigma, H, "up-and-in", "put")
        vanilla = bs_put_price(S, K, T, r, sigma)
        assert abs((out + in_) - vanilla) < 0.01

    def test_out_barrier_cheaper_than_vanilla(self):
        """Knock-out options must be cheaper than vanilla."""
        H = 80.0
        out = barrier_analytical(S, K, T, r, sigma, H, "down-and-out", "call")
        vanilla = bs_call_price(S, K, T, r, sigma)
        assert out <= vanilla + 0.01

    def test_barrier_mc_vs_analytical_down_out_call(self):
        """MC barrier should converge to analytical price."""
        H = 80.0
        analytical = barrier_analytical(S, K, T, r, sigma, H, "down-and-out", "call")
        mc = barrier_mc(S, K, T, r, sigma, H, "down-and-out", "call",
                        n_paths=200_000, n_steps=500)
        # MC with discrete monitoring has a bias, allow wider tolerance
        assert abs(mc.price - analytical) < 1.0, f"MC={mc.price}, analytical={analytical}"

    def test_barrier_mc_vs_analytical_up_out_call(self):
        """MC up-and-out call should be close to analytical."""
        H = 130.0
        analytical = barrier_analytical(S, K, T, r, sigma, H, "up-and-out", "call")
        mc = barrier_mc(S, K, T, r, sigma, H, "up-and-out", "call",
                        n_paths=200_000, n_steps=500)
        assert abs(mc.price - analytical) < 1.0


class TestAsianOptions:
    """Tests for Asian options."""

    def test_geometric_mc_vs_analytical(self):
        """MC geometric Asian should match analytical formula."""
        n_steps = 252
        analytical = asian_geometric_analytical(S, K, T, r, sigma, n_steps)
        mc = asian_mc(S, K, T, r, sigma, n_paths=200_000, n_steps=n_steps,
                      average_type="geometric")
        assert abs(mc.price - analytical) < 0.2, f"MC={mc.price}, analytical={analytical}"

    def test_asian_cheaper_than_vanilla(self):
        """Asian call (averaging reduces volatility) should be cheaper than vanilla."""
        vanilla = bs_call_price(S, K, T, r, sigma)
        asian_arith = asian_mc(S, K, T, r, sigma, n_paths=200_000, average_type="arithmetic")
        assert asian_arith.price < vanilla + 0.5

    def test_arithmetic_mean_geq_geometric_mean(self):
        """Arithmetic average Asian ≥ geometric average Asian (AM-GM inequality)."""
        arith = asian_mc(S, K, T, r, sigma, n_paths=100_000, average_type="arithmetic")
        geom = asian_mc(S, K, T, r, sigma, n_paths=100_000, average_type="geometric")
        # Arithmetic payoff >= geometric payoff on average
        assert arith.price >= geom.price - 0.5

    def test_asian_cv_reduces_variance(self):
        """Control variate should reduce standard error vs plain MC."""
        mc_plain = asian_mc(S, K, T, r, sigma, n_paths=100_000, average_type="arithmetic")
        mc_cv = asian_arithmetic_cv(S, K, T, r, sigma, n_paths=100_000)
        assert mc_cv.std_error < mc_plain.std_error

    def test_asian_put_positive(self):
        """Asian put should have positive price."""
        result = asian_mc(S, K, T, r, sigma, n_paths=50_000, option_type="put")
        assert result.price > 0


class TestLookbackOptions:
    """Tests for lookback options."""

    def test_lookback_call_analytical_positive(self):
        """Lookback call price should be positive."""
        price = lookback_floating_analytical(S, T, r, sigma, "call")
        assert price > 0

    def test_lookback_put_analytical_positive(self):
        """Lookback put price should be positive."""
        price = lookback_floating_analytical(S, T, r, sigma, "put")
        assert price > 0

    def test_lookback_more_expensive_than_vanilla(self):
        """Floating-strike lookback should be more expensive than ATM vanilla."""
        vanilla_call = bs_call_price(S, S, T, r, sigma)
        lookback_call = lookback_floating_analytical(S, T, r, sigma, "call")
        assert lookback_call > vanilla_call

    def test_lookback_mc_vs_analytical_call(self):
        """MC lookback call should be close to analytical (discrete bias expected)."""
        analytical = lookback_floating_analytical(S, T, r, sigma, "call")
        mc = lookback_mc(S, T, r, sigma, "call", n_paths=200_000, n_steps=500)
        # Discrete monitoring underestimates the continuous extremum
        assert abs(mc.price - analytical) < 2.0, f"MC={mc.price}, analytical={analytical}"

    def test_lookback_mc_vs_analytical_put(self):
        """MC lookback put should be close to analytical."""
        analytical = lookback_floating_analytical(S, T, r, sigma, "put")
        mc = lookback_mc(S, T, r, sigma, "put", n_paths=200_000, n_steps=500)
        assert abs(mc.price - analytical) < 2.0, f"MC={mc.price}, analytical={analytical}"

    def test_lookback_increases_with_volatility(self):
        """Higher vol → higher lookback price (more extreme min/max)."""
        p1 = lookback_floating_analytical(S, T, r, 0.15, "call")
        p2 = lookback_floating_analytical(S, T, r, 0.30, "call")
        assert p2 > p1
