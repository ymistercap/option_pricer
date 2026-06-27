"""
Tests for the Black-Scholes analytical pricing module.

Test strategy:
    - Known-value tests against hand-computed / well-known results
    - Put-call parity verification: C - P = S - K·e^{-rT}
    - Edge cases: T=0 (expiry), S=0, sigma=0, deep ITM/OTM
    - Digital option complementarity: digital_call + digital_put = e^{-rT}
    - Monotonicity and boundary conditions
"""

import numpy as np
import pytest

from pricing.black_scholes import (
    bs_call_price,
    bs_digital_call,
    bs_digital_put,
    bs_price,
    bs_put_price,
)


# ---------------------------------------------------------------------------
# Standard test parameters
# ---------------------------------------------------------------------------
# ATM option: S=100, K=100, T=1y, r=5%, σ=20%
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


class TestBSCallPrice:
    """Tests for bs_call_price."""

    def test_bs_call_atm_known_value(self):
        """ATM call with standard params should match known BS value."""
        price = bs_call_price(S, K, T, r, sigma)
        assert abs(price - 10.4506) < 0.01

    def test_bs_call_deep_itm(self):
        """Deep ITM call (S >> K) should be close to S - K·e^{-rT}."""
        price = bs_call_price(200.0, 100.0, 1.0, 0.05, 0.20)
        intrinsic = 200.0 - 100.0 * np.exp(-0.05)
        assert abs(price - intrinsic) < 1.0

    def test_bs_call_deep_otm(self):
        """Deep OTM call (S << K) should be close to 0."""
        price = bs_call_price(50.0, 200.0, 1.0, 0.05, 0.20)
        assert price < 0.01

    def test_bs_call_positive(self):
        """Call price should always be non-negative."""
        assert bs_call_price(S, K, T, r, sigma) >= 0

    def test_bs_call_upper_bound(self):
        """Call price should be <= S (no dividends)."""
        price = bs_call_price(S, K, T, r, sigma)
        assert price <= S

    def test_bs_call_increases_with_spot(self):
        """Call price is monotonically increasing in S."""
        p1 = bs_call_price(90.0, K, T, r, sigma)
        p2 = bs_call_price(100.0, K, T, r, sigma)
        p3 = bs_call_price(110.0, K, T, r, sigma)
        assert p1 < p2 < p3

    def test_bs_call_increases_with_volatility(self):
        """Call price is monotonically increasing in sigma."""
        p1 = bs_call_price(S, K, T, r, 0.10)
        p2 = bs_call_price(S, K, T, r, 0.20)
        p3 = bs_call_price(S, K, T, r, 0.30)
        assert p1 < p2 < p3

    def test_bs_call_increases_with_maturity(self):
        """Call price generally increases with maturity (for r >= 0)."""
        p1 = bs_call_price(S, K, 0.25, r, sigma)
        p2 = bs_call_price(S, K, 1.0, r, sigma)
        p3 = bs_call_price(S, K, 2.0, r, sigma)
        assert p1 < p2 < p3


class TestBSPutPrice:
    """Tests for bs_put_price."""

    def test_bs_put_atm_known_value(self):
        """ATM put with standard params should match known BS value."""
        price = bs_put_price(S, K, T, r, sigma)
        assert abs(price - 5.5735) < 0.01

    def test_bs_put_deep_itm(self):
        """Deep ITM put (S << K) should be close to K·e^{-rT} - S."""
        price = bs_put_price(50.0, 200.0, 1.0, 0.05, 0.20)
        intrinsic = 200.0 * np.exp(-0.05) - 50.0
        assert abs(price - intrinsic) < 1.0

    def test_bs_put_deep_otm(self):
        """Deep OTM put (S >> K) should be close to 0."""
        price = bs_put_price(200.0, 100.0, 1.0, 0.05, 0.20)
        assert price < 0.01

    def test_bs_put_positive(self):
        """Put price should always be non-negative."""
        assert bs_put_price(S, K, T, r, sigma) >= 0

    def test_bs_put_decreases_with_spot(self):
        """Put price is monotonically decreasing in S."""
        p1 = bs_put_price(90.0, K, T, r, sigma)
        p2 = bs_put_price(100.0, K, T, r, sigma)
        p3 = bs_put_price(110.0, K, T, r, sigma)
        assert p1 > p2 > p3


class TestPutCallParity:
    """
    Put-call parity: C - P = S - K·e^{-rT}.

    This is a model-free arbitrage relationship for European options.
    """

    @pytest.mark.parametrize(
        "S,K,T,r,sigma",
        [
            (100, 100, 1.0, 0.05, 0.20),  # ATM
            (120, 100, 1.0, 0.05, 0.20),  # ITM call
            (80, 100, 1.0, 0.05, 0.20),   # OTM call
            (100, 100, 0.1, 0.05, 0.20),  # Short maturity
            (100, 100, 5.0, 0.05, 0.20),  # Long maturity
            (100, 100, 1.0, 0.10, 0.40),  # High vol / high rate
            (100, 100, 1.0, 0.00, 0.20),  # Zero rate
            (50, 150, 2.0, 0.03, 0.35),   # Deep OTM call
        ],
    )
    def test_put_call_parity(self, S, K, T, r, sigma):
        """Verify C - P = S - K·e^{-rT} for various parameter sets."""
        call = bs_call_price(S, K, T, r, sigma)
        put = bs_put_price(S, K, T, r, sigma)
        parity = S - K * np.exp(-r * T)
        assert abs((call - put) - parity) < 1e-10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_call_at_expiry_itm(self):
        """At expiry (T=0), ITM call = S - K."""
        assert bs_call_price(110.0, 100.0, 0.0, 0.05, 0.20) == 10.0

    def test_call_at_expiry_otm(self):
        """At expiry (T=0), OTM call = 0."""
        assert bs_call_price(90.0, 100.0, 0.0, 0.05, 0.20) == 0.0

    def test_put_at_expiry_itm(self):
        """At expiry (T=0), ITM put = K - S."""
        assert bs_put_price(90.0, 100.0, 0.0, 0.05, 0.20) == 10.0

    def test_put_at_expiry_otm(self):
        """At expiry (T=0), OTM put = 0."""
        assert bs_put_price(110.0, 100.0, 0.0, 0.05, 0.20) == 0.0

    def test_call_spot_zero(self):
        """If S=0, call is worthless."""
        assert bs_call_price(0.0, 100.0, 1.0, 0.05, 0.20) == 0.0

    def test_put_spot_zero(self):
        """If S=0, put = K·e^{-rT}."""
        price = bs_put_price(0.0, 100.0, 1.0, 0.05, 0.20)
        assert abs(price - 100.0 * np.exp(-0.05)) < 1e-10

    def test_call_zero_vol(self):
        """With σ=0, call = max(S - K·e^{-rT}, 0) (deterministic)."""
        price = bs_call_price(100.0, 95.0, 1.0, 0.05, 0.0)
        expected = max(100.0 - 95.0 * np.exp(-0.05), 0.0)
        assert abs(price - expected) < 1e-10

    def test_put_zero_vol(self):
        """With σ=0, put = max(K·e^{-rT} - S, 0) (deterministic)."""
        price = bs_put_price(90.0, 100.0, 1.0, 0.05, 0.0)
        expected = max(100.0 * np.exp(-0.05) - 90.0, 0.0)
        assert abs(price - expected) < 1e-10


class TestDigitalOptions:
    """Tests for digital (binary/cash-or-nothing) options."""

    def test_digital_call_atm(self):
        """ATM digital call known value."""
        price = bs_digital_call(S, K, T, r, sigma)
        assert abs(price - 0.5323) < 0.01

    def test_digital_put_atm(self):
        """ATM digital put known value."""
        price = bs_digital_put(S, K, T, r, sigma)
        assert abs(price - 0.4189) < 0.01

    def test_digital_complementarity(self):
        """Digital call + digital put = e^{-rT} (covers all outcomes)."""
        dc = bs_digital_call(S, K, T, r, sigma)
        dp = bs_digital_put(S, K, T, r, sigma)
        assert abs(dc + dp - np.exp(-r * T)) < 1e-10

    @pytest.mark.parametrize(
        "S,K,T,r,sigma",
        [
            (100, 100, 1.0, 0.05, 0.20),
            (120, 80, 0.5, 0.03, 0.30),
            (80, 120, 2.0, 0.08, 0.15),
        ],
    )
    def test_digital_complementarity_parametric(self, S, K, T, r, sigma):
        """Digital call + digital put = e^{-rT} across parameter sets."""
        dc = bs_digital_call(S, K, T, r, sigma)
        dp = bs_digital_put(S, K, T, r, sigma)
        assert abs(dc + dp - np.exp(-r * T)) < 1e-10

    def test_digital_call_at_expiry_itm(self):
        """At expiry, digital call pays 1 if ITM."""
        assert bs_digital_call(110.0, 100.0, 0.0, 0.05, 0.20) == 1.0

    def test_digital_call_at_expiry_otm(self):
        """At expiry, digital call pays 0 if OTM."""
        assert bs_digital_call(90.0, 100.0, 0.0, 0.05, 0.20) == 0.0

    def test_digital_put_at_expiry_itm(self):
        """At expiry, digital put pays 1 if ITM."""
        assert bs_digital_put(90.0, 100.0, 0.0, 0.05, 0.20) == 1.0

    def test_digital_put_at_expiry_otm(self):
        """At expiry, digital put pays 0 if OTM."""
        assert bs_digital_put(110.0, 100.0, 0.0, 0.05, 0.20) == 0.0

    def test_digital_call_deep_itm(self):
        """Deep ITM digital call ≈ e^{-rT}."""
        price = bs_digital_call(200.0, 100.0, 1.0, 0.05, 0.20)
        assert abs(price - np.exp(-0.05)) < 0.01

    def test_digital_call_deep_otm(self):
        """Deep OTM digital call ≈ 0."""
        price = bs_digital_call(50.0, 200.0, 1.0, 0.05, 0.20)
        assert price < 0.001


class TestBSPriceWrapper:
    """Tests for the bs_price convenience wrapper."""

    def test_wrapper_call(self):
        """Wrapper 'call' matches bs_call_price."""
        assert bs_price(S, K, T, r, sigma, "call") == bs_call_price(S, K, T, r, sigma)

    def test_wrapper_put(self):
        """Wrapper 'put' matches bs_put_price."""
        assert bs_price(S, K, T, r, sigma, "put") == bs_put_price(S, K, T, r, sigma)

    def test_wrapper_invalid_type(self):
        """Invalid option type raises ValueError."""
        with pytest.raises(ValueError, match="option_type"):
            bs_price(S, K, T, r, sigma, "straddle")
