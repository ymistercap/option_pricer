"""
Tests for the SABR model module.

Validates:
    - ATM implied vol from SABR ≈ α/F^{1-β} (leading term).
    - SABR with ν=0 gives flat smile (no stochastic vol).
    - Negative ρ produces a skew (higher vol at low strikes).
    - Calibration recovers known parameters from synthetic data.
    - Smile is continuous and positive.
"""

import numpy as np
import pytest

from models.sabr import sabr_calibrate, sabr_implied_vol, sabr_smile


class TestSABRImpliedVol:
    """SABR implied vol formula tests."""

    F, T = 100.0, 1.0
    alpha, beta, rho, nu = 0.2, 0.5, -0.3, 0.4

    def test_atm_leading_order(self):
        iv = sabr_implied_vol(self.F, self.F, self.T, self.alpha, self.beta, 0.0, 0.0)
        expected = self.alpha / (self.F ** (1.0 - self.beta))
        assert abs(iv - expected) < 0.001

    def test_zero_volvol_flat_smile(self):
        strikes = np.linspace(80, 120, 9)
        vols = sabr_smile(self.F, strikes, self.T, self.alpha, self.beta, 0.0, 0.0)
        assert np.std(vols) < 0.005

    def test_negative_rho_skew(self):
        low_K_vol = sabr_implied_vol(self.F, 80, self.T, self.alpha, self.beta, -0.5, 0.4)
        high_K_vol = sabr_implied_vol(self.F, 120, self.T, self.alpha, self.beta, -0.5, 0.4)
        assert low_K_vol > high_K_vol

    def test_positive_rho_reverse_skew(self):
        low_K_vol = sabr_implied_vol(self.F, 80, self.T, self.alpha, self.beta, 0.5, 0.4)
        high_K_vol = sabr_implied_vol(self.F, 120, self.T, self.alpha, self.beta, 0.5, 0.4)
        assert high_K_vol > low_K_vol

    def test_higher_nu_wider_smile(self):
        vols_low = sabr_smile(self.F, np.array([80, 100, 120]), self.T, self.alpha, self.beta, self.rho, 0.1)
        vols_high = sabr_smile(self.F, np.array([80, 100, 120]), self.T, self.alpha, self.beta, self.rho, 0.8)
        spread_low = vols_low[0] - vols_low[2]
        spread_high = vols_high[0] - vols_high[2]
        assert spread_high > spread_low

    def test_all_positive(self):
        strikes = np.linspace(70, 130, 13)
        vols = sabr_smile(self.F, strikes, self.T, self.alpha, self.beta, self.rho, self.nu)
        assert np.all(vols > 0)

    def test_beta_one_lognormal(self):
        iv = sabr_implied_vol(100, 100, 1.0, 0.2, 1.0, 0.0, 0.0)
        assert abs(iv - 0.2) < 0.001

    def test_smile_array_length(self):
        strikes = np.linspace(80, 120, 9)
        vols = sabr_smile(self.F, strikes, self.T, self.alpha, self.beta, self.rho, self.nu)
        assert len(vols) == 9


class TestSABRCalibration:
    """SABR calibration tests."""

    F, T = 100.0, 1.0
    true_alpha, true_beta, true_rho, true_nu = 0.3, 0.5, -0.3, 0.4

    def test_recovers_known_params(self):
        strikes = np.linspace(75, 125, 11)
        market_vols = sabr_smile(
            self.F, strikes, self.T, self.true_alpha, self.true_beta, self.true_rho, self.true_nu
        )
        result = sabr_calibrate(self.F, strikes, market_vols, self.T, self.true_beta)
        assert abs(result["alpha"] - self.true_alpha) < 0.02
        assert abs(result["rho"] - self.true_rho) < 0.05
        assert abs(result["nu"] - self.true_nu) < 0.05

    def test_low_rmse(self):
        strikes = np.linspace(75, 125, 11)
        market_vols = sabr_smile(
            self.F, strikes, self.T, self.true_alpha, self.true_beta, self.true_rho, self.true_nu
        )
        result = sabr_calibrate(self.F, strikes, market_vols, self.T, self.true_beta)
        assert result["rmse"] < 0.001

    def test_calibration_returns_valid_rho(self):
        strikes = np.linspace(80, 120, 9)
        market_vols = sabr_smile(self.F, strikes, self.T, 0.25, 0.5, -0.5, 0.6)
        result = sabr_calibrate(self.F, strikes, market_vols, self.T, 0.5)
        assert -1.0 < result["rho"] < 1.0
        assert result["nu"] >= 0
