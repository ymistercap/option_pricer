"""
Tests for the calibration engine.

Validates:
    - Heston calibration recovers params from synthetic Heston data.
    - Merton calibration recovers params from synthetic Merton data.
    - CalibrationResult has correct structure.
    - RMSE is low for self-consistent data.
"""

import numpy as np
import pytest

from calibration.calibrate import CalibrationResult, calibrate_heston, calibrate_merton
from pricing.heston import heston_price
from pricing.jump_diffusion import merton_price
from volatility.implied_vol import implied_vol


class TestHestonCalibration:
    """Heston calibration tests."""

    S, r = 100.0, 0.05
    true_v0, true_kappa, true_theta, true_xi, true_rho = 0.04, 2.0, 0.04, 0.3, -0.5

    def _synthetic_data(self):
        strikes = np.array([90, 95, 100, 105, 110], dtype=float)
        maturities = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        market_vols = np.zeros(5)
        for i in range(5):
            price = heston_price(
                self.S, strikes[i], maturities[i], self.r,
                self.true_v0, self.true_kappa, self.true_theta,
                self.true_xi, self.true_rho, "call",
            )
            market_vols[i] = implied_vol(price, self.S, strikes[i], maturities[i], self.r, "call")
        return strikes, maturities, market_vols

    def test_recovers_parameters(self):
        strikes, maturities, market_vols = self._synthetic_data()
        result = calibrate_heston(self.S, self.r, strikes, maturities, market_vols)
        assert isinstance(result, CalibrationResult)
        assert result.rmse < 0.01

    def test_model_vols_close_to_market(self):
        strikes, maturities, market_vols = self._synthetic_data()
        result = calibrate_heston(self.S, self.r, strikes, maturities, market_vols)
        valid = ~np.isnan(result.model_vols)
        max_err = np.max(np.abs(result.model_vols[valid] - market_vols[valid]))
        assert max_err < 0.01

    def test_output_structure(self):
        strikes, maturities, market_vols = self._synthetic_data()
        result = calibrate_heston(self.S, self.r, strikes, maturities, market_vols)
        assert "v0" in result.params
        assert "kappa" in result.params
        assert "theta" in result.params
        assert "xi" in result.params
        assert "rho" in result.params
        assert len(result.model_vols) == len(strikes)


class TestMertonCalibration:
    """Merton calibration tests."""

    S, r = 100.0, 0.05
    true_sigma, true_lam, true_m, true_delta = 0.15, 1.0, -0.05, 0.1

    def _synthetic_data(self):
        strikes = np.array([90, 95, 100, 105, 110], dtype=float)
        maturities = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        market_vols = np.zeros(5)
        for i in range(5):
            price = merton_price(
                self.S, strikes[i], maturities[i], self.r,
                self.true_sigma, self.true_lam, self.true_m, self.true_delta, "call",
            )
            market_vols[i] = implied_vol(price, self.S, strikes[i], maturities[i], self.r, "call")
        return strikes, maturities, market_vols

    def test_low_rmse(self):
        strikes, maturities, market_vols = self._synthetic_data()
        result = calibrate_merton(self.S, self.r, strikes, maturities, market_vols)
        assert isinstance(result, CalibrationResult)
        assert result.rmse < 0.01

    def test_model_vols_close_to_market(self):
        strikes, maturities, market_vols = self._synthetic_data()
        result = calibrate_merton(self.S, self.r, strikes, maturities, market_vols)
        valid = ~np.isnan(result.model_vols)
        max_err = np.max(np.abs(result.model_vols[valid] - market_vols[valid]))
        assert max_err < 0.01

    def test_output_structure(self):
        strikes, maturities, market_vols = self._synthetic_data()
        result = calibrate_merton(self.S, self.r, strikes, maturities, market_vols)
        assert "sigma" in result.params
        assert "lam" in result.params
        assert "m" in result.params
        assert "delta" in result.params
