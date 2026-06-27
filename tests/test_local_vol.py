"""
Tests for the Dupire local volatility module.

Validates:
    - Flat IV surface → flat local vol (σ_loc = σ_imp).
    - Local vol surface has correct shape.
    - MC under flat local vol matches BS.
    - Local vol values are positive where defined.
"""

import numpy as np
import pytest

from models.local_vol import dupire_local_vol, local_vol_mc
from pricing.black_scholes import bs_price


class TestDupireLocalVol:
    """Dupire local vol extraction tests."""

    S, r = 100.0, 0.05
    sigma_flat = 0.2

    def _flat_surface(self):
        strikes = np.linspace(70, 130, 13)
        maturities = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        iv_surface = np.full((len(maturities), len(strikes)), self.sigma_flat)
        return strikes, maturities, iv_surface

    def test_flat_iv_gives_flat_local_vol(self):
        strikes, maturities, iv_surface = self._flat_surface()
        lv = dupire_local_vol(strikes, maturities, iv_surface, self.S, self.r)
        # Interior points (away from boundaries) should be close to flat sigma
        interior = lv[1:-1, 2:-2]
        valid = interior[~np.isnan(interior)]
        assert len(valid) > 0
        assert np.all(np.abs(valid - self.sigma_flat) < 0.02)

    def test_output_shape(self):
        strikes, maturities, iv_surface = self._flat_surface()
        lv = dupire_local_vol(strikes, maturities, iv_surface, self.S, self.r)
        assert lv.shape == iv_surface.shape

    def test_positive_local_vols(self):
        strikes, maturities, iv_surface = self._flat_surface()
        lv = dupire_local_vol(strikes, maturities, iv_surface, self.S, self.r)
        valid = lv[~np.isnan(lv)]
        assert np.all(valid > 0)

    def test_skewed_surface_varies(self):
        strikes = np.linspace(80, 120, 9)
        maturities = np.array([0.25, 0.5, 1.0])
        iv_surface = np.zeros((3, 9))
        for i in range(3):
            for j in range(9):
                moneyness = strikes[j] / self.S
                iv_surface[i, j] = 0.2 + 0.1 * (1.0 - moneyness)
        lv = dupire_local_vol(strikes, maturities, iv_surface, self.S, self.r)
        valid = lv[~np.isnan(lv)]
        assert valid.std() > 0.001


class TestLocalVolMC:
    """Local vol MC pricing tests."""

    S, r, sigma = 100.0, 0.05, 0.2

    def test_flat_local_vol_mc_matches_bs(self):
        strikes = np.linspace(60, 140, 17)
        maturities = np.linspace(0.0, 1.5, 7)
        lv_surface = np.full((len(maturities), len(strikes)), self.sigma)

        mc = local_vol_mc(
            self.S, 100.0, 1.0, self.r,
            strikes, maturities, lv_surface,
            "call", n_paths=100_000, n_steps=100,
        )
        bs = bs_price(self.S, 100.0, 1.0, self.r, self.sigma, "call")
        assert abs(mc.price - bs) < 0.5

    def test_mc_std_error_positive(self):
        strikes = np.linspace(60, 140, 17)
        maturities = np.linspace(0.0, 1.5, 7)
        lv_surface = np.full((len(maturities), len(strikes)), self.sigma)

        mc = local_vol_mc(
            self.S, 100.0, 1.0, self.r,
            strikes, maturities, lv_surface,
            "call", n_paths=10_000, n_steps=50,
        )
        assert mc.std_error > 0
