"""Tests for risk overlay implementations.

Tests for IMPL-010: Risk Overlays Module
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantetf.risk.overlays import (
    DrawdownCircuitBreaker,
    PositionLimitOverlay,
    RiskOverlay,
    VIXRegimeOverlay,
    VolatilityTargeting,
    apply_overlay_chain,
)


# --- Fixtures ---


@dataclass
class MockPortfolioState:
    """Mock portfolio state for testing."""

    nav: float
    peak_nav: float
    as_of: pd.Timestamp = pd.Timestamp("2024-01-15")


class MockDataStore:
    """Mock data store for testing."""

    def __init__(
        self,
        prices: Optional[pd.DataFrame] = None,
        tickers: Optional[list[str]] = None,
    ):
        self._prices = prices
        self.tickers = tickers or ["SPY", "QQQ", "AAPL", "MSFT", "AGG", "TLT", "GLD"]

    def get_close_prices(
        self,
        tickers: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        if self._prices is not None:
            return self._prices
        # Generate default price data
        dates = pd.date_range(start=start, end=end, freq="B")
        data = {}
        for ticker in tickers:
            # Random walk prices
            np.random.seed(hash(ticker) % 2**32)
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = 100 * np.cumprod(1 + returns)
            data[ticker] = prices
        return pd.DataFrame(data, index=dates)


@pytest.fixture
def mock_store() -> MockDataStore:
    return MockDataStore()


@pytest.fixture
def sample_weights() -> pd.Series:
    return pd.Series({"SPY": 0.4, "QQQ": 0.3, "AAPL": 0.2, "MSFT": 0.1})


# --- VolatilityTargeting Tests ---


class TestVolatilityTargeting:
    """Tests for VolatilityTargeting overlay."""

    def test_volatility_targeting_scales_correctly(self):
        """Verify scale factor calculation based on realized vs target vol."""
        # Create store with known volatility data
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        # Create prices with ~20% annualized volatility
        np.random.seed(42)
        daily_vol = 0.20 / np.sqrt(252)
        returns = np.random.normal(0, daily_vol, 60)
        prices = 100 * np.cumprod(1 + returns)
        price_df = pd.DataFrame({"SPY": prices}, index=dates)

        store = MockDataStore(prices=price_df)
        weights = pd.Series({"SPY": 1.0})

        overlay = VolatilityTargeting(
            target_vol=0.15,
            lookback_days=60,
            min_scale=0.25,
            max_scale=1.50,
        )

        result, diag = overlay.apply(
            weights,
            as_of=dates[-1],
            store=store,
            portfolio_state=None,
        )

        # With ~20% realized vol and 15% target, scale should be ~0.75
        assert "scale_factor" in diag
        assert "realized_vol" in diag
        assert diag["scale_factor"] > 0
        # Result should be scaled
        assert result["SPY"] == pytest.approx(weights["SPY"] * diag["scale_factor"])

    def test_volatility_targeting_clamps_bounds(self):
        """Verify min/max scale limits are enforced."""
        overlay = VolatilityTargeting(
            target_vol=0.15,
            min_scale=0.25,
            max_scale=1.50,
        )

        dates = pd.date_range("2024-01-01", periods=60, freq="B")

        # Test with very high volatility (should clamp to min_scale)
        np.random.seed(123)
        high_vol_prices = pd.DataFrame(
            {"SPY": 100 * np.cumprod(1 + np.random.normal(0, 0.10, 60))},
            index=dates,
        )
        store_high_vol = MockDataStore(prices=high_vol_prices)
        weights = pd.Series({"SPY": 1.0})

        result_high, diag_high = overlay.apply(weights, dates[-1], store_high_vol, None)

        # High vol should clamp to min_scale
        assert diag_high["scale_factor"] == pytest.approx(0.25, rel=0.01)

        # Test that scale factor is always within bounds
        assert diag_high["scale_factor"] >= 0.25
        assert diag_high["scale_factor"] <= 1.50

    def test_volatility_targeting_handles_insufficient_data(self):
        """Verify graceful handling when not enough data."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        prices = pd.DataFrame({"SPY": [100] * 10}, index=dates)
        store = MockDataStore(prices=prices)

        overlay = VolatilityTargeting()
        weights = pd.Series({"SPY": 1.0})

        result, diag = overlay.apply(weights, dates[-1], store, None)

        # Should pass through unchanged with error
        assert "error" in diag
        assert diag["scale_factor"] == 1.0
        assert result["SPY"] == weights["SPY"]


# --- PositionLimitOverlay Tests ---


class TestPositionLimitOverlay:
    """Tests for PositionLimitOverlay."""

    def test_position_limit_caps_weights(self):
        """Verify 25% cap is applied to positions."""
        overlay = PositionLimitOverlay(max_weight=0.25, redistribute=False)

        weights = pd.Series({"AAPL": 0.50, "MSFT": 0.30, "GOOGL": 0.20})

        result, diag = overlay.apply(weights, pd.Timestamp("2024-01-15"), None, None)

        # All positions should be capped at 25%
        assert result["AAPL"] == 0.25
        assert result["MSFT"] == 0.25
        assert result["GOOGL"] == 0.20  # Already under limit

        assert "AAPL" in diag["capped_tickers"]
        assert "MSFT" in diag["capped_tickers"]
        assert diag["num_capped"] == 2

    def test_position_limit_redistributes(self):
        """Verify excess weight is redistributed to other positions."""
        overlay = PositionLimitOverlay(max_weight=0.25, redistribute=True)

        # Single position over limit
        weights = pd.Series({"AAPL": 0.50, "MSFT": 0.25, "GOOGL": 0.25})

        result, diag = overlay.apply(weights, pd.Timestamp("2024-01-15"), None, None)

        # AAPL should be capped
        assert result["AAPL"] == 0.25

        # Excess (0.25) should be redistributed
        assert diag["excess_weight"] == pytest.approx(0.25)
        assert diag["capped_tickers"] == ["AAPL"]

    def test_position_limit_tracks_capped_tickers(self):
        """Verify diagnostics correctly track which tickers were capped."""
        overlay = PositionLimitOverlay(max_weight=0.20)

        weights = pd.Series(
            {"A": 0.30, "B": 0.25, "C": 0.15, "D": 0.15, "E": 0.15}
        )

        result, diag = overlay.apply(weights, pd.Timestamp("2024-01-15"), None, None)

        assert set(diag["capped_tickers"]) == {"A", "B"}
        assert diag["num_capped"] == 2

    def test_position_limit_no_change_when_under_limit(self):
        """Verify no changes when all positions under limit."""
        overlay = PositionLimitOverlay(max_weight=0.25)

        weights = pd.Series({"A": 0.20, "B": 0.20, "C": 0.20, "D": 0.20, "E": 0.20})

        result, diag = overlay.apply(weights, pd.Timestamp("2024-01-15"), None, None)

        assert result.equals(weights)
        assert diag["capped_tickers"] == []
        assert diag["num_capped"] == 0


# --- DrawdownCircuitBreaker Tests ---


class TestDrawdownCircuitBreaker:
    """Tests for DrawdownCircuitBreaker overlay."""

    def test_drawdown_circuit_breaker_thresholds(self):
        """Verify each drawdown level triggers correctly."""
        overlay = DrawdownCircuitBreaker(
            soft_threshold=0.10,
            hard_threshold=0.20,
            exit_threshold=0.30,
            soft_exposure=0.75,
            hard_exposure=0.50,
            exit_exposure=0.25,
        )

        weights = pd.Series({"SPY": 0.6, "QQQ": 0.4})

        # No drawdown
        state_none = MockPortfolioState(nav=100, peak_nav=100)
        result, diag = overlay.apply(weights, pd.Timestamp("2024-01-15"), None, state_none)
        assert diag["level"] == "NONE"
        assert diag["exposure_factor"] == 1.0
        assert result["SPY"] == 0.6

        # Soft threshold (10% drawdown)
        state_soft = MockPortfolioState(nav=90, peak_nav=100)
        result, diag = overlay.apply(weights, pd.Timestamp("2024-01-15"), None, state_soft)
        assert diag["level"] == "SOFT"
        assert diag["exposure_factor"] == 0.75
        assert result["SPY"] == pytest.approx(0.6 * 0.75)

        # Hard threshold (20% drawdown)
        state_hard = MockPortfolioState(nav=80, peak_nav=100)
        result, diag = overlay.apply(weights, pd.Timestamp("2024-01-15"), None, state_hard)
        assert diag["level"] == "HARD"
        assert diag["exposure_factor"] == 0.50
        assert result["SPY"] == pytest.approx(0.6 * 0.50)

        # Exit threshold (30% drawdown)
        state_exit = MockPortfolioState(nav=70, peak_nav=100)
        result, diag = overlay.apply(weights, pd.Timestamp("2024-01-15"), None, state_exit)
        assert diag["level"] == "EXIT"
        assert diag["exposure_factor"] == 0.25
        assert result["SPY"] == pytest.approx(0.6 * 0.25)

    def test_drawdown_circuit_breaker_no_state(self):
        """Verify pass-through when no portfolio state available."""
        overlay = DrawdownCircuitBreaker()
        weights = pd.Series({"SPY": 0.6, "QQQ": 0.4})

        result, diag = overlay.apply(weights, pd.Timestamp("2024-01-15"), None, None)

        assert diag["level"] == "NONE"
        assert diag["exposure_factor"] == 1.0
        assert result.equals(weights)

    def test_drawdown_circuit_breaker_diagnostics(self):
        """Verify diagnostics contain required information."""
        overlay = DrawdownCircuitBreaker()
        state = MockPortfolioState(nav=85, peak_nav=100)
        weights = pd.Series({"SPY": 1.0})

        result, diag = overlay.apply(weights, pd.Timestamp("2024-01-15"), None, state)

        assert "drawdown" in diag
        assert "peak_nav" in diag
        assert "current_nav" in diag
        assert "exposure_factor" in diag
        assert "level" in diag
        assert "thresholds" in diag
        assert diag["drawdown"] == pytest.approx(0.15)


# --- VIXRegimeOverlay Tests ---


class TestVIXRegimeOverlay:
    """Tests for VIXRegimeOverlay."""

    def test_vix_regime_shifts_to_defensive(self):
        """Verify defensive allocation when VIX is high."""
        overlay = VIXRegimeOverlay(
            high_vix_threshold=30.0,
            elevated_vix_threshold=25.0,
            defensive_tickers=("AGG", "TLT", "GLD"),
            high_vix_defensive_weight=0.50,
            elevated_vix_defensive_weight=0.25,
        )

        weights = pd.Series({"SPY": 0.6, "QQQ": 0.4})
        store = MockDataStore(tickers=["SPY", "QQQ", "AGG", "TLT", "GLD"])

        # Mock VIX at 35 (high regime)
        with patch("quantetf.data.macro_loader.MacroDataLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.get_vix.return_value = 35.0
            mock_loader_class.return_value = mock_loader

            result, diag = overlay.apply(
                weights, pd.Timestamp("2024-01-15"), store, None
            )

        assert diag["regime"] == "HIGH_VOL"
        assert diag["vix"] == 35.0
        assert diag["defensive_weight"] == 0.50

        # Defensive tickers should have allocation
        defensive_total = sum(
            result.get(t, 0) for t in ["AGG", "TLT", "GLD"]
        )
        assert defensive_total > 0

    def test_vix_regime_elevated(self):
        """Verify reduced defensive allocation at elevated VIX."""
        overlay = VIXRegimeOverlay()
        weights = pd.Series({"SPY": 0.6, "QQQ": 0.4})
        store = MockDataStore()

        with patch("quantetf.data.macro_loader.MacroDataLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.get_vix.return_value = 27.0  # Elevated
            mock_loader_class.return_value = mock_loader

            result, diag = overlay.apply(
                weights, pd.Timestamp("2024-01-15"), store, None
            )

        assert diag["regime"] == "ELEVATED_VOL"
        assert diag["defensive_weight"] == 0.25

    def test_vix_regime_normal(self):
        """Verify no defensive shift in normal VIX regime."""
        overlay = VIXRegimeOverlay()
        weights = pd.Series({"SPY": 0.6, "QQQ": 0.4})
        store = MockDataStore()

        with patch("quantetf.data.macro_loader.MacroDataLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.get_vix.return_value = 15.0  # Normal
            mock_loader_class.return_value = mock_loader

            result, diag = overlay.apply(
                weights, pd.Timestamp("2024-01-15"), store, None
            )

        assert diag["regime"] == "NORMAL"
        assert diag["defensive_weight"] == 0.0
        # Weights should be unchanged
        assert result["SPY"] == pytest.approx(0.6)
        assert result["QQQ"] == pytest.approx(0.4)

    def test_vix_regime_handles_data_error(self):
        """Verify graceful handling when VIX data unavailable."""
        overlay = VIXRegimeOverlay()
        weights = pd.Series({"SPY": 0.6, "QQQ": 0.4})
        store = MockDataStore()

        with patch("quantetf.data.macro_loader.MacroDataLoader") as mock_loader_class:
            mock_loader_class.side_effect = Exception("VIX data not available")

            result, diag = overlay.apply(
                weights, pd.Timestamp("2024-01-15"), store, None
            )

        assert diag["regime"] == "UNKNOWN"
        assert "error" in diag
        # Weights should pass through unchanged
        assert result.equals(weights)

    def test_vix_overlay_is_frozen(self):
        """Verify VIXRegimeOverlay is a frozen dataclass."""
        overlay = VIXRegimeOverlay()

        with pytest.raises(AttributeError):
            overlay.high_vix_threshold = 50.0


# --- apply_overlay_chain Tests ---


class TestApplyOverlayChain:
    """Tests for apply_overlay_chain function."""

    def test_chain_applies_overlays_sequentially(self, mock_store, sample_weights):
        """Verify overlays are applied in order."""
        overlays = [
            PositionLimitOverlay(max_weight=0.30),
            PositionLimitOverlay(max_weight=0.25),
        ]

        result, diagnostics = apply_overlay_chain(
            target_weights=sample_weights,
            overlays=overlays,
            as_of=pd.Timestamp("2024-01-15"),
            store=mock_store,
            portfolio_state=None,
        )

        # Both overlays should have run
        assert "PositionLimitOverlay" in diagnostics
        # Final result should respect 25% limit
        assert all(w <= 0.25 + 0.001 for w in result)

    def test_chain_collects_all_diagnostics(self, mock_store):
        """Verify diagnostics from all overlays are collected."""
        weights = pd.Series({"SPY": 0.5, "QQQ": 0.5})
        state = MockPortfolioState(nav=90, peak_nav=100)

        overlays = [
            PositionLimitOverlay(max_weight=0.40),
            DrawdownCircuitBreaker(),
        ]

        result, diagnostics = apply_overlay_chain(
            target_weights=weights,
            overlays=overlays,
            as_of=pd.Timestamp("2024-01-15"),
            store=mock_store,
            portfolio_state=state,
        )

        assert "PositionLimitOverlay" in diagnostics
        assert "DrawdownCircuitBreaker" in diagnostics
        assert diagnostics["DrawdownCircuitBreaker"]["level"] == "SOFT"

    def test_chain_handles_empty_overlays(self, mock_store, sample_weights):
        """Verify chain handles empty overlay list."""
        result, diagnostics = apply_overlay_chain(
            target_weights=sample_weights,
            overlays=[],
            as_of=pd.Timestamp("2024-01-15"),
            store=mock_store,
            portfolio_state=None,
        )

        assert result.equals(sample_weights)
        assert diagnostics == {}

    def test_chain_preserves_weight_structure(self, mock_store):
        """Verify chain preserves Series structure."""
        weights = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
        overlays = [PositionLimitOverlay(max_weight=0.35)]

        result, _ = apply_overlay_chain(
            target_weights=weights,
            overlays=overlays,
            as_of=pd.Timestamp("2024-01-15"),
            store=mock_store,
            portfolio_state=None,
        )

        assert isinstance(result, pd.Series)
        assert set(result.index) == set(weights.index)
