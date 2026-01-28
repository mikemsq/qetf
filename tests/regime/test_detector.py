"""Comprehensive tests for regime detection with hysteresis."""

import pytest
import pandas as pd

from quantetf.regime.detector import RegimeDetector
from quantetf.regime.types import (
    RegimeConfig,
    RegimeState,
    TrendState,
    VolatilityState,
)


class TestRegimeDetector:
    """Test regime detection with hysteresis."""

    @pytest.fixture
    def detector(self):
        return RegimeDetector()

    @pytest.fixture
    def as_of(self):
        return pd.Timestamp("2026-01-24")

    # --- Trend Detection Tests ---

    def test_clear_uptrend(self, detector, as_of):
        """SPY well above 200MA should be uptrend."""
        state = detector.detect(
            spy_price=600, spy_200ma=500, vix=15, previous_state=None, as_of=as_of
        )
        assert state.trend == TrendState.UPTREND

    def test_clear_downtrend(self, detector, as_of):
        """SPY well below 200MA should be downtrend."""
        state = detector.detect(
            spy_price=480, spy_200ma=500, vix=15, previous_state=None, as_of=as_of
        )
        assert state.trend == TrendState.DOWNTREND

    def test_trend_hysteresis_maintains_uptrend(self, detector, as_of):
        """SPY slightly below 200MA but inside band should maintain uptrend."""
        previous = RegimeState(
            trend=TrendState.UPTREND, vol=VolatilityState.LOW_VOL, as_of=as_of
        )
        # 499 is only 0.2% below 500 (inside 2% band)
        state = detector.detect(
            spy_price=499, spy_200ma=500, vix=15, previous_state=previous, as_of=as_of
        )
        assert state.trend == TrendState.UPTREND

    def test_trend_hysteresis_maintains_downtrend(self, detector, as_of):
        """SPY slightly above 200MA but inside band should maintain downtrend."""
        previous = RegimeState(
            trend=TrendState.DOWNTREND, vol=VolatilityState.LOW_VOL, as_of=as_of
        )
        # 501 is only 0.2% above 500 (inside 2% band)
        state = detector.detect(
            spy_price=501, spy_200ma=500, vix=15, previous_state=previous, as_of=as_of
        )
        assert state.trend == TrendState.DOWNTREND

    def test_trend_crosses_hysteresis_to_downtrend(self, detector, as_of):
        """SPY dropping below lower band should trigger downtrend."""
        previous = RegimeState(
            trend=TrendState.UPTREND, vol=VolatilityState.LOW_VOL, as_of=as_of
        )
        # 489 is 2.2% below 500 (outside 2% band)
        state = detector.detect(
            spy_price=489, spy_200ma=500, vix=15, previous_state=previous, as_of=as_of
        )
        assert state.trend == TrendState.DOWNTREND

    def test_trend_crosses_hysteresis_to_uptrend(self, detector, as_of):
        """SPY rising above upper band should trigger uptrend."""
        previous = RegimeState(
            trend=TrendState.DOWNTREND, vol=VolatilityState.LOW_VOL, as_of=as_of
        )
        # 511 is 2.2% above 500 (outside 2% band)
        state = detector.detect(
            spy_price=511, spy_200ma=500, vix=15, previous_state=previous, as_of=as_of
        )
        assert state.trend == TrendState.UPTREND

    # --- Volatility Detection Tests ---

    def test_clear_low_vol(self, detector, as_of):
        """VIX well below 20 should be low_vol."""
        state = detector.detect(
            spy_price=500, spy_200ma=500, vix=12, previous_state=None, as_of=as_of
        )
        assert state.vol == VolatilityState.LOW_VOL

    def test_clear_high_vol(self, detector, as_of):
        """VIX above 25 should be high_vol."""
        state = detector.detect(
            spy_price=500, spy_200ma=500, vix=30, previous_state=None, as_of=as_of
        )
        assert state.vol == VolatilityState.HIGH_VOL

    def test_vol_hysteresis_maintains_low_vol(self, detector, as_of):
        """VIX in 20-25 band should maintain low_vol."""
        previous = RegimeState(
            trend=TrendState.UPTREND, vol=VolatilityState.LOW_VOL, as_of=as_of
        )
        state = detector.detect(
            spy_price=500, spy_200ma=500, vix=22, previous_state=previous, as_of=as_of
        )
        assert state.vol == VolatilityState.LOW_VOL

    def test_vol_hysteresis_maintains_high_vol(self, detector, as_of):
        """VIX in 20-25 band should maintain high_vol."""
        previous = RegimeState(
            trend=TrendState.UPTREND, vol=VolatilityState.HIGH_VOL, as_of=as_of
        )
        state = detector.detect(
            spy_price=500, spy_200ma=500, vix=22, previous_state=previous, as_of=as_of
        )
        assert state.vol == VolatilityState.HIGH_VOL

    def test_vol_crosses_to_high_vol(self, detector, as_of):
        """VIX rising above 25 should trigger high_vol."""
        previous = RegimeState(
            trend=TrendState.UPTREND, vol=VolatilityState.LOW_VOL, as_of=as_of
        )
        state = detector.detect(
            spy_price=500, spy_200ma=500, vix=26, previous_state=previous, as_of=as_of
        )
        assert state.vol == VolatilityState.HIGH_VOL

    def test_vol_crosses_to_low_vol(self, detector, as_of):
        """VIX dropping below 20 should trigger low_vol."""
        previous = RegimeState(
            trend=TrendState.UPTREND, vol=VolatilityState.HIGH_VOL, as_of=as_of
        )
        state = detector.detect(
            spy_price=500, spy_200ma=500, vix=18, previous_state=previous, as_of=as_of
        )
        assert state.vol == VolatilityState.LOW_VOL

    # --- Combined Regime Tests ---

    def test_all_four_regimes(self, detector, as_of):
        """Verify all 4 regime combinations work."""
        # Uptrend + Low Vol
        state = detector.detect(600, 500, 15, None, as_of)
        assert state.name == "uptrend_low_vol"

        # Uptrend + High Vol
        state = detector.detect(600, 500, 30, None, as_of)
        assert state.name == "uptrend_high_vol"

        # Downtrend + Low Vol
        state = detector.detect(480, 500, 15, None, as_of)
        assert state.name == "downtrend_low_vol"

        # Downtrend + High Vol
        state = detector.detect(480, 500, 30, None, as_of)
        assert state.name == "downtrend_high_vol"

    def test_regime_state_stores_indicators(self, detector, as_of):
        """RegimeState should store indicator values for debugging."""
        state = detector.detect(
            spy_price=600, spy_200ma=500, vix=15, previous_state=None, as_of=as_of
        )
        assert state.spy_price == 600
        assert state.spy_200ma == 500
        assert state.vix == 15
        assert state.as_of == as_of

    # --- Serialization Tests ---

    def test_state_to_dict_roundtrip(self, as_of):
        """RegimeState should survive JSON serialization."""
        state = RegimeState(
            trend=TrendState.UPTREND,
            vol=VolatilityState.HIGH_VOL,
            as_of=as_of,
            spy_price=600.0,
            spy_200ma=550.0,
            vix=28.5,
        )
        data = state.to_dict()
        restored = RegimeState.from_dict(data)

        assert restored.name == state.name
        assert restored.spy_price == state.spy_price
        assert restored.vix == state.vix
        assert restored.trend == state.trend
        assert restored.vol == state.vol

    def test_state_to_dict_structure(self, as_of):
        """Verify to_dict produces expected structure."""
        state = RegimeState(
            trend=TrendState.DOWNTREND,
            vol=VolatilityState.HIGH_VOL,
            as_of=as_of,
            spy_price=480.0,
            spy_200ma=500.0,
            vix=30.0,
        )
        data = state.to_dict()

        assert data["regime"] == "downtrend_high_vol"
        assert data["trend"] == "downtrend"
        assert data["vol"] == "high_vol"
        assert "as_of" in data
        assert data["indicators"]["spy_price"] == 480.0
        assert data["indicators"]["spy_200ma"] == 500.0
        assert data["indicators"]["vix"] == 30.0

    # --- Edge Cases ---

    def test_no_previous_state_defaults(self, detector, as_of):
        """Without previous state, should make reasonable defaults."""
        # Exactly at MA, low VIX -> uptrend_low_vol
        state = detector.detect(500, 500, 15, None, as_of)
        assert state.name == "uptrend_low_vol"

    def test_no_previous_state_below_ma(self, detector, as_of):
        """Without previous state, below MA should be downtrend."""
        # Slightly below MA (but inside band), low VIX -> downtrend_low_vol
        state = detector.detect(499, 500, 15, None, as_of)
        assert state.trend == TrendState.DOWNTREND

    def test_no_previous_state_in_vol_band(self, detector, as_of):
        """Without previous state, VIX in band should default to low_vol."""
        state = detector.detect(600, 500, 22, None, as_of)
        assert state.vol == VolatilityState.LOW_VOL

    def test_custom_config(self, as_of):
        """Custom thresholds should be respected."""
        config = RegimeConfig(
            trend_hysteresis_pct=0.05,  # 5% instead of 2%
            vix_high_threshold=30,  # 30 instead of 25
            vix_low_threshold=15,  # 15 instead of 20
        )
        detector = RegimeDetector(config)

        # VIX 25 should be low_vol with higher threshold
        state = detector.detect(500, 500, 25, None, as_of)
        assert state.vol == VolatilityState.LOW_VOL

        # With previous uptrend state, SPY at 96% of MA should maintain uptrend with 5% band
        previous = RegimeState(
            trend=TrendState.UPTREND, vol=VolatilityState.LOW_VOL, as_of=as_of
        )
        state = detector.detect(480, 500, 15, previous, as_of)
        assert state.trend == TrendState.UPTREND  # 4% below, inside 5% band

    def test_default_state(self, as_of):
        """get_default_state should return uptrend_low_vol."""
        state = RegimeDetector.get_default_state(as_of)
        assert state.name == "uptrend_low_vol"
        assert state.trend == TrendState.UPTREND
        assert state.vol == VolatilityState.LOW_VOL
        assert state.as_of == as_of

    # --- Boundary Tests ---

    def test_exactly_at_vix_high_threshold(self, detector, as_of):
        """VIX exactly at 25 should NOT trigger high_vol (must be > 25)."""
        state = detector.detect(600, 500, 25, None, as_of)
        assert state.vol == VolatilityState.LOW_VOL

    def test_exactly_at_vix_low_threshold(self, detector, as_of):
        """VIX exactly at 20 should NOT trigger low_vol (must be < 20)."""
        previous = RegimeState(
            trend=TrendState.UPTREND, vol=VolatilityState.HIGH_VOL, as_of=as_of
        )
        state = detector.detect(600, 500, 20, previous, as_of)
        assert state.vol == VolatilityState.HIGH_VOL

    def test_exactly_at_trend_lower_band(self, detector, as_of):
        """SPY exactly at 98% of MA should NOT trigger downtrend (must be < 98%)."""
        previous = RegimeState(
            trend=TrendState.UPTREND, vol=VolatilityState.LOW_VOL, as_of=as_of
        )
        # 490 is exactly 98% of 500
        state = detector.detect(490, 500, 15, previous, as_of)
        assert state.trend == TrendState.UPTREND

    def test_exactly_at_trend_upper_band(self, detector, as_of):
        """SPY exactly at 102% of MA should NOT trigger uptrend (must be > 102%)."""
        previous = RegimeState(
            trend=TrendState.DOWNTREND, vol=VolatilityState.LOW_VOL, as_of=as_of
        )
        # 510 is exactly 102% of 500
        state = detector.detect(510, 500, 15, previous, as_of)
        assert state.trend == TrendState.DOWNTREND


class TestRegimeConfig:
    """Tests for RegimeConfig dataclass."""

    def test_default_values(self):
        """Default config should have expected values."""
        config = RegimeConfig()
        assert config.trend_hysteresis_pct == 0.02
        assert config.vix_high_threshold == 25.0
        assert config.vix_low_threshold == 20.0

    def test_custom_values(self):
        """Custom config values should be stored."""
        config = RegimeConfig(
            trend_hysteresis_pct=0.03,
            vix_high_threshold=28.0,
            vix_low_threshold=18.0,
        )
        assert config.trend_hysteresis_pct == 0.03
        assert config.vix_high_threshold == 28.0
        assert config.vix_low_threshold == 18.0

    def test_config_is_frozen(self):
        """RegimeConfig should be immutable."""
        config = RegimeConfig()
        with pytest.raises(AttributeError):
            config.vix_high_threshold = 30.0


class TestTrendState:
    """Tests for TrendState enum."""

    def test_values(self):
        """TrendState should have expected values."""
        assert TrendState.UPTREND.value == "uptrend"
        assert TrendState.DOWNTREND.value == "downtrend"

    def test_from_string(self):
        """TrendState should be constructable from string value."""
        assert TrendState("uptrend") == TrendState.UPTREND
        assert TrendState("downtrend") == TrendState.DOWNTREND


class TestVolatilityState:
    """Tests for VolatilityState enum."""

    def test_values(self):
        """VolatilityState should have expected values."""
        assert VolatilityState.LOW_VOL.value == "low_vol"
        assert VolatilityState.HIGH_VOL.value == "high_vol"

    def test_from_string(self):
        """VolatilityState should be constructable from string value."""
        assert VolatilityState("low_vol") == VolatilityState.LOW_VOL
        assert VolatilityState("high_vol") == VolatilityState.HIGH_VOL
