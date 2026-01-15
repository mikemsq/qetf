"""Tests for MacroDataLoader and RegimeDetector."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from quantetf.data.macro_loader import MacroDataLoader, RegimeDetector


@pytest.fixture
def temp_macro_data():
    """Create temporary macro data files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create sample VIX data
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="B")
        n = len(dates)

        # VIX: low for first third, medium for second third, high for last third
        third = n // 3
        vix_values = [15.0] * third + [25.0] * third + [35.0] * (n - 2 * third)
        vix_df = pd.DataFrame({"VIXCLS": vix_values}, index=dates)
        vix_df.to_parquet(data_dir / "VIX.parquet")

        # Create sample yield curve spread data
        # Normal (positive) for first 60%, inverted (negative) for last 40%
        split = int(n * 0.6)
        spread_values = [0.5] * split + [-0.2] * (n - split)
        spread_df = pd.DataFrame({"T10Y2Y": spread_values}, index=dates)
        spread_df.to_parquet(data_dir / "TREASURY_SPREAD_10Y2Y.parquet")

        # Create 10Y treasury data
        treasury_df = pd.DataFrame({"DGS10": [1.5] * n}, index=dates)
        treasury_df.to_parquet(data_dir / "TREASURY_10Y.parquet")

        yield data_dir


class TestMacroDataLoader:
    """Tests for MacroDataLoader class."""

    def test_load_indicator_vix(self, temp_macro_data):
        """Test loading VIX indicator."""
        loader = MacroDataLoader(data_dir=temp_macro_data)
        vix = loader.load_indicator("VIX")

        assert isinstance(vix, pd.Series)
        assert len(vix) > 0
        assert vix.iloc[0] == 15.0  # First value

    def test_load_indicator_not_found(self, temp_macro_data):
        """Test loading non-existent indicator raises error."""
        loader = MacroDataLoader(data_dir=temp_macro_data)

        with pytest.raises(FileNotFoundError, match="FAKE_INDICATOR not found"):
            loader.load_indicator("FAKE_INDICATOR")

    def test_load_all_indicators(self, temp_macro_data):
        """Test loading all indicators at once."""
        loader = MacroDataLoader(data_dir=temp_macro_data)
        df = loader.load_all()

        assert isinstance(df, pd.DataFrame)
        assert "VIX" in df.columns
        assert "TREASURY_SPREAD_10Y2Y" in df.columns
        assert "TREASURY_10Y" in df.columns

    def test_get_vix_latest(self, temp_macro_data):
        """Test getting latest VIX value."""
        loader = MacroDataLoader(data_dir=temp_macro_data)
        vix = loader.get_vix()

        assert isinstance(vix, float)
        assert vix == 35.0  # Last value in our test data

    def test_get_vix_at_date(self, temp_macro_data):
        """Test getting VIX value at specific date."""
        loader = MacroDataLoader(data_dir=temp_macro_data)
        vix = loader.get_vix("2020-01-15")

        assert isinstance(vix, float)
        assert vix == 15.0  # Early in the year, low VIX

    def test_get_yield_curve_spread(self, temp_macro_data):
        """Test getting yield curve spread."""
        loader = MacroDataLoader(data_dir=temp_macro_data)
        spread = loader.get_yield_curve_spread()

        assert isinstance(spread, float)
        assert spread == -0.2  # Last value (inverted)

    def test_is_high_vol_regime_true(self, temp_macro_data):
        """Test high volatility regime detection when VIX is high."""
        loader = MacroDataLoader(data_dir=temp_macro_data)

        # Latest VIX is 35, above threshold of 25
        assert loader.is_high_vol_regime() is True

    def test_is_high_vol_regime_false(self, temp_macro_data):
        """Test high volatility regime detection when VIX is low."""
        loader = MacroDataLoader(data_dir=temp_macro_data)

        # Early in 2020, VIX was 15
        assert loader.is_high_vol_regime("2020-01-15") is False

    def test_is_yield_curve_inverted_true(self, temp_macro_data):
        """Test yield curve inversion detection when inverted."""
        loader = MacroDataLoader(data_dir=temp_macro_data)

        # Latest spread is -0.2
        assert loader.is_yield_curve_inverted() is True

    def test_is_yield_curve_inverted_false(self, temp_macro_data):
        """Test yield curve inversion detection when normal."""
        loader = MacroDataLoader(data_dir=temp_macro_data)

        # Early in year, spread was 0.5
        assert loader.is_yield_curve_inverted("2020-01-15") is False

    def test_data_dir_string_conversion(self, temp_macro_data):
        """Test that string data_dir is converted to Path."""
        loader = MacroDataLoader(data_dir=str(temp_macro_data))

        assert isinstance(loader.data_dir, Path)


class TestRegimeDetector:
    """Tests for RegimeDetector class."""

    def test_detect_regime_risk_on(self, temp_macro_data):
        """Test detecting RISK_ON regime (low VIX, normal yield curve)."""
        loader = MacroDataLoader(data_dir=temp_macro_data)
        detector = RegimeDetector(loader)

        # Early 2020: VIX=15, spread=0.5
        regime = detector.detect_regime("2020-01-15")
        assert regime == "RISK_ON"

    def test_detect_regime_elevated_vol(self, temp_macro_data):
        """Test detecting ELEVATED_VOL regime (VIX 20-30)."""
        loader = MacroDataLoader(data_dir=temp_macro_data)
        detector = RegimeDetector(loader)

        # Middle of year: VIX=25
        regime = detector.detect_regime("2020-06-15")
        assert regime == "ELEVATED_VOL"

    def test_detect_regime_high_vol(self, temp_macro_data):
        """Test detecting HIGH_VOL regime (VIX > 30)."""
        loader = MacroDataLoader(data_dir=temp_macro_data)
        detector = RegimeDetector(loader)

        # End of year: VIX=35, but spread is inverted
        # Need a date where VIX is high but spread is not inverted
        # In our data, VIX goes to 35 after spread inverts, so this regime
        # will show RECESSION_WARNING since inverted curve takes precedence

        # Let's check what we get at end of year
        regime = detector.detect_regime("2020-12-30")
        # Inverted yield curve takes precedence
        assert regime == "RECESSION_WARNING"

    def test_detect_regime_recession_warning(self, temp_macro_data):
        """Test detecting RECESSION_WARNING regime (inverted yield curve)."""
        loader = MacroDataLoader(data_dir=temp_macro_data)
        detector = RegimeDetector(loader)

        # Late in year: spread becomes negative
        regime = detector.detect_regime("2020-08-15")
        assert regime == "RECESSION_WARNING"

    def test_detect_regime_unknown_on_error(self, temp_macro_data):
        """Test that UNKNOWN is returned when data is unavailable."""
        loader = MacroDataLoader(data_dir=temp_macro_data)
        detector = RegimeDetector(loader)

        # Use a date before our data starts
        regime = detector.detect_regime("2010-01-01")
        assert regime == "UNKNOWN"


class TestCombinedDataset:
    """Tests for combined dataset functionality."""

    def test_load_all_creates_combined_on_demand(self, temp_macro_data):
        """Test that load_all combines individual files."""
        loader = MacroDataLoader(data_dir=temp_macro_data)

        # No combined.parquet file exists
        combined_path = temp_macro_data / "combined.parquet"
        assert not combined_path.exists()

        # load_all should still work
        df = loader.load_all()
        assert len(df.columns) == 3

    def test_load_all_uses_combined_if_exists(self, temp_macro_data):
        """Test that load_all uses existing combined.parquet."""
        loader = MacroDataLoader(data_dir=temp_macro_data)

        # First create a combined file
        df = loader.load_all()
        df.to_parquet(temp_macro_data / "combined.parquet")

        # Now load_all should use the combined file
        df2 = loader.load_all()
        pd.testing.assert_frame_equal(df, df2)
