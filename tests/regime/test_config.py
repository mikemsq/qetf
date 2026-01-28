"""Tests for regime configuration loading."""

import pytest
from pathlib import Path

from quantetf.regime.config import (
    load_thresholds,
    load_regime_mapping,
    get_strategy_for_regime,
)
from quantetf.regime.types import RegimeConfig


class TestLoadThresholds:
    """Test threshold configuration loading."""

    def test_load_default_thresholds(self, tmp_path):
        """Load thresholds from valid YAML."""
        config_file = tmp_path / "thresholds.yaml"
        config_file.write_text("""
version: "1.0"
trend:
  hysteresis_pct: 0.03
volatility:
  high_threshold: 30
  low_threshold: 18
        """)

        config = load_thresholds(config_file)

        assert config.trend_hysteresis_pct == 0.03
        assert config.vix_high_threshold == 30
        assert config.vix_low_threshold == 18

    def test_missing_file_returns_defaults(self, tmp_path):
        """Missing file should return default config."""
        config = load_thresholds(tmp_path / "nonexistent.yaml")

        assert config.trend_hysteresis_pct == 0.02
        assert config.vix_high_threshold == 25
        assert config.vix_low_threshold == 20

    def test_malformed_config_raises(self, tmp_path):
        """Malformed config should raise ValueError."""
        config_file = tmp_path / "thresholds.yaml"
        config_file.write_text("""
trend:
  wrong_field: 0.03
        """)

        with pytest.raises(ValueError, match="Missing required field"):
            load_thresholds(config_file)

    def test_load_production_thresholds(self):
        """Load actual production thresholds.yaml if it exists."""
        prod_path = Path("configs/regimes/thresholds.yaml")
        if prod_path.exists():
            config = load_thresholds(prod_path)
            assert isinstance(config, RegimeConfig)
            assert config.trend_hysteresis_pct > 0
            assert config.vix_high_threshold > config.vix_low_threshold


class TestLoadRegimeMapping:
    """Test regime mapping loading."""

    def test_load_valid_mapping(self, tmp_path):
        """Load mapping with all 4 regimes."""
        config_file = tmp_path / "mapping.yaml"
        config_file.write_text("""
mapping:
  uptrend_low_vol:
    strategy: momentum_accel
    config_path: path/to/config.yaml
  uptrend_high_vol:
    strategy: vol_mom
    config_path: path/to/config2.yaml
  downtrend_low_vol:
    strategy: short_mom
    config_path: path/to/config3.yaml
  downtrend_high_vol:
    strategy: defensive
    config_path: path/to/config4.yaml
fallback:
  strategy: momentum_accel
  config_path: path/to/fallback.yaml
        """)

        mapping = load_regime_mapping(config_file)

        assert mapping["uptrend_low_vol"]["strategy"] == "momentum_accel"
        assert mapping["downtrend_high_vol"]["strategy"] == "defensive"
        assert mapping["fallback"]["strategy"] == "momentum_accel"

    def test_missing_regime_raises(self, tmp_path):
        """Missing regime should raise ValueError."""
        config_file = tmp_path / "mapping.yaml"
        config_file.write_text("""
mapping:
  uptrend_low_vol:
    strategy: test
fallback:
  strategy: test
        """)

        with pytest.raises(ValueError, match="Missing mapping for regime"):
            load_regime_mapping(config_file)

    def test_missing_file_raises(self, tmp_path):
        """Missing mapping file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_regime_mapping(tmp_path / "nonexistent.yaml")

    def test_load_production_mapping(self):
        """Load actual production default_mapping.yaml if it exists."""
        prod_path = Path("configs/regimes/default_mapping.yaml")
        if prod_path.exists():
            mapping = load_regime_mapping(prod_path)
            assert "uptrend_low_vol" in mapping
            assert "uptrend_high_vol" in mapping
            assert "downtrend_low_vol" in mapping
            assert "downtrend_high_vol" in mapping
            assert "fallback" in mapping


class TestGetStrategyForRegime:
    """Test strategy lookup."""

    @pytest.fixture
    def mapping(self):
        return {
            "uptrend_low_vol": {"strategy": "momentum", "config_path": "a.yaml"},
            "uptrend_high_vol": {"strategy": "vol_adj", "config_path": "b.yaml"},
            "downtrend_low_vol": {"strategy": "short", "config_path": "c.yaml"},
            "downtrend_high_vol": {"strategy": "defensive", "config_path": "d.yaml"},
            "fallback": {"strategy": "default", "config_path": "default.yaml"},
        }

    def test_get_known_regime(self, mapping):
        """Known regime returns correct strategy."""
        result = get_strategy_for_regime("uptrend_low_vol", mapping)
        assert result["strategy"] == "momentum"

    def test_get_all_regimes(self, mapping):
        """All 4 regimes return correct strategies."""
        assert get_strategy_for_regime("uptrend_low_vol", mapping)["strategy"] == "momentum"
        assert get_strategy_for_regime("uptrend_high_vol", mapping)["strategy"] == "vol_adj"
        assert get_strategy_for_regime("downtrend_low_vol", mapping)["strategy"] == "short"
        assert get_strategy_for_regime("downtrend_high_vol", mapping)["strategy"] == "defensive"

    def test_unknown_regime_returns_fallback(self, mapping):
        """Unknown regime returns fallback."""
        result = get_strategy_for_regime("invalid_regime", mapping)
        assert result["strategy"] == "default"

    def test_fallback_regime_name(self, mapping):
        """Requesting 'fallback' directly works."""
        result = get_strategy_for_regime("fallback", mapping)
        assert result["strategy"] == "default"

    def test_strategy_has_config_path(self, mapping):
        """Each strategy should have a config_path."""
        result = get_strategy_for_regime("uptrend_low_vol", mapping)
        assert "config_path" in result
        assert result["config_path"] == "a.yaml"


class TestProductionConfigs:
    """Integration tests with actual production config files."""

    def test_thresholds_and_mapping_compatible(self):
        """Production thresholds and mapping should be loadable together."""
        thresh_path = Path("configs/regimes/thresholds.yaml")
        mapping_path = Path("configs/regimes/default_mapping.yaml")

        if thresh_path.exists() and mapping_path.exists():
            config = load_thresholds(thresh_path)
            mapping = load_regime_mapping(mapping_path)

            # All regimes from thresholds should be in mapping
            expected_regimes = [
                "uptrend_low_vol",
                "uptrend_high_vol",
                "downtrend_low_vol",
                "downtrend_high_vol",
            ]
            for regime in expected_regimes:
                assert regime in mapping
                strategy_info = get_strategy_for_regime(regime, mapping)
                assert "strategy" in strategy_info
