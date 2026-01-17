"""Tests for alpha model selection framework."""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, List
import pandas as pd

from quantetf.alpha.selector import (
    AlphaSelector,
    AlphaSelection,
    MarketRegime,
    RegimeBasedSelector,
    RegimeWeightedSelector,
    ConfigurableSelector,
    compute_alpha_with_selection,
)


class TestMarketRegimeEnum:
    """Test MarketRegime enum."""

    def test_all_regimes_defined(self):
        """Test that all expected regimes exist."""
        expected = {"risk_on", "elevated_vol", "high_vol", "recession_warning", "trending", "mean_reverting", "unknown"}
        actual = {r.value for r in MarketRegime}
        assert actual == expected

    def test_regime_values(self):
        """Test that regime values are strings."""
        for regime in MarketRegime:
            assert isinstance(regime.value, str)
            assert regime.value.islower()

    def test_regime_equality(self):
        """Test regime equality and identity."""
        assert MarketRegime.RISK_ON == MarketRegime.RISK_ON
        assert MarketRegime.RISK_ON is MarketRegime.RISK_ON
        assert MarketRegime.RISK_ON != MarketRegime.HIGH_VOL


class TestAlphaSelectionDataclass:
    """Test AlphaSelection dataclass."""

    def test_single_model_creation(self):
        """Test creating AlphaSelection with single model."""
        model = Mock()
        selection = AlphaSelection(
            model=model,
            regime=MarketRegime.RISK_ON,
            confidence=0.95,
        )
        assert selection.is_single_model
        assert not selection.is_ensemble
        assert selection.model is model
        assert selection.model_weights is None

    def test_ensemble_creation(self):
        """Test creating AlphaSelection with ensemble weights."""
        weights = {"model1": 0.6, "model2": 0.4}
        selection = AlphaSelection(
            model_weights=weights,
            regime=MarketRegime.HIGH_VOL,
            confidence=0.85,
        )
        assert selection.is_ensemble
        assert not selection.is_single_model
        assert selection.model_weights == weights
        assert selection.model is None

    def test_cannot_specify_both_model_and_weights(self):
        """Test that both model and model_weights cannot be specified."""
        model = Mock()
        weights = {"model1": 1.0}
        with pytest.raises(ValueError, match="Cannot specify both model and model_weights"):
            AlphaSelection(
                model=model,
                model_weights=weights,
            )

    def test_must_specify_one_of_model_or_weights(self):
        """Test that at least one of model or model_weights must be specified."""
        with pytest.raises(ValueError, match="Must specify either model or model_weights"):
            AlphaSelection(regime=MarketRegime.UNKNOWN)

    def test_confidence_validation(self):
        """Test that confidence must be between 0 and 1."""
        model = Mock()
        
        # Valid values
        AlphaSelection(model=model, confidence=0.0)
        AlphaSelection(model=model, confidence=1.0)
        AlphaSelection(model=model, confidence=0.5)
        
        # Invalid values
        with pytest.raises(ValueError, match="Confidence must be between"):
            AlphaSelection(model=model, confidence=-0.1)
        
        with pytest.raises(ValueError, match="Confidence must be between"):
            AlphaSelection(model=model, confidence=1.1)

    def test_default_regime_is_unknown(self):
        """Test that default regime is UNKNOWN."""
        model = Mock()
        selection = AlphaSelection(model=model)
        assert selection.regime == MarketRegime.UNKNOWN

    def test_default_confidence_is_one(self):
        """Test that default confidence is 1.0."""
        model = Mock()
        selection = AlphaSelection(model=model)
        assert selection.confidence == 1.0


class TestRegimeBasedSelector:
    """Test RegimeBasedSelector implementation."""

    def test_creation(self):
        """Test creating a regime-based selector."""
        mapping = {
            MarketRegime.RISK_ON: "momentum",
            MarketRegime.HIGH_VOL: "vol_momentum",
        }
        selector = RegimeBasedSelector(regime_model_map=mapping, default_model="safe_momentum")
        assert selector.regime_model_map == mapping
        assert selector.default_model == "safe_momentum"

    def test_select_mapped_regime(self):
        """Test selection for a mapped regime."""
        mapping = {MarketRegime.RISK_ON: "momentum_model"}
        selector = RegimeBasedSelector(regime_model_map=mapping)
        
        model = Mock()
        available = {"momentum_model": model}
        
        selection = selector.select(MarketRegime.RISK_ON, pd.Timestamp("2024-01-01"), available)
        
        assert selection.is_single_model
        assert selection.model is model
        assert selection.regime == MarketRegime.RISK_ON
        assert selection.confidence == 1.0

    def test_select_unmapped_regime_uses_default(self):
        """Test selection for unmapped regime falls back to default."""
        mapping = {MarketRegime.RISK_ON: "momentum_model"}
        selector = RegimeBasedSelector(regime_model_map=mapping, default_model="safe_model")
        
        model = Mock()
        available = {"safe_model": model}
        
        selection = selector.select(MarketRegime.HIGH_VOL, pd.Timestamp("2024-01-01"), available)
        
        assert selection.model is model
        assert selection.regime == MarketRegime.HIGH_VOL

    def test_select_fails_when_model_not_available(self):
        """Test that selection fails when mapped model is not available."""
        mapping = {MarketRegime.RISK_ON: "missing_model"}
        selector = RegimeBasedSelector(regime_model_map=mapping)
        
        available = {"other_model": Mock()}
        
        with pytest.raises(ValueError, match="not in available"):
            selector.select(MarketRegime.RISK_ON, pd.Timestamp("2024-01-01"), available)

    def test_get_supported_regimes(self):
        """Test get_supported_regimes returns mapped regimes."""
        mapping = {
            MarketRegime.RISK_ON: "model1",
            MarketRegime.HIGH_VOL: "model2",
        }
        selector = RegimeBasedSelector(regime_model_map=mapping)
        
        supported = selector.get_supported_regimes()
        assert set(supported) == {MarketRegime.RISK_ON, MarketRegime.HIGH_VOL}


class TestRegimeWeightedSelector:
    """Test RegimeWeightedSelector implementation."""

    def test_creation(self):
        """Test creating a regime-weighted selector."""
        weights = {
            MarketRegime.RISK_ON: {"model1": 0.6, "model2": 0.4},
        }
        defaults = {"model1": 1.0}
        selector = RegimeWeightedSelector(regime_weights=weights, default_weights=defaults)
        
        assert selector.regime_weights == weights
        assert selector.default_weights == defaults

    def test_select_mapped_regime(self):
        """Test selection for a mapped regime."""
        weights = {
            MarketRegime.RISK_ON: {"model1": 0.6, "model2": 0.4},
        }
        selector = RegimeWeightedSelector(regime_weights=weights, default_weights={"model1": 1.0})
        
        available = {"model1": Mock(), "model2": Mock()}
        selection = selector.select(MarketRegime.RISK_ON, pd.Timestamp("2024-01-01"), available)
        
        assert selection.is_ensemble
        assert selection.model_weights == {"model1": 0.6, "model2": 0.4}

    def test_select_unmapped_regime_uses_default_weights(self):
        """Test selection for unmapped regime uses default weights."""
        weights = {MarketRegime.RISK_ON: {"model1": 1.0}}
        defaults = {"model1": 0.5, "model2": 0.5}
        selector = RegimeWeightedSelector(regime_weights=weights, default_weights=defaults)
        
        available = {"model1": Mock(), "model2": Mock()}
        selection = selector.select(MarketRegime.HIGH_VOL, pd.Timestamp("2024-01-01"), available)
        
        assert selection.model_weights == defaults

    def test_weights_are_normalized(self):
        """Test that weights are normalized to sum to 1."""
        weights = {
            MarketRegime.RISK_ON: {"model1": 2.0, "model2": 2.0},  # total 4.0
        }
        selector = RegimeWeightedSelector(regime_weights=weights, default_weights={"model1": 1.0})
        
        available = {"model1": Mock(), "model2": Mock()}
        selection = selector.select(MarketRegime.RISK_ON, pd.Timestamp("2024-01-01"), available)
        
        assert selection.model_weights == {"model1": 0.5, "model2": 0.5}
        assert abs(sum(selection.model_weights.values()) - 1.0) < 1e-10

    def test_fails_when_required_model_not_available(self):
        """Test that selection fails when required model is not available."""
        weights = {
            MarketRegime.RISK_ON: {"model1": 0.6, "missing": 0.4},
        }
        selector = RegimeWeightedSelector(regime_weights=weights, default_weights={"model1": 1.0})
        
        available = {"model1": Mock()}
        
        with pytest.raises(ValueError, match="not available"):
            selector.select(MarketRegime.RISK_ON, pd.Timestamp("2024-01-01"), available)

    def test_fails_on_negative_weight(self):
        """Test that negative weights are rejected."""
        with pytest.raises(ValueError, match="weight must be >= 0"):
            RegimeWeightedSelector(
                regime_weights={MarketRegime.RISK_ON: {"model1": -0.1}},
                default_weights={"model1": 1.0},
            )

    def test_fails_on_all_zero_weights(self):
        """Test that all-zero weights are rejected."""
        weights = {MarketRegime.RISK_ON: {"model1": 0.0, "model2": 0.0}}
        selector = RegimeWeightedSelector(regime_weights=weights, default_weights={"model1": 1.0})
        
        available = {"model1": Mock(), "model2": Mock()}
        
        with pytest.raises(ValueError, match="total weight is 0"):
            selector.select(MarketRegime.RISK_ON, pd.Timestamp("2024-01-01"), available)

    def test_get_supported_regimes(self):
        """Test get_supported_regimes returns configured regimes."""
        weights = {
            MarketRegime.RISK_ON: {"model1": 1.0},
            MarketRegime.HIGH_VOL: {"model2": 1.0},
        }
        selector = RegimeWeightedSelector(regime_weights=weights, default_weights={"model1": 1.0})
        
        supported = selector.get_supported_regimes()
        assert set(supported) == {MarketRegime.RISK_ON, MarketRegime.HIGH_VOL}


class TestConfigurableSelector:
    """Test ConfigurableSelector YAML-driven configuration."""

    def test_regime_based_selector_from_config(self):
        """Test building regime-based selector from config."""
        config = {
            "type": "regime_based",
            "default_model": "default_momentum",
            "regime_mapping": {
                "risk_on": "aggressive_momentum",
                "high_vol": "conservative_momentum",
            }
        }
        selector = ConfigurableSelector(config)
        
        # Verify it's using the right inner type
        assert isinstance(selector._inner, RegimeBasedSelector)
        
        # Verify it works
        model = Mock()
        available = {"conservative_momentum": model}
        selection = selector.select(MarketRegime.HIGH_VOL, pd.Timestamp("2024-01-01"), available)
        
        assert selection.model is model

    def test_regime_weighted_selector_from_config(self):
        """Test building regime-weighted selector from config."""
        config = {
            "type": "regime_weighted",
            "default_weights": {"momentum": 1.0},
            "regime_mapping": {
                "risk_on": {"momentum": 0.6, "acceleration": 0.4},
            }
        }
        selector = ConfigurableSelector(config)
        
        assert isinstance(selector._inner, RegimeWeightedSelector)
        
        available = {"momentum": Mock(), "acceleration": Mock()}
        selection = selector.select(MarketRegime.RISK_ON, pd.Timestamp("2024-01-01"), available)
        
        assert selection.is_ensemble
        assert selection.model_weights == {"momentum": 0.6, "acceleration": 0.4}

    def test_converts_string_regime_keys_to_enum(self):
        """Test that string regime keys are converted to MarketRegime enum."""
        config = {
            "type": "regime_based",
            "default_model": "default",
            "regime_mapping": {
                "risk_on": "model1",
                "high_vol": "model2",
            }
        }
        selector = ConfigurableSelector(config)
        
        # Verify regime_model_map has enum keys
        assert all(isinstance(k, MarketRegime) for k in selector._inner.regime_model_map.keys())

    def test_fails_on_invalid_regime_name(self):
        """Test that invalid regime names are rejected."""
        config = {
            "type": "regime_based",
            "default_model": "default",
            "regime_mapping": {
                "invalid_regime_xyz": "model1",
            }
        }
        with pytest.raises(ValueError, match="Invalid regime name"):
            ConfigurableSelector(config)

    def test_fails_on_invalid_selector_type(self):
        """Test that invalid selector type is rejected."""
        config = {
            "type": "unknown_type",
            "regime_mapping": {"risk_on": "model1"},
        }
        with pytest.raises(ValueError, match="Unknown selector type"):
            ConfigurableSelector(config)

    def test_fails_when_regime_mapping_missing(self):
        """Test that missing regime_mapping is rejected."""
        config = {"type": "regime_based"}
        with pytest.raises(ValueError, match="regime_mapping"):
            ConfigurableSelector(config)

    def test_get_supported_regimes_delegates_to_inner(self):
        """Test that get_supported_regimes delegates to inner selector."""
        config = {
            "type": "regime_based",
            "default_model": "default",
            "regime_mapping": {
                "risk_on": "model1",
                "high_vol": "model2",
            }
        }
        selector = ConfigurableSelector(config)
        
        supported = selector.get_supported_regimes()
        assert set(supported) == {MarketRegime.RISK_ON, MarketRegime.HIGH_VOL}


class TestComputeAlphaWithSelection:
    """Test compute_alpha_with_selection helper function."""

    def test_single_model_scoring(self):
        """Test scoring with single selected model."""
        # Create mock model that returns scores
        model = Mock()
        scores = pd.Series([0.1, 0.2, 0.3], index=["AAPL", "MSFT", "GOOGL"])
        model.score.return_value = scores
        
        # Create selector that returns that model
        selector = Mock(spec=AlphaSelector)
        selection = AlphaSelection(model=model, regime=MarketRegime.RISK_ON)
        selector.select.return_value = selection
        
        # Score
        result = compute_alpha_with_selection(
            selector=selector,
            regime=MarketRegime.RISK_ON,
            as_of=pd.Timestamp("2024-01-01"),
            available_models={"model": model},
            universe=["AAPL", "MSFT", "GOOGL"],
            data_store=Mock(),
        )
        
        # Verify
        pd.testing.assert_series_equal(result, scores)
        model.score.assert_called_once()

    def test_ensemble_scoring_with_equal_weights(self):
        """Test scoring with equally weighted ensemble."""
        # Create models
        model1 = Mock()
        model1.score.return_value = pd.Series([0.2, 0.4], index=["A", "B"])
        
        model2 = Mock()
        model2.score.return_value = pd.Series([0.4, 0.2], index=["A", "B"])
        
        # Create selector
        selector = Mock(spec=AlphaSelector)
        weights = {"model1": 0.5, "model2": 0.5}
        selection = AlphaSelection(model_weights=weights, regime=MarketRegime.RISK_ON)
        selector.select.return_value = selection
        
        # Score
        result = compute_alpha_with_selection(
            selector=selector,
            regime=MarketRegime.RISK_ON,
            as_of=pd.Timestamp("2024-01-01"),
            available_models={"model1": model1, "model2": model2},
            universe=["A", "B"],
            data_store=Mock(),
        )
        
        # Expected: average of the two scores
        expected = pd.Series([0.3, 0.3], index=["A", "B"])
        pd.testing.assert_series_equal(result, expected)

    def test_ensemble_scoring_with_unequal_weights(self):
        """Test scoring with unequally weighted ensemble."""
        model1 = Mock()
        model1.score.return_value = pd.Series([1.0], index=["X"])
        
        model2 = Mock()
        model2.score.return_value = pd.Series([2.0], index=["X"])
        
        selector = Mock(spec=AlphaSelector)
        weights = {"model1": 0.25, "model2": 0.75}
        selection = AlphaSelection(model_weights=weights, regime=MarketRegime.RISK_ON)
        selector.select.return_value = selection
        
        result = compute_alpha_with_selection(
            selector=selector,
            regime=MarketRegime.RISK_ON,
            as_of=pd.Timestamp("2024-01-01"),
            available_models={"model1": model1, "model2": model2},
            universe=["X"],
            data_store=Mock(),
        )
        
        # Expected: 1.0 * 0.25 + 2.0 * 0.75 = 1.75
        expected = pd.Series([1.75], index=["X"])
        pd.testing.assert_series_equal(result, expected)

    def test_fails_when_model_scoring_raises(self):
        """Test that model scoring errors are propagated."""
        model = Mock()
        model.score.side_effect = RuntimeError("Score failed")
        
        selector = Mock(spec=AlphaSelector)
        weights = {"model": 1.0}
        selection = AlphaSelection(model_weights=weights, regime=MarketRegime.RISK_ON)
        selector.select.return_value = selection
        
        with pytest.raises(RuntimeError, match="Failed to score with model"):
            compute_alpha_with_selection(
                selector=selector,
                regime=MarketRegime.RISK_ON,
                as_of=pd.Timestamp("2024-01-01"),
                available_models={"model": model},
                universe=["A"],
                data_store=Mock(),
            )

    def test_ensemble_with_missing_scores(self):
        """Test ensemble scoring when model returns partial results."""
        model1 = Mock()
        model1.score.return_value = pd.Series([1.0, 2.0], index=["A", "B"])
        
        model2 = Mock()
        model2.score.return_value = pd.Series([0.5, 1.5], index=["A", "B"])
        
        selector = Mock(spec=AlphaSelector)
        weights = {"model1": 0.5, "model2": 0.5}
        selection = AlphaSelection(model_weights=weights, regime=MarketRegime.RISK_ON)
        selector.select.return_value = selection
        
        result = compute_alpha_with_selection(
            selector=selector,
            regime=MarketRegime.RISK_ON,
            as_of=pd.Timestamp("2024-01-01"),
            available_models={"model1": model1, "model2": model2},
            universe=["A", "B"],
            data_store=Mock(),
        )
        
        expected = pd.Series([0.75, 1.75], index=["A", "B"])
        pd.testing.assert_series_equal(result, expected)
