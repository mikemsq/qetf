"""Tests for regime-aware alpha integration (IMPL-018)."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from quantetf.alpha.base import AlphaModel
from quantetf.alpha.selector import (
    AlphaSelector,
    AlphaSelection,
    MarketRegime,
    RegimeBasedSelector,
)
from quantetf.alpha.regime_aware import (
    RegimeDetector,
    RegimeAwareAlpha,
)
from quantetf.data.macro_loader import MacroDataLoader
from quantetf.types import Universe


class DummyAlpha(AlphaModel):
    """Simple alpha model for testing."""

    def __init__(self, name: str, scores: dict):
        self._name = name
        self.scores = scores

    @property
    def name(self) -> str:
        return self._name

    def score(self, *, as_of, universe, features, store, dataset_version=None):
        # Return scores for all tickers in universe
        return pd.Series(
            {t: self.scores.get(t, 0.5) for t in universe.tickers}
        )


class TestRegimeDetector:
    """Test suite for RegimeDetector class."""

    @pytest.fixture
    def macro_loader(self):
        """Create mock macro loader."""
        loader = Mock(spec=MacroDataLoader)
        return loader

    @pytest.fixture
    def detector(self, macro_loader):
        """Create detector instance."""
        return RegimeDetector(macro_loader)

    def test_init(self, macro_loader):
        """Test detector initialization."""
        detector = RegimeDetector(macro_loader)
        assert detector.macro is macro_loader

    def test_detect_regime_risk_on(self, detector, macro_loader):
        """Test regime detection for risk-on conditions."""
        # Low volatility, positive yield curve
        macro_loader.get_macro_snapshot.return_value = {
            "vix": 15.0,
            "yield_curve_10y2y": 0.5,
            "hy_spread": 250.0,
        }

        regime = detector.detect_regime(pd.Timestamp("2024-01-01"))
        assert regime == MarketRegime.RISK_ON

    def test_detect_regime_elevated_vol(self, detector, macro_loader):
        """Test regime detection for elevated volatility."""
        macro_loader.get_macro_snapshot.return_value = {
            "vix": 22.0,
            "yield_curve_10y2y": 0.5,
            "hy_spread": 300.0,
        }

        regime = detector.detect_regime(pd.Timestamp("2024-01-01"))
        assert regime == MarketRegime.ELEVATED_VOL

    def test_detect_regime_high_vol(self, detector, macro_loader):
        """Test regime detection for high volatility."""
        macro_loader.get_macro_snapshot.return_value = {
            "vix": 35.0,
            "yield_curve_10y2y": 0.5,
            "hy_spread": 400.0,
        }

        regime = detector.detect_regime(pd.Timestamp("2024-01-01"))
        assert regime == MarketRegime.HIGH_VOL

    def test_detect_regime_recession_warning(self, detector, macro_loader):
        """Test regime detection for recession warning (inverted curve)."""
        macro_loader.get_macro_snapshot.return_value = {
            "vix": 18.0,
            "yield_curve_10y2y": -0.2,  # Inverted
            "hy_spread": 350.0,
        }

        regime = detector.detect_regime(pd.Timestamp("2024-01-01"))
        assert regime == MarketRegime.RECESSION_WARNING

    def test_detect_regime_missing_data(self, detector, macro_loader):
        """Test regime detection with missing macro data."""
        macro_loader.get_macro_snapshot.side_effect = Exception("Data not available")

        regime = detector.detect_regime(pd.Timestamp("2024-01-01"))
        assert regime == MarketRegime.UNKNOWN

    def test_classify_regime_with_all_data(self):
        """Test static regime classification method."""
        regime = RegimeDetector._classify_regime(
            vix=18.0,
            yield_spread=0.5,
            hy_spread=250.0,
        )
        assert regime == MarketRegime.RISK_ON

    def test_classify_regime_with_missing_vix(self):
        """Test regime classification with missing VIX."""
        regime = RegimeDetector._classify_regime(
            vix=None,
            yield_spread=0.5,
            hy_spread=250.0,
        )
        assert regime == MarketRegime.UNKNOWN

    def test_classify_regime_with_missing_yield(self):
        """Test regime classification with missing yield spread."""
        regime = RegimeDetector._classify_regime(
            vix=18.0,
            yield_spread=None,
            hy_spread=250.0,
        )
        assert regime == MarketRegime.UNKNOWN


class TestRegimeAwareAlpha:
    """Test suite for RegimeAwareAlpha class."""

    @pytest.fixture
    def mock_macro_loader(self):
        """Create mock macro loader."""
        loader = Mock(spec=MacroDataLoader)
        loader.get_macro_snapshot.return_value = {
            "vix": 16.0,
            "yield_curve_10y2y": 0.5,
            "hy_spread": 250.0,
        }
        return loader

    @pytest.fixture
    def mock_selector(self):
        """Create mock selector."""
        selector = Mock(spec=AlphaSelector)
        return selector

    @pytest.fixture
    def alpha_models(self):
        """Create test alpha models."""
        return {
            "momentum": DummyAlpha("momentum", {"AAPL": 0.8, "MSFT": 0.6}),
            "value": DummyAlpha("value", {"AAPL": 0.4, "MSFT": 0.7}),
        }

    @pytest.fixture
    def regime_aware_alpha(self, mock_selector, alpha_models, mock_macro_loader):
        """Create RegimeAwareAlpha instance."""
        return RegimeAwareAlpha(
            selector=mock_selector,
            models=alpha_models,
            macro_loader=mock_macro_loader,
        )

    def test_init_valid(self, mock_selector, alpha_models, mock_macro_loader):
        """Test initialization with valid inputs."""
        raa = RegimeAwareAlpha(
            selector=mock_selector,
            models=alpha_models,
            macro_loader=mock_macro_loader,
            name="TestAlpha",
        )
        assert raa.selector is mock_selector
        assert raa.models is alpha_models
        assert raa.macro_loader is mock_macro_loader
        assert raa.name == "TestAlpha"

    def test_init_empty_models(self, mock_selector, mock_macro_loader):
        """Test that init raises error with empty models dict."""
        with pytest.raises(ValueError, match="Must provide at least one"):
            RegimeAwareAlpha(
                selector=mock_selector,
                models={},
                macro_loader=mock_macro_loader,
            )

    def test_name_property(self, regime_aware_alpha):
        """Test name property."""
        assert regime_aware_alpha.name == "RegimeAwareAlpha"

    def test_score_single_model(
        self, regime_aware_alpha, mock_selector, alpha_models
    ):
        """Test scoring with single model selection."""
        # Setup selector to return single model
        selection = AlphaSelection(
            model=alpha_models["momentum"],
            model_weights=None,
        )
        mock_selector.select.return_value = selection

        # Score universe
        universe = Mock(spec=Universe)
        universe.tickers = ["AAPL", "MSFT", "GOOG"]
        as_of = pd.Timestamp("2024-01-01")
        data_store = Mock()

        scores = regime_aware_alpha.score(
            as_of=as_of,
            universe=universe,
            features=None,
            store=data_store,
        )

        assert isinstance(scores, pd.Series)
        assert scores["AAPL"] == 0.8
        assert scores["MSFT"] == 0.6
        assert scores["GOOG"] == 0.5

    def test_score_ensemble_models(
        self, regime_aware_alpha, mock_selector, alpha_models
    ):
        """Test scoring with ensemble model selection."""
        # Setup selector to return ensemble
        selection = AlphaSelection(
            model=None,
            model_weights={
                "momentum": 0.6,
                "value": 0.4,
            },
        )
        mock_selector.select.return_value = selection

        # Score universe
        universe = Mock(spec=Universe)
        universe.tickers = ["AAPL", "MSFT", "GOOG"]
        as_of = pd.Timestamp("2024-01-01")
        data_store = Mock()

        scores = regime_aware_alpha.score(
            as_of=as_of,
            universe=universe,
            features=None,
            store=data_store,
        )

        assert isinstance(scores, pd.Series)
        # AAPL: momentum(0.8)*0.6 + value(0.4)*0.4 = 0.48 + 0.16 = 0.64
        assert abs(scores["AAPL"] - 0.64) < 0.01
        # MSFT: momentum(0.6)*0.6 + value(0.7)*0.4 = 0.36 + 0.28 = 0.64
        assert abs(scores["MSFT"] - 0.64) < 0.01

    def test_score_tracks_history(self, regime_aware_alpha, mock_selector, alpha_models):
        """Test that scoring records regime and selection history."""
        selection = AlphaSelection(
            model=alpha_models["momentum"],
            model_weights=None,
        )
        mock_selector.select.return_value = selection

        # Score multiple times
        universe = Mock(spec=Universe)
        universe.tickers = ["AAPL"]
        data_store = Mock()

        for i in range(3):
            as_of = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
            regime_aware_alpha.score(
                as_of=as_of,
                universe=universe,
                features=None,
                store=data_store,
            )

        # Check history was recorded
        history = regime_aware_alpha.get_regime_history()
        assert len(history) == 3
        assert list(history.columns) == ["date", "regime", "selection"]

    def test_score_error_handling(self, regime_aware_alpha, mock_selector):
        """Test error handling during scoring."""
        mock_selector.select.side_effect = Exception("Selection failed")

        universe = Mock(spec=Universe)
        universe.tickers = ["AAPL"]
        as_of = pd.Timestamp("2024-01-01")
        data_store = Mock()

        with pytest.raises(RuntimeError, match="RegimeAwareAlpha scoring failed"):
            regime_aware_alpha.score(
                as_of=as_of,
                universe=universe,
                features=None,
                store=data_store,
            )

    def test_get_regime_history_empty(self, regime_aware_alpha):
        """Test regime history when nothing has been scored."""
        history = regime_aware_alpha.get_regime_history()
        assert history.empty

    def test_get_regime_history_populated(
        self, regime_aware_alpha, mock_selector, alpha_models
    ):
        """Test regime history after scoring."""
        selection = AlphaSelection(
            model=alpha_models["momentum"],
            model_weights=None,
        )
        mock_selector.select.return_value = selection

        universe = Mock(spec=Universe)
        universe.tickers = ["AAPL"]
        data_store = Mock()

        # Score on two dates
        regime_aware_alpha.score(
            as_of=pd.Timestamp("2024-01-01"),
            universe=universe,
            features=None,
            store=data_store,
        )
        regime_aware_alpha.score(
            as_of=pd.Timestamp("2024-01-02"),
            universe=universe,
            features=None,
            store=data_store,
        )

        history = regime_aware_alpha.get_regime_history()
        assert len(history) == 2
        assert "risk_on" in history["regime"].values  # Default regime

    def test_get_selection_stats(
        self, regime_aware_alpha, mock_selector, alpha_models
    ):
        """Test selection statistics."""
        selection = AlphaSelection(
            model=alpha_models["momentum"],
            model_weights=None,
        )
        mock_selector.select.return_value = selection

        universe = Mock(spec=Universe)
        universe.tickers = ["AAPL"]
        data_store = Mock()

        # Score multiple times
        for i in range(3):
            as_of = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
            regime_aware_alpha.score(
                as_of=as_of,
                universe=universe,
                features=None,
                store=data_store,
            )

        stats = regime_aware_alpha.get_selection_stats()
        assert "regime_risk_on" in stats
        assert stats["regime_risk_on"] == 3

    def test_describe_selection_single_model(self, regime_aware_alpha, alpha_models):
        """Test description generation for single model selection."""
        selection = AlphaSelection(
            model=alpha_models["momentum"],
            model_weights=None,
        )

        desc = regime_aware_alpha._describe_selection(selection)
        assert desc == "model=momentum"

    def test_describe_selection_ensemble(self, regime_aware_alpha):
        """Test description generation for ensemble selection."""
        selection = AlphaSelection(
            model=None,
            model_weights={
                "momentum": 0.6,
                "value": 0.4,
            },
        )

        desc = regime_aware_alpha._describe_selection(selection)
        assert "ensemble=" in desc
        assert "momentum" in desc
        assert "value" in desc


class TestRegimeAwareAlphaIntegration:
    """Integration tests with real selectors."""

    def test_with_regime_based_selector(self):
        """Test RegimeAwareAlpha with RegimeBasedSelector."""
        # Create config: momentum for risk-on, value for defensive
        config = {
            MarketRegime.RISK_ON: "momentum",
            MarketRegime.ELEVATED_VOL: "value",
            MarketRegime.HIGH_VOL: "value",
            MarketRegime.RECESSION_WARNING: "value",
            MarketRegime.UNKNOWN: "momentum",
        }

        models = {
            "momentum": DummyAlpha("momentum", {"AAPL": 0.8}),
            "value": DummyAlpha("value", {"AAPL": 0.4}),
        }

        selector = RegimeBasedSelector(config)

        macro_loader = Mock(spec=MacroDataLoader)
        macro_loader.get_macro_snapshot.return_value = {
            "vix": 16.0,
            "yield_curve_10y2y": 0.5,
            "hy_spread": 250.0,
        }

        raa = RegimeAwareAlpha(
            selector=selector,
            models=models,
            macro_loader=macro_loader,
        )

        # Should use momentum model (risk-on regime)
        scores = raa.score(
            as_of=pd.Timestamp("2024-01-01"),
            universe=Mock(spec=Universe, tickers=["AAPL"]),
            features=None,
            store=Mock(),
        )
        assert scores["AAPL"] == 0.8

    def test_with_real_regime_detector(self):
        """Test RegimeAwareAlpha with real RegimeDetector."""
        models = {
            "momentum": DummyAlpha("momentum", {"AAPL": 0.8}),
            "value": DummyAlpha("value", {"AAPL": 0.4}),
        }

        # RegimeBasedSelector expects MarketRegime enum keys, not strings
        config = {
            MarketRegime.RISK_ON: "momentum",
            MarketRegime.ELEVATED_VOL: "value",
            MarketRegime.HIGH_VOL: "value",
            MarketRegime.RECESSION_WARNING: "value",
            MarketRegime.UNKNOWN: "momentum",
        }

        selector = RegimeBasedSelector(config)

        macro_loader = Mock(spec=MacroDataLoader)
        macro_loader.get_macro_snapshot.return_value = {
            "vix": 25.0,  # Elevated volatility
            "yield_curve_10y2y": 0.5,
            "hy_spread": 300.0,
        }

        detector = RegimeDetector(macro_loader)

        raa = RegimeAwareAlpha(
            selector=selector,
            models=models,
            macro_loader=macro_loader,
            regime_detector=detector,
        )

        # Should detect elevated volatility and use value model
        scores = raa.score(
            as_of=pd.Timestamp("2024-01-01"),
            universe=Mock(spec=Universe, tickers=["AAPL"]),
            features=None,
            store=Mock(),
        )
        assert abs(scores["AAPL"] - 0.4) < 0.01
