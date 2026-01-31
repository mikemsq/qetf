"""Regime-aware alpha model integration (IMPL-018).

This module integrates regime detection with alpha model selection,
creating a unified interface that adapts to market conditions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING
import pandas as pd
import logging

from quantetf.alpha.base import AlphaModel
from quantetf.alpha.selector import AlphaSelector, AlphaSelection, MarketRegime
from quantetf.data.macro_loader import MacroDataLoader
from quantetf.types import AlphaScores, FeatureFrame, Universe, DatasetVersion

if TYPE_CHECKING:
    from quantetf.data.store import DataStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class RegimeDetector:
    """Unified interface for detecting market regimes.
    
    Combines multiple macro signals to classify current market regime.
    """

    def __init__(self, macro_loader: MacroDataLoader):
        """Initialize regime detector with macro data source.
        
        Args:
            macro_loader: MacroDataLoader instance for accessing macro data
        """
        self.macro = macro_loader

    def detect_regime(
        self,
        as_of: pd.Timestamp,
    ) -> MarketRegime:
        """Detect market regime as-of a given date.

        Uses multiple macro signals to classify regime:
        - VIX (volatility level)
        - Yield curve (economic health)
        - Credit spreads (risk appetite)

        Args:
            as_of: Point-in-time date for regime detection

        Returns:
            MarketRegime classification

        Note:
            Returns UNKNOWN if unable to determine regime.
        """
        try:
            # Get macro snapshot
            snapshot = self.macro.get_macro_snapshot(as_of)

            # Extract key signals
            vix = snapshot.get("vix")
            yield_spread = snapshot.get("yield_curve_10y2y")
            hy_spread = snapshot.get("hy_spread")

            # Classify regime based on signals
            return self._classify_regime(vix, yield_spread, hy_spread)

        except Exception as e:
            logger.warning(f"Failed to detect regime as-of {as_of}: {e}")
            return MarketRegime.UNKNOWN

    @staticmethod
    def _classify_regime(
        vix: Optional[float],
        yield_spread: Optional[float],
        hy_spread: Optional[float],
    ) -> MarketRegime:
        """Classify regime based on macro signals.

        Args:
            vix: VIX level (volatility)
            yield_spread: 10Y-2Y spread (recession signal)
            hy_spread: High yield credit spread (risk appetite)

        Returns:
            MarketRegime classification
        """
        # Handle missing data
        if vix is None or yield_spread is None:
            return MarketRegime.UNKNOWN

        # Recession warning: inverted yield curve
        if yield_spread < 0:
            return MarketRegime.RECESSION_WARNING

        # High volatility regimes
        if vix > 30:
            return MarketRegime.HIGH_VOL
        elif vix > 20:
            return MarketRegime.ELEVATED_VOL

        # Risk-on regime
        return MarketRegime.RISK_ON


class RegimeAwareAlpha(AlphaModel):
    """Alpha model that dynamically selects models based on market regime.

    This is the main integration point between regime detection and alpha
    selection. It looks like a regular AlphaModel to the backtest engine,
    but internally uses regime detection to choose the appropriate model(s).

    The system flow:
    1. Detect market regime using macro data
    2. Select alpha model(s) appropriate for that regime
    3. Score universe using selected model(s)
    4. Track selection history for analysis
    """

    def __init__(
        self,
        selector: AlphaSelector,
        models: Dict[str, AlphaModel],
        macro_loader: MacroDataLoader,
        regime_detector: Optional[RegimeDetector] = None,
        name: str = "RegimeAwareAlpha",
    ):
        """Initialize regime-aware alpha model.

        Args:
            selector: AlphaSelector for choosing model based on regime
            models: Dict of model_name -> AlphaModel instances
            macro_loader: For accessing macro data and regime detection
            regime_detector: Optional custom RegimeDetector (creates default if None)
            name: Name of this alpha model

        Raises:
            ValueError: If models dict is empty
        """
        if not models:
            raise ValueError("Must provide at least one alpha model")

        self.selector = selector
        self.models = models
        self.macro_loader = macro_loader
        self.regime_detector = regime_detector or RegimeDetector(macro_loader)
        self._name = name

        # Track regime and selection history for analysis
        self._history: List[tuple] = []

    def score(
        self,
        *,
        as_of: pd.Timestamp,
        universe: Universe,
        features: FeatureFrame,
        store: DataStore,
        dataset_version: Optional[DatasetVersion] = None,
    ) -> AlphaScores:
        """Score universe using regime-appropriate model(s).

        Args:
            as_of: Scoring date (T-1, not including as_of date)
            universe: Universe of tickers to score
            features: Feature frame (not used for regime-aware alpha)
            store: Data store for accessing historical prices
            dataset_version: Optional dataset version (passed to models)

        Returns:
            Series of alpha scores indexed by ticker

        Raises:
            RuntimeError: If scoring fails
        """
        try:
            # 1. Detect current regime
            regime = self.regime_detector.detect_regime(as_of)
            logger.debug(f"Detected regime {regime.value} as-of {as_of.date()}")

            # 2. Select model(s) for this regime
            selection = self.selector.select(regime, as_of, self.models)
            logger.debug(f"Selected: {self._describe_selection(selection)}")

            # 3. Record for analysis
            self._history.append((as_of, regime, selection))

            # 4. Compute scores
            if selection.is_single_model:
                # Simple case: single model
                scores = selection.model.score(
                    as_of=as_of,
                    universe=universe,
                    features=features,
                    store=store,
                    dataset_version=dataset_version,
                )

            else:
                # Ensemble case: weighted average scores
                scores_dict = {}
                for model_name, weight in selection.model_weights.items():
                    model = self.models[model_name]
                    model_scores = model.score(
                        as_of=as_of,
                        universe=universe,
                        features=features,
                        store=store,
                        dataset_version=dataset_version,
                    )
                    scores_dict[model_name] = model_scores * weight

                # Sum weighted scores
                scores_df = pd.DataFrame(scores_dict)
                scores = scores_df.sum(axis=1)

            return scores

        except Exception as e:
            logger.error(f"Failed to score universe as-of {as_of}: {e}")
            raise RuntimeError(f"RegimeAwareAlpha scoring failed: {e}") from e

    def get_regime_history(self) -> pd.DataFrame:
        """Get regime detection and model selection history.

        Returns:
            DataFrame with columns: date, regime, selection
        """
        if not self._history:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "date": as_of,
                "regime": regime.value,
                "selection": self._describe_selection(selection),
            }
            for as_of, regime, selection in self._history
        ])

    def get_selection_stats(self) -> Dict[str, int]:
        """Get statistics on model selection history.

        Returns:
            Dictionary with regime and model selection counts
        """
        if not self._history:
            return {}

        stats = {}
        for _, regime, _ in self._history:
            regime_key = f"regime_{regime.value}"
            stats[regime_key] = stats.get(regime_key, 0) + 1

        return stats

    @property
    def name(self) -> str:
        """Return model name."""
        return self._name

    def _describe_selection(self, selection: AlphaSelection) -> str:
        """Create human-readable description of selection.

        Args:
            selection: AlphaSelection result

        Returns:
            Human-readable string (e.g., "model=momentum" or "ensemble=[momentum:60%, accel:40%]")
        """
        if selection.is_single_model:
            # Find model name by matching object
            for name, model in self.models.items():
                if model is selection.model:
                    return f"model={name}"
            return "model=unknown"

        else:
            # Ensemble format
            weights_str = ", ".join(
                f"{k}:{v:.0%}" for k, v in selection.model_weights.items()
            )
            return f"ensemble=[{weights_str}]"
