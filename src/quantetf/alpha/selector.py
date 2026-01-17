"""Alpha model selection based on market regime.

This module provides the AlphaSelector abstraction for dynamically selecting
or weighting alpha models based on detected market regimes. Instead of using
a fixed alpha model, the system can now adapt model selection to market conditions.

Core Components:
    - AlphaSelector: Abstract base class for selection strategies
    - MarketRegime: Enum of market regime classifications
    - AlphaSelection: Result of model selection (single or ensemble)
    - RegimeBasedSelector: Simple rule-based single model selection
    - RegimeWeightedSelector: Regime-specific ensemble weighting
    - ConfigurableSelector: YAML-driven configuration
    - compute_alpha_with_selection: Helper to execute selected models

Architecture:
    RegimeDetector → AlphaSelector → Selected Model(s) → Alpha Scores
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quantetf.alpha.base import AlphaModel


class MarketRegime(Enum):
    """Market regime classification.

    Aligned with RegimeMonitor but can be extended for more specific classifications.
    """

    RISK_ON = "risk_on"
    ELEVATED_VOL = "elevated_vol"
    HIGH_VOL = "high_vol"
    RECESSION_WARNING = "recession_warning"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    UNKNOWN = "unknown"


@dataclass
class AlphaSelection:
    """Result of alpha model selection.

    Contains either a single selected model OR an ensemble weight configuration,
    along with metadata about the selection decision.

    Attributes:
        model: Single selected AlphaModel (mutually exclusive with model_weights)
        model_weights: Dict of model_name -> weight for ensemble (mutually exclusive with model)
        regime: The market regime for which selection was made
        confidence: Confidence level in the regime detection (0.0 to 1.0)
    """

    model: Optional[AlphaModel] = None
    model_weights: Optional[Dict[str, float]] = None  # model_name -> weight
    regime: MarketRegime = MarketRegime.UNKNOWN
    confidence: float = 1.0

    @property
    def is_single_model(self) -> bool:
        """True if selection contains a single model."""
        return self.model is not None

    @property
    def is_ensemble(self) -> bool:
        """True if selection contains ensemble weights."""
        return self.model_weights is not None

    def __post_init__(self):
        """Validate that exactly one of model or model_weights is set."""
        if self.is_single_model and self.is_ensemble:
            raise ValueError("Cannot specify both model and model_weights")
        if not self.is_single_model and not self.is_ensemble:
            raise ValueError("Must specify either model or model_weights")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


class AlphaSelector(ABC):
    """Base class for dynamic alpha model selection.

    AlphaSelector chooses which alpha model(s) to use based on the detected
    market regime. Different selector implementations can use different strategies:

    - RegimeBasedSelector: Maps each regime to a single model
    - RegimeWeightedSelector: Maps each regime to ensemble weights
    - Custom implementations: ML-based, volatility-adjusted, etc.

    Selectors are stateless - selection depends only on current regime.
    """

    @abstractmethod
    def select(
        self,
        regime: MarketRegime,
        as_of: pd.Timestamp,
        available_models: Dict[str, AlphaModel],
    ) -> AlphaSelection:
        """Select alpha model(s) for the given regime.

        Args:
            regime: Detected market regime
            as_of: Current date (for date-dependent logic if needed)
            available_models: Dict of model_name -> AlphaModel instances

        Returns:
            AlphaSelection with either single model or ensemble weights

        Raises:
            ValueError: If required models are not available
        """
        pass

    @abstractmethod
    def get_supported_regimes(self) -> List[MarketRegime]:
        """Return list of regimes this selector handles."""
        pass


class RegimeBasedSelector(AlphaSelector):
    """Selects a single alpha model based on regime-to-model mapping.

    Simple rule-based selection: each regime is mapped to a specific model.
    Unmapped regimes fall back to a default model.

    Example:
        selector = RegimeBasedSelector(
            regime_model_map={
                MarketRegime.RISK_ON: "momentum_acceleration",
                MarketRegime.HIGH_VOL: "vol_adjusted_momentum",
                MarketRegime.RECESSION_WARNING: "dual_momentum",
            },
            default_model="momentum"
        )
    """

    def __init__(
        self,
        regime_model_map: Dict[MarketRegime, str],
        default_model: str = "momentum",
    ):
        """Initialize with regime-to-model mapping.

        Args:
            regime_model_map: Maps MarketRegime to model name string
            default_model: Fallback model name for unmapped regimes
        """
        self.regime_model_map = regime_model_map
        self.default_model = default_model

    def select(
        self,
        regime: MarketRegime,
        as_of: pd.Timestamp,
        available_models: Dict[str, AlphaModel],
    ) -> AlphaSelection:
        """Select single model for regime."""
        model_name = self.regime_model_map.get(regime, self.default_model)

        if model_name not in available_models:
            raise ValueError(
                f"Model '{model_name}' for regime {regime.value} not in available "
                f"models: {list(available_models.keys())}"
            )

        return AlphaSelection(
            model=available_models[model_name],
            regime=regime,
            confidence=1.0,
        )

    def get_supported_regimes(self) -> List[MarketRegime]:
        """Return regimes with explicit mappings."""
        return list(self.regime_model_map.keys())


class RegimeWeightedSelector(AlphaSelector):
    """Adjusts ensemble weights based on market regime.

    Uses different model weight configurations for different regimes,
    allowing dynamic blending of multiple models based on market conditions.

    Example:
        selector = RegimeWeightedSelector(
            regime_weights={
                MarketRegime.RISK_ON: {
                    "momentum": 0.6,
                    "momentum_acceleration": 0.4,
                },
                MarketRegime.HIGH_VOL: {
                    "vol_adjusted_momentum": 0.7,
                    "dual_momentum": 0.3,
                },
            },
            default_weights={"momentum": 1.0},
        )
    """

    def __init__(
        self,
        regime_weights: Dict[MarketRegime, Dict[str, float]],
        default_weights: Dict[str, float],
    ):
        """Initialize with regime-specific weight configurations.

        Args:
            regime_weights: Maps MarketRegime to dict of model_name -> weight
            default_weights: Fallback weights for unmapped regimes
        """
        self.regime_weights = regime_weights
        self.default_weights = default_weights
        self._validate()

    def _validate(self):
        """Validate that weights are valid (positive, will be normalized)."""
        for regime, weights in self.regime_weights.items():
            for model, weight in weights.items():
                if weight < 0:
                    raise ValueError(
                        f"Regime {regime.value}, model {model}: weight must be >= 0, got {weight}"
                    )
        for model, weight in self.default_weights.items():
            if weight < 0:
                raise ValueError(
                    f"Default weights, model {model}: weight must be >= 0, got {weight}"
                )

    def select(
        self,
        regime: MarketRegime,
        as_of: pd.Timestamp,
        available_models: Dict[str, AlphaModel],
    ) -> AlphaSelection:
        """Select ensemble weights for regime."""
        weights = self.regime_weights.get(regime, self.default_weights)

        # Validate all models exist
        missing = set(weights.keys()) - set(available_models.keys())
        if missing:
            raise ValueError(
                f"For regime {regime.value}, models not available: {missing}. "
                f"Available: {list(available_models.keys())}"
            )

        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total == 0:
            raise ValueError(
                f"For regime {regime.value}, total weight is 0 (all weights are 0)"
            )
        normalized_weights = {k: v / total for k, v in weights.items()}

        return AlphaSelection(
            model_weights=normalized_weights,
            regime=regime,
            confidence=1.0,
        )

    def get_supported_regimes(self) -> List[MarketRegime]:
        """Return regimes with explicit weight configurations."""
        return list(self.regime_weights.keys())


class ConfigurableSelector(AlphaSelector):
    """Alpha selector configured from dict/YAML.

    Allows regime-model mappings and ensemble weights to be defined
    in configuration files for easy updates without code changes.

    Expected config structure:
        {
            "type": "regime_based",  # or "regime_weighted"
            "default_model": "momentum",
            "regime_mapping": {
                "risk_on": "momentum_acceleration",
                "high_vol": "vol_adjusted_momentum",
            }
        }

    Or for weighted:
        {
            "type": "regime_weighted",
            "default_weights": {"momentum": 1.0},
            "regime_mapping": {
                "risk_on": {"momentum": 0.6, "momentum_acceleration": 0.4},
                "high_vol": {"vol_adjusted_momentum": 0.7, "dual_momentum": 0.3},
            }
        }
    """

    def __init__(self, config: dict):
        """Initialize from configuration dict.

        Args:
            config: Configuration dict with type, default settings, and regime_mapping
        """
        self.config = config
        self._inner: Optional[AlphaSelector] = None
        self._build_selector()

    def _build_selector(self):
        """Build appropriate selector from config."""
        selector_type = self.config.get("type", "regime_based")
        regime_mapping = self.config.get("regime_mapping", {})

        if not regime_mapping:
            raise ValueError("Config must contain 'regime_mapping'")

        try:
            if selector_type == "regime_based":
                # Convert string keys to MarketRegime enum
                mapping = {
                    MarketRegime(k) if isinstance(k, str) else k: v
                    for k, v in regime_mapping.items()
                }
                self._inner = RegimeBasedSelector(
                    regime_model_map=mapping,
                    default_model=self.config.get("default_model", "momentum"),
                )
            elif selector_type == "regime_weighted":
                # Convert string keys to MarketRegime enum, keep value dicts as-is
                weights = {
                    MarketRegime(k) if isinstance(k, str) else k: v
                    for k, v in regime_mapping.items()
                }
                self._inner = RegimeWeightedSelector(
                    regime_weights=weights,
                    default_weights=self.config.get(
                        "default_weights", {"momentum": 1.0}
                    ),
                )
            else:
                raise ValueError(f"Unknown selector type: {selector_type}")
        except ValueError as e:
            if "is not a valid" in str(e):
                raise ValueError(f"Invalid regime name in config: {e}")
            raise

    def select(
        self,
        regime: MarketRegime,
        as_of: pd.Timestamp,
        available_models: Dict[str, AlphaModel],
    ) -> AlphaSelection:
        """Select using inner selector."""
        if self._inner is None:
            raise RuntimeError("Selector not initialized")
        return self._inner.select(regime, as_of, available_models)

    def get_supported_regimes(self) -> List[MarketRegime]:
        """Return supported regimes from inner selector."""
        if self._inner is None:
            raise RuntimeError("Selector not initialized")
        return self._inner.get_supported_regimes()


def compute_alpha_with_selection(
    selector: AlphaSelector,
    regime: MarketRegime,
    as_of: pd.Timestamp,
    available_models: Dict[str, AlphaModel],
    universe: List[str],
    data_store: "DataStore",
) -> pd.Series:
    """Compute alpha scores using selected model(s).

    Handles both single model and ensemble weighting cases transparently.

    Args:
        selector: AlphaSelector instance to make selection
        regime: Current market regime
        as_of: Scoring date (T-1, not including as_of date)
        available_models: Dict of model_name -> AlphaModel instances
        universe: List of tickers to score
        data_store: Data store for accessing historical prices

    Returns:
        Series of alpha scores indexed by ticker

    Raises:
        ValueError: If selection fails or models cannot be found
        RuntimeError: If scoring fails
    """
    # Get the selection
    selection = selector.select(regime, as_of, available_models)

    if selection.is_single_model:
        # Simple case: use single model
        return selection.model.score(universe, as_of, data_store)

    elif selection.is_ensemble:
        # Ensemble case: weight and average model scores
        scores_dict = {}
        for model_name, weight in selection.model_weights.items():
            model = available_models[model_name]
            try:
                model_scores = model.score(universe, as_of, data_store)
                scores_dict[model_name] = model_scores * weight
            except Exception as e:
                raise RuntimeError(
                    f"Failed to score with model '{model_name}': {e}"
                ) from e

        # Stack and sum weighted scores
        scores_df = pd.DataFrame(scores_dict)
        return scores_df.sum(axis=1)

    else:
        raise ValueError("AlphaSelection must contain either model or model_weights")
