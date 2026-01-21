# Task Handoff: IMPL-016 - Alpha Selector Framework

**Task ID:** IMPL-016
**Status:** ready
**Priority:** critical
**Estimated Effort:** 6-8 hours
**Dependencies:** None (can run parallel with IMPL-015)

---

## Quick Context

You are implementing the **AlphaSelector** abstraction - the core mechanism for dynamically selecting or weighting alpha models based on detected market regime.

From the README:
> **Model Selection**: Select the most appropriate alpha model or model weights for the detected regime

Currently, the system has:
- 7 alpha models in `src/quantetf/alpha/`
- `WeightedEnsemble` that uses fixed weights
- `RegimeMonitor` that detects regimes but doesn't influence model selection

**What's missing:** A way to dynamically choose WHICH model to use (or how to weight models) based on the current market regime.

---

## What You Need to Know

### Architecture Goal

```
RegimeDetector → AlphaSelector → AlphaModel(s) → Scores
       ↓              ↓
   "HIGH_VOL"    "Use VolAdjustedMomentum"
       ↓              ↓
   "RISK_ON"     "Use MomentumAcceleration"
```

### Key Design Decisions

1. **AlphaSelector is an interface** - Multiple implementations possible
2. **Selector receives regime state** - From RegimeMonitor or RegimeDetector
3. **Selector returns model OR weights** - Can switch models or blend them
4. **Selector is stateless** - Decision based only on current regime

### Existing Components to Integrate

- `src/quantetf/alpha/base.py` - AlphaModel base class
- `src/quantetf/alpha/factory.py` - AlphaModelRegistry
- `src/quantetf/alpha/ensemble.py` - WeightedEnsemble
- `src/quantetf/monitoring/regime.py` - RegimeMonitor with MarketRegime enum

---

## Files to Read First

1. **`/workspaces/qetf/CLAUDE_CONTEXT.md`** - Coding standards
2. **`/workspaces/qetf/README.md`** - Regime-aware alpha generation description
3. **`/workspaces/qetf/src/quantetf/alpha/base.py`** - AlphaModel interface
4. **`/workspaces/qetf/src/quantetf/alpha/ensemble.py`** - WeightedEnsemble
5. **`/workspaces/qetf/src/quantetf/alpha/factory.py`** - AlphaModelRegistry
6. **`/workspaces/qetf/src/quantetf/monitoring/regime.py`** - MarketRegime enum

---

## Implementation Steps

### 1. Create AlphaSelector base class

Create `src/quantetf/alpha/selector.py`:

```python
"""Alpha model selection based on market regime."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
import pandas as pd

from quantetf.alpha.base import AlphaModel


class MarketRegime(Enum):
    """Market regime classification.

    Aligned with RegimeMonitor but can be extended.
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

    Either a single model or weighted ensemble configuration.
    """
    model: Optional[AlphaModel] = None
    model_weights: Optional[Dict[str, float]] = None  # model_name -> weight
    regime: MarketRegime = MarketRegime.UNKNOWN
    confidence: float = 1.0  # Confidence in regime detection

    @property
    def is_single_model(self) -> bool:
        return self.model is not None

    @property
    def is_ensemble(self) -> bool:
        return self.model_weights is not None


class AlphaSelector(ABC):
    """Base class for dynamic alpha model selection.

    AlphaSelector chooses which alpha model(s) to use based on
    the detected market regime.
    """

    @abstractmethod
    def select(
        self,
        regime: MarketRegime,
        as_of: pd.Timestamp,
        available_models: Dict[str, AlphaModel],
    ) -> AlphaSelection:
        """
        Select alpha model(s) for the given regime.

        Args:
            regime: Detected market regime
            as_of: Current date (for any date-dependent logic)
            available_models: Dict of model_name -> AlphaModel instances

        Returns:
            AlphaSelection with either single model or ensemble weights
        """
        pass

    @abstractmethod
    def get_supported_regimes(self) -> List[MarketRegime]:
        """Return list of regimes this selector handles."""
        pass
```

### 2. Implement RegimeBasedSelector

```python
class RegimeBasedSelector(AlphaSelector):
    """Selects alpha model based on regime-to-model mapping.

    Simple rule-based selection: each regime maps to a specific model.
    """

    def __init__(
        self,
        regime_model_map: Dict[MarketRegime, str],
        default_model: str = "momentum",
    ):
        """
        Initialize with regime-to-model mapping.

        Args:
            regime_model_map: Maps regime to model name
            default_model: Fallback model for unmapped regimes

        Example:
            RegimeBasedSelector({
                MarketRegime.RISK_ON: "momentum_acceleration",
                MarketRegime.HIGH_VOL: "vol_adjusted_momentum",
                MarketRegime.RECESSION_WARNING: "dual_momentum",
            })
        """
        self.regime_model_map = regime_model_map
        self.default_model = default_model

    def select(
        self,
        regime: MarketRegime,
        as_of: pd.Timestamp,
        available_models: Dict[str, AlphaModel],
    ) -> AlphaSelection:
        model_name = self.regime_model_map.get(regime, self.default_model)

        if model_name not in available_models:
            raise ValueError(
                f"Model '{model_name}' for regime {regime} not in available models: "
                f"{list(available_models.keys())}"
            )

        return AlphaSelection(
            model=available_models[model_name],
            regime=regime,
            confidence=1.0,
        )

    def get_supported_regimes(self) -> List[MarketRegime]:
        return list(self.regime_model_map.keys())
```

### 3. Implement RegimeWeightedEnsemble

```python
class RegimeWeightedSelector(AlphaSelector):
    """Adjusts ensemble weights based on regime.

    Uses different model weight configurations for different regimes.
    """

    def __init__(
        self,
        regime_weights: Dict[MarketRegime, Dict[str, float]],
        default_weights: Dict[str, float],
    ):
        """
        Initialize with regime-specific weight configurations.

        Args:
            regime_weights: Maps regime to model weights
            default_weights: Fallback weights for unmapped regimes

        Example:
            RegimeWeightedSelector(
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
        self.regime_weights = regime_weights
        self.default_weights = default_weights

    def select(
        self,
        regime: MarketRegime,
        as_of: pd.Timestamp,
        available_models: Dict[str, AlphaModel],
    ) -> AlphaSelection:
        weights = self.regime_weights.get(regime, self.default_weights)

        # Validate all models exist
        missing = set(weights.keys()) - set(available_models.keys())
        if missing:
            raise ValueError(f"Models not available: {missing}")

        # Normalize weights
        total = sum(weights.values())
        normalized_weights = {k: v / total for k, v in weights.items()}

        return AlphaSelection(
            model_weights=normalized_weights,
            regime=regime,
            confidence=1.0,
        )

    def get_supported_regimes(self) -> List[MarketRegime]:
        return list(self.regime_weights.keys())
```

### 4. Create ConfigurableSelector (YAML-driven)

```python
from quantetf.alpha.factory import AlphaModelRegistry


class ConfigurableSelector(AlphaSelector):
    """Alpha selector configured from YAML.

    Allows regime-model mappings to be defined in config files.
    """

    def __init__(self, config: dict, registry: AlphaModelRegistry):
        """
        Initialize from config dict.

        Expected config structure:
        ```yaml
        alpha_selector:
          type: regime_based  # or regime_weighted
          default_model: momentum
          regime_mapping:
            risk_on: momentum_acceleration
            high_vol: vol_adjusted_momentum
            recession_warning: dual_momentum
        ```
        """
        self.config = config
        self.registry = registry
        self._build_selector()

    def _build_selector(self):
        selector_type = self.config.get("type", "regime_based")
        regime_mapping = self.config.get("regime_mapping", {})

        if selector_type == "regime_based":
            # Convert string keys to MarketRegime
            mapping = {
                MarketRegime(k): v
                for k, v in regime_mapping.items()
            }
            self._inner = RegimeBasedSelector(
                regime_model_map=mapping,
                default_model=self.config.get("default_model", "momentum"),
            )
        elif selector_type == "regime_weighted":
            weights = {
                MarketRegime(k): v
                for k, v in regime_mapping.items()
            }
            self._inner = RegimeWeightedSelector(
                regime_weights=weights,
                default_weights=self.config.get("default_weights", {"momentum": 1.0}),
            )
        else:
            raise ValueError(f"Unknown selector type: {selector_type}")

    def select(
        self,
        regime: MarketRegime,
        as_of: pd.Timestamp,
        available_models: Dict[str, AlphaModel],
    ) -> AlphaSelection:
        return self._inner.select(regime, as_of, available_models)

    def get_supported_regimes(self) -> List[MarketRegime]:
        return self._inner.get_supported_regimes()
```

### 5. Create helper function to execute selection

```python
def compute_alpha_with_selection(
    selector: AlphaSelector,
    regime: MarketRegime,
    as_of: pd.Timestamp,
    available_models: Dict[str, AlphaModel],
    universe: List[str],
    data_store: "DataStore",
) -> pd.Series:
    """
    Compute alpha scores using selected model(s).

    Handles both single model and ensemble cases.

    Args:
        selector: AlphaSelector instance
        regime: Current market regime
        as_of: Scoring date
        available_models: Available alpha models
        universe: List of tickers to score
        data_store: Data store for price data

    Returns:
        Series of alpha scores indexed by ticker
    """
    selection = selector.select(regime, as_of, available_models)

    if selection.is_single_model:
        return selection.model.score(universe, as_of, data_store)

    elif selection.is_ensemble:
        # Weighted average of model scores
        scores = pd.DataFrame()
        for model_name, weight in selection.model_weights.items():
            model = available_models[model_name]
            model_scores = model.score(universe, as_of, data_store)
            scores[model_name] = model_scores * weight

        return scores.sum(axis=1)

    else:
        raise ValueError("AlphaSelection must have model or model_weights")
```

### 6. Write tests

Create `tests/alpha/test_selector.py`:

```python
import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock

from quantetf.alpha.selector import (
    MarketRegime,
    AlphaSelection,
    RegimeBasedSelector,
    RegimeWeightedSelector,
    compute_alpha_with_selection,
)


class TestRegimeBasedSelector:
    """Tests for regime-based model selection."""

    def test_selects_correct_model_for_regime(self):
        """Test that correct model is selected for each regime."""
        mock_momentum = Mock()
        mock_vol_adj = Mock()

        selector = RegimeBasedSelector(
            regime_model_map={
                MarketRegime.RISK_ON: "momentum",
                MarketRegime.HIGH_VOL: "vol_adjusted",
            },
            default_model="momentum",
        )

        available = {"momentum": mock_momentum, "vol_adjusted": mock_vol_adj}

        # Test RISK_ON -> momentum
        selection = selector.select(
            MarketRegime.RISK_ON,
            pd.Timestamp("2024-01-01"),
            available,
        )
        assert selection.model == mock_momentum
        assert selection.regime == MarketRegime.RISK_ON

        # Test HIGH_VOL -> vol_adjusted
        selection = selector.select(
            MarketRegime.HIGH_VOL,
            pd.Timestamp("2024-01-01"),
            available,
        )
        assert selection.model == mock_vol_adj

    def test_uses_default_for_unmapped_regime(self):
        """Test fallback to default model."""
        mock_default = Mock()

        selector = RegimeBasedSelector(
            regime_model_map={MarketRegime.RISK_ON: "special"},
            default_model="default",
        )

        selection = selector.select(
            MarketRegime.UNKNOWN,
            pd.Timestamp("2024-01-01"),
            {"default": mock_default, "special": Mock()},
        )

        assert selection.model == mock_default

    def test_raises_for_missing_model(self):
        """Test error when mapped model doesn't exist."""
        selector = RegimeBasedSelector(
            regime_model_map={MarketRegime.RISK_ON: "nonexistent"},
            default_model="momentum",
        )

        with pytest.raises(ValueError, match="not in available models"):
            selector.select(
                MarketRegime.RISK_ON,
                pd.Timestamp("2024-01-01"),
                {"momentum": Mock()},
            )


class TestRegimeWeightedSelector:
    """Tests for regime-weighted ensemble selection."""

    def test_returns_correct_weights_for_regime(self):
        """Test weight configuration per regime."""
        selector = RegimeWeightedSelector(
            regime_weights={
                MarketRegime.RISK_ON: {"a": 0.6, "b": 0.4},
                MarketRegime.HIGH_VOL: {"a": 0.3, "b": 0.7},
            },
            default_weights={"a": 0.5, "b": 0.5},
        )

        selection = selector.select(
            MarketRegime.RISK_ON,
            pd.Timestamp("2024-01-01"),
            {"a": Mock(), "b": Mock()},
        )

        assert selection.is_ensemble
        assert selection.model_weights["a"] == pytest.approx(0.6)
        assert selection.model_weights["b"] == pytest.approx(0.4)

    def test_normalizes_weights(self):
        """Test that weights are normalized to sum to 1."""
        selector = RegimeWeightedSelector(
            regime_weights={MarketRegime.RISK_ON: {"a": 2, "b": 2}},
            default_weights={"a": 1},
        )

        selection = selector.select(
            MarketRegime.RISK_ON,
            pd.Timestamp("2024-01-01"),
            {"a": Mock(), "b": Mock()},
        )

        assert sum(selection.model_weights.values()) == pytest.approx(1.0)


class TestComputeAlphaWithSelection:
    """Tests for alpha computation with selection."""

    def test_single_model_scoring(self):
        """Test scoring with single selected model."""
        mock_model = Mock()
        mock_model.score.return_value = pd.Series({"SPY": 0.5, "QQQ": 0.3})

        selector = RegimeBasedSelector(
            regime_model_map={MarketRegime.RISK_ON: "test"},
            default_model="test",
        )

        scores = compute_alpha_with_selection(
            selector=selector,
            regime=MarketRegime.RISK_ON,
            as_of=pd.Timestamp("2024-01-01"),
            available_models={"test": mock_model},
            universe=["SPY", "QQQ"],
            data_store=Mock(),
        )

        mock_model.score.assert_called_once()
        assert "SPY" in scores.index

    def test_ensemble_scoring(self):
        """Test weighted ensemble scoring."""
        mock_model_a = Mock()
        mock_model_a.score.return_value = pd.Series({"SPY": 1.0, "QQQ": 0.0})

        mock_model_b = Mock()
        mock_model_b.score.return_value = pd.Series({"SPY": 0.0, "QQQ": 1.0})

        selector = RegimeWeightedSelector(
            regime_weights={MarketRegime.RISK_ON: {"a": 0.5, "b": 0.5}},
            default_weights={"a": 1.0},
        )

        scores = compute_alpha_with_selection(
            selector=selector,
            regime=MarketRegime.RISK_ON,
            as_of=pd.Timestamp("2024-01-01"),
            available_models={"a": mock_model_a, "b": mock_model_b},
            universe=["SPY", "QQQ"],
            data_store=Mock(),
        )

        # 50% weight on each model
        assert scores["SPY"] == pytest.approx(0.5)
        assert scores["QQQ"] == pytest.approx(0.5)
```

---

## Acceptance Criteria

- [ ] `AlphaSelector` abstract base class defined
- [ ] `AlphaSelection` dataclass with model/weights
- [ ] `RegimeBasedSelector` implemented (single model per regime)
- [ ] `RegimeWeightedSelector` implemented (ensemble weights per regime)
- [ ] `ConfigurableSelector` loads from YAML config
- [ ] `compute_alpha_with_selection()` helper function works
- [ ] Unit tests pass for all selectors
- [ ] Integration test with real alpha models
- [ ] MarketRegime enum aligned with existing RegimeMonitor
- [ ] All code has type hints and docstrings

---

## Example Config Structure

Create example config `configs/selectors/regime_based_example.yaml`:

```yaml
alpha_selector:
  type: regime_based
  default_model: momentum

  regime_mapping:
    risk_on: momentum_acceleration
    elevated_vol: vol_adjusted_momentum
    high_vol: dual_momentum
    recession_warning: trend_filtered_momentum
```

And `configs/selectors/regime_weighted_example.yaml`:

```yaml
alpha_selector:
  type: regime_weighted
  default_weights:
    momentum: 0.5
    vol_adjusted_momentum: 0.5

  regime_mapping:
    risk_on:
      momentum: 0.4
      momentum_acceleration: 0.4
      value_momentum: 0.2
    high_vol:
      vol_adjusted_momentum: 0.6
      dual_momentum: 0.4
    recession_warning:
      dual_momentum: 0.5
      trend_filtered_momentum: 0.5
```

---

## Definition of Done

1. All acceptance criteria met
2. `pytest tests/alpha/test_selector.py` passes
3. Example configs created in `configs/selectors/`
4. PROGRESS_LOG.md updated
5. Completion note created: `handoffs/completion-IMPL-016.md`
6. TASKS.md status updated to `completed`
7. Code committed with clear message

---

## Notes

- This is the core abstraction for regime-aware investing
- Keep selectors simple and stateless
- The actual regime detection is handled elsewhere (RegimeMonitor/RegimeDetector)
- Consider adding logging to track which model is selected per rebalance
- Future enhancement: ML-based selector that learns optimal mapping
