# Task Handoff: IMPL-018 - Regime-Alpha Pipeline Integration

**Task ID:** IMPL-018
**Status:** blocked
**Priority:** high
**Estimated Effort:** 6-8 hours
**Dependencies:** IMPL-016 (Alpha Selector), IMPL-017 (Macro Data API)

---

## Quick Context

You are integrating the regime detection and alpha selection components into the main backtest and production pipelines.

Currently:
- `RegimeMonitor` detects regimes but doesn't influence alpha selection
- `AlphaSelector` (from IMPL-016) selects models but isn't connected to the pipeline
- Backtest engine uses a single fixed alpha model

**Goal:** Connect RegimeMonitor → AlphaSelector → BacktestEngine so that the system automatically uses the appropriate alpha model(s) for each market regime.

---

## What You Need to Know

### Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Backtest/Production Pipeline              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  For each rebalance date:                                    │
│                                                              │
│  1. MacroDataLoader.get_macro_snapshot(as_of)               │
│              ↓                                               │
│  2. RegimeDetector.detect_regime(macro_snapshot)            │
│              ↓                                               │
│  3. AlphaSelector.select(regime, available_models)          │
│              ↓                                               │
│  4. Selected Model(s).score(universe, as_of)                │
│              ↓                                               │
│  5. PortfolioConstructor.construct(scores, constraints)     │
│              ↓                                               │
│  6. Execute trades / record weights                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Components to Integrate

From existing code:
- `src/quantetf/monitoring/regime.py` - `RegimeMonitor` class
- `src/quantetf/backtest/simple_engine.py` - `SimpleBacktestEngine`
- `src/quantetf/production/pipeline.py` - `ProductionPipeline`

From IMPL-016/017:
- `src/quantetf/alpha/selector.py` - `AlphaSelector`, `RegimeBasedSelector`
- `src/quantetf/data/macro_loader.py` - Enhanced `MacroDataLoader`

---

## Files to Read First

1. **`/workspaces/qetf/CLAUDE_CONTEXT.md`** - Coding standards
2. **`/workspaces/qetf/src/quantetf/backtest/simple_engine.py`** - Current engine
3. **`/workspaces/qetf/src/quantetf/monitoring/regime.py`** - RegimeMonitor
4. **`/workspaces/qetf/src/quantetf/alpha/selector.py`** - AlphaSelector (from IMPL-016)
5. **`/workspaces/qetf/src/quantetf/production/pipeline.py`** - Production pipeline

---

## Implementation Steps

### 1. Create RegimeAwareAlphaModel wrapper

Create `src/quantetf/alpha/regime_aware.py`:

```python
"""Regime-aware alpha model that delegates to appropriate model per regime."""

from typing import Dict, Optional, List
import pandas as pd
import logging

from quantetf.alpha.base import AlphaModel
from quantetf.alpha.selector import AlphaSelector, AlphaSelection, MarketRegime
from quantetf.data.macro_loader import MacroDataLoader

logger = logging.getLogger(__name__)


class RegimeAwareAlpha(AlphaModel):
    """
    Alpha model that dynamically selects underlying model based on regime.

    This is the main integration point - it looks like a regular AlphaModel
    to the backtest engine, but internally uses regime detection to choose
    which model to use.
    """

    def __init__(
        self,
        selector: AlphaSelector,
        models: Dict[str, AlphaModel],
        macro_loader: MacroDataLoader,
        regime_detector: "RegimeDetector",
    ):
        """
        Initialize regime-aware alpha.

        Args:
            selector: AlphaSelector for choosing model based on regime
            models: Dict of model_name -> AlphaModel instances
            macro_loader: For accessing macro data
            regime_detector: For detecting current regime
        """
        self.selector = selector
        self.models = models
        self.macro_loader = macro_loader
        self.regime_detector = regime_detector

        # Track regime history for analysis
        self._regime_history: List[tuple] = []

    def score(
        self,
        universe: List[str],
        as_of: pd.Timestamp,
        data_store: "DataStore",
    ) -> pd.Series:
        """
        Score universe using regime-appropriate model.

        Args:
            universe: Tickers to score
            as_of: Scoring date
            data_store: Data store for prices

        Returns:
            Alpha scores Series
        """
        # 1. Detect current regime
        regime = self.regime_detector.detect_regime(as_of, self.macro_loader)
        logger.info(f"Detected regime {regime.value} as of {as_of.date()}")

        # 2. Select model for this regime
        selection = self.selector.select(regime, as_of, self.models)
        logger.info(f"Selected: {self._describe_selection(selection)}")

        # 3. Record for analysis
        self._regime_history.append((as_of, regime, selection))

        # 4. Compute scores
        if selection.is_single_model:
            return selection.model.score(universe, as_of, data_store)
        else:
            # Weighted ensemble
            scores = pd.DataFrame()
            for model_name, weight in selection.model_weights.items():
                model_scores = self.models[model_name].score(universe, as_of, data_store)
                scores[model_name] = model_scores * weight
            return scores.sum(axis=1)

    def _describe_selection(self, selection: AlphaSelection) -> str:
        """Create human-readable description of selection."""
        if selection.is_single_model:
            # Find model name
            for name, model in self.models.items():
                if model is selection.model:
                    return f"model={name}"
            return "model=unknown"
        else:
            weights_str = ", ".join(
                f"{k}:{v:.1%}" for k, v in selection.model_weights.items()
            )
            return f"ensemble=[{weights_str}]"

    def get_regime_history(self) -> pd.DataFrame:
        """Get regime detection history as DataFrame."""
        if not self._regime_history:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "date": as_of,
                "regime": regime.value,
                "selection": self._describe_selection(selection),
            }
            for as_of, regime, selection in self._regime_history
        ])

    @property
    def name(self) -> str:
        return "RegimeAwareAlpha"
```

### 2. Create RegimeDetector class (unified interface)

Add to `src/quantetf/alpha/regime_aware.py` or `src/quantetf/regime/detector.py`:

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class RegimeThresholds:
    """Thresholds for regime classification."""
    vix_elevated: float = 20.0
    vix_high: float = 30.0
    yield_curve_inverted: float = 0.0
    hy_spread_elevated: float = 4.0
    hy_spread_stressed: float = 6.0


class RegimeDetector:
    """
    Detect market regime from macro indicators.

    Uses configurable thresholds to classify current conditions.
    """

    def __init__(self, thresholds: Optional[RegimeThresholds] = None):
        self.thresholds = thresholds or RegimeThresholds()

    def detect_regime(
        self,
        as_of: pd.Timestamp,
        macro_loader: MacroDataLoader,
    ) -> MarketRegime:
        """
        Detect market regime as-of date.

        Decision tree:
        1. If yield curve inverted AND HY spread elevated → RECESSION_WARNING
        2. If VIX > 30 → HIGH_VOL
        3. If VIX > 20 OR HY spread elevated → ELEVATED_VOL
        4. Otherwise → RISK_ON
        """
        vix = macro_loader.get_vix(as_of)
        yield_spread = macro_loader.get_yield_curve_spread(as_of)
        hy_spread = macro_loader.get_credit_spread(as_of, high_yield=True)

        # Handle missing data
        if vix is None:
            vix = 20.0  # Assume normal
        if yield_spread is None:
            yield_spread = 1.0  # Assume normal
        if hy_spread is None:
            hy_spread = 3.0  # Assume normal

        # Classification logic
        yield_inverted = yield_spread < self.thresholds.yield_curve_inverted
        hy_stressed = hy_spread > self.thresholds.hy_spread_stressed
        hy_elevated = hy_spread > self.thresholds.hy_spread_elevated

        if yield_inverted and hy_elevated:
            return MarketRegime.RECESSION_WARNING
        elif vix > self.thresholds.vix_high:
            return MarketRegime.HIGH_VOL
        elif vix > self.thresholds.vix_elevated or hy_elevated:
            return MarketRegime.ELEVATED_VOL
        else:
            return MarketRegime.RISK_ON

    def detect_with_details(
        self,
        as_of: pd.Timestamp,
        macro_loader: MacroDataLoader,
    ) -> dict:
        """Detect regime and return detailed breakdown."""
        regime = self.detect_regime(as_of, macro_loader)

        return {
            "regime": regime.value,
            "as_of": as_of.isoformat(),
            "indicators": macro_loader.get_macro_snapshot(as_of),
            "thresholds": {
                "vix_elevated": self.thresholds.vix_elevated,
                "vix_high": self.thresholds.vix_high,
                "yield_curve_inverted": self.thresholds.yield_curve_inverted,
            },
        }
```

### 3. Update BacktestConfig to support regime-aware alpha

Update `src/quantetf/backtest/simple_engine.py`:

```python
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    rebalance_frequency: str = "M"  # M=monthly, W=weekly
    initial_capital: float = 100_000.0

    # Alpha configuration
    alpha_model: Optional[AlphaModel] = None

    # Regime-aware configuration (new)
    regime_aware: bool = False
    alpha_selector: Optional["AlphaSelector"] = None
    alpha_models: Optional[Dict[str, AlphaModel]] = None
    macro_loader: Optional["MacroDataLoader"] = None
    regime_detector: Optional["RegimeDetector"] = None


class SimpleBacktestEngine:
    """Event-driven backtest engine."""

    def __init__(self, config: BacktestConfig, data_store: "DataStore"):
        self.config = config
        self.data_store = data_store

        # Build alpha model
        self.alpha_model = self._build_alpha_model()

    def _build_alpha_model(self) -> AlphaModel:
        """Build alpha model from config, handling regime-aware case."""
        if self.config.regime_aware:
            # Validate required components
            if not all([
                self.config.alpha_selector,
                self.config.alpha_models,
                self.config.macro_loader,
                self.config.regime_detector,
            ]):
                raise ValueError(
                    "regime_aware=True requires alpha_selector, alpha_models, "
                    "macro_loader, and regime_detector"
                )

            return RegimeAwareAlpha(
                selector=self.config.alpha_selector,
                models=self.config.alpha_models,
                macro_loader=self.config.macro_loader,
                regime_detector=self.config.regime_detector,
            )
        else:
            if self.config.alpha_model is None:
                raise ValueError("Must provide alpha_model or enable regime_aware")
            return self.config.alpha_model
```

### 4. Create factory function for easy setup

Add to `src/quantetf/alpha/regime_aware.py`:

```python
def create_regime_aware_backtest(
    data_store: "DataStore",
    macro_dir: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    regime_model_map: Dict[str, str],  # regime_name -> model_name
    model_configs: Dict[str, dict],  # model_name -> config
    rebalance_frequency: str = "M",
) -> SimpleBacktestEngine:
    """
    Factory to create regime-aware backtest engine.

    Example:
        engine = create_regime_aware_backtest(
            data_store=store,
            macro_dir="data/raw/macro",
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2024-12-31"),
            regime_model_map={
                "risk_on": "momentum_acceleration",
                "elevated_vol": "vol_adjusted_momentum",
                "high_vol": "dual_momentum",
                "recession_warning": "trend_filtered_momentum",
            },
            model_configs={
                "momentum_acceleration": {"short_window": 21, "long_window": 63},
                "vol_adjusted_momentum": {"lookback": 252},
                "dual_momentum": {"lookback": 252},
                "trend_filtered_momentum": {"trend_window": 200},
            },
        )
    """
    from quantetf.alpha.factory import AlphaModelRegistry
    from quantetf.alpha.selector import RegimeBasedSelector, MarketRegime

    # Load macro data
    macro_loader = MacroDataLoader(macro_dir)

    # Create regime detector
    regime_detector = RegimeDetector()

    # Create alpha models
    registry = AlphaModelRegistry()
    models = {}
    for model_name, config in model_configs.items():
        models[model_name] = registry.create(model_name, config)

    # Create selector
    mapping = {MarketRegime(k): v for k, v in regime_model_map.items()}
    selector = RegimeBasedSelector(
        regime_model_map=mapping,
        default_model=list(model_configs.keys())[0],
    )

    # Build config
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        rebalance_frequency=rebalance_frequency,
        regime_aware=True,
        alpha_selector=selector,
        alpha_models=models,
        macro_loader=macro_loader,
        regime_detector=regime_detector,
    )

    return SimpleBacktestEngine(config, data_store)
```

### 5. Update production pipeline

Update `src/quantetf/production/pipeline.py`:

```python
class ProductionPipeline:
    """Production pipeline with regime-aware alpha."""

    def __init__(
        self,
        alpha_model: AlphaModel,  # Can be RegimeAwareAlpha
        portfolio_constructor: PortfolioConstructor,
        # ... other params
    ):
        self.alpha_model = alpha_model
        # ...

    def run(self, as_of: pd.Timestamp) -> PipelineResult:
        """Run pipeline for given date."""
        # Get universe
        universe = self.universe_provider.get_universe(as_of)

        # Score using alpha model (regime detection happens inside if regime-aware)
        scores = self.alpha_model.score(universe, as_of, self.data_store)

        # If using RegimeAwareAlpha, log regime info
        if isinstance(self.alpha_model, RegimeAwareAlpha):
            history = self.alpha_model.get_regime_history()
            if not history.empty:
                latest = history.iloc[-1]
                logger.info(f"Production run using regime: {latest['regime']}")

        # Continue with portfolio construction...
```

### 6. Create integration tests

Create `tests/integration/test_regime_aware_backtest.py`:

```python
import pytest
import pandas as pd
from pathlib import Path

from quantetf.alpha.regime_aware import (
    RegimeAwareAlpha,
    RegimeDetector,
    create_regime_aware_backtest,
)
from quantetf.alpha.selector import RegimeBasedSelector, MarketRegime
from quantetf.data.macro_loader import MacroDataLoader


class TestRegimeAwareBacktest:
    """Integration tests for regime-aware backtesting."""

    @pytest.fixture
    def sample_setup(self, tmp_path):
        """Create minimal setup for testing."""
        # Create mock macro data
        macro_dir = tmp_path / "macro"
        macro_dir.mkdir()

        dates = pd.date_range("2023-01-01", "2024-01-01", freq="D")

        # VIX with varying regimes
        vix_values = [15.0] * 100 + [25.0] * 100 + [35.0] * 50 + [20.0] * 116
        vix_df = pd.DataFrame({"value": vix_values[:len(dates)]}, index=dates)
        vix_df.to_parquet(macro_dir / "VIX.parquet")

        # Yield curve
        yc_df = pd.DataFrame({"value": [1.0] * len(dates)}, index=dates)
        yc_df.to_parquet(macro_dir / "T10Y2Y.parquet")

        return {"macro_dir": str(macro_dir), "dates": dates}

    def test_regime_detection_changes_model(self, sample_setup):
        """Test that regime changes trigger model switches."""
        macro_loader = MacroDataLoader(sample_setup["macro_dir"])
        detector = RegimeDetector()

        # Check regime at different points
        regime_jan = detector.detect_regime(
            pd.Timestamp("2023-01-15"), macro_loader
        )
        regime_may = detector.detect_regime(
            pd.Timestamp("2023-05-15"), macro_loader
        )
        regime_aug = detector.detect_regime(
            pd.Timestamp("2023-08-15"), macro_loader
        )

        # VIX ~15 -> RISK_ON
        assert regime_jan == MarketRegime.RISK_ON

        # VIX ~25 -> ELEVATED_VOL
        assert regime_may == MarketRegime.ELEVATED_VOL

        # VIX ~35 -> HIGH_VOL
        assert regime_aug == MarketRegime.HIGH_VOL

    def test_regime_aware_alpha_tracks_history(self, sample_setup):
        """Test that regime history is recorded."""
        from unittest.mock import Mock

        macro_loader = MacroDataLoader(sample_setup["macro_dir"])
        detector = RegimeDetector()

        # Create mock models
        mock_model = Mock()
        mock_model.score.return_value = pd.Series({"SPY": 0.5})

        selector = RegimeBasedSelector(
            regime_model_map={
                MarketRegime.RISK_ON: "test",
                MarketRegime.ELEVATED_VOL: "test",
                MarketRegime.HIGH_VOL: "test",
            },
            default_model="test",
        )

        alpha = RegimeAwareAlpha(
            selector=selector,
            models={"test": mock_model},
            macro_loader=macro_loader,
            regime_detector=detector,
        )

        # Score at multiple dates
        alpha.score(["SPY"], pd.Timestamp("2023-01-15"), Mock())
        alpha.score(["SPY"], pd.Timestamp("2023-05-15"), Mock())
        alpha.score(["SPY"], pd.Timestamp("2023-08-15"), Mock())

        history = alpha.get_regime_history()

        assert len(history) == 3
        assert history.iloc[0]["regime"] == "risk_on"
        assert history.iloc[1]["regime"] == "elevated_vol"
        assert history.iloc[2]["regime"] == "high_vol"
```

---

## Acceptance Criteria

- [ ] `RegimeAwareAlpha` class wraps regime detection + model selection
- [ ] `RegimeDetector` class with configurable thresholds
- [ ] `BacktestConfig` supports `regime_aware=True` mode
- [ ] `SimpleBacktestEngine` correctly uses regime-aware alpha
- [ ] `create_regime_aware_backtest()` factory function works
- [ ] Production pipeline supports `RegimeAwareAlpha`
- [ ] Regime history tracked and accessible for analysis
- [ ] Integration tests pass
- [ ] Logging shows regime and model selection at each rebalance

---

## Example Usage

```python
# Create regime-aware backtest
engine = create_regime_aware_backtest(
    data_store=snapshot_store,
    macro_dir="data/raw/macro",
    start_date=pd.Timestamp("2020-01-01"),
    end_date=pd.Timestamp("2024-12-31"),
    regime_model_map={
        "risk_on": "momentum_acceleration",
        "elevated_vol": "vol_adjusted_momentum",
        "high_vol": "dual_momentum",
        "recession_warning": "trend_filtered_momentum",
    },
    model_configs={
        "momentum_acceleration": {"short_window": 21, "long_window": 63},
        "vol_adjusted_momentum": {"lookback": 252},
        "dual_momentum": {"lookback": 252},
        "trend_filtered_momentum": {"trend_window": 200},
    },
)

# Run backtest
result = engine.run()

# Analyze regime history
regime_history = engine.alpha_model.get_regime_history()
print(regime_history.groupby("regime").size())
```

---

## Definition of Done

1. All acceptance criteria met
2. Integration tests pass
3. Example script demonstrating usage created
4. PROGRESS_LOG.md updated
5. Completion note created: `handoffs/completion-IMPL-018.md`
6. TASKS.md status updated to `completed`
7. Code committed with clear message

---

## Notes

- This is the "glue" task that connects all the pieces
- Depends on IMPL-016 and IMPL-017 being complete
- Keep the interface clean so non-regime-aware backtests still work
- Consider adding regime visualization to artifacts output
- Future: Add regime-conditional risk overlays
