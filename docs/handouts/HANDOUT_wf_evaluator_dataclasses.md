# Task IMPL-036-A: Walk-Forward Evaluator Dataclasses

## File to Create
`src/quantetf/optimization/walk_forward_evaluator.py`

## Purpose
Create the configuration and result dataclasses for walk-forward evaluation. These are the data structures that will be used by the WalkForwardEvaluator class.

---

## Background

The current optimizer uses in-sample performance to rank strategies, leading to overfitting. Walk-forward validation evaluates strategies on **out-of-sample (OOS)** data to detect overfitting before deployment.

---

## Implementation

```python
"""Walk-forward evaluation for strategy optimization.

This module provides walk-forward validation to evaluate strategies
on out-of-sample data, preventing overfitting to recent market regimes.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import pandas as pd


@dataclass
class WalkForwardEvaluatorConfig:
    """Configuration for walk-forward evaluation.

    Attributes:
        train_years: Number of years for each training window.
        test_years: Number of years for each test window.
        step_months: Months to slide window forward each iteration.
        min_windows: Minimum number of windows required for valid evaluation.
        require_positive_oos: If True, filter out strategies with negative OOS return.

    Example:
        With train_years=3, test_years=1, step_months=6:
        Window 1: Train [2016-2019] → Test [2019-2020]
        Window 2: Train [2016.5-2019.5] → Test [2019.5-2020.5]
        etc.
    """
    train_years: int = 3
    test_years: int = 1
    step_months: int = 6
    min_windows: int = 4
    require_positive_oos: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.train_years < 1:
            raise ValueError(f"train_years must be >= 1, got {self.train_years}")
        if self.test_years < 1:
            raise ValueError(f"test_years must be >= 1, got {self.test_years}")
        if self.step_months < 1 or self.step_months > 12:
            raise ValueError(f"step_months must be 1-12, got {self.step_months}")
        if self.min_windows < 1:
            raise ValueError(f"min_windows must be >= 1, got {self.min_windows}")


@dataclass
class WindowResult:
    """Results from a single walk-forward window.

    Attributes:
        window_id: Sequential identifier for this window.
        train_start: Start date of training period.
        train_end: End date of training period.
        test_start: Start date of test period (same as train_end).
        test_end: End date of test period.

        # In-sample (training) metrics - for reference only
        is_return: Total return during training period.
        is_sharpe: Sharpe ratio during training period.
        is_volatility: Annualized volatility during training.

        # Out-of-sample (test) metrics - PRIMARY for ranking
        oos_return: Total return during test period.
        oos_sharpe: Sharpe ratio during test period.
        oos_volatility: Annualized volatility during test.
        oos_max_drawdown: Maximum drawdown during test.

        # Benchmark comparison (test period only)
        spy_return: SPY total return during test period.
        active_return: oos_return - spy_return

        # Daily returns for further analysis
        oos_daily_returns: Series of daily returns during test period.
    """
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # In-sample metrics (reference only)
    is_return: float
    is_sharpe: float
    is_volatility: float

    # Out-of-sample metrics (PRIMARY)
    oos_return: float
    oos_sharpe: float
    oos_volatility: float
    oos_max_drawdown: float

    # Benchmark comparison
    spy_return: float
    active_return: float  # oos_return - spy_return

    # Daily returns for analysis
    oos_daily_returns: Optional[pd.Series] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        return {
            'window_id': self.window_id,
            'train_start': self.train_start.isoformat() if self.train_start else None,
            'train_end': self.train_end.isoformat() if self.train_end else None,
            'test_start': self.test_start.isoformat() if self.test_start else None,
            'test_end': self.test_end.isoformat() if self.test_end else None,
            'is_return': self.is_return,
            'is_sharpe': self.is_sharpe,
            'is_volatility': self.is_volatility,
            'oos_return': self.oos_return,
            'oos_sharpe': self.oos_sharpe,
            'oos_volatility': self.oos_volatility,
            'oos_max_drawdown': self.oos_max_drawdown,
            'spy_return': self.spy_return,
            'active_return': self.active_return,
        }


@dataclass
class WalkForwardEvaluationResult:
    """Results from walk-forward evaluation of a single strategy.

    This aggregates results across all walk-forward windows and provides
    summary statistics for strategy ranking.

    Attributes:
        config_name: Unique identifier for the strategy configuration.
        config: The StrategyConfig object (for reference).
        num_windows: Number of walk-forward windows evaluated.

        # Out-of-sample metrics (PRIMARY - use these for ranking)
        oos_sharpe_mean: Mean Sharpe ratio across test windows.
        oos_sharpe_std: Standard deviation of Sharpe across windows.
        oos_return_mean: Mean total return per test window.
        oos_active_return_mean: Mean active return vs SPY per window.
        oos_win_rate: Fraction of windows with positive active return.

        # In-sample metrics (for reference/debugging)
        is_sharpe_mean: Mean Sharpe during training windows.
        is_return_mean: Mean return during training windows.

        # Degradation metrics (IS - OOS; lower is better)
        sharpe_degradation: is_sharpe_mean - oos_sharpe_mean
        return_degradation: is_return_mean - oos_return_mean

        # Per-window details
        window_results: List of WindowResult objects.

        # Composite score for ranking (based on OOS metrics)
        composite_score: Final score for ranking strategies.
    """
    config_name: str
    config: Optional[object] = None  # StrategyConfig, optional for serialization
    num_windows: int = 0

    # Out-of-sample metrics (PRIMARY)
    oos_sharpe_mean: float = 0.0
    oos_sharpe_std: float = 0.0
    oos_return_mean: float = 0.0
    oos_active_return_mean: float = 0.0
    oos_win_rate: float = 0.0

    # In-sample metrics (reference)
    is_sharpe_mean: float = 0.0
    is_return_mean: float = 0.0

    # Degradation metrics
    sharpe_degradation: float = 0.0
    return_degradation: float = 0.0

    # Per-window details
    window_results: List[WindowResult] = field(default_factory=list)

    # Composite score
    composite_score: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        return {
            'config_name': self.config_name,
            'num_windows': self.num_windows,
            'oos_sharpe_mean': self.oos_sharpe_mean,
            'oos_sharpe_std': self.oos_sharpe_std,
            'oos_return_mean': self.oos_return_mean,
            'oos_active_return_mean': self.oos_active_return_mean,
            'oos_win_rate': self.oos_win_rate,
            'is_sharpe_mean': self.is_sharpe_mean,
            'is_return_mean': self.is_return_mean,
            'sharpe_degradation': self.sharpe_degradation,
            'return_degradation': self.return_degradation,
            'composite_score': self.composite_score,
        }

    def passed_filter(self) -> bool:
        """Check if this strategy passes OOS filters."""
        return self.oos_active_return_mean > 0
```

---

## Key Files to Reference

| File | What to Import/Reference |
|------|-------------------------|
| `src/quantetf/evaluation/walk_forward.py` | `WalkForwardConfig`, `WalkForwardWindow` patterns |
| `src/quantetf/optimization/evaluator.py` | `PeriodMetrics`, `MultiPeriodResult` patterns |
| `src/quantetf/optimization/grid.py` | `StrategyConfig` type |

---

## Testing

Create `tests/optimization/test_walk_forward_evaluator.py`:

```python
"""Tests for walk-forward evaluator dataclasses."""
import pytest
from datetime import datetime
import pandas as pd

from quantetf.optimization.walk_forward_evaluator import (
    WalkForwardEvaluatorConfig,
    WindowResult,
    WalkForwardEvaluationResult,
)


class TestWalkForwardEvaluatorConfig:
    """Tests for WalkForwardEvaluatorConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = WalkForwardEvaluatorConfig()
        assert config.train_years == 3
        assert config.test_years == 1
        assert config.step_months == 6
        assert config.min_windows == 4
        assert config.require_positive_oos is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = WalkForwardEvaluatorConfig(
            train_years=2,
            test_years=2,
            step_months=12,
            min_windows=3,
            require_positive_oos=False,
        )
        assert config.train_years == 2
        assert config.test_years == 2
        assert config.step_months == 12
        assert config.min_windows == 3
        assert config.require_positive_oos is False

    def test_invalid_train_years(self):
        """Test validation rejects invalid train_years."""
        with pytest.raises(ValueError, match="train_years must be >= 1"):
            WalkForwardEvaluatorConfig(train_years=0)

    def test_invalid_test_years(self):
        """Test validation rejects invalid test_years."""
        with pytest.raises(ValueError, match="test_years must be >= 1"):
            WalkForwardEvaluatorConfig(test_years=0)

    def test_invalid_step_months_low(self):
        """Test validation rejects step_months < 1."""
        with pytest.raises(ValueError, match="step_months must be 1-12"):
            WalkForwardEvaluatorConfig(step_months=0)

    def test_invalid_step_months_high(self):
        """Test validation rejects step_months > 12."""
        with pytest.raises(ValueError, match="step_months must be 1-12"):
            WalkForwardEvaluatorConfig(step_months=13)

    def test_invalid_min_windows(self):
        """Test validation rejects invalid min_windows."""
        with pytest.raises(ValueError, match="min_windows must be >= 1"):
            WalkForwardEvaluatorConfig(min_windows=0)


class TestWindowResult:
    """Tests for WindowResult dataclass."""

    def test_creation(self):
        """Test creating a WindowResult."""
        result = WindowResult(
            window_id=1,
            train_start=datetime(2016, 1, 1),
            train_end=datetime(2019, 1, 1),
            test_start=datetime(2019, 1, 1),
            test_end=datetime(2020, 1, 1),
            is_return=0.25,
            is_sharpe=1.2,
            is_volatility=0.15,
            oos_return=0.10,
            oos_sharpe=0.8,
            oos_volatility=0.18,
            oos_max_drawdown=-0.15,
            spy_return=0.08,
            active_return=0.02,
        )
        assert result.window_id == 1
        assert result.oos_return == 0.10
        assert result.active_return == 0.02

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = WindowResult(
            window_id=1,
            train_start=datetime(2016, 1, 1),
            train_end=datetime(2019, 1, 1),
            test_start=datetime(2019, 1, 1),
            test_end=datetime(2020, 1, 1),
            is_return=0.25,
            is_sharpe=1.2,
            is_volatility=0.15,
            oos_return=0.10,
            oos_sharpe=0.8,
            oos_volatility=0.18,
            oos_max_drawdown=-0.15,
            spy_return=0.08,
            active_return=0.02,
        )
        d = result.to_dict()
        assert d['window_id'] == 1
        assert d['oos_return'] == 0.10
        assert 'oos_daily_returns' not in d  # Excluded from dict


class TestWalkForwardEvaluationResult:
    """Tests for WalkForwardEvaluationResult dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        result = WalkForwardEvaluationResult(config_name="test_strategy")
        assert result.config_name == "test_strategy"
        assert result.num_windows == 0
        assert result.oos_sharpe_mean == 0.0
        assert result.window_results == []

    def test_passed_filter_positive(self):
        """Test passed_filter with positive active return."""
        result = WalkForwardEvaluationResult(
            config_name="test",
            oos_active_return_mean=0.05,
        )
        assert result.passed_filter() is True

    def test_passed_filter_negative(self):
        """Test passed_filter with negative active return."""
        result = WalkForwardEvaluationResult(
            config_name="test",
            oos_active_return_mean=-0.02,
        )
        assert result.passed_filter() is False

    def test_passed_filter_zero(self):
        """Test passed_filter with zero active return."""
        result = WalkForwardEvaluationResult(
            config_name="test",
            oos_active_return_mean=0.0,
        )
        assert result.passed_filter() is False

    def test_to_dict(self):
        """Test conversion to dictionary for CSV export."""
        result = WalkForwardEvaluationResult(
            config_name="momentum_top5_monthly",
            num_windows=7,
            oos_sharpe_mean=0.82,
            oos_sharpe_std=0.15,
            oos_return_mean=0.12,
            oos_active_return_mean=0.042,
            oos_win_rate=0.71,
            is_sharpe_mean=0.95,
            is_return_mean=0.18,
            sharpe_degradation=0.13,
            return_degradation=0.06,
            composite_score=1.42,
        )
        d = result.to_dict()
        assert d['config_name'] == "momentum_top5_monthly"
        assert d['num_windows'] == 7
        assert d['oos_sharpe_mean'] == 0.82
        assert d['composite_score'] == 1.42
        # window_results and config should not be in dict
        assert 'window_results' not in d
        assert 'config' not in d
```

---

## Acceptance Checklist

- [ ] `WalkForwardEvaluatorConfig` has all required fields with defaults
- [ ] `WalkForwardEvaluatorConfig.__post_init__` validates all parameters
- [ ] `WindowResult` captures both IS and OOS metrics for a single window
- [ ] `WindowResult.to_dict()` exports all numeric fields
- [ ] `WalkForwardEvaluationResult` aggregates across windows
- [ ] `WalkForwardEvaluationResult.passed_filter()` returns correct boolean
- [ ] `WalkForwardEvaluationResult.to_dict()` exports for CSV
- [ ] All tests pass
- [ ] Type hints on all parameters and return types
- [ ] Docstrings on all classes and methods

---

## Dependencies

None - this is a standalone file with only standard library and pandas imports.

---

## Next Task

After completing this task, proceed to **IMPL-036-B**: Implement the `WalkForwardEvaluator` class that uses these dataclasses.
