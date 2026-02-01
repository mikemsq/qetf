# HANDOUT: Walk-Forward Optimizer Implementation

**Task ID**: IMPL-036
**Type**: Implementation
**Priority**: High
**Estimated Complexity**: Medium-High

---

## Executive Summary

The current strategy optimizer selects strategies based on **in-sample performance**, leading to overfitting. A strategy that shows +12% active return in 1-year optimization underperforms SPY by 100%+ over 10 years.

This task modifies the optimizer to use **walk-forward validation** internally, scoring strategies by their **out-of-sample (OOS) performance** rather than in-sample performance.

---

## Problem Statement

### Current Behavior

```
Optimizer Flow (FLAWED):
┌─────────────────────────────────────────────────────┐
│  Full History: 2016 ──────────────────────── 2026   │
│                                                     │
│  Evaluation Window: ─────────────────[2025──2026]   │
│                     (1-year trailing)               │
│                                                     │
│  Selection: Pick strategy with best 1yr return      │
│  Result: OVERFIT to recent market regime            │
└─────────────────────────────────────────────────────┘
```

### Desired Behavior

```
Optimizer Flow (CORRECT):
┌─────────────────────────────────────────────────────┐
│  Full History: 2016 ──────────────────────── 2026   │
│                                                     │
│  Window 1: [Train: 2016-2018] → [Test: 2019]        │
│  Window 2: [Train: 2017-2019] → [Test: 2020]        │
│  Window 3: [Train: 2018-2020] → [Test: 2021]        │
│  Window 4: [Train: 2019-2021] → [Test: 2022]        │
│  Window 5: [Train: 2020-2022] → [Test: 2023]        │
│  Window 6: [Train: 2021-2023] → [Test: 2024]        │
│  Window 7: [Train: 2022-2024] → [Test: 2025]        │
│                                                     │
│  Selection: Pick strategy with best AVERAGE OOS     │
│  Result: Robust to regime changes                   │
└─────────────────────────────────────────────────────┘
```

---

## Architecture Overview

### Existing Components

| File | Purpose | Modification |
|------|---------|--------------|
| `src/quantetf/evaluation/walk_forward.py` | Walk-forward validation logic | Extend for multi-alpha support |
| `src/quantetf/optimization/evaluator.py` | `MultiPeriodEvaluator` class | Add walk-forward evaluation mode |
| `src/quantetf/optimization/optimizer.py` | `StrategyOptimizer` class | Use WF evaluator, new scoring |
| `scripts/run_backtests.py` | CLI for running optimization | Add `--walk-forward` flag |

### New Components

| File | Purpose |
|------|---------|
| `src/quantetf/optimization/walk_forward_evaluator.py` | New evaluator using walk-forward windows |

---

## Detailed Requirements

### 1. Walk-Forward Evaluator

Create `src/quantetf/optimization/walk_forward_evaluator.py`:

```python
@dataclass
class WalkForwardEvaluatorConfig:
    """Configuration for walk-forward evaluation."""
    train_years: int = 3          # Years of training data per window
    test_years: int = 1           # Years of test data per window
    step_months: int = 6          # How far to slide window each iteration
    min_windows: int = 4          # Minimum windows required for valid evaluation
    require_positive_oos: bool = True  # Require positive OOS return to pass

@dataclass
class WalkForwardEvaluationResult:
    """Results from walk-forward evaluation of a single strategy."""
    config_name: str
    num_windows: int

    # Out-of-sample metrics (PRIMARY - use these for ranking)
    oos_sharpe_mean: float
    oos_sharpe_std: float
    oos_return_mean: float
    oos_active_return_mean: float  # vs SPY
    oos_win_rate: float            # % of windows with positive active return

    # In-sample metrics (for reference/debugging)
    is_sharpe_mean: float
    is_return_mean: float

    # Degradation metrics
    sharpe_degradation: float      # IS - OOS (lower is better, negative means OOS > IS)
    return_degradation: float

    # Per-window details
    window_results: List[WindowResult]

    # Composite score for ranking (based on OOS metrics)
    composite_score: float
```

**Key Methods:**

```python
class WalkForwardEvaluator:
    def __init__(
        self,
        data_access: DataAccessContext,
        wf_config: WalkForwardEvaluatorConfig,
        cost_bps: float = 10.0,
    ):
        """Initialize with data access and walk-forward configuration."""

    def evaluate(self, strategy_config: StrategyConfig) -> WalkForwardEvaluationResult:
        """
        Evaluate a single strategy using walk-forward validation.

        For each window:
        1. Run backtest on TEST period only (train period establishes warmup)
        2. Calculate OOS metrics vs SPY benchmark
        3. Aggregate across all windows

        Returns WalkForwardEvaluationResult with OOS-based scoring.
        """

    def _generate_windows(self) -> List[WalkForwardWindow]:
        """Generate train/test windows based on config."""
        # Reuse logic from walk_forward.py:generate_walk_forward_windows()

    def _evaluate_window(
        self,
        strategy_config: StrategyConfig,
        window: WalkForwardWindow,
    ) -> WindowResult:
        """
        Run backtest for a single test window.

        IMPORTANT: Only evaluate on TEST period. Train period provides
        warmup data for indicators but is NOT scored.
        """

    def _calculate_composite_score(self, result: WalkForwardEvaluationResult) -> float:
        """
        Calculate composite score for ranking.

        Suggested formula:
        score = (oos_active_return_mean * oos_win_rate) + oos_sharpe_mean

        This rewards:
        - Higher average outperformance vs SPY
        - Consistency (high win rate across windows)
        - Risk-adjusted returns (Sharpe)
        """
```

### 2. Modify StrategyOptimizer

Update `src/quantetf/optimization/optimizer.py`:

```python
class StrategyOptimizer:
    def __init__(
        self,
        data_access: DataAccessContext,
        output_dir: str,
        # REMOVE: periods_years parameter (replaced by walk-forward)
        # ADD: walk-forward config
        wf_config: Optional[WalkForwardEvaluatorConfig] = None,
        use_walk_forward: bool = True,  # Default to walk-forward
        max_workers: int = 1,
        cost_bps: float = 10.0,
        regime_analysis_enabled: bool = False,
    ):
        if use_walk_forward:
            self.evaluator = WalkForwardEvaluator(
                data_access=data_access,
                wf_config=wf_config or WalkForwardEvaluatorConfig(),
                cost_bps=cost_bps,
            )
        else:
            # Legacy mode for backwards compatibility
            self.evaluator = MultiPeriodEvaluator(...)
```

**Modify `run()` method:**

```python
def run(self, ...) -> List[WalkForwardEvaluationResult]:
    """
    Run optimization using walk-forward evaluation.

    Changes from current implementation:
    1. Use WalkForwardEvaluator instead of MultiPeriodEvaluator
    2. Score by OOS metrics, not in-sample
    3. Filter strategies that don't meet OOS criteria
    """
    results = []

    for config in strategy_configs:
        result = self.evaluator.evaluate(config)

        # Filter: require positive OOS active return
        if result.oos_active_return_mean > 0:
            results.append(result)
        else:
            logger.info(f"Filtered {config.name}: negative OOS active return")

    # Rank by composite score (OOS-based)
    results.sort(key=lambda r: r.composite_score, reverse=True)

    return results
```

### 3. Update CLI Script

Modify `scripts/run_backtests.py`:

```python
parser.add_argument(
    '--walk-forward',
    action='store_true',
    default=True,
    help='Use walk-forward validation for evaluation (default: True)'
)

parser.add_argument(
    '--train-years',
    type=int,
    default=3,
    help='Training window size in years (default: 3)'
)

parser.add_argument(
    '--test-years',
    type=int,
    default=1,
    help='Test window size in years (default: 1)'
)

parser.add_argument(
    '--step-months',
    type=int,
    default=6,
    help='Window step size in months (default: 6)'
)

# REMOVE or deprecate --periods argument
parser.add_argument(
    '--periods',
    type=str,
    default=None,
    help='DEPRECATED: Use --walk-forward instead'
)
```

### 4. Alpha Model Factory

The walk-forward evaluator needs to create alpha models from StrategyConfig. Ensure `StrategyConfig` can instantiate any alpha type:

```python
# In src/quantetf/config/loader.py or strategy_config.py

class StrategyConfig:
    def create_alpha_model(self) -> AlphaModel:
        """Create alpha model instance from config."""
        alpha_type = self.alpha_config.get('type')

        if alpha_type == 'momentum':
            return MomentumAlpha(**self.alpha_config)
        elif alpha_type == 'momentum_acceleration':
            return MomentumAccelerationAlpha(**self.alpha_config)
        elif alpha_type == 'vol_adjusted_momentum':
            return VolAdjustedMomentumAlpha(**self.alpha_config)
        elif alpha_type == 'residual_momentum':
            return ResidualMomentumAlpha(**self.alpha_config)
        # ... etc
        else:
            raise ValueError(f"Unknown alpha type: {alpha_type}")
```

---

## Implementation Plan

### Phase 1: Core Walk-Forward Evaluator

**Tasks:**

1. **IMPL-036-A**: Create `WalkForwardEvaluatorConfig` and `WalkForwardEvaluationResult` dataclasses
2. **IMPL-036-B**: Implement `WalkForwardEvaluator` class with `evaluate()` method
3. **IMPL-036-C**: Add unit tests for walk-forward window generation
4. **IMPL-036-D**: Add unit tests for single-window evaluation

### Phase 2: Optimizer Integration

**Tasks:**

5. **IMPL-036-E**: Modify `StrategyOptimizer` to use `WalkForwardEvaluator`
6. **IMPL-036-F**: Update composite scoring to use OOS metrics
7. **IMPL-036-G**: Add filtering for strategies with negative OOS performance

### Phase 3: CLI and Output

**Tasks:**

8. **IMPL-036-H**: Update `run_backtests.py` with walk-forward CLI arguments
9. **IMPL-036-I**: Update output format to include OOS metrics prominently
10. **IMPL-036-J**: Add summary statistics (% strategies filtered, OOS vs IS comparison)

### Phase 4: Validation

**Tasks:**

11. **IMPL-036-K**: Run walk-forward optimization on existing strategy universe
12. **IMPL-036-L**: Compare results to previous (flawed) optimization
13. **IMPL-036-M**: Document findings and update STATUS.md

---

## Success Criteria

### Functional Requirements

- [ ] Walk-forward evaluator generates correct train/test windows
- [ ] Each strategy is evaluated on TEST data only (train provides warmup)
- [ ] Composite score is based on OOS metrics, not in-sample
- [ ] Strategies with negative OOS active return are filtered
- [ ] Output includes both OOS and IS metrics for comparison

### Performance Requirements

- [ ] Evaluation of single strategy completes in <30 seconds
- [ ] Full optimization (1000+ configs) completes in <2 hours with parallel=4

### Quality Requirements

- [ ] All existing tests pass
- [ ] New unit tests for walk-forward evaluator (>80% coverage)
- [ ] Integration test: known overfit strategy shows poor OOS metrics

---

## Testing Strategy

### Unit Tests

```python
# tests/optimization/test_walk_forward_evaluator.py

def test_window_generation_basic():
    """Test that windows are generated correctly."""
    config = WalkForwardEvaluatorConfig(
        train_years=2, test_years=1, step_months=6
    )
    evaluator = WalkForwardEvaluator(mock_data_access, config)
    windows = evaluator._generate_windows()

    assert len(windows) >= 4
    for w in windows:
        assert w.test_start == w.train_end
        assert (w.test_end - w.test_start).days >= 365

def test_oos_scoring():
    """Test that composite score uses OOS metrics only."""
    result = WalkForwardEvaluationResult(
        oos_sharpe_mean=0.5,
        oos_active_return_mean=0.05,
        oos_win_rate=0.7,
        is_sharpe_mean=2.0,  # High IS should NOT inflate score
        ...
    )
    score = evaluator._calculate_composite_score(result)

    # Score should be moderate despite high IS
    assert 0 < score < 2.0

def test_overfit_strategy_detection():
    """Test that overfit strategy shows IS >> OOS."""
    # Create strategy that was optimized on recent data
    overfit_config = create_overfit_strategy_config()

    result = evaluator.evaluate(overfit_config)

    # Should show significant degradation
    assert result.sharpe_degradation > 0.5
    assert result.is_sharpe_mean > result.oos_sharpe_mean
```

### Integration Tests

```python
def test_full_optimization_walk_forward():
    """Test that optimizer uses walk-forward correctly."""
    optimizer = StrategyOptimizer(
        data_access=test_data_access,
        use_walk_forward=True,
        wf_config=WalkForwardEvaluatorConfig(
            train_years=2, test_years=1, step_months=12
        ),
    )

    results = optimizer.run(max_configs=10)

    # All results should have OOS metrics
    for r in results:
        assert r.oos_sharpe_mean is not None
        assert r.num_windows >= 4
        assert r.oos_active_return_mean > 0  # Filtered
```

---

## Edge Cases and Considerations

### 1. Warmup Period Handling

Some alpha models need significant warmup (e.g., 126-day momentum needs 126 days of data before first signal).

**Solution**: The train period provides warmup. When evaluating the test period, the model has already "warmed up" during the train period.

```python
def _evaluate_window(self, config, window):
    # Run backtest starting from TRAIN_START but only score TEST period
    backtest_config = BacktestConfig(
        start_date=window.train_start,  # Start early for warmup
        end_date=window.test_end,
    )

    result = engine.run(...)

    # Extract only TEST period returns for scoring
    test_returns = result.equity_curve[
        (result.equity_curve.index >= window.test_start) &
        (result.equity_curve.index <= window.test_end)
    ]

    return self._calculate_metrics(test_returns)
```

### 2. Insufficient Data

If the data range is too short for the configured windows:

```python
if len(windows) < self.config.min_windows:
    raise ValueError(
        f"Insufficient data for walk-forward validation. "
        f"Need {self.config.min_windows} windows but could only generate {len(windows)}. "
        f"Try reducing train_years or test_years."
    )
```

### 3. SPY Benchmark Alignment

SPY benchmark must be calculated for each TEST window separately, not the full period:

```python
def _get_spy_return_for_window(self, window: WalkForwardWindow) -> float:
    """Get SPY total return for the test period only."""
    spy_prices = self.data_access.prices.read_prices_as_of(
        as_of=window.test_end,
        tickers=['SPY'],
    )
    # Calculate return from test_start to test_end
    ...
```

### 4. Parallel Execution

Walk-forward evaluation is embarrassingly parallel across strategies:

```python
from concurrent.futures import ProcessPoolExecutor

def run_parallel(self, configs: List[StrategyConfig]) -> List[Result]:
    with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
        results = list(executor.map(self.evaluator.evaluate, configs))
    return results
```

---

## Output Format

### Console Output

```
================================================================================
WALK-FORWARD OPTIMIZATION RESULTS
================================================================================
Walk-Forward Config:
  Train Period: 3 years
  Test Period:  1 year
  Step Size:    6 months
  Windows:      7

Strategies Evaluated:    1,024
Strategies Passed OOS:     127  (12.4%)
Strategies Filtered:       897  (negative OOS active return)

TOP 10 STRATEGIES (by OOS composite score):
--------------------------------------------------------------------------------
Rank  Strategy                                    OOS_Sharpe  OOS_Active  Win%
--------------------------------------------------------------------------------
  1   momentum_lookback252_top5_monthly              0.82      +4.2%     71%
  2   vol_adj_momentum_lookback126_top7_weekly       0.75      +3.8%     71%
  3   residual_momentum_lookback63_top5_monthly      0.71      +3.5%     57%
...

DEGRADATION ANALYSIS (IS vs OOS):
--------------------------------------------------------------------------------
Strategy                                    IS_Sharpe  OOS_Sharpe  Degradation
--------------------------------------------------------------------------------
momentum_accel_126_63_top7_tier3 (PREV BEST)   2.72      0.21       -92%
momentum_lookback252_top5_monthly              0.95      0.82       -14%
...
================================================================================
```

### CSV Output

```csv
config_name,composite_score,num_windows,oos_sharpe_mean,oos_sharpe_std,oos_return_mean,oos_active_return_mean,oos_win_rate,is_sharpe_mean,is_return_mean,sharpe_degradation,passed_filter
momentum_lookback252_top5_monthly,1.42,7,0.82,0.31,0.12,0.042,0.71,0.95,0.18,-0.13,True
vol_adj_momentum_lookback126_top7_weekly,1.35,7,0.75,0.28,0.11,0.038,0.71,0.88,0.15,-0.13,True
...
```

---

## References

- Current walk-forward implementation: [src/quantetf/evaluation/walk_forward.py](src/quantetf/evaluation/walk_forward.py)
- Current optimizer: [src/quantetf/optimization/optimizer.py](src/quantetf/optimization/optimizer.py)
- Current evaluator: [src/quantetf/optimization/evaluator.py](src/quantetf/optimization/evaluator.py)
- Walk-forward test script: [scripts/walk_forward_test.py](scripts/walk_forward_test.py)
- Evidence of overfitting: [artifacts/backtests/20260201_023759_momentum_acceleration_*/out.txt](artifacts/backtests/20260201_023759_momentum_acceleration_long_lookback_days126_min_periods100_short_lookback_days63_top7_tier3_monthly/out.txt)

---

## Appendix: Composite Score Formula Options

### Option A: Weighted Sum (Recommended)

```python
score = (
    0.4 * oos_active_return_mean * 10 +  # Scale to ~0-1 range
    0.3 * oos_sharpe_mean +
    0.3 * oos_win_rate
)
```

### Option B: Information Ratio Focus

```python
score = oos_information_ratio * oos_win_rate
```

### Option C: Risk-Adjusted with Penalty

```python
score = oos_sharpe_mean * (1 - max(0, sharpe_degradation))
# Penalizes strategies that degrade significantly from IS to OOS
```

The architect should evaluate these options and select based on project goals (maximizing active return vs. consistency vs. risk-adjusted performance).
