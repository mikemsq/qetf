# Task IMPL-036-E/F/G: Optimizer Integration with Walk-Forward Evaluator

## Files to Modify
- `src/quantetf/optimization/optimizer.py`

## Purpose
Modify `StrategyOptimizer` to use `WalkForwardEvaluator` instead of `MultiPeriodEvaluator` for strategy evaluation. This changes the scoring from in-sample to out-of-sample metrics.

---

## Background

The current `StrategyOptimizer` uses `MultiPeriodEvaluator` which scores strategies on **in-sample** performance. This leads to overfitting - strategies that look great in historical backtests but fail in production.

The new approach:
- Use `WalkForwardEvaluator` to evaluate strategies on **out-of-sample** data
- Score strategies by their OOS composite score
- Filter out strategies with negative OOS active return

---

## Changes Required

### 1. Add New Constructor Parameters

```python
# In StrategyOptimizer.__init__

def __init__(
    self,
    data_access: DataAccessContext,
    output_dir: str | Path = "artifacts/optimization",
    # EXISTING: periods_years - keep for backwards compatibility but deprecate
    periods_years: Optional[List[int]] = None,
    max_workers: int = 1,
    cost_bps: float = 10.0,
    regime_analysis_enabled: bool = True,
    num_finalists: int = 6,
    scoring_method: str = "regime_weighted",
    trailing_days: int = 252,
    # NEW: Walk-forward configuration
    use_walk_forward: bool = True,  # Default to walk-forward
    wf_config: Optional['WalkForwardEvaluatorConfig'] = None,
):
    """Initialize the optimizer.

    Args:
        data_access: DataAccessContext for historical prices and macro data
        output_dir: Base directory for output files
        periods_years: DEPRECATED - use wf_config instead. Evaluation periods
            in years, only used when use_walk_forward=False.
        max_workers: Number of parallel workers (1 = sequential)
        cost_bps: Transaction cost in basis points (default: 10)
        regime_analysis_enabled: Whether to run regime analysis on winners
        num_finalists: Number of top strategies for regime mapping
        scoring_method: Method for ranking (multi_period, trailing_1y, regime_weighted)
        trailing_days: Days for trailing window evaluation (default: 252)
        use_walk_forward: If True, use WalkForwardEvaluator (default: True).
            If False, use legacy MultiPeriodEvaluator.
        wf_config: Walk-forward configuration. Uses defaults if None.
    """
```

### 2. Initialize Evaluator Based on Mode

```python
# Add to __init__ after existing initialization

from quantetf.optimization.walk_forward_evaluator import (
    WalkForwardEvaluator,
    WalkForwardEvaluatorConfig,
    WalkForwardEvaluationResult,
)

# Store mode
self.use_walk_forward = use_walk_forward
self.wf_config = wf_config or WalkForwardEvaluatorConfig()

# Log deprecation warning if using old mode with periods_years
if not use_walk_forward and periods_years:
    import warnings
    warnings.warn(
        "periods_years parameter is deprecated. "
        "Use use_walk_forward=True with wf_config instead.",
        DeprecationWarning,
        stacklevel=2,
    )

logger.info(
    f"StrategyOptimizer initialized: "
    f"use_walk_forward={use_walk_forward}, "
    f"periods={self.periods_years if not use_walk_forward else 'N/A'}, "
    f"wf_config={self.wf_config if use_walk_forward else 'N/A'}, "
    f"max_workers={max_workers}, "
    f"regime_analysis={self.regime_analysis_enabled}"
)
```

### 3. Create Appropriate Evaluator in _run_sequential

```python
def _run_sequential(
    self,
    configs: List[StrategyConfig],
) -> Tuple[List, int]:
    """Run evaluations sequentially.

    Returns:
        Tuple of (results list, failure count)
    """
    results = []
    failed_count = 0

    # Create evaluator based on mode
    if self.use_walk_forward:
        evaluator = WalkForwardEvaluator(
            data_access=self.data_access,
            wf_config=self.wf_config,
            cost_bps=self.cost_bps,
        )
    else:
        # Legacy mode
        evaluator = MultiPeriodEvaluator(
            data_access=self.data_access,
            cost_bps=self.cost_bps,
        )

    # Progress bar
    iterator = configs
    if TQDM_AVAILABLE:
        iterator = tqdm(configs, desc="Evaluating strategies", unit="config")

    for config in iterator:
        try:
            if self.use_walk_forward:
                result = evaluator.evaluate(config)
            else:
                result = evaluator.evaluate(config, periods_years=self.periods_years)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to evaluate {config.generate_name()}: {e}")
            failed_count += 1

    return results, failed_count
```

### 4. Modify rank_results for Walk-Forward Results

```python
def rank_results(
    self,
    results: List,  # Can be MultiPeriodResult or WalkForwardEvaluationResult
) -> Tuple[List, List, Optional[StrategyConfig]]:
    """Rank and filter results based on evaluation mode.

    Args:
        results: List of evaluation results

    Returns:
        Tuple of (all_results_sorted, winners, best_config)
    """
    if self.use_walk_forward:
        return self._rank_walk_forward_results(results)
    else:
        return self._rank_multi_period_results(results)

def _rank_walk_forward_results(
    self,
    results: List['WalkForwardEvaluationResult'],
) -> Tuple[List, List, Optional[StrategyConfig]]:
    """Rank results from walk-forward evaluation.

    Filters strategies with negative OOS active return, then ranks
    by composite score (OOS-based).

    Args:
        results: List of WalkForwardEvaluationResult

    Returns:
        Tuple of (all_results_sorted, winners, best_config)
    """
    # Filter: require positive OOS active return
    winners = [
        r for r in results
        if r.oos_active_return_mean > 0
    ]

    filtered_count = len(results) - len(winners)
    if filtered_count > 0:
        logger.info(
            f"Filtered {filtered_count} strategies with negative OOS active return"
        )

    # Sort all results by composite score (higher is better)
    all_sorted = sorted(results, key=lambda r: r.composite_score, reverse=True)
    winners_sorted = sorted(winners, key=lambda r: r.composite_score, reverse=True)

    # Get best config
    best_config = winners_sorted[0].config if winners_sorted else None

    return all_sorted, winners_sorted, best_config

def _rank_multi_period_results(
    self,
    results: List['MultiPeriodResult'],
) -> Tuple[List, List, Optional[StrategyConfig]]:
    """Legacy ranking for multi-period results.

    (Keep existing logic from current rank_results method)
    """
    # ... existing implementation ...
```

### 5. Update Output Methods for Walk-Forward Results

```python
def _save_all_results_csv(
    self,
    results: List,
    path: Path,
) -> None:
    """Save all results to CSV.

    Handles both WalkForwardEvaluationResult and MultiPeriodResult.
    """
    if not results:
        logger.warning("No results to save")
        return

    # Convert to dicts
    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)

    # Add walk-forward specific columns if present
    if self.use_walk_forward:
        # Reorder columns to show OOS metrics first
        oos_cols = [c for c in df.columns if c.startswith('oos_')]
        is_cols = [c for c in df.columns if c.startswith('is_')]
        other_cols = [c for c in df.columns if c not in oos_cols + is_cols]
        df = df[other_cols + oos_cols + is_cols]

    df.to_csv(path, index=False)
    logger.info(f"Saved all results to {path}")

def _save_report(
    self,
    results: List,
    winners: List,
    best_config: Optional[StrategyConfig],
    path: Path,
) -> None:
    """Generate markdown report."""
    lines = [
        "# Strategy Optimization Report",
        "",
        f"**Run Timestamp**: {self.run_timestamp}",
        f"**Evaluation Mode**: {'Walk-Forward' if self.use_walk_forward else 'Multi-Period'}",
        "",
    ]

    if self.use_walk_forward:
        lines.extend([
            "## Walk-Forward Configuration",
            "",
            f"- Train Period: {self.wf_config.train_years} years",
            f"- Test Period: {self.wf_config.test_years} years",
            f"- Step Size: {self.wf_config.step_months} months",
            f"- Minimum Windows: {self.wf_config.min_windows}",
            "",
            "## Summary",
            "",
            f"- Strategies Evaluated: {len(results)}",
            f"- Strategies with Positive OOS Active Return: {len(winners)} ({100*len(winners)/max(len(results),1):.1f}%)",
            f"- Strategies Filtered: {len(results) - len(winners)}",
            "",
        ])

        if winners:
            lines.extend([
                "## Top 10 Strategies (by OOS Composite Score)",
                "",
                "| Rank | Strategy | OOS Sharpe | OOS Active | Win Rate |",
                "|------|----------|------------|------------|----------|",
            ])
            for i, r in enumerate(winners[:10], 1):
                lines.append(
                    f"| {i} | {r.config_name[:45]} | {r.oos_sharpe_mean:.2f} | "
                    f"{r.oos_active_return_mean:+.1%} | {r.oos_win_rate:.0%} |"
                )
            lines.append("")

            # Degradation analysis
            lines.extend([
                "## Degradation Analysis (IS vs OOS)",
                "",
                "| Strategy | IS Sharpe | OOS Sharpe | Degradation |",
                "|----------|-----------|------------|-------------|",
            ])
            for r in winners[:5]:
                lines.append(
                    f"| {r.config_name[:40]} | {r.is_sharpe_mean:.2f} | "
                    f"{r.oos_sharpe_mean:.2f} | {r.sharpe_degradation:+.2f} |"
                )
            lines.append("")
    else:
        # Legacy multi-period report
        lines.extend([
            "## Multi-Period Configuration",
            "",
            f"- Periods: {self.periods_years}",
            "",
        ])
        # ... existing report logic ...

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"Saved report to {path}")
```

### 6. Update run() Method

```python
def run(
    self,
    max_configs: Optional[int] = None,
    schedules: Optional[List[str]] = None,
    alpha_types: Optional[List[str]] = None,
) -> OptimizationResult:
    """Run the full optimization pipeline.

    Args:
        max_configs: Limit number of configs (for testing)
        schedules: Filter to specific schedules (e.g., ['monthly'])
        alpha_types: Filter to specific alpha types

    Returns:
        OptimizationResult with all results and winners
    """
    # Create run directory
    self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    self._run_dir = self.output_dir / f"run_{self._run_timestamp}"
    self._run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting optimization run: {self._run_dir}")
    logger.info(f"Evaluation mode: {'Walk-Forward' if self.use_walk_forward else 'Multi-Period'}")

    # Generate configurations
    configs = generate_configs()
    total_configs = len(configs)

    # Apply filters
    if schedules:
        configs = [c for c in configs if c.schedule_name in schedules]
    if alpha_types:
        configs = [c for c in configs if c.alpha_type in alpha_types]
    if max_configs:
        configs = configs[:max_configs]

    logger.info(f"Evaluating {len(configs)} of {total_configs} configurations")

    # Run evaluations
    if self.max_workers > 1:
        results, failed_count = self._run_parallel(configs)
    else:
        results, failed_count = self._run_sequential(configs)

    # Rank and filter results
    all_sorted, winners, best_config = self.rank_results(results)

    # Save outputs
    self._save_results(all_sorted, winners, best_config)

    # Run regime analysis on finalists (if enabled)
    regime_outputs = None
    if self.regime_analysis_enabled and winners:
        regime_outputs = self._run_regime_analysis(winners[:self.num_finalists])

    return OptimizationResult(
        all_results=all_sorted,
        winners=winners,
        best_config=best_config,
        run_timestamp=self._run_timestamp,
        total_configs=total_configs,
        successful_configs=len(results),
        failed_configs=failed_count,
        output_dir=self._run_dir,
        regime_outputs=regime_outputs,
    )
```

---

## Console Output Format

When using walk-forward mode, the optimizer should print:

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
================================================================================
```

---

## Key Files to Reference

| File | Purpose |
|------|---------|
| `src/quantetf/optimization/optimizer.py` | File to modify |
| `src/quantetf/optimization/walk_forward_evaluator.py` | New evaluator to use |
| `src/quantetf/optimization/evaluator.py` | Legacy evaluator (keep for backwards compat) |

---

## Testing

```python
"""Tests for optimizer with walk-forward mode."""
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from quantetf.optimization.optimizer import StrategyOptimizer
from quantetf.optimization.walk_forward_evaluator import WalkForwardEvaluatorConfig


class TestOptimizerWalkForward:
    """Tests for walk-forward optimization mode."""

    @pytest.fixture
    def mock_data_access(self):
        """Create mock DataAccessContext."""
        mock = Mock()
        return mock

    def test_default_uses_walk_forward(self, mock_data_access):
        """Test that walk-forward is default mode."""
        optimizer = StrategyOptimizer(
            data_access=mock_data_access,
            output_dir="/tmp/test_opt",
        )
        assert optimizer.use_walk_forward is True

    def test_legacy_mode_with_flag(self, mock_data_access):
        """Test legacy mode can be enabled."""
        optimizer = StrategyOptimizer(
            data_access=mock_data_access,
            output_dir="/tmp/test_opt",
            use_walk_forward=False,
            periods_years=[3, 5, 10],
        )
        assert optimizer.use_walk_forward is False
        assert optimizer.periods_years == [3, 5, 10]

    def test_custom_wf_config(self, mock_data_access):
        """Test custom walk-forward configuration."""
        wf_config = WalkForwardEvaluatorConfig(
            train_years=2,
            test_years=1,
            step_months=12,
        )
        optimizer = StrategyOptimizer(
            data_access=mock_data_access,
            output_dir="/tmp/test_opt",
            wf_config=wf_config,
        )
        assert optimizer.wf_config.train_years == 2
        assert optimizer.wf_config.step_months == 12

    def test_rank_walk_forward_filters_negative(self, mock_data_access):
        """Test that negative OOS active return strategies are filtered."""
        from quantetf.optimization.walk_forward_evaluator import (
            WalkForwardEvaluationResult,
        )

        optimizer = StrategyOptimizer(
            data_access=mock_data_access,
            output_dir="/tmp/test_opt",
        )

        results = [
            WalkForwardEvaluationResult(
                config_name="good_strategy",
                oos_active_return_mean=0.05,
                composite_score=1.5,
            ),
            WalkForwardEvaluationResult(
                config_name="bad_strategy",
                oos_active_return_mean=-0.02,
                composite_score=0.3,
            ),
        ]

        all_sorted, winners, best = optimizer._rank_walk_forward_results(results)

        assert len(winners) == 1
        assert winners[0].config_name == "good_strategy"

    def test_rank_walk_forward_sorts_by_composite(self, mock_data_access):
        """Test that results are sorted by composite score."""
        from quantetf.optimization.walk_forward_evaluator import (
            WalkForwardEvaluationResult,
        )

        optimizer = StrategyOptimizer(
            data_access=mock_data_access,
            output_dir="/tmp/test_opt",
        )

        results = [
            WalkForwardEvaluationResult(
                config_name="second",
                oos_active_return_mean=0.03,
                composite_score=1.0,
            ),
            WalkForwardEvaluationResult(
                config_name="first",
                oos_active_return_mean=0.05,
                composite_score=1.5,
            ),
        ]

        all_sorted, winners, best = optimizer._rank_walk_forward_results(results)

        assert all_sorted[0].config_name == "first"
        assert winners[0].config_name == "first"
```

---

## Acceptance Checklist

- [ ] `use_walk_forward` parameter defaults to `True`
- [ ] `wf_config` parameter accepts `WalkForwardEvaluatorConfig`
- [ ] Deprecation warning shown when using `periods_years` with legacy mode
- [ ] `_run_sequential()` creates appropriate evaluator based on mode
- [ ] `_run_parallel()` creates appropriate evaluator based on mode
- [ ] `rank_results()` dispatches to correct ranking method
- [ ] `_rank_walk_forward_results()` filters negative OOS active return
- [ ] `_rank_walk_forward_results()` sorts by composite score
- [ ] Output CSV includes OOS metrics prominently
- [ ] Report markdown shows walk-forward config and results
- [ ] Console output shows walk-forward summary
- [ ] All existing tests still pass
- [ ] New tests for walk-forward mode pass

---

## Backwards Compatibility

The existing `MultiPeriodEvaluator` and related code should **not be deleted**. Users can still use legacy mode by setting `use_walk_forward=False`.

```python
# Legacy mode - still works
optimizer = StrategyOptimizer(
    data_access=ctx,
    use_walk_forward=False,
    periods_years=[3, 5, 10],
)
```

---

## Next Task

After completing this task, proceed to **IMPL-036-H**: Update CLI script with walk-forward arguments.
