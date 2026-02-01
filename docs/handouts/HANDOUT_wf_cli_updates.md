# Task IMPL-036-H/I/J: CLI and Output Updates for Walk-Forward

## Files to Modify
- `scripts/run_backtests.py`
- `scripts/run_optimization.py` (if exists, or create)

## Purpose
Update CLI scripts to support walk-forward evaluation mode with appropriate command-line arguments and output formatting.

---

## Changes to `scripts/run_backtests.py`

### 1. Add New CLI Arguments

```python
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run backtests for all strategy configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Walk-forward mode (default)
    python scripts/run_backtests.py --snapshot data/snapshots/latest

    # Walk-forward with custom windows
    python scripts/run_backtests.py --snapshot data/snapshots/latest \\
        --train-years 2 --test-years 1 --step-months 12

    # Legacy multi-period mode
    python scripts/run_backtests.py --snapshot data/snapshots/latest \\
        --no-walk-forward --periods 3,5,10
""",
    )

    # EXISTING ARGS (keep as-is)
    parser.add_argument(
        '--snapshot',
        type=str,
        required=True,
        help='Path to data snapshot directory',
    )

    parser.add_argument(
        '--output',
        type=str,
        default='artifacts/optimization',
        help='Output directory for results (default: artifacts/optimization)',
    )

    parser.add_argument(
        '--max-configs',
        type=int,
        default=None,
        help='Maximum number of configs to test (for debugging)',
    )

    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)',
    )

    parser.add_argument(
        '--cost-bps',
        type=float,
        default=10.0,
        help='Transaction cost in basis points (default: 10.0)',
    )

    parser.add_argument(
        '--schedules',
        type=str,
        default=None,
        help='Comma-separated list of schedules to test (default: all)',
    )

    parser.add_argument(
        '--alpha-types',
        type=str,
        default=None,
        help='Comma-separated list of alpha types to test (default: all)',
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging',
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Just count configs without running',
    )

    # NEW: Walk-forward arguments
    wf_group = parser.add_argument_group('Walk-Forward Options')

    wf_group.add_argument(
        '--walk-forward',
        action='store_true',
        default=True,
        dest='walk_forward',
        help='Use walk-forward validation (default: True)',
    )

    wf_group.add_argument(
        '--no-walk-forward',
        action='store_false',
        dest='walk_forward',
        help='Use legacy multi-period evaluation instead of walk-forward',
    )

    wf_group.add_argument(
        '--train-years',
        type=int,
        default=3,
        help='Training window size in years (default: 3)',
    )

    wf_group.add_argument(
        '--test-years',
        type=int,
        default=1,
        help='Test window size in years (default: 1)',
    )

    wf_group.add_argument(
        '--step-months',
        type=int,
        default=6,
        help='Window step size in months (default: 6)',
    )

    wf_group.add_argument(
        '--min-windows',
        type=int,
        default=4,
        help='Minimum walk-forward windows required (default: 4)',
    )

    # MODIFIED: Deprecate --periods for walk-forward mode
    legacy_group = parser.add_argument_group('Legacy Options (deprecated)')

    legacy_group.add_argument(
        '--periods',
        type=str,
        default=None,
        help='DEPRECATED: Use --no-walk-forward with this option. '
             'Comma-separated evaluation periods in years.',
    )

    return parser.parse_args()
```

### 2. Update main() Function

```python
def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    # Handle deprecation warning for --periods with walk-forward
    if args.walk_forward and args.periods:
        import warnings
        warnings.warn(
            "--periods is ignored when using walk-forward mode. "
            "Use --no-walk-forward if you want multi-period evaluation.",
            DeprecationWarning,
        )
        logger.warning("--periods ignored in walk-forward mode")

    # Parse evaluation periods (for legacy mode)
    periods = None
    if not args.walk_forward:
        if args.periods:
            try:
                periods = [int(p.strip()) for p in args.periods.split(',')]
            except ValueError:
                logger.error(f"Invalid periods format: {args.periods}")
                return 1
        else:
            periods = [3, 5, 10]  # Default legacy periods
        logger.info(f"Legacy mode: evaluation periods = {periods} years")

    # Log walk-forward config
    if args.walk_forward:
        logger.info(
            f"Walk-forward mode: train={args.train_years}yr, "
            f"test={args.test_years}yr, step={args.step_months}mo"
        )

    # ... existing snapshot validation ...

    # Create walk-forward config if using walk-forward mode
    wf_config = None
    if args.walk_forward:
        from quantetf.optimization.walk_forward_evaluator import WalkForwardEvaluatorConfig
        wf_config = WalkForwardEvaluatorConfig(
            train_years=args.train_years,
            test_years=args.test_years,
            step_months=args.step_months,
            min_windows=args.min_windows,
        )

    # ... existing config counting ...

    try:
        data_access = DataAccessFactory.create_context(
            config={"snapshot_path": str(snapshot_path)},
            enable_caching=True,
        )

        optimizer = StrategyOptimizer(
            data_access=data_access,
            output_dir=args.output,
            # Walk-forward mode
            use_walk_forward=args.walk_forward,
            wf_config=wf_config,
            # Legacy mode
            periods_years=periods,
            # Common
            max_workers=args.parallel,
            cost_bps=args.cost_bps,
            regime_analysis_enabled=False,
        )

        # Run optimization
        result = optimizer.run(
            max_configs=args.max_configs,
            schedules=schedule_names,
            alpha_types=alpha_types,
        )

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=args.verbose)
        return 1

    # Print summary (mode-specific)
    print_summary(result, args.walk_forward)

    return 0
```

### 3. Add Mode-Specific Summary Printing

```python
def print_summary(result: 'OptimizationResult', walk_forward: bool) -> None:
    """Print optimization summary based on mode."""
    print("\n" + "=" * 80)

    if walk_forward:
        print("WALK-FORWARD OPTIMIZATION RESULTS")
        print("=" * 80)

        # Get WF config from first result if available
        if result.winners:
            first = result.winners[0]
            print(f"\nWalk-Forward Config:")
            print(f"  Training Period:  {first.num_windows} windows evaluated")

        print(f"\nStrategies Evaluated:     {result.successful_configs}")
        print(f"Strategies Passed OOS:    {len(result.winners)}")
        filtered = result.successful_configs - len(result.winners)
        print(f"Strategies Filtered:      {filtered} (negative OOS active return)")

        if result.winners:
            print("\nTOP 10 STRATEGIES (by OOS composite score):")
            print("-" * 80)
            print(f"{'Rank':<5} {'Strategy':<45} {'OOS_Sharpe':<12} {'OOS_Active':<12} {'Win%':<8}")
            print("-" * 80)

            for i, r in enumerate(result.winners[:10], 1):
                name = r.config_name[:43]
                print(
                    f"{i:<5} {name:<45} "
                    f"{r.oos_sharpe_mean:<12.2f} "
                    f"{r.oos_active_return_mean:+11.1%} "
                    f"{r.oos_win_rate:<8.0%}"
                )

            # Degradation analysis
            print("\nDEGRADATION ANALYSIS (IS vs OOS):")
            print("-" * 80)
            print(f"{'Strategy':<45} {'IS_Sharpe':<12} {'OOS_Sharpe':<12} {'Degradation':<12}")
            print("-" * 80)

            for r in result.winners[:5]:
                name = r.config_name[:43]
                print(
                    f"{name:<45} "
                    f"{r.is_sharpe_mean:<12.2f} "
                    f"{r.oos_sharpe_mean:<12.2f} "
                    f"{r.sharpe_degradation:+11.2f}"
                )

    else:
        # Legacy multi-period summary
        print("MULTI-PERIOD OPTIMIZATION RESULTS")
        print("=" * 80)
        print(f"\nStrategies Evaluated:     {result.successful_configs}")
        print(f"Strategies Beat SPY:      {len(result.winners)}")

        if result.winners:
            print("\nTOP 10 STRATEGIES (by composite score):")
            print("-" * 80)
            for i, r in enumerate(result.winners[:10], 1):
                print(f"  {i}. {r.config_name}")
                print(f"     Score: {r.composite_score:.3f}")

    print("\n" + "=" * 80)
    print(f"Output saved to: {result.output_dir}")
    print("=" * 80)
```

---

## Output Files

### Walk-Forward Mode Outputs

When `--walk-forward` (default):

```
artifacts/optimization/run_20260201_143022/
├── all_results.csv           # All strategies with OOS metrics
├── winners.csv               # Strategies with positive OOS active return
├── best_strategy.yaml        # Top strategy configuration
├── optimization_report.md    # Human-readable report
├── walk_forward_details.csv  # Per-window results for top strategies
└── backtest_results.pkl      # Full results with daily returns
```

### CSV Column Order for Walk-Forward

```csv
config_name,composite_score,num_windows,oos_sharpe_mean,oos_sharpe_std,oos_return_mean,oos_active_return_mean,oos_win_rate,is_sharpe_mean,is_return_mean,sharpe_degradation,return_degradation
```

OOS columns should come BEFORE IS columns to emphasize out-of-sample metrics.

---

## Testing

```python
"""Tests for CLI with walk-forward support."""
import pytest
import subprocess
import tempfile
from pathlib import Path


class TestCLIWalkForward:
    """Tests for walk-forward CLI arguments."""

    def test_default_walk_forward(self):
        """Test that walk-forward is the default mode."""
        from scripts.run_backtests import parse_args
        import sys

        # Simulate CLI args
        sys.argv = ['run_backtests.py', '--snapshot', '/tmp/fake']
        args = parse_args()

        assert args.walk_forward is True

    def test_no_walk_forward_flag(self):
        """Test --no-walk-forward disables walk-forward."""
        from scripts.run_backtests import parse_args
        import sys

        sys.argv = ['run_backtests.py', '--snapshot', '/tmp/fake', '--no-walk-forward']
        args = parse_args()

        assert args.walk_forward is False

    def test_wf_config_args(self):
        """Test walk-forward configuration arguments."""
        from scripts.run_backtests import parse_args
        import sys

        sys.argv = [
            'run_backtests.py',
            '--snapshot', '/tmp/fake',
            '--train-years', '2',
            '--test-years', '2',
            '--step-months', '12',
            '--min-windows', '3',
        ]
        args = parse_args()

        assert args.train_years == 2
        assert args.test_years == 2
        assert args.step_months == 12
        assert args.min_windows == 3

    def test_periods_warning_with_walk_forward(self, capsys):
        """Test deprecation warning when using --periods with walk-forward."""
        from scripts.run_backtests import parse_args
        import sys
        import warnings

        sys.argv = [
            'run_backtests.py',
            '--snapshot', '/tmp/fake',
            '--periods', '3,5,10',  # Should be ignored
        ]
        args = parse_args()

        # walk_forward should still be True
        assert args.walk_forward is True
        # periods should be set but ignored
        assert args.periods == '3,5,10'

    def test_legacy_mode_with_periods(self):
        """Test legacy mode with --periods."""
        from scripts.run_backtests import parse_args
        import sys

        sys.argv = [
            'run_backtests.py',
            '--snapshot', '/tmp/fake',
            '--no-walk-forward',
            '--periods', '1,3,5',
        ]
        args = parse_args()

        assert args.walk_forward is False
        assert args.periods == '1,3,5'

    def test_dry_run_shows_mode(self, capsys):
        """Test dry run shows evaluation mode."""
        # This would require mocking or a real integration test
        pass


class TestCLIHelp:
    """Tests for CLI help output."""

    def test_help_shows_walk_forward_options(self):
        """Test that --help shows walk-forward options."""
        result = subprocess.run(
            ['python', 'scripts/run_backtests.py', '--help'],
            capture_output=True,
            text=True,
        )

        assert '--train-years' in result.stdout
        assert '--test-years' in result.stdout
        assert '--step-months' in result.stdout
        assert '--no-walk-forward' in result.stdout
```

---

## Example Usage

```bash
# Default: Walk-forward with 3yr train, 1yr test, 6mo step
python scripts/run_backtests.py --snapshot data/snapshots/latest

# Custom walk-forward windows
python scripts/run_backtests.py --snapshot data/snapshots/latest \
    --train-years 2 --test-years 1 --step-months 12

# Quick test with limited configs
python scripts/run_backtests.py --snapshot data/snapshots/latest \
    --max-configs 20 --schedules monthly

# Legacy multi-period mode
python scripts/run_backtests.py --snapshot data/snapshots/latest \
    --no-walk-forward --periods 3,5,10

# Dry run to see config counts
python scripts/run_backtests.py --snapshot data/snapshots/latest --dry-run
```

---

## Acceptance Checklist

- [ ] `--walk-forward` is default (True)
- [ ] `--no-walk-forward` disables walk-forward mode
- [ ] `--train-years` sets training window size
- [ ] `--test-years` sets test window size
- [ ] `--step-months` sets window step size
- [ ] `--min-windows` sets minimum required windows
- [ ] Deprecation warning when using `--periods` with walk-forward
- [ ] Help text shows walk-forward options clearly
- [ ] Console output shows walk-forward config when in that mode
- [ ] Console output shows OOS metrics table
- [ ] Console output shows degradation analysis
- [ ] CSV output has OOS columns first
- [ ] All tests pass

---

## Dependencies

- `quantetf.optimization.optimizer` (updated in IMPL-036-E)
- `quantetf.optimization.walk_forward_evaluator` (from IMPL-036-A/B)

---

## Next Task

After completing this task, proceed to **IMPL-036-K/L/M**: Validation and Documentation.
