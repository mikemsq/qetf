#!/usr/bin/env python3
"""Run backtests for all strategy configurations and save results.

This script supports two evaluation modes:
1. Walk-forward validation (default): Evaluates strategies on out-of-sample data
   across multiple rolling windows to prevent overfitting.
2. Legacy multi-period: Evaluates on fixed historical periods (deprecated).

Example:
    # Walk-forward mode (default)
    $ python scripts/run_backtests.py --snapshot data/snapshots/snapshot_latest

    # Walk-forward with custom windows
    $ python scripts/run_backtests.py --snapshot data/snapshots/snapshot_latest \
        --train-years 2 --test-years 1 --step-months 12

    # Legacy multi-period mode
    $ python scripts/run_backtests.py --snapshot data/snapshots/snapshot_latest \
        --no-walk-forward --periods 3,5,10

    # Quick test with limited configs
    $ python scripts/run_backtests.py --snapshot data/snapshots/snapshot_latest \
        --max-configs 20 --schedules monthly
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

from quantetf.data.access import DataAccessFactory
from quantetf.optimization.grid import count_configs, get_alpha_types, get_schedule_names
from quantetf.optimization.optimizer import StrategyOptimizer
from quantetf.optimization.walk_forward_evaluator import WalkForwardEvaluatorConfig

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


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

    # Required arguments
    parser.add_argument(
        '--snapshot',
        type=str,
        required=True,
        help='Path to data snapshot directory',
    )

    # Common arguments
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

    # Walk-forward arguments
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

    # Legacy arguments
    legacy_group = parser.add_argument_group('Legacy Options (deprecated)')

    legacy_group.add_argument(
        '--periods',
        type=str,
        default=None,
        help='DEPRECATED: Use --no-walk-forward with this option. '
             'Comma-separated evaluation periods in years.',
    )

    return parser.parse_args()


def print_config_summary(counts: dict) -> None:
    """Print configuration count summary."""
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)

    for schedule in ['weekly', 'monthly']:
        if schedule not in counts:
            continue
        print(f"\n{schedule.upper()} schedule:")
        for alpha_type, count in counts[schedule].items():
            if alpha_type == 'subtotal':
                print(f"  {'Subtotal':<30} {count:>6}")
            else:
                print(f"  {alpha_type:<30} {count:>6}")

    print(f"\n{'TOTAL CONFIGURATIONS':<32} {counts['total']:>6}")
    print("=" * 60)


def print_walk_forward_summary(result, wf_config: WalkForwardEvaluatorConfig) -> None:
    """Print walk-forward optimization summary."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD OPTIMIZATION RESULTS")
    print("=" * 80)

    print(f"\nWalk-Forward Config:")
    print(f"  Train Period: {wf_config.train_years} years")
    print(f"  Test Period:  {wf_config.test_years} years")
    print(f"  Step Size:    {wf_config.step_months} months")
    if result.winners:
        print(f"  Windows:      {result.winners[0].num_windows}")

    print(f"\nStrategies Evaluated:    {result.successful_configs:,}")
    print(f"Strategies Passed OOS:   {len(result.winners):>5}  "
          f"({100 * len(result.winners) / max(result.successful_configs, 1):.1f}%)")
    filtered = result.successful_configs - len(result.winners)
    print(f"Strategies Filtered:     {filtered:>5}  (negative OOS active return)")

    if result.winners:
        print("\nTOP 10 STRATEGIES (by OOS composite score):")
        print("-" * 80)
        print(f"{'Rank':<5} {'Strategy':<45} {'OOS_Sharpe':<12} {'OOS_Active':<12} {'Win%':<8}")
        print("-" * 80)

        for i, winner in enumerate(result.winners[:10], 1):
            name = winner.config_name[:43]
            print(
                f"{i:<5} {name:<45} "
                f"{winner.oos_sharpe_mean:<12.2f} "
                f"{winner.oos_active_return_mean:+11.1%} "
                f"{winner.oos_win_rate:<8.0%}"
            )

        print("\nDEGRADATION ANALYSIS (IS vs OOS):")
        print("-" * 80)
        print(f"{'Strategy':<45} {'IS_Sharpe':<12} {'OOS_Sharpe':<12} {'Degradation':<12}")
        print("-" * 80)

        for winner in result.winners[:5]:
            name = winner.config_name[:43]
            print(
                f"{name:<45} "
                f"{winner.is_sharpe_mean:<12.2f} "
                f"{winner.oos_sharpe_mean:<12.2f} "
                f"{winner.sharpe_degradation:+11.2f}"
            )

    print("\n" + "=" * 80)
    print(f"Output saved to: {result.output_dir}")
    print("=" * 80)


def print_legacy_summary(result, winners) -> None:
    """Print legacy multi-period summary."""
    print("\n" + "=" * 60)
    print("BACKTEST RUN COMPLETE")
    print("=" * 60)
    print(f"Total configs tested:     {result.successful_configs}")
    print(f"Failed configs:           {result.failed_configs}")
    print(f"Strategies that beat SPY: {len(winners)}")
    print(f"\nResults saved to: {result.output_dir}")
    print("=" * 60)


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    # Handle deprecation warning for --periods with walk-forward
    if args.walk_forward and args.periods:
        warnings.warn(
            "--periods is ignored when using walk-forward mode. "
            "Use --no-walk-forward if you want multi-period evaluation.",
            DeprecationWarning,
            stacklevel=2,
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

    # Validate snapshot
    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        logger.error(f"Snapshot not found: {snapshot_path}")
        return 1
    if snapshot_path.is_dir():
        data_file = snapshot_path / 'data.parquet'
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            return 1
        snapshot_path = data_file

    # Parse optional filters
    schedule_names = [s.strip() for s in args.schedules.split(',')] if args.schedules else None
    alpha_types = [a.strip() for a in args.alpha_types.split(',')] if args.alpha_types else None

    # Validate filters
    if schedule_names:
        valid_schedules = get_schedule_names()
        for s in schedule_names:
            if s not in valid_schedules:
                logger.error(f"Invalid schedule: {s}. Valid: {valid_schedules}")
                return 1

    if alpha_types:
        valid_alphas = get_alpha_types()
        for a in alpha_types:
            if a not in valid_alphas:
                logger.error(f"Invalid alpha type: {a}. Valid: {valid_alphas}")
                return 1

    # Count and show configurations
    counts = count_configs(schedule_names=schedule_names, alpha_types=alpha_types)
    print_config_summary(counts)

    if args.dry_run:
        if args.walk_forward:
            print(f"\nWalk-forward config: train={args.train_years}yr, "
                  f"test={args.test_years}yr, step={args.step_months}mo")
        else:
            print(f"\nLegacy mode: periods={periods}yr")
        print("\nDry run complete.")
        return 0

    if counts['total'] == 0:
        logger.error("No configurations to test")
        return 1

    # Create walk-forward config if using walk-forward mode
    wf_config = None
    if args.walk_forward:
        wf_config = WalkForwardEvaluatorConfig(
            train_years=args.train_years,
            test_years=args.test_years,
            step_months=args.step_months,
            min_windows=args.min_windows,
        )

    # Run optimization
    print(f"\nRunning backtests with {counts['total']} configurations...")
    if args.max_configs:
        print(f"(Limited to {args.max_configs} for testing)")
    if args.walk_forward:
        print(f"Mode: Walk-Forward (train={args.train_years}yr, "
              f"test={args.test_years}yr, step={args.step_months}mo)")
    else:
        print(f"Mode: Legacy Multi-Period ({periods}yr)")

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

        # Run full optimization (includes ranking)
        result = optimizer.run(
            max_configs=args.max_configs,
            schedule_names=schedule_names,
            alpha_types=alpha_types,
        )

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=args.verbose)
        return 1

    # Print mode-specific summary
    if args.walk_forward:
        print_walk_forward_summary(result, wf_config)
    else:
        winners = [r for r in result.all_results if r.beats_spy_all_periods]
        print_legacy_summary(result, winners)

    return 0


if __name__ == '__main__':
    sys.exit(main())
