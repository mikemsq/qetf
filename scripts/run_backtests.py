#!/usr/bin/env python3
"""Run backtests for all strategy configurations and save results.

This script runs backtests without applying ranking or regime analysis.
Results are saved to a pickle file that preserves daily returns for later
analysis steps.

Example:
    $ python scripts/run_backtests.py --snapshot data/snapshots/snapshot_latest

    $ python scripts/run_backtests.py --snapshot data/snapshots/snapshot_latest \
        --output artifacts/optimization/my_run --periods 1,3 --max-configs 20
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from quantetf.data.access import DataAccessFactory
from quantetf.optimization.grid import count_configs, get_alpha_types, get_schedule_names
from quantetf.optimization.optimizer import StrategyOptimizer

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
    )

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
        '--periods',
        type=str,
        default='1,3',
        help='Comma-separated evaluation periods in years (default: 1,3)',
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


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    # Parse evaluation periods
    try:
        periods = [int(p.strip()) for p in args.periods.split(',')]
    except ValueError:
        logger.error(f"Invalid periods format: {args.periods}")
        return 1

    logger.info(f"Evaluation periods: {periods} years")

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
        print("\nDry run complete.")
        return 0

    if counts['total'] == 0:
        logger.error("No configurations to test")
        return 1

    # Run backtests
    print(f"\nRunning backtests with {counts['total']} configurations...")
    if args.max_configs:
        print(f"(Limited to {args.max_configs} for testing)")

    try:
        data_access = DataAccessFactory.create_context(
            config={"snapshot_path": str(snapshot_path)},
            enable_caching=True,
        )

        optimizer = StrategyOptimizer(
            data_access=data_access,
            output_dir=args.output,
            periods_years=periods,
            max_workers=args.parallel,
            cost_bps=args.cost_bps,
            regime_analysis_enabled=False,
        )

        results, total_configs, failed, configs = optimizer.run_backtests(
            max_configs=args.max_configs,
            schedule_names=schedule_names,
            alpha_types=alpha_types,
        )

        # Save results to pickle (includes daily returns)
        results_path = optimizer.run_dir / 'backtest_results.pkl'
        optimizer.save_backtest_results(results, results_path)

        # Also save CSV for human readability
        df = pd.DataFrame([r.to_dict() for r in results])
        df.to_csv(optimizer.run_dir / 'backtest_results.csv', index=False)

    except Exception as e:
        logger.error(f"Backtest run failed: {e}", exc_info=args.verbose)
        return 1

    # Print summary
    winners = [r for r in results if r.beats_spy_all_periods]

    print("\n" + "=" * 60)
    print("BACKTEST RUN COMPLETE")
    print("=" * 60)
    print(f"Total configs tested:     {len(results)}")
    print(f"Failed configs:           {failed}")
    print(f"Strategies that beat SPY: {len(winners)}")
    print(f"\nResults saved to: {optimizer.run_dir}")
    print(f"  - backtest_results.pkl")
    print(f"  - backtest_results.csv")
    print("=" * 60)

    # Print the results path for shell script to capture
    print(f"\nRESULTS_FILE={results_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
