#!/usr/bin/env python3
"""Find the best strategy that beats SPY across multiple time periods.

This script runs the strategy optimizer to search across all parameter
combinations and find strategies that consistently outperform SPY.

Uses DataAccessContext (DAL) for all data access, enabling:
- Decoupling from specific data storage implementations
- Transparent caching for improved performance

Example:
    # Basic run with defaults
    $ python scripts/find_best_strategy.py \\
        --snapshot data/snapshots/snapshot_20260122_010523

    # Quick test with 20 configs
    $ python scripts/find_best_strategy.py \\
        --snapshot data/snapshots/snapshot_20260122_010523 \\
        --max-configs 20 --verbose

    # Parallel execution with 4 workers
    $ python scripts/find_best_strategy.py \\
        --snapshot data/snapshots/snapshot_20260122_010523 \\
        --parallel 4

    # Custom evaluation periods
    $ python scripts/find_best_strategy.py \\
        --snapshot data/snapshots/snapshot_20260122_010523 \\
        --periods 1,3,5

    # Dry run (just count configs)
    $ python scripts/find_best_strategy.py \\
        --snapshot data/snapshots/snapshot_20260122_010523 \\
        --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

from quantetf.data.access import DataAccessFactory
from quantetf.optimization.optimizer import StrategyOptimizer
from quantetf.optimization.grid import count_configs, get_alpha_types, get_schedule_names

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, log_file: Path = None) -> None:
    """Configure logging for the script.

    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO.
        log_file: Optional path to write logs to file.
    """
    level = logging.DEBUG if verbose else logging.INFO

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='Find strategies that beat SPY across multiple time periods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  python scripts/find_best_strategy.py --snapshot data/snapshots/snapshot_20260122_010523

  # Quick test with limited configs
  python scripts/find_best_strategy.py --snapshot data/snapshots/snapshot_20260122_010523 --max-configs 20 -v

  # Parallel execution
  python scripts/find_best_strategy.py --snapshot data/snapshots/snapshot_20260122_010523 --parallel 4

  # Dry run to count configs
  python scripts/find_best_strategy.py --snapshot data/snapshots/snapshot_20260122_010523 --dry-run
        """,
    )

    parser.add_argument(
        '--snapshot',
        type=str,
        required=True,
        help='Path to data snapshot directory (required)',
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
        help='Number of parallel workers (default: 1 = sequential)',
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
        help='Enable verbose (DEBUG) logging',
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Just count configs without running optimization',
    )

    return parser.parse_args()


def print_config_summary(counts: dict) -> None:
    """Print configuration count summary to console.

    Args:
        counts: Dictionary from count_configs() function.
    """
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
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    setup_logging(args.verbose)

    # Parse evaluation periods
    try:
        periods = [int(p.strip()) for p in args.periods.split(',')]
    except ValueError as e:
        logger.error(f"Invalid periods format: {args.periods}. Use comma-separated integers (e.g., 3,5,10)")
        return 1

    logger.info(f"Evaluation periods: {periods} years")

    # Validate snapshot path exists
    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        logger.error(f"Snapshot not found: {snapshot_path}")
        return 1

    # Resolve snapshot path to data file if directory
    if snapshot_path.is_dir():
        data_file = snapshot_path / 'data.parquet'
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            return 1
        snapshot_path = data_file

    logger.info(f"Snapshot: {snapshot_path}")

    # Parse optional filters
    schedule_names = None
    if args.schedules:
        schedule_names = [s.strip() for s in args.schedules.split(',')]
        valid_schedules = get_schedule_names()
        for s in schedule_names:
            if s not in valid_schedules:
                logger.error(f"Invalid schedule: {s}. Valid options: {valid_schedules}")
                return 1
        logger.info(f"Filtering schedules: {schedule_names}")

    alpha_types = None
    if args.alpha_types:
        alpha_types = [a.strip() for a in args.alpha_types.split(',')]
        valid_alphas = get_alpha_types()
        for a in alpha_types:
            if a not in valid_alphas:
                logger.error(f"Invalid alpha type: {a}. Valid options: {valid_alphas}")
                return 1
        logger.info(f"Filtering alpha types: {alpha_types}")

    # Count configurations
    counts = count_configs(schedule_names=schedule_names, alpha_types=alpha_types)
    print_config_summary(counts)

    if args.dry_run:
        print("\nDry run complete. Use without --dry-run to execute optimization.")
        return 0

    # Validate we have configs to run
    if counts['total'] == 0:
        logger.error("No configurations to test with the given filters")
        return 1

    # Run optimization
    logger.info("Starting optimization...")
    print(f"\nRunning optimization with {counts['total']} configurations...")
    if args.max_configs:
        print(f"(Limited to {args.max_configs} for testing)")

    try:
        # Create DataAccessContext using factory
        logger.info("Creating DataAccessContext...")
        data_access = DataAccessFactory.create_context(
            config={"snapshot_path": str(snapshot_path)},
            enable_caching=True  # Enable caching for sequential execution performance
        )

        optimizer = StrategyOptimizer(
            data_access=data_access,
            output_dir=args.output,
            periods_years=periods,
            max_workers=args.parallel,
            cost_bps=args.cost_bps,
        )

        result = optimizer.run(
            max_configs=args.max_configs,
            schedule_names=schedule_names,
            alpha_types=alpha_types,
        )
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=args.verbose)
        return 1

    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Total configs tested:  {result.successful_configs}")
    print(f"Failed configs:        {result.failed_configs}")
    print(f"Strategies that beat SPY in ALL periods: {len(result.winners)}")

    if result.winners:
        print(f"\nBest strategy: {result.best_config.generate_name()}")
        best = result.winners[0]
        print(f"  Composite score: {best.composite_score:.3f}")

        for period_name, metrics in sorted(best.periods.items()):
            active_pct = metrics.active_return * 100
            print(f"  {period_name}: {active_pct:+.1f}% excess return, IR={metrics.information_ratio:.2f}")

        print(f"\nResults saved to: {optimizer.run_dir}")
        print(f"  - all_results.csv ({len(result.all_results)} strategies)")
        print(f"  - winners.csv ({len(result.winners)} strategies)")
        print(f"  - best_strategy.yaml")
        print(f"  - optimization_report.md")
    else:
        print("\nNo strategy beat SPY in all periods.")
        print(f"Results saved to: {optimizer.run_dir}")
        print(f"  - all_results.csv ({len(result.all_results)} strategies)")
        print(f"  - optimization_report.md")

    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
