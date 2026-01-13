#!/usr/bin/env python3
"""Run multiple strategy configs and compare results.

This script runs backtests for multiple strategy configurations and
automatically generates a comparative analysis report.

Examples:
    # Compare all Phase 1 momentum strategies
    $ python scripts/run_strategy_sweep.py \\
        --configs configs/strategies/momentum_*.yaml configs/strategies/*residual*.yaml \\
        --output artifacts/comparisons/phase1_momentum

    # Compare with custom date range
    $ python scripts/run_strategy_sweep.py \\
        --configs configs/strategies/*.yaml \\
        --start 2022-01-01 \\
        --end 2024-12-31 \\
        --output artifacts/comparisons/recent_period

    # Quick run with verbose logging
    $ python scripts/run_strategy_sweep.py \\
        --configs configs/strategies/momentum_acceleration_top5.yaml \\
                  configs/strategies/vol_adjusted_momentum_top5.yaml \\
        --verbose
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List
import glob

import pandas as pd

from quantetf.config.loader import load_strategy_config
from quantetf.backtest.simple_engine import SimpleBacktestEngine, BacktestConfig
from quantetf.data.snapshot_store import SnapshotDataStore
from quantetf.evaluation.comparison import (
    load_backtest_result,
    compute_comparison_metrics,
    compute_returns_correlation,
    sharpe_ratio_ttest,
    generate_comparison_report,
    StrategyResult
)
from quantetf.evaluation.benchmarks import run_spy_benchmark
from quantetf.types import Universe

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run multiple strategy configs and compare results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare Phase 1 momentum strategies
  python scripts/run_strategy_sweep.py \\
    --configs configs/strategies/momentum_*.yaml

  # Custom date range and output
  python scripts/run_strategy_sweep.py \\
    --configs configs/strategies/*.yaml \\
    --start 2022-01-01 --end 2024-12-31 \\
    --output artifacts/comparisons/recent

  # Specific configs
  python scripts/run_strategy_sweep.py \\
    --configs \\
      configs/strategies/momentum_acceleration_top5.yaml \\
      configs/strategies/residual_momentum_top5.yaml \\
    --output artifacts/comparisons/momentum_vs_residual
        """
    )

    parser.add_argument(
        '--configs',
        nargs='+',
        required=True,
        help='Strategy config files to run (supports glob patterns)'
    )

    parser.add_argument(
        '--snapshot',
        type=str,
        default='data/snapshots/snapshot_5yr_20etfs',
        help='Path to snapshot directory (default: data/snapshots/snapshot_5yr_20etfs)'
    )

    parser.add_argument(
        '--start',
        type=str,
        default='2021-01-01',
        help='Backtest start date (YYYY-MM-DD, default: 2021-01-01)'
    )

    parser.add_argument(
        '--end',
        type=str,
        default='2025-12-31',
        help='Backtest end date (YYYY-MM-DD, default: 2025-12-31)'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='Initial capital in dollars (default: 100000.0)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for comparison results (default: artifacts/comparisons/TIMESTAMP)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--no-spy-benchmark',
        action='store_true',
        help='Disable automatic SPY benchmark comparison'
    )

    return parser.parse_args()


def expand_config_patterns(patterns: List[str]) -> List[Path]:
    """Expand glob patterns to list of config file paths.

    Args:
        patterns: List of file paths or glob patterns

    Returns:
        List of resolved Path objects

    Raises:
        ValueError: If no config files are found
    """
    config_paths = []

    for pattern in patterns:
        # Try glob expansion
        matches = glob.glob(pattern)

        if matches:
            config_paths.extend([Path(m) for m in matches if Path(m).is_file()])
        else:
            # Try as direct path
            p = Path(pattern)
            if p.is_file():
                config_paths.append(p)
            else:
                logger.warning(f"No files found matching: {pattern}")

    if not config_paths:
        raise ValueError(f"No config files found matching patterns: {patterns}")

    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for p in config_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    return unique_paths


def run_single_backtest(strategy_config, args, store, backtest_dir):
    """Run backtest for a single strategy config.

    Args:
        strategy_config: StrategyConfig object
        args: Command-line arguments
        store: SnapshotDataStore object
        backtest_dir: Directory to save results

    Returns:
        BacktestResult object
    """
    logger.info(f"Running backtest: {strategy_config.name}")

    # Create universe
    universe = strategy_config.create_universe(as_of=pd.Timestamp(args.end))

    # Configure backtest
    config = BacktestConfig(
        start_date=pd.Timestamp(args.start),
        end_date=pd.Timestamp(args.end),
        universe=universe,
        initial_capital=args.capital,
        rebalance_frequency=strategy_config.rebalance_frequency
    )

    # Run backtest
    engine = SimpleBacktestEngine()

    result = engine.run(
        config=config,
        alpha_model=strategy_config.alpha_model,
        portfolio=strategy_config.portfolio_construction,
        cost_model=strategy_config.cost_model,
        store=store
    )

    # Save results
    backtest_dir.mkdir(parents=True, exist_ok=True)

    result.equity_curve.to_csv(backtest_dir / 'equity_curve.csv')
    result.holdings_history.to_csv(backtest_dir / 'holdings_history.csv')
    result.weights_history.to_csv(backtest_dir / 'weights_history.csv')

    import json
    with open(backtest_dir / 'metrics.json', 'w') as f:
        json.dump(result.metrics, f, indent=2, default=str)

    with open(backtest_dir / 'config.json', 'w') as f:
        config_dict = {
            'strategy_name': strategy_config.name,
            'snapshot_dir': str(args.snapshot),
            'start_date': str(config.start_date),
            'end_date': str(config.end_date),
            'initial_capital': config.initial_capital,
            'rebalance_frequency': config.rebalance_frequency,
            'universe': list(universe.tickers),
            'alpha_model': strategy_config.raw_config.get('alpha_model', {}),
            'portfolio_construction': strategy_config.raw_config.get('portfolio_construction', {}),
            'cost_bps': strategy_config.cost_model.cost_bps,
        }
        json.dump(config_dict, f, indent=2)

    logger.info(f"  Sharpe: {result.metrics['sharpe_ratio']:.2f}, "
                f"Return: {result.metrics['total_return']:.2%}, "
                f"Max DD: {result.metrics['max_drawdown']:.2%}")

    return result


def add_spy_benchmark(store, args) -> StrategyResult:
    """Run SPY benchmark and return as StrategyResult.

    Args:
        store: SnapshotDataStore object
        args: Command-line arguments

    Returns:
        StrategyResult for SPY benchmark
    """
    logger.info("Running SPY benchmark...")

    dummy_universe = Universe(
        as_of=pd.Timestamp(args.start),
        tickers=('SPY',)
    )

    spy_config = BacktestConfig(
        start_date=pd.Timestamp(args.start),
        end_date=pd.Timestamp(args.end),
        universe=dummy_universe,
        initial_capital=args.capital,
        rebalance_frequency='monthly'
    )

    spy_result = run_spy_benchmark(config=spy_config, store=store)

    return StrategyResult(
        name='SPY Benchmark',
        backtest_dir=Path('benchmark_spy'),
        equity_curve=spy_result.equity_curve['nav'],
        weights_history=pd.DataFrame(),
        holdings_history=pd.DataFrame(),
        metrics=spy_result.metrics,
        config={'description': 'SPY buy-and-hold benchmark'}
    )


def main():
    """Main entry point."""
    args = parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    try:
        # Expand config patterns
        logger.info("Expanding config patterns...")
        config_paths = expand_config_patterns(args.configs)
        logger.info(f"Found {len(config_paths)} config files")

        # Load snapshot
        logger.info(f"Loading snapshot: {args.snapshot}")
        snapshot_path = Path(args.snapshot)
        if snapshot_path.is_dir():
            data_path = snapshot_path / 'data.parquet'
        else:
            data_path = snapshot_path

        if not data_path.exists():
            raise FileNotFoundError(f"Snapshot data not found: {data_path}")

        store = SnapshotDataStore(data_path)

        # Create output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path('artifacts/comparisons') / timestamp

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Run backtests for each config
        logger.info(f"\nRunning {len(config_paths)} backtests...")
        logger.info("=" * 80)

        backtest_dirs = []

        for i, config_path in enumerate(config_paths, 1):
            logger.info(f"\n[{i}/{len(config_paths)}] Loading config: {config_path.name}")

            # Load strategy config
            strategy_config = load_strategy_config(config_path)

            # Create backtest subdirectory
            backtest_dir = output_dir / 'backtests' / strategy_config.name

            # Run backtest
            result = run_single_backtest(strategy_config, args, store, backtest_dir)

            backtest_dirs.append(str(backtest_dir))

        logger.info("\n" + "=" * 80)
        logger.info("All backtests complete!")

        # Load results for comparison
        logger.info("\nLoading backtest results for comparison...")
        results = []
        for backtest_dir in backtest_dirs:
            result = load_backtest_result(backtest_dir)
            results.append(result)

        # Add SPY benchmark
        if not args.no_spy_benchmark:
            spy_result = add_spy_benchmark(store, args)
            results.append(spy_result)

        # Generate comparison report
        logger.info("\nGenerating comparison report...")
        comparison_metrics = compute_comparison_metrics(results)

        # Print summary table
        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON SUMMARY")
        print("=" * 80)

        display_cols = ['total_return', 'cagr', 'sharpe_ratio', 'max_drawdown', 'volatility']
        available_cols = [col for col in display_cols if col in comparison_metrics.columns]
        display_df = comparison_metrics[available_cols].copy()

        # Format
        for col in ['total_return', 'cagr', 'max_drawdown', 'volatility']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f'{x:.2%}')

        if 'sharpe_ratio' in display_df.columns:
            display_df['sharpe_ratio'] = display_df['sharpe_ratio'].apply(lambda x: f'{x:.3f}')

        print(display_df.to_string())
        print("=" * 80 + "\n")

        # Generate full report
        report_paths = generate_comparison_report(
            results,
            str(output_dir),
            'strategy_comparison'
        )

        print("\n" + "=" * 80)
        print("GENERATED FILES")
        print("=" * 80)
        for file_type, path in report_paths.items():
            print(f"{file_type:20s}: {path}")
        print("=" * 80 + "\n")

        logger.info(f"\nComparison complete! Results saved to: {output_dir}")

        return 0

    except Exception as e:
        logger.error(f"Strategy sweep failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
