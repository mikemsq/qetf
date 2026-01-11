#!/usr/bin/env python3
"""Compare multiple strategy backtests and generate comparative analysis.

This script loads multiple backtest results, computes comparative metrics,
performs statistical significance tests, and generates comprehensive reports
with charts and tables.

Examples:
    # Compare all backtests in a directory
    $ python scripts/compare_strategies.py \\
        --backtest-dirs artifacts/backtests/*/ \\
        --output artifacts/comparisons/latest

    # Compare specific strategies
    $ python scripts/compare_strategies.py \\
        --backtest-dirs \\
            artifacts/backtests/20260111_023934_momentum-ew-top5 \\
            artifacts/backtests/20260111_024110_momentum-ew-top3 \\
        --output artifacts/comparisons/momentum_comparison \\
        --report-name momentum_3v5

    # Run new backtests and compare them
    $ python scripts/compare_strategies.py \\
        --configs configs/strategies/*.yaml \\
        --snapshot data/snapshots/snapshot_5yr_20etfs \\
        --output artifacts/comparisons/config_sweep
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List

from quantetf.evaluation.comparison import (
    load_backtest_result,
    compute_comparison_metrics,
    compute_returns_correlation,
    sharpe_ratio_ttest,
    create_equity_overlay_chart,
    create_risk_return_scatter,
    generate_comparison_report,
    StrategyResult
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare multiple strategy backtests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare existing backtests
  python scripts/compare_strategies.py \\
    --backtest-dirs artifacts/backtests/20260111_*/  \\
    --output artifacts/comparisons/latest

  # Compare with custom report name
  python scripts/compare_strategies.py \\
    --backtest-dirs artifacts/backtests/momentum_* \\
    --output artifacts/comparisons/ \\
    --report-name momentum_comparison

  # Run new backtests from configs (future enhancement)
  python scripts/compare_strategies.py \\
    --configs configs/strategies/*.yaml \\
    --snapshot data/snapshots/snapshot_5yr_20etfs \\
    --output artifacts/comparisons/config_sweep
        """
    )

    parser.add_argument(
        '--backtest-dirs',
        nargs='+',
        type=str,
        help='Paths to backtest result directories to compare'
    )

    parser.add_argument(
        '--configs',
        nargs='+',
        type=str,
        help='Strategy config files to run and compare (future feature)'
    )

    parser.add_argument(
        '--snapshot',
        type=str,
        default='data/snapshots/snapshot_5yr_20etfs',
        help='Snapshot directory for running new backtests (used with --configs)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for comparison results'
    )

    parser.add_argument(
        '--report-name',
        type=str,
        default='comparison_report',
        help='Base name for report files (default: comparison_report)'
    )

    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='Display plots interactively (default: save only)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def load_all_backtests(backtest_dirs: List[str]) -> List[StrategyResult]:
    """Load all backtest results from directories.

    Args:
        backtest_dirs: List of directory paths

    Returns:
        List of StrategyResult objects

    Raises:
        ValueError: If no valid backtests are found
    """
    results = []
    failed_loads = []

    for dir_path in backtest_dirs:
        try:
            result = load_backtest_result(dir_path)
            results.append(result)
            logger.info(f"Loaded backtest: {result.name} from {dir_path}")
        except Exception as e:
            logger.warning(f"Failed to load backtest from {dir_path}: {e}")
            failed_loads.append(dir_path)

    if not results:
        raise ValueError(f"No valid backtests loaded. Failed: {failed_loads}")

    if failed_loads:
        logger.warning(f"Failed to load {len(failed_loads)} backtest(s)")

    return results


def print_summary_table(comparison_df):
    """Print formatted comparison table to console.

    Args:
        comparison_df: DataFrame from compute_comparison_metrics()
    """
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 80)

    # Select key metrics for console display
    display_cols = [
        'total_return', 'cagr', 'sharpe_ratio', 'sortino_ratio',
        'max_drawdown', 'volatility', 'win_rate', 'num_rebalances'
    ]

    available_cols = [col for col in display_cols if col in comparison_df.columns]
    display_df = comparison_df[available_cols].copy()

    # Format percentages
    pct_cols = ['total_return', 'cagr', 'max_drawdown', 'volatility', 'win_rate']
    for col in pct_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f'{x:.2%}')

    # Format ratios
    ratio_cols = ['sharpe_ratio', 'sortino_ratio']
    for col in ratio_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f'{x:.3f}')

    # Format integers
    if 'num_rebalances' in display_df.columns:
        display_df['num_rebalances'] = display_df['num_rebalances'].apply(lambda x: f'{int(x)}')

    print(display_df.to_string())
    print("=" * 80 + "\n")


def print_correlation_matrix(corr_df):
    """Print correlation matrix to console.

    Args:
        corr_df: Correlation DataFrame
    """
    print("\n" + "=" * 80)
    print("STRATEGY RETURNS CORRELATION MATRIX")
    print("=" * 80)
    print(corr_df.to_string(float_format=lambda x: f'{x:.3f}'))
    print("=" * 80 + "\n")


def print_significance_tests(results: List[StrategyResult]):
    """Print pairwise Sharpe ratio t-tests.

    Args:
        results: List of StrategyResult objects
    """
    if len(results) < 2:
        return

    print("\n" + "=" * 80)
    print("SHARPE RATIO SIGNIFICANCE TESTS (Pairwise)")
    print("=" * 80)

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            test = sharpe_ratio_ttest(results[i], results[j])

            sig_marker = "***" if test['is_significant'] else ""
            print(f"\n{results[i].name} vs {results[j].name}:")

            if 'sharpe_1' in test and 'sharpe_2' in test:
                print(f"  Sharpe ratios: {test['sharpe_1']:.3f} vs {test['sharpe_2']:.3f}")
                print(f"  t-statistic: {test['t_statistic']:.3f}")
                print(f"  p-value: {test['p_value']:.4f} {sig_marker}")
                print(f"  Significant at 5%: {test['is_significant']}")
            else:
                print(f"  {test.get('message', 'Unable to perform test')}")

    print("\n*** = Significant at 5% level")
    print("=" * 80 + "\n")


def main():
    """Main execution function."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    if not args.backtest_dirs and not args.configs:
        logger.error("Must provide either --backtest-dirs or --configs")
        sys.exit(1)

    if args.configs:
        logger.error("--configs option not yet implemented. Use --backtest-dirs to compare existing backtests.")
        sys.exit(1)

    # Expand glob patterns in backtest directories
    backtest_paths = []
    for pattern in args.backtest_dirs:
        expanded = list(Path('.').glob(pattern))
        if expanded:
            backtest_paths.extend([str(p) for p in expanded if p.is_dir()])
        else:
            # Try as direct path
            p = Path(pattern)
            if p.is_dir():
                backtest_paths.append(str(p))

    if not backtest_paths:
        logger.error(f"No backtest directories found matching: {args.backtest_dirs}")
        sys.exit(1)

    logger.info(f"Found {len(backtest_paths)} backtest directories")

    # Load all backtests
    try:
        results = load_all_backtests(backtest_paths)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Successfully loaded {len(results)} backtest(s)")

    if len(results) < 2:
        logger.warning("Only 1 backtest loaded. Comparison requires at least 2.")
        logger.info("Generating single-strategy report...")

    # Compute comparison metrics
    logger.info("Computing comparison metrics...")
    comparison_df = compute_comparison_metrics(results)

    # Print summary to console
    print_summary_table(comparison_df)

    # Compute and print correlation matrix
    if len(results) > 1:
        logger.info("Computing returns correlation...")
        corr_df = compute_returns_correlation(results)
        print_correlation_matrix(corr_df)

        # Print significance tests
        logger.info("Performing Sharpe ratio significance tests...")
        print_significance_tests(results)

    # Generate comprehensive report
    logger.info(f"Generating comparison report in {args.output}...")
    output_paths = generate_comparison_report(
        results,
        args.output,
        args.report_name
    )

    # Print output file paths
    print("\n" + "=" * 80)
    print("GENERATED FILES")
    print("=" * 80)
    for file_type, path in output_paths.items():
        print(f"{file_type:20s}: {path}")
    print("=" * 80 + "\n")

    logger.info("Comparison complete!")

    # Show plots if requested
    if args.show_plots:
        import matplotlib.pyplot as plt
        logger.info("Displaying plots...")
        plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
