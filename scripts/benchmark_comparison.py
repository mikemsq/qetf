#!/usr/bin/env python3
"""Compare a strategy against standard benchmarks.

This script runs a strategy backtest alongside multiple benchmark strategies
and generates a comprehensive comparison report with performance attribution.

Example:
    # Compare strategy from config file
    $ python scripts/benchmark_comparison.py \\
        --strategy artifacts/backtests/momentum-ew-top5_20260110_092345/results.pkl \\
        --snapshot data/snapshots/snapshot_5yr_20etfs \\
        --output artifacts/benchmark_comparisons/

    # Run all benchmarks on a date range
    $ python scripts/benchmark_comparison.py \\
        --strategy artifacts/backtests/my_strategy/results.pkl \\
        --snapshot data/snapshots/snapshot_5yr_20etfs \\
        --start 2021-01-01 \\
        --end 2023-12-31 \\
        --benchmarks spy 60_40 equal_weight random oracle
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import pickle
import json
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from quantetf.backtest.simple_engine import BacktestConfig, BacktestResult
from quantetf.data.access import DataAccessFactory
from quantetf.evaluation import benchmarks
from quantetf.evaluation import metrics
from quantetf.evaluation import comparison
from quantetf.types import Universe

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare strategy against benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare with all benchmarks
  python scripts/benchmark_comparison.py \\
      --strategy artifacts/backtests/momentum-ew-top5_latest/results.pkl \\
      --snapshot data/snapshots/snapshot_5yr_20etfs

  # Compare with selected benchmarks
  python scripts/benchmark_comparison.py \\
      --strategy artifacts/backtests/my_strategy/results.pkl \\
      --snapshot data/snapshots/snapshot_5yr_20etfs \\
      --benchmarks spy equal_weight oracle
        """
    )

    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        help='Path to strategy backtest results.pkl file'
    )

    parser.add_argument(
        '--snapshot',
        type=str,
        default='data/snapshots/snapshot_latest',
        help='Path to snapshot directory'
    )

    parser.add_argument(
        '--start',
        type=str,
        default=None,
        help='Start date (YYYY-MM-DD), overrides strategy start'
    )

    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD), overrides strategy end'
    )

    parser.add_argument(
        '--benchmarks',
        nargs='+',
        choices=['spy', '60_40', 'equal_weight', 'random', 'oracle'],
        default=['spy', '60_40', 'equal_weight', 'random', 'oracle'],
        help='Which benchmarks to run (default: all)'
    )

    parser.add_argument(
        '--random-trials',
        type=int,
        default=100,
        help='Number of Monte Carlo trials for random benchmark (default: 100)'
    )

    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='artifacts/benchmark_comparisons',
        help='Output directory for comparison results'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )

    return parser.parse_args()


def load_strategy_result(path: str) -> BacktestResult:
    """Load strategy backtest result from pickle file."""
    with open(path, 'rb') as f:
        result = pickle.load(f)

    if not isinstance(result, BacktestResult):
        raise ValueError(f"Expected BacktestResult, got {type(result)}")

    return result


def run_benchmarks(
    strategy_result: BacktestResult,
    data_access,
    benchmark_names: list[str],
    random_trials: int,
    random_seed: int
) -> dict[str, benchmarks.BenchmarkResult]:
    """Run all requested benchmarks."""
    results = {}

    config = strategy_result.config

    logger.info("=" * 80)
    logger.info("Running Benchmarks")
    logger.info("=" * 80)

    if 'spy' in benchmark_names:
        logger.info("Running SPY benchmark...")
        results['SPY Buy-and-Hold'] = benchmarks.run_spy_benchmark(
            config=config,
            data_access=data_access
        )

    if '60_40' in benchmark_names:
        logger.info("Running 60/40 benchmark...")
        results['60/40 Portfolio'] = benchmarks.run_60_40_benchmark(
            config=config,
            data_access=data_access
        )

    if 'equal_weight' in benchmark_names:
        logger.info("Running equal weight benchmark...")
        results['Equal Weight Universe'] = benchmarks.run_equal_weight_benchmark(
            config=config,
            data_access=data_access,
            rebalance_frequency=config.rebalance_frequency
        )

    if 'random' in benchmark_names:
        logger.info(f"Running random selection benchmark ({random_trials} trials)...")
        # Determine n_selections from strategy weights
        strategy_weights = strategy_result.weights_history
        avg_positions = (strategy_weights > 0).sum(axis=1).mean()
        n_selections = max(1, int(round(avg_positions)))

        results[f'Random Selection (N={n_selections})'] = benchmarks.run_random_selection_benchmark(
            config=config,
            data_access=data_access,
            n_selections=n_selections,
            n_trials=random_trials,
            rebalance_frequency=config.rebalance_frequency,
            seed=random_seed
        )

    if 'oracle' in benchmark_names:
        logger.info("Running oracle benchmark...")
        # Use same n_selections as random
        strategy_weights = strategy_result.weights_history
        avg_positions = (strategy_weights > 0).sum(axis=1).mean()
        n_selections = max(1, int(round(avg_positions)))

        results[f'Oracle (N={n_selections})'] = benchmarks.run_oracle_benchmark(
            config=config,
            data_access=data_access,
            n_selections=n_selections,
            rebalance_frequency=config.rebalance_frequency
        )

    logger.info(f"Completed {len(results)} benchmarks")
    return results


def calculate_comparison_metrics(
    strategy_result: BacktestResult,
    benchmark_results: dict[str, benchmarks.BenchmarkResult]
) -> pd.DataFrame:
    """Calculate comparison metrics for strategy vs benchmarks."""

    all_metrics = []

    # Strategy metrics
    strategy_returns = strategy_result.equity_curve['nav'].pct_change().dropna()
    strategy_metrics = {
        'Name': 'Strategy',
        'Total Return': strategy_result.metrics.get('total_return', 0.0),
        'CAGR': metrics.cagr(strategy_result.equity_curve['nav']),
        'Sharpe Ratio': metrics.sharpe(strategy_returns),
        'Sortino Ratio': metrics.sortino_ratio(strategy_returns),
        'Max Drawdown': metrics.max_drawdown(strategy_result.equity_curve['nav']),
        'Calmar Ratio': metrics.calmar_ratio(strategy_returns),
        'Win Rate': metrics.win_rate(strategy_returns),
        'VaR (95%)': metrics.value_at_risk(strategy_returns),
        'CVaR (95%)': metrics.conditional_value_at_risk(strategy_returns),
    }
    all_metrics.append(strategy_metrics)

    # Benchmark metrics
    for name, bench_result in benchmark_results.items():
        bench_returns = bench_result.equity_curve['nav'].pct_change().dropna()

        if len(bench_returns) == 0:
            continue

        bench_metrics = {
            'Name': name,
            'Total Return': bench_result.metrics.get('total_return', 0.0),
            'CAGR': metrics.cagr(bench_result.equity_curve['nav']),
            'Sharpe Ratio': metrics.sharpe(bench_returns),
            'Sortino Ratio': metrics.sortino_ratio(bench_returns),
            'Max Drawdown': metrics.max_drawdown(bench_result.equity_curve['nav']),
            'Calmar Ratio': metrics.calmar_ratio(bench_returns),
            'Win Rate': metrics.win_rate(bench_returns),
            'VaR (95%)': metrics.value_at_risk(bench_returns),
            'CVaR (95%)': metrics.conditional_value_at_risk(bench_returns),
        }
        all_metrics.append(bench_metrics)

    return pd.DataFrame(all_metrics)


def calculate_attribution_metrics(
    strategy_result: BacktestResult,
    benchmark_results: dict[str, benchmarks.BenchmarkResult]
) -> pd.DataFrame:
    """Calculate attribution metrics (excess return, tracking error, IR, beta, alpha)."""

    attribution = []

    strategy_returns = strategy_result.equity_curve['nav'].pct_change().dropna()

    for name, bench_result in benchmark_results.items():
        bench_returns = bench_result.equity_curve['nav'].pct_change().dropna()

        if len(bench_returns) == 0:
            continue

        # Align returns
        aligned_strategy, aligned_bench = strategy_returns.align(bench_returns, join='inner')

        if len(aligned_strategy) < 2:
            continue

        # Excess return
        excess_returns = aligned_strategy - aligned_bench
        excess_return_ann = excess_returns.mean() * 252

        # Tracking error
        tracking_error = excess_returns.std() * np.sqrt(252)

        # Information ratio
        ir = metrics.information_ratio(aligned_strategy, aligned_bench)

        # Beta and alpha (regression)
        if len(aligned_bench) > 1 and aligned_bench.std() > 0:
            covariance = np.cov(aligned_strategy, aligned_bench)[0, 1]
            variance = np.var(aligned_bench)
            beta = covariance / variance if variance > 0 else 0.0

            # Jensen's alpha
            alpha_ann = (aligned_strategy.mean() - beta * aligned_bench.mean()) * 252
        else:
            beta = 0.0
            alpha_ann = 0.0

        attribution.append({
            'Benchmark': name,
            'Excess Return': excess_return_ann,
            'Tracking Error': tracking_error,
            'Information Ratio': ir,
            'Beta': beta,
            'Alpha': alpha_ann,
        })

    return pd.DataFrame(attribution)


def generate_plots(
    strategy_result: BacktestResult,
    benchmark_results: dict[str, benchmarks.BenchmarkResult],
    output_dir: Path
):
    """Generate comparison plots."""

    # 1. Equity curves overlay
    fig, ax = plt.subplots(figsize=(12, 6))

    # Normalize to $1 starting value
    strategy_nav = strategy_result.equity_curve['nav']
    norm_strategy = strategy_nav / strategy_nav.iloc[0]
    ax.plot(norm_strategy.index, norm_strategy.values,
            label='Strategy', linewidth=2, color='black')

    colors = sns.color_palette('husl', n_colors=len(benchmark_results))
    for (name, bench_result), color in zip(benchmark_results.items(), colors):
        bench_nav = bench_result.equity_curve['nav']
        norm_bench = bench_nav / bench_nav.iloc[0]
        ax.plot(norm_bench.index, norm_bench.values,
                label=name, linewidth=1.5, alpha=0.7, color=color)

    ax.set_title('Strategy vs Benchmarks (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity (Normalized to $1)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curves_comparison.png', dpi=150)
    plt.close()

    logger.info(f"Saved equity curves to {output_dir / 'equity_curves_comparison.png'}")

    # 2. Risk-return scatter
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate risk/return for each
    points = []

    strategy_returns = strategy_result.equity_curve['nav'].pct_change().dropna()
    strategy_vol = strategy_returns.std() * np.sqrt(252)
    strategy_ret = metrics.cagr(strategy_result.equity_curve['nav'])
    points.append({
        'name': 'Strategy',
        'volatility': strategy_vol,
        'return': strategy_ret,
        'color': 'black',
        'size': 200
    })

    colors = sns.color_palette('husl', n_colors=len(benchmark_results))
    for (name, bench_result), color in zip(benchmark_results.items(), colors):
        bench_returns = bench_result.equity_curve['nav'].pct_change().dropna()
        if len(bench_returns) == 0:
            continue
        bench_vol = bench_returns.std() * np.sqrt(252)
        bench_ret = metrics.cagr(bench_result.equity_curve['nav'])
        points.append({
            'name': name,
            'volatility': bench_vol,
            'return': bench_ret,
            'color': color,
            'size': 150
        })

    for point in points:
        ax.scatter(point['volatility'], point['return'],
                  color=point['color'], s=point['size'], alpha=0.7,
                  label=point['name'], edgecolors='black', linewidths=1)

    ax.set_title('Risk-Return Scatter', fontsize=14, fontweight='bold')
    ax.set_xlabel('Volatility (Annualized)')
    ax.set_ylabel('CAGR')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'risk_return_scatter.png', dpi=150)
    plt.close()

    logger.info(f"Saved risk-return scatter to {output_dir / 'risk_return_scatter.png'}")


def generate_html_report(
    strategy_result: BacktestResult,
    benchmark_results: dict[str, benchmarks.BenchmarkResult],
    metrics_df: pd.DataFrame,
    attribution_df: pd.DataFrame,
    output_dir: Path
):
    """Generate HTML report."""

    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th {{ background-color: #4CAF50; color: white; padding: 12px; text-align: left; }}
        td {{ border: 1px solid #ddd; padding: 8px; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .highlight {{ background-color: #ffffcc; font-weight: bold; }}
        img {{ max-width: 100%; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Benchmark Comparison Report</h1>
    <p>Generated: {timestamp}</p>

    <h2>Performance Metrics</h2>
    {metrics_table}

    <h2>Attribution Analysis</h2>
    {attribution_table}

    <h2>Equity Curves</h2>
    <img src="equity_curves_comparison.png" alt="Equity Curves">

    <h2>Risk-Return Profile</h2>
    <img src="risk_return_scatter.png" alt="Risk-Return Scatter">
</body>
</html>
"""

    # Format tables
    metrics_html = metrics_df.to_html(index=False, float_format='%.4f')
    attribution_html = attribution_df.to_html(index=False, float_format='%.4f')

    # Fill template
    html = html.format(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        metrics_table=metrics_html,
        attribution_table=attribution_html
    )

    # Write to file
    report_path = output_dir / 'benchmark_comparison_report.html'
    with open(report_path, 'w') as f:
        f.write(html)

    logger.info(f"Saved HTML report to {report_path}")


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load strategy result
    logger.info(f"Loading strategy result from {args.strategy}")
    strategy_result = load_strategy_result(args.strategy)

    # Load snapshot data
    logger.info(f"Loading snapshot data from {args.snapshot}")
    snapshot_path = Path(args.snapshot)
    if snapshot_path.is_dir():
        data_path = snapshot_path / 'data.parquet'
    else:
        data_path = snapshot_path

    # Create DataAccessContext
    data_access = DataAccessFactory.create_context(
        config={"snapshot_path": str(data_path)},
        enable_caching=True
    )

    # Run benchmarks
    benchmark_results = run_benchmarks(
        strategy_result=strategy_result,
        data_access=data_access,
        benchmark_names=args.benchmarks,
        random_trials=args.random_trials,
        random_seed=args.random_seed
    )

    # Calculate metrics
    logger.info("Calculating comparison metrics...")
    metrics_df = calculate_comparison_metrics(strategy_result, benchmark_results)
    attribution_df = calculate_attribution_metrics(strategy_result, benchmark_results)

    # Print to console
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("ATTRIBUTION ANALYSIS")
    print("=" * 80)
    print(attribution_df.to_string(index=False))

    # Save metrics
    metrics_df.to_csv(output_dir / 'performance_metrics.csv', index=False)
    attribution_df.to_csv(output_dir / 'attribution_metrics.csv', index=False)
    logger.info(f"Saved metrics to {output_dir}")

    # Generate plots
    if not args.no_plots:
        logger.info("Generating plots...")
        generate_plots(strategy_result, benchmark_results, output_dir)

        # Generate HTML report
        logger.info("Generating HTML report...")
        generate_html_report(
            strategy_result, benchmark_results,
            metrics_df, attribution_df, output_dir
        )

    logger.info("=" * 80)
    logger.info("Benchmark comparison complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
