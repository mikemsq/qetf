#!/usr/bin/env python3
"""Run walk-forward validation on strategy.

This script performs rolling window validation to test strategy robustness
and out-of-sample performance. It generates train/test windows, runs backtests
for each window, and produces a comprehensive analysis report.

Walk-forward validation helps detect overfitting by testing the strategy on
data it hasn't seen before, using a rolling window approach.

Example:
    # Run with defaults (2-year train, 1-year test, 6-month step)
    $ python scripts/walk_forward_test.py

    # Run with custom windows
    $ python scripts/walk_forward_test.py \\
        --train-years 3 \\
        --test-years 1 \\
        --step-months 3

    # Run with custom strategy parameters
    $ python scripts/walk_forward_test.py \\
        --top-n 7 \\
        --lookback 126 \\
        --cost-bps 5

    # Save detailed results
    $ python scripts/walk_forward_test.py \\
        --output artifacts/walk_forward/custom_test \\
        --save-plots
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from quantetf.evaluation.walk_forward import (
    WalkForwardConfig,
    run_walk_forward_validation,
    analyze_walk_forward_results,
    create_walk_forward_summary_table,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run walk-forward validation on strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python scripts/walk_forward_test.py

  # Custom train/test windows
  python scripts/walk_forward_test.py --train-years 3 --test-years 1 --step-months 3

  # Custom strategy parameters
  python scripts/walk_forward_test.py --top-n 7 --lookback 126 --cost-bps 5

  # Save detailed output
  python scripts/walk_forward_test.py --output artifacts/walk_forward/test1 --save-plots
        """,
    )

    # Data configuration
    parser.add_argument(
        "--snapshot",
        type=str,
        default="data/snapshots/snapshot_latest",
        help="Path to snapshot directory (default: data/snapshots/snapshot_latest)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Overall start date (YYYY-MM-DD, default: 2020-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-12-31",
        help="Overall end date (YYYY-MM-DD, default: 2025-12-31)",
    )

    # Walk-forward configuration
    parser.add_argument(
        "--train-years",
        type=int,
        default=2,
        help="Training window size in years (default: 2)",
    )
    parser.add_argument(
        "--test-years",
        type=int,
        default=1,
        help="Testing window size in years (default: 1)",
    )
    parser.add_argument(
        "--step-months",
        type=int,
        default=6,
        help="Step size in months between windows (default: 6)",
    )

    # Strategy parameters
    parser.add_argument(
        "--top-n", type=int, default=5, help="Number of top ETFs to hold (default: 5)"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=252,
        help="Momentum lookback in days (default: 252)",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help="Transaction cost in basis points (default: 10.0)",
    )
    parser.add_argument(
        "--rebalance-frequency",
        type=str,
        default="monthly",
        choices=["weekly", "monthly", "quarterly"],
        help="Rebalance frequency (default: monthly)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital in dollars (default: 100000.0)",
    )

    # Output configuration
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: artifacts/walk_forward/YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save visualization plots to output directory",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def create_visualization_plots(
    analysis, output_dir: Path, save_plots: bool = False
) -> None:
    """Create visualization plots for walk-forward results.

    Args:
        analysis: WalkForwardAnalysis object
        output_dir: Directory to save plots
        save_plots: Whether to save plots to disk
    """
    # Set style
    sns.set_style("whitegrid")

    # 1. In-sample vs Out-of-sample Sharpe Ratios
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data
    window_ids = [r.window.window_id for r in analysis.window_results]
    train_sharpes = [r.train_metrics.get("sharpe_ratio", np.nan) for r in analysis.window_results]
    test_sharpes = [r.test_metrics.get("sharpe_ratio", np.nan) for r in analysis.window_results]
    train_returns = [r.train_metrics.get("total_return", np.nan) for r in analysis.window_results]
    test_returns = [r.test_metrics.get("total_return", np.nan) for r in analysis.window_results]

    # Plot 1: Sharpe ratios over windows
    axes[0, 0].plot(window_ids, train_sharpes, marker="o", label="In-Sample", linewidth=2)
    axes[0, 0].plot(window_ids, test_sharpes, marker="s", label="Out-of-Sample", linewidth=2)
    axes[0, 0].axhline(0, color="black", linestyle="--", alpha=0.3)
    axes[0, 0].set_xlabel("Window ID")
    axes[0, 0].set_ylabel("Sharpe Ratio")
    axes[0, 0].set_title("Sharpe Ratio: In-Sample vs Out-of-Sample")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Returns over windows
    axes[0, 1].plot(window_ids, train_returns, marker="o", label="In-Sample", linewidth=2)
    axes[0, 1].plot(window_ids, test_returns, marker="s", label="Out-of-Sample", linewidth=2)
    axes[0, 1].axhline(0, color="black", linestyle="--", alpha=0.3)
    axes[0, 1].set_xlabel("Window ID")
    axes[0, 1].set_ylabel("Total Return")
    axes[0, 1].set_title("Total Return: In-Sample vs Out-of-Sample")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Plot 3: Distribution of OOS Sharpe ratios
    clean_test_sharpes = [s for s in test_sharpes if not np.isnan(s)]
    if clean_test_sharpes:
        axes[1, 0].hist(clean_test_sharpes, bins=10, edgecolor="black", alpha=0.7)
        axes[1, 0].axvline(
            np.mean(clean_test_sharpes),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(clean_test_sharpes):.2f}",
        )
        axes[1, 0].axvline(
            np.median(clean_test_sharpes),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(clean_test_sharpes):.2f}",
        )
        axes[1, 0].set_xlabel("Sharpe Ratio")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Distribution of Out-of-Sample Sharpe Ratios")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Degradation (IS - OOS)
    degradation = [
        (train - test)
        for train, test in zip(train_sharpes, test_sharpes)
        if not (np.isnan(train) or np.isnan(test))
    ]
    if degradation:
        axes[1, 1].bar(range(len(degradation)), degradation, edgecolor="black", alpha=0.7)
        axes[1, 1].axhline(0, color="black", linestyle="-", linewidth=1)
        axes[1, 1].axhline(
            np.mean(degradation),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(degradation):.2f}",
        )
        axes[1, 1].set_xlabel("Window ID")
        axes[1, 1].set_ylabel("Sharpe Degradation (IS - OOS)")
        axes[1, 1].set_title("Performance Degradation by Window")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        plot_path = output_dir / "walk_forward_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {plot_path}")

    plt.close()


def print_summary(analysis) -> None:
    """Print summary statistics to console.

    Args:
        analysis: WalkForwardAnalysis object
    """
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 80)

    # Overall statistics
    print(f"\nNumber of Windows: {analysis.summary_stats['num_windows']}")

    print("\n--- In-Sample Performance ---")
    print(f"  Sharpe Ratio:  {analysis.summary_stats['is_sharpe_mean']:.3f} ± {analysis.summary_stats['is_sharpe_std']:.3f}")
    print(f"  Total Return:  {analysis.summary_stats['is_return_mean']:.2%}")
    print(f"  Max Drawdown:  {analysis.summary_stats['is_drawdown_mean']:.2%}")

    print("\n--- Out-of-Sample Performance ---")
    print(f"  Sharpe Ratio:  {analysis.summary_stats['oos_sharpe_mean']:.3f} ± {analysis.summary_stats['oos_sharpe_std']:.3f}")
    print(f"  Total Return:  {analysis.summary_stats['oos_return_mean']:.2%}")
    print(f"  Max Drawdown:  {analysis.summary_stats['oos_drawdown_mean']:.2%}")

    print("\n--- Degradation Metrics ---")
    print(f"  Sharpe Degradation:        {analysis.degradation_metrics['sharpe_degradation']:.3f}")
    print(f"  Return Degradation:        {analysis.degradation_metrics['return_degradation']:.2%}")
    print(f"  % Windows OOS Positive:    {analysis.degradation_metrics['pct_windows_oos_positive']:.1%}")
    print(f"  % Windows OOS Beats IS:    {analysis.degradation_metrics['pct_windows_oos_beats_is']:.1%}")

    print("\n--- Stability Metrics ---")
    print(f"  IS Sharpe CV:              {analysis.stability_metrics['is_sharpe_cv']:.3f}")
    print(f"  OOS Sharpe CV:             {analysis.stability_metrics['oos_sharpe_cv']:.3f}")
    print(f"  OOS Sharpe Positive %:     {analysis.stability_metrics['oos_sharpe_positive_pct']:.1%}")

    print("\n" + "=" * 80)


def save_results(analysis, output_dir: Path, config: dict) -> None:
    """Save results to output directory.

    Args:
        analysis: WalkForwardAnalysis object
        output_dir: Directory to save results
        config: Configuration dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary statistics
    summary_path = output_dir / "summary.json"
    summary_data = {
        "summary_stats": analysis.summary_stats,
        "degradation_metrics": analysis.degradation_metrics,
        "stability_metrics": analysis.stability_metrics,
        "config": config,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2, default=str)
    logger.info(f"Saved summary to {summary_path}")

    # Save detailed window results
    df = create_walk_forward_summary_table(analysis)
    table_path = output_dir / "window_results.csv"
    df.to_csv(table_path, index=False)
    logger.info(f"Saved window results to {table_path}")

    # Save window details (equity curves for each window)
    for r in analysis.window_results:
        window_dir = output_dir / f"window_{r.window.window_id:02d}"
        window_dir.mkdir(parents=True, exist_ok=True)

        # Save train equity curve
        train_path = window_dir / "train_equity.csv"
        r.train_result.equity_curve.to_csv(train_path)

        # Save test equity curve
        test_path = window_dir / "test_equity.csv"
        r.test_result.equity_curve.to_csv(test_path)

    logger.info(f"Saved individual window data to {output_dir}/window_*/")


def main():
    """Main execution function."""
    args = parse_args()
    setup_logging(args.verbose)

    logger.info("Starting walk-forward validation")
    logger.info(f"Snapshot: {args.snapshot}")
    logger.info(f"Date range: {args.start} to {args.end}")
    logger.info(f"Window config: {args.train_years}y train, {args.test_years}y test, {args.step_months}m step")
    logger.info(f"Strategy: top_n={args.top_n}, lookback={args.lookback}, cost={args.cost_bps}bps")

    # Create walk-forward configuration
    wf_config = WalkForwardConfig(
        train_years=args.train_years,
        test_years=args.test_years,
        step_months=args.step_months,
    )

    # Strategy parameters
    strategy_params = {
        "top_n": args.top_n,
        "lookback_days": args.lookback,
        "cost_bps": args.cost_bps,
        "rebalance_frequency": args.rebalance_frequency,
    }

    # Run walk-forward validation
    # Note: run_walk_forward_validation already uses DataAccessContext internally
    try:
        results = run_walk_forward_validation(
            snapshot_path=args.snapshot,
            start_date=args.start,
            end_date=args.end,
            wf_config=wf_config,
            strategy_params=strategy_params,
            initial_capital=args.capital,
        )
    except Exception as e:
        logger.error(f"Walk-forward validation failed: {e}", exc_info=True)
        sys.exit(1)

    # Analyze results
    logger.info("\nAnalyzing results...")
    analysis = analyze_walk_forward_results(results)

    # Print summary
    print_summary(analysis)

    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("artifacts") / "walk_forward" / timestamp

    # Save results
    config_dict = {
        "snapshot": args.snapshot,
        "start_date": args.start,
        "end_date": args.end,
        "walk_forward_config": {
            "train_years": args.train_years,
            "test_years": args.test_years,
            "step_months": args.step_months,
        },
        "strategy_params": strategy_params,
        "initial_capital": args.capital,
    }

    save_results(analysis, output_dir, config_dict)

    # Create plots
    if args.save_plots:
        logger.info("\nCreating visualization plots...")
        create_visualization_plots(analysis, output_dir, save_plots=True)

    logger.info(f"\nResults saved to {output_dir}")
    logger.info("\nWalk-forward validation complete!")

    # Return exit code based on OOS performance
    oos_sharpe = analysis.summary_stats.get("oos_sharpe_mean", 0)
    if oos_sharpe > 0.5:
        logger.info(f"✓ Strategy shows robust OOS performance (Sharpe={oos_sharpe:.2f})")
        sys.exit(0)
    else:
        logger.warning(f"⚠ Strategy shows weak OOS performance (Sharpe={oos_sharpe:.2f})")
        sys.exit(0)  # Still exit 0 for successful run


if __name__ == "__main__":
    main()
