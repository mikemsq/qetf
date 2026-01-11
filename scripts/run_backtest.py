#!/usr/bin/env python3
"""Run a backtest on historical data.

This script orchestrates a complete backtest run using the SimpleBacktestEngine.
It loads snapshot data, configures the strategy components, executes the backtest,
and saves comprehensive results to the artifacts directory.

Example:
    # Run with defaults (5-year snapshot, momentum top-5 strategy)
    $ python scripts/run_backtest.py

    # Run with custom parameters
    $ python scripts/run_backtest.py \\
        --snapshot data/snapshots/snapshot_5yr_20etfs \\
        --start 2021-01-01 \\
        --end 2025-12-31 \\
        --strategy momentum-ew-top5 \\
        --top-n 5 \\
        --lookback 252 \\
        --cost-bps 10.0

    # Run with smaller universe
    $ python scripts/run_backtest.py \\
        --top-n 3 \\
        --lookback 126 \\
        --cost-bps 5
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import yaml
import pandas as pd
import sys
import os

from quantetf.backtest.simple_engine import SimpleBacktestEngine, BacktestConfig
from quantetf.alpha.momentum import MomentumAlpha
from quantetf.portfolio.equal_weight import EqualWeightTopN
from quantetf.portfolio.costs import FlatTransactionCost
from quantetf.data.snapshot_store import SnapshotDataStore
from quantetf.types import Universe

# Logging is set up in main() to write to out.txt file
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run ETF backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python scripts/run_backtest.py

  # Run with custom date range
  python scripts/run_backtest.py --start 2022-01-01 --end 2024-12-31

  # Run with different strategy parameters
  python scripts/run_backtest.py --top-n 3 --lookback 126 --cost-bps 5
        """
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
        '--strategy',
        type=str,
        default='momentum-ew-top5',
        help='Strategy name for output directory (default: momentum-ew-top5)'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='Initial capital in dollars (default: 100000.0)'
    )

    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Number of ETFs to hold (default: 5)'
    )

    parser.add_argument(
        '--lookback',
        type=int,
        default=252,
        help='Momentum lookback days (default: 252)'
    )

    parser.add_argument(
        '--cost-bps',
        type=float,
        default=10.0,
        help='Transaction cost in basis points (default: 10.0)'
    )

    parser.add_argument(
        '--rebalance',
        type=str,
        default='monthly',
        choices=['monthly', 'weekly'],
        help='Rebalance frequency (default: monthly)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='artifacts/backtests',
        help='Output directory for results (default: artifacts/backtests)'
    )

    return parser.parse_args()


def run_backtest(args, output_dir):
    """Run the backtest with given parameters.

    Args:
        args: Parsed command-line arguments
        output_dir: Path to output directory (already created)

    Returns:
        BacktestResult object containing equity curve, holdings, metrics
    """
    logger.info("=" * 80)
    logger.info("QuantETF Backtest")
    logger.info("=" * 80)

    # 1. Load snapshot
    logger.info(f"Loading snapshot: {args.snapshot}")
    snapshot_path = Path(args.snapshot)

    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    # Check if path is directory or file
    if snapshot_path.is_dir():
        data_path = snapshot_path / 'data.parquet'
        metadata_path = snapshot_path / 'manifest.yaml'
    else:
        data_path = snapshot_path
        metadata_path = snapshot_path.parent / 'manifest.yaml'

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    store = SnapshotDataStore(data_path)

    # 2. Load universe from snapshot metadata
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
        tickers = tuple(metadata['data_summary']['tickers'])
    else:
        # Fallback to extracting tickers from data
        logger.warning(f"Metadata file not found: {metadata_path}, extracting tickers from data")
        tickers = tuple(store.tickers)

    logger.info(f"Universe: {len(tickers)} ETFs")
    logger.info(f"Tickers: {', '.join(tickers)}")

    universe = Universe(
        as_of=pd.Timestamp(args.end),
        tickers=tickers
    )

    # 3. Configure backtest
    config = BacktestConfig(
        start_date=pd.Timestamp(args.start),
        end_date=pd.Timestamp(args.end),
        universe=universe,
        initial_capital=args.capital,
        rebalance_frequency=args.rebalance
    )

    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Initial capital: ${args.capital:,.2f}")
    logger.info(f"Rebalance frequency: {args.rebalance}")

    # 4. Create components
    alpha_model = MomentumAlpha(lookback_days=args.lookback)
    portfolio = EqualWeightTopN(top_n=args.top_n)
    cost_model = FlatTransactionCost(cost_bps=args.cost_bps)

    logger.info(f"Alpha: {args.lookback}-day momentum")
    logger.info(f"Portfolio: Equal-weight top {args.top_n}")
    logger.info(f"Costs: {args.cost_bps} bps per trade")

    # 5. Run backtest
    logger.info("Running backtest...")
    engine = SimpleBacktestEngine()

    result = engine.run(
        config=config,
        alpha_model=alpha_model,
        portfolio=portfolio,
        cost_model=cost_model,
        store=store
    )

    logger.info("Backtest complete!")

    # 6. Print metrics
    print_metrics(result)

    # 7. Save results
    output_dir = save_results(result, args, output_dir)
    logger.info(f"Results saved to: {output_dir}")

    return result


def print_metrics(result):
    """Print backtest metrics to console.

    Args:
        result: BacktestResult object containing metrics and equity curve
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)

    metrics = result.metrics

    logger.info(f"\nTotal Return:     {metrics['total_return']:>10.2%}")
    logger.info(f"Sharpe Ratio:     {metrics['sharpe_ratio']:>10.2f}")
    logger.info(f"Max Drawdown:     {metrics['max_drawdown']:>10.2%}")
    logger.info(f"Total Costs:      ${metrics['total_costs']:>10,.2f}")
    logger.info(f"Num Rebalances:   {metrics['num_rebalances']:>10,}")

    # Additional metrics
    final_nav = result.equity_curve['nav'].iloc[-1]
    initial_nav = result.config.initial_capital
    logger.info(f"\nInitial NAV:      ${initial_nav:>10,.2f}")
    logger.info(f"Final NAV:        ${final_nav:>10,.2f}")
    logger.info(f"Profit/Loss:      ${final_nav - initial_nav:>10,.2f}")

    logger.info("=" * 80)


def save_results(result, args, output_dir):
    """Save backtest results to disk.

    Saves results to the already-created output directory containing:
    - equity_curve.csv: NAV and costs over time
    - holdings_history.csv: Share holdings at each rebalance
    - weights_history.csv: Portfolio weights at each rebalance
    - metrics.json: Performance metrics
    - config.json: Backtest configuration
    - out.txt: Console and logging output (created in main())

    Args:
        result: BacktestResult object
        args: Parsed command-line arguments
        output_dir: Path to output directory (already created)

    Returns:
        Path object pointing to the output directory
    """

    # Save equity curve
    equity_path = output_dir / 'equity_curve.csv'
    result.equity_curve.to_csv(equity_path)
    logger.info(f"Saved equity curve: {equity_path}")

    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(result.metrics, f, indent=2, default=str)
    logger.info(f"Saved metrics: {metrics_path}")

    # Save holdings history
    holdings_path = output_dir / 'holdings_history.csv'
    result.holdings_history.to_csv(holdings_path)
    logger.info(f"Saved holdings: {holdings_path}")

    # Save weights history
    weights_path = output_dir / 'weights_history.csv'
    result.weights_history.to_csv(weights_path)
    logger.info(f"Saved weights: {weights_path}")

    # Save config
    config_path = output_dir / 'config.json'
    config_dict = {
        'start_date': str(result.config.start_date),
        'end_date': str(result.config.end_date),
        'initial_capital': result.config.initial_capital,
        'rebalance_frequency': result.config.rebalance_frequency,
        'universe': list(result.config.universe.tickers),
        'strategy': args.strategy,
        'top_n': args.top_n,
        'lookback_days': args.lookback,
        'cost_bps': args.cost_bps,
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Saved config: {config_path}")

    return output_dir


def main():
    """Main entry point."""
    args = parse_args()

    # Create output directory early so we can set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = base_output_dir / f"{timestamp}_{args.strategy}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to both console and file
    log_file = output_dir / 'out.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Console output
            logging.FileHandler(log_file)        # File output
        ]
    )
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Starting backtest, output will be saved to: {output_dir}")
        result = run_backtest(args, output_dir)
        logger.info("SUCCESS!")
        return 0

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
