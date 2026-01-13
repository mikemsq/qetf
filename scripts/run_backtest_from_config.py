#!/usr/bin/env python3
"""Run a backtest from a YAML strategy configuration file.

This script loads a strategy configuration from YAML and runs a complete
backtest using the configured alpha model, portfolio construction, and
cost model.

Examples:
    # Run a single strategy from config
    $ python scripts/run_backtest_from_config.py \\
        --config configs/strategies/momentum_acceleration_top5.yaml

    # Run with custom date range and snapshot
    $ python scripts/run_backtest_from_config.py \\
        --config configs/strategies/residual_momentum_top5.yaml \\
        --snapshot data/snapshots/snapshot_5yr_20etfs \\
        --start 2022-01-01 \\
        --end 2024-12-31

    # Run with custom initial capital
    $ python scripts/run_backtest_from_config.py \\
        --config configs/strategies/vol_adjusted_momentum_top5.yaml \\
        --capital 250000
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import yaml
import pandas as pd

from quantetf.config.loader import load_strategy_config
from quantetf.backtest.simple_engine import SimpleBacktestEngine, BacktestConfig
from quantetf.data.snapshot_store import SnapshotDataStore

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run backtest from strategy configuration file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single strategy
  python scripts/run_backtest_from_config.py \\
    --config configs/strategies/momentum_acceleration_top5.yaml

  # Custom date range
  python scripts/run_backtest_from_config.py \\
    --config configs/strategies/residual_momentum_top5.yaml \\
    --start 2022-01-01 --end 2024-12-31

  # Custom snapshot and capital
  python scripts/run_backtest_from_config.py \\
    --config configs/strategies/vol_adjusted_momentum_top5.yaml \\
    --snapshot data/snapshots/snapshot_5yr_20etfs \\
    --capital 250000
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to strategy config YAML file'
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
        '--output-dir',
        type=str,
        default='artifacts/backtests',
        help='Output directory for results (default: artifacts/backtests)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def run_backtest(args, strategy_config, output_dir):
    """Run the backtest with given strategy configuration.

    Args:
        args: Parsed command-line arguments
        strategy_config: Loaded StrategyConfig object
        output_dir: Path to output directory (already created)

    Returns:
        BacktestResult object containing equity curve, holdings, metrics
    """
    logger.info("=" * 80)
    logger.info("QuantETF Backtest from Config")
    logger.info("=" * 80)
    logger.info(f"Strategy: {strategy_config.name}")
    logger.info(f"Alpha Model: {type(strategy_config.alpha_model).__name__}")
    logger.info(f"Portfolio: {type(strategy_config.portfolio_construction).__name__}")

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

    # 2. Create universe from config
    logger.info(f"Universe: {len(strategy_config.universe_tickers)} ETFs")
    logger.info(f"Tickers: {', '.join(strategy_config.universe_tickers)}")

    universe = strategy_config.create_universe(as_of=pd.Timestamp(args.end))

    # 3. Configure backtest
    config = BacktestConfig(
        start_date=pd.Timestamp(args.start),
        end_date=pd.Timestamp(args.end),
        universe=universe,
        initial_capital=args.capital,
        rebalance_frequency=strategy_config.rebalance_frequency
    )

    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Initial capital: ${args.capital:,.2f}")
    logger.info(f"Rebalance frequency: {strategy_config.rebalance_frequency}")

    # 4. Get components from strategy config
    alpha_model = strategy_config.alpha_model
    portfolio = strategy_config.portfolio_construction
    cost_model = strategy_config.cost_model

    logger.info(f"Cost model: {cost_model.cost_bps} bps per trade")

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
    save_results(result, args, strategy_config, output_dir)
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


def save_results(result, args, strategy_config, output_dir):
    """Save backtest results to disk.

    Args:
        result: BacktestResult object
        args: Parsed command-line arguments
        strategy_config: StrategyConfig object
        output_dir: Path to output directory
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

    # Save config (merge strategy config with run parameters)
    config_path = output_dir / 'config.json'
    config_dict = {
        'strategy_name': strategy_config.name,
        'config_file': str(args.config),
        'snapshot_dir': str(args.snapshot),
        'start_date': str(result.config.start_date),
        'end_date': str(result.config.end_date),
        'initial_capital': result.config.initial_capital,
        'rebalance_frequency': result.config.rebalance_frequency,
        'universe': list(result.config.universe.tickers),
        'alpha_model': {
            'type': type(strategy_config.alpha_model).__name__,
            'config': strategy_config.raw_config.get('alpha_model', {})
        },
        'portfolio_construction': {
            'type': type(strategy_config.portfolio_construction).__name__,
            'config': strategy_config.raw_config.get('portfolio_construction', {})
        },
        'cost_bps': strategy_config.cost_model.cost_bps,
        'description': strategy_config.description
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Saved config: {config_path}")


def main():
    """Main entry point."""
    args = parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Load strategy configuration
        logger.info(f"Loading strategy config: {args.config}")
        strategy_config = load_strategy_config(args.config)

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_output_dir = Path(args.output_dir)
        base_output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = base_output_dir / f"{timestamp}_{strategy_config.name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Add file handler for logging
        log_file = output_dir / 'out.txt'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(file_handler)

        logger.info(f"Output directory: {output_dir}")

        # Run backtest
        result = run_backtest(args, strategy_config, output_dir)

        logger.info("SUCCESS!")
        return 0

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
