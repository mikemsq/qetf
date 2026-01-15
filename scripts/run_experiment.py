#!/usr/bin/env python3
"""Run strategy experiments with configurable alpha models.

This script runs backtests using any registered alpha model and saves
comprehensive results for analysis.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantetf.alpha.factory import create_alpha_model, AlphaModelRegistry
from quantetf.backtest.simple_engine import SimpleBacktestEngine
from quantetf.data.snapshot_store import SnapshotDataStore
from quantetf.evaluation.metrics import (
    cagr, max_drawdown, sharpe, sortino_ratio, calmar_ratio,
    win_rate, value_at_risk, conditional_value_at_risk, information_ratio,
    calculate_active_metrics
)
from quantetf.portfolio.equal_weight import EqualWeightTopN
from quantetf.portfolio.costs import FlatTransactionCost
from quantetf.types import Universe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run strategy experiments with configurable alpha models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run trend-filtered momentum (EXP-002)
  python scripts/run_experiment.py \\
      --snapshot data/snapshots/snapshot_20260115_170559 \\
      --alpha-type trend_filtered_momentum \\
      --experiment exp002_trend_filtered

  # Run dual momentum
  python scripts/run_experiment.py \\
      --snapshot data/snapshots/snapshot_20260115_170559 \\
      --alpha-type dual_momentum \\
      --alpha-params '{"momentum_lookback": 252, "absolute_threshold": 0.0}'

  # List available alpha models
  python scripts/run_experiment.py --list-models
        """
    )

    parser.add_argument('--snapshot', type=str, required=False,
                        help='Path to snapshot directory')
    parser.add_argument('--alpha-type', type=str, default='momentum',
                        help='Alpha model type (default: momentum)')
    parser.add_argument('--alpha-params', type=str, default='{}',
                        help='JSON string of alpha model parameters')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of ETFs to hold (default: 5)')
    parser.add_argument('--cost-bps', type=float, default=10.0,
                        help='Transaction cost in basis points (default: 10)')
    parser.add_argument('--rebalance', choices=['monthly', 'weekly'], default='monthly',
                        help='Rebalance frequency (default: monthly)')
    parser.add_argument('--start', type=str, default=None,
                        help='Start date YYYY-MM-DD (default: use all data)')
    parser.add_argument('--end', type=str, default=None,
                        help='End date YYYY-MM-DD (default: use all data)')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Experiment name for output directory')
    parser.add_argument('--output-dir', type=str, default='artifacts/experiments',
                        help='Base output directory (default: artifacts/experiments)')
    parser.add_argument('--list-models', action='store_true',
                        help='List available alpha models and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable debug logging')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # List models and exit if requested
    if args.list_models:
        print("Available alpha models:")
        for model in AlphaModelRegistry.list_models():
            print(f"  - {model}")
        return 0

    if not args.snapshot:
        print("ERROR: --snapshot is required")
        return 1

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        print(f"ERROR: Snapshot not found: {snapshot_path}")
        return 1

    # Parse alpha params
    try:
        alpha_params = json.loads(args.alpha_params)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON for --alpha-params: {e}")
        return 1

    # Create alpha model
    alpha_config = {'type': args.alpha_type, **alpha_params}
    try:
        alpha_model = create_alpha_model(alpha_config)
        logger.info(f"Created alpha model: {type(alpha_model).__name__}")
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    # Load data - handle both directory and direct parquet file paths
    if snapshot_path.is_dir():
        parquet_path = snapshot_path / 'data.parquet'
    else:
        parquet_path = snapshot_path

    logger.info(f"Loading snapshot from {parquet_path}")
    store = SnapshotDataStore(parquet_path)

    # Get date range and tickers from data
    all_dates = store._data.index
    start_date = pd.Timestamp(args.start) if args.start else all_dates[0]
    end_date = pd.Timestamp(args.end) if args.end else all_dates[-1]

    logger.info(f"Backtest period: {start_date.date()} to {end_date.date()}")

    # Get tickers
    tickers = store._data.columns.get_level_values('Ticker').unique().tolist()
    logger.info(f"Universe: {len(tickers)} tickers")

    # Import BacktestConfig
    from quantetf.backtest.simple_engine import BacktestConfig

    # Create components
    portfolio_constructor = EqualWeightTopN(top_n=args.top_n)
    cost_model = FlatTransactionCost(cost_bps=args.cost_bps)

    # Create universe and config
    universe = Universe(as_of=start_date, tickers=tuple(tickers))
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        universe=universe,
        initial_capital=100000.0,
        rebalance_frequency=args.rebalance,
    )

    # Create engine and run backtest
    logger.info("Running backtest...")
    engine = SimpleBacktestEngine()

    result = engine.run(
        config=config,
        alpha_model=alpha_model,
        portfolio=portfolio_constructor,
        cost_model=cost_model,
        store=store,
    )

    # Calculate metrics - extract NAV column from equity curve DataFrame
    equity = result.equity_curve['nav'] if isinstance(result.equity_curve, pd.DataFrame) else result.equity_curve
    returns = equity.pct_change().dropna()
    metrics = {
        'total_return': (equity.iloc[-1] / equity.iloc[0]) - 1,
        'cagr': cagr(equity),
        'sharpe_ratio': sharpe(returns),
        'sortino_ratio': sortino_ratio(returns),
        'max_drawdown': max_drawdown(equity),
        'calmar_ratio': calmar_ratio(equity),
        'volatility': returns.std() * (252 ** 0.5),
        'win_rate': win_rate(returns),
        'var_95': value_at_risk(returns, confidence_level=0.95),
        'cvar_95': conditional_value_at_risk(returns, confidence_level=0.95),
    }

    # Get SPY returns for comparison
    try:
        spy_prices = store.get_close_prices(as_of=end_date, tickers=['SPY'], lookback_days=2600)
        spy_prices = spy_prices['SPY'].loc[start_date:end_date]
        spy_returns = spy_prices.pct_change().dropna()
        # Align returns
        common_dates = returns.index.intersection(spy_returns.index)
        aligned_returns = returns.loc[common_dates]
        aligned_spy = spy_returns.loc[common_dates]
        active_metrics = calculate_active_metrics(aligned_returns, aligned_spy)
        has_spy = True
    except Exception as e:
        logger.warning(f"Could not calculate active metrics: {e}")
        active_metrics = {}
        has_spy = False

    # Print results
    print("\n" + "="*60)
    print(f"EXPERIMENT RESULTS: {args.alpha_type}")
    print("="*60)

    print(f"\nBacktest Period: {start_date.date()} to {end_date.date()}")
    print(f"Alpha Model: {args.alpha_type}")
    print(f"Top N: {args.top_n}")
    print(f"Rebalance: {args.rebalance}")
    print(f"Cost: {args.cost_bps} bps")

    print("\n--- Portfolio Metrics ---")
    print(f"Total Return: {metrics.get('total_return', 0)*100:.1f}%")
    print(f"CAGR: {metrics.get('cagr', 0)*100:.1f}%")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.1f}%")
    print(f"Volatility: {metrics.get('volatility', 0)*100:.1f}%")

    if has_spy:
        print("\n--- vs SPY Benchmark ---")
        print(f"SPY Total Return: {active_metrics.get('benchmark_total_return', 0)*100:.1f}%")
        print(f"Active Return: {active_metrics.get('active_return', 0)*100:.1f}%")
        print(f"Information Ratio: {active_metrics.get('information_ratio', 0):.2f}")
        print(f"Tracking Error: {active_metrics.get('tracking_error', 0)*100:.1f}%")

        beat_spy = active_metrics.get('active_return', 0) > 0
        print(f"\nBEAT SPY: {'YES' if beat_spy else 'NO'}")

    # Save results
    experiment_name = args.experiment or f"{args.alpha_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save equity curve
    result.equity_curve.to_csv(output_dir / 'equity_curve.csv')

    # Save metrics
    all_metrics = {**metrics, **active_metrics}
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # Save config
    config = {
        'alpha_type': args.alpha_type,
        'alpha_params': alpha_params,
        'top_n': args.top_n,
        'cost_bps': args.cost_bps,
        'rebalance': args.rebalance,
        'start_date': str(start_date.date()),
        'end_date': str(end_date.date()),
        'snapshot': str(snapshot_path),
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save holdings history
    if hasattr(result, 'holdings_history') and result.holdings_history is not None and not result.holdings_history.empty:
        holdings_df = pd.DataFrame(result.holdings_history)
        holdings_df.to_csv(output_dir / 'holdings.csv', index=False)

    print(f"\nResults saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
