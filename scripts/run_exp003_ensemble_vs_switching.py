#!/usr/bin/env python3
"""EXP-003: Ensemble vs Switching Strategy Comparison.

This experiment tests whether blending strategies beats regime-switching.

Strategies compared:
1. Pure momentum (baseline)
2. Trend-filtered momentum (switching based on MA200)
3. Ensemble blend (momentum + value + dual momentum)

Hypothesis: Blending strategies reduces variance vs switching.
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

from quantetf.alpha.factory import create_alpha_model
from quantetf.alpha.ensemble import WeightedEnsemble
from quantetf.alpha.momentum import MomentumAlpha
from quantetf.alpha.trend_filtered_momentum import TrendFilteredMomentum
from quantetf.alpha.dual_momentum import DualMomentum
from quantetf.alpha.value_momentum import ValueMomentum
from quantetf.backtest.simple_engine import SimpleBacktestEngine, BacktestConfig
from quantetf.data.access import DataAccessFactory
from quantetf.evaluation.metrics import (
    cagr, max_drawdown, sharpe, sortino_ratio, calmar_ratio,
    win_rate, calculate_active_metrics
)
from quantetf.portfolio.equal_weight import EqualWeightTopN
from quantetf.portfolio.costs import FlatTransactionCost
from quantetf.types import Universe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_backtest(data_access, alpha_model, start_date, end_date, tickers, top_n=5, cost_bps=10.0):
    """Run a single backtest with given alpha model."""
    universe = Universe(as_of=start_date, tickers=tuple(tickers))
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        universe=universe,
        initial_capital=100000.0,
        rebalance_frequency='monthly',
    )

    portfolio_constructor = EqualWeightTopN(top_n=top_n)
    cost_model = FlatTransactionCost(cost_bps=cost_bps)
    engine = SimpleBacktestEngine()

    result = engine.run(
        config=config,
        alpha_model=alpha_model,
        portfolio=portfolio_constructor,
        cost_model=cost_model,
        data_access=data_access,
    )

    return result


def calculate_all_metrics(result, spy_returns):
    """Calculate comprehensive metrics for a backtest result."""
    equity = result.equity_curve['nav'] if isinstance(result.equity_curve, pd.DataFrame) else result.equity_curve
    returns = equity.pct_change().dropna()

    # Align with SPY
    common_dates = returns.index.intersection(spy_returns.index)
    aligned_returns = returns.loc[common_dates]
    aligned_spy = spy_returns.loc[common_dates]

    metrics = {
        'total_return': float((equity.iloc[-1] / equity.iloc[0]) - 1),
        'cagr': float(cagr(equity)),
        'sharpe_ratio': float(sharpe(returns)),
        'sortino_ratio': float(sortino_ratio(returns)),
        'max_drawdown': float(max_drawdown(equity)),
        'calmar_ratio': float(calmar_ratio(equity)),
        'volatility': float(returns.std() * (252 ** 0.5)),
        'win_rate': float(win_rate(returns)),
    }

    # Active metrics
    try:
        active = calculate_active_metrics(aligned_returns, aligned_spy)
        metrics.update({
            'active_return': active.get('active_return', 0),
            'information_ratio': active.get('information_ratio', 0),
            'tracking_error': active.get('tracking_error', 0),
        })
    except Exception as e:
        logger.warning(f"Could not calculate active metrics: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='EXP-003: Ensemble vs Switching Strategy Comparison'
    )
    parser.add_argument('--snapshot', type=str, required=True,
                        help='Path to snapshot directory')
    parser.add_argument('--start', type=str, default='2017-01-01',
                        help='Start date (default: 2017-01-01, allows warmup)')
    parser.add_argument('--end', type=str, default=None,
                        help='End date (default: use all data)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of ETFs to hold (default: 5)')
    parser.add_argument('--output-dir', type=str, default='artifacts/experiments',
                        help='Output directory')

    args = parser.parse_args()

    # Load data
    snapshot_path = Path(args.snapshot)
    if snapshot_path.is_dir():
        parquet_path = snapshot_path / 'data.parquet'
    else:
        parquet_path = snapshot_path

    logger.info(f"Loading snapshot from {parquet_path}")

    # Create DataAccessContext using factory
    data_access = DataAccessFactory.create_context(
        config={"snapshot_path": str(parquet_path)},
        enable_caching=True
    )

    # Get date range and tickers
    latest_date = data_access.prices.get_latest_price_date()
    prices = data_access.prices.read_prices_as_of(as_of=latest_date + pd.Timedelta(days=1))
    all_dates = prices.index
    start_date = pd.Timestamp(args.start) if args.start else all_dates[0]
    end_date = pd.Timestamp(args.end) if args.end else all_dates[-1]
    tickers = prices.columns.get_level_values('Ticker').unique().tolist()

    logger.info(f"Backtest period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Universe: {len(tickers)} tickers")

    # Get SPY returns for comparison
    spy_prices_data = data_access.prices.read_prices_as_of(as_of=end_date + pd.Timedelta(days=1), tickers=['SPY'])
    spy_prices = spy_prices_data.xs('Close', level='Price', axis=1)['SPY'].loc[start_date:end_date]
    spy_returns = spy_prices.pct_change().dropna()

    # Define strategies to compare
    strategies = {}

    # 1. Pure Momentum (baseline)
    logger.info("\n" + "="*60)
    logger.info("Strategy 1: Pure Momentum (baseline)")
    logger.info("="*60)
    strategies['momentum'] = {
        'model': MomentumAlpha(lookback_days=252, min_periods=200),
        'description': 'Pure 12-month momentum'
    }

    # 2. Trend-Filtered Momentum (switching)
    logger.info("\n" + "="*60)
    logger.info("Strategy 2: Trend-Filtered Momentum (switching)")
    logger.info("="*60)
    strategies['trend_filtered'] = {
        'model': TrendFilteredMomentum(
            momentum_lookback=252,
            ma_period=150,  # Best from optimizer
            min_periods=100
        ),
        'description': 'Momentum with MA150 trend filter (switches to defensive in bear markets)'
    }

    # 3. Dual Momentum (absolute + relative)
    logger.info("\n" + "="*60)
    logger.info("Strategy 3: Dual Momentum")
    logger.info("="*60)
    strategies['dual_momentum'] = {
        'model': DualMomentum(
            lookback=252,
            risk_free_rate=0.02,
            min_periods=200
        ),
        'description': 'Gary Antonacci style dual momentum'
    }

    # 4. Value-Momentum Blend
    logger.info("\n" + "="*60)
    logger.info("Strategy 4: Value-Momentum Blend")
    logger.info("="*60)
    strategies['value_momentum'] = {
        'model': ValueMomentum(
            momentum_weight=0.5,
            value_weight=0.5,
            momentum_lookback=252,
            value_lookback=252,
            min_periods=200
        ),
        'description': 'Equal blend of momentum and mean-reversion'
    }

    # 5. Ensemble (blend of all)
    logger.info("\n" + "="*60)
    logger.info("Strategy 5: Ensemble Blend")
    logger.info("="*60)
    ensemble_models = [
        MomentumAlpha(lookback_days=252, min_periods=200),
        DualMomentum(lookback=252, risk_free_rate=0.02, min_periods=200),
        ValueMomentum(momentum_weight=0.5, value_weight=0.5, momentum_lookback=252, min_periods=200),
    ]
    ensemble_weights = [0.4, 0.3, 0.3]  # From research spec
    strategies['ensemble'] = {
        'model': WeightedEnsemble(models=tuple(ensemble_models), weights=tuple(ensemble_weights)),
        'description': 'Blend: 40% momentum + 30% dual momentum + 30% value-momentum'
    }

    # Run all backtests
    results = {}
    for name, strategy in strategies.items():
        logger.info(f"\nRunning backtest: {name}")
        try:
            result = run_backtest(
                data_access=data_access,
                alpha_model=strategy['model'],
                start_date=start_date,
                end_date=end_date,
                tickers=tickers,
                top_n=args.top_n,
            )
            metrics = calculate_all_metrics(result, spy_returns)
            results[name] = {
                'description': strategy['description'],
                'metrics': metrics,
                'equity_curve': result.equity_curve,
            }
            logger.info(f"  Total Return: {metrics['total_return']*100:.1f}%")
            logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"  Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
            logger.info(f"  Active Return: {metrics.get('active_return', 0)*100:.1f}%")
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results[name] = {'error': str(e)}

    # Print comparison table
    print("\n" + "="*80)
    print("EXP-003 RESULTS: ENSEMBLE VS SWITCHING")
    print("="*80)
    print(f"\nBacktest Period: {start_date.date()} to {end_date.date()}")
    print(f"Top N: {args.top_n}")
    print()

    # Create comparison table
    print(f"{'Strategy':<25} {'Total Ret':>10} {'Sharpe':>8} {'Max DD':>10} {'Active':>10} {'IR':>8}")
    print("-"*80)

    for name, data in results.items():
        if 'error' in data:
            print(f"{name:<25} ERROR: {data['error']}")
        else:
            m = data['metrics']
            print(f"{name:<25} {m['total_return']*100:>9.1f}% {m['sharpe_ratio']:>8.2f} {m['max_drawdown']*100:>9.1f}% {m.get('active_return', 0)*100:>9.1f}% {m.get('information_ratio', 0):>8.2f}")

    print("-"*80)

    # Determine winners
    valid_results = {k: v for k, v in results.items() if 'error' not in v}

    if valid_results:
        # Best by Sharpe
        best_sharpe = max(valid_results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'])
        print(f"\nBest Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1]['metrics']['sharpe_ratio']:.2f})")

        # Best by Active Return
        best_active = max(valid_results.items(), key=lambda x: x[1]['metrics'].get('active_return', -999))
        print(f"Best Active Return: {best_active[0]} ({best_active[1]['metrics'].get('active_return', 0)*100:.1f}%)")

        # Lowest Max Drawdown
        best_dd = min(valid_results.items(), key=lambda x: abs(x[1]['metrics']['max_drawdown']))
        print(f"Lowest Max Drawdown: {best_dd[0]} ({best_dd[1]['metrics']['max_drawdown']*100:.1f}%)")

        # Compare ensemble vs trend_filtered (main hypothesis)
        print("\n" + "="*80)
        print("KEY FINDING: Ensemble vs Switching (Trend-Filtered)")
        print("="*80)

        if 'ensemble' in valid_results and 'trend_filtered' in valid_results:
            ens = valid_results['ensemble']['metrics']
            tf = valid_results['trend_filtered']['metrics']

            print(f"\n{'Metric':<25} {'Ensemble':>15} {'Trend-Filtered':>15} {'Difference':>15}")
            print("-"*70)
            print(f"{'Total Return':<25} {ens['total_return']*100:>14.1f}% {tf['total_return']*100:>14.1f}% {(ens['total_return']-tf['total_return'])*100:>+14.1f}%")
            print(f"{'Sharpe Ratio':<25} {ens['sharpe_ratio']:>15.2f} {tf['sharpe_ratio']:>15.2f} {ens['sharpe_ratio']-tf['sharpe_ratio']:>+15.2f}")
            print(f"{'Max Drawdown':<25} {ens['max_drawdown']*100:>14.1f}% {tf['max_drawdown']*100:>14.1f}% {(ens['max_drawdown']-tf['max_drawdown'])*100:>+14.1f}%")
            print(f"{'Volatility':<25} {ens['volatility']*100:>14.1f}% {tf['volatility']*100:>14.1f}% {(ens['volatility']-tf['volatility'])*100:>+14.1f}%")
            print(f"{'Active Return':<25} {ens.get('active_return',0)*100:>14.1f}% {tf.get('active_return',0)*100:>14.1f}% {(ens.get('active_return',0)-tf.get('active_return',0))*100:>+14.1f}%")

            # Conclusion
            print("\n" + "-"*70)
            if ens['sharpe_ratio'] > tf['sharpe_ratio']:
                print("CONCLUSION: Ensemble WINS on risk-adjusted basis (higher Sharpe)")
            else:
                print("CONCLUSION: Trend-Filtered WINS on risk-adjusted basis (higher Sharpe)")

            if abs(ens['max_drawdown']) < abs(tf['max_drawdown']):
                print("           Ensemble has LOWER max drawdown (more stable)")
            else:
                print("           Trend-Filtered has LOWER max drawdown (more stable)")

    # Save results
    output_dir = Path(args.output_dir) / f"exp003_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics summary
    summary = {
        name: {
            'description': data.get('description', ''),
            'metrics': data.get('metrics', {}),
            'error': data.get('error', None)
        }
        for name, data in results.items()
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save equity curves
    for name, data in results.items():
        if 'equity_curve' in data:
            data['equity_curve'].to_csv(output_dir / f'{name}_equity.csv')

    print(f"\nResults saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
