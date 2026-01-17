"""Example: Using RegimeAwareAlpha with SimpleBacktestEngine.

This example demonstrates how to integrate regime-aware alpha selection
into a backtest using the SimpleBacktestEngine.
"""

from pathlib import Path
import pandas as pd

from quantetf.backtest.simple_engine import SimpleBacktestEngine, BacktestConfig
from quantetf.alpha.momentum import MomentumAlpha
from quantetf.alpha.value_momentum import ValueMomentum
from quantetf.alpha.selector import RegimeBasedSelector, MarketRegime
from quantetf.alpha.regime_aware import RegimeAwareAlpha, RegimeDetector
from quantetf.data.macro_loader import MacroDataLoader
from quantetf.portfolio.equal_weight import EqualWeightTopN
from quantetf.portfolio.costs import FlatTransactionCost
from quantetf.data.snapshot_store import SnapshotDataStore
from quantetf.universe import Universe


def create_regime_aware_alpha_example():
    """Create a RegimeAwareAlpha instance for backtesting.

    This example shows:
    1. Creating individual alpha models (momentum, value-momentum)
    2. Creating a selector that maps regimes to models
    3. Wrapping everything in RegimeAwareAlpha
    4. Using it with the backtest engine

    Returns:
        RegimeAwareAlpha instance ready for backtesting
    """

    # 1. Create individual alpha models
    momentum_model = MomentumAlpha(
        lookback_days=252,
        min_price=5.0,
    )

    value_momentum_model = ValueMomentum(
        momentum_lookback=252,
        value_lookback=252,
    )

    models = {
        "momentum": momentum_model,
        "value_momentum": value_momentum_model,
    }

    # 2. Create a selector that adapts models to regime
    # In risk-on: use pure momentum (higher beta)
    # In defensive/high-vol: use value-momentum (better risk-adjusted)
    regime_model_map = {
        MarketRegime.RISK_ON: "momentum",
        MarketRegime.ELEVATED_VOL: "value_momentum",
        MarketRegime.HIGH_VOL: "value_momentum",
        MarketRegime.RECESSION_WARNING: "value_momentum",
        MarketRegime.UNKNOWN: "momentum",  # Default: optimistic
    }

    selector = RegimeBasedSelector(
        regime_model_map=regime_model_map,
        default_model="momentum",
    )

    # 3. Create macro data loader for regime detection
    macro_loader = MacroDataLoader(data_dir=Path("data/raw/macro"))

    # 4. Create RegimeAwareAlpha wrapper
    regime_aware = RegimeAwareAlpha(
        selector=selector,
        models=models,
        macro_loader=macro_loader,
        name="RegimeAdaptiveMomentum",
    )

    return regime_aware


def example_backtest_with_regime_alpha():
    """Run a complete backtest using regime-aware alpha.

    This demonstrates the end-to-end flow:
    1. Create regime-aware alpha model
    2. Set up backtest configuration
    3. Run the backtest
    4. Print regime history and performance
    """

    # Create regime-aware alpha model
    alpha_model = create_regime_aware_alpha_example()

    # Create backtest config
    universe = Universe(
        tickers=["SPY", "QQQ", "IWM", "EEM", "AGG"],
        tier="tier_2",
    )

    config = BacktestConfig(
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2023-12-31"),
        universe=universe,
        initial_capital=100_000.0,
        rebalance_frequency="monthly",
    )

    # Create supporting components
    portfolio = EqualWeightTopN(top_n=3)
    cost_model = FlatTransactionCost(cost_bps=10.0)

    # Load data store
    store = SnapshotDataStore(snapshot_dir=Path("data/snapshots"))

    # Run backtest
    engine = SimpleBacktestEngine()
    result = engine.run(
        config=config,
        alpha_model=alpha_model,
        portfolio=portfolio,
        cost_model=cost_model,
        store=store,
    )

    # Print results
    print("Backtest Complete!")
    print(f"Total Return: {result.metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")

    # Print regime history
    print("\nRegime History:")
    regime_history = alpha_model.get_regime_history()
    print(regime_history.head(10))

    # Print selection statistics
    print("\nModel Selection Statistics:")
    stats = alpha_model.get_selection_stats()
    for regime, count in stats.items():
        print(f"  {regime}: {count} days")

    return result


if __name__ == "__main__":
    # This would be run from the project root
    # result = example_backtest_with_regime_alpha()
    print("This is an example showing how to use RegimeAwareAlpha.")
    print("To run this example, ensure you have:")
    print("1. Macro data in data/raw/macro/")
    print("2. Price snapshots in data/snapshots/")
    print("3. All dependencies installed")
