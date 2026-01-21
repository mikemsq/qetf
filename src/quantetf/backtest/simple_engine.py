"""Simple event-driven backtest engine.

This module provides a straightforward backtest implementation that orchestrates
the various Phase 2 components (alpha, portfolio construction, costs) into a
complete backtesting system.

Migrated to use DataAccessContext (DAL) instead of direct SnapshotDataStore dependency.
"""

from __future__ import annotations

import logging
from typing import Optional, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np

from quantetf.types import Universe, CASH_TICKER
from quantetf.alpha.base import AlphaModel
from quantetf.portfolio.base import PortfolioConstructor, CostModel
from quantetf.data.access import DataAccessContext, PriceDataAccessor
from quantetf.data.store import DataStore
from quantetf.types import DatasetVersion

logger = logging.getLogger(__name__)


class _DataAccessAdapter(DataStore):
    """Adapter that wraps DataAccessContext to provide DataStore interface.

    This enables backward compatibility with alpha models and portfolio
    constructors that still use the DataStore interface. This adapter will
    be removed once IMPL-026 (alpha models migration) and related tasks
    migrate those components to use DataAccessContext directly.
    """

    def __init__(self, data_access: DataAccessContext):
        """Initialize adapter with DataAccessContext.

        Args:
            data_access: The DataAccessContext to wrap
        """
        self._data_access = data_access
        self._prices_accessor = data_access.prices

    def read_prices(
        self,
        as_of: pd.Timestamp,
        tickers: Optional[list[str]] = None,
        lookback_days: Optional[int] = None
    ) -> pd.DataFrame:
        """Read price data as-of a specific date (point-in-time).

        Delegates to the underlying PriceDataAccessor.
        """
        return self._prices_accessor.read_prices_as_of(
            as_of=as_of,
            tickers=tickers,
            lookback_days=lookback_days,
        )

    def get_close_prices(
        self,
        as_of: pd.Timestamp,
        tickers: Optional[list[str]] = None,
        lookback_days: Optional[int] = None
    ) -> pd.DataFrame:
        """Get close prices in simple format (date × ticker).

        Extracts Close prices from OHLCV data returned by the accessor.
        """
        data = self.read_prices(as_of, tickers, lookback_days)

        # Extract Close prices and pivot to simple format
        close_prices = data.xs('Close', level='Price', axis=1)

        return close_prices

    def read_prices_total_return(
        self,
        version: Optional[DatasetVersion] = None
    ) -> pd.DataFrame:
        """Return DataFrame of daily total returns.

        Note: Gets all data up to latest date and computes returns.
        """
        latest_date = self._prices_accessor.get_latest_price_date()
        # Add one day to ensure we get data up to latest_date
        data = self._prices_accessor.read_prices_as_of(
            as_of=latest_date + pd.Timedelta(days=1)
        )
        close_prices = data.xs('Close', level='Price', axis=1)
        returns = close_prices.pct_change()
        return returns

    def read_instrument_master(
        self,
        version: Optional[DatasetVersion] = None
    ) -> pd.DataFrame:
        """Return instrument metadata.

        For compatibility, returns basic metadata from the price accessor.
        """
        # Get available tickers if the accessor supports it
        if hasattr(self._prices_accessor, 'get_available_tickers'):
            tickers = self._prices_accessor.get_available_tickers()
        else:
            # Fallback: read all data and extract tickers
            latest_date = self._prices_accessor.get_latest_price_date()
            data = self._prices_accessor.read_prices_as_of(
                as_of=latest_date + pd.Timedelta(days=1)
            )
            tickers = data.columns.get_level_values('Ticker').unique().tolist()

        return pd.DataFrame({'ticker': tickers}).set_index('ticker')

    def create_snapshot(
        self,
        *,
        snapshot_id: str,
        as_of: pd.Timestamp,
        description: str = ""
    ):
        """Not implemented - adapter is read-only."""
        raise NotImplementedError("DataAccessAdapter is read-only")


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    universe: Universe
    initial_capital: float = 100_000.0
    rebalance_frequency: str = 'monthly'  # 'monthly' or 'weekly'


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    equity_curve: pd.DataFrame  # columns: nav, cost
    holdings_history: pd.DataFrame  # date x ticker (shares)
    weights_history: pd.DataFrame  # date x ticker (weights)
    metrics: dict
    config: BacktestConfig
    rebalance_dates: list  # List of rebalance dates used in backtest


class SimpleBacktestEngine:
    """Simple event-driven backtest engine.

    Iterates through rebalance dates chronologically, making portfolio decisions
    using only historical data (T-1), and tracking performance over time.

    The engine enforces point-in-time correctness by:
    - Only accessing data before the decision date (T-1 and earlier)
    - Making decisions sequentially (no vectorization across future dates)
    - Tracking state explicitly (NAV, holdings, weights)

    Uses DataAccessContext (DAL) for all data access, enabling:
    - Decoupling from specific data storage implementations
    - Easy mocking in tests
    - Transparent caching

    Example:
        >>> from quantetf.data.access import DataAccessFactory
        >>> ctx = DataAccessFactory.create_context(
        ...     config={"snapshot_path": "data/snapshots/latest/data.parquet"}
        ... )
        >>> engine = SimpleBacktestEngine()
        >>> config = BacktestConfig(
        ...     start_date=pd.Timestamp("2021-01-01"),
        ...     end_date=pd.Timestamp("2023-12-31"),
        ...     universe=universe,
        ...     initial_capital=100_000.0
        ... )
        >>> result = engine.run(
        ...     config=config,
        ...     alpha_model=MomentumAlpha(lookback_days=252),
        ...     portfolio=EqualWeightTopN(top_n=5),
        ...     cost_model=FlatTransactionCost(cost_bps=10.0),
        ...     data_access=ctx
        ... )
        >>> print(f"Total Return: {result.metrics['total_return']:.2%}")
    """

    def run(
        self,
        *,
        config: BacktestConfig,
        alpha_model: AlphaModel,
        portfolio: PortfolioConstructor,
        cost_model: CostModel,
        data_access: DataAccessContext,
    ) -> BacktestResult:
        """Run the backtest.

        Args:
            config: Backtest configuration (dates, universe, initial capital)
            alpha_model: Alpha model to generate signals
            portfolio: Portfolio constructor for target weights
            cost_model: Transaction cost model
            data_access: DataAccessContext for historical prices and macro data

        Returns:
            BacktestResult with equity curve, holdings, metrics

        Raises:
            ValueError: If configuration is invalid or insufficient data
        """
        # Create adapter for backward compatibility with alpha models and
        # portfolio constructors that still use DataStore interface
        store = _DataAccessAdapter(data_access)
        logger.info("=" * 80)
        logger.info("Starting SimpleBacktestEngine")
        logger.info("=" * 80)
        logger.info(f"Period: {config.start_date} to {config.end_date}")
        logger.info(f"Universe: {len(config.universe.tickers)} tickers")
        logger.info(f"Initial capital: ${config.initial_capital:,.2f}")
        logger.info(f"Rebalance frequency: {config.rebalance_frequency}")

        # 1. Initialize tracking
        # Add cash to universe if not already present
        all_tickers = list(config.universe.tickers)
        if CASH_TICKER not in all_tickers:
            all_tickers.append(CASH_TICKER)
        
        nav = config.initial_capital
        holdings = pd.Series(0.0, index=all_tickers)  # shares
        weights = pd.Series(0.0, index=all_tickers)  # portfolio weights
        
        # Initialize 100% in cash
        cash_shares = config.initial_capital / 1.0  # Cash price is $1.00
        holdings[CASH_TICKER] = cash_shares
        weights[CASH_TICKER] = 1.0

        # History tracking
        nav_history = []
        holdings_history = []
        weights_history = []
        costs_history = []

        # 2. Generate rebalance dates
        rebalance_dates = _generate_rebalance_dates(
            start=config.start_date,
            end=config.end_date,
            frequency=config.rebalance_frequency
        )

        logger.info(f"Generated {len(rebalance_dates)} rebalance dates")

        if len(rebalance_dates) == 0:
            raise ValueError("No rebalance dates generated - check start/end dates")

        # 3. Event loop - iterate through rebalance dates
        for i, rebalance_date in enumerate(rebalance_dates):
            logger.info(f"Rebalance {i+1}/{len(rebalance_dates)}: {rebalance_date}")

            # 3a. Get prices (T-1 data only!)
            try:
                prices = store.get_close_prices(
                    as_of=rebalance_date,
                    tickers=list(config.universe.tickers)
                )
            except ValueError as e:
                logger.warning(f"No data available for {rebalance_date}, skipping: {e}")
                continue

            if prices.empty:
                logger.warning(f"Empty price data for {rebalance_date}, skipping")
                continue

            # Add cash price ($1.00 constant)
            if CASH_TICKER not in prices.columns:
                # Create cash price series with $1.00 for all dates
                cash_prices = pd.Series(1.0, index=prices.index, name=CASH_TICKER)
                prices = pd.concat([prices, cash_prices], axis=1)

            # Get latest prices for universe tickers (most recent T-1)
            latest_prices = prices.iloc[-1]  # Most recent available (T-1)

            # 3b. Mark to market - update NAV with current holdings
            if i > 0:  # Not first iteration
                portfolio_value = (holdings * latest_prices).sum()
                nav = portfolio_value
                logger.debug(f"  Mark-to-market NAV: ${nav:,.2f}")

            # 3c. Generate alpha scores
            try:
                alpha_scores = alpha_model.score(
                    as_of=rebalance_date,
                    universe=config.universe,
                    features=None,  # Not used in simple momentum
                    store=store
                )
            except Exception as e:
                logger.error(f"Alpha model failed at {rebalance_date}: {e}")
                continue

            # 3d. Construct target portfolio
            try:
                target_weights = portfolio.construct(
                    as_of=rebalance_date,
                    universe=config.universe,
                    alpha=alpha_scores,
                    risk=None,  # Not used yet
                    store=store,
                    prev_weights=weights
                )
            except Exception as e:
                logger.error(f"Portfolio construction failed at {rebalance_date}: {e}")
                continue

            # 3e. Calculate costs
            # Special case: no costs on initial deployment from cash
            if i == 0 and weights.sum() == 0.0:
                # First rebalance from all cash - no transaction costs
                cost = 0.0
                cost_dollars = 0.0
                logger.debug("  Initial deployment from cash: no transaction costs")
            else:
                cost = cost_model.estimate_rebalance_cost(
                    prev_weights=weights,
                    next_weights=target_weights.weights,
                    prices=latest_prices
                )
                cost_dollars = cost * nav  # Convert from fraction to dollars

                logger.debug(f"  Transaction cost: ${cost_dollars:,.2f} ({cost*100:.4f}%)")

            # 3f. Apply costs to NAV
            nav -= cost_dollars

            # 3g. Calculate new holdings (shares to buy/sell)
            target_dollars = target_weights.weights * nav
            new_holdings = target_dollars / latest_prices
            new_holdings = new_holdings.fillna(0.0)

            # 3h. Update state
            holdings = new_holdings
            weights = target_weights.weights

            logger.debug(f"  Post-rebalance NAV: ${nav:,.2f}")
            logger.debug(f"  Holdings: {(holdings > 0).sum()} positions")

            # 3i. Track history
            nav_history.append({
                'date': rebalance_date,
                'nav': nav,
                'cost': cost_dollars
            })
            holdings_history.append(holdings.copy())
            weights_history.append(weights.copy())
            costs_history.append(cost_dollars)

        if len(nav_history) == 0:
            raise ValueError("No successful rebalance dates - backtest failed")

        # 4. Create result DataFrames
        nav_df = pd.DataFrame(nav_history).set_index('date')
        nav_df['returns'] = nav_df['nav'].pct_change()

        holdings_df = pd.DataFrame(
            holdings_history,
            index=[h['date'] for h in nav_history],
            columns=list(config.universe.tickers)
        )

        weights_df = pd.DataFrame(
            weights_history,
            index=[h['date'] for h in nav_history],
            columns=list(config.universe.tickers)
        )

        # 5. Calculate metrics
        total_return = (nav_df['nav'].iloc[-1] / config.initial_capital) - 1.0
        sharpe_ratio = _calculate_sharpe(nav_df['returns'].dropna())
        max_drawdown = _calculate_max_drawdown(nav_df['nav'])
        total_costs = nav_df['cost'].sum()

        metrics = {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'total_costs': float(total_costs),
            'num_rebalances': len(rebalance_dates),
            'final_nav': float(nav_df['nav'].iloc[-1]),
            'initial_nav': float(config.initial_capital),
        }

        logger.info("=" * 80)
        logger.info("Backtest Complete")
        logger.info("=" * 80)
        logger.info(f"Total Return:    {metrics['total_return']:>10.2%}")
        logger.info(f"Sharpe Ratio:    {metrics['sharpe_ratio']:>10.2f}")
        logger.info(f"Max Drawdown:    {metrics['max_drawdown']:>10.2%}")
        logger.info(f"Total Costs:     ${metrics['total_costs']:>10,.2f}")
        logger.info(f"Num Rebalances:  {metrics['num_rebalances']:>10,}")
        logger.info("=" * 80)

        # 6. Return results
        return BacktestResult(
            equity_curve=nav_df,
            holdings_history=holdings_df,
            weights_history=weights_df,
            metrics=metrics,
            config=config,
            rebalance_dates=rebalance_dates,
        )


def _generate_rebalance_dates(
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str = 'monthly'
) -> list[pd.Timestamp]:
    """Generate rebalance dates between start and end.

    Args:
        start: Start date for backtest
        end: End date for backtest
        frequency: Rebalance frequency ('monthly' or 'weekly')

    Returns:
        List of rebalance dates (pd.Timestamp)

    Raises:
        ValueError: If frequency is not recognized
    """
    if frequency == 'monthly':
        # Last business day of each month
        dates = pd.date_range(start, end, freq='BME')  # Business Month End
    elif frequency == 'weekly':
        # Every Friday
        dates = pd.date_range(start, end, freq='W-FRI')
    else:
        raise ValueError(f"Unknown frequency: {frequency}. Use 'monthly' or 'weekly'.")

    return dates.tolist()


def _calculate_sharpe(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Calculate annualized Sharpe ratio.

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year (12 for monthly, 52 for weekly)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    mean_return = returns.mean()
    std_return = returns.std()

    if std_return == 0 or pd.isna(std_return):
        return 0.0

    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    return sharpe


def _calculate_max_drawdown(nav_series: pd.Series) -> float:
    """Calculate maximum drawdown.

    Args:
        nav_series: Series of NAV values

    Returns:
        Maximum drawdown as a negative fraction (e.g., -0.25 = -25%)
    """
    if len(nav_series) == 0:
        return 0.0

    running_max = nav_series.expanding().max()
    drawdown = (nav_series - running_max) / running_max
    return drawdown.min()
