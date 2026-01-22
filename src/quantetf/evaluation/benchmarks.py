"""Benchmark strategies for comparing portfolio performance.

This module provides standard benchmark implementations for strategy evaluation:
- SPY buy-and-hold
- 60/40 portfolio (SPY/AGG)
- Equal-weight universe
- Random selection (Monte Carlo)
- Oracle (perfect foresight upper bound)

These benchmarks help contextualize strategy performance and assess whether
active management adds value over simple passive alternatives.
"""

from __future__ import annotations

import logging
from typing import Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

from quantetf.types import Universe, CASH_TICKER
from quantetf.data.access import DataAccessContext
from quantetf.backtest.simple_engine import BacktestConfig, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from running a benchmark strategy."""

    name: str
    equity_curve: pd.DataFrame  # columns: nav, cost
    holdings_history: pd.DataFrame  # date x ticker (shares)
    weights_history: pd.DataFrame  # date x ticker (weights)
    metrics: dict
    description: str


def run_spy_benchmark(
    *,
    config: BacktestConfig,
    data_access: DataAccessContext,
) -> BenchmarkResult:
    """Run SPY buy-and-hold benchmark.

    Invests 100% of capital in SPY at start date and holds until end.
    This represents the passive market beta exposure.

    Args:
        config: Backtest configuration (dates, initial capital)
        data_access: Data access context for historical prices

    Returns:
        BenchmarkResult with equity curve and metrics

    Example:
        >>> result = run_spy_benchmark(config=config, data_access=data_access)
        >>> print(f"SPY Return: {result.metrics['total_return']:.2%}")
    """
    logger.info("Running SPY buy-and-hold benchmark")

    # Get rebalance dates (we only need first and last)
    rebalance_dates = _generate_rebalance_dates_simple(
        start=config.start_date,
        end=config.end_date
    )

    if len(rebalance_dates) == 0:
        raise ValueError("No valid dates in date range")

    # Initialize tracking
    nav = config.initial_capital
    holdings = pd.Series(0.0, index=['SPY', CASH_TICKER])
    weights = pd.Series(0.0, index=['SPY', CASH_TICKER])

    nav_history = []
    holdings_history = []
    weights_history = []

    # Buy SPY on first date
    first_date = rebalance_dates[0]
    prices = data_access.prices.read_prices_as_of(as_of=first_date, tickers=['SPY'])
    spy_prices_df = prices.xs('Close', level='Price', axis=1) if not prices.empty else pd.DataFrame()

    if spy_prices_df.empty or 'SPY' not in spy_prices_df.columns:
        raise ValueError(f"No SPY price data available as of {first_date}")

    spy_price = spy_prices_df['SPY'].iloc[-1]  # Get most recent price

    if pd.isna(spy_price) or spy_price <= 0:
        raise ValueError(f"Invalid SPY price on {first_date}: {spy_price}")

    spy_shares = nav / spy_price
    holdings['SPY'] = spy_shares
    weights['SPY'] = 1.0

    logger.info(f"Bought {spy_shares:.2f} shares of SPY at ${spy_price:.2f}")

    # Track performance on all dates
    for date in rebalance_dates:
        prices = data_access.prices.read_prices_as_of(as_of=date, tickers=['SPY'])
        spy_prices_df = prices.xs('Close', level='Price', axis=1) if not prices.empty else pd.DataFrame()

        if spy_prices_df.empty or 'SPY' not in spy_prices_df.columns:
            logger.warning(f"No SPY price data on {date}, using previous NAV")
            nav_history.append({'date': date, 'nav': nav, 'cost': 0.0})
            continue

        spy_price = spy_prices_df['SPY'].iloc[-1]

        if pd.isna(spy_price) or spy_price <= 0:
            logger.warning(f"Invalid SPY price on {date}, using previous NAV")
            nav_history.append({'date': date, 'nav': nav, 'cost': 0.0})
        else:
            nav = holdings['SPY'] * spy_price
            nav_history.append({'date': date, 'nav': nav, 'cost': 0.0})

        holdings_history.append({
            'date': date,
            **holdings.to_dict()
        })
        weights_history.append({
            'date': date,
            **weights.to_dict()
        })

    # Convert to DataFrames
    equity_df = pd.DataFrame(nav_history).set_index('date')
    holdings_df = pd.DataFrame(holdings_history).set_index('date')
    weights_df = pd.DataFrame(weights_history).set_index('date')

    # Calculate metrics
    total_return = (equity_df['nav'].iloc[-1] / config.initial_capital) - 1.0

    metrics = {
        'total_return': total_return,
        'final_nav': equity_df['nav'].iloc[-1],
        'initial_capital': config.initial_capital,
    }

    return BenchmarkResult(
        name='SPY Buy-and-Hold',
        equity_curve=equity_df,
        holdings_history=holdings_df,
        weights_history=weights_df,
        metrics=metrics,
        description='100% SPY passive allocation'
    )


def run_60_40_benchmark(
    *,
    config: BacktestConfig,
    data_access: DataAccessContext,
    rebalance_frequency: str = 'quarterly'
) -> BenchmarkResult:
    """Run 60/40 portfolio benchmark (60% SPY, 40% AGG).

    Classic balanced portfolio with periodic rebalancing.
    Represents traditional diversified passive allocation.

    Args:
        config: Backtest configuration (dates, initial capital)
        data_access: Data access context for historical prices
        rebalance_frequency: How often to rebalance ('quarterly', 'monthly')

    Returns:
        BenchmarkResult with equity curve and metrics

    Example:
        >>> result = run_60_40_benchmark(config=config, data_access=data_access)
        >>> print(f"60/40 Return: {result.metrics['total_return']:.2%}")
    """
    logger.info("Running 60/40 (SPY/AGG) benchmark")

    target_weights = {'SPY': 0.6, 'AGG': 0.4, CASH_TICKER: 0.0}
    rebalance_dates = _generate_rebalance_dates_simple(
        start=config.start_date,
        end=config.end_date,
        frequency=rebalance_frequency
    )

    if len(rebalance_dates) == 0:
        raise ValueError("No valid dates in date range")

    # Initialize
    nav = config.initial_capital
    holdings = pd.Series(0.0, index=['SPY', 'AGG', CASH_TICKER])
    weights = pd.Series(0.0, index=['SPY', 'AGG', CASH_TICKER])

    nav_history = []
    holdings_history = []
    weights_history = []

    for i, date in enumerate(rebalance_dates):
        # Get current prices
        price_data = data_access.prices.read_prices_as_of(as_of=date, tickers=['SPY', 'AGG'])
        prices_df = price_data.xs('Close', level='Price', axis=1) if not price_data.empty else pd.DataFrame()

        if prices_df.empty:
            logger.warning(f"No price data on {date}")
            if i > 0:
                nav_history.append({'date': date, 'nav': nav, 'cost': 0.0})
                holdings_history.append({'date': date, **holdings.to_dict()})
                weights_history.append({'date': date, **weights.to_dict()})
            continue

        # Get most recent prices
        prices = prices_df.iloc[-1]

        # Check for missing prices
        if prices.isna().any():
            missing = prices[prices.isna()].index.tolist()
            logger.warning(f"Missing prices on {date}: {missing}")
            # Use previous NAV
            if i > 0:
                nav_history.append({'date': date, 'nav': nav, 'cost': 0.0})
                holdings_history.append({'date': date, **holdings.to_dict()})
                weights_history.append({'date': date, **weights.to_dict()})
            continue

        # Update NAV from current holdings
        if i > 0:
            nav = (holdings['SPY'] * prices['SPY'] +
                   holdings['AGG'] * prices['AGG'] +
                   holdings[CASH_TICKER])

        # Rebalance to target weights
        target_values = {ticker: nav * weight
                        for ticker, weight in target_weights.items()}

        holdings['SPY'] = target_values['SPY'] / prices['SPY']
        holdings['AGG'] = target_values['AGG'] / prices['AGG']
        holdings[CASH_TICKER] = target_values[CASH_TICKER]

        weights = pd.Series(target_weights)

        # Record
        nav_history.append({'date': date, 'nav': nav, 'cost': 0.0})
        holdings_history.append({'date': date, **holdings.to_dict()})
        weights_history.append({'date': date, **weights.to_dict()})

    # Convert to DataFrames
    equity_df = pd.DataFrame(nav_history).set_index('date')
    holdings_df = pd.DataFrame(holdings_history).set_index('date')
    weights_df = pd.DataFrame(weights_history).set_index('date')

    # Calculate metrics
    total_return = (equity_df['nav'].iloc[-1] / config.initial_capital) - 1.0

    metrics = {
        'total_return': total_return,
        'final_nav': equity_df['nav'].iloc[-1],
        'initial_capital': config.initial_capital,
    }

    return BenchmarkResult(
        name='60/40 Portfolio',
        equity_curve=equity_df,
        holdings_history=holdings_df,
        weights_history=weights_df,
        metrics=metrics,
        description='60% SPY, 40% AGG with periodic rebalancing'
    )


def run_equal_weight_benchmark(
    *,
    config: BacktestConfig,
    data_access: DataAccessContext,
    rebalance_frequency: str = 'monthly'
) -> BenchmarkResult:
    """Run equal-weight universe benchmark.

    Equally weights all tickers in the universe with periodic rebalancing.
    Represents naive diversification strategy.

    Args:
        config: Backtest configuration with universe
        data_access: Data access context for historical prices
        rebalance_frequency: How often to rebalance

    Returns:
        BenchmarkResult with equity curve and metrics

    Example:
        >>> result = run_equal_weight_benchmark(config=config, data_access=data_access)
        >>> print(f"Equal Weight Return: {result.metrics['total_return']:.2%}")
    """
    logger.info(f"Running equal-weight benchmark ({len(config.universe.tickers)} tickers)")

    tickers = list(config.universe.tickers)
    all_tickers = tickers + [CASH_TICKER]
    target_weight = 1.0 / len(tickers)

    rebalance_dates = _generate_rebalance_dates_simple(
        start=config.start_date,
        end=config.end_date,
        frequency=rebalance_frequency
    )

    if len(rebalance_dates) == 0:
        raise ValueError("No valid dates in date range")

    # Initialize
    nav = config.initial_capital
    holdings = pd.Series(0.0, index=all_tickers)
    weights = pd.Series(0.0, index=all_tickers)

    nav_history = []
    holdings_history = []
    weights_history = []

    for i, date in enumerate(rebalance_dates):
        # Get current prices
        price_data = data_access.prices.read_prices_as_of(as_of=date, tickers=tickers)
        prices_df = price_data.xs('Close', level='Price', axis=1) if not price_data.empty else pd.DataFrame()

        if prices_df.empty:
            logger.warning(f"No price data on {date}")
            if i > 0:
                nav_history.append({'date': date, 'nav': nav, 'cost': 0.0})
                holdings_history.append({'date': date, **holdings.to_dict()})
                weights_history.append({'date': date, **weights.to_dict()})
            continue

        # Get most recent prices
        prices = prices_df.iloc[-1]

        # Filter out missing prices
        valid_tickers = prices[~prices.isna() & (prices > 0)].index.tolist()

        if len(valid_tickers) == 0:
            logger.warning(f"No valid prices on {date}")
            if i > 0:
                nav_history.append({'date': date, 'nav': nav, 'cost': 0.0})
                holdings_history.append({'date': date, **holdings.to_dict()})
                weights_history.append({'date': date, **weights.to_dict()})
            continue

        # Update NAV from current holdings
        if i > 0:
            nav = holdings[CASH_TICKER]
            for ticker in valid_tickers:
                nav += holdings[ticker] * prices[ticker]

        # Rebalance to equal weights (only valid tickers)
        valid_weight = 1.0 / len(valid_tickers)
        holdings = pd.Series(0.0, index=all_tickers)
        weights = pd.Series(0.0, index=all_tickers)

        for ticker in valid_tickers:
            target_value = nav * valid_weight
            holdings[ticker] = target_value / prices[ticker]
            weights[ticker] = valid_weight

        # Record
        nav_history.append({'date': date, 'nav': nav, 'cost': 0.0})
        holdings_history.append({'date': date, **holdings.to_dict()})
        weights_history.append({'date': date, **weights.to_dict()})

    # Convert to DataFrames
    equity_df = pd.DataFrame(nav_history).set_index('date')
    holdings_df = pd.DataFrame(holdings_history).set_index('date')
    weights_df = pd.DataFrame(weights_history).set_index('date')

    # Calculate metrics
    total_return = (equity_df['nav'].iloc[-1] / config.initial_capital) - 1.0

    metrics = {
        'total_return': total_return,
        'final_nav': equity_df['nav'].iloc[-1],
        'initial_capital': config.initial_capital,
        'universe_size': len(tickers),
    }

    return BenchmarkResult(
        name='Equal Weight Universe',
        equity_curve=equity_df,
        holdings_history=holdings_df,
        weights_history=weights_df,
        metrics=metrics,
        description=f'Equal weight allocation across {len(tickers)} tickers'
    )


def run_random_selection_benchmark(
    *,
    config: BacktestConfig,
    data_access: DataAccessContext,
    n_selections: int = 5,
    n_trials: int = 100,
    rebalance_frequency: str = 'monthly',
    seed: Optional[int] = None
) -> BenchmarkResult:
    """Run random selection Monte Carlo benchmark.

    Randomly selects N tickers and equal-weights them, averaging over many trials.
    Provides a baseline for skill vs luck assessment.

    Args:
        config: Backtest configuration with universe
        data_access: Data access context for historical prices
        n_selections: Number of tickers to randomly select
        n_trials: Number of random trials to average
        rebalance_frequency: How often to rebalance
        seed: Random seed for reproducibility

    Returns:
        BenchmarkResult with average equity curve and metrics

    Example:
        >>> result = run_random_selection_benchmark(
        ...     config=config, data_access=data_access, n_selections=5, n_trials=100
        ... )
        >>> print(f"Random (N={n_selections}) Return: {result.metrics['total_return']:.2%}")
    """
    logger.info(f"Running random selection benchmark (N={n_selections}, trials={n_trials})")

    if seed is not None:
        np.random.seed(seed)

    tickers = list(config.universe.tickers)

    if n_selections > len(tickers):
        raise ValueError(f"n_selections ({n_selections}) > universe size ({len(tickers)})")

    rebalance_dates = _generate_rebalance_dates_simple(
        start=config.start_date,
        end=config.end_date,
        frequency=rebalance_frequency
    )

    if len(rebalance_dates) == 0:
        raise ValueError("No valid dates in date range")

    # Run multiple trials
    trial_equity_curves = []

    for trial in range(n_trials):
        # Initialize for this trial
        nav = config.initial_capital
        holdings = pd.Series(0.0, index=tickers + [CASH_TICKER])
        nav_history = []

        for i, date in enumerate(rebalance_dates):
            # Get prices
            price_data = data_access.prices.read_prices_as_of(as_of=date, tickers=tickers)
            prices_df = price_data.xs('Close', level='Price', axis=1) if not price_data.empty else pd.DataFrame()

            if prices_df.empty:
                if i > 0:
                    nav_history.append({'date': date, 'nav': nav})
                continue

            prices = prices_df.iloc[-1]
            valid_tickers = prices[~prices.isna() & (prices > 0)].index.tolist()

            if len(valid_tickers) < n_selections:
                logger.warning(
                    f"Trial {trial}, {date}: Only {len(valid_tickers)} valid tickers"
                )
                if i > 0:
                    nav_history.append({'date': date, 'nav': nav})
                continue

            # Update NAV
            if i > 0:
                nav = holdings[CASH_TICKER]
                for ticker in valid_tickers:
                    nav += holdings[ticker] * prices[ticker]

            # Randomly select N tickers
            selected_tickers = np.random.choice(
                valid_tickers, size=n_selections, replace=False
            ).tolist()

            # Equal weight selected tickers
            holdings = pd.Series(0.0, index=tickers + [CASH_TICKER])
            weight_per_ticker = 1.0 / n_selections

            for ticker in selected_tickers:
                target_value = nav * weight_per_ticker
                holdings[ticker] = target_value / prices[ticker]

            nav_history.append({'date': date, 'nav': nav})

        trial_equity_df = pd.DataFrame(nav_history).set_index('date')
        trial_equity_curves.append(trial_equity_df['nav'])

    # Average across trials
    all_equity = pd.concat(trial_equity_curves, axis=1)
    avg_equity = all_equity.mean(axis=1)
    std_equity = all_equity.std(axis=1)

    equity_df = pd.DataFrame({
        'nav': avg_equity,
        'cost': 0.0,
        'std': std_equity
    })

    # Placeholder for holdings/weights (not meaningful when averaged)
    holdings_df = pd.DataFrame(index=equity_df.index)
    weights_df = pd.DataFrame(index=equity_df.index)

    # Calculate metrics
    total_return = (equity_df['nav'].iloc[-1] / config.initial_capital) - 1.0

    metrics = {
        'total_return': total_return,
        'final_nav': equity_df['nav'].iloc[-1],
        'initial_capital': config.initial_capital,
        'n_selections': n_selections,
        'n_trials': n_trials,
        'std_final_nav': equity_df['std'].iloc[-1],
    }

    return BenchmarkResult(
        name=f'Random Selection (N={n_selections})',
        equity_curve=equity_df,
        holdings_history=holdings_df,
        weights_history=weights_df,
        metrics=metrics,
        description=f'Average of {n_trials} random {n_selections}-ticker portfolios'
    )


def run_oracle_benchmark(
    *,
    config: BacktestConfig,
    data_access: DataAccessContext,
    n_selections: int = 5,
    rebalance_frequency: str = 'monthly'
) -> BenchmarkResult:
    """Run oracle benchmark (perfect foresight upper bound).

    At each rebalance, selects the N best-performing tickers over the NEXT period.
    This represents the theoretical upper bound with perfect foresight.
    Useful for understanding strategy potential and realistic performance limits.

    Args:
        config: Backtest configuration with universe
        data_access: Data access context for historical prices
        n_selections: Number of top performers to select
        rebalance_frequency: How often to rebalance

    Returns:
        BenchmarkResult with equity curve and metrics

    Example:
        >>> result = run_oracle_benchmark(config=config, data_access=data_access, n_selections=5)
        >>> print(f"Oracle Return: {result.metrics['total_return']:.2%}")
    """
    logger.info(f"Running oracle benchmark (perfect foresight, N={n_selections})")

    tickers = list(config.universe.tickers)
    all_tickers = tickers + [CASH_TICKER]

    rebalance_dates = _generate_rebalance_dates_simple(
        start=config.start_date,
        end=config.end_date,
        frequency=rebalance_frequency
    )

    if len(rebalance_dates) < 2:
        raise ValueError("Need at least 2 dates for oracle benchmark")

    # Initialize
    nav = config.initial_capital
    holdings = pd.Series(0.0, index=all_tickers)
    weights = pd.Series(0.0, index=all_tickers)

    nav_history = []
    holdings_history = []
    weights_history = []

    for i, date in enumerate(rebalance_dates[:-1]):  # Stop at second-to-last
        # Get current prices
        price_data = data_access.prices.read_prices_as_of(as_of=date, tickers=tickers)
        current_prices_df = price_data.xs('Close', level='Price', axis=1) if not price_data.empty else pd.DataFrame()

        if current_prices_df.empty:
            logger.warning(f"No price data on {date}")
            continue

        current_prices = current_prices_df.iloc[-1]
        valid_current = current_prices[~current_prices.isna() & (current_prices > 0)]

        if len(valid_current) == 0:
            logger.warning(f"No valid prices on {date}")
            continue

        # Update NAV from current holdings
        if i > 0:
            nav = holdings[CASH_TICKER]
            for ticker in valid_current.index:
                nav += holdings[ticker] * current_prices[ticker]

        # Look ahead to next period to get future returns (ORACLE!)
        next_date = rebalance_dates[i + 1]
        next_price_data = data_access.prices.read_prices_as_of(as_of=next_date, tickers=tickers)
        next_prices_df = next_price_data.xs('Close', level='Price', axis=1) if not next_price_data.empty else pd.DataFrame()

        if next_prices_df.empty:
            logger.warning(f"No price data on {next_date}")
            # Hold cash
            holdings = pd.Series(0.0, index=all_tickers)
            holdings[CASH_TICKER] = nav
            weights = pd.Series(0.0, index=all_tickers)
            weights[CASH_TICKER] = 1.0
            nav_history.append({'date': date, 'nav': nav, 'cost': 0.0})
            holdings_history.append({'date': date, **holdings.to_dict()})
            weights_history.append({'date': date, **weights.to_dict()})
            continue

        next_prices = next_prices_df.iloc[-1]

        # Calculate forward returns
        forward_returns = {}
        for ticker in valid_current.index:
            if ticker in next_prices.index and not pd.isna(next_prices[ticker]):
                forward_returns[ticker] = (
                    next_prices[ticker] / current_prices[ticker] - 1.0
                )

        if len(forward_returns) < n_selections:
            logger.warning(
                f"Only {len(forward_returns)} valid forward returns on {date}"
            )
            # Hold cash
            holdings = pd.Series(0.0, index=all_tickers)
            holdings[CASH_TICKER] = nav
            weights = pd.Series(0.0, index=all_tickers)
            weights[CASH_TICKER] = 1.0
        else:
            # Select top N by forward return
            sorted_tickers = sorted(
                forward_returns.items(), key=lambda x: x[1], reverse=True
            )
            top_tickers = [t[0] for t in sorted_tickers[:n_selections]]

            # Equal weight top performers
            holdings = pd.Series(0.0, index=all_tickers)
            weights = pd.Series(0.0, index=all_tickers)
            weight_per_ticker = 1.0 / n_selections

            for ticker in top_tickers:
                target_value = nav * weight_per_ticker
                holdings[ticker] = target_value / current_prices[ticker]
                weights[ticker] = weight_per_ticker

        # Record
        nav_history.append({'date': date, 'nav': nav, 'cost': 0.0})
        holdings_history.append({'date': date, **holdings.to_dict()})
        weights_history.append({'date': date, **weights.to_dict()})

    # Final update for last date
    last_date = rebalance_dates[-1]
    last_price_data = data_access.prices.read_prices_as_of(as_of=last_date, tickers=tickers)
    last_prices_df = last_price_data.xs('Close', level='Price', axis=1) if not last_price_data.empty else pd.DataFrame()

    if not last_prices_df.empty:
        last_prices = last_prices_df.iloc[-1]
        nav = holdings[CASH_TICKER]
        for ticker in last_prices.index:
            if not pd.isna(last_prices[ticker]) and last_prices[ticker] > 0:
                nav += holdings[ticker] * last_prices[ticker]

    nav_history.append({'date': last_date, 'nav': nav, 'cost': 0.0})
    holdings_history.append({'date': last_date, **holdings.to_dict()})
    weights_history.append({'date': last_date, **weights.to_dict()})

    # Convert to DataFrames
    equity_df = pd.DataFrame(nav_history).set_index('date')
    holdings_df = pd.DataFrame(holdings_history).set_index('date')
    weights_df = pd.DataFrame(weights_history).set_index('date')

    # Calculate metrics
    total_return = (equity_df['nav'].iloc[-1] / config.initial_capital) - 1.0

    metrics = {
        'total_return': total_return,
        'final_nav': equity_df['nav'].iloc[-1],
        'initial_capital': config.initial_capital,
        'n_selections': n_selections,
    }

    return BenchmarkResult(
        name=f'Oracle (N={n_selections})',
        equity_curve=equity_df,
        holdings_history=holdings_df,
        weights_history=weights_df,
        metrics=metrics,
        description=f'Perfect foresight selection of top {n_selections} performers'
    )


def _generate_rebalance_dates_simple(
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str = 'monthly'
) -> list[pd.Timestamp]:
    """Generate rebalance dates for benchmarks.

    Args:
        start: Start date
        end: End date
        frequency: 'monthly', 'quarterly', or 'weekly'

    Returns:
        List of rebalance dates
    """
    if frequency == 'monthly':
        freq = 'ME'  # Month end
    elif frequency == 'quarterly':
        freq = 'QE'  # Quarter end
    elif frequency == 'weekly':
        freq = 'W-FRI'  # Week end (Friday)
    else:
        raise ValueError(f"Unknown frequency: {frequency}")

    dates = pd.date_range(start=start, end=end, freq=freq)
    return [pd.Timestamp(d) for d in dates]
