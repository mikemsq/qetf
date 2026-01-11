"""Risk analytics for portfolio and strategy evaluation.

This module provides comprehensive risk analysis tools for understanding
portfolio exposures, concentration, volatility characteristics, and
correlation patterns in backtest results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any


def holdings_correlation_matrix(
    holdings_history: pd.DataFrame,
    window: Optional[int] = None
) -> pd.DataFrame:
    """Calculate correlation matrix of holdings over time.

    Computes pairwise correlations between holdings based on their
    share quantities over the backtest period. Useful for understanding
    diversification and identifying clustered positions.

    Args:
        holdings_history: DataFrame with dates as index, tickers as columns,
                         values are share quantities
        window: Optional rolling window size in periods. If None, uses full history.

    Returns:
        Correlation matrix (DataFrame) with tickers as both index and columns

    Raises:
        ValueError: If holdings_history is empty or has fewer than 2 columns

    Example:
        >>> holdings = pd.DataFrame({
        ...     'SPY': [100, 105, 110],
        ...     'TLT': [50, 48, 45]
        ... })
        >>> corr = holdings_correlation_matrix(holdings)
        >>> corr.loc['SPY', 'TLT']
        -1.0
    """
    if holdings_history.empty:
        raise ValueError("holdings_history is empty")

    if len(holdings_history.columns) < 2:
        raise ValueError("Need at least 2 tickers to compute correlation")

    # Filter to only holdings that ever had non-zero positions
    active_tickers = holdings_history.columns[
        (holdings_history != 0).any(axis=0)
    ]

    if len(active_tickers) < 2:
        raise ValueError("Need at least 2 active tickers to compute correlation")

    holdings_subset = holdings_history[active_tickers]

    if window is not None:
        # Use rolling window correlation (return last matrix)
        corr = holdings_subset.rolling(window=window).corr().iloc[-len(active_tickers):]
    else:
        # Full period correlation
        corr = holdings_subset.corr()

    return corr


def portfolio_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """Calculate portfolio beta relative to a benchmark.

    Beta measures systematic risk: how much the portfolio moves with
    the benchmark. Beta > 1 means more volatile than benchmark,
    beta < 1 means less volatile.

    Args:
        portfolio_returns: Series of portfolio period returns
        benchmark_returns: Series of benchmark period returns (same frequency)

    Returns:
        Portfolio beta (float)

    Raises:
        ValueError: If returns series are empty or misaligned
        ValueError: If benchmark variance is zero

    Example:
        >>> port_ret = pd.Series([0.01, -0.02, 0.015], index=pd.date_range('2021-01-01', periods=3))
        >>> bench_ret = pd.Series([0.005, -0.01, 0.008], index=pd.date_range('2021-01-01', periods=3))
        >>> beta = portfolio_beta(port_ret, bench_ret)
        >>> beta
        1.5
    """
    # Align the two series
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()

    if len(aligned) < 2:
        raise ValueError("Need at least 2 aligned observations to compute beta")

    port = aligned['portfolio']
    bench = aligned['benchmark']

    bench_var = bench.var()
    if bench_var == 0:
        raise ValueError("Benchmark variance is zero, cannot compute beta")

    # Beta = Cov(portfolio, benchmark) / Var(benchmark)
    covariance = port.cov(bench)
    beta = covariance / bench_var

    return float(beta)


def portfolio_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0
) -> float:
    """Calculate Jensen's alpha (excess return after adjusting for beta).

    Alpha measures the value added by active management after accounting
    for systematic risk. Positive alpha indicates outperformance vs
    risk-adjusted benchmark.

    Uses CAPM: Alpha = Rp - [Rf + Beta * (Rb - Rf)]

    Args:
        portfolio_returns: Series of portfolio period returns (e.g., daily)
        benchmark_returns: Series of benchmark period returns
        risk_free_rate: Annualized risk-free rate (default 0.0)

    Returns:
        Annualized alpha

    Raises:
        ValueError: If returns series are empty or misaligned

    Example:
        >>> port_ret = pd.Series([0.01, -0.01, 0.02])
        >>> bench_ret = pd.Series([0.005, -0.005, 0.01])
        >>> alpha = portfolio_alpha(port_ret, bench_ret)
    """
    # Align the two series
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()

    if len(aligned) < 2:
        raise ValueError("Need at least 2 aligned observations to compute alpha")

    port = aligned['portfolio']
    bench = aligned['benchmark']

    # Calculate beta
    beta = portfolio_beta(port, bench)

    # Calculate average returns
    avg_port = port.mean()
    avg_bench = bench.mean()

    # Assume returns are daily, annualize the risk-free rate to daily
    periods_per_year = 252  # Trading days
    rf_daily = risk_free_rate / periods_per_year

    # Alpha = Rp - [Rf + Beta * (Rb - Rf)]
    alpha_daily = avg_port - (rf_daily + beta * (avg_bench - rf_daily))

    # Annualize alpha
    alpha_annual = alpha_daily * periods_per_year

    return float(alpha_annual)


def concentration_herfindahl(weights: pd.Series) -> float:
    """Calculate Herfindahl-Hirschman Index (HHI) for portfolio concentration.

    HHI = sum(weight_i^2) where weights are fractions of portfolio.
    - HHI = 1.0: fully concentrated (single asset)
    - HHI = 1/N: equally weighted across N assets
    - Lower HHI = more diversified

    Args:
        weights: Series of portfolio weights (should sum to ~1.0)

    Returns:
        HHI concentration index (float between 0 and 1)

    Raises:
        ValueError: If weights is empty or all zero

    Example:
        >>> weights = pd.Series([0.5, 0.3, 0.2])  # 3 assets
        >>> hhi = concentration_herfindahl(weights)
        >>> hhi
        0.38  # (0.5^2 + 0.3^2 + 0.2^2)
    """
    w = weights[weights != 0]  # Ignore zero weights

    if len(w) == 0:
        raise ValueError("All weights are zero")

    # Normalize to ensure sum = 1 (handle numerical errors)
    w_norm = w / w.sum()

    hhi = (w_norm ** 2).sum()

    return float(hhi)


def effective_n_holdings(weights: pd.Series) -> float:
    """Calculate effective number of holdings (inverse of HHI).

    Effective N = 1 / HHI represents the "equivalent number of equal-weight
    positions". Useful for understanding true diversification.

    Args:
        weights: Series of portfolio weights

    Returns:
        Effective number of holdings (float)

    Example:
        >>> weights = pd.Series([0.25, 0.25, 0.25, 0.25])  # 4 equal positions
        >>> eff_n = effective_n_holdings(weights)
        >>> eff_n
        4.0
        >>> weights2 = pd.Series([0.7, 0.15, 0.15])  # Concentrated
        >>> effective_n_holdings(weights2)
        2.04  # Behaves like ~2 positions
    """
    hhi = concentration_herfindahl(weights)
    return float(1.0 / hhi)


def volatility_clustering(
    returns: pd.Series,
    window: int = 21
) -> pd.Series:
    """Detect volatility clustering using rolling standard deviation.

    Returns the rolling volatility, which can be used to identify periods
    of high vs low volatility. Persistent high or low values indicate clustering.

    Args:
        returns: Series of period returns
        window: Rolling window size (default 21 for ~monthly)

    Returns:
        Series of rolling standard deviations (annualized)

    Raises:
        ValueError: If returns is empty or window is too large

    Example:
        >>> returns = pd.Series(np.random.randn(100) * 0.01)
        >>> vol = volatility_clustering(returns, window=21)
        >>> vol.mean()  # Average rolling volatility
    """
    if len(returns) < window:
        raise ValueError(f"Returns length {len(returns)} < window {window}")

    # Calculate rolling std and annualize
    rolling_std = returns.rolling(window=window).std()

    # Annualize (assuming daily returns)
    annualized_vol = rolling_std * np.sqrt(252)

    return annualized_vol


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Calculate drawdown series (distance from running maximum).

    Drawdown at time t = (equity_t / running_max_t) - 1

    Args:
        equity: Series of equity curve values

    Returns:
        Series of drawdowns (negative values, 0 = at high water mark)

    Example:
        >>> equity = pd.Series([100, 110, 105, 120, 115])
        >>> dd = drawdown_series(equity)
        >>> dd.iloc[2]
        -0.0454  # (105/110 - 1)
    """
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown


def exposure_summary(
    weights_history: pd.DataFrame,
    ticker_metadata: Optional[Dict[str, Dict[str, Any]]] = None
) -> pd.DataFrame:
    """Summarize portfolio exposures over time.

    Provides statistics on how weights evolved, including average weight,
    max weight, frequency of holding, and turnover.

    Args:
        weights_history: DataFrame with dates as index, tickers as columns
        ticker_metadata: Optional dict mapping ticker -> {sector, country, etc.}

    Returns:
        DataFrame with summary statistics per ticker

    Example:
        >>> weights = pd.DataFrame({
        ...     'SPY': [0.5, 0.4, 0.3],
        ...     'TLT': [0.5, 0.6, 0.7]
        ... })
        >>> summary = exposure_summary(weights)
        >>> summary.loc['SPY', 'avg_weight']
        0.4
    """
    summary = pd.DataFrame(index=weights_history.columns)

    # Average weight when held
    summary['avg_weight'] = weights_history.mean()

    # Max weight
    summary['max_weight'] = weights_history.max()

    # Min weight (non-zero)
    summary['min_weight_nonzero'] = weights_history.replace(0, np.nan).min()

    # Frequency of being held (fraction of periods)
    summary['hold_frequency'] = (weights_history > 0).sum() / len(weights_history)

    # Standard deviation of weights
    summary['weight_std'] = weights_history.std()

    # Add metadata if provided
    if ticker_metadata is not None:
        for ticker in summary.index:
            if ticker in ticker_metadata:
                meta = ticker_metadata[ticker]
                for key, value in meta.items():
                    if key not in summary.columns:
                        summary[key] = None
                    summary.loc[ticker, key] = value

    return summary


def rolling_correlation(
    returns1: pd.Series,
    returns2: pd.Series,
    window: int = 63
) -> pd.Series:
    """Calculate rolling correlation between two return series.

    Useful for monitoring how portfolio correlation with benchmark evolves
    over time. Can detect regime changes or correlation breakdown.

    Args:
        returns1: First return series
        returns2: Second return series
        window: Rolling window size (default 63 ~ quarterly)

    Returns:
        Series of rolling correlations

    Raises:
        ValueError: If series cannot be aligned or window too large

    Example:
        >>> r1 = pd.Series(np.random.randn(200))
        >>> r2 = pd.Series(np.random.randn(200))
        >>> roll_corr = rolling_correlation(r1, r2, window=63)
    """
    # Align the series
    aligned = pd.DataFrame({
        'r1': returns1,
        'r2': returns2
    }).dropna()

    if len(aligned) < window:
        raise ValueError(f"Aligned length {len(aligned)} < window {window}")

    # Calculate rolling correlation
    roll_corr = aligned['r1'].rolling(window=window).corr(aligned['r2'])

    return roll_corr


def max_drawdown_duration(equity: pd.Series) -> int:
    """Calculate maximum drawdown duration in periods.

    Returns the longest consecutive period below the previous high water mark.

    Args:
        equity: Series of equity curve values

    Returns:
        Maximum drawdown duration in number of periods

    Example:
        >>> equity = pd.Series([100, 110, 105, 100, 95, 100, 105, 115])
        >>> max_drawdown_duration(equity)
        4  # Periods from peak at index 1 to recovery at index 6
    """
    dd = drawdown_series(equity)

    # Find all underwater periods (drawdown < 0)
    underwater = dd < 0

    if not underwater.any():
        return 0

    # Calculate consecutive underwater periods
    underwater_cumsum = (~underwater).cumsum()
    durations = underwater.groupby(underwater_cumsum).sum()

    max_duration = int(durations.max())

    return max_duration


def tail_ratio(returns: pd.Series, percentile: float = 5.0) -> float:
    """Calculate tail ratio (right tail / left tail).

    Measures the ratio of gains in the right tail vs losses in the left tail.
    - Tail ratio > 1: larger positive outliers than negative
    - Tail ratio < 1: larger negative outliers than positive

    Args:
        returns: Series of period returns
        percentile: Percentile for defining tails (default 5.0 for top/bottom 5%)

    Returns:
        Tail ratio (float)

    Raises:
        ValueError: If returns is empty or percentile invalid

    Example:
        >>> returns = pd.Series([-0.05, -0.02, 0.01, 0.015, 0.08])
        >>> tail_ratio(returns, percentile=20.0)
        # Compares top 20% gains vs bottom 20% losses
    """
    if len(returns) == 0:
        raise ValueError("Returns is empty")

    if not (0 < percentile < 50):
        raise ValueError("Percentile must be between 0 and 50")

    r = returns.dropna()

    # Calculate percentiles
    right_tail = np.percentile(r, 100 - percentile)  # Top percentile
    left_tail = np.percentile(r, percentile)  # Bottom percentile

    # Tail ratio = abs(right_tail) / abs(left_tail)
    if left_tail == 0:
        return np.inf if right_tail > 0 else 0.0

    ratio = abs(right_tail) / abs(left_tail)

    return float(ratio)
