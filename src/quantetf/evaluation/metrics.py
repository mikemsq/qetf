"""Performance metrics for strategy evaluation.

This module provides comprehensive metrics for analyzing backtest
and live trading performance, including risk-adjusted returns,
drawdown analysis, and tail risk measures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate Compound Annual Growth Rate (CAGR).

    Args:
        equity: Series of equity curve values
        periods_per_year: Trading periods per year (default 252 for daily)

    Returns:
        Annualized compound growth rate
    """
    if len(equity) < 2:
        return 0.0
    years = len(equity) / float(periods_per_year)
    return float(equity.iloc[-1] ** (1.0 / years) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    """Calculate maximum drawdown from peak equity.

    Args:
        equity: Series of equity curve values

    Returns:
        Maximum drawdown as negative number (e.g., -0.15 = 15% drawdown)
    """
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate Sharpe ratio (annualized).

    Args:
        returns: Series of period returns
        periods_per_year: Trading periods per year (default 252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    r = returns.dropna()
    if r.std() == 0 or len(r) < 2:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate Sortino ratio (return / downside deviation).

    Measures risk-adjusted return using only downside volatility.
    Better than Sharpe for strategies with asymmetric returns as it
    only penalizes downside volatility.

    Args:
        returns: Series of period returns (e.g., daily)
        risk_free_rate: Annualized risk-free rate (default 0.0)
        periods_per_year: Trading periods per year (default 252 for daily)

    Returns:
        Sortino ratio (annualized)

    Raises:
        ValueError: If returns is empty or all NaN

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02])
        >>> sortino_ratio(returns)
        1.23
    """
    r = returns.dropna()

    if len(r) == 0:
        raise ValueError("Returns series is empty or all NaN")

    if len(r) < 2:
        return 0.0

    # Calculate excess return
    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = r - rf_per_period

    # Calculate downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        # No downside - return a large positive number or inf
        return np.inf if excess_returns.mean() > 0 else 0.0

    downside_std = downside_returns.std()

    if downside_std == 0:
        return 0.0

    # Annualize
    annualized_return = excess_returns.mean() * periods_per_year
    annualized_downside_std = downside_std * np.sqrt(periods_per_year)

    return float(annualized_return / annualized_downside_std)


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown).

    Measures annualized return per unit of maximum drawdown.
    Popular metric in hedge fund industry for assessing return
    vs worst-case drawdown risk.

    Args:
        returns: Series of period returns (e.g., daily)
        periods_per_year: Trading periods per year (default 252 for daily)

    Returns:
        Calmar ratio

    Raises:
        ValueError: If returns is empty or all NaN

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])
        >>> calmar_ratio(returns)
        2.45
    """
    r = returns.dropna()

    if len(r) == 0:
        raise ValueError("Returns series is empty or all NaN")

    if len(r) < 2:
        return 0.0

    # Calculate annualized return
    annualized_return = r.mean() * periods_per_year

    # Calculate max drawdown from equity curve
    equity = (1 + r).cumprod()
    mdd = abs(max_drawdown(equity))

    if mdd == 0:
        # No drawdown occurred
        return np.inf if annualized_return > 0 else 0.0

    return float(annualized_return / mdd)


def win_rate(returns: pd.Series) -> float:
    """Calculate win rate (percentage of positive return periods).

    Args:
        returns: Series of period returns

    Returns:
        Win rate as percentage (0-100)

    Raises:
        ValueError: If returns is empty or all NaN

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02])
        >>> win_rate(returns)
        60.0  # 3 out of 5 positive
    """
    r = returns.dropna()

    if len(r) == 0:
        raise ValueError("Returns series is empty or all NaN")

    # Count positive returns (exclude zeros)
    positive_count = (r > 0).sum()
    total_count = len(r)

    return float(positive_count / total_count * 100.0)


def value_at_risk(
    returns: pd.Series,
    confidence_level: float = 0.95,
    periods_per_year: int = 252
) -> float:
    """Calculate Value at Risk (VaR) at given confidence level.

    VaR estimates the maximum loss over a time period at a given
    confidence level. For example, 95% VaR of -2% means there's a
    5% chance of losing more than 2% in a period.

    Uses historical simulation (empirical percentile method).

    Args:
        returns: Series of period returns
        confidence_level: Confidence level (0.95 = 95%)
        periods_per_year: Trading periods per year (for reference, not used in calculation)

    Returns:
        VaR as a negative number (e.g., -0.02 = 2% loss at given confidence level)

    Raises:
        ValueError: If returns is empty or all NaN
        ValueError: If confidence_level not in (0, 1)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.03, 0.02])
        >>> value_at_risk(returns, confidence_level=0.95)
        -0.025  # 95% VaR is 2.5% loss
    """
    r = returns.dropna()

    if len(r) == 0:
        raise ValueError("Returns series is empty or all NaN")

    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be between 0 and 1, got {confidence_level}")

    # Calculate percentile (lower tail)
    alpha = 1 - confidence_level
    var = float(r.quantile(alpha))

    return var


def conditional_value_at_risk(
    returns: pd.Series,
    confidence_level: float = 0.95,
    periods_per_year: int = 252
) -> float:
    """Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

    CVaR is the expected loss given that VaR threshold is exceeded.
    It captures tail risk better than VaR alone by averaging all
    losses beyond the VaR threshold.

    Args:
        returns: Series of period returns
        confidence_level: Confidence level (0.95 = 95%)
        periods_per_year: Trading periods per year (for reference, not used)

    Returns:
        CVaR as a negative number (expected loss in tail)

    Raises:
        ValueError: If returns is empty or all NaN
        ValueError: If confidence_level not in (0, 1)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.03, 0.02])
        >>> conditional_value_at_risk(returns, confidence_level=0.95)
        -0.030  # Average loss when VaR exceeded
    """
    r = returns.dropna()

    if len(r) == 0:
        raise ValueError("Returns series is empty or all NaN")

    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be between 0 and 1, got {confidence_level}")

    # Get VaR threshold
    var = value_at_risk(r, confidence_level, periods_per_year)

    # Calculate mean of returns worse than VaR
    tail_returns = r[r <= var]

    if len(tail_returns) == 0:
        # No returns in tail (very small dataset)
        return var

    cvar = float(tail_returns.mean())

    return cvar


def rolling_sharpe_ratio(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.0,
    min_periods: int = 126
) -> pd.Series:
    """Calculate rolling Sharpe ratio over time.

    Computes Sharpe ratio over a rolling window. Useful for
    visualizing performance stability and identifying periods
    of strong/weak performance.

    Args:
        returns: Series of period returns (e.g., daily)
        window: Rolling window size in periods (default 252 = 1 year daily)
        risk_free_rate: Annualized risk-free rate (default 0.0)
        min_periods: Minimum periods required for calculation

    Returns:
        Series of rolling Sharpe ratios (same index as returns)

    Raises:
        ValueError: If returns is empty or all NaN

    Example:
        >>> returns = pd.Series([...])  # 500 days of returns
        >>> rolling_sharpe = rolling_sharpe_ratio(returns, window=252)
        >>> rolling_sharpe.plot()  # Visualize Sharpe over time
    """
    r = returns.dropna()

    if len(r) == 0:
        raise ValueError("Returns series is empty or all NaN")

    # Calculate rolling mean and std
    rolling_mean = r.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = r.rolling(window=window, min_periods=min_periods).std()

    # Handle zero std
    rolling_std = rolling_std.replace(0, np.nan)

    # Calculate Sharpe ratio (annualized)
    rf_per_period = risk_free_rate / 252  # Assume 252 periods per year for now
    rolling_sharpe = (rolling_mean - rf_per_period) / rolling_std * np.sqrt(window)

    return rolling_sharpe


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """Calculate Information Ratio vs benchmark.

    Measures excess return per unit of tracking error.
    Indicates skill in active management - how much additional
    return is generated per unit of deviation from benchmark.

    Formula: (R_p - R_b) / tracking_error
    Where tracking_error = std(R_p - R_b)

    Args:
        returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns (must be aligned with returns)
        periods_per_year: Trading periods per year (default 252 for daily)

    Returns:
        Information ratio (annualized)

    Raises:
        ValueError: If returns and benchmark have different lengths
        ValueError: If tracking error is zero (perfect tracking)
        ValueError: If returns is empty or all NaN

    Example:
        >>> portfolio_returns = pd.Series([0.01, 0.02, -0.01])
        >>> spy_returns = pd.Series([0.008, 0.015, -0.008])
        >>> information_ratio(portfolio_returns, spy_returns)
        0.85
    """
    r = returns.dropna()
    b = benchmark_returns.dropna()

    if len(r) == 0:
        raise ValueError("Returns series is empty or all NaN")

    # Align indices
    aligned_returns = r.reindex(b.index)
    aligned_benchmark = b.reindex(r.index)

    # Drop any NaN after alignment
    mask = ~(aligned_returns.isna() | aligned_benchmark.isna())
    aligned_returns = aligned_returns[mask]
    aligned_benchmark = aligned_benchmark[mask]

    if len(aligned_returns) == 0:
        raise ValueError("No overlapping returns between portfolio and benchmark")

    if len(aligned_returns) < 2:
        return 0.0

    # Calculate excess returns
    excess_returns = aligned_returns - aligned_benchmark

    # Calculate tracking error
    tracking_error = excess_returns.std()

    if tracking_error == 0:
        # Perfect tracking
        return 0.0

    # Annualize
    annualized_excess = excess_returns.mean() * periods_per_year
    annualized_te = tracking_error * np.sqrt(periods_per_year)

    return float(annualized_excess / annualized_te)


def calculate_active_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0
) -> dict[str, float]:
    """Calculate comprehensive active performance metrics vs benchmark.

    This is the STANDARD format for all performance reporting in QuantETF.
    The goal is to beat SPY, so active metrics are the primary focus.

    Args:
        strategy_returns: Strategy period returns (e.g., daily)
        benchmark_returns: Benchmark period returns (typically SPY)
        periods_per_year: Trading periods per year (default 252 for daily)
        risk_free_rate: Annualized risk-free rate (default 0.0)

    Returns:
        Dictionary with comprehensive active metrics:

        Strategy Metrics:
        - strategy_total_return: Cumulative strategy return
        - strategy_sharpe: Strategy Sharpe ratio
        - strategy_sortino: Strategy Sortino ratio
        - strategy_max_dd: Strategy maximum drawdown
        - strategy_cagr: Strategy CAGR

        Benchmark Metrics:
        - benchmark_total_return: Cumulative benchmark return
        - benchmark_sharpe: Benchmark Sharpe ratio
        - benchmark_sortino: Benchmark Sortino ratio
        - benchmark_max_dd: Benchmark maximum drawdown
        - benchmark_cagr: Benchmark CAGR

        Active Metrics (THE KEY METRICS):
        - active_return: Total excess return vs benchmark
        - active_return_ann: Annualized excess return
        - tracking_error: Volatility of excess returns (annualized)
        - information_ratio: Active return / tracking error
        - beta: Strategy beta to benchmark
        - alpha: Jensen's alpha (risk-adjusted excess return)
        - sharpe_difference: Strategy Sharpe - Benchmark Sharpe
        - win_rate: % of periods where strategy beat benchmark

    Raises:
        ValueError: If returns series are empty or misaligned

    Example:
        >>> strategy_rets = pd.Series([0.01, 0.02, -0.01, 0.015])
        >>> spy_rets = pd.Series([0.008, 0.015, -0.008, 0.012])
        >>> metrics = calculate_active_metrics(strategy_rets, spy_rets)
        >>> print(f"Beat SPY by {metrics['active_return']:.2%}")
        >>> print(f"Information Ratio: {metrics['information_ratio']:.2f}")
    """
    # Input validation
    strat = strategy_returns.dropna()
    bench = benchmark_returns.dropna()

    if len(strat) == 0:
        raise ValueError("Strategy returns series is empty or all NaN")
    if len(bench) == 0:
        raise ValueError("Benchmark returns series is empty or all NaN")

    # Align indices
    common_index = strat.index.intersection(bench.index)
    if len(common_index) == 0:
        raise ValueError("No overlapping returns between strategy and benchmark")

    aligned_strat = strat[common_index]
    aligned_bench = bench[common_index]

    # Calculate strategy metrics
    strat_equity = (1 + aligned_strat).cumprod()
    bench_equity = (1 + aligned_bench).cumprod()

    strategy_metrics = {
        'strategy_total_return': float(strat_equity.iloc[-1] - 1.0),
        'strategy_sharpe': sharpe(aligned_strat, periods_per_year),
        'strategy_sortino': sortino_ratio(aligned_strat, risk_free_rate, periods_per_year),
        'strategy_max_dd': max_drawdown(strat_equity),
        'strategy_cagr': cagr(strat_equity, periods_per_year),
    }

    # Calculate benchmark metrics
    benchmark_metrics = {
        'benchmark_total_return': float(bench_equity.iloc[-1] - 1.0),
        'benchmark_sharpe': sharpe(aligned_bench, periods_per_year),
        'benchmark_sortino': sortino_ratio(aligned_bench, risk_free_rate, periods_per_year),
        'benchmark_max_dd': max_drawdown(bench_equity),
        'benchmark_cagr': cagr(bench_equity, periods_per_year),
    }

    # Calculate active metrics
    excess_returns = aligned_strat - aligned_bench
    tracking_error = excess_returns.std() * np.sqrt(periods_per_year)

    # Information ratio
    ir = information_ratio(aligned_strat, aligned_bench, periods_per_year)

    # Beta and alpha (using linear regression)
    if aligned_bench.std() > 0:
        covariance = aligned_strat.cov(aligned_bench)
        beta = covariance / aligned_bench.var()

        # Jensen's alpha: alpha = Rp - [Rf + beta * (Rb - Rf)]
        rf_per_period = risk_free_rate / periods_per_year
        alpha = (aligned_strat.mean() - rf_per_period) - beta * (aligned_bench.mean() - rf_per_period)
        alpha_annualized = alpha * periods_per_year
    else:
        beta = 0.0
        alpha_annualized = 0.0

    # Win rate (% periods beating benchmark)
    win_rate_pct = float((excess_returns > 0).sum() / len(excess_returns) * 100.0)

    active_metrics = {
        'active_return': float(strategy_metrics['strategy_total_return'] - benchmark_metrics['benchmark_total_return']),
        'active_return_ann': float(excess_returns.mean() * periods_per_year),
        'tracking_error': float(tracking_error),
        'information_ratio': float(ir),
        'beta': float(beta),
        'alpha': float(alpha_annualized),
        'sharpe_difference': float(strategy_metrics['strategy_sharpe'] - benchmark_metrics['benchmark_sharpe']),
        'win_rate': float(win_rate_pct),
    }

    # Combine all metrics
    return {**strategy_metrics, **benchmark_metrics, **active_metrics}
