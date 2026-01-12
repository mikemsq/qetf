"""Tests for advanced performance metrics."""

import numpy as np
import pandas as pd
import pytest

from quantetf.evaluation.metrics import (
    sortino_ratio,
    calmar_ratio,
    win_rate,
    value_at_risk,
    conditional_value_at_risk,
    rolling_sharpe_ratio,
    information_ratio,
    calculate_active_metrics,
)


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_sortino_typical_case(self):
        """Test Sortino with mixed positive and negative returns."""
        returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02, 0.01, -0.01, 0.005])
        ratio = sortino_ratio(returns)
        assert isinstance(ratio, float)
        assert ratio > 0  # Positive expected return

    def test_sortino_all_positive(self):
        """Test Sortino when no downside (all gains)."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.005, 0.01])
        ratio = sortino_ratio(returns)
        # Should return inf since no downside volatility
        assert np.isinf(ratio) and ratio > 0

    def test_sortino_all_negative(self):
        """Test Sortino with all negative returns."""
        returns = pd.Series([-0.01, -0.02, -0.015, -0.005])
        ratio = sortino_ratio(returns)
        # Should be negative (negative return, positive downside std)
        assert ratio < 0

    def test_sortino_empty_raises(self):
        """Test Sortino with empty series raises ValueError."""
        with pytest.raises(ValueError, match="empty or all NaN"):
            sortino_ratio(pd.Series([]))

    def test_sortino_all_nan_raises(self):
        """Test Sortino with all NaN raises ValueError."""
        with pytest.raises(ValueError, match="empty or all NaN"):
            sortino_ratio(pd.Series([np.nan, np.nan, np.nan]))

    def test_sortino_single_value(self):
        """Test Sortino with single value returns 0."""
        returns = pd.Series([0.01])
        ratio = sortino_ratio(returns)
        assert ratio == 0.0


class TestCalmarRatio:
    """Tests for Calmar ratio calculation."""

    def test_calmar_typical_case(self):
        """Test Calmar with returns that have drawdown."""
        # Create returns with known drawdown and positive mean
        returns = pd.Series([0.02, -0.03, 0.02, 0.01, -0.01, 0.02])
        ratio = calmar_ratio(returns)
        assert isinstance(ratio, float)
        # Positive average returns should give positive Calmar
        assert ratio > 0

    def test_calmar_no_drawdown(self):
        """Test Calmar when all returns are positive (no drawdown)."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.01])
        ratio = calmar_ratio(returns)
        # Should return inf (positive return, zero drawdown)
        assert np.isinf(ratio) and ratio > 0

    def test_calmar_negative_returns(self):
        """Test Calmar with negative average returns."""
        returns = pd.Series([-0.01, -0.02, 0.005, -0.01])
        ratio = calmar_ratio(returns)
        # Negative return / positive drawdown = negative
        assert ratio < 0

    def test_calmar_empty_raises(self):
        """Test Calmar with empty series raises ValueError."""
        with pytest.raises(ValueError, match="empty or all NaN"):
            calmar_ratio(pd.Series([]))

    def test_calmar_single_value(self):
        """Test Calmar with single value returns 0."""
        returns = pd.Series([0.01])
        ratio = calmar_ratio(returns)
        assert ratio == 0.0


class TestWinRate:
    """Tests for win rate calculation."""

    def test_win_rate_mixed(self):
        """Test win rate with mixed wins and losses."""
        returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02])
        wr = win_rate(returns)
        # 3 out of 5 positive = 60%
        assert wr == 60.0

    def test_win_rate_all_wins(self):
        """Test win rate when all returns are positive."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.005])
        wr = win_rate(returns)
        assert wr == 100.0

    def test_win_rate_all_losses(self):
        """Test win rate when all returns are negative."""
        returns = pd.Series([-0.01, -0.02, -0.015, -0.005])
        wr = win_rate(returns)
        assert wr == 0.0

    def test_win_rate_with_zeros(self):
        """Test win rate with zero returns (zeros not counted as wins)."""
        returns = pd.Series([0.01, 0.0, 0.0, -0.01])
        wr = win_rate(returns)
        # Only 1 out of 4 is positive
        assert wr == 25.0

    def test_win_rate_empty_raises(self):
        """Test win rate with empty series raises ValueError."""
        with pytest.raises(ValueError, match="empty or all NaN"):
            win_rate(pd.Series([]))


class TestValueAtRisk:
    """Tests for Value at Risk (VaR) calculation."""

    def test_var_95_confidence(self):
        """Test VaR at 95% confidence with known distribution."""
        # Create normal distribution
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        var_95 = value_at_risk(returns, confidence_level=0.95)

        # VaR should be negative (it's a loss)
        assert var_95 < 0

        # Approximately 5% of returns should be worse than VaR
        worse_than_var = (returns < var_95).sum()
        pct_worse = worse_than_var / len(returns)
        # Should be around 5% with some tolerance
        assert 0.03 < pct_worse < 0.07

    def test_var_different_confidence_levels(self):
        """Test that higher confidence gives more extreme VaR."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))

        var_90 = value_at_risk(returns, confidence_level=0.90)
        var_95 = value_at_risk(returns, confidence_level=0.95)
        var_99 = value_at_risk(returns, confidence_level=0.99)

        # Higher confidence should give more extreme (more negative) VaR
        assert var_99 < var_95 < var_90

    def test_var_invalid_confidence_raises(self):
        """Test that invalid confidence level raises ValueError."""
        returns = pd.Series([0.01, -0.02, 0.015])

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            value_at_risk(returns, confidence_level=1.5)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            value_at_risk(returns, confidence_level=-0.1)

    def test_var_empty_raises(self):
        """Test VaR with empty series raises ValueError."""
        with pytest.raises(ValueError, match="empty or all NaN"):
            value_at_risk(pd.Series([]))

    def test_var_all_positive(self):
        """Test VaR with all positive returns (VaR could be positive)."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.025])
        var_95 = value_at_risk(returns, confidence_level=0.95)
        # With all positive returns, 5th percentile could still be positive
        assert isinstance(var_95, float)


class TestConditionalValueAtRisk:
    """Tests for Conditional Value at Risk (CVaR) calculation."""

    def test_cvar_95_confidence(self):
        """Test CVaR at 95% confidence."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))

        var_95 = value_at_risk(returns, confidence_level=0.95)
        cvar_95 = conditional_value_at_risk(returns, confidence_level=0.95)

        # CVaR should be negative (expected loss in tail)
        assert cvar_95 < 0

        # CVaR should be more extreme (more negative) than VaR
        assert cvar_95 <= var_95

    def test_cvar_is_average_of_tail(self):
        """Test that CVaR is mean of returns worse than VaR."""
        returns = pd.Series([0.01, -0.01, -0.02, -0.03, -0.04, 0.02, 0.01, -0.01, 0.015, 0.005])

        var_90 = value_at_risk(returns, confidence_level=0.90)
        cvar_90 = conditional_value_at_risk(returns, confidence_level=0.90)

        # Manually calculate CVaR
        tail_returns = returns[returns <= var_90]
        expected_cvar = tail_returns.mean()

        assert abs(cvar_90 - expected_cvar) < 1e-10

    def test_cvar_invalid_confidence_raises(self):
        """Test that invalid confidence level raises ValueError."""
        returns = pd.Series([0.01, -0.02, 0.015])

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            conditional_value_at_risk(returns, confidence_level=1.5)

    def test_cvar_empty_raises(self):
        """Test CVaR with empty series raises ValueError."""
        with pytest.raises(ValueError, match="empty or all NaN"):
            conditional_value_at_risk(pd.Series([]))


class TestRollingSharpeRatio:
    """Tests for rolling Sharpe ratio calculation."""

    def test_rolling_sharpe_basic(self):
        """Test rolling Sharpe with sufficient data."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 500))

        rolling_sharpe = rolling_sharpe_ratio(returns, window=252, min_periods=126)

        # Should return a Series
        assert isinstance(rolling_sharpe, pd.Series)
        # Should have same length as input
        assert len(rolling_sharpe) == len(returns)
        # First values should be NaN (before min_periods)
        assert rolling_sharpe.iloc[:125].isna().all()
        # Later values should be non-NaN
        assert rolling_sharpe.iloc[126:].notna().any()

    def test_rolling_sharpe_short_window(self):
        """Test rolling Sharpe with short window."""
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))

        rolling_sharpe = rolling_sharpe_ratio(returns, window=20, min_periods=10)

        # Should have values after min_periods
        assert rolling_sharpe.iloc[10:].notna().any()

    def test_rolling_sharpe_insufficient_data(self):
        """Test rolling Sharpe when data shorter than min_periods."""
        returns = pd.Series(np.random.normal(0.001, 0.01, 50))

        rolling_sharpe = rolling_sharpe_ratio(returns, window=252, min_periods=126)

        # All values should be NaN (insufficient data)
        assert rolling_sharpe.isna().all()

    def test_rolling_sharpe_empty_raises(self):
        """Test rolling Sharpe with empty series raises ValueError."""
        with pytest.raises(ValueError, match="empty or all NaN"):
            rolling_sharpe_ratio(pd.Series([]))


class TestInformationRatio:
    """Tests for Information Ratio calculation."""

    def test_information_ratio_basic(self):
        """Test Information Ratio with portfolio outperforming benchmark."""
        # Portfolio with higher returns than benchmark
        portfolio_returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.01])
        benchmark_returns = pd.Series([0.008, 0.015, -0.008, 0.01, 0.008])

        ir = information_ratio(portfolio_returns, benchmark_returns)

        # Should be positive (outperformance)
        assert ir > 0

    def test_information_ratio_underperformance(self):
        """Test Information Ratio with portfolio underperforming."""
        portfolio_returns = pd.Series([0.005, 0.01, -0.015, 0.008, 0.005])
        benchmark_returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.01])

        ir = information_ratio(portfolio_returns, benchmark_returns)

        # Should be negative (underperformance)
        assert ir < 0

    def test_information_ratio_perfect_tracking(self):
        """Test Information Ratio when perfectly tracking benchmark."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.01])
        benchmark_returns = returns.copy()

        ir = information_ratio(returns, benchmark_returns)

        # Should be zero (perfect tracking, no tracking error)
        assert ir == 0.0

    def test_information_ratio_misaligned_indices(self):
        """Test Information Ratio handles misaligned indices."""
        # Different indices
        portfolio_returns = pd.Series(
            [0.01, 0.02, -0.01, 0.015],
            index=pd.date_range('2024-01-01', periods=4)
        )
        benchmark_returns = pd.Series(
            [0.008, 0.015, -0.008, 0.01],
            index=pd.date_range('2024-01-01', periods=4)
        )

        ir = information_ratio(portfolio_returns, benchmark_returns)

        # Should handle alignment and calculate ratio
        assert isinstance(ir, float)

    def test_information_ratio_empty_raises(self):
        """Test Information Ratio with empty series raises ValueError."""
        with pytest.raises(ValueError, match="empty or all NaN"):
            information_ratio(pd.Series([]), pd.Series([0.01, 0.02]))

    def test_information_ratio_no_overlap_raises(self):
        """Test Information Ratio with no overlapping data raises ValueError."""
        portfolio_returns = pd.Series(
            [0.01, 0.02],
            index=pd.date_range('2024-01-01', periods=2)
        )
        benchmark_returns = pd.Series(
            [0.008, 0.015],
            index=pd.date_range('2024-02-01', periods=2)
        )

        with pytest.raises(ValueError, match="No overlapping returns"):
            information_ratio(portfolio_returns, benchmark_returns)


class TestMetricsIntegration:
    """Integration tests for metrics working together."""

    def test_all_metrics_on_sample_returns(self):
        """Test that all metrics can be calculated on sample data."""
        np.random.seed(42)
        # Generate realistic returns
        returns = pd.Series(np.random.normal(0.0005, 0.01, 252))  # 1 year daily
        benchmark_returns = pd.Series(np.random.normal(0.0004, 0.009, 252))

        # Calculate all metrics (should not raise)
        sortino = sortino_ratio(returns)
        calmar = calmar_ratio(returns)
        wr = win_rate(returns)
        var = value_at_risk(returns, 0.95)
        cvar = conditional_value_at_risk(returns, 0.95)
        rolling_sharpe = rolling_sharpe_ratio(returns, window=60, min_periods=30)
        ir = information_ratio(returns, benchmark_returns)

        # All should return valid numbers
        assert isinstance(sortino, float)
        assert isinstance(calmar, float)
        assert isinstance(wr, float)
        assert isinstance(var, float)
        assert isinstance(cvar, float)
        assert isinstance(rolling_sharpe, pd.Series)
        assert isinstance(ir, float)

    def test_metrics_with_real_backtest_structure(self):
        """Test metrics with data structure similar to real backtest."""
        # Simulate backtest equity curve
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        np.random.seed(42)
        daily_returns = pd.Series(
            np.random.normal(0.0005, 0.01, 252),
            index=dates
        )

        # Test metrics work with datetime index
        sortino = sortino_ratio(daily_returns)
        wr = win_rate(daily_returns)
        var_95 = value_at_risk(daily_returns, 0.95)

        assert sortino != 0
        assert 0 <= wr <= 100
        assert var_95 < 0  # VaR should be negative (loss)


class TestCalculateActiveMetrics:
    """Tests for calculate_active_metrics function."""

    def test_calculate_active_metrics_basic(self):
        """Test basic active metrics calculation."""
        strategy = pd.Series([0.01, 0.02, -0.01, 0.015, 0.02])
        benchmark = pd.Series([0.008, 0.015, -0.008, 0.012, 0.018])

        metrics = calculate_active_metrics(strategy, benchmark)

        # Check all keys present
        assert 'active_return' in metrics
        assert 'information_ratio' in metrics
        assert 'beta' in metrics
        assert 'alpha' in metrics
        assert 'tracking_error' in metrics
        assert 'win_rate' in metrics
        assert 'strategy_total_return' in metrics
        assert 'benchmark_total_return' in metrics
        assert 'strategy_sharpe' in metrics
        assert 'benchmark_sharpe' in metrics

        # Check strategy outperformed
        assert metrics['active_return'] > 0
        assert isinstance(metrics['information_ratio'], float)
        assert isinstance(metrics['beta'], float)

    def test_calculate_active_metrics_underperformance(self):
        """Test when strategy underperforms benchmark."""
        strategy = pd.Series([0.005, 0.01, -0.015, 0.008, 0.01])
        benchmark = pd.Series([0.01, 0.02, -0.01, 0.015, 0.02])

        metrics = calculate_active_metrics(strategy, benchmark)

        assert metrics['active_return'] < 0
        assert metrics['win_rate'] < 50.0
        assert metrics['strategy_total_return'] < metrics['benchmark_total_return']

    def test_calculate_active_metrics_perfect_tracking(self):
        """Test when strategy perfectly tracks benchmark."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.02])

        metrics = calculate_active_metrics(returns, returns)

        assert np.isclose(metrics['active_return'], 0.0, atol=1e-10)
        assert np.isclose(metrics['tracking_error'], 0.0, atol=1e-10)
        assert np.isclose(metrics['beta'], 1.0, atol=1e-6)
        # Win rate should be 0% when perfectly tracking (excess returns are exactly 0, not > 0)
        assert metrics['win_rate'] == 0.0

    def test_calculate_active_metrics_empty_strategy(self):
        """Test error handling for empty strategy returns."""
        empty_strategy = pd.Series([])
        benchmark = pd.Series([0.01, 0.02, 0.015])

        with pytest.raises(ValueError, match="Strategy returns series is empty"):
            calculate_active_metrics(empty_strategy, benchmark)

    def test_calculate_active_metrics_empty_benchmark(self):
        """Test error handling for empty benchmark returns."""
        strategy = pd.Series([0.01, 0.02, 0.015])
        empty_benchmark = pd.Series([])

        with pytest.raises(ValueError, match="Benchmark returns series is empty"):
            calculate_active_metrics(strategy, empty_benchmark)

    def test_calculate_active_metrics_no_overlap(self):
        """Test error handling when there's no overlap in indices."""
        strategy = pd.Series([0.01, 0.02, 0.015], index=pd.date_range('2024-01-01', periods=3))
        benchmark = pd.Series([0.01, 0.02, 0.015], index=pd.date_range('2024-02-01', periods=3))

        with pytest.raises(ValueError, match="No overlapping returns"):
            calculate_active_metrics(strategy, benchmark)

    def test_calculate_active_metrics_with_nans(self):
        """Test handling of NaN values in returns."""
        strategy = pd.Series([0.01, np.nan, 0.02, -0.01, 0.015])
        benchmark = pd.Series([0.008, 0.015, np.nan, -0.008, 0.012])

        # Should drop NaNs and align properly
        metrics = calculate_active_metrics(strategy, benchmark)

        # Should have valid metrics based on overlapping non-NaN values
        assert np.isfinite(metrics['active_return'])
        assert np.isfinite(metrics['information_ratio'])

    def test_calculate_active_metrics_with_datetime_index(self):
        """Test with datetime-indexed returns (realistic scenario)."""
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        np.random.seed(42)
        strategy = pd.Series(np.random.normal(0.0006, 0.01, 252), index=dates)
        benchmark = pd.Series(np.random.normal(0.0005, 0.009, 252), index=dates)

        metrics = calculate_active_metrics(strategy, benchmark, periods_per_year=252)

        # All metrics should be valid
        for key in ['active_return', 'information_ratio', 'beta', 'alpha', 'tracking_error']:
            assert np.isfinite(metrics[key])

        # Beta should be reasonable (typically between 0.5 and 1.5 for equity strategies)
        assert -2.0 < metrics['beta'] < 3.0

    def test_calculate_active_metrics_risk_free_rate(self):
        """Test with non-zero risk-free rate."""
        strategy = pd.Series([0.02, 0.03, -0.01, 0.025, 0.03])
        benchmark = pd.Series([0.015, 0.025, -0.008, 0.020, 0.028])

        metrics_rf0 = calculate_active_metrics(strategy, benchmark, risk_free_rate=0.0)
        metrics_rf2 = calculate_active_metrics(strategy, benchmark, risk_free_rate=0.02)

        # Alpha should differ with different risk-free rates
        assert metrics_rf0['alpha'] != metrics_rf2['alpha']

    def test_calculate_active_metrics_all_values_reasonable(self):
        """Test that all returned values are reasonable."""
        np.random.seed(42)
        strategy = pd.Series(np.random.normal(0.0008, 0.015, 500))
        benchmark = pd.Series(np.random.normal(0.0006, 0.012, 500))

        metrics = calculate_active_metrics(strategy, benchmark)

        # Win rate should be between 0 and 100
        assert 0 <= metrics['win_rate'] <= 100

        # Total returns should be reasonable (e.g., between -50% and +200%)
        assert -0.5 < metrics['strategy_total_return'] < 2.0
        assert -0.5 < metrics['benchmark_total_return'] < 2.0

        # Sharpe ratios should be reasonable (typically -2 to +4 for daily data)
        assert -3.0 < metrics['strategy_sharpe'] < 5.0
        assert -3.0 < metrics['benchmark_sharpe'] < 5.0

        # Max drawdown should be negative and reasonable
        assert -1.0 < metrics['strategy_max_dd'] <= 0.0
        assert -1.0 < metrics['benchmark_max_dd'] <= 0.0
