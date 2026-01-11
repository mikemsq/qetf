"""Tests for risk analytics module."""

import numpy as np
import pandas as pd
import pytest

from quantetf.evaluation.risk_analytics import (
    holdings_correlation_matrix,
    portfolio_beta,
    portfolio_alpha,
    concentration_herfindahl,
    effective_n_holdings,
    volatility_clustering,
    drawdown_series,
    exposure_summary,
    rolling_correlation,
    max_drawdown_duration,
    tail_ratio
)


class TestHoldingsCorrelationMatrix:
    """Test holdings correlation matrix calculation."""

    def test_basic_correlation(self):
        """Test basic correlation calculation."""
        holdings = pd.DataFrame({
            'A': [100, 105, 110, 115],
            'B': [50, 48, 46, 44]
        })
        corr = holdings_correlation_matrix(holdings)

        # A increases, B decreases -> negative correlation
        assert corr.loc['A', 'B'] < 0
        assert corr.loc['A', 'A'] == pytest.approx(1.0)
        assert corr.shape == (2, 2)

    def test_empty_holdings_raises(self):
        """Test empty DataFrame raises error."""
        holdings = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            holdings_correlation_matrix(holdings)

    def test_single_ticker_raises(self):
        """Test single ticker raises error."""
        holdings = pd.DataFrame({'A': [100, 110, 120]})
        with pytest.raises(ValueError, match="at least 2 tickers"):
            holdings_correlation_matrix(holdings)

    def test_filters_zero_holdings(self):
        """Test filters out tickers never held."""
        holdings = pd.DataFrame({
            'A': [100, 105, 110],
            'B': [50, 48, 46],
            'C': [0, 0, 0]  # Never held
        })
        corr = holdings_correlation_matrix(holdings)

        # Should only include A and B
        assert 'A' in corr.index
        assert 'B' in corr.index
        assert 'C' not in corr.index

    def test_all_zero_raises(self):
        """Test all zero holdings raises error."""
        holdings = pd.DataFrame({
            'A': [0, 0, 0],
            'B': [0, 0, 0]
        })
        with pytest.raises(ValueError, match="at least 2 active tickers"):
            holdings_correlation_matrix(holdings)


class TestPortfolioBeta:
    """Test portfolio beta calculation."""

    def test_beta_equals_one(self):
        """Test beta = 1 when portfolio matches benchmark."""
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])
        beta = portfolio_beta(returns, returns)
        assert beta == pytest.approx(1.0)

    def test_beta_greater_than_one(self):
        """Test beta > 1 for amplified moves."""
        bench_ret = pd.Series([0.01, -0.01, 0.02, -0.015, 0.01])
        port_ret = bench_ret * 1.5  # 1.5x leverage
        beta = portfolio_beta(port_ret, bench_ret)
        assert beta == pytest.approx(1.5, rel=0.01)

    def test_beta_less_than_one(self):
        """Test beta < 1 for dampened moves."""
        bench_ret = pd.Series([0.02, -0.02, 0.04, -0.03, 0.02])
        port_ret = bench_ret * 0.5
        beta = portfolio_beta(port_ret, bench_ret)
        assert beta == pytest.approx(0.5, rel=0.01)

    def test_negative_beta(self):
        """Test negative beta for inverse moves."""
        bench_ret = pd.Series([0.01, -0.01, 0.02, -0.015])
        port_ret = -bench_ret
        beta = portfolio_beta(port_ret, bench_ret)
        assert beta == pytest.approx(-1.0, rel=0.01)

    def test_empty_raises(self):
        """Test empty series raises error."""
        with pytest.raises(ValueError, match="at least 2"):
            portfolio_beta(pd.Series([]), pd.Series([]))

    def test_zero_variance_benchmark_raises(self):
        """Test zero variance benchmark raises error."""
        port_ret = pd.Series([0.01, 0.02, 0.03])
        bench_ret = pd.Series([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="variance is zero"):
            portfolio_beta(port_ret, bench_ret)

    def test_misaligned_series(self):
        """Test handles misaligned index gracefully."""
        port_ret = pd.Series([0.01, 0.02, 0.03], index=[0, 1, 2])
        bench_ret = pd.Series([0.005, 0.01, 0.015], index=[1, 2, 3])
        # Should align on common indices [1, 2]
        beta = portfolio_beta(port_ret, bench_ret)
        assert isinstance(beta, float)


class TestPortfolioAlpha:
    """Test portfolio alpha calculation."""

    def test_zero_alpha_for_benchmark(self):
        """Test alpha ≈ 0 when portfolio = benchmark."""
        returns = pd.Series([0.01, -0.01, 0.02, -0.015, 0.01])
        alpha = portfolio_alpha(returns, returns, risk_free_rate=0.0)
        assert alpha == pytest.approx(0.0, abs=1e-10)

    def test_positive_alpha_for_outperformance(self):
        """Test positive alpha when outperforming."""
        # Use varying returns to avoid zero variance issues
        np.random.seed(42)
        bench_ret = pd.Series(0.005 + np.random.randn(100) * 0.001)
        port_ret = pd.Series(0.01 + np.random.randn(100) * 0.001)  # Higher mean
        alpha = portfolio_alpha(port_ret, bench_ret)
        assert alpha > 0

    def test_negative_alpha_for_underperformance(self):
        """Test negative alpha when underperforming."""
        # Use deterministic data: portfolio underperforms with same beta
        bench_ret = pd.Series([0.01, -0.01, 0.02, -0.015, 0.01] * 20)
        port_ret = bench_ret * 1.0 - 0.001  # Same volatility but lower return
        alpha = portfolio_alpha(port_ret, bench_ret)
        assert alpha < 0

    def test_empty_raises(self):
        """Test empty series raises error."""
        with pytest.raises(ValueError, match="at least 2"):
            portfolio_alpha(pd.Series([]), pd.Series([]))


class TestConcentrationHerfindahl:
    """Test HHI concentration index."""

    def test_single_asset_concentration(self):
        """Test HHI = 1 for single asset."""
        weights = pd.Series([1.0])
        hhi = concentration_herfindahl(weights)
        assert hhi == pytest.approx(1.0)

    def test_equal_weights(self):
        """Test HHI = 1/N for equal weights."""
        weights = pd.Series([0.25, 0.25, 0.25, 0.25])
        hhi = concentration_herfindahl(weights)
        assert hhi == pytest.approx(0.25)  # 1/4

    def test_concentrated_portfolio(self):
        """Test high HHI for concentrated portfolio."""
        weights = pd.Series([0.8, 0.1, 0.1])
        hhi = concentration_herfindahl(weights)
        assert hhi == pytest.approx(0.66, abs=0.01)  # 0.8^2 + 0.1^2 + 0.1^2

    def test_ignores_zero_weights(self):
        """Test ignores zero weights."""
        weights = pd.Series([0.5, 0.5, 0.0, 0.0])
        hhi = concentration_herfindahl(weights)
        assert hhi == pytest.approx(0.5)  # Two equal weights

    def test_all_zero_raises(self):
        """Test all zero weights raises error."""
        weights = pd.Series([0.0, 0.0])
        with pytest.raises(ValueError, match="All weights are zero"):
            concentration_herfindahl(weights)

    def test_normalizes_weights(self):
        """Test normalizes weights to sum to 1."""
        weights = pd.Series([50, 50])  # Sum = 100, not 1
        hhi = concentration_herfindahl(weights)
        assert hhi == pytest.approx(0.5)  # Should normalize


class TestEffectiveNHoldings:
    """Test effective N holdings calculation."""

    def test_equal_weights(self):
        """Test effective N = N for equal weights."""
        weights = pd.Series([0.25, 0.25, 0.25, 0.25])
        eff_n = effective_n_holdings(weights)
        assert eff_n == pytest.approx(4.0)

    def test_concentrated_portfolio(self):
        """Test effective N < N for concentrated portfolio."""
        weights = pd.Series([0.7, 0.15, 0.15])
        eff_n = effective_n_holdings(weights)
        assert eff_n < 3  # Less diversified than 3 equal weights
        # HHI = 0.7^2 + 0.15^2 + 0.15^2 = 0.49 + 0.0225 + 0.0225 = 0.535
        # Effective N = 1/0.535 = 1.869
        assert eff_n == pytest.approx(1.869, abs=0.01)

    def test_single_asset(self):
        """Test effective N = 1 for single asset."""
        weights = pd.Series([1.0])
        eff_n = effective_n_holdings(weights)
        assert eff_n == pytest.approx(1.0)


class TestVolatilityClustering:
    """Test volatility clustering detection."""

    def test_basic_volatility(self):
        """Test basic rolling volatility calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.01)
        vol = volatility_clustering(returns, window=21)

        # Should return series of same length
        assert len(vol) == len(returns)
        # First window-1 values should be NaN
        assert vol.iloc[:20].isna().all()
        # Rest should be positive
        assert (vol.iloc[20:] > 0).all()

    def test_window_too_large_raises(self):
        """Test window larger than series raises error."""
        returns = pd.Series([0.01, 0.02, 0.03])
        with pytest.raises(ValueError, match="< window"):
            volatility_clustering(returns, window=10)

    def test_high_volatility_period(self):
        """Test detects high volatility periods."""
        # Low vol followed by high vol
        low_vol = np.random.randn(50) * 0.005
        high_vol = np.random.randn(50) * 0.03
        returns = pd.Series(np.concatenate([low_vol, high_vol]))

        vol = volatility_clustering(returns, window=21)

        # Average volatility in second half should be higher
        avg_vol_first = vol.iloc[30:50].mean()
        avg_vol_second = vol.iloc[70:].mean()
        assert avg_vol_second > avg_vol_first


class TestDrawdownSeries:
    """Test drawdown series calculation."""

    def test_no_drawdown(self):
        """Test no drawdown when equity always increases."""
        equity = pd.Series([100, 110, 120, 130])
        dd = drawdown_series(equity)
        assert (dd == 0).all()

    def test_simple_drawdown(self):
        """Test simple drawdown calculation."""
        equity = pd.Series([100, 110, 105, 120])
        dd = drawdown_series(equity)

        assert dd.iloc[0] == pytest.approx(0.0)  # At starting point
        assert dd.iloc[1] == pytest.approx(0.0)  # At new high
        assert dd.iloc[2] == pytest.approx(-0.0454, abs=0.001)  # (105/110 - 1)
        assert dd.iloc[3] == pytest.approx(0.0)  # New high

    def test_max_drawdown(self):
        """Test identifies max drawdown correctly."""
        equity = pd.Series([100, 120, 90, 100, 110])
        dd = drawdown_series(equity)

        max_dd = dd.min()
        assert max_dd == pytest.approx(-0.25)  # (90/120 - 1) = -25%


class TestExposureSummary:
    """Test exposure summary statistics."""

    def test_basic_summary(self):
        """Test basic exposure summary."""
        weights = pd.DataFrame({
            'A': [0.5, 0.4, 0.3, 0.0],
            'B': [0.5, 0.6, 0.7, 1.0]
        })
        summary = exposure_summary(weights)

        assert summary.loc['A', 'avg_weight'] == pytest.approx(0.3)
        assert summary.loc['B', 'avg_weight'] == pytest.approx(0.7)
        assert summary.loc['A', 'max_weight'] == pytest.approx(0.5)
        assert summary.loc['B', 'max_weight'] == pytest.approx(1.0)

    def test_hold_frequency(self):
        """Test hold frequency calculation."""
        weights = pd.DataFrame({
            'A': [0.5, 0.5, 0.0, 0.0],  # Held 50% of time
            'B': [0.5, 0.5, 1.0, 1.0]   # Held 100% of time
        })
        summary = exposure_summary(weights)

        assert summary.loc['A', 'hold_frequency'] == pytest.approx(0.5)
        assert summary.loc['B', 'hold_frequency'] == pytest.approx(1.0)

    def test_with_metadata(self):
        """Test adding metadata to summary."""
        weights = pd.DataFrame({
            'SPY': [0.5, 0.5],
            'TLT': [0.5, 0.5]
        })
        metadata = {
            'SPY': {'sector': 'Equity', 'country': 'US'},
            'TLT': {'sector': 'Fixed Income', 'country': 'US'}
        }
        summary = exposure_summary(weights, ticker_metadata=metadata)

        assert summary.loc['SPY', 'sector'] == 'Equity'
        assert summary.loc['TLT', 'sector'] == 'Fixed Income'
        assert summary.loc['SPY', 'country'] == 'US'


class TestRollingCorrelation:
    """Test rolling correlation calculation."""

    def test_basic_rolling_correlation(self):
        """Test basic rolling correlation."""
        np.random.seed(42)
        r1 = pd.Series(np.random.randn(200))
        r2 = pd.Series(np.random.randn(200))

        roll_corr = rolling_correlation(r1, r2, window=63)

        # Should return series of same length
        assert len(roll_corr) == len(r1)
        # First window-1 should be NaN
        assert roll_corr.iloc[:62].isna().all()
        # Correlations should be between -1 and 1
        assert (roll_corr.iloc[62:].abs() <= 1).all()

    def test_perfect_correlation(self):
        """Test rolling correlation with identical series."""
        returns = pd.Series(np.random.randn(100))
        roll_corr = rolling_correlation(returns, returns, window=21)

        # Should be 1.0 (or very close due to numerical precision)
        valid_corr = roll_corr.iloc[20:]
        assert (valid_corr.apply(lambda x: pytest.approx(x, abs=1e-10) == 1.0)).all()

    def test_window_too_large_raises(self):
        """Test window larger than series raises error."""
        r1 = pd.Series([0.01, 0.02])
        r2 = pd.Series([0.01, 0.02])
        with pytest.raises(ValueError, match="< window"):
            rolling_correlation(r1, r2, window=10)


class TestMaxDrawdownDuration:
    """Test maximum drawdown duration calculation."""

    def test_no_drawdown(self):
        """Test no drawdown when equity increases."""
        equity = pd.Series([100, 110, 120, 130])
        duration = max_drawdown_duration(equity)
        assert duration == 0

    def test_simple_drawdown_duration(self):
        """Test simple drawdown duration."""
        equity = pd.Series([100, 110, 105, 100, 95, 100, 105, 115])
        #                   0    1    2    3    4   5    6    7
        # Peak at 1, underwater from 2-6 (5 periods including recovery), new high at 7
        duration = max_drawdown_duration(equity)
        # The function counts consecutive periods below previous high
        assert duration >= 4  # Should be at least 4 periods

    def test_multiple_drawdowns(self):
        """Test identifies longest drawdown."""
        equity = pd.Series([100, 110, 105, 110, 100, 95, 90, 95, 100, 110])
        #                   0    1    2    3    4   5   6   7   8    9
        # First DD: 2 (1 period), Second DD: 4-8 (5 periods)
        duration = max_drawdown_duration(equity)
        assert duration >= 4  # Should capture the longer drawdown


class TestTailRatio:
    """Test tail ratio calculation."""

    def test_symmetric_tails(self):
        """Test tail ratio ≈ 1 for symmetric distribution."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.01)
        ratio = tail_ratio(returns, percentile=5.0)

        # Should be close to 1 for symmetric normal distribution
        assert ratio == pytest.approx(1.0, rel=0.3)  # Allow some variance

    def test_positive_skew(self):
        """Test tail ratio > 1 for positive skew."""
        # Add large positive outliers
        returns = pd.Series(np.concatenate([
            np.random.randn(95) * 0.01,
            [0.1, 0.12, 0.15, 0.2, 0.25]  # Large positive
        ]))
        ratio = tail_ratio(returns, percentile=5.0)
        assert ratio > 1.0

    def test_negative_skew(self):
        """Test tail ratio < 1 for negative skew."""
        # Add large negative outliers
        returns = pd.Series(np.concatenate([
            np.random.randn(95) * 0.01,
            [-0.1, -0.12, -0.15, -0.2, -0.25]
        ]))
        ratio = tail_ratio(returns, percentile=5.0)
        assert ratio < 1.0

    def test_empty_raises(self):
        """Test empty returns raises error."""
        with pytest.raises(ValueError, match="empty"):
            tail_ratio(pd.Series([]))

    def test_invalid_percentile_raises(self):
        """Test invalid percentile raises error."""
        returns = pd.Series([0.01, 0.02, 0.03])
        with pytest.raises(ValueError, match="between 0 and 50"):
            tail_ratio(returns, percentile=60)
