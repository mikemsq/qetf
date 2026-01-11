"""Tests for strategy comparison module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json
from datetime import datetime, timedelta

from quantetf.evaluation.comparison import (
    load_backtest_result,
    compute_comparison_metrics,
    compute_returns_correlation,
    sharpe_ratio_ttest,
    create_equity_overlay_chart,
    create_risk_return_scatter,
    create_comparison_table_html,
    generate_comparison_report,
    StrategyResult
)


@pytest.fixture
def mock_backtest_dir():
    """Create a temporary backtest directory with mock data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dates
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # Create config
        config = {
            'strategy': 'test_strategy',
            'start_date': '2023-01-01',
            'end_date': '2023-04-10',
            'initial_capital': 100000.0,
            'top_n': 5,
            'lookback_days': 252,
            'cost_bps': 10.0
        }
        with open(tmpdir / 'config.json', 'w') as f:
            json.dump(config, f)

        # Create metrics
        metrics = {
            'total_return': 0.15,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.10,
            'total_costs': 50.0,
            'num_rebalances': 10,
            'final_nav': 115000.0,
            'initial_nav': 100000.0
        }
        with open(tmpdir / 'metrics.json', 'w') as f:
            json.dump(metrics, f)

        # Create equity curve
        equity = pd.Series(
            np.linspace(100000, 115000, 100),
            index=dates,
            name='nav'
        )
        equity_df = pd.DataFrame({'nav': equity})
        equity_df.to_csv(tmpdir / 'equity_curve.csv')

        # Create weights history
        weights = pd.DataFrame(
            np.random.dirichlet([1]*5, 100),
            index=dates,
            columns=['SPY', 'QQQ', 'DIA', 'IWM', 'VTI']
        )
        weights.to_csv(tmpdir / 'weights_history.csv')

        # Create holdings history
        holdings = pd.DataFrame(
            np.random.randint(10, 100, (100, 5)),
            index=dates,
            columns=['SPY', 'QQQ', 'DIA', 'IWM', 'VTI']
        )
        holdings.to_csv(tmpdir / 'holdings_history.csv')

        yield tmpdir


@pytest.fixture
def mock_backtest_dir_2():
    """Create a second temporary backtest directory with different performance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dates
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # Create config
        config = {
            'strategy': 'test_strategy_2',
            'start_date': '2023-01-01',
            'end_date': '2023-04-10',
            'initial_capital': 100000.0,
            'top_n': 3,
            'lookback_days': 126,
            'cost_bps': 5.0
        }
        with open(tmpdir / 'config.json', 'w') as f:
            json.dump(config, f)

        # Create metrics (worse performance)
        metrics = {
            'total_return': 0.08,
            'sharpe_ratio': 0.9,
            'max_drawdown': -0.15,
            'total_costs': 30.0,
            'num_rebalances': 8,
            'final_nav': 108000.0,
            'initial_nav': 100000.0
        }
        with open(tmpdir / 'metrics.json', 'w') as f:
            json.dump(metrics, f)

        # Create equity curve (more volatile)
        equity = pd.Series(
            100000 + np.cumsum(np.random.randn(100) * 500),
            index=dates,
            name='nav'
        )
        equity.iloc[-1] = 108000  # Ensure final value matches metrics
        equity_df = pd.DataFrame({'nav': equity})
        equity_df.to_csv(tmpdir / 'equity_curve.csv')

        # Create weights history
        weights = pd.DataFrame(
            np.random.dirichlet([1]*5, 100),
            index=dates,
            columns=['SPY', 'QQQ', 'DIA', 'IWM', 'VTI']
        )
        weights.to_csv(tmpdir / 'weights_history.csv')

        # Create holdings history
        holdings = pd.DataFrame(
            np.random.randint(10, 100, (100, 5)),
            index=dates,
            columns=['SPY', 'QQQ', 'DIA', 'IWM', 'VTI']
        )
        holdings.to_csv(tmpdir / 'holdings_history.csv')

        yield tmpdir


class TestLoadBacktestResult:
    """Tests for load_backtest_result function."""

    def test_load_valid_backtest(self, mock_backtest_dir):
        """Test loading a valid backtest directory."""
        result = load_backtest_result(mock_backtest_dir)

        assert isinstance(result, StrategyResult)
        assert result.name == 'test_strategy'
        assert result.config['initial_capital'] == 100000.0
        assert result.metrics['sharpe_ratio'] == 1.5
        assert len(result.equity_curve) == 100
        assert result.equity_curve.iloc[0] == 100000.0
        assert len(result.weights_history) == 100
        assert len(result.holdings_history) == 100

    def test_load_nonexistent_directory(self):
        """Test loading from non-existent directory."""
        with pytest.raises(FileNotFoundError, match="Backtest directory not found"):
            load_backtest_result('/nonexistent/path')

    def test_load_missing_config(self, mock_backtest_dir):
        """Test loading with missing config file."""
        (mock_backtest_dir / 'config.json').unlink()

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_backtest_result(mock_backtest_dir)

    def test_load_missing_metrics(self, mock_backtest_dir):
        """Test loading with missing metrics file."""
        (mock_backtest_dir / 'metrics.json').unlink()

        with pytest.raises(FileNotFoundError, match="Metrics file not found"):
            load_backtest_result(mock_backtest_dir)

    def test_load_missing_equity_curve(self, mock_backtest_dir):
        """Test loading with missing equity curve file."""
        (mock_backtest_dir / 'equity_curve.csv').unlink()

        with pytest.raises(FileNotFoundError, match="Equity curve not found"):
            load_backtest_result(mock_backtest_dir)

    def test_load_path_as_string(self, mock_backtest_dir):
        """Test loading with path as string."""
        result = load_backtest_result(str(mock_backtest_dir))
        assert result.name == 'test_strategy'


class TestComputeComparisonMetrics:
    """Tests for compute_comparison_metrics function."""

    def test_single_strategy(self, mock_backtest_dir):
        """Test computing metrics for a single strategy."""
        result = load_backtest_result(mock_backtest_dir)
        df = compute_comparison_metrics([result])

        assert len(df) == 1
        assert 'total_return' in df.columns
        assert 'sharpe_ratio' in df.columns
        assert 'max_drawdown' in df.columns
        assert df.loc['test_strategy', 'total_return'] == 0.15
        assert df.loc['test_strategy', 'sharpe_ratio'] == 1.5

    def test_multiple_strategies(self, mock_backtest_dir, mock_backtest_dir_2):
        """Test computing metrics for multiple strategies."""
        result1 = load_backtest_result(mock_backtest_dir)
        result2 = load_backtest_result(mock_backtest_dir_2)
        df = compute_comparison_metrics([result1, result2])

        assert len(df) == 2
        assert 'test_strategy' in df.index
        assert 'test_strategy_2' in df.index
        assert df.loc['test_strategy', 'total_return'] > df.loc['test_strategy_2', 'total_return']

    def test_computed_metrics_present(self, mock_backtest_dir):
        """Test that all expected metrics are computed."""
        result = load_backtest_result(mock_backtest_dir)
        df = compute_comparison_metrics([result])

        expected_metrics = [
            'total_return', 'cagr', 'sharpe_ratio', 'sortino_ratio',
            'calmar_ratio', 'max_drawdown', 'win_rate', 'var_95',
            'cvar_95', 'volatility', 'total_costs', 'num_rebalances',
            'final_nav', 'initial_nav'
        ]

        for metric in expected_metrics:
            assert metric in df.columns, f"Missing metric: {metric}"

    def test_empty_results_list(self):
        """Test with empty results list."""
        df = compute_comparison_metrics([])
        assert len(df) == 0


class TestComputeReturnsCorrelation:
    """Tests for compute_returns_correlation function."""

    def test_correlation_single_strategy(self, mock_backtest_dir):
        """Test correlation with single strategy."""
        result = load_backtest_result(mock_backtest_dir)
        corr = compute_returns_correlation([result])

        assert corr.shape == (1, 1)
        assert corr.loc['test_strategy', 'test_strategy'] == 1.0

    def test_correlation_multiple_strategies(self, mock_backtest_dir, mock_backtest_dir_2):
        """Test correlation with multiple strategies."""
        result1 = load_backtest_result(mock_backtest_dir)
        result2 = load_backtest_result(mock_backtest_dir_2)
        corr = compute_returns_correlation([result1, result2])

        assert corr.shape == (2, 2)
        assert corr.loc['test_strategy', 'test_strategy'] == 1.0
        assert corr.loc['test_strategy_2', 'test_strategy_2'] == 1.0
        # Correlation should be between -1 and 1
        assert -1 <= corr.loc['test_strategy', 'test_strategy_2'] <= 1

    def test_correlation_symmetric(self, mock_backtest_dir, mock_backtest_dir_2):
        """Test that correlation matrix is symmetric."""
        result1 = load_backtest_result(mock_backtest_dir)
        result2 = load_backtest_result(mock_backtest_dir_2)
        corr = compute_returns_correlation([result1, result2])

        assert np.allclose(corr.values, corr.values.T)


class TestSharpeRatioTTest:
    """Tests for sharpe_ratio_ttest function."""

    def test_ttest_same_strategy(self, mock_backtest_dir):
        """Test t-test comparing strategy to itself."""
        result = load_backtest_result(mock_backtest_dir)
        test = sharpe_ratio_ttest(result, result)

        # When comparing same strategy, t-stat should be 0 or NaN
        assert test['t_statistic'] == 0.0 or np.isnan(test['t_statistic'])
        # p-value should be high (not significant) or NaN
        assert test['p_value'] >= 0.05 or np.isnan(test['p_value'])
        assert not test['is_significant']
        assert test['sharpe_1'] == test['sharpe_2']

    def test_ttest_different_strategies(self, mock_backtest_dir, mock_backtest_dir_2):
        """Test t-test comparing two different strategies."""
        result1 = load_backtest_result(mock_backtest_dir)
        result2 = load_backtest_result(mock_backtest_dir_2)
        test = sharpe_ratio_ttest(result1, result2)

        assert 't_statistic' in test
        assert 'p_value' in test
        assert 'is_significant' in test
        assert 'sharpe_1' in test
        assert 'sharpe_2' in test
        assert isinstance(test['is_significant'], bool)

    def test_ttest_output_structure(self, mock_backtest_dir, mock_backtest_dir_2):
        """Test that t-test returns all required fields."""
        result1 = load_backtest_result(mock_backtest_dir)
        result2 = load_backtest_result(mock_backtest_dir_2)
        test = sharpe_ratio_ttest(result1, result2)

        required_fields = ['t_statistic', 'p_value', 'is_significant', 'sharpe_1', 'sharpe_2', 'message']
        for field in required_fields:
            assert field in test


class TestCreateEquityOverlayChart:
    """Tests for create_equity_overlay_chart function."""

    def test_chart_creation(self, mock_backtest_dir, mock_backtest_dir_2):
        """Test creating equity overlay chart."""
        result1 = load_backtest_result(mock_backtest_dir)
        result2 = load_backtest_result(mock_backtest_dir_2)

        fig = create_equity_overlay_chart([result1, result2])

        assert fig is not None
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert len(ax.lines) == 2  # Two equity curves

    def test_chart_saved_to_file(self, mock_backtest_dir, tmp_path):
        """Test saving chart to file."""
        result = load_backtest_result(mock_backtest_dir)
        output_path = tmp_path / 'chart.png'

        fig = create_equity_overlay_chart([result], output_path=output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_single_strategy_chart(self, mock_backtest_dir):
        """Test chart with single strategy."""
        result = load_backtest_result(mock_backtest_dir)
        fig = create_equity_overlay_chart([result])

        assert fig is not None
        ax = fig.axes[0]
        assert len(ax.lines) == 1


class TestCreateRiskReturnScatter:
    """Tests for create_risk_return_scatter function."""

    def test_scatter_creation(self, mock_backtest_dir, mock_backtest_dir_2):
        """Test creating risk-return scatter plot."""
        result1 = load_backtest_result(mock_backtest_dir)
        result2 = load_backtest_result(mock_backtest_dir_2)

        fig = create_risk_return_scatter([result1, result2])

        assert fig is not None
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        # Should have scatter points
        assert len(ax.collections) > 0

    def test_scatter_saved_to_file(self, mock_backtest_dir, tmp_path):
        """Test saving scatter plot to file."""
        result = load_backtest_result(mock_backtest_dir)
        output_path = tmp_path / 'scatter.png'

        fig = create_risk_return_scatter([result], output_path=output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestCreateComparisonTableHTML:
    """Tests for create_comparison_table_html function."""

    def test_html_generation(self, mock_backtest_dir):
        """Test HTML table generation."""
        result = load_backtest_result(mock_backtest_dir)
        df = compute_comparison_metrics([result])

        html = create_comparison_table_html(df)

        assert isinstance(html, str)
        assert '<table' in html
        assert '<th>' in html
        assert 'test_strategy' in html

    def test_html_saved_to_file(self, mock_backtest_dir, tmp_path):
        """Test saving HTML to file."""
        result = load_backtest_result(mock_backtest_dir)
        df = compute_comparison_metrics([result])
        output_path = tmp_path / 'table.html'

        html = create_comparison_table_html(df, output_path=output_path)

        assert output_path.exists()
        with open(output_path) as f:
            content = f.read()
        assert content == html

    def test_html_formatting(self, mock_backtest_dir, mock_backtest_dir_2):
        """Test HTML includes proper formatting."""
        result1 = load_backtest_result(mock_backtest_dir)
        result2 = load_backtest_result(mock_backtest_dir_2)
        df = compute_comparison_metrics([result1, result2])

        html = create_comparison_table_html(df)

        # Check for CSS styling
        assert '<style>' in html
        assert 'border-collapse' in html
        # Check for both strategies
        assert 'test_strategy' in html
        assert 'test_strategy_2' in html


class TestGenerateComparisonReport:
    """Tests for generate_comparison_report function."""

    def test_report_generation(self, mock_backtest_dir, mock_backtest_dir_2, tmp_path):
        """Test generating complete comparison report."""
        result1 = load_backtest_result(mock_backtest_dir)
        result2 = load_backtest_result(mock_backtest_dir_2)

        output_dir = tmp_path / 'comparison'
        paths = generate_comparison_report([result1, result2], output_dir, 'test_report')

        # Check that output directory was created
        assert output_dir.exists()

        # Check that all expected files are present
        assert 'metrics_csv' in paths
        assert 'table_html' in paths
        assert 'equity_chart' in paths
        assert 'risk_return_chart' in paths
        assert 'correlation_csv' in paths
        assert 'significance_tests' in paths

        # Verify files exist
        for path in paths.values():
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_single_strategy(self, mock_backtest_dir, tmp_path):
        """Test report generation with single strategy."""
        result = load_backtest_result(mock_backtest_dir)

        output_dir = tmp_path / 'comparison'
        paths = generate_comparison_report([result], output_dir, 'single_report')

        # Should still generate metrics and charts
        assert 'metrics_csv' in paths
        assert 'table_html' in paths
        assert 'equity_chart' in paths
        assert 'risk_return_chart' in paths

        # Should not generate correlation or significance tests
        assert 'correlation_csv' not in paths
        assert 'significance_tests' not in paths

    def test_report_creates_directory(self, mock_backtest_dir, tmp_path):
        """Test that report creates output directory if it doesn't exist."""
        result = load_backtest_result(mock_backtest_dir)

        output_dir = tmp_path / 'new_dir' / 'comparison'
        assert not output_dir.exists()

        paths = generate_comparison_report([result], output_dir)

        assert output_dir.exists()
        assert output_dir.is_dir()


class TestStrategyResultDataclass:
    """Tests for StrategyResult dataclass."""

    def test_strategy_result_creation(self):
        """Test creating StrategyResult object."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        equity = pd.Series(range(100000, 110000, 1000), index=dates)
        weights = pd.DataFrame(np.random.rand(10, 3), index=dates, columns=['A', 'B', 'C'])
        holdings = pd.DataFrame(np.random.randint(1, 100, (10, 3)), index=dates, columns=['A', 'B', 'C'])

        result = StrategyResult(
            name='test',
            config={'key': 'value'},
            metrics={'sharpe': 1.5},
            equity_curve=equity,
            weights_history=weights,
            holdings_history=holdings,
            backtest_dir=Path('/tmp')
        )

        assert result.name == 'test'
        assert result.config == {'key': 'value'}
        assert result.metrics == {'sharpe': 1.5}
        assert len(result.equity_curve) == 10
        assert result.backtest_dir == Path('/tmp')
