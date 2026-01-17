"""Tests for run_backtest.py script.

This module tests the end-to-end backtest script including:
- Argument parsing
- Backtest execution
- Output file generation
- Error handling
"""

import pytest
import pandas as pd
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
import argparse

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
import run_backtest

from quantetf.backtest.simple_engine import BacktestResult, BacktestConfig
from quantetf.types import Universe


@pytest.fixture
def mock_args():
    """Create mock command-line arguments."""
    args = argparse.Namespace(
        snapshot='data/snapshots/test_snapshot',
        start='2021-01-01',
        end='2023-12-31',
        strategy='test-strategy',
        capital=100000.0,
        top_n=5,
        lookback=252,
        cost_bps=10.0,
        rebalance='monthly',
        output_dir='artifacts/backtests'
    )
    return args


@pytest.fixture
def mock_backtest_result():
    """Create mock BacktestResult for testing."""
    # Create sample equity curve
    dates = pd.date_range('2021-01-01', '2021-04-01', freq='BME')
    equity_curve = pd.DataFrame({
        'nav': [100000.0, 105000.0, 110000.0],
        'cost': [50.0, 60.0, 55.0],
        'returns': [0.0, 0.05, 0.047619]
    }, index=dates)

    # Create sample holdings
    holdings = pd.DataFrame({
        'SPY': [100.0, 105.0, 110.0],
        'QQQ': [50.0, 52.0, 55.0],
        'AGG': [200.0, 195.0, 190.0]
    }, index=dates)

    # Create sample weights
    weights = pd.DataFrame({
        'SPY': [0.4, 0.42, 0.44],
        'QQQ': [0.3, 0.31, 0.32],
        'AGG': [0.3, 0.27, 0.24]
    }, index=dates)

    # Create metrics
    metrics = {
        'total_return': 0.10,
        'sharpe_ratio': 1.5,
        'max_drawdown': -0.05,
        'total_costs': 165.0,
        'num_rebalances': 3,
        'final_nav': 110000.0,
        'initial_nav': 100000.0
    }

    # Create config
    config = BacktestConfig(
        start_date=pd.Timestamp('2021-01-01'),
        end_date=pd.Timestamp('2023-12-31'),
        universe=Universe(
            as_of=pd.Timestamp('2023-12-31'),
            tickers=('SPY', 'QQQ', 'AGG')
        ),
        initial_capital=100000.0
    )

    rebalance_dates = pd.date_range("2023-01-01", "2023-12-31", freq="MS").tolist()

    return BacktestResult(
        equity_curve=equity_curve,
        holdings_history=holdings,
        weights_history=weights,
        metrics=metrics,
        config=config,
        rebalance_dates=rebalance_dates
    )


class TestArgumentParsing:
    """Test command-line argument parsing."""

    def test_parse_args_defaults(self):
        """Test that default arguments are set correctly."""
        with patch('sys.argv', ['run_backtest.py']):
            args = run_backtest.parse_args()

            assert args.snapshot == 'data/snapshots/snapshot_5yr_20etfs'
            assert args.start == '2021-01-01'
            assert args.end == '2025-12-31'
            assert args.strategy == 'momentum-ew-top5'
            assert args.capital == 100000.0
            assert args.top_n == 5
            assert args.lookback == 252
            assert args.cost_bps == 10.0
            assert args.rebalance == 'monthly'
            assert args.output_dir == 'artifacts/backtests'

    def test_parse_args_custom(self):
        """Test parsing custom arguments."""
        with patch('sys.argv', [
            'run_backtest.py',
            '--snapshot', 'custom/path',
            '--start', '2022-01-01',
            '--end', '2024-12-31',
            '--strategy', 'custom-strategy',
            '--capital', '50000',
            '--top-n', '3',
            '--lookback', '126',
            '--cost-bps', '5.0',
            '--rebalance', 'weekly',
            '--output-dir', 'custom/output'
        ]):
            args = run_backtest.parse_args()

            assert args.snapshot == 'custom/path'
            assert args.start == '2022-01-01'
            assert args.end == '2024-12-31'
            assert args.strategy == 'custom-strategy'
            assert args.capital == 50000.0
            assert args.top_n == 3
            assert args.lookback == 126
            assert args.cost_bps == 5.0
            assert args.rebalance == 'weekly'
            assert args.output_dir == 'custom/output'


class TestPrintMetrics:
    """Test metrics printing function."""

    def test_print_metrics(self, mock_backtest_result, caplog):
        """Test that metrics are printed correctly."""
        import logging
        
        # Mock the store to avoid needing SPY data
        mock_store = MagicMock()
        mock_store.get_prices.return_value = None  # SPY prices unavailable
        
        # Ensure logger is set to INFO level
        with caplog.at_level(logging.INFO):
            run_backtest.print_metrics(mock_backtest_result, mock_store)

        # Check logged output (logger writes to caplog)
        output = caplog.text

        # Check that key metrics are in output
        assert 'Total Return:' in output
        assert '10.00%' in output
        assert 'Sharpe Ratio:' in output
        assert '1.50' in output
        assert 'Max Drawdown:' in output
        assert '-5.00%' in output
        assert 'Total Costs:' in output
        assert '165.00' in output  # Check without $ to avoid spacing issues
        assert 'Num Rebalances:' in output
        assert '3' in output
        assert 'Initial NAV:' in output
        assert '100,000.00' in output
        assert 'Final NAV:' in output
        assert '110,000.00' in output
        assert 'Profit/Loss:' in output
        assert '10,000.00' in output


class TestSaveResults:
    """Test results saving functionality."""

    def test_save_results_creates_directory(self, mock_backtest_result, mock_args, tmp_path):
        """Test that save_results creates output directory."""
        mock_args.output_dir = str(tmp_path / 'backtests')
        
        output_dir_path = tmp_path / 'test_output'
        output_dir_path.mkdir()

        output_dir = run_backtest.save_results(mock_backtest_result, mock_args, output_dir_path)

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_save_results_creates_all_files(self, mock_backtest_result, mock_args, tmp_path):
        """Test that all expected output files are created."""
        output_dir_path = tmp_path / 'test_output'
        output_dir_path.mkdir()

        output_dir = run_backtest.save_results(mock_backtest_result, mock_args, output_dir_path)

        # Check that all files exist
        assert (output_dir / 'equity_curve.csv').exists()
        assert (output_dir / 'holdings_history.csv').exists()
        assert (output_dir / 'weights_history.csv').exists()
        assert (output_dir / 'metrics.json').exists()
        assert (output_dir / 'config.json').exists()

    def test_save_results_equity_curve_content(self, mock_backtest_result, mock_args, tmp_path):
        """Test equity curve CSV content."""
        output_dir_path = tmp_path / 'test_output'
        output_dir_path.mkdir()

        output_dir = run_backtest.save_results(mock_backtest_result, mock_args, output_dir_path)

        # Read and verify equity curve
        equity_df = pd.read_csv(output_dir / 'equity_curve.csv', index_col=0, parse_dates=True)
        assert 'nav' in equity_df.columns
        assert 'cost' in equity_df.columns
        assert len(equity_df) == 3
        assert equity_df['nav'].iloc[0] == 100000.0

    def test_save_results_metrics_content(self, mock_backtest_result, mock_args, tmp_path):
        """Test metrics JSON content."""
        output_dir_path = tmp_path / 'test_output'
        output_dir_path.mkdir()

        output_dir = run_backtest.save_results(mock_backtest_result, mock_args, output_dir_path)

        # Read and verify metrics
        with open(output_dir / 'metrics.json') as f:
            metrics = json.load(f)

        assert metrics['total_return'] == 0.10
        assert metrics['sharpe_ratio'] == 1.5
        assert metrics['max_drawdown'] == -0.05
        assert metrics['total_costs'] == 165.0
        assert metrics['num_rebalances'] == 3

    def test_save_results_config_content(self, mock_backtest_result, mock_args, tmp_path):
        """Test config JSON content."""
        output_dir_path = tmp_path / 'test_output'
        output_dir_path.mkdir()

        output_dir = run_backtest.save_results(mock_backtest_result, mock_args, output_dir_path)

        # Read and verify config
        with open(output_dir / 'config.json') as f:
            config = json.load(f)

        assert config['strategy'] == 'test-strategy'
        assert config['top_n'] == 5
        assert config['lookback_days'] == 252
        assert config['cost_bps'] == 10.0
        assert config['initial_capital'] == 100000.0
        assert isinstance(config['universe'], list)


class TestRunBacktest:
    """Test main backtest execution function."""

    @patch('run_backtest.SnapshotDataStore')
    @patch('run_backtest.SimpleBacktestEngine')
    @patch('run_backtest.save_results')
    def test_run_backtest_success(
        self,
        mock_save,
        mock_engine_class,
        mock_store_class,
        mock_args,
        mock_backtest_result,
        tmp_path
    ):
        """Test successful backtest execution."""
        # Setup mocks
        mock_store = Mock()
        mock_store.tickers = ['SPY', 'QQQ', 'AGG']
        mock_store_class.return_value = mock_store

        mock_engine = Mock()
        mock_engine.run.return_value = mock_backtest_result
        mock_engine_class.return_value = mock_engine

        mock_save.return_value = tmp_path / 'output'

        # Create temporary snapshot directory structure
        snapshot_dir = tmp_path / 'snapshot'
        snapshot_dir.mkdir()
        (snapshot_dir / 'data.parquet').touch()

        # Create manifest file
        manifest = {
            'data_summary': {
                'tickers': ['SPY', 'QQQ', 'AGG']
            }
        }
        with open(snapshot_dir / 'manifest.yaml', 'w') as f:
            yaml.dump(manifest, f)

        mock_args.snapshot = str(snapshot_dir)

        # Run backtest
        result = run_backtest.run_backtest(mock_args, tmp_path / 'output')

        # Verify calls
        assert mock_store_class.called
        assert mock_engine.run.called
        assert mock_save.called
        assert result == mock_backtest_result

    def test_run_backtest_snapshot_not_found(self, mock_args, tmp_path):
        """Test error handling when snapshot not found."""
        mock_args.snapshot = 'nonexistent/path'

        with pytest.raises(FileNotFoundError, match="Snapshot not found"):
            run_backtest.run_backtest(mock_args, tmp_path / 'output')

    @patch('run_backtest.SnapshotDataStore')
    def test_run_backtest_loads_metadata(self, mock_store_class, mock_args, tmp_path):
        """Test that backtest correctly loads metadata."""
        # Create temporary snapshot directory
        snapshot_dir = tmp_path / 'snapshot'
        snapshot_dir.mkdir()
        (snapshot_dir / 'data.parquet').touch()

        # Create manifest with tickers
        manifest = {
            'data_summary': {
                'tickers': ['SPY', 'QQQ', 'IWM', 'AGG', 'TLT']
            }
        }
        with open(snapshot_dir / 'manifest.yaml', 'w') as f:
            yaml.dump(manifest, f)

        mock_args.snapshot = str(snapshot_dir)

        # Setup mock store
        mock_store = Mock()
        mock_store_class.return_value = mock_store

        # Setup mock engine
        with patch('run_backtest.SimpleBacktestEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine

            # Create a minimal result to avoid print_metrics issues
            minimal_result = Mock()
            minimal_result.metrics = {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_costs': 0.0,
                'num_rebalances': 0
            }
            minimal_result.equity_curve = pd.DataFrame({'nav': [100000.0]})
            minimal_result.config = Mock(initial_capital=100000.0)
            mock_engine.run.return_value = minimal_result

            with patch('run_backtest.save_results'):
                run_backtest.run_backtest(mock_args, tmp_path / 'output')

            # Verify engine.run was called with correct universe
            call_args = mock_engine.run.call_args
            universe = call_args.kwargs['config'].universe
            assert len(universe.tickers) == 5
            assert 'SPY' in universe.tickers
            assert 'QQQ' in universe.tickers


class TestMain:
    """Test main entry point."""

    @patch('run_backtest.run_backtest')
    def test_main_success(self, mock_run):
        """Test successful main execution."""
        mock_run.return_value = Mock()

        with patch('sys.argv', ['run_backtest.py']):
            exit_code = run_backtest.main()

        assert exit_code == 0
        assert mock_run.called

    @patch('run_backtest.run_backtest')
    def test_main_failure(self, mock_run):
        """Test main handles exceptions."""
        mock_run.side_effect = Exception("Test error")

        with patch('sys.argv', ['run_backtest.py']):
            exit_code = run_backtest.main()

        assert exit_code == 1


class TestIntegration:
    """Integration tests (require real data)."""

    @pytest.mark.skipif(
        not (Path('data/snapshots/snapshot_5yr_20etfs/data.parquet').exists()),
        reason="Real snapshot data not available"
    )
    def test_full_backtest_with_real_data(self):
        """Test full backtest execution with real snapshot data."""
        with patch('sys.argv', [
            'run_backtest.py',
            '--start', '2021-01-01',
            '--end', '2021-12-31',
            '--top-n', '3',
            '--lookback', '126'
        ]):
            args = run_backtest.parse_args()
            # Create a temporary output directory for integration test
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                result = run_backtest.run_backtest(args, Path(tmpdir))

                # Verify result structure
                assert result is not None
                assert hasattr(result, 'equity_curve')
                assert hasattr(result, 'holdings_history')
                assert hasattr(result, 'weights_history')
                assert hasattr(result, 'metrics')

            # Verify metrics
            assert 'total_return' in result.metrics
            assert 'sharpe_ratio' in result.metrics
            assert 'max_drawdown' in result.metrics


class TestErrorHandling:
    """Test error handling scenarios."""

    @patch('run_backtest.SnapshotDataStore')
    def test_handles_store_initialization_error(self, mock_store_class, mock_args, tmp_path):
        """Test handling of SnapshotDataStore initialization errors."""
        snapshot_dir = tmp_path / 'snapshot'
        snapshot_dir.mkdir()
        (snapshot_dir / 'data.parquet').touch()

        mock_args.snapshot = str(snapshot_dir)
        mock_store_class.side_effect = ValueError("Invalid data format")

        with pytest.raises(ValueError, match="Invalid data format"):
            run_backtest.run_backtest(mock_args, tmp_path / 'output')

    @patch('run_backtest.SnapshotDataStore')
    @patch('run_backtest.SimpleBacktestEngine')
    def test_handles_backtest_engine_error(
        self,
        mock_engine_class,
        mock_store_class,
        mock_args,
        tmp_path
    ):
        """Test handling of backtest engine errors."""
        # Setup snapshot
        snapshot_dir = tmp_path / 'snapshot'
        snapshot_dir.mkdir()
        (snapshot_dir / 'data.parquet').touch()
        (snapshot_dir / 'manifest.yaml').write_text('data_summary:\n  tickers: [SPY]')

        mock_args.snapshot = str(snapshot_dir)

        # Setup mocks
        mock_store_class.return_value = Mock()

        mock_engine = Mock()
        mock_engine.run.side_effect = ValueError("Insufficient data")
        mock_engine_class.return_value = mock_engine

        with pytest.raises(ValueError, match="Insufficient data"):
            run_backtest.run_backtest(mock_args, tmp_path / 'output')
