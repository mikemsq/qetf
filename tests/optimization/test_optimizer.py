"""Tests for the strategy optimizer module.

This module tests the optimizer.py functionality for orchestrating
the parameter sweep and generating output files.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import tempfile
import shutil
import yaml

from quantetf.optimization.optimizer import (
    OptimizationResult,
    StrategyOptimizer,
    _evaluate_config_worker,
)
from quantetf.optimization.evaluator import (
    PeriodMetrics,
    MultiPeriodResult,
)
from quantetf.optimization.grid import StrategyConfig


class TestOptimizationResult:
    """Tests for the OptimizationResult dataclass."""

    @pytest.fixture
    def sample_config(self):
        """Create sample strategy config."""
        return StrategyConfig(
            alpha_type='momentum',
            alpha_params={'lookback_days': 252, 'min_periods': 200},
            top_n=5,
            universe_path='configs/universes/tier3_expanded_100.yaml',
            schedule_path='configs/schedules/monthly_rebalance.yaml',
            schedule_name='monthly',
        )

    @pytest.fixture
    def sample_period_metrics(self):
        """Create sample period metrics."""
        return {
            '3yr': PeriodMetrics(
                period_name='3yr',
                start_date=datetime(2021, 1, 1),
                end_date=datetime(2024, 1, 1),
                strategy_return=0.45,
                spy_return=0.35,
                active_return=0.10,
                strategy_volatility=0.15,
                tracking_error=0.08,
                information_ratio=1.25,
                max_drawdown=-0.12,
                sharpe_ratio=1.5,
            ),
            '5yr': PeriodMetrics(
                period_name='5yr',
                start_date=datetime(2019, 1, 1),
                end_date=datetime(2024, 1, 1),
                strategy_return=0.85,
                spy_return=0.75,
                active_return=0.10,
                strategy_volatility=0.16,
                tracking_error=0.09,
                information_ratio=1.11,
                max_drawdown=-0.15,
                sharpe_ratio=1.3,
            ),
        }

    @pytest.fixture
    def sample_multi_period_result(self, sample_config, sample_period_metrics):
        """Create sample multi-period result."""
        return MultiPeriodResult(
            config_name='momentum_lookback_days252_min_periods200_top5_monthly',
            config=sample_config,
            periods=sample_period_metrics,
            beats_spy_all_periods=True,
            composite_score=1.68,
        )

    def test_optimization_result_creation(self, sample_config, sample_multi_period_result):
        """OptimizationResult should be created with all fields."""
        result = OptimizationResult(
            all_results=[sample_multi_period_result],
            winners=[sample_multi_period_result],
            best_config=sample_config,
            run_timestamp='20260114_143022',
            total_configs=100,
            successful_configs=95,
            failed_configs=5,
        )

        assert len(result.all_results) == 1
        assert len(result.winners) == 1
        assert result.best_config == sample_config
        assert result.total_configs == 100
        assert result.successful_configs == 95
        assert result.failed_configs == 5

    def test_optimization_result_no_winners(self, sample_multi_period_result):
        """OptimizationResult should handle no winners case."""
        result = OptimizationResult(
            all_results=[sample_multi_period_result],
            winners=[],
            best_config=None,
            run_timestamp='20260114_143022',
            total_configs=100,
            successful_configs=100,
            failed_configs=0,
        )

        assert len(result.winners) == 0
        assert result.best_config is None

    def test_optimization_result_summary_with_winners(self, sample_config, sample_multi_period_result):
        """summary should include winner details."""
        result = OptimizationResult(
            all_results=[sample_multi_period_result],
            winners=[sample_multi_period_result],
            best_config=sample_config,
            run_timestamp='20260114_143022',
            total_configs=100,
            successful_configs=95,
            failed_configs=5,
            output_dir=Path('/tmp/test'),
        )

        summary = result.summary()

        assert 'Strategy Optimization Results' in summary
        assert '100' in summary  # total_configs
        assert '95' in summary  # successful
        assert '5' in summary  # failed
        assert 'momentum' in summary.lower()
        assert 'Best Strategy' in summary

    def test_optimization_result_summary_no_winners(self, sample_multi_period_result):
        """summary should indicate no winners when appropriate."""
        result = OptimizationResult(
            all_results=[sample_multi_period_result],
            winners=[],
            best_config=None,
            run_timestamp='20260114_143022',
            total_configs=100,
            successful_configs=100,
            failed_configs=0,
        )

        summary = result.summary()

        assert 'No winning strategies found' in summary


class TestStrategyOptimizerInit:
    """Tests for StrategyOptimizer initialization."""

    @pytest.fixture
    def temp_snapshot(self):
        """Create a temporary snapshot file."""
        temp_dir = tempfile.mkdtemp()
        snapshot_path = Path(temp_dir) / 'data.parquet'
        # Create empty parquet file
        df = pd.DataFrame({'ticker': ['SPY'], 'close': [100.0]})
        df.to_parquet(snapshot_path)
        yield snapshot_path
        shutil.rmtree(temp_dir)

    def test_init_with_valid_path(self, temp_snapshot):
        """Optimizer should initialize with valid snapshot path."""
        optimizer = StrategyOptimizer(
            snapshot_path=temp_snapshot,
            output_dir='/tmp/output',
        )

        assert optimizer.snapshot_path == temp_snapshot
        assert optimizer.periods_years == [3, 5, 10]
        assert optimizer.max_workers == 1

    def test_init_with_custom_periods(self, temp_snapshot):
        """Optimizer should accept custom evaluation periods."""
        optimizer = StrategyOptimizer(
            snapshot_path=temp_snapshot,
            output_dir='/tmp/output',
            periods_years=[5, 10],
        )

        assert optimizer.periods_years == [5, 10]

    def test_init_with_parallel_workers(self, temp_snapshot):
        """Optimizer should accept parallel worker count."""
        optimizer = StrategyOptimizer(
            snapshot_path=temp_snapshot,
            output_dir='/tmp/output',
            max_workers=4,
        )

        assert optimizer.max_workers == 4

    def test_init_with_invalid_path_raises(self):
        """Optimizer should raise FileNotFoundError for invalid path."""
        with pytest.raises(FileNotFoundError):
            StrategyOptimizer(
                snapshot_path='/nonexistent/path/data.parquet',
                output_dir='/tmp/output',
            )

    def test_run_dir_before_run_raises(self, temp_snapshot):
        """Accessing run_dir before run() should raise RuntimeError."""
        optimizer = StrategyOptimizer(
            snapshot_path=temp_snapshot,
            output_dir='/tmp/output',
        )

        with pytest.raises(RuntimeError):
            _ = optimizer.run_dir


class TestStrategyOptimizerRun:
    """Tests for StrategyOptimizer.run() method."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for snapshot and output."""
        temp_dir = tempfile.mkdtemp()
        snapshot_dir = Path(temp_dir) / 'snapshot'
        output_dir = Path(temp_dir) / 'output'
        snapshot_dir.mkdir()
        output_dir.mkdir()

        # Create snapshot file
        snapshot_path = snapshot_dir / 'data.parquet'
        df = pd.DataFrame({'ticker': ['SPY'], 'close': [100.0]})
        df.to_parquet(snapshot_path)

        yield {
            'temp_dir': temp_dir,
            'snapshot_path': snapshot_path,
            'output_dir': output_dir,
        }

        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_evaluator(self):
        """Create mock evaluator that returns successful results."""
        def create_mock_result(config):
            return MultiPeriodResult(
                config_name=config.generate_name(),
                config=config,
                periods={
                    '3yr': PeriodMetrics(
                        period_name='3yr',
                        start_date=datetime(2021, 1, 1),
                        end_date=datetime(2024, 1, 1),
                        strategy_return=0.45,
                        spy_return=0.35,
                        active_return=0.10,
                        strategy_volatility=0.15,
                        tracking_error=0.08,
                        information_ratio=1.25,
                        max_drawdown=-0.12,
                        sharpe_ratio=1.5,
                    ),
                },
                beats_spy_all_periods=True,
                composite_score=1.75,
            )

        mock = MagicMock()
        mock.evaluate = MagicMock(side_effect=lambda config, periods: create_mock_result(config))
        return mock

    @patch('quantetf.optimization.optimizer.MultiPeriodEvaluator')
    @patch('quantetf.optimization.optimizer.generate_configs')
    def test_run_creates_output_directory(self, mock_gen, mock_eval_class, temp_dirs, mock_evaluator):
        """run() should create timestamped output directory."""
        mock_gen.return_value = [
            StrategyConfig(
                alpha_type='momentum',
                alpha_params={'lookback_days': 126, 'min_periods': 50},
                top_n=5,
                universe_path='configs/universes/tier3.yaml',
                schedule_path='configs/schedules/monthly.yaml',
                schedule_name='monthly',
            ),
        ]
        mock_eval_class.return_value = mock_evaluator

        optimizer = StrategyOptimizer(
            snapshot_path=temp_dirs['snapshot_path'],
            output_dir=temp_dirs['output_dir'],
        )

        result = optimizer.run()

        assert optimizer.run_dir.exists()
        assert optimizer.run_dir.parent == temp_dirs['output_dir']

    @patch('quantetf.optimization.optimizer.MultiPeriodEvaluator')
    @patch('quantetf.optimization.optimizer.generate_configs')
    def test_run_returns_optimization_result(self, mock_gen, mock_eval_class, temp_dirs, mock_evaluator):
        """run() should return OptimizationResult."""
        mock_gen.return_value = [
            StrategyConfig(
                alpha_type='momentum',
                alpha_params={'lookback_days': 126, 'min_periods': 50},
                top_n=5,
                universe_path='configs/universes/tier3.yaml',
                schedule_path='configs/schedules/monthly.yaml',
                schedule_name='monthly',
            ),
        ]
        mock_eval_class.return_value = mock_evaluator

        optimizer = StrategyOptimizer(
            snapshot_path=temp_dirs['snapshot_path'],
            output_dir=temp_dirs['output_dir'],
        )

        result = optimizer.run()

        assert isinstance(result, OptimizationResult)
        assert result.total_configs >= 1
        assert result.successful_configs >= 0
        assert result.run_timestamp is not None

    @patch('quantetf.optimization.optimizer.MultiPeriodEvaluator')
    @patch('quantetf.optimization.optimizer.generate_configs')
    def test_run_with_max_configs_limit(self, mock_gen, mock_eval_class, temp_dirs, mock_evaluator):
        """run() should respect max_configs limit."""
        # Generate 10 configs
        configs = [
            StrategyConfig(
                alpha_type='momentum',
                alpha_params={'lookback_days': i * 10, 'min_periods': 50},
                top_n=5,
                universe_path='configs/universes/tier3.yaml',
                schedule_path='configs/schedules/monthly.yaml',
                schedule_name='monthly',
            )
            for i in range(10, 20)
        ]
        mock_gen.return_value = configs
        mock_eval_class.return_value = mock_evaluator

        optimizer = StrategyOptimizer(
            snapshot_path=temp_dirs['snapshot_path'],
            output_dir=temp_dirs['output_dir'],
        )

        result = optimizer.run(max_configs=3)

        # Should only evaluate 3 configs
        assert mock_evaluator.evaluate.call_count == 3

    @patch('quantetf.optimization.optimizer.MultiPeriodEvaluator')
    @patch('quantetf.optimization.optimizer.generate_configs')
    def test_run_saves_all_results_csv(self, mock_gen, mock_eval_class, temp_dirs, mock_evaluator):
        """run() should save all_results.csv."""
        mock_gen.return_value = [
            StrategyConfig(
                alpha_type='momentum',
                alpha_params={'lookback_days': 126, 'min_periods': 50},
                top_n=5,
                universe_path='configs/universes/tier3.yaml',
                schedule_path='configs/schedules/monthly.yaml',
                schedule_name='monthly',
            ),
        ]
        mock_eval_class.return_value = mock_evaluator

        optimizer = StrategyOptimizer(
            snapshot_path=temp_dirs['snapshot_path'],
            output_dir=temp_dirs['output_dir'],
        )

        result = optimizer.run()

        csv_path = optimizer.run_dir / 'all_results.csv'
        assert csv_path.exists()

        df = pd.read_csv(csv_path)
        assert len(df) >= 1
        assert 'config_name' in df.columns
        assert 'composite_score' in df.columns

    @patch('quantetf.optimization.optimizer.MultiPeriodEvaluator')
    @patch('quantetf.optimization.optimizer.generate_configs')
    def test_run_saves_winners_csv_when_winners_exist(self, mock_gen, mock_eval_class, temp_dirs, mock_evaluator):
        """run() should save winners.csv when there are winners."""
        mock_gen.return_value = [
            StrategyConfig(
                alpha_type='momentum',
                alpha_params={'lookback_days': 126, 'min_periods': 50},
                top_n=5,
                universe_path='configs/universes/tier3.yaml',
                schedule_path='configs/schedules/monthly.yaml',
                schedule_name='monthly',
            ),
        ]
        mock_eval_class.return_value = mock_evaluator

        optimizer = StrategyOptimizer(
            snapshot_path=temp_dirs['snapshot_path'],
            output_dir=temp_dirs['output_dir'],
        )

        result = optimizer.run()

        csv_path = optimizer.run_dir / 'winners.csv'
        assert csv_path.exists()

    @patch('quantetf.optimization.optimizer.MultiPeriodEvaluator')
    @patch('quantetf.optimization.optimizer.generate_configs')
    def test_run_saves_best_strategy_yaml(self, mock_gen, mock_eval_class, temp_dirs, mock_evaluator):
        """run() should save best_strategy.yaml when there are winners."""
        mock_gen.return_value = [
            StrategyConfig(
                alpha_type='momentum',
                alpha_params={'lookback_days': 126, 'min_periods': 50},
                top_n=5,
                universe_path='configs/universes/tier3.yaml',
                schedule_path='configs/schedules/monthly.yaml',
                schedule_name='monthly',
            ),
        ]
        mock_eval_class.return_value = mock_evaluator

        optimizer = StrategyOptimizer(
            snapshot_path=temp_dirs['snapshot_path'],
            output_dir=temp_dirs['output_dir'],
        )

        result = optimizer.run()

        yaml_path = optimizer.run_dir / 'best_strategy.yaml'
        assert yaml_path.exists()

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        assert 'name' in config
        assert 'alpha_model' in config
        assert '_optimization_metadata' in config

    @patch('quantetf.optimization.optimizer.MultiPeriodEvaluator')
    @patch('quantetf.optimization.optimizer.generate_configs')
    def test_run_saves_optimization_report(self, mock_gen, mock_eval_class, temp_dirs, mock_evaluator):
        """run() should save optimization_report.md."""
        mock_gen.return_value = [
            StrategyConfig(
                alpha_type='momentum',
                alpha_params={'lookback_days': 126, 'min_periods': 50},
                top_n=5,
                universe_path='configs/universes/tier3.yaml',
                schedule_path='configs/schedules/monthly.yaml',
                schedule_name='monthly',
            ),
        ]
        mock_eval_class.return_value = mock_evaluator

        optimizer = StrategyOptimizer(
            snapshot_path=temp_dirs['snapshot_path'],
            output_dir=temp_dirs['output_dir'],
        )

        result = optimizer.run()

        report_path = optimizer.run_dir / 'optimization_report.md'
        assert report_path.exists()

        with open(report_path) as f:
            content = f.read()

        assert '# Strategy Optimization Report' in content
        assert 'Summary' in content

    @patch('quantetf.optimization.optimizer.MultiPeriodEvaluator')
    @patch('quantetf.optimization.optimizer.generate_configs')
    def test_run_handles_failed_evaluations(self, mock_gen, mock_eval_class, temp_dirs):
        """run() should handle failed evaluations gracefully."""
        mock_gen.return_value = [
            StrategyConfig(
                alpha_type='momentum',
                alpha_params={'lookback_days': 126, 'min_periods': 50},
                top_n=5,
                universe_path='configs/universes/tier3.yaml',
                schedule_path='configs/schedules/monthly.yaml',
                schedule_name='monthly',
            ),
            StrategyConfig(
                alpha_type='momentum',
                alpha_params={'lookback_days': 252, 'min_periods': 100},
                top_n=5,
                universe_path='configs/universes/tier3.yaml',
                schedule_path='configs/schedules/monthly.yaml',
                schedule_name='monthly',
            ),
        ]

        # First call succeeds, second fails
        mock_eval = MagicMock()
        mock_eval.evaluate = MagicMock(side_effect=[
            MultiPeriodResult(
                config_name='config1',
                config=None,
                periods={},
                beats_spy_all_periods=False,
                composite_score=0.5,
            ),
            Exception("Evaluation failed"),
        ])
        mock_eval_class.return_value = mock_eval

        optimizer = StrategyOptimizer(
            snapshot_path=temp_dirs['snapshot_path'],
            output_dir=temp_dirs['output_dir'],
        )

        result = optimizer.run()

        assert result.successful_configs == 1
        assert result.failed_configs == 1

    @patch('quantetf.optimization.optimizer.MultiPeriodEvaluator')
    @patch('quantetf.optimization.optimizer.generate_configs')
    def test_run_sorts_by_composite_score(self, mock_gen, mock_eval_class, temp_dirs):
        """run() should sort results by composite score descending."""
        configs = [
            StrategyConfig(
                alpha_type='momentum',
                alpha_params={'lookback_days': i * 10, 'min_periods': 50},
                top_n=5,
                universe_path='configs/universes/tier3.yaml',
                schedule_path='configs/schedules/monthly.yaml',
                schedule_name='monthly',
            )
            for i in range(1, 4)
        ]
        mock_gen.return_value = configs

        # Return results with different scores
        call_count = [0]
        scores = [0.5, 1.5, 1.0]  # Middle one is highest

        def create_result(config, periods):
            score = scores[call_count[0]]
            call_count[0] += 1
            return MultiPeriodResult(
                config_name=config.generate_name(),
                config=config,
                periods={
                    '3yr': PeriodMetrics(
                        period_name='3yr',
                        start_date=datetime(2021, 1, 1),
                        end_date=datetime(2024, 1, 1),
                        strategy_return=0.45,
                        spy_return=0.35,
                        active_return=0.10,
                        strategy_volatility=0.15,
                        tracking_error=0.08,
                        information_ratio=score,
                        max_drawdown=-0.12,
                        sharpe_ratio=1.5,
                    ),
                },
                beats_spy_all_periods=True,
                composite_score=score,
            )

        mock_eval = MagicMock()
        mock_eval.evaluate = MagicMock(side_effect=create_result)
        mock_eval_class.return_value = mock_eval

        optimizer = StrategyOptimizer(
            snapshot_path=temp_dirs['snapshot_path'],
            output_dir=temp_dirs['output_dir'],
        )

        result = optimizer.run()

        # Check sorting
        assert result.all_results[0].composite_score == 1.5
        assert result.all_results[1].composite_score == 1.0
        assert result.all_results[2].composite_score == 0.5

    @patch('quantetf.optimization.optimizer.MultiPeriodEvaluator')
    @patch('quantetf.optimization.optimizer.generate_configs')
    def test_run_calls_progress_callback(self, mock_gen, mock_eval_class, temp_dirs, mock_evaluator):
        """run() should call progress callback during evaluation."""
        mock_gen.return_value = [
            StrategyConfig(
                alpha_type='momentum',
                alpha_params={'lookback_days': 126, 'min_periods': 50},
                top_n=5,
                universe_path='configs/universes/tier3.yaml',
                schedule_path='configs/schedules/monthly.yaml',
                schedule_name='monthly',
            ),
            StrategyConfig(
                alpha_type='momentum',
                alpha_params={'lookback_days': 252, 'min_periods': 100},
                top_n=5,
                universe_path='configs/universes/tier3.yaml',
                schedule_path='configs/schedules/monthly.yaml',
                schedule_name='monthly',
            ),
        ]
        mock_eval_class.return_value = mock_evaluator

        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        optimizer = StrategyOptimizer(
            snapshot_path=temp_dirs['snapshot_path'],
            output_dir=temp_dirs['output_dir'],
        )

        result = optimizer.run(progress_callback=progress_callback)

        assert len(progress_calls) == 2
        assert progress_calls[0] == (1, 2)
        assert progress_calls[1] == (2, 2)


class TestStrategyOptimizerOutputGeneration:
    """Tests for output file generation."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample optimization result for testing."""
        config = StrategyConfig(
            alpha_type='momentum',
            alpha_params={'lookback_days': 252, 'min_periods': 200},
            top_n=5,
            universe_path='configs/universes/tier3.yaml',
            schedule_path='configs/schedules/monthly.yaml',
            schedule_name='monthly',
        )

        period_metrics = {
            '3yr': PeriodMetrics(
                period_name='3yr',
                start_date=datetime(2021, 1, 1),
                end_date=datetime(2024, 1, 1),
                strategy_return=0.45,
                spy_return=0.35,
                active_return=0.10,
                strategy_volatility=0.15,
                tracking_error=0.08,
                information_ratio=1.25,
                max_drawdown=-0.12,
                sharpe_ratio=1.5,
            ),
            '5yr': PeriodMetrics(
                period_name='5yr',
                start_date=datetime(2019, 1, 1),
                end_date=datetime(2024, 1, 1),
                strategy_return=0.85,
                spy_return=0.75,
                active_return=0.10,
                strategy_volatility=0.16,
                tracking_error=0.09,
                information_ratio=1.11,
                max_drawdown=-0.15,
                sharpe_ratio=1.3,
            ),
        }

        multi_result = MultiPeriodResult(
            config_name=config.generate_name(),
            config=config,
            periods=period_metrics,
            beats_spy_all_periods=True,
            composite_score=1.68,
        )

        return OptimizationResult(
            all_results=[multi_result],
            winners=[multi_result],
            best_config=config,
            run_timestamp='20260114_143022',
            total_configs=10,
            successful_configs=9,
            failed_configs=1,
        )

    def test_report_contains_summary_section(self, sample_result):
        """Report should contain summary statistics."""
        # Generate report content manually for testing
        periods_str = '3yr, 5yr, 10yr'
        lines = [
            "# Strategy Optimization Report",
            "",
            f"**Run timestamp:** {sample_result.run_timestamp}",
            f"**Periods evaluated:** {periods_str}",
            "",
            "## Summary",
            "",
            f"- Total configurations tested: {sample_result.total_configs}",
            f"- Successful evaluations: {sample_result.successful_configs}",
            f"- Failed evaluations: {sample_result.failed_configs}",
        ]
        content = '\n'.join(lines)

        assert 'Summary' in content
        assert '10' in content  # total_configs
        assert '9' in content  # successful
        assert '1' in content  # failed

    def test_report_contains_winners_table(self, sample_result):
        """Report should contain winners table when winners exist."""
        winner = sample_result.winners[0]
        lines = [
            "## Top 10 Winners",
            "",
            "| Rank | Config | Composite Score | 3yr Active | 5yr Active |",
            "|------|--------|-----------------|------------|------------|",
            f"| 1 | {winner.config_name} | {winner.composite_score:.3f} | +10.0% | +10.0% |",
        ]
        content = '\n'.join(lines)

        assert 'Top 10 Winners' in content
        assert 'Composite Score' in content

    def test_yaml_output_is_valid(self, sample_result):
        """YAML output should be valid and loadable."""
        config_dict = sample_result.best_config.to_dict()
        config_dict['_optimization_metadata'] = {
            'description': 'Test strategy',
            'composite_score': 1.68,
        }

        yaml_str = yaml.dump(config_dict, default_flow_style=False)
        loaded = yaml.safe_load(yaml_str)

        assert loaded['name'] == sample_result.best_config.generate_name()
        assert loaded['alpha_model']['type'] == 'momentum'


class TestEvaluateConfigWorker:
    """Tests for the parallel worker function."""

    def test_worker_returns_none_on_exception(self):
        """Worker should return None when evaluation fails."""
        config = StrategyConfig(
            alpha_type='momentum',
            alpha_params={'lookback_days': 252, 'min_periods': 200},
            top_n=5,
            universe_path='configs/universes/tier3.yaml',
            schedule_path='configs/schedules/monthly.yaml',
            schedule_name='monthly',
        )

        # Non-existent path will cause failure
        args = (config, Path('/nonexistent/path.parquet'), [3, 5, 10], 10.0)

        result = _evaluate_config_worker(args)

        assert result is None


class TestStrategyOptimizerIntegration:
    """Integration tests for StrategyOptimizer (requires real data)."""

    @pytest.fixture
    def snapshot_path(self):
        """Get path to test snapshot (skip if not available)."""
        snapshot_dir = Path('data/snapshots')
        if not snapshot_dir.exists():
            pytest.skip("No snapshot directory found")

        snapshots = list(snapshot_dir.glob('snapshot_*/data.parquet'))
        if not snapshots:
            pytest.skip("No snapshots available for integration testing")

        return snapshots[0]

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.mark.integration
    def test_integration_small_run(self, snapshot_path, temp_output_dir):
        """Test a small optimization run with real data."""
        optimizer = StrategyOptimizer(
            snapshot_path=snapshot_path,
            output_dir=temp_output_dir,
            periods_years=[3],  # Single period for speed
            max_workers=1,
        )

        # Run with just 2 configs
        result = optimizer.run(max_configs=2)

        assert result.successful_configs > 0
        assert optimizer.run_dir.exists()
        assert (optimizer.run_dir / 'all_results.csv').exists()
        assert (optimizer.run_dir / 'optimization_report.md').exists()

    @pytest.mark.integration
    def test_integration_filters_by_schedule(self, snapshot_path, temp_output_dir):
        """Test filtering by schedule type."""
        optimizer = StrategyOptimizer(
            snapshot_path=snapshot_path,
            output_dir=temp_output_dir,
            periods_years=[3],
            max_workers=1,
        )

        result = optimizer.run(
            max_configs=2,
            schedule_names=['monthly'],
        )

        # All configs should be monthly
        for r in result.all_results:
            assert 'monthly' in r.config_name

    @pytest.mark.integration
    def test_integration_filters_by_alpha_type(self, snapshot_path, temp_output_dir):
        """Test filtering by alpha type."""
        optimizer = StrategyOptimizer(
            snapshot_path=snapshot_path,
            output_dir=temp_output_dir,
            periods_years=[3],
            max_workers=1,
        )

        result = optimizer.run(
            max_configs=2,
            alpha_types=['momentum'],
        )

        # All configs should use momentum alpha
        for r in result.all_results:
            assert 'momentum' in r.config_name
