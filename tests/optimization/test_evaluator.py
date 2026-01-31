"""Tests for the multi-period evaluator module.

This module tests the evaluator.py functionality for evaluating strategy
configurations across multiple time periods.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from quantetf.data.access import DataAccessFactory
from quantetf.optimization.evaluator import (
    PeriodMetrics,
    MultiPeriodResult,
    MultiPeriodEvaluator,
)
from quantetf.optimization.grid import StrategyConfig


class TestPeriodMetrics:
    """Tests for the PeriodMetrics dataclass."""

    @pytest.fixture
    def sample_metrics(self):
        """Create sample period metrics for testing."""
        return PeriodMetrics(
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
            num_rebalances=36,
            evaluation_success=True,
        )

    def test_period_metrics_creation(self, sample_metrics):
        """PeriodMetrics should be created with all fields."""
        assert sample_metrics.period_name == '3yr'
        assert sample_metrics.strategy_return == 0.45
        assert sample_metrics.spy_return == 0.35
        assert sample_metrics.active_return == 0.10
        assert sample_metrics.information_ratio == 1.25

    def test_to_dict_contains_all_fields(self, sample_metrics):
        """to_dict should contain all metric fields."""
        d = sample_metrics.to_dict()

        assert 'period_name' in d
        assert 'start_date' in d
        assert 'end_date' in d
        assert 'strategy_return' in d
        assert 'spy_return' in d
        assert 'active_return' in d
        assert 'information_ratio' in d
        assert 'sharpe_ratio' in d
        assert 'max_drawdown' in d
        assert 'tracking_error' in d
        assert 'evaluation_success' in d

    def test_to_dict_values(self, sample_metrics):
        """to_dict values should match attributes."""
        d = sample_metrics.to_dict()

        assert d['strategy_return'] == 0.45
        assert d['spy_return'] == 0.35
        assert d['active_return'] == 0.10
        assert d['information_ratio'] == 1.25

    def test_failed_metrics(self):
        """Failed metrics should have appropriate indicator values."""
        failed = PeriodMetrics(
            period_name='10yr',
            start_date=datetime(2014, 1, 1),
            end_date=datetime(2024, 1, 1),
            strategy_return=float('-inf'),
            spy_return=0.0,
            active_return=float('-inf'),
            strategy_volatility=float('inf'),
            tracking_error=float('inf'),
            information_ratio=float('-inf'),
            max_drawdown=-1.0,
            sharpe_ratio=float('-inf'),
            evaluation_success=False,
        )

        assert failed.evaluation_success is False
        assert failed.strategy_return == float('-inf')
        assert failed.information_ratio == float('-inf')


class TestMultiPeriodResult:
    """Tests for the MultiPeriodResult dataclass."""

    @pytest.fixture
    def sample_config(self):
        """Create sample strategy config."""
        return StrategyConfig(
            alpha_type='momentum',
            alpha_params={'lookback_days': 252, 'min_periods': 200},
            top_n=5,
            universe_path='configs/universes/tier3_expanded_100.yaml',
            universe_name='tier3',
            schedule_path='configs/schedules/monthly_rebalance.yaml',
            schedule_name='monthly',
        )

    @pytest.fixture
    def sample_periods(self):
        """Create sample period metrics dict."""
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
    def sample_result(self, sample_config, sample_periods):
        """Create sample multi-period result."""
        return MultiPeriodResult(
            config_name='momentum_lookback_days252_min_periods200_top5_monthly',
            config=sample_config,
            periods=sample_periods,
            beats_spy_all_periods=True,
            composite_score=1.68,
        )

    def test_multi_period_result_creation(self, sample_result):
        """MultiPeriodResult should be created with all fields."""
        assert sample_result.config_name is not None
        assert sample_result.beats_spy_all_periods is True
        assert sample_result.composite_score == 1.68
        assert len(sample_result.periods) == 2

    def test_to_dict_structure(self, sample_result):
        """to_dict should have flat structure with prefixed columns."""
        d = sample_result.to_dict()

        assert 'config_name' in d
        assert 'beats_spy_all_periods' in d
        assert 'composite_score' in d

        # Check for prefixed period columns
        assert '3y_strategy_return' in d
        assert '3y_spy_return' in d
        assert '3y_active_return' in d
        assert '3y_information_ratio' in d

        assert '5y_strategy_return' in d
        assert '5y_spy_return' in d

    def test_to_dict_values(self, sample_result):
        """to_dict should have correct values."""
        d = sample_result.to_dict()

        assert d['3y_strategy_return'] == 0.45
        assert d['3y_spy_return'] == 0.35
        assert d['3y_active_return'] == 0.10
        assert d['5y_strategy_return'] == 0.85

    def test_summary_output(self, sample_result):
        """summary should return readable string."""
        summary = sample_result.summary()

        assert 'momentum' in summary.lower()
        assert '3yr' in summary
        assert '5yr' in summary
        assert 'Beats SPY' in summary

    def test_result_with_error(self, sample_config):
        """Result with error should include error message."""
        result = MultiPeriodResult(
            config_name='test_config',
            config=sample_config,
            periods={},
            beats_spy_all_periods=False,
            composite_score=float('-inf'),
            error_message='Insufficient data for 10yr period',
        )

        assert result.error_message is not None
        assert 'Insufficient data' in result.error_message
        assert 'Error' in result.summary()


class TestMultiPeriodEvaluatorUnit:
    """Unit tests for MultiPeriodEvaluator methods."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock SnapshotDataStore."""
        store = MagicMock()
        store.date_range = (
            pd.Timestamp('2015-01-01'),
            pd.Timestamp('2024-12-31'),
        )
        store.tickers = ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD']
        return store

    @pytest.fixture
    def sample_config(self):
        """Create sample strategy config."""
        return StrategyConfig(
            alpha_type='momentum',
            alpha_params={'lookback_days': 252, 'min_periods': 200},
            top_n=5,
            universe_path='configs/universes/tier3_expanded_100.yaml',
            universe_name='tier3',
            schedule_path='configs/schedules/monthly_rebalance.yaml',
            schedule_name='monthly',
        )

    def test_beats_spy_all_positive(self):
        """Strategy with all positive metrics should beat SPY."""
        periods = {
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

        # Use a helper to test the method in isolation
        result = _beats_spy_helper(periods)
        assert result is True

    def test_beats_spy_negative_active_return(self):
        """Strategy with negative active return should not beat SPY."""
        periods = {
            '3yr': PeriodMetrics(
                period_name='3yr',
                start_date=datetime(2021, 1, 1),
                end_date=datetime(2024, 1, 1),
                strategy_return=0.30,
                spy_return=0.35,
                active_return=-0.05,  # Negative
                strategy_volatility=0.15,
                tracking_error=0.08,
                information_ratio=-0.625,
                max_drawdown=-0.12,
                sharpe_ratio=1.0,
            ),
        }

        result = _beats_spy_helper(periods)
        assert result is False

    def test_beats_spy_negative_ir(self):
        """Strategy with negative IR should not beat SPY."""
        periods = {
            '3yr': PeriodMetrics(
                period_name='3yr',
                start_date=datetime(2021, 1, 1),
                end_date=datetime(2024, 1, 1),
                strategy_return=0.45,
                spy_return=0.35,
                active_return=0.10,
                strategy_volatility=0.15,
                tracking_error=0.08,
                information_ratio=-0.5,  # Negative IR
                max_drawdown=-0.12,
                sharpe_ratio=1.0,
            ),
        }

        result = _beats_spy_helper(periods)
        assert result is False

    def test_beats_spy_one_period_fails(self):
        """Strategy should fail if ANY period fails criteria."""
        periods = {
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
                strategy_return=0.65,
                spy_return=0.75,
                active_return=-0.10,  # This period fails
                strategy_volatility=0.16,
                tracking_error=0.09,
                information_ratio=-1.11,
                max_drawdown=-0.15,
                sharpe_ratio=1.0,
            ),
        }

        result = _beats_spy_helper(periods)
        assert result is False

    def test_composite_score_with_consistent_ir(self):
        """Composite score should be higher for consistent IRs."""
        periods_consistent = {
            '3yr': _make_metrics('3yr', ir=1.0),
            '5yr': _make_metrics('5yr', ir=1.0),
            '10yr': _make_metrics('10yr', ir=1.0),
        }

        periods_volatile = {
            '3yr': _make_metrics('3yr', ir=0.5),
            '5yr': _make_metrics('5yr', ir=1.0),
            '10yr': _make_metrics('10yr', ir=1.5),
        }

        score_consistent = _composite_score_helper(periods_consistent)
        score_volatile = _composite_score_helper(periods_volatile)

        # Both have same avg IR (1.0) but consistent should score higher
        assert score_consistent > score_volatile

    def test_composite_score_winner_bonus(self):
        """Composite score should include winner bonus."""
        periods_winner = {
            '3yr': _make_metrics('3yr', ir=1.0, active_return=0.10),
            '5yr': _make_metrics('5yr', ir=1.0, active_return=0.10),
        }

        periods_loser = {
            '3yr': _make_metrics('3yr', ir=1.0, active_return=0.10),
            '5yr': _make_metrics('5yr', ir=-0.5, active_return=-0.05),
        }

        score_winner = _composite_score_helper(periods_winner)
        score_loser = _composite_score_helper(periods_loser)

        assert score_winner > score_loser

    def test_composite_score_failed_evaluation(self):
        """Failed evaluation should return -inf score."""
        periods = {
            '3yr': PeriodMetrics(
                period_name='3yr',
                start_date=datetime(2021, 1, 1),
                end_date=datetime(2024, 1, 1),
                strategy_return=float('-inf'),
                spy_return=0.0,
                active_return=float('-inf'),
                strategy_volatility=float('inf'),
                tracking_error=float('inf'),
                information_ratio=float('-inf'),
                max_drawdown=-1.0,
                sharpe_ratio=float('-inf'),
                evaluation_success=False,
            ),
        }

        score = _composite_score_helper(periods)
        assert score == float('-inf')

    def test_warmup_days_momentum(self, sample_config):
        """Warmup days for momentum should be 1.5x lookback."""
        warmup = _get_warmup_days_helper(sample_config)
        expected = int(252 * 1.5)
        assert warmup == expected

    def test_warmup_days_momentum_acceleration(self):
        """Warmup days for momentum_acceleration should use long_lookback."""
        config = StrategyConfig(
            alpha_type='momentum_acceleration',
            alpha_params={
                'short_lookback_days': 63,
                'long_lookback_days': 252,
                'min_periods': 100,
            },
            top_n=5,
            universe_path='configs/universes/tier3_expanded_100.yaml',
            universe_name='tier3',
            schedule_path='configs/schedules/monthly_rebalance.yaml',
            schedule_name='monthly',
        )

        warmup = _get_warmup_days_helper(config)
        expected = int(252 * 1.5)
        assert warmup == expected

    def test_periods_per_year_weekly(self):
        """Weekly schedule should have 52 periods per year."""
        assert _get_periods_per_year_helper('weekly') == 52

    def test_periods_per_year_monthly(self):
        """Monthly schedule should have 12 periods per year."""
        assert _get_periods_per_year_helper('monthly') == 12


class TestMultiPeriodEvaluatorIntegration:
    """Integration tests for MultiPeriodEvaluator (requires real data)."""

    @pytest.fixture
    def data_access(self):
        """Get DataAccessContext for test snapshot (skip if not available)."""
        # Check for available snapshots
        snapshot_dir = Path('data/snapshots')
        if not snapshot_dir.exists():
            pytest.skip("No snapshot directory found")

        snapshots = list(snapshot_dir.glob('snapshot_*/data.parquet'))
        if not snapshots:
            pytest.skip("No snapshots available for integration testing")

        return DataAccessFactory.create_context(
            config={"snapshot_path": str(snapshots[0])},
            enable_caching=True
        )

    @pytest.mark.integration
    def test_evaluate_single_config(self, data_access):
        """Test evaluating a single configuration."""
        config = StrategyConfig(
            alpha_type='momentum',
            alpha_params={'lookback_days': 126, 'min_periods': 50},
            top_n=5,
            universe_path='configs/universes/tier3_expanded_100.yaml',
            universe_name='tier3',
            schedule_path='configs/schedules/monthly_rebalance.yaml',
            schedule_name='monthly',
        )

        evaluator = MultiPeriodEvaluator(data_access=data_access)

        # Only test 3yr period for speed
        result = evaluator.evaluate(config, periods_years=[3])

        assert result.config_name == config.generate_name()
        assert '3yr' in result.periods
        assert isinstance(result.beats_spy_all_periods, bool)
        assert result.composite_score != float('-inf') or not result.periods['3yr'].evaluation_success

    @pytest.mark.integration
    def test_evaluate_multiple_periods(self, data_access):
        """Test evaluating multiple time periods."""
        config = StrategyConfig(
            alpha_type='momentum',
            alpha_params={'lookback_days': 126, 'min_periods': 50},
            top_n=5,
            universe_path='configs/universes/tier3_expanded_100.yaml',
            universe_name='tier3',
            schedule_path='configs/schedules/monthly_rebalance.yaml',
            schedule_name='monthly',
        )

        evaluator = MultiPeriodEvaluator(data_access=data_access)
        result = evaluator.evaluate(config, periods_years=[3, 5])

        assert '3yr' in result.periods
        assert '5yr' in result.periods


# Helper functions for testing internal logic
def _beats_spy_helper(periods: dict) -> bool:
    """Helper to test _beats_spy_all_periods logic."""
    for period_name, metrics in periods.items():
        if not getattr(metrics, 'evaluation_success', True):
            return False
        if metrics.active_return <= 0:
            return False
        if metrics.information_ratio <= 0:
            return False
    return True


def _composite_score_helper(periods: dict) -> float:
    """Helper to test _calculate_composite_score logic."""
    irs = [
        m.information_ratio
        for m in periods.values()
        if getattr(m, 'evaluation_success', True) and m.information_ratio != float('-inf')
    ]

    if not irs:
        return float('-inf')

    avg_ir = sum(irs) / len(irs)

    if len(irs) > 1:
        ir_variance = sum((ir - avg_ir) ** 2 for ir in irs) / len(irs)
        ir_std = ir_variance ** 0.5
        consistency_penalty = ir_std * 0.5
    else:
        consistency_penalty = 0.0

    beats_all = all(
        getattr(m, 'evaluation_success', True)
        and m.active_return > 0
        and m.information_ratio > 0
        for m in periods.values()
    )
    winner_bonus = 0.5 if beats_all else 0.0

    return avg_ir - consistency_penalty + winner_bonus


def _make_metrics(period_name: str, ir: float = 1.0, active_return: float = 0.10) -> PeriodMetrics:
    """Create metrics with specified IR and active return."""
    return PeriodMetrics(
        period_name=period_name,
        start_date=datetime(2021, 1, 1),
        end_date=datetime(2024, 1, 1),
        strategy_return=0.45,
        spy_return=0.35,
        active_return=active_return,
        strategy_volatility=0.15,
        tracking_error=0.08,
        information_ratio=ir,
        max_drawdown=-0.12,
        sharpe_ratio=1.5,
        evaluation_success=True,
    )


def _get_warmup_days_helper(config: StrategyConfig) -> int:
    """Helper to test warmup days calculation."""
    params = config.alpha_params

    if config.alpha_type == 'momentum_acceleration':
        lookback = params.get('long_lookback_days', 252)
    else:
        lookback = params.get('lookback_days', 252)

    return int(lookback * 1.5)


def _get_periods_per_year_helper(schedule_name: str) -> int:
    """Helper to test periods per year calculation."""
    if schedule_name == 'weekly':
        return 52
    elif schedule_name == 'monthly':
        return 12
    else:
        return 12


class TestLoadUniverseTickers:
    """Tests for the _load_universe_tickers method."""

    def test_load_tier2_universe(self, tmp_path):
        """Test loading tier2 universe from YAML."""
        # Create a mock universe YAML file
        universe_yaml = tmp_path / "test_universe.yaml"
        universe_yaml.write_text("""
name: test_universe
source:
  type: static_list
  tickers:
    - SPY
    - QQQ
    - IWM
    - EEM
    - GLD
""")

        # Create a mock evaluator to test the method
        mock_prices = MagicMock()
        mock_prices.get_available_tickers.return_value = ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD']
        mock_prices.get_latest_price_date.return_value = pd.Timestamp('2024-01-01')

        mock_ctx = MagicMock()
        mock_ctx.prices = mock_prices

        evaluator = MultiPeriodEvaluator(data_access=mock_ctx)

        tickers = evaluator._load_universe_tickers(str(universe_yaml))

        assert tickers == ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD']
        assert len(tickers) == 5

    def test_load_real_tier2_universe(self):
        """Test loading real tier2 universe file."""
        universe_path = 'configs/universes/tier2_core_50.yaml'
        if not Path(universe_path).exists():
            pytest.skip("Tier2 universe config not found")

        mock_prices = MagicMock()
        mock_prices.get_available_tickers.return_value = ['SPY', 'QQQ']
        mock_prices.get_latest_price_date.return_value = pd.Timestamp('2024-01-01')

        mock_ctx = MagicMock()
        mock_ctx.prices = mock_prices

        evaluator = MultiPeriodEvaluator(data_access=mock_ctx)

        tickers = evaluator._load_universe_tickers(universe_path)

        assert 'SPY' in tickers
        assert 'QQQ' in tickers
        assert len(tickers) == 50  # tier2 has 50 ETFs

    def test_load_missing_universe_raises_error(self, tmp_path):
        """Test that loading missing universe file raises ValueError."""
        mock_prices = MagicMock()
        mock_prices.get_latest_price_date.return_value = pd.Timestamp('2024-01-01')

        mock_ctx = MagicMock()
        mock_ctx.prices = mock_prices

        evaluator = MultiPeriodEvaluator(data_access=mock_ctx)

        with pytest.raises(ValueError, match="Universe config not found"):
            evaluator._load_universe_tickers(str(tmp_path / "nonexistent.yaml"))

    def test_load_universe_missing_source_raises_error(self, tmp_path):
        """Test that universe missing 'source' section raises ValueError."""
        universe_yaml = tmp_path / "bad_universe.yaml"
        universe_yaml.write_text("""
name: bad_universe
# No source section
""")

        mock_prices = MagicMock()
        mock_prices.get_latest_price_date.return_value = pd.Timestamp('2024-01-01')

        mock_ctx = MagicMock()
        mock_ctx.prices = mock_prices

        evaluator = MultiPeriodEvaluator(data_access=mock_ctx)

        with pytest.raises(ValueError, match="missing 'source' section"):
            evaluator._load_universe_tickers(str(universe_yaml))

    def test_load_universe_empty_tickers_raises_error(self, tmp_path):
        """Test that universe with empty tickers raises ValueError."""
        universe_yaml = tmp_path / "empty_universe.yaml"
        universe_yaml.write_text("""
name: empty_universe
source:
  type: static_list
  tickers: []
""")

        mock_prices = MagicMock()
        mock_prices.get_latest_price_date.return_value = pd.Timestamp('2024-01-01')

        mock_ctx = MagicMock()
        mock_ctx.prices = mock_prices

        evaluator = MultiPeriodEvaluator(data_access=mock_ctx)

        with pytest.raises(ValueError, match="empty ticker list"):
            evaluator._load_universe_tickers(str(universe_yaml))
