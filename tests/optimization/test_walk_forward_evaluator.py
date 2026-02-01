"""Tests for walk-forward evaluator module.

Tests cover:
- WalkForwardEvaluatorConfig validation
- WindowResult and WalkForwardEvaluationResult dataclasses
- WalkForwardEvaluator class with mocked data
- Composite score calculation
- Window aggregation logic
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from quantetf.optimization.walk_forward_evaluator import (
    WalkForwardEvaluationResult,
    WalkForwardEvaluator,
    WalkForwardEvaluatorConfig,
    WindowResult,
)


class TestWalkForwardEvaluatorConfig:
    """Tests for WalkForwardEvaluatorConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = WalkForwardEvaluatorConfig()
        assert config.train_years == 3
        assert config.test_years == 1
        assert config.step_months == 6
        assert config.min_windows == 4
        assert config.require_positive_oos is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = WalkForwardEvaluatorConfig(
            train_years=2,
            test_years=2,
            step_months=12,
            min_windows=3,
            require_positive_oos=False,
        )
        assert config.train_years == 2
        assert config.test_years == 2
        assert config.step_months == 12
        assert config.min_windows == 3
        assert config.require_positive_oos is False

    def test_invalid_train_years(self):
        """Test validation rejects invalid train_years."""
        with pytest.raises(ValueError, match="train_years must be >= 1"):
            WalkForwardEvaluatorConfig(train_years=0)

    def test_invalid_test_years(self):
        """Test validation rejects invalid test_years."""
        with pytest.raises(ValueError, match="test_years must be >= 1"):
            WalkForwardEvaluatorConfig(test_years=0)

    def test_invalid_step_months_low(self):
        """Test validation rejects step_months < 1."""
        with pytest.raises(ValueError, match="step_months must be 1-12"):
            WalkForwardEvaluatorConfig(step_months=0)

    def test_invalid_step_months_high(self):
        """Test validation rejects step_months > 12."""
        with pytest.raises(ValueError, match="step_months must be 1-12"):
            WalkForwardEvaluatorConfig(step_months=13)

    def test_invalid_min_windows(self):
        """Test validation rejects invalid min_windows."""
        with pytest.raises(ValueError, match="min_windows must be >= 1"):
            WalkForwardEvaluatorConfig(min_windows=0)


class TestWindowResult:
    """Tests for WindowResult dataclass."""

    def test_creation(self):
        """Test creating a WindowResult."""
        result = WindowResult(
            window_id=1,
            train_start=datetime(2016, 1, 1),
            train_end=datetime(2019, 1, 1),
            test_start=datetime(2019, 1, 1),
            test_end=datetime(2020, 1, 1),
            is_return=0.25,
            is_sharpe=1.2,
            is_volatility=0.15,
            oos_return=0.10,
            oos_sharpe=0.8,
            oos_volatility=0.18,
            oos_max_drawdown=-0.15,
            spy_return=0.08,
            active_return=0.02,
        )
        assert result.window_id == 1
        assert result.oos_return == 0.10
        assert result.active_return == 0.02

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = WindowResult(
            window_id=1,
            train_start=datetime(2016, 1, 1),
            train_end=datetime(2019, 1, 1),
            test_start=datetime(2019, 1, 1),
            test_end=datetime(2020, 1, 1),
            is_return=0.25,
            is_sharpe=1.2,
            is_volatility=0.15,
            oos_return=0.10,
            oos_sharpe=0.8,
            oos_volatility=0.18,
            oos_max_drawdown=-0.15,
            spy_return=0.08,
            active_return=0.02,
        )
        d = result.to_dict()
        assert d["window_id"] == 1
        assert d["oos_return"] == 0.10
        assert "oos_daily_returns" not in d  # Excluded from dict


class TestWalkForwardEvaluationResult:
    """Tests for WalkForwardEvaluationResult dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        result = WalkForwardEvaluationResult(config_name="test_strategy")
        assert result.config_name == "test_strategy"
        assert result.num_windows == 0
        assert result.oos_sharpe_mean == 0.0
        assert result.window_results == []

    def test_passed_filter_positive(self):
        """Test passed_filter with positive active return."""
        result = WalkForwardEvaluationResult(
            config_name="test",
            oos_active_return_mean=0.05,
        )
        assert result.passed_filter() is True

    def test_passed_filter_negative(self):
        """Test passed_filter with negative active return."""
        result = WalkForwardEvaluationResult(
            config_name="test",
            oos_active_return_mean=-0.02,
        )
        assert result.passed_filter() is False

    def test_passed_filter_zero(self):
        """Test passed_filter with zero active return."""
        result = WalkForwardEvaluationResult(
            config_name="test",
            oos_active_return_mean=0.0,
        )
        assert result.passed_filter() is False

    def test_to_dict(self):
        """Test conversion to dictionary for CSV export."""
        result = WalkForwardEvaluationResult(
            config_name="momentum_top5_monthly",
            num_windows=7,
            oos_sharpe_mean=0.82,
            oos_sharpe_std=0.15,
            oos_return_mean=0.12,
            oos_active_return_mean=0.042,
            oos_win_rate=0.71,
            is_sharpe_mean=0.95,
            is_return_mean=0.18,
            sharpe_degradation=0.13,
            return_degradation=0.06,
            composite_score=1.42,
        )
        d = result.to_dict()
        assert d["config_name"] == "momentum_top5_monthly"
        assert d["num_windows"] == 7
        assert d["oos_sharpe_mean"] == 0.82
        assert d["composite_score"] == 1.42
        # window_results and config should not be in dict
        assert "window_results" not in d
        assert "config" not in d


class TestWalkForwardEvaluator:
    """Tests for WalkForwardEvaluator class."""

    @pytest.fixture
    def mock_data_access(self):
        """Create mock DataAccessContext."""
        mock = Mock()
        # Create price data spanning 10 years
        dates = pd.date_range("2016-01-01", "2026-01-01", freq="B")
        prices = pd.DataFrame(
            {
                "SPY": np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01)),
                "QQQ": np.exp(np.cumsum(np.random.randn(len(dates)) * 0.015)),
            },
            index=dates,
        )
        mock.prices.read_prices_as_of = Mock(return_value=prices)
        mock.prices.get_available_tickers = Mock(return_value=["SPY", "QQQ"])
        return mock

    @pytest.fixture
    def evaluator(self, mock_data_access):
        """Create evaluator instance."""
        return WalkForwardEvaluator(
            data_access=mock_data_access,
            wf_config=WalkForwardEvaluatorConfig(
                train_years=2,
                test_years=1,
                step_months=6,
                min_windows=2,
            ),
            cost_bps=10.0,
        )

    def test_initialization(self, mock_data_access):
        """Test evaluator initialization."""
        evaluator = WalkForwardEvaluator(
            data_access=mock_data_access,
            wf_config=WalkForwardEvaluatorConfig(train_years=2, test_years=1),
            cost_bps=15.0,
        )
        assert evaluator.wf_config.train_years == 2
        assert evaluator.wf_config.test_years == 1
        assert evaluator.cost_bps == 15.0

    def test_default_config(self, mock_data_access):
        """Test evaluator uses default config if none provided."""
        evaluator = WalkForwardEvaluator(data_access=mock_data_access)
        assert evaluator.wf_config.train_years == 3
        assert evaluator.wf_config.test_years == 1
        assert evaluator.wf_config.step_months == 6

    def test_calculate_metrics_basic(self, evaluator):
        """Test metric calculation with realistic returns."""
        # Create realistic return series with some variance
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)  # ~0.05% daily + noise
        metrics = evaluator._calculate_metrics(returns)

        assert metrics["total_return"] != 0  # Some non-zero return
        assert metrics["sharpe"] != 0  # Non-zero Sharpe
        assert metrics["volatility"] > 0  # Has volatility

    def test_calculate_metrics_empty(self, evaluator):
        """Test metric calculation with empty series."""
        returns = pd.Series(dtype=float)
        metrics = evaluator._calculate_metrics(returns)

        assert metrics["total_return"] == 0.0
        assert metrics["sharpe"] == 0.0
        assert metrics["volatility"] == 0.0
        assert metrics["max_drawdown"] == 0.0

    def test_calculate_metrics_with_drawdown(self, evaluator):
        """Test max drawdown calculation."""
        # Create series with a drawdown
        returns = pd.Series([0.05, 0.05, -0.15, 0.05, 0.05])
        metrics = evaluator._calculate_metrics(returns)

        assert metrics["max_drawdown"] < 0  # Should be negative

    def test_composite_score_calculation(self, evaluator):
        """Test composite score uses OOS metrics only."""
        result = WalkForwardEvaluationResult(
            config_name="test",
            oos_sharpe_mean=0.5,
            oos_active_return_mean=0.05,  # 5%
            oos_win_rate=0.7,
            is_sharpe_mean=2.0,  # High IS should NOT inflate score
            is_return_mean=0.30,
        )
        score = evaluator._calculate_composite_score(result)

        # Score should be moderate despite high IS
        # 0.4 * (0.05 * 10) + 0.3 * 0.5 + 0.3 * 0.7 = 0.2 + 0.15 + 0.21 = 0.56
        expected = 0.4 * 0.5 + 0.3 * 0.5 + 0.3 * 0.7
        assert abs(score - expected) < 0.01

    def test_composite_score_negative_active_return(self, evaluator):
        """Test composite score with negative active return."""
        result = WalkForwardEvaluationResult(
            config_name="test",
            oos_sharpe_mean=0.5,
            oos_active_return_mean=-0.05,  # -5%
            oos_win_rate=0.3,
        )
        score = evaluator._calculate_composite_score(result)

        # Should be low due to negative active return
        assert score < 0.5


class TestOverfitDetection:
    """Tests for detecting overfit strategies."""

    def test_overfit_strategy_shows_degradation(self):
        """Test that overfit strategy shows IS >> OOS."""
        # Simulate an overfit strategy result
        result = WalkForwardEvaluationResult(
            config_name="overfit_strategy",
            num_windows=7,
            oos_sharpe_mean=0.21,
            is_sharpe_mean=2.72,
            sharpe_degradation=2.51,  # Large degradation!
        )

        # Should show significant degradation
        assert result.sharpe_degradation > 0.5
        assert result.is_sharpe_mean > result.oos_sharpe_mean

    def test_robust_strategy_low_degradation(self):
        """Test that robust strategy shows low degradation."""
        result = WalkForwardEvaluationResult(
            config_name="robust_strategy",
            num_windows=7,
            oos_sharpe_mean=0.82,
            is_sharpe_mean=0.95,
            sharpe_degradation=0.13,  # Low degradation
        )

        assert result.sharpe_degradation < 0.5


class TestAggregateResults:
    """Tests for result aggregation logic."""

    @pytest.fixture
    def mock_data_access(self):
        """Create mock DataAccessContext."""
        mock = Mock()
        dates = pd.date_range("2016-01-01", "2026-01-01", freq="B")
        prices = pd.DataFrame(
            {"SPY": np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01))},
            index=dates,
        )
        mock.prices.read_prices_as_of = Mock(return_value=prices)
        mock.prices.get_available_tickers = Mock(return_value=["SPY"])
        return mock

    @pytest.fixture
    def evaluator(self, mock_data_access):
        """Create evaluator instance."""
        return WalkForwardEvaluator(
            data_access=mock_data_access,
            wf_config=WalkForwardEvaluatorConfig(
                train_years=2,
                test_years=1,
                step_months=6,
                min_windows=2,
            ),
            cost_bps=10.0,
        )

    def test_aggregate_results_win_rate(self, evaluator):
        """Test win rate calculation across windows."""
        window_results = [
            WindowResult(
                window_id=i,
                train_start=datetime(2016, 1, 1),
                train_end=datetime(2019, 1, 1),
                test_start=datetime(2019, 1, 1),
                test_end=datetime(2020, 1, 1),
                is_return=0.20,
                is_sharpe=1.0,
                is_volatility=0.15,
                oos_return=0.10,
                oos_sharpe=0.7,
                oos_volatility=0.18,
                oos_max_drawdown=-0.10,
                spy_return=0.08,
                active_return=0.02 if i % 2 == 0 else -0.01,  # Alternating
            )
            for i in range(4)
        ]

        # Create a mock strategy config
        mock_config = Mock()
        mock_config.generate_name = Mock(return_value="test_strategy")

        result = evaluator._aggregate_results(
            config_name="test",
            config=mock_config,
            window_results=window_results,
        )

        # 2 out of 4 windows have positive active return
        assert result.oos_win_rate == 0.5

    def test_aggregate_results_means(self, evaluator):
        """Test mean calculation across windows."""
        window_results = [
            WindowResult(
                window_id=0,
                train_start=datetime(2016, 1, 1),
                train_end=datetime(2019, 1, 1),
                test_start=datetime(2019, 1, 1),
                test_end=datetime(2020, 1, 1),
                is_return=0.20,
                is_sharpe=1.0,
                is_volatility=0.15,
                oos_return=0.10,
                oos_sharpe=0.8,
                oos_volatility=0.18,
                oos_max_drawdown=-0.10,
                spy_return=0.08,
                active_return=0.02,
            ),
            WindowResult(
                window_id=1,
                train_start=datetime(2016, 7, 1),
                train_end=datetime(2019, 7, 1),
                test_start=datetime(2019, 7, 1),
                test_end=datetime(2020, 7, 1),
                is_return=0.30,
                is_sharpe=1.2,
                is_volatility=0.18,
                oos_return=0.12,
                oos_sharpe=0.6,
                oos_volatility=0.20,
                oos_max_drawdown=-0.15,
                spy_return=0.10,
                active_return=0.02,
            ),
        ]

        mock_config = Mock()
        mock_config.generate_name = Mock(return_value="test_strategy")

        result = evaluator._aggregate_results(
            config_name="test",
            config=mock_config,
            window_results=window_results,
        )

        # Check means
        assert result.oos_sharpe_mean == pytest.approx(0.7, abs=0.01)
        assert result.oos_return_mean == pytest.approx(0.11, abs=0.01)
        assert result.is_sharpe_mean == pytest.approx(1.1, abs=0.01)

        # Check degradation
        assert result.sharpe_degradation == pytest.approx(0.4, abs=0.01)


class TestWindowGeneration:
    """Tests for window generation (IMPL-036-C)."""

    @pytest.fixture
    def mock_data_access(self):
        """Create mock DataAccessContext with 10 years of data."""
        mock = Mock()
        dates = pd.date_range("2016-01-01", "2026-01-01", freq="B")
        prices = pd.DataFrame(
            {
                "SPY": np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01)),
                "QQQ": np.exp(np.cumsum(np.random.randn(len(dates)) * 0.015)),
            },
            index=dates,
        )
        mock.prices.read_prices_as_of = Mock(return_value=prices)
        mock.prices.get_available_tickers = Mock(return_value=["SPY", "QQQ"])
        return mock

    @pytest.fixture
    def evaluator(self, mock_data_access):
        """Create evaluator with 2yr train, 1yr test, 6mo step."""
        return WalkForwardEvaluator(
            data_access=mock_data_access,
            wf_config=WalkForwardEvaluatorConfig(
                train_years=2,
                test_years=1,
                step_months=6,
                min_windows=2,
            ),
            cost_bps=10.0,
        )

    def test_window_generation_basic(self, evaluator):
        """Test that windows are generated correctly."""
        windows = evaluator._generate_windows()

        assert len(windows) >= 2
        for w in windows:
            # Test start should equal train end (no gap)
            assert w.test_start == w.train_end
            # Test period should be ~1 year (300-400 days)
            test_days = (w.test_end - w.test_start).days
            assert 300 < test_days < 400

    def test_windows_cached(self, evaluator):
        """Test that windows are cached after first generation."""
        windows1 = evaluator._generate_windows()
        windows2 = evaluator._generate_windows()
        assert windows1 is windows2  # Same object reference

    def test_window_generation_sequential_ids(self, evaluator):
        """Test that window IDs are sequential starting from 0."""
        windows = evaluator._generate_windows()

        for i, window in enumerate(windows):
            assert window.window_id == i

    def test_window_generation_train_period_length(self, evaluator):
        """Test that training periods are approximately correct length."""
        windows = evaluator._generate_windows()

        for w in windows:
            train_days = (w.train_end - w.train_start).days
            # 2 years = ~730 days, allow some variance
            assert 700 < train_days < 760

    def test_window_generation_step_forward(self, evaluator):
        """Test that windows step forward by configured months."""
        windows = evaluator._generate_windows()

        if len(windows) >= 2:
            # Each window should start 6 months after the previous
            expected_step = pd.DateOffset(months=6)
            assert windows[1].train_start == windows[0].train_start + expected_step

    def test_window_generation_insufficient_windows(self, mock_data_access):
        """Test error when min_windows cannot be met."""
        # Create evaluator requiring 20 windows (impossible with 10yr data)
        evaluator = WalkForwardEvaluator(
            data_access=mock_data_access,
            wf_config=WalkForwardEvaluatorConfig(
                train_years=3,
                test_years=1,
                step_months=6,
                min_windows=20,  # Too many
            ),
        )

        with pytest.raises(ValueError, match="Insufficient data"):
            evaluator._generate_windows()


class TestSingleWindowEvaluation:
    """Tests for single-window evaluation (IMPL-036-D)."""

    @pytest.fixture
    def mock_data_access(self):
        """Create mock DataAccessContext."""
        mock = Mock()
        np.random.seed(42)
        dates = pd.date_range("2016-01-01", "2026-01-01", freq="B")

        # Create SPY prices with realistic growth
        spy_returns = np.random.randn(len(dates)) * 0.01 + 0.0003  # Slight positive drift
        spy_prices = 100 * np.exp(np.cumsum(spy_returns))

        prices = pd.DataFrame({"SPY": spy_prices}, index=dates)
        mock.prices.read_prices_as_of = Mock(return_value=prices)
        mock.prices.get_available_tickers = Mock(return_value=["SPY"])
        return mock

    @pytest.fixture
    def evaluator(self, mock_data_access):
        """Create evaluator instance."""
        return WalkForwardEvaluator(
            data_access=mock_data_access,
            wf_config=WalkForwardEvaluatorConfig(
                train_years=2,
                test_years=1,
                step_months=6,
                min_windows=2,
            ),
            cost_bps=10.0,
        )

    def test_get_spy_return_for_window(self, evaluator):
        """Test SPY return calculation for a window."""
        from quantetf.evaluation.walk_forward import WalkForwardWindow

        window = WalkForwardWindow(
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2022-01-01"),
            test_start=pd.Timestamp("2022-01-01"),
            test_end=pd.Timestamp("2023-01-01"),
            window_id=0,
        )

        spy_return = evaluator._get_spy_return_for_window(window)

        # Should be a float representing the return
        assert isinstance(spy_return, float)
        # SPY return should be reasonable (between -50% and +100%)
        assert -0.5 < spy_return < 1.0

    def test_get_spy_return_for_window_empty_data(self, mock_data_access):
        """Test SPY return with empty data returns 0."""
        # Override mock to return empty DataFrame
        mock_data_access.prices.read_prices_as_of = Mock(return_value=pd.DataFrame())

        evaluator = WalkForwardEvaluator(
            data_access=mock_data_access,
            wf_config=WalkForwardEvaluatorConfig(min_windows=1),
        )

        from quantetf.evaluation.walk_forward import WalkForwardWindow

        window = WalkForwardWindow(
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2022-01-01"),
            test_start=pd.Timestamp("2022-01-01"),
            test_end=pd.Timestamp("2023-01-01"),
            window_id=0,
        )

        spy_return = evaluator._get_spy_return_for_window(window)
        assert spy_return == 0.0

    def test_extract_period_returns_basic(self, evaluator):
        """Test extracting returns for a specific period."""
        # Create a mock backtest result with equity curve
        dates = pd.date_range("2020-01-01", "2023-01-01", freq="B")
        equity_values = 100000 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01))
        equity_curve = pd.DataFrame({"portfolio_value": equity_values}, index=dates)

        from quantetf.backtest.simple_engine import BacktestResult, BacktestConfig

        mock_result = Mock(spec=BacktestResult)
        mock_result.equity_curve = equity_curve

        # Extract returns for 2021
        returns = evaluator._extract_period_returns(
            mock_result,
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2021-12-31"),
        )

        assert isinstance(returns, pd.Series)
        assert len(returns) > 0
        # Returns should be within reasonable bounds
        assert returns.min() > -0.5
        assert returns.max() < 0.5

    def test_extract_period_returns_empty_period(self, evaluator):
        """Test extracting returns for period with no data."""
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="B")
        equity_curve = pd.DataFrame(
            {"portfolio_value": np.linspace(100000, 110000, len(dates))},
            index=dates,
        )

        mock_result = Mock()
        mock_result.equity_curve = equity_curve

        # Extract returns for period outside data range
        returns = evaluator._extract_period_returns(
            mock_result,
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-12-31"),
        )

        assert isinstance(returns, pd.Series)
        assert len(returns) == 0

    def test_calculate_metrics_positive_returns(self, evaluator):
        """Test metrics with consistent positive returns."""
        # Consistent 1% daily returns
        returns = pd.Series([0.01] * 252)
        metrics = evaluator._calculate_metrics(returns)

        assert metrics["total_return"] > 0
        # With zero volatility, sharpe is undefined (returns 0)
        assert metrics["sharpe"] == 0.0
        assert metrics["volatility"] == 0.0  # No variance
        assert metrics["max_drawdown"] == 0.0  # No drawdown

    def test_calculate_metrics_mixed_returns(self, evaluator):
        """Test metrics with realistic mixed returns."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0003)
        metrics = evaluator._calculate_metrics(returns)

        assert "total_return" in metrics
        assert "sharpe" in metrics
        assert "volatility" in metrics
        assert metrics["volatility"] > 0
        assert "max_drawdown" in metrics
        assert metrics["max_drawdown"] <= 0  # Drawdown is negative or zero

    def test_calculate_metrics_single_value(self, evaluator):
        """Test metrics with single return value returns zeros."""
        returns = pd.Series([0.05])
        metrics = evaluator._calculate_metrics(returns)

        # With only one return, metrics are set to 0 (need >=2 for meaningful stats)
        assert metrics["total_return"] == 0.0
        assert metrics["sharpe"] == 0.0
        assert metrics["volatility"] == 0.0
        assert metrics["max_drawdown"] == 0.0


class TestEvaluatorIntegration:
    """Integration tests for WalkForwardEvaluator."""

    @pytest.fixture
    def mock_data_access(self):
        """Create mock DataAccessContext with MultiIndex columns."""
        mock = Mock()
        np.random.seed(42)
        dates = pd.date_range("2016-01-01", "2026-01-01", freq="B")

        # Create realistic price data
        spy_returns = np.random.randn(len(dates)) * 0.01 + 0.0003
        spy_prices = 100 * np.exp(np.cumsum(spy_returns))

        # Create DataFrame with MultiIndex columns like real data
        columns = pd.MultiIndex.from_tuples(
            [("SPY", "Close")], names=["Ticker", "Price"]
        )
        prices = pd.DataFrame(spy_prices, index=dates, columns=columns)

        # Also support simple column access
        def mock_read_prices(as_of, tickers):
            simple_prices = pd.DataFrame({"SPY": spy_prices}, index=dates)
            mask = dates <= as_of
            return simple_prices[mask]

        mock.prices.read_prices_as_of = Mock(side_effect=mock_read_prices)
        mock.prices.get_available_tickers = Mock(return_value=["SPY"])
        return mock

    def test_evaluator_handles_failed_window_gracefully(self, mock_data_access):
        """Test that evaluator continues if individual window fails."""
        evaluator = WalkForwardEvaluator(
            data_access=mock_data_access,
            wf_config=WalkForwardEvaluatorConfig(
                train_years=2,
                test_years=1,
                step_months=6,
                min_windows=2,
            ),
        )

        # Mock strategy config that will fail
        mock_config = Mock()
        mock_config.generate_name = Mock(return_value="failing_strategy")

        # Make _evaluate_window raise an exception
        with patch.object(
            evaluator, "_evaluate_window", side_effect=Exception("Window failed")
        ):
            result = evaluator.evaluate(mock_config)

        # Should return a failed result, not crash
        assert result.num_windows == 0
        assert result.composite_score == float("-inf")
