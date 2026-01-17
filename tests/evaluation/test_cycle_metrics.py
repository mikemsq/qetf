"""Tests for per-cycle metrics.

Tests the CycleMetrics and related functions to ensure accurate decomposition
of backtest results into per-rebalance-cycle metrics.
"""

import pytest
import pandas as pd
import numpy as np

from quantetf.evaluation.cycle_metrics import (
    CycleResult,
    CycleMetrics,
    decompose_by_cycles,
    calculate_cycle_metrics,
    cycle_metrics_dataframe,
    print_cycle_summary,
)


class TestCycleResult:
    """Tests for CycleResult dataclass."""

    def test_basic_cycle_result(self):
        """Test creating a basic cycle result."""
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-02-01")
        result = CycleResult(
            cycle_start=start,
            cycle_end=end,
            strategy_return=0.05,
            benchmark_return=0.03,
            active_return=0.02,
            is_win=True,
        )

        assert result.cycle_start == start
        assert result.cycle_end == end
        assert result.strategy_return == 0.05
        assert result.benchmark_return == 0.03
        assert result.active_return == 0.02
        assert result.is_win is True

    def test_cycle_result_repr(self):
        """Test string representation of cycle result."""
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-02-01")
        result = CycleResult(
            cycle_start=start,
            cycle_end=end,
            strategy_return=0.05,
            benchmark_return=0.03,
            active_return=0.02,
            is_win=True,
        )

        repr_str = repr(result)
        assert "2024-01-01" in repr_str
        assert "2024-02-01" in repr_str
        assert "strat=" in repr_str


class TestCycleMetrics:
    """Tests for CycleMetrics dataclass and methods."""

    def test_meets_success_criterion_pass(self):
        """Test success criterion with 80% threshold."""
        metrics = CycleMetrics(
            total_cycles=10,
            winning_cycles=8,
            losing_cycles=2,
            win_rate=0.80,
            avg_active_return=0.01,
            avg_win_return=0.02,
            avg_loss_return=-0.03,
            best_cycle=CycleResult(
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-02-01"),
                0.10,
                0.05,
                0.05,
                True,
            ),
            worst_cycle=CycleResult(
                pd.Timestamp("2024-11-01"),
                pd.Timestamp("2024-12-01"),
                -0.05,
                0.02,
                -0.07,
                False,
            ),
            cycle_results=[],
        )

        assert metrics.meets_success_criterion(0.80) is True
        assert metrics.meets_success_criterion(0.81) is False
        assert metrics.meets_success_criterion(0.79) is True

    def test_meets_success_criterion_fail(self):
        """Test success criterion with below-threshold win rate."""
        metrics = CycleMetrics(
            total_cycles=10,
            winning_cycles=7,
            losing_cycles=3,
            win_rate=0.70,
            avg_active_return=-0.001,
            avg_win_return=0.015,
            avg_loss_return=-0.045,
            best_cycle=CycleResult(
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-02-01"),
                0.08,
                0.06,
                0.02,
                True,
            ),
            worst_cycle=CycleResult(
                pd.Timestamp("2024-11-01"),
                pd.Timestamp("2024-12-01"),
                -0.08,
                0.01,
                -0.09,
                False,
            ),
            cycle_results=[],
        )

        assert metrics.meets_success_criterion(0.80) is False
        assert metrics.meets_success_criterion(0.70) is True

    def test_summary_dict(self):
        """Test summary() method returns correct dictionary."""
        metrics = CycleMetrics(
            total_cycles=10,
            winning_cycles=8,
            losing_cycles=2,
            win_rate=0.80,
            avg_active_return=0.01,
            avg_win_return=0.02,
            avg_loss_return=-0.03,
            best_cycle=CycleResult(
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-02-01"),
                0.10,
                0.05,
                0.05,
                True,
            ),
            worst_cycle=CycleResult(
                pd.Timestamp("2024-11-01"),
                pd.Timestamp("2024-12-01"),
                -0.05,
                0.02,
                -0.07,
                False,
            ),
            cycle_results=[],
        )

        summary = metrics.summary()
        assert summary['total_cycles'] == 10
        assert summary['winning_cycles'] == 8
        assert summary['win_rate'] == 0.80
        assert summary['meets_success_criterion'] is True


class TestDecomposeByCycles:
    """Tests for decompose_by_cycles function."""

    def test_basic_decomposition(self):
        """Test basic cycle decomposition with synthetic data."""
        # Create daily dates and monthly rebalance dates
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        rebalance_dates = pd.date_range("2024-01-01", "2024-12-31", freq="MS").tolist()

        # Strategy outperforms on a trending basis
        strategy_nav = pd.Series(
            [1.0 + 0.001 * i for i in range(len(dates))],
            index=dates,
        )
        benchmark_nav = pd.Series(
            [1.0 + 0.0008 * i for i in range(len(dates))],
            index=dates,
        )

        metrics = decompose_by_cycles(strategy_nav, benchmark_nav, rebalance_dates)

        assert metrics.total_cycles == len(rebalance_dates) - 1
        assert 0 <= metrics.win_rate <= 1
        assert len(metrics.cycle_results) == metrics.total_cycles
        assert metrics.winning_cycles + metrics.losing_cycles == metrics.total_cycles

    def test_all_winning_cycles(self):
        """Test when strategy beats benchmark every cycle."""
        dates = pd.date_range("2024-01-01", "2024-06-30", freq="D")
        rebalance_dates = pd.date_range("2024-01-01", "2024-06-30", freq="MS").tolist()

        # Strategy always outperforms
        strategy_nav = pd.Series(
            [1.0 + 0.02 * i for i in range(len(dates))],
            index=dates,
        )
        benchmark_nav = pd.Series(
            [1.0 + 0.01 * i for i in range(len(dates))],
            index=dates,
        )

        metrics = decompose_by_cycles(strategy_nav, benchmark_nav, rebalance_dates)

        assert metrics.win_rate == 1.0
        assert metrics.losing_cycles == 0
        assert metrics.winning_cycles == metrics.total_cycles
        assert metrics.meets_success_criterion() is True

    def test_all_losing_cycles(self):
        """Test when benchmark beats strategy every cycle."""
        dates = pd.date_range("2024-01-01", "2024-06-30", freq="D")
        rebalance_dates = pd.date_range("2024-01-01", "2024-06-30", freq="MS").tolist()

        # Benchmark always outperforms
        strategy_nav = pd.Series(
            [1.0 + 0.01 * i for i in range(len(dates))],
            index=dates,
        )
        benchmark_nav = pd.Series(
            [1.0 + 0.02 * i for i in range(len(dates))],
            index=dates,
        )

        metrics = decompose_by_cycles(strategy_nav, benchmark_nav, rebalance_dates)

        assert metrics.win_rate == 0.0
        assert metrics.winning_cycles == 0
        assert metrics.losing_cycles == metrics.total_cycles
        assert metrics.meets_success_criterion() is False

    def test_mixed_winning_losing(self):
        """Test with mixed winning and losing cycles."""
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        rebalance_dates = pd.date_range("2024-01-01", "2024-12-31", freq="MS").tolist()

        # Create alternating performance pattern
        strategy_nav = pd.Series(
            [
                1.0 + 0.02 * i if i % 2 == 0 else 1.0 + 0.005 * i
                for i in range(len(dates))
            ],
            index=dates,
        )
        benchmark_nav = pd.Series(
            [1.0 + 0.01 * i for i in range(len(dates))],
            index=dates,
        )

        metrics = decompose_by_cycles(strategy_nav, benchmark_nav, rebalance_dates)

        assert metrics.total_cycles > 0
        assert metrics.winning_cycles > 0
        assert metrics.losing_cycles > 0
        assert (
            metrics.winning_cycles + metrics.losing_cycles == metrics.total_cycles
        )

    def test_returns_are_correct(self):
        """Test that individual cycle returns are calculated correctly."""
        # Create dates that include the rebalance dates
        dates = pd.date_range("2024-01-01", "2024-03-05", freq="D")
        rebalance_dates = [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-02-01"),
            pd.Timestamp("2024-03-01"),
        ]

        # Fixed returns for manual verification
        # First cycle (Jan 1 -> Feb 1): Strategy 100->105 (5%), Benchmark 100->104 (4%)
        # Second cycle (Feb 1 -> Mar 1): Strategy 105->110 (4.76%), Benchmark 104->108 (3.85%)
        strategy_nav = pd.Series(
            [100.0] * 31 + [105.0] * 29 + [110.0] * 5,
            index=dates,
        )
        benchmark_nav = pd.Series(
            [100.0] * 31 + [104.0] * 29 + [108.0] * 5,
            index=dates,
        )

        metrics = decompose_by_cycles(strategy_nav, benchmark_nav, rebalance_dates)

        # First cycle: 100->105 (5%), 100->104 (4%), active=1%
        assert abs(metrics.cycle_results[0].strategy_return - 0.05) < 1e-3
        assert abs(metrics.cycle_results[0].benchmark_return - 0.04) < 1e-3
        assert abs(metrics.cycle_results[0].active_return - 0.01) < 1e-3
        assert metrics.cycle_results[0].is_win is True

    def test_best_worst_cycles(self):
        """Test that best and worst cycles are identified correctly."""
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        rebalance_dates = pd.date_range("2024-01-01", "2024-12-31", freq="MS").tolist()

        strategy_nav = pd.Series(
            [1.0 + 0.001 * i for i in range(len(dates))],
            index=dates,
        )
        benchmark_nav = pd.Series(
            [1.0 + 0.0008 * i for i in range(len(dates))],
            index=dates,
        )

        metrics = decompose_by_cycles(strategy_nav, benchmark_nav, rebalance_dates)

        # Best cycle should have highest active return
        best_active = max(c.active_return for c in metrics.cycle_results)
        assert abs(metrics.best_cycle.active_return - best_active) < 1e-6

        # Worst cycle should have lowest active return
        worst_active = min(c.active_return for c in metrics.cycle_results)
        assert abs(metrics.worst_cycle.active_return - worst_active) < 1e-6

    def test_invalid_rebalance_dates(self):
        """Test error handling with invalid rebalance dates."""
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        strategy_nav = pd.Series([1.0 + 0.001 * i for i in range(len(dates))], index=dates)
        benchmark_nav = pd.Series([1.0 + 0.0008 * i for i in range(len(dates))], index=dates)

        # Single rebalance date (need at least 2)
        with pytest.raises(ValueError, match="at least 2 rebalance dates"):
            decompose_by_cycles(strategy_nav, benchmark_nav, [pd.Timestamp("2024-01-01")])

    def test_empty_curves(self):
        """Test error handling with empty curves."""
        rebalance_dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")]

        with pytest.raises(ValueError, match="cannot be empty"):
            decompose_by_cycles(
                pd.Series([], dtype=float),
                pd.Series([1.0, 1.1], index=rebalance_dates),
                rebalance_dates,
            )

    def test_negative_nav_values(self):
        """Test error handling with negative NAV values."""
        dates = pd.date_range("2024-01-01", "2024-06-30", freq="D")
        rebalance_dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30")]

        strategy_nav = pd.Series([-1.0] + [1.0 + 0.001 * i for i in range(len(dates) - 1)], index=dates)
        benchmark_nav = pd.Series([1.0] + [1.0 + 0.0008 * i for i in range(len(dates) - 1)], index=dates)

        with pytest.raises(ValueError, match="Invalid NAV values"):
            decompose_by_cycles(strategy_nav, benchmark_nav, rebalance_dates)


class TestCalculateCycleMetrics:
    """Tests for calculate_cycle_metrics convenience function."""

    def test_empty_benchmark_prices(self):
        """Test error handling with empty benchmark prices."""
        from quantetf.backtest.simple_engine import BacktestResult, BacktestConfig

        config = BacktestConfig(
            start_date=pd.Timestamp("2024-01-01"),
            end_date=pd.Timestamp("2024-12-31"),
            universe=None,
            initial_capital=100000.0,
        )

        result = BacktestResult(
            equity_curve=pd.DataFrame(),
            holdings_history=pd.DataFrame(),
            weights_history=pd.DataFrame(),
            metrics={},
            config=config,
            rebalance_dates=[],
        )

        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_cycle_metrics(
                result,
                pd.Series([], dtype=float),
                [],
            )


class TestCycleMetricsDataframe:
    """Tests for cycle_metrics_dataframe function."""

    def test_dataframe_conversion(self):
        """Test converting cycle results to DataFrame."""
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        rebalance_dates = pd.date_range("2024-01-01", "2024-12-31", freq="MS").tolist()

        strategy_nav = pd.Series(
            [1.0 + 0.001 * i for i in range(len(dates))],
            index=dates,
        )
        benchmark_nav = pd.Series(
            [1.0 + 0.0008 * i for i in range(len(dates))],
            index=dates,
        )

        metrics = decompose_by_cycles(strategy_nav, benchmark_nav, rebalance_dates)
        df = cycle_metrics_dataframe(metrics)

        assert len(df) == metrics.total_cycles
        assert 'start_date' in df.columns
        assert 'end_date' in df.columns
        assert 'strat_return' in df.columns
        assert 'bench_return' in df.columns
        assert 'active_return' in df.columns
        assert 'is_win' in df.columns

    def test_dataframe_values(self):
        """Test that DataFrame values match cycle results."""
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        rebalance_dates = pd.date_range("2024-01-01", "2024-12-31", freq="MS").tolist()

        strategy_nav = pd.Series(
            [1.0 + 0.001 * i for i in range(len(dates))],
            index=dates,
        )
        benchmark_nav = pd.Series(
            [1.0 + 0.0008 * i for i in range(len(dates))],
            index=dates,
        )

        metrics = decompose_by_cycles(strategy_nav, benchmark_nav, rebalance_dates)
        df = cycle_metrics_dataframe(metrics)

        # Check first cycle
        assert df.loc[0, 'start_date'] == metrics.cycle_results[0].cycle_start
        assert df.loc[0, 'end_date'] == metrics.cycle_results[0].cycle_end
        assert (
            abs(df.loc[0, 'strat_return'] - metrics.cycle_results[0].strategy_return)
            < 1e-6
        )
        assert (
            abs(df.loc[0, 'active_return'] - metrics.cycle_results[0].active_return)
            < 1e-6
        )


class TestPrintCycleSummary:
    """Tests for print_cycle_summary function."""

    def test_print_cycle_summary_passes(self, capsys):
        """Test print output for passing strategy."""
        metrics = CycleMetrics(
            total_cycles=10,
            winning_cycles=8,
            losing_cycles=2,
            win_rate=0.80,
            avg_active_return=0.01,
            avg_win_return=0.02,
            avg_loss_return=-0.03,
            best_cycle=CycleResult(
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-02-01"),
                0.10,
                0.05,
                0.05,
                True,
            ),
            worst_cycle=CycleResult(
                pd.Timestamp("2024-11-01"),
                pd.Timestamp("2024-12-01"),
                -0.05,
                0.02,
                -0.07,
                False,
            ),
            cycle_results=[],
        )

        print_cycle_summary(metrics)
        captured = capsys.readouterr()

        assert "CYCLE WIN RATE ANALYSIS" in captured.out
        assert "80.0%" in captured.out
        assert "✓ PASS" in captured.out
        assert "10" in captured.out
        assert "8" in captured.out

    def test_print_cycle_summary_fails(self, capsys):
        """Test print output for failing strategy."""
        metrics = CycleMetrics(
            total_cycles=10,
            winning_cycles=7,
            losing_cycles=3,
            win_rate=0.70,
            avg_active_return=-0.001,
            avg_win_return=0.015,
            avg_loss_return=-0.045,
            best_cycle=CycleResult(
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-02-01"),
                0.08,
                0.06,
                0.02,
                True,
            ),
            worst_cycle=CycleResult(
                pd.Timestamp("2024-11-01"),
                pd.Timestamp("2024-12-01"),
                -0.08,
                0.01,
                -0.09,
                False,
            ),
            cycle_results=[],
        )

        print_cycle_summary(metrics)
        captured = capsys.readouterr()

        assert "70.0%" in captured.out
        assert "✗ FAIL" in captured.out
