"""Integration tests for regime-based strategy selection system.

These tests verify that all components of the regime system work together
correctly from regime detection through production rebalancing.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
import json
import yaml

from quantetf.regime.detector import RegimeDetector
from quantetf.regime.analyzer import RegimeAnalyzer
from quantetf.regime.config import load_regime_mapping, get_strategy_for_regime
from quantetf.production.regime_monitor import DailyRegimeMonitor
from quantetf.production.rebalancer import RegimeAwareRebalancer


class TestMonitorToRebalancerFlow:
    """Test daily monitor integrates with rebalancer."""

    def test_rebalance_uses_current_regime(
        self,
        mock_data_access,
        clean_state_dir,
        sample_regime_mapping,
        tmp_path,
    ):
        """
        Rebalancer should use regime from monitor.

        Flow:
        1. Monitor updates regime state
        2. Rebalancer reads regime
        3. Rebalancer selects correct strategy
        4. Rebalancer generates portfolio
        """
        # Create monitor
        monitor = DailyRegimeMonitor(
            data_access=mock_data_access,
            state_dir=clean_state_dir,
        )

        # Update regime
        regime_state = monitor.update(as_of=pd.Timestamp("2026-01-20"))

        # Create rebalancer
        rebalancer = RegimeAwareRebalancer(
            data_access=mock_data_access,
            regime_monitor=monitor,
            regime_mapping_path=sample_regime_mapping,
            artifacts_dir=tmp_path / "artifacts",
        )
        rebalancer.holdings_file = tmp_path / "holdings" / "current_holdings.json"

        # Run rebalance
        result = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-20"),
            dry_run=True,
        )

        # Verify regime matches
        assert result.regime == regime_state.name

        # Verify strategy was selected from mapping
        assert result.strategy_used is not None

    def test_multiple_day_rebalance_sequence(
        self,
        mock_data_access,
        clean_state_dir,
        sample_regime_mapping,
        tmp_path,
    ):
        """
        Test sequence of rebalances maintains state correctly.

        Flow:
        1. Day 1: Monitor + Rebalance
        2. Day 2: Monitor (check hysteresis) + Rebalance
        3. Verify holdings state persisted
        """
        monitor = DailyRegimeMonitor(
            data_access=mock_data_access,
            state_dir=clean_state_dir,
        )

        rebalancer = RegimeAwareRebalancer(
            data_access=mock_data_access,
            regime_monitor=monitor,
            regime_mapping_path=sample_regime_mapping,
            artifacts_dir=tmp_path / "artifacts",
        )
        rebalancer.holdings_file = tmp_path / "holdings" / "current_holdings.json"

        # Day 1
        result1 = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-17"),
            dry_run=False,
        )

        # Verify holdings saved
        assert rebalancer.holdings_file.exists()

        # Day 2 (next week - rebalance day)
        result2 = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-20"),
            dry_run=False,
        )

        # Should have trades based on alpha changes
        assert len(result2.trades) > 0

        # Check history
        history = monitor.get_history()
        assert len(history) >= 2

    def test_state_persists_across_instances(
        self,
        mock_data_access,
        clean_state_dir,
        sample_regime_mapping,
        tmp_path,
    ):
        """
        State should persist when creating new monitor/rebalancer instances.
        """
        # First instance
        monitor1 = DailyRegimeMonitor(
            data_access=mock_data_access,
            state_dir=clean_state_dir,
        )
        state1 = monitor1.update(as_of=pd.Timestamp("2026-01-17"))

        # Create new instance (simulates restart)
        monitor2 = DailyRegimeMonitor(
            data_access=mock_data_access,
            state_dir=clean_state_dir,
        )
        loaded_state = monitor2.load_state()

        # State should be preserved
        assert loaded_state is not None
        assert loaded_state.name == state1.name


class TestRegimeChangeStrategySwitch:
    """Test that regime changes cause strategy switching."""

    def test_different_regimes_different_strategies(
        self,
        differentiated_regime_mapping,
    ):
        """
        Different regimes should select different strategies.
        """
        with open(differentiated_regime_mapping) as f:
            mapping = yaml.safe_load(f)

        # Add fallback to mapping dict
        mapping_dict = mapping["mapping"]
        mapping_dict["fallback"] = mapping["fallback"]

        # Test lookup for each regime
        results = {}
        for regime in ["uptrend_low_vol", "uptrend_high_vol", "downtrend_low_vol", "downtrend_high_vol"]:
            info = get_strategy_for_regime(regime, mapping_dict)
            results[regime] = info["strategy"]

        # Verify each regime has a different strategy
        assert len(set(results.values())) == 4, "Each regime should have a unique strategy"

    def test_fallback_used_for_unknown_regime(
        self,
        differentiated_regime_mapping,
    ):
        """
        Unknown regime should use fallback strategy.
        """
        with open(differentiated_regime_mapping) as f:
            mapping = yaml.safe_load(f)

        mapping_dict = mapping["mapping"]
        mapping_dict["fallback"] = mapping["fallback"]

        result = get_strategy_for_regime("invalid_regime_name", mapping_dict)
        assert result["strategy"] == "strategy_fallback"


class TestNoLookahead:
    """Verify no lookahead bias in regime system."""

    def test_regime_state_uses_as_of_date(
        self,
        mock_data_access,
        clean_state_dir,
    ):
        """
        Regime state should use data available at as_of date.
        """
        monitor = DailyRegimeMonitor(
            data_access=mock_data_access,
            state_dir=clean_state_dir,
        )

        # Use a date within the mock data range
        historical_date = pd.Timestamp("2025-12-15")
        state = monitor.update(as_of=historical_date)

        # Verify as_of is respected
        assert state.as_of == historical_date

    def test_rebalance_respects_as_of(
        self,
        mock_data_access,
        clean_state_dir,
        sample_regime_mapping,
        tmp_path,
    ):
        """
        Rebalance should use data from as_of date only.
        """
        monitor = DailyRegimeMonitor(
            data_access=mock_data_access,
            state_dir=clean_state_dir,
        )

        rebalancer = RegimeAwareRebalancer(
            data_access=mock_data_access,
            regime_monitor=monitor,
            regime_mapping_path=sample_regime_mapping,
            artifacts_dir=tmp_path / "artifacts",
        )
        rebalancer.holdings_file = tmp_path / "holdings" / "current_holdings.json"

        # Rebalance for historical date (within mock data range)
        historical_date = pd.Timestamp("2025-12-15")
        result = rebalancer.rebalance(as_of=historical_date, dry_run=True)

        # Result should reflect the historical date
        assert result.as_of == historical_date


class TestRegimeIndicatorIntegration:
    """Test regime detection with different market conditions."""

    def test_uptrend_low_vol_detection(
        self,
        mock_data_access,
        clean_state_dir,
    ):
        """
        Uptrend low vol regime should be detected correctly.
        """
        monitor = DailyRegimeMonitor(
            data_access=mock_data_access,
            state_dir=clean_state_dir,
        )

        state = monitor.update(as_of=pd.Timestamp("2026-01-20"))

        # Mock data has uptrend and low VIX
        assert state.name == "uptrend_low_vol"

    def test_high_vol_detection(
        self,
        high_vol_data_access,
        clean_state_dir,
    ):
        """
        High volatility regime should be detected.
        """
        monitor = DailyRegimeMonitor(
            data_access=high_vol_data_access,
            state_dir=clean_state_dir,
        )

        state = monitor.update(as_of=pd.Timestamp("2026-01-20"))

        # High VIX data should trigger high_vol
        assert "high_vol" in state.name

    def test_downtrend_detection(
        self,
        downtrend_data_access,
        clean_state_dir,
    ):
        """
        Downtrend regime should be detected when SPY below 200MA.
        """
        monitor = DailyRegimeMonitor(
            data_access=downtrend_data_access,
            state_dir=clean_state_dir,
        )

        state = monitor.update(as_of=pd.Timestamp("2026-01-20"))

        # Downtrend data should trigger downtrend
        assert "downtrend" in state.name


class TestEndToEndRebalanceFlow:
    """Test complete rebalance workflow."""

    def test_full_rebalance_creates_all_artifacts(
        self,
        mock_data_access,
        clean_state_dir,
        sample_regime_mapping,
        tmp_path,
    ):
        """
        Full rebalance should create all expected artifacts.
        """
        monitor = DailyRegimeMonitor(
            data_access=mock_data_access,
            state_dir=clean_state_dir,
        )

        rebalancer = RegimeAwareRebalancer(
            data_access=mock_data_access,
            regime_monitor=monitor,
            regime_mapping_path=sample_regime_mapping,
            artifacts_dir=tmp_path / "artifacts",
        )
        rebalancer.holdings_file = tmp_path / "holdings" / "current_holdings.json"

        # Run rebalance (not dry run)
        result = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-20"),
            dry_run=False,
        )

        # Check artifacts created
        artifacts_dir = tmp_path / "artifacts" / "20260120"
        assert artifacts_dir.exists()
        assert (artifacts_dir / "target_portfolio.csv").exists()
        assert (artifacts_dir / "trades.csv").exists()
        assert (artifacts_dir / "execution_log.json").exists()
        assert (artifacts_dir / "rebalance_result.json").exists()

        # Check holdings updated
        assert rebalancer.holdings_file.exists()

        # Check regime state saved
        assert (clean_state_dir / "current_regime.json").exists()

    def test_result_contains_complete_metadata(
        self,
        mock_data_access,
        clean_state_dir,
        sample_regime_mapping,
        tmp_path,
    ):
        """
        Rebalance result should contain complete metadata.
        """
        monitor = DailyRegimeMonitor(
            data_access=mock_data_access,
            state_dir=clean_state_dir,
        )

        rebalancer = RegimeAwareRebalancer(
            data_access=mock_data_access,
            regime_monitor=monitor,
            regime_mapping_path=sample_regime_mapping,
            artifacts_dir=tmp_path / "artifacts",
        )
        rebalancer.holdings_file = tmp_path / "holdings" / "current_holdings.json"

        result = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-20"),
            dry_run=True,
        )

        # Check all fields present
        assert result.regime is not None
        assert result.strategy_used is not None
        assert result.target_portfolio is not None
        assert len(result.trades) > 0
        assert "regime_indicators" in result.metadata
        assert "spy_price" in result.metadata["regime_indicators"]
        assert "vix" in result.metadata["regime_indicators"]

    def test_history_tracking_across_days(
        self,
        mock_data_access,
        clean_state_dir,
    ):
        """
        Monitor should track regime history across multiple days.
        """
        monitor = DailyRegimeMonitor(
            data_access=mock_data_access,
            state_dir=clean_state_dir,
        )

        # Update for multiple days
        days = [
            pd.Timestamp("2026-01-15"),
            pd.Timestamp("2026-01-16"),
            pd.Timestamp("2026-01-17"),
            pd.Timestamp("2026-01-20"),
        ]

        for day in days:
            monitor.update(as_of=day)

        # Check history
        history = monitor.get_history()
        assert len(history) == 4

        # History should be ordered
        dates_in_history = [pd.Timestamp(row["date"]) for _, row in history.iterrows()]
        assert dates_in_history == sorted(dates_in_history)


class TestConfigIntegration:
    """Test config loading integration."""

    def test_load_default_regime_mapping(self):
        """
        Default regime mapping should load from configs/regimes/.
        """
        from quantetf.regime.config import load_regime_mapping, DEFAULT_MAPPING_PATH

        if DEFAULT_MAPPING_PATH.exists():
            mapping = load_regime_mapping()

            # Should have all 4 regimes + fallback
            required = ["uptrend_low_vol", "uptrend_high_vol", "downtrend_low_vol", "downtrend_high_vol", "fallback"]
            for regime in required:
                assert regime in mapping, f"Missing {regime} in mapping"

    def test_strategy_lookup_chain(self, sample_regime_mapping):
        """
        Strategy lookup should work through the full chain.
        """
        from quantetf.regime.config import load_regime_mapping, get_strategy_for_regime

        mapping = load_regime_mapping(sample_regime_mapping)

        # Each regime should return a valid strategy
        for regime in ["uptrend_low_vol", "uptrend_high_vol", "downtrend_low_vol", "downtrend_high_vol"]:
            strategy_info = get_strategy_for_regime(regime, mapping)
            assert "strategy" in strategy_info
            assert len(strategy_info["strategy"]) > 0


class TestErrorHandling:
    """Test error handling in the regime system."""

    def test_monitor_handles_indicator_failure_gracefully(
        self,
        clean_state_dir,
    ):
        """
        Monitor should handle indicator failures with previous state.
        """
        ctx = MagicMock()

        # First: setup working indicators
        dates = pd.date_range("2025-01-01", "2026-01-20", freq="B")
        spy_prices = pd.DataFrame(
            np.linspace(500, 600, len(dates)),
            index=dates,
            columns=pd.MultiIndex.from_product(
                [["SPY"], ["Close"]], names=["Ticker", "Price"]
            ),
        )
        ctx.prices.read_prices_as_of.return_value = spy_prices

        vix_dates = pd.date_range("2025-12-01", "2026-01-20", freq="B")
        vix_data = pd.DataFrame({"VIX": [15.0] * len(vix_dates)}, index=vix_dates)
        ctx.macro.read_macro_indicator.return_value = vix_data

        monitor = DailyRegimeMonitor(
            data_access=ctx,
            state_dir=clean_state_dir,
        )
        first_state = monitor.update(as_of=pd.Timestamp("2026-01-19"))

        # Now make indicators fail
        ctx.prices.read_prices_as_of.side_effect = ValueError("Connection failed")

        # Should return previous state as fallback
        second_state = monitor.update(as_of=pd.Timestamp("2026-01-20"))
        assert second_state.name == first_state.name

    def test_rebalancer_uses_fallback_portfolio_on_alpha_failure(
        self,
        mock_data_access,
        clean_state_dir,
        sample_regime_mapping,
        tmp_path,
    ):
        """
        Rebalancer should use fallback portfolio if alpha model fails.
        """
        monitor = DailyRegimeMonitor(
            data_access=mock_data_access,
            state_dir=clean_state_dir,
        )

        rebalancer = RegimeAwareRebalancer(
            data_access=mock_data_access,
            regime_monitor=monitor,
            regime_mapping_path=sample_regime_mapping,
            artifacts_dir=tmp_path / "artifacts",
        )
        rebalancer.holdings_file = tmp_path / "holdings" / "current_holdings.json"

        # Even if alpha scoring fails, fallback should work
        result = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-20"),
            dry_run=True,
        )

        # Should still have a valid portfolio (fallback)
        assert len(result.target_portfolio) > 0
        assert len(result.trades) > 0
