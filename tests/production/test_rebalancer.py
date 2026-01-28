"""Tests for regime-aware production rebalancer."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

from quantetf.production.rebalancer import (
    RegimeAwareRebalancer,
    Trade,
    RebalanceResult,
)
from quantetf.production.regime_monitor import DailyRegimeMonitor
from quantetf.regime.types import RegimeState, TrendState, VolatilityState


class TestTrade:
    """Test Trade dataclass."""

    def test_shares_delta_buy(self):
        """Positive delta for buy."""
        trade = Trade(
            ticker="SPY",
            action="BUY",
            current_shares=10,
            target_shares=15,
            current_weight=0.1,
            target_weight=0.15,
            notional_value=500,
        )
        assert trade.shares_delta == 5

    def test_shares_delta_sell(self):
        """Negative delta for sell."""
        trade = Trade(
            ticker="SPY",
            action="SELL",
            current_shares=15,
            target_shares=10,
            current_weight=0.15,
            target_weight=0.1,
            notional_value=500,
        )
        assert trade.shares_delta == -5

    def test_shares_delta_hold(self):
        """Zero delta for hold."""
        trade = Trade(
            ticker="SPY",
            action="HOLD",
            current_shares=10,
            target_shares=10,
            current_weight=0.1,
            target_weight=0.1,
            notional_value=0,
        )
        assert trade.shares_delta == 0


class TestRebalanceResult:
    """Test RebalanceResult dataclass."""

    def test_to_dict(self):
        """to_dict should include key fields."""
        target = pd.DataFrame(
            {"weight": [0.5, 0.5], "shares": [10, 10]},
            index=["SPY", "QQQ"],
        )
        trades = [
            Trade("SPY", "BUY", 0, 10, 0, 0.5, 5000),
            Trade("QQQ", "BUY", 0, 10, 0, 0.5, 5000),
        ]

        result = RebalanceResult(
            as_of=pd.Timestamp("2026-01-20"),
            regime="uptrend_low_vol",
            strategy_used="momentum_acceleration",
            target_portfolio=target,
            trades=trades,
            current_portfolio=pd.DataFrame(),
            metadata={},
        )

        d = result.to_dict()

        assert d["regime"] == "uptrend_low_vol"
        assert d["strategy_used"] == "momentum_acceleration"
        assert d["num_positions"] == 2
        assert d["num_trades"] == 2

    def test_turnover_calculation(self):
        """Turnover should be calculated correctly."""
        trades = [
            Trade("SPY", "BUY", 0, 10, 0, 0.5, 5000),
            Trade("QQQ", "HOLD", 10, 10, 0.5, 0.5, 0),
        ]

        result = RebalanceResult(
            as_of=pd.Timestamp("2026-01-20"),
            regime="uptrend_low_vol",
            strategy_used="test",
            target_portfolio=pd.DataFrame(),
            trades=trades,
            current_portfolio=pd.DataFrame(),
            metadata={},
        )

        # Only SPY is traded, notional = 5000
        # Total = 5000, turnover = 5000/5000 = 1.0
        assert result._calculate_turnover() == 1.0


class TestRegimeAwareRebalancer:
    """Test regime-aware rebalancing."""

    @pytest.fixture
    def mock_data_access(self):
        """Create mock data access context."""
        ctx = MagicMock()

        # Mock SPY prices for regime detection
        dates = pd.date_range("2025-01-01", "2026-01-20", freq="B")
        spy_prices = pd.DataFrame(
            np.linspace(500, 600, len(dates)),
            index=dates,
            columns=pd.MultiIndex.from_product(
                [["SPY"], ["Close"]], names=["Ticker", "Price"]
            ),
        )
        ctx.prices.read_prices_as_of.return_value = spy_prices

        # Mock VIX data
        vix_dates = pd.date_range("2025-12-01", "2026-01-20", freq="B")
        vix_data = pd.DataFrame(
            {"VIX": np.random.uniform(12, 20, len(vix_dates))},
            index=vix_dates,
        )
        ctx.macro.read_macro_indicator.return_value = vix_data

        return ctx

    @pytest.fixture
    def state_dir(self, tmp_path):
        return tmp_path / "state"

    @pytest.fixture
    def rebalancer(self, mock_data_access, tmp_path, state_dir):
        monitor = DailyRegimeMonitor(
            data_access=mock_data_access,
            state_dir=state_dir,
        )
        rebalancer = RegimeAwareRebalancer(
            data_access=mock_data_access,
            regime_monitor=monitor,
            artifacts_dir=tmp_path / "artifacts",
        )
        # Use temp directory for holdings file
        rebalancer.holdings_file = tmp_path / "holdings" / "current_holdings.json"
        return rebalancer

    def test_rebalance_returns_result(self, rebalancer):
        """Rebalance should return RebalanceResult."""
        result = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-20"),
            dry_run=True,
        )

        assert result is not None
        assert isinstance(result, RebalanceResult)
        assert result.regime in [
            "uptrend_low_vol",
            "uptrend_high_vol",
            "downtrend_low_vol",
            "downtrend_high_vol",
        ]

    def test_dry_run_no_state_change(self, rebalancer):
        """Dry run should not update holdings."""
        rebalancer.rebalance(as_of=pd.Timestamp("2026-01-20"), dry_run=True)

        # Holdings file should not exist
        assert not rebalancer.holdings_file.exists()

    def test_rebalance_saves_artifacts(self, rebalancer, tmp_path):
        """Rebalance should save artifacts."""
        result = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-20"),
            dry_run=False,
        )

        artifacts_dir = tmp_path / "artifacts" / "20260120"
        assert (artifacts_dir / "target_portfolio.csv").exists()
        assert (artifacts_dir / "trades.csv").exists()
        assert (artifacts_dir / "execution_log.json").exists()

    def test_strategy_selected_by_regime(self, rebalancer):
        """Strategy should be selected based on regime."""
        result = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-20"),
            dry_run=True,
        )

        assert result.strategy_used is not None
        assert len(result.strategy_used) > 0

    def test_trades_generated(self, rebalancer):
        """Trades should be generated."""
        result = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-20"),
            dry_run=True,
        )

        # Should have some trades (all buys for empty portfolio)
        assert len(result.trades) > 0
        for trade in result.trades:
            assert trade.action in ["BUY", "SELL", "HOLD"]

    def test_target_portfolio_has_positions(self, rebalancer):
        """Target portfolio should have positions."""
        result = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-20"),
            dry_run=True,
        )

        assert len(result.target_portfolio) > 0
        assert "weight" in result.target_portfolio.columns
        assert "shares" in result.target_portfolio.columns

    def test_metadata_includes_regime_indicators(self, rebalancer):
        """Metadata should include regime indicators."""
        result = rebalancer.rebalance(
            as_of=pd.Timestamp("2026-01-20"),
            dry_run=True,
        )

        assert "regime_indicators" in result.metadata
        assert "spy_price" in result.metadata["regime_indicators"]
        assert "vix" in result.metadata["regime_indicators"]

    def test_holdings_updated_after_rebalance(self, rebalancer):
        """Holdings should be updated after non-dry-run rebalance."""
        rebalancer.rebalance(as_of=pd.Timestamp("2026-01-20"), dry_run=False)

        assert rebalancer.holdings_file.exists()

        with open(rebalancer.holdings_file) as f:
            data = json.load(f)

        assert "holdings" in data
        assert "updated_at" in data


class TestRebalancerWithExistingHoldings:
    """Test rebalancer behavior with existing holdings."""

    @pytest.fixture
    def mock_data_access(self):
        ctx = MagicMock()

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
        vix_data = pd.DataFrame(
            {"VIX": np.random.uniform(12, 20, len(vix_dates))},
            index=vix_dates,
        )
        ctx.macro.read_macro_indicator.return_value = vix_data

        return ctx

    def test_generates_sells_for_removed_positions(self, mock_data_access, tmp_path):
        """Should generate SELL trades for positions not in target."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)

        # Create existing holdings with OLD_TICKER
        holdings_file = state_dir / "current_holdings.json"
        with open(holdings_file, "w") as f:
            json.dump(
                {
                    "updated_at": "2026-01-19",
                    "holdings": [
                        {
                            "ticker": "OLD_TICKER",
                            "weight": 1.0,
                            "shares": 100,
                            "price": 50,
                            "notional": 5000,
                        }
                    ],
                },
                f,
            )

        monitor = DailyRegimeMonitor(data_access=mock_data_access, state_dir=state_dir)
        rebalancer = RegimeAwareRebalancer(
            data_access=mock_data_access,
            regime_monitor=monitor,
            artifacts_dir=tmp_path / "artifacts",
        )
        rebalancer.holdings_file = holdings_file

        result = rebalancer.rebalance(as_of=pd.Timestamp("2026-01-20"), dry_run=True)

        # Should have a SELL for OLD_TICKER
        old_ticker_trade = next(
            (t for t in result.trades if t.ticker == "OLD_TICKER"), None
        )
        assert old_ticker_trade is not None
        assert old_ticker_trade.action == "SELL"
