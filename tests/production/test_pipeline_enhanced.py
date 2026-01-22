"""Tests for enhanced production pipeline (IMPL-012, IMPL-028).

Tests for:
- Pre-trade checks (MaxTurnoverCheck, SectorConcentrationCheck, MinTradeThresholdCheck)
- Rebalance scheduling (should_rebalance, get_next_rebalance_date)
- PipelineConfig and PipelineResult
- ProductionPipeline.run_enhanced()

IMPL-028: Updated to use mock DataAccessContext instead of MockDataStore.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantetf.production.pipeline import (
    MaxTurnoverCheck,
    MinTradeThresholdCheck,
    PipelineConfig,
    PipelineResult,
    ProductionPipeline,
    SectorConcentrationCheck,
    get_next_rebalance_date,
    should_rebalance,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@dataclass
class MockPortfolioState:
    """Mock portfolio state for testing."""

    nav: float
    peak_nav: float
    weights: pd.Series
    as_of: pd.Timestamp = pd.Timestamp("2024-01-15")


class MockStateManager:
    """Mock state manager for testing."""

    def __init__(self, state: Optional[MockPortfolioState] = None):
        self._state = state

    def get_latest_state(self) -> Optional[MockPortfolioState]:
        return self._state


class MockPriceAccessor:
    """Mock price accessor for testing."""

    def __init__(self, prices: Optional[pd.DataFrame] = None):
        self._prices = prices

    def read_prices_as_of(
        self,
        as_of: pd.Timestamp,
        tickers: list[str],
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        """Return mock OHLCV data."""
        if self._prices is not None:
            return self._prices
        start = as_of - pd.tseries.offsets.BDay(lookback_days)
        dates = pd.date_range(start=start, end=as_of, freq="B")

        # Create multi-level columns (ticker, price_type)
        columns = pd.MultiIndex.from_product(
            [tickers, ['Open', 'High', 'Low', 'Close', 'Volume']],
            names=['Ticker', 'Price']
        )
        data = np.random.randn(len(dates), len(columns)) * 10 + 100
        df = pd.DataFrame(data, index=dates, columns=columns)
        return df


class MockMacroAccessor:
    """Mock macro accessor for testing."""

    def __init__(self, vix_value: float = 20.0):
        self._vix = vix_value

    def read_macro_indicator(self, indicator: str, as_of: pd.Timestamp) -> float:
        """Return mock macro indicator value."""
        if indicator == "VIX":
            return self._vix
        return 0.0

    def get_regime(self, as_of: pd.Timestamp) -> str:
        """Return mock regime."""
        return "NORMAL"


class MockUniverseAccessor:
    """Mock universe accessor for testing."""

    def get_universe(self, name: str) -> list[str]:
        """Return mock universe tickers."""
        return ["SPY", "QQQ", "AAPL", "MSFT", "AGG"]


class MockReferenceAccessor:
    """Mock reference accessor for testing."""

    def get_ticker_info(self, ticker: str) -> dict:
        """Return mock ticker info."""
        return {"ticker": ticker, "sector": "Unknown"}


@dataclass
class MockDataAccessContext:
    """Mock DataAccessContext for testing (IMPL-028)."""

    prices: MockPriceAccessor
    macro: MockMacroAccessor
    universes: MockUniverseAccessor
    references: MockReferenceAccessor

    @classmethod
    def create_default(cls, prices: Optional[pd.DataFrame] = None, vix: float = 20.0):
        """Create a default mock context for testing."""
        return cls(
            prices=MockPriceAccessor(prices),
            macro=MockMacroAccessor(vix),
            universes=MockUniverseAccessor(),
            references=MockReferenceAccessor(),
        )


@pytest.fixture
def sample_trades() -> pd.DataFrame:
    """Sample trades DataFrame."""
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL", "SPY"],
        "current_weight": [0.20, 0.30, 0.25, 0.25],
        "target_weight": [0.30, 0.20, 0.25, 0.25],
        "delta_weight": [0.10, -0.10, 0.00, 0.00],
    })


@pytest.fixture
def high_turnover_trades() -> pd.DataFrame:
    """Trades with high turnover."""
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL", "SPY"],
        "current_weight": [0.50, 0.50, 0.00, 0.00],
        "target_weight": [0.00, 0.00, 0.50, 0.50],
        "delta_weight": [-0.50, -0.50, 0.50, 0.50],
    })


# -----------------------------------------------------------------------------
# MaxTurnoverCheck Tests
# -----------------------------------------------------------------------------


class TestMaxTurnoverCheck:
    """Tests for MaxTurnoverCheck."""

    def test_turnover_within_limit(self, sample_trades):
        """Turnover within limit should pass."""
        check = MaxTurnoverCheck(max_turnover=0.50)
        passed, reason = check.check(sample_trades, None, pd.Timestamp("2024-01-15"))

        assert passed is True
        assert "within limit" in reason

    def test_turnover_exceeds_limit(self, high_turnover_trades):
        """Turnover exceeding limit should fail."""
        check = MaxTurnoverCheck(max_turnover=0.50)
        passed, reason = check.check(high_turnover_trades, None, pd.Timestamp("2024-01-15"))

        # Turnover = 0.5 * (0.5 + 0.5 + 0.5 + 0.5) = 1.0, exceeds 0.50
        assert passed is False
        assert "exceeds max" in reason

    def test_empty_trades_passes(self):
        """Empty trades should pass."""
        check = MaxTurnoverCheck(max_turnover=0.50)
        empty_trades = pd.DataFrame(columns=["ticker", "current_weight", "target_weight", "delta_weight"])

        passed, reason = check.check(empty_trades, None, pd.Timestamp("2024-01-15"))

        assert passed is True
        assert "No trades" in reason

    def test_custom_max_turnover(self, sample_trades):
        """Custom max_turnover should be respected."""
        # Turnover = 0.5 * (0.10 + 0.10) = 0.10
        check = MaxTurnoverCheck(max_turnover=0.05)
        passed, reason = check.check(sample_trades, None, pd.Timestamp("2024-01-15"))

        assert passed is False
        assert "10.00%" in reason

    def test_is_frozen_dataclass(self):
        """MaxTurnoverCheck should be immutable."""
        check = MaxTurnoverCheck(max_turnover=0.50)
        with pytest.raises(AttributeError):
            check.max_turnover = 0.60


# -----------------------------------------------------------------------------
# SectorConcentrationCheck Tests
# -----------------------------------------------------------------------------


class TestSectorConcentrationCheck:
    """Tests for SectorConcentrationCheck."""

    def test_no_sector_map_skips(self, sample_trades):
        """Without sector map, check should skip."""
        check = SectorConcentrationCheck(max_sector_weight=0.40)
        passed, reason = check.check(sample_trades, None, pd.Timestamp("2024-01-15"))

        assert passed is True
        assert "skipped" in reason

    def test_sector_within_limit(self):
        """Sector concentrations within limit should pass."""
        sector_map = (
            ("AAPL", "Technology"),
            ("MSFT", "Technology"),
            ("XOM", "Energy"),
        )
        check = SectorConcentrationCheck(max_sector_weight=0.50, sector_map=sector_map)

        trades = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "XOM"],
            "current_weight": [0.0, 0.0, 0.0],
            "target_weight": [0.20, 0.20, 0.30],
            "delta_weight": [0.20, 0.20, 0.30],
        })

        passed, reason = check.check(trades, None, pd.Timestamp("2024-01-15"))

        assert passed is True
        assert "within limits" in reason

    def test_sector_exceeds_limit(self):
        """Sector concentration exceeding limit should fail."""
        sector_map = (
            ("AAPL", "Technology"),
            ("MSFT", "Technology"),
            ("GOOGL", "Technology"),
        )
        check = SectorConcentrationCheck(max_sector_weight=0.40, sector_map=sector_map)

        trades = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "current_weight": [0.0, 0.0, 0.0],
            "target_weight": [0.20, 0.20, 0.20],
            "delta_weight": [0.20, 0.20, 0.20],
        })

        passed, reason = check.check(trades, None, pd.Timestamp("2024-01-15"))

        # Technology = 0.60, exceeds 0.40
        assert passed is False
        assert "Technology" in reason
        assert "60.00%" in reason

    def test_empty_trades_passes(self):
        """Empty trades should pass."""
        check = SectorConcentrationCheck(max_sector_weight=0.40)
        empty_trades = pd.DataFrame(columns=["ticker", "current_weight", "target_weight", "delta_weight"])

        passed, reason = check.check(empty_trades, None, pd.Timestamp("2024-01-15"))

        assert passed is True


# -----------------------------------------------------------------------------
# MinTradeThresholdCheck Tests
# -----------------------------------------------------------------------------


class TestMinTradeThresholdCheck:
    """Tests for MinTradeThresholdCheck."""

    def test_always_passes(self, sample_trades):
        """MinTradeThresholdCheck should always pass."""
        check = MinTradeThresholdCheck(min_trade_weight=0.05)
        passed, reason = check.check(sample_trades, None, pd.Timestamp("2024-01-15"))

        assert passed is True

    def test_reports_below_threshold_count(self):
        """Should report count of trades below threshold."""
        check = MinTradeThresholdCheck(min_trade_weight=0.05)

        trades = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "current_weight": [0.0, 0.0, 0.0],
            "target_weight": [0.01, 0.02, 0.10],
            "delta_weight": [0.01, 0.02, 0.10],
        })

        passed, reason = check.check(trades, None, pd.Timestamp("2024-01-15"))

        assert passed is True
        assert "2 trades below" in reason

    def test_empty_trades(self):
        """Empty trades should pass with appropriate message."""
        check = MinTradeThresholdCheck(min_trade_weight=0.05)
        empty_trades = pd.DataFrame(columns=["ticker", "current_weight", "target_weight", "delta_weight"])

        passed, reason = check.check(empty_trades, None, pd.Timestamp("2024-01-15"))

        assert passed is True
        assert "No trades" in reason


# -----------------------------------------------------------------------------
# Rebalance Scheduling Tests
# -----------------------------------------------------------------------------


class TestShouldRebalance:
    """Tests for should_rebalance function."""

    def test_daily_always_true(self):
        """Daily schedule should always return True."""
        assert should_rebalance(pd.Timestamp("2024-01-15"), "daily") is True
        assert should_rebalance(pd.Timestamp("2024-01-16"), "daily") is True
        assert should_rebalance(pd.Timestamp("2024-01-17"), "daily") is True

    def test_weekly_only_friday(self):
        """Weekly schedule should only return True on Friday."""
        # 2024-01-15 is Monday
        assert should_rebalance(pd.Timestamp("2024-01-15"), "weekly") is False
        # 2024-01-16 is Tuesday
        assert should_rebalance(pd.Timestamp("2024-01-16"), "weekly") is False
        # 2024-01-19 is Friday
        assert should_rebalance(pd.Timestamp("2024-01-19"), "weekly") is True

    def test_monthly_last_business_day(self):
        """Monthly schedule should return True on last business day of month."""
        # 2024-01-31 is last business day of January
        assert should_rebalance(pd.Timestamp("2024-01-31"), "monthly") is True
        # 2024-01-30 is not last business day
        assert should_rebalance(pd.Timestamp("2024-01-30"), "monthly") is False
        # 2024-02-29 is last business day of February 2024 (leap year)
        assert should_rebalance(pd.Timestamp("2024-02-29"), "monthly") is True

    def test_invalid_schedule_raises(self):
        """Invalid schedule should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown schedule"):
            should_rebalance(pd.Timestamp("2024-01-15"), "quarterly")


class TestGetNextRebalanceDate:
    """Tests for get_next_rebalance_date function."""

    def test_daily_next_business_day(self):
        """Daily schedule should return next business day."""
        # 2024-01-15 is Monday
        result = get_next_rebalance_date(pd.Timestamp("2024-01-15"), "daily")
        assert result == pd.Timestamp("2024-01-16")

    def test_weekly_next_friday(self):
        """Weekly schedule should return next Friday."""
        # 2024-01-15 is Monday
        result = get_next_rebalance_date(pd.Timestamp("2024-01-15"), "weekly")
        assert result == pd.Timestamp("2024-01-19")  # Friday

    def test_monthly_next_month_end(self):
        """Monthly schedule should return next month-end business day."""
        # 2024-01-15 -> should get 2024-01-31
        result = get_next_rebalance_date(pd.Timestamp("2024-01-15"), "monthly")
        assert result == pd.Timestamp("2024-01-31")

    def test_handles_weekend(self):
        """Should handle weekend start dates."""
        # 2024-01-13 is Saturday
        result = get_next_rebalance_date(pd.Timestamp("2024-01-13"), "weekly")
        assert result.weekday() == 4  # Friday


# -----------------------------------------------------------------------------
# PipelineConfig Tests
# -----------------------------------------------------------------------------


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_values(self):
        """PipelineConfig should have sensible defaults."""
        config = PipelineConfig()

        assert config.strategy_config_path is None
        assert config.risk_overlays == []
        assert config.pre_trade_checks == []
        assert config.state_manager is None
        assert config.rebalance_schedule == "monthly"
        assert config.trade_threshold == 0.005
        assert config.dry_run is False

    def test_custom_values(self):
        """PipelineConfig should accept custom values."""
        check = MaxTurnoverCheck(max_turnover=0.30)
        config = PipelineConfig(
            rebalance_schedule="weekly",
            trade_threshold=0.01,
            pre_trade_checks=[check],
            dry_run=True,
        )

        assert config.rebalance_schedule == "weekly"
        assert config.trade_threshold == 0.01
        assert len(config.pre_trade_checks) == 1
        assert config.dry_run is True


# -----------------------------------------------------------------------------
# PipelineResult Tests
# -----------------------------------------------------------------------------


class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_to_dict(self):
        """PipelineResult.to_dict() should serialize correctly."""
        result = PipelineResult(
            as_of=pd.Timestamp("2024-01-31"),
            target_weights=pd.Series({"SPY": 0.6, "QQQ": 0.4}),
            adjusted_weights=pd.Series({"SPY": 0.55, "QQQ": 0.35}),
            trades=pd.DataFrame({
                "ticker": ["SPY", "QQQ"],
                "current_weight": [0.5, 0.5],
                "target_weight": [0.55, 0.35],
                "delta_weight": [0.05, -0.15],
            }),
            pre_trade_checks_passed=True,
            check_results=(("MaxTurnoverCheck", True, "OK"),),
            overlay_diagnostics=(("PositionLimitOverlay", (("capped", 0),)),),
            execution_status="pending",
        )

        d = result.to_dict()

        assert d["as_of"] == "2024-01-31 00:00:00"
        assert d["execution_status"] == "pending"
        assert d["pre_trade_checks_passed"] is True
        assert "MaxTurnoverCheck" in d["check_results"]
        assert "PositionLimitOverlay" in d["overlay_diagnostics"]

    def test_get_check_results_dict(self):
        """get_check_results_dict should return proper format."""
        result = PipelineResult(
            as_of=pd.Timestamp("2024-01-31"),
            target_weights=pd.Series({"SPY": 1.0}),
            adjusted_weights=pd.Series({"SPY": 1.0}),
            trades=pd.DataFrame(),
            pre_trade_checks_passed=True,
            check_results=(
                ("MaxTurnoverCheck", True, "OK"),
                ("SectorConcentrationCheck", False, "Failed"),
            ),
            overlay_diagnostics=(),
            execution_status="blocked",
        )

        check_dict = result.get_check_results_dict()

        assert check_dict["MaxTurnoverCheck"] == (True, "OK")
        assert check_dict["SectorConcentrationCheck"] == (False, "Failed")

    def test_frozen(self):
        """PipelineResult should be immutable."""
        result = PipelineResult(
            as_of=pd.Timestamp("2024-01-31"),
            target_weights=pd.Series({"SPY": 1.0}),
            adjusted_weights=pd.Series({"SPY": 1.0}),
            trades=pd.DataFrame(),
            pre_trade_checks_passed=True,
            check_results=(),
            overlay_diagnostics=(),
            execution_status="pending",
        )

        with pytest.raises(AttributeError):
            result.execution_status = "executed"


# -----------------------------------------------------------------------------
# ProductionPipeline.run_enhanced Tests
# -----------------------------------------------------------------------------


class TestProductionPipelineRunEnhanced:
    """Tests for ProductionPipeline.run_enhanced()."""

    def test_basic_run(self):
        """Basic run without overlays or checks."""
        config = PipelineConfig(rebalance_schedule="daily")
        pipeline = ProductionPipeline(config=config)

        target_weights = pd.Series({"SPY": 0.6, "QQQ": 0.4})
        current_weights = pd.Series({"SPY": 0.5, "QQQ": 0.5})

        result = pipeline.run_enhanced(
            as_of=pd.Timestamp("2024-01-15"),
            target_weights=target_weights,
            current_weights=current_weights,
            force_rebalance=True,
        )

        assert result.execution_status == "pending"
        assert result.pre_trade_checks_passed is True
        assert len(result.trades) > 0

    def test_skips_non_rebalance_date(self):
        """Should skip on non-rebalance dates."""
        config = PipelineConfig(rebalance_schedule="monthly")
        pipeline = ProductionPipeline(config=config)

        target_weights = pd.Series({"SPY": 0.6, "QQQ": 0.4})

        # 2024-01-15 is not end of month
        result = pipeline.run_enhanced(
            as_of=pd.Timestamp("2024-01-15"),
            target_weights=target_weights,
        )

        assert result.execution_status == "skipped"
        assert result.trades.empty

    def test_force_rebalance_ignores_schedule(self):
        """force_rebalance should ignore schedule."""
        config = PipelineConfig(rebalance_schedule="monthly")
        pipeline = ProductionPipeline(config=config)

        target_weights = pd.Series({"SPY": 0.6, "QQQ": 0.4})
        current_weights = pd.Series(dtype=float)

        result = pipeline.run_enhanced(
            as_of=pd.Timestamp("2024-01-15"),
            target_weights=target_weights,
            current_weights=current_weights,
            force_rebalance=True,
        )

        assert result.execution_status != "skipped"

    def test_pre_trade_check_blocks(self):
        """Failed pre-trade check should block execution."""
        config = PipelineConfig(
            rebalance_schedule="daily",
            pre_trade_checks=[MaxTurnoverCheck(max_turnover=0.01)],  # Very low threshold
        )
        pipeline = ProductionPipeline(config=config)

        target_weights = pd.Series({"SPY": 0.6, "QQQ": 0.4})
        current_weights = pd.Series({"SPY": 0.0, "QQQ": 0.0, "AAPL": 1.0})

        result = pipeline.run_enhanced(
            as_of=pd.Timestamp("2024-01-15"),
            target_weights=target_weights,
            current_weights=current_weights,
            force_rebalance=True,
        )

        assert result.execution_status == "blocked"
        assert result.pre_trade_checks_passed is False

    def test_loads_state_from_manager(self):
        """Should load current weights from state manager."""
        state = MockPortfolioState(
            nav=100000,
            peak_nav=100000,
            weights=pd.Series({"SPY": 0.5, "QQQ": 0.5}),
        )
        state_manager = MockStateManager(state=state)

        config = PipelineConfig(
            rebalance_schedule="daily",
            state_manager=state_manager,
        )
        pipeline = ProductionPipeline(config=config)

        target_weights = pd.Series({"SPY": 0.6, "QQQ": 0.4})

        result = pipeline.run_enhanced(
            as_of=pd.Timestamp("2024-01-15"),
            target_weights=target_weights,
            force_rebalance=True,
        )

        # Should have generated trades from state weights
        assert len(result.trades) > 0
        assert result.execution_status == "pending"

    def test_empty_current_weights_no_state(self):
        """Should handle empty portfolio when no state."""
        config = PipelineConfig(rebalance_schedule="daily")
        pipeline = ProductionPipeline(config=config)

        target_weights = pd.Series({"SPY": 0.6, "QQQ": 0.4})

        result = pipeline.run_enhanced(
            as_of=pd.Timestamp("2024-01-15"),
            target_weights=target_weights,
            force_rebalance=True,
        )

        # Should buy full positions from empty
        assert len(result.trades) == 2
        assert result.execution_status == "pending"

    def test_multiple_pre_trade_checks(self):
        """Should run all pre-trade checks and collect results."""
        config = PipelineConfig(
            rebalance_schedule="daily",
            pre_trade_checks=[
                MaxTurnoverCheck(max_turnover=0.50),
                MinTradeThresholdCheck(min_trade_weight=0.01),
            ],
        )
        pipeline = ProductionPipeline(config=config)

        target_weights = pd.Series({"SPY": 0.6, "QQQ": 0.4})
        current_weights = pd.Series({"SPY": 0.5, "QQQ": 0.5})

        result = pipeline.run_enhanced(
            as_of=pd.Timestamp("2024-01-15"),
            target_weights=target_weights,
            current_weights=current_weights,
            force_rebalance=True,
        )

        check_dict = result.get_check_results_dict()
        assert "MaxTurnoverCheck" in check_dict
        assert "MinTradeThresholdCheck" in check_dict

    def test_trade_threshold_filters_small_trades(self):
        """trade_threshold should filter small trades."""
        config = PipelineConfig(
            rebalance_schedule="daily",
            trade_threshold=0.10,  # 10% threshold
        )
        pipeline = ProductionPipeline(config=config)

        target_weights = pd.Series({"SPY": 0.55, "QQQ": 0.45})
        current_weights = pd.Series({"SPY": 0.50, "QQQ": 0.50})

        result = pipeline.run_enhanced(
            as_of=pd.Timestamp("2024-01-15"),
            target_weights=target_weights,
            current_weights=current_weights,
            force_rebalance=True,
        )

        # 5% trades should be filtered out
        assert result.trades.empty or all(
            abs(row["delta_weight"]) >= 0.10
            for _, row in result.trades.iterrows()
        )


class TestProductionPipelineBackwardCompatibility:
    """Tests for backward compatibility with legacy run() method."""

    def test_legacy_run_method(self):
        """Legacy run() method should still work."""
        from datetime import datetime

        from quantetf.types import DatasetVersion

        pipeline = ProductionPipeline()

        current_weights = pd.Series({"SPY": 0.5, "QQQ": 0.5})
        target_weights = pd.Series({"SPY": 0.6, "QQQ": 0.4})
        dataset_version = DatasetVersion(id="test_v1", created_at=datetime.now())

        packet = pipeline.run(
            as_of=pd.Timestamp("2024-01-15"),
            dataset_version=dataset_version,
            current_weights=current_weights,
            target_weights=target_weights,
        )

        assert packet.as_of == pd.Timestamp("2024-01-15")
        assert len(packet.trades) > 0
        assert packet.summary["dataset_id"] == "test_v1"
