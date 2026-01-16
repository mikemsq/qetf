"""Tests for the data quality monitoring module."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from quantetf.monitoring.alerts import AlertManager, InMemoryAlertHandler
from quantetf.monitoring.quality import (
    AnomalyResult,
    DataQualityChecker,
    GapResult,
    QualityCheckResult,
    StalenessResult,
)


class TestStalenessResult:
    """Tests for StalenessResult dataclass."""

    def test_create_result(self):
        """Test creating a staleness result."""
        result = StalenessResult(
            ticker="SPY",
            last_date=pd.Timestamp("2024-01-10"),
            days_stale=5,
            is_stale=True,
        )

        assert result.ticker == "SPY"
        assert result.days_stale == 5
        assert result.is_stale is True

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = StalenessResult(
            ticker="QQQ",
            last_date=pd.Timestamp("2024-01-10"),
            days_stale=3,
            is_stale=False,
        )

        data = result.to_dict()

        assert data["ticker"] == "QQQ"
        assert data["days_stale"] == 3


class TestGapResult:
    """Tests for GapResult dataclass."""

    def test_create_result(self):
        """Test creating a gap result."""
        result = GapResult(
            ticker="IWM",
            max_gap_days=10,
            gap_count=2,
            gap_dates=(pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-15")),
        )

        assert result.ticker == "IWM"
        assert result.max_gap_days == 10
        assert result.gap_count == 2

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = GapResult(
            ticker="IWM",
            max_gap_days=10,
            gap_count=2,
            gap_dates=(pd.Timestamp("2024-01-05"),),
        )

        data = result.to_dict()

        assert data["max_gap_days"] == 10
        assert len(data["gap_dates"]) == 1


class TestAnomalyResult:
    """Tests for AnomalyResult dataclass."""

    def test_create_result(self):
        """Test creating an anomaly result."""
        result = AnomalyResult(
            ticker="TSLA",
            anomaly_type="PRICE_SPIKE",
            anomaly_date=pd.Timestamp("2024-01-15"),
            value=0.15,
            details={"threshold": 0.10},
        )

        assert result.ticker == "TSLA"
        assert result.anomaly_type == "PRICE_SPIKE"
        assert result.value == 0.15

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = AnomalyResult(
            ticker="TSLA",
            anomaly_type="PRICE_SPIKE",
            anomaly_date=pd.Timestamp("2024-01-15"),
            value=0.15,
        )

        data = result.to_dict()

        assert data["anomaly_type"] == "PRICE_SPIKE"


class TestDataQualityChecker:
    """Tests for DataQualityChecker class."""

    @pytest.fixture
    def alert_handler(self):
        """Create in-memory alert handler."""
        return InMemoryAlertHandler()

    @pytest.fixture
    def alert_manager(self, alert_handler):
        """Create alert manager with in-memory handler."""
        return AlertManager(handlers=[alert_handler])

    @pytest.fixture
    def checker(self, alert_manager):
        """Create data quality checker."""
        return DataQualityChecker(
            alert_manager=alert_manager,
            stale_threshold_days=3,
            gap_threshold_days=5,
            spike_threshold=0.10,
        )

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        dates = pd.bdate_range("2024-01-01", periods=30)
        return pd.DataFrame(
            {
                "SPY": np.random.randn(30).cumsum() + 450,
                "QQQ": np.random.randn(30).cumsum() + 380,
                "IWM": np.random.randn(30).cumsum() + 200,
            },
            index=dates,
        )

    def test_check_staleness_no_stale_data(self, checker, sample_prices):
        """Test checking staleness when data is fresh."""
        results = checker.check_price_staleness(sample_prices)

        assert len(results) == 0

    def test_check_staleness_stale_ticker(self, checker):
        """Test detecting stale data."""
        dates = pd.bdate_range("2024-01-01", periods=30)

        # SPY has data up to day 20, others have full data
        prices = pd.DataFrame(
            {
                "SPY": [100.0] * 20 + [np.nan] * 10,
                "QQQ": [380.0] * 30,
            },
            index=dates,
        )

        results = checker.check_price_staleness(
            prices, tickers=["SPY", "QQQ"], as_of=dates[-1]
        )

        stale_tickers = [r.ticker for r in results]
        assert "SPY" in stale_tickers
        assert "QQQ" not in stale_tickers

    def test_check_staleness_missing_ticker(self, checker, sample_prices):
        """Test checking staleness for missing ticker."""
        results = checker.check_price_staleness(
            sample_prices, tickers=["MISSING"]
        )

        assert len(results) == 1
        assert results[0].ticker == "MISSING"
        assert results[0].is_stale is True

    def test_check_gaps_no_gaps(self, checker, sample_prices):
        """Test checking for gaps when data is complete."""
        results = checker.check_price_gaps(sample_prices)

        assert len(results) == 0

    def test_check_gaps_with_gap(self, checker):
        """Test detecting data gaps."""
        # Create data with a 10-day gap
        dates = list(pd.bdate_range("2024-01-01", periods=10))
        dates.extend(list(pd.bdate_range("2024-01-22", periods=10)))  # Skip ~10 days

        prices = pd.DataFrame(
            {"SPY": [100.0] * 20},
            index=pd.DatetimeIndex(dates),
        )

        results = checker.check_price_gaps(prices)

        assert len(results) == 1
        assert results[0].ticker == "SPY"
        assert results[0].max_gap_days > 5

    def test_check_price_spikes(self, checker):
        """Test detecting price spikes."""
        dates = pd.bdate_range("2024-01-01", periods=10)

        # Create data with a spike (15% move on day 5)
        prices_data = [100.0, 101.0, 102.0, 101.0, 115.0, 116.0, 115.0, 116.0, 117.0, 118.0]

        prices = pd.DataFrame({"SPY": prices_data}, index=dates)

        results = checker.check_price_spikes(prices)

        assert len(results) >= 1
        spike = results[0]
        assert spike.ticker == "SPY"
        assert spike.anomaly_type == "PRICE_SPIKE"
        assert abs(spike.value) > 0.10

    def test_check_price_spikes_no_spikes(self, checker):
        """Test when there are no price spikes."""
        dates = pd.bdate_range("2024-01-01", periods=10)

        # Create smooth data
        prices = pd.DataFrame(
            {"SPY": [100.0 + i * 0.5 for i in range(10)]},
            index=dates,
        )

        results = checker.check_price_spikes(prices)

        assert len(results) == 0

    def test_check_zero_volume(self, checker):
        """Test detecting zero volume days."""
        dates = pd.bdate_range("2024-01-01", periods=10)

        volume = pd.DataFrame(
            {"SPY": [1000000, 0, 1500000, 0, 0, 2000000, 1800000, 1900000, 2100000, 2200000]},
            index=dates,
        )

        results = checker.check_zero_volume(volume)

        zero_vol_count = len(results)
        assert zero_vol_count == 3  # Days 2, 4, 5 have zero volume

    def test_check_negative_prices(self, checker):
        """Test detecting negative prices."""
        dates = pd.bdate_range("2024-01-01", periods=5)

        prices = pd.DataFrame(
            {"SPY": [100.0, 101.0, -5.0, 102.0, 0.0]},
            index=dates,
        )

        results = checker.check_negative_prices(prices)

        assert len(results) == 2
        types = [r.anomaly_type for r in results]
        assert "NEGATIVE_PRICE" in types
        assert "ZERO_PRICE" in types

    def test_check_all(self, checker, sample_prices):
        """Test running all checks."""
        result = checker.check_all(sample_prices)

        assert isinstance(result, QualityCheckResult)
        assert result.check_timestamp is not None
        assert result.overall_status in ("OK", "WARNING", "CRITICAL")

    def test_check_all_emits_alerts(self, checker, alert_handler):
        """Test that check_all emits alerts for issues."""
        dates = pd.bdate_range("2024-01-01", periods=30)

        # Create data with staleness issue
        prices = pd.DataFrame(
            {
                "SPY": [100.0] * 20 + [np.nan] * 10,
                "QQQ": [380.0] * 30,
            },
            index=dates,
        )

        checker.check_all(prices, tickers=["SPY", "QQQ"], as_of=dates[-1])

        alerts = alert_handler.get_alerts()
        assert len(alerts) > 0
        assert any(a.category == "DATA_QUALITY" for a in alerts)

    def test_check_all_overall_status(self, checker):
        """Test that overall status is determined correctly."""
        dates = pd.bdate_range("2024-01-01", periods=10)

        # Create clean data
        prices = pd.DataFrame(
            {"SPY": [100.0 + i * 0.5 for i in range(10)]},
            index=dates,
        )

        result = checker.check_all(prices)

        assert result.overall_status == "OK"

    def test_get_quality_summary(self, checker, sample_prices):
        """Test getting quality summary."""
        summary = checker.get_quality_summary(sample_prices)

        assert summary["total_tickers_requested"] == 3
        assert summary["tickers_found"] == 3
        assert summary["date_range"]["trading_days"] == 30
        assert "completeness" in summary
        assert "thresholds" in summary

    def test_get_quality_summary_missing_tickers(self, checker, sample_prices):
        """Test quality summary with missing tickers."""
        summary = checker.get_quality_summary(
            sample_prices, tickers=["SPY", "MISSING"]
        )

        assert summary["total_tickers_requested"] == 2
        assert summary["tickers_found"] == 1
        assert "MISSING" in summary["tickers_missing"]

    def test_empty_prices_dataframe(self, checker):
        """Test handling empty DataFrame."""
        empty_prices = pd.DataFrame()

        stale = checker.check_price_staleness(empty_prices)
        gaps = checker.check_price_gaps(empty_prices)
        spikes = checker.check_price_spikes(empty_prices)

        assert stale == []
        assert gaps == []
        assert spikes == []

    def test_checker_without_alert_manager(self):
        """Test checker works without alert manager."""
        checker = DataQualityChecker()

        dates = pd.bdate_range("2024-01-01", periods=10)
        prices = pd.DataFrame({"SPY": [100.0] * 10}, index=dates)

        # Should not raise
        result = checker.check_all(prices)

        assert result.overall_status == "OK"

    def test_result_to_dict(self, checker, sample_prices):
        """Test converting result to dictionary."""
        result = checker.check_all(sample_prices)

        data = result.to_dict()

        assert "check_timestamp" in data
        assert "overall_status" in data
        assert "summary" in data
        assert "stale_count" in data["summary"]


class TestQualityCheckResult:
    """Tests for QualityCheckResult dataclass."""

    def test_create_result(self):
        """Test creating a quality check result."""
        result = QualityCheckResult(
            check_timestamp=datetime.now(timezone.utc),
            overall_status="OK",
        )

        assert result.overall_status == "OK"
        assert result.stale_tickers == []
        assert result.gap_issues == []
        assert result.anomalies == []

    def test_to_dict_with_issues(self):
        """Test converting result with issues to dictionary."""
        result = QualityCheckResult(
            check_timestamp=datetime.now(timezone.utc),
            stale_tickers=[
                StalenessResult(
                    ticker="SPY",
                    last_date=pd.Timestamp("2024-01-10"),
                    days_stale=5,
                    is_stale=True,
                )
            ],
            overall_status="WARNING",
        )

        data = result.to_dict()

        assert data["overall_status"] == "WARNING"
        assert data["summary"]["stale_count"] == 1
        assert len(data["stale_tickers"]) == 1
