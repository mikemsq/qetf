"""Data quality monitoring.

This module provides data quality checks for price data including staleness
detection, gap detection, and anomaly identification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from quantetf.monitoring.alerts import Alert, AlertManager

if TYPE_CHECKING:
    from quantetf.data.access import DataAccessContext
    from quantetf.data.store import DataStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StalenessResult:
    """Result of staleness check for a single ticker.

    Attributes:
        ticker: Ticker symbol.
        last_date: Last date with data.
        days_stale: Number of business days since last data.
        is_stale: Whether the data exceeds staleness threshold.
    """

    ticker: str
    last_date: pd.Timestamp | None
    days_stale: int
    is_stale: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "last_date": self.last_date.isoformat() if self.last_date else None,
            "days_stale": self.days_stale,
            "is_stale": self.is_stale,
        }


@dataclass(frozen=True)
class GapResult:
    """Result of gap check for a single ticker.

    Attributes:
        ticker: Ticker symbol.
        max_gap_days: Maximum gap in trading days.
        gap_count: Number of gaps exceeding threshold.
        gap_dates: List of dates where gaps start.
    """

    ticker: str
    max_gap_days: int
    gap_count: int
    gap_dates: tuple[pd.Timestamp, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "max_gap_days": self.max_gap_days,
            "gap_count": self.gap_count,
            "gap_dates": [d.isoformat() for d in self.gap_dates],
        }


@dataclass(frozen=True)
class AnomalyResult:
    """Result of anomaly check for a single ticker.

    Attributes:
        ticker: Ticker symbol.
        anomaly_type: Type of anomaly detected.
        anomaly_date: Date of the anomaly.
        value: Anomalous value.
        details: Additional details about the anomaly.
    """

    ticker: str
    anomaly_type: str
    anomaly_date: pd.Timestamp
    value: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "anomaly_type": self.anomaly_type,
            "anomaly_date": self.anomaly_date.isoformat(),
            "value": self.value,
            "details": self.details,
        }


@dataclass
class QualityCheckResult:
    """Comprehensive result of data quality checks.

    Attributes:
        check_timestamp: When the check was performed.
        stale_tickers: Tickers with stale data.
        gap_issues: Tickers with data gaps.
        anomalies: Detected anomalies.
        alerts_emitted: Alerts that were emitted.
        overall_status: Summary status (OK, WARNING, CRITICAL).
    """

    check_timestamp: datetime
    stale_tickers: list[StalenessResult] = field(default_factory=list)
    gap_issues: list[GapResult] = field(default_factory=list)
    anomalies: list[AnomalyResult] = field(default_factory=list)
    alerts_emitted: list[Alert] = field(default_factory=list)
    overall_status: str = "OK"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_timestamp": self.check_timestamp.isoformat(),
            "stale_tickers": [s.to_dict() for s in self.stale_tickers],
            "gap_issues": [g.to_dict() for g in self.gap_issues],
            "anomalies": [a.to_dict() for a in self.anomalies],
            "alerts_emitted": [a.to_dict() for a in self.alerts_emitted],
            "overall_status": self.overall_status,
            "summary": {
                "stale_count": len(self.stale_tickers),
                "gap_count": len(self.gap_issues),
                "anomaly_count": len(self.anomalies),
            },
        }


class DataQualityChecker:
    """Check data quality and emit alerts for issues.

    The DataQualityChecker performs various quality checks on price data:
    - Staleness: Identifies tickers without recent data
    - Gaps: Identifies tickers with missing data periods
    - Anomalies: Identifies unusual price movements or values

    Example using DataAccessContext (recommended):
        >>> from quantetf.data.access import DataAccessFactory
        >>> from quantetf.monitoring import AlertManager, DataQualityChecker
        >>> ctx = DataAccessFactory.create_context(config={"snapshot_path": "..."})
        >>> alert_manager = AlertManager()
        >>> checker = DataQualityChecker(data_access=ctx, alert_manager=alert_manager)
        >>> result = checker.check_all(tickers=["SPY", "QQQ", "IWM"], as_of=pd.Timestamp("2024-01-31"))

    Example with direct DataFrame (legacy):
        >>> checker = DataQualityChecker(alert_manager=alert_manager)
        >>> result = checker.check_all(prices=prices_df, tickers=["SPY", "QQQ", "IWM"])
    """

    def __init__(
        self,
        data_access: "DataAccessContext | None" = None,
        alert_manager: AlertManager | None = None,
        stale_threshold_days: int = 3,
        gap_threshold_days: int = 5,
        spike_threshold: float = 0.10,
    ) -> None:
        """Initialize data quality checker.

        Args:
            data_access: Optional DataAccessContext for fetching price data.
                        If provided, check_all() can be called without a prices DataFrame.
            alert_manager: Optional AlertManager for emitting notifications.
            stale_threshold_days: Days without data to consider stale.
            gap_threshold_days: Gap size in trading days to flag.
            spike_threshold: Price change threshold for spike detection (e.g., 0.10 = 10%).
        """
        self._data_access = data_access
        self._alert_manager = alert_manager
        self._stale_threshold_days = stale_threshold_days
        self._gap_threshold_days = gap_threshold_days
        self._spike_threshold = spike_threshold

    def check_price_staleness(
        self,
        prices: pd.DataFrame,
        tickers: list[str] | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> list[StalenessResult]:
        """Check for stale price data.

        Args:
            prices: DataFrame with price data (index=dates, columns=tickers).
            tickers: Optional list of tickers to check. Checks all if None.
            as_of: Reference date for staleness. Uses last date in data if None.

        Returns:
            List of StalenessResult for stale tickers.
        """
        if prices.empty:
            return []

        tickers_to_check = tickers if tickers else list(prices.columns)
        as_of_date = as_of if as_of else prices.index[-1]

        stale_results: list[StalenessResult] = []

        for ticker in tickers_to_check:
            if ticker not in prices.columns:
                stale_results.append(
                    StalenessResult(
                        ticker=ticker,
                        last_date=None,
                        days_stale=999,
                        is_stale=True,
                    )
                )
                continue

            ticker_data = prices[ticker].dropna()
            if ticker_data.empty:
                stale_results.append(
                    StalenessResult(
                        ticker=ticker,
                        last_date=None,
                        days_stale=999,
                        is_stale=True,
                    )
                )
                continue

            last_date = ticker_data.index[-1]

            # Calculate business days between last date and as_of
            business_days = pd.bdate_range(start=last_date, end=as_of_date)
            days_stale = len(business_days) - 1  # Don't count the start date

            is_stale = days_stale > self._stale_threshold_days

            if is_stale:
                stale_results.append(
                    StalenessResult(
                        ticker=ticker,
                        last_date=last_date,
                        days_stale=days_stale,
                        is_stale=True,
                    )
                )

        return stale_results

    def check_price_gaps(
        self,
        prices: pd.DataFrame,
        tickers: list[str] | None = None,
    ) -> list[GapResult]:
        """Check for gaps in price data.

        Args:
            prices: DataFrame with price data.
            tickers: Optional list of tickers to check.

        Returns:
            List of GapResult for tickers with gaps.
        """
        if prices.empty:
            return []

        tickers_to_check = tickers if tickers else list(prices.columns)
        gap_results: list[GapResult] = []

        for ticker in tickers_to_check:
            if ticker not in prices.columns:
                continue

            ticker_data = prices[ticker].dropna()
            if len(ticker_data) < 2:
                continue

            # Find gaps in the data
            dates = ticker_data.index
            gaps: list[tuple[pd.Timestamp, int]] = []

            for i in range(1, len(dates)):
                business_days = pd.bdate_range(start=dates[i - 1], end=dates[i])
                gap_days = len(business_days) - 1

                if gap_days > self._gap_threshold_days:
                    gaps.append((dates[i - 1], gap_days))

            if gaps:
                max_gap = max(g[1] for g in gaps)
                gap_dates = tuple(g[0] for g in gaps)

                gap_results.append(
                    GapResult(
                        ticker=ticker,
                        max_gap_days=max_gap,
                        gap_count=len(gaps),
                        gap_dates=gap_dates,
                    )
                )

        return gap_results

    def check_price_spikes(
        self,
        prices: pd.DataFrame,
        tickers: list[str] | None = None,
        lookback_days: int = 252,
    ) -> list[AnomalyResult]:
        """Check for suspicious price spikes.

        Args:
            prices: DataFrame with price data.
            tickers: Optional list of tickers to check.
            lookback_days: Number of days to check for spikes.

        Returns:
            List of AnomalyResult for detected spikes.
        """
        if prices.empty:
            return []

        tickers_to_check = tickers if tickers else list(prices.columns)
        anomalies: list[AnomalyResult] = []

        for ticker in tickers_to_check:
            if ticker not in prices.columns:
                continue

            ticker_prices = prices[ticker].dropna().iloc[-lookback_days:]
            if len(ticker_prices) < 2:
                continue

            returns = ticker_prices.pct_change().dropna()

            # Find large moves
            large_moves = returns[returns.abs() > self._spike_threshold]

            for date, ret in large_moves.items():
                anomalies.append(
                    AnomalyResult(
                        ticker=ticker,
                        anomaly_type="PRICE_SPIKE",
                        anomaly_date=date,
                        value=ret,
                        details={
                            "return_pct": f"{ret:.1%}",
                            "threshold": f"{self._spike_threshold:.1%}",
                            "price_before": float(ticker_prices.loc[:date].iloc[-2])
                            if len(ticker_prices.loc[:date]) > 1
                            else None,
                            "price_after": float(ticker_prices.loc[date]),
                        },
                    )
                )

        return anomalies

    def check_zero_volume(
        self,
        volume: pd.DataFrame,
        tickers: list[str] | None = None,
        lookback_days: int = 20,
    ) -> list[AnomalyResult]:
        """Check for zero volume days.

        Args:
            volume: DataFrame with volume data.
            tickers: Optional list of tickers to check.
            lookback_days: Number of recent days to check.

        Returns:
            List of AnomalyResult for zero volume days.
        """
        if volume.empty:
            return []

        tickers_to_check = tickers if tickers else list(volume.columns)
        anomalies: list[AnomalyResult] = []

        for ticker in tickers_to_check:
            if ticker not in volume.columns:
                continue

            recent_volume = volume[ticker].iloc[-lookback_days:]
            zero_volume_days = recent_volume[recent_volume == 0]

            for date, vol in zero_volume_days.items():
                anomalies.append(
                    AnomalyResult(
                        ticker=ticker,
                        anomaly_type="ZERO_VOLUME",
                        anomaly_date=date,
                        value=0.0,
                        details={"lookback_days": lookback_days},
                    )
                )

        return anomalies

    def check_negative_prices(
        self,
        prices: pd.DataFrame,
        tickers: list[str] | None = None,
    ) -> list[AnomalyResult]:
        """Check for negative or zero prices.

        Args:
            prices: DataFrame with price data.
            tickers: Optional list of tickers to check.

        Returns:
            List of AnomalyResult for invalid prices.
        """
        if prices.empty:
            return []

        tickers_to_check = tickers if tickers else list(prices.columns)
        anomalies: list[AnomalyResult] = []

        for ticker in tickers_to_check:
            if ticker not in prices.columns:
                continue

            ticker_prices = prices[ticker].dropna()
            invalid = ticker_prices[ticker_prices <= 0]

            for date, price in invalid.items():
                anomalies.append(
                    AnomalyResult(
                        ticker=ticker,
                        anomaly_type="NEGATIVE_PRICE" if price < 0 else "ZERO_PRICE",
                        anomaly_date=date,
                        value=price,
                        details={},
                    )
                )

        return anomalies

    def check_all(
        self,
        prices: pd.DataFrame | None = None,
        volume: pd.DataFrame | None = None,
        tickers: list[str] | None = None,
        as_of: pd.Timestamp | None = None,
        lookback_days: int = 252,
    ) -> QualityCheckResult:
        """Run all quality checks and emit alerts.

        Args:
            prices: DataFrame with price data. If None and data_access is configured,
                   prices will be fetched using the accessor.
            volume: Optional DataFrame with volume data.
            tickers: Optional list of tickers to check.
            as_of: Reference date for checks. Required if prices is None.
            lookback_days: Number of days to look back when fetching from accessor.

        Returns:
            QualityCheckResult with all findings.

        Raises:
            ValueError: If prices is None and no data_access is configured,
                       or if prices is None and as_of is not provided.
        """
        # Fetch prices from data_access if not provided directly
        if prices is None:
            if self._data_access is None:
                raise ValueError(
                    "prices DataFrame required when DataAccessContext not configured"
                )
            if as_of is None:
                raise ValueError(
                    "as_of is required when fetching prices from DataAccessContext"
                )

            # Fetch prices using the accessor
            ohlcv = self._data_access.prices.read_prices_as_of(
                as_of=as_of,
                tickers=tickers,
                lookback_days=lookback_days,
            )
            # Extract Close prices - ohlcv has MultiIndex columns (Ticker, Field)
            if ohlcv.empty:
                prices = pd.DataFrame()
            else:
                # Get Close prices for each ticker
                prices = ohlcv.xs("Close", axis=1, level="Field")
        check_timestamp = datetime.now(timezone.utc)
        alerts_emitted: list[Alert] = []

        # Run all checks
        stale_results = self.check_price_staleness(prices, tickers, as_of)
        gap_results = self.check_price_gaps(prices, tickers)
        anomalies = self.check_price_spikes(prices, tickers)
        anomalies.extend(self.check_negative_prices(prices, tickers))

        if volume is not None:
            anomalies.extend(self.check_zero_volume(volume, tickers))

        # Determine overall status
        overall_status = "OK"
        if anomalies or gap_results:
            overall_status = "WARNING"
        if stale_results:
            overall_status = "CRITICAL" if len(stale_results) > 5 else "WARNING"

        # Emit alerts if alert manager is configured
        if self._alert_manager:
            if stale_results:
                stale_tickers = [s.ticker for s in stale_results]
                alert = Alert(
                    timestamp=check_timestamp,
                    level="WARNING" if len(stale_results) <= 5 else "CRITICAL",
                    category="DATA_QUALITY",
                    message=f"Found {len(stale_results)} tickers with stale data: "
                    f"{', '.join(stale_tickers[:5])}{'...' if len(stale_tickers) > 5 else ''}",
                    data={
                        "stale_count": len(stale_results),
                        "stale_tickers": stale_tickers[:20],
                        "threshold_days": self._stale_threshold_days,
                    },
                )
                self._alert_manager.emit(alert)
                alerts_emitted.append(alert)

            if gap_results:
                gap_tickers = [g.ticker for g in gap_results]
                alert = Alert(
                    timestamp=check_timestamp,
                    level="WARNING",
                    category="DATA_QUALITY",
                    message=f"Found {len(gap_results)} tickers with data gaps: "
                    f"{', '.join(gap_tickers[:5])}{'...' if len(gap_tickers) > 5 else ''}",
                    data={
                        "gap_count": len(gap_results),
                        "gap_tickers": gap_tickers[:20],
                        "threshold_days": self._gap_threshold_days,
                    },
                )
                self._alert_manager.emit(alert)
                alerts_emitted.append(alert)

            if anomalies:
                anomaly_count_by_type: dict[str, int] = {}
                for a in anomalies:
                    anomaly_count_by_type[a.anomaly_type] = (
                        anomaly_count_by_type.get(a.anomaly_type, 0) + 1
                    )

                alert = Alert(
                    timestamp=check_timestamp,
                    level="WARNING",
                    category="DATA_QUALITY",
                    message=f"Found {len(anomalies)} data anomalies: "
                    f"{', '.join(f'{k}={v}' for k, v in anomaly_count_by_type.items())}",
                    data={
                        "anomaly_count": len(anomalies),
                        "anomaly_types": anomaly_count_by_type,
                    },
                )
                self._alert_manager.emit(alert)
                alerts_emitted.append(alert)

        return QualityCheckResult(
            check_timestamp=check_timestamp,
            stale_tickers=stale_results,
            gap_issues=gap_results,
            anomalies=anomalies,
            alerts_emitted=alerts_emitted,
            overall_status=overall_status,
        )

    def get_quality_summary(
        self,
        prices: pd.DataFrame,
        tickers: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get a summary of data quality metrics.

        Args:
            prices: DataFrame with price data.
            tickers: Optional list of tickers to check.

        Returns:
            Dictionary with quality summary metrics.
        """
        tickers_to_check = tickers if tickers else list(prices.columns)

        # Calculate basic statistics
        total_tickers = len(tickers_to_check)
        available_tickers = [t for t in tickers_to_check if t in prices.columns]
        missing_tickers = [t for t in tickers_to_check if t not in prices.columns]

        # NaN statistics
        nan_counts = prices[available_tickers].isna().sum()
        total_values = len(prices) * len(available_tickers)
        nan_total = nan_counts.sum()

        return {
            "total_tickers_requested": total_tickers,
            "tickers_found": len(available_tickers),
            "tickers_missing": missing_tickers,
            "date_range": {
                "start": prices.index[0].isoformat() if len(prices) > 0 else None,
                "end": prices.index[-1].isoformat() if len(prices) > 0 else None,
                "trading_days": len(prices),
            },
            "completeness": {
                "total_values": total_values,
                "nan_count": int(nan_total),
                "nan_percentage": float(nan_total / total_values * 100) if total_values > 0 else 0,
                "tickers_with_nan": int((nan_counts > 0).sum()),
            },
            "thresholds": {
                "stale_threshold_days": self._stale_threshold_days,
                "gap_threshold_days": self._gap_threshold_days,
                "spike_threshold": self._spike_threshold,
            },
        }
